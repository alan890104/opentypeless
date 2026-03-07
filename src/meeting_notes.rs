use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use unicode_segmentation::UnicodeSegmentation;
use hound;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingNote {
    pub id: String,
    pub title: String,
    pub transcript: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub duration_secs: f64,
    pub stt_model: String,
    pub is_recording: bool,
    pub word_count: u64,
    pub summary: String,
    /// Absolute path to the archived WAV file, or None if audio was not recorded.
    pub audio_path: Option<String>,
}

/// Run schema migrations. Called once at app startup.
pub fn init_db(history_dir: &Path) {
    match open_db(history_dir) {
        Ok(conn) => {
            if let Err(e) = conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS meeting_notes (
                    id            TEXT PRIMARY KEY,
                    title         TEXT NOT NULL,
                    transcript    TEXT NOT NULL DEFAULT '',
                    created_at    INTEGER NOT NULL,
                    updated_at    INTEGER NOT NULL,
                    duration_secs REAL NOT NULL DEFAULT 0.0,
                    stt_model     TEXT NOT NULL DEFAULT '',
                    is_recording  INTEGER NOT NULL DEFAULT 0,
                    word_count    INTEGER NOT NULL DEFAULT 0,
                    summary       TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_meeting_notes_created ON meeting_notes(created_at DESC);",
            ) {
                tracing::error!("Failed to init meeting_notes schema: {}", e);
            }
            // Migration: add summary column for existing databases.
            let _ = conn.execute_batch(
                "ALTER TABLE meeting_notes ADD COLUMN summary TEXT NOT NULL DEFAULT '';",
            );
            // Migration: add audio_path column for existing databases.
            let _ = conn.execute_batch(
                "ALTER TABLE meeting_notes ADD COLUMN audio_path TEXT;",
            );
        }
        Err(e) => tracing::error!("Failed to open DB for meeting_notes init: {}", e),
    }
}

fn open_db(history_dir: &Path) -> Result<Connection, rusqlite::Error> {
    let _ = std::fs::create_dir_all(history_dir);
    let conn = Connection::open(history_dir.join("history.db"))?;
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;
    Ok(conn)
}

pub fn create_note(history_dir: &Path, note: &MeetingNote) -> Result<(), String> {
    let conn = open_db(history_dir).map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT INTO meeting_notes (id, title, transcript, created_at, updated_at, duration_secs, stt_model, is_recording, word_count, summary)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            note.id,
            note.title,
            note.transcript,
            note.created_at,
            note.updated_at,
            note.duration_secs,
            note.stt_model,
            note.is_recording as i32,
            note.word_count as i64,
            note.summary,
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn get_note(history_dir: &Path, id: &str) -> Result<MeetingNote, String> {
    let conn = open_db(history_dir).map_err(|e| e.to_string())?;
    let mut note = conn
        .query_row(
            "SELECT id, title, transcript, created_at, updated_at, duration_secs, stt_model, is_recording, word_count, summary, audio_path
             FROM meeting_notes WHERE id = ?1",
            params![id],
            map_row,
        )
        .map_err(|e| e.to_string())?;
    // For recording notes the live transcript lives on disk, not in SQLite.
    if note.is_recording {
        note.transcript = read_wal(history_dir, id);
        note.word_count = note.transcript.unicode_words().count() as u64;
    }
    Ok(note)
}

pub fn list_notes(history_dir: &Path) -> Vec<MeetingNote> {
    let conn = match open_db(history_dir) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Failed to open DB for meeting notes list: {}", e);
            return Vec::new();
        }
    };
    let mut stmt = match conn.prepare(
        "SELECT id, title, transcript, created_at, updated_at, duration_secs, stt_model, is_recording, word_count, summary, audio_path
         FROM meeting_notes ORDER BY created_at DESC",
    ) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("Failed to prepare meeting notes query: {}", e);
            return Vec::new();
        }
    };
    let result = stmt.query_map([], map_row);
    let mut notes: Vec<MeetingNote> = match result {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            tracing::error!("Failed to query meeting notes: {}", e);
            return Vec::new();
        }
    };
    // Merge live transcript from WAL file for notes still recording.
    for note in &mut notes {
        if note.is_recording {
            note.transcript = read_wal(history_dir, &note.id);
            note.word_count = note.transcript.unicode_words().count() as u64;
        }
    }
    notes
}

// ── Transcript file ──
// During recording the transcript lives on disk, NOT in memory.
// Each new STT segment is appended to a text file. This avoids holding a
// growing String in the backend for the entire meeting duration.
// On normal stop: read the file, write once to SQLite, delete file.
// On crash: startup recovery reads the file back into SQLite.

fn wal_path(history_dir: &Path, id: &str) -> PathBuf {
    debug_assert!(
        !id.contains('/') && !id.contains('\\') && !id.contains(".."),
        "meeting note id must not contain path separators: {id}"
    );
    history_dir.join(format!("{}.meeting_wal", id))
}

/// Append a new text segment to the transcript file.
/// Called from the feeder thread each time a STT segment is produced.
pub fn append_wal(history_dir: &Path, id: &str, segment: &str) {
    use std::io::Write;
    let path = wal_path(history_dir, id);
    match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            if let Err(e) = f.write_all(segment.as_bytes()) {
                tracing::warn!("Failed to append meeting WAL: {}", e);
            }
        }
        Err(e) => tracing::warn!("Failed to open meeting WAL for append: {}", e),
    }
}

/// Read the full transcript from the file. Used by `stop_meeting_mode`
/// and `get_note` (for notes still recording).
pub fn read_wal(history_dir: &Path, id: &str) -> String {
    let path = wal_path(history_dir, id);
    std::fs::read_to_string(&path).unwrap_or_default()
}

/// Remove the transcript file after a successful finalize.
pub fn remove_wal(history_dir: &Path, id: &str) {
    let path = wal_path(history_dir, id);
    let _ = std::fs::remove_file(&path);
}

pub fn finalize_note(
    history_dir: &Path,
    id: &str,
    transcript: &str,
    duration_secs: f64,
) -> Result<(), String> {
    let conn = open_db(history_dir).map_err(|e| e.to_string())?;
    let now = now_millis();
    let wc = transcript.unicode_words().count() as i64;
    conn.execute(
        "UPDATE meeting_notes SET transcript = ?1, updated_at = ?2, duration_secs = ?3, is_recording = 0, word_count = ?4 WHERE id = ?5",
        params![transcript, now, duration_secs, wc, id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn rename_note(history_dir: &Path, id: &str, title: &str) -> Result<(), String> {
    let conn = open_db(history_dir).map_err(|e| e.to_string())?;
    let now = now_millis();
    conn.execute(
        "UPDATE meeting_notes SET title = ?1, updated_at = ?2 WHERE id = ?3",
        params![title, now, id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn delete_note(history_dir: &Path, id: &str) -> Result<(), String> {
    let conn = open_db(history_dir).map_err(|e| e.to_string())?;
    // Read audio_path before deleting the row so we can clean up the file.
    let audio_path: Option<String> = conn
        .query_row(
            "SELECT audio_path FROM meeting_notes WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )
        .unwrap_or(None);
    conn.execute("DELETE FROM meeting_notes WHERE id = ?1", params![id])
        .map_err(|e| e.to_string())?;
    // Best-effort cleanup of the transcript WAL (may not exist for finalized notes).
    remove_wal(history_dir, id);
    // Best-effort cleanup of audio files.
    if let Some(p) = audio_path {
        let _ = std::fs::remove_file(&p);
    }
    let _ = std::fs::remove_file(audio_wal_path(history_dir, id));
    Ok(())
}

pub fn delete_all_notes(history_dir: &Path) -> Result<(), String> {
    let conn = open_db(history_dir).map_err(|e| e.to_string())?;
    // Collect IDs and audio_paths before deleting so we can clean up files.
    let mut stmt = conn
        .prepare("SELECT id, audio_path FROM meeting_notes")
        .map_err(|e| e.to_string())?;
    let rows: Vec<(String, Option<String>)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();
    drop(stmt);
    conn.execute("DELETE FROM meeting_notes", [])
        .map_err(|e| e.to_string())?;
    for (id, audio_path) in &rows {
        remove_wal(history_dir, id);
        if let Some(p) = audio_path {
            let _ = std::fs::remove_file(p);
        }
        let _ = std::fs::remove_file(audio_wal_path(history_dir, id));
    }
    Ok(())
}

/// On startup, recover notes stuck in is_recording=1 from a previous crash.
/// Reads any WAL file to restore the transcript, then marks the note as finalized.
/// Also finalizes any pending audio WAL into a WAV file.
pub fn recover_stuck_notes(history_dir: &Path, audio_dir: &Path) {
    let conn = match open_db(history_dir) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Failed to open DB for stuck notes recovery: {}", e);
            return;
        }
    };
    // Find stuck notes.
    let stuck_ids: Vec<String> = {
        let mut stmt = match conn.prepare(
            "SELECT id FROM meeting_notes WHERE is_recording = 1",
        ) {
            Ok(s) => s,
            Err(_) => return,
        };
        stmt.query_map([], |row| row.get(0))
            .ok()
            .map(|iter| iter.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    };
    if stuck_ids.is_empty() {
        return;
    }
    tracing::info!("Recovering {} stuck meeting notes", stuck_ids.len());
    let now = now_millis();
    for id in &stuck_ids {
        // Try to read WAL file for the transcript.
        let wal = wal_path(history_dir, id);
        let transcript = std::fs::read_to_string(&wal).unwrap_or_default();
        let wc = transcript.unicode_words().count() as i64;
        // Also finalize any pending audio WAL.
        let audio_path = finalize_audio(history_dir, id, audio_dir);
        let _ = conn.execute(
            "UPDATE meeting_notes SET transcript = ?1, updated_at = ?2, is_recording = 0, word_count = ?3, audio_path = ?4 WHERE id = ?5",
            params![transcript, now, wc, audio_path, id],
        );
        let _ = std::fs::remove_file(&wal);
    }
}

fn map_row(row: &rusqlite::Row) -> Result<MeetingNote, rusqlite::Error> {
    Ok(MeetingNote {
        id: row.get(0)?,
        title: row.get(1)?,
        transcript: row.get(2)?,
        created_at: row.get(3)?,
        updated_at: row.get(4)?,
        duration_secs: row.get(5)?,
        stt_model: row.get(6)?,
        is_recording: row.get::<_, i32>(7)? != 0,
        word_count: row.get::<_, i64>(8).unwrap_or(0) as u64,
        summary: row.get::<_, String>(9).unwrap_or_default(),
        audio_path: row.get::<_, Option<String>>(10).unwrap_or(None),
    })
}

// ── Audio WAL file ────────────────────────────────────────────────────────────
// When `record_meeting_audio` is enabled, raw f32 audio samples (16 kHz, mono,
// little-endian) are appended here during recording.  On normal stop
// `finalize_audio` converts this to a WAV file and removes the temp file.
// On crash, `recover_stuck_notes` performs the same finalization on startup.

fn audio_wal_path(history_dir: &Path, id: &str) -> PathBuf {
    debug_assert!(
        !id.contains('/') && !id.contains('\\') && !id.contains(".."),
        "meeting note id must not contain path separators: {id}"
    );
    history_dir.join(format!("{}.audio_raw", id))
}

/// Append raw 16 kHz mono f32 samples to the audio WAL file.
/// Called from the meeting feeder segmenter thread for each segment.
pub fn append_audio_wal(history_dir: &Path, id: &str, samples: &[f32]) {
    use std::io::Write;
    let path = audio_wal_path(history_dir, id);
    match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            let bytes: Vec<u8> = samples.iter().flat_map(|&s| s.to_le_bytes()).collect();
            if let Err(e) = f.write_all(&bytes) {
                tracing::warn!("Failed to append meeting audio WAL: {}", e);
            }
        }
        Err(e) => tracing::warn!("Failed to open meeting audio WAL for append: {}", e),
    }
}

/// Convert the raw audio WAL into a 16-bit PCM WAV file stored in
/// `audio_dir/meetings/{id}.wav`.  Deletes the raw temp file on success or
/// failure.  Returns the WAV path as a String, or None if no audio WAL exists
/// or conversion fails.
pub fn finalize_audio(history_dir: &Path, id: &str, audio_dir: &Path) -> Option<String> {
    let raw_path = audio_wal_path(history_dir, id);
    if !raw_path.exists() {
        return None;
    }
    let bytes = match std::fs::read(&raw_path) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!("Failed to read meeting audio WAL: {}", e);
            let _ = std::fs::remove_file(&raw_path);
            return None;
        }
    };
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // Always remove the raw temp file, even if WAV writing fails.
    let _ = std::fs::remove_file(&raw_path);
    if samples.is_empty() {
        return None;
    }
    let meetings_dir = audio_dir.join("meetings");
    if std::fs::create_dir_all(&meetings_dir).is_err() {
        return None;
    }
    let wav_path = meetings_dir.join(format!("{}.wav", id));
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let ok = match hound::WavWriter::create(&wav_path, spec) {
        Ok(mut w) => {
            let mut success = true;
            for &s in &samples {
                let val = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                if w.write_sample(val).is_err() {
                    success = false;
                    break;
                }
            }
            success && w.finalize().is_ok()
        }
        Err(_) => false,
    };
    if ok {
        Some(wav_path.to_string_lossy().into_owned())
    } else {
        None
    }
}

/// Delete only the audio file for a note, setting `audio_path = NULL` in SQLite.
/// The transcript and summary are preserved.
pub fn delete_audio_file(history_dir: &Path, id: &str) -> Result<(), String> {
    let conn = open_db(history_dir).map_err(|e| e.to_string())?;
    let audio_path: Option<String> = conn
        .query_row(
            "SELECT audio_path FROM meeting_notes WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )
        .unwrap_or(None);
    conn.execute(
        "UPDATE meeting_notes SET audio_path = NULL WHERE id = ?1",
        params![id],
    )
    .map_err(|e| e.to_string())?;
    if let Some(p) = audio_path {
        let _ = std::fs::remove_file(&p);
    }
    Ok(())
}

/// Persist the finalized audio WAV path into the SQLite row.
pub fn update_audio_path(history_dir: &Path, id: &str, path: &str) -> Result<(), String> {
    let conn = open_db(history_dir).map_err(|e| e.to_string())?;
    conn.execute(
        "UPDATE meeting_notes SET audio_path = ?1 WHERE id = ?2",
        params![path, id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn save_summary(
    history_dir: &Path,
    id: &str,
    title: &str,
    summary: &str,
) -> Result<(), String> {
    let conn = open_db(history_dir).map_err(|e| e.to_string())?;
    let now = now_millis();
    conn.execute(
        "UPDATE meeting_notes SET title = ?1, summary = ?2, updated_at = ?3 WHERE id = ?4",
        params![title, summary, now, id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_note(id: &str) -> MeetingNote {
        MeetingNote {
            id: id.to_string(),
            title: format!("Test Note {}", id),
            transcript: String::new(),
            created_at: now_millis(),
            updated_at: now_millis(),
            duration_secs: 0.0,
            stt_model: "test".to_string(),
            is_recording: false,
            word_count: 0,
            summary: String::new(),
            audio_path: None,
        }
    }

    // ── WAL file: the "everything is a file" design ──

    #[test]
    fn wal_multi_append_accumulates() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        append_wal(p, "n1", "Hello ");
        append_wal(p, "n1", "world ");
        append_wal(p, "n1", "foo");
        assert_eq!(read_wal(p, "n1"), "Hello world foo");
    }

    #[test]
    fn wal_read_nonexistent_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_wal(dir.path(), "missing"), "");
    }

    /// Recording notes read transcript from WAL, not SQLite.
    /// This is the core design rule — verify it works.
    #[test]
    fn get_note_reads_wal_for_recording_note() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        init_db(p);
        let mut note = make_note("rec");
        note.is_recording = true;
        create_note(p, &note).unwrap();
        append_wal(p, "rec", "live transcript data");
        let fetched = get_note(p, "rec").unwrap();
        assert_eq!(fetched.transcript, "live transcript data");
        // word_count should be recomputed from WAL content
        assert!(fetched.word_count > 0);
    }

    /// finalize writes WAL content to SQLite and clears is_recording.
    #[test]
    fn finalize_persists_transcript_to_sqlite() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        init_db(p);
        let mut note = make_note("fin");
        note.is_recording = true;
        create_note(p, &note).unwrap();
        finalize_note(p, "fin", "Hello world transcript", 42.5).unwrap();
        let fetched = get_note(p, "fin").unwrap();
        assert!(!fetched.is_recording);
        assert_eq!(fetched.transcript, "Hello world transcript");
        assert!((fetched.duration_secs - 42.5).abs() < 0.01);
        assert!(fetched.word_count > 0);
    }

    // ── Crash recovery: the most critical safety path ──

    #[test]
    fn recover_stuck_notes_restores_from_wal() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        init_db(p);
        let mut note = make_note("stuck");
        note.is_recording = true;
        create_note(p, &note).unwrap();
        append_wal(p, "stuck", "recovered text");

        recover_stuck_notes(p, p);

        let fetched = get_note(p, "stuck").unwrap();
        assert!(!fetched.is_recording);
        assert_eq!(fetched.transcript, "recovered text");
        assert_eq!(read_wal(p, "stuck"), "", "WAL should be cleaned up");
    }

    #[test]
    fn recover_stuck_notes_handles_crash_before_any_audio() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        init_db(p);
        let mut note = make_note("stuck2");
        note.is_recording = true;
        create_note(p, &note).unwrap();
        // No WAL file — crash happened before any STT output

        recover_stuck_notes(p, p);

        let fetched = get_note(p, "stuck2").unwrap();
        assert!(!fetched.is_recording);
        assert_eq!(fetched.transcript, "");
    }

    /// delete_all must also clean up WAL files (not just SQLite rows).
    #[test]
    fn delete_all_cleans_up_wal_files() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        init_db(p);
        create_note(p, &make_note("1")).unwrap();
        append_wal(p, "1", "data");
        delete_all_notes(p).unwrap();
        assert!(list_notes(p).is_empty());
        assert_eq!(read_wal(p, "1"), "");
    }
}

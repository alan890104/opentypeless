use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use unicode_segmentation::UnicodeSegmentation;
use hound;

// ── WAL segment types ─────────────────────────────────────────────────────────
//
// Each transcribed VAD segment is written to the WAL file as one JSON line
// (JSONL format). This replaces the old plain-text append format and enables
// speaker labels and word-level timestamps alongside the text.
//
// Backward compat: `read_wal` returns raw file content unchanged. Callers that
// need displayable text call `transcript_from_wal`. Old finalized notes keep
// their plain-text transcript in SQLite; the frontend detects the format by
// checking whether each line parses as a JSON object.

/// One transcribed segment written to the WAL file as a single JSON line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalSegment {
    /// Speaker label, e.g. `"SPEAKER_00"`. Empty when diarization is disabled.
    pub speaker: String,
    /// Segment start time in seconds from meeting start.
    pub start: f64,
    /// Segment end time in seconds from meeting start.
    pub end: f64,
    /// Transcribed text for this segment.
    pub text: String,
    /// Word-level timestamps. Empty when unavailable (Qwen3-ASR, most cloud).
    pub words: Vec<WordTs>,
}

/// Single word with start/end timestamps.
/// Short field names keep WAL files compact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTs {
    /// Word text.
    pub w: String,
    /// Start time in seconds from meeting start.
    pub s: f64,
    /// End time in seconds from meeting start.
    pub e: f64,
}

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
        note.word_count = count_words_in_transcript(&note.transcript);
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
            note.word_count = count_words_in_transcript(&note.transcript);
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

/// Append a new segment to the WAL file as a single JSON line.
/// Called from the feeder thread each time a STT segment is produced.
/// Segments with empty text are written so that `update_wal_speakers` can
/// re-label them during agglomerative finalization; they carry (start, end,
/// speaker) even without transcribed text.
pub fn append_wal(history_dir: &Path, id: &str, segment: &WalSegment) {
    use std::io::Write;
    let path = wal_path(history_dir, id);
    let line = match serde_json::to_string(segment) {
        Ok(s) => s + "\n",
        Err(e) => {
            tracing::warn!("Failed to serialize WAL segment: {}", e);
            return;
        }
    };
    match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            if let Err(e) = f.write_all(line.as_bytes()) {
                tracing::warn!("Failed to append meeting WAL: {}", e);
            }
        }
        Err(e) => tracing::warn!("Failed to open meeting WAL for append: {}", e),
    }
}

/// Convert a JSONL WAL to `"SPEAKER_XX: text\n"` format for LLM polishing.
///
/// Lines that fail to parse as JSON are treated as legacy plain text and
/// emitted as-is, so notes written before the JSONL migration still work.
pub fn transcript_from_wal(wal_content: &str) -> String {
    let mut out = String::new();
    let mut prev_speaker = String::new();
    for line in wal_content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(seg) = serde_json::from_str::<WalSegment>(line) {
            if seg.text.is_empty() {
                continue;
            }
            if seg.speaker != prev_speaker {
                if !out.is_empty() {
                    out.push('\n');
                }
                if !seg.speaker.is_empty() {
                    out.push_str(&seg.speaker);
                    out.push_str(": ");
                }
                prev_speaker = seg.speaker.clone();
            }
            out.push_str(&seg.text);
            out.push('\n');
        } else {
            // Legacy plain-text line.
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Extract concatenated plain text from JSONL WAL for use as Whisper
/// `initial_prompt` context (last `max_chars` characters).
pub fn wal_text_for_context(wal_content: &str, max_chars: usize) -> String {
    let mut text = String::new();
    for line in wal_content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(seg) = serde_json::from_str::<WalSegment>(line) {
            if !text.is_empty() && !seg.text.is_empty() {
                text.push(' ');
            }
            text.push_str(&seg.text);
        } else {
            // Legacy plain text.
            text.push_str(line);
        }
    }
    let trimmed = text.trim_end();
    let char_count = trimmed.chars().count();
    if char_count > max_chars {
        trimmed
            .char_indices()
            .nth(char_count - max_chars)
            .map(|(i, _)| trimmed[i..].to_string())
            .unwrap_or_else(|| trimmed.to_string())
    } else {
        trimmed.to_string()
    }
}

/// Read the full transcript from the file. Used by `stop_meeting_mode`
/// and `get_note` (for notes still recording).
pub fn read_wal(history_dir: &Path, id: &str) -> String {
    let path = wal_path(history_dir, id);
    std::fs::read_to_string(&path).unwrap_or_default()
}

/// Overwrite the WAL file with new content.
/// Used after agglomerative speaker relabeling in the import pipeline to
/// replace online speaker labels with globally-optimal ones before finalizing.
#[allow(dead_code)]
pub fn write_wal(history_dir: &Path, id: &str, content: &str) {
    let path = wal_path(history_dir, id);
    if let Err(e) = std::fs::write(&path, content) {
        tracing::warn!("Failed to overwrite meeting WAL: {}", e);
    }
}

/// Remove the transcript file after a successful finalize.
pub fn remove_wal(history_dir: &Path, id: &str) {
    let path = wal_path(history_dir, id);
    let _ = std::fs::remove_file(&path);
}

/// Rewrite speaker labels in a JSONL WAL transcript using agglomerative labels.
///
/// Each WAL line that matches a `(start_secs, end_secs)` pair in `labels`
/// has its `speaker` field updated.  Lines that do not match (plain text or
/// mismatched timestamps) are passed through unchanged.
///
/// Called in `stop_meeting_mode` after `finalize_labels()` to upgrade the
/// real-time online labels to globally-optimal agglomerative labels before
/// writing the transcript to SQLite.
pub fn update_wal_speakers(wal: &str, labels: &[(f64, f64, String)]) -> String {
    wal.lines()
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return line.to_string();
            }
            let mut seg: WalSegment = match serde_json::from_str(trimmed) {
                Ok(s) => s,
                Err(_) => return line.to_string(), // plain-text line — keep as-is
            };
            // Match by (start, end) within 10 ms tolerance.
            let matched = labels.iter().find(|(s, e, _)| {
                (seg.start - s).abs() < 0.01 && (seg.end - e).abs() < 0.01
            });
            if let Some((_, _, spk)) = matched {
                seg.speaker = spk.clone();
            } else if !seg.speaker.is_empty() {
                tracing::warn!(
                    "[diarization] WAL segment [{:.3}–{:.3}s] not in agglomerative labels \
                     — keeping online label {:?}",
                    seg.start, seg.end, seg.speaker
                );
            }
            serde_json::to_string(&seg).unwrap_or_else(|_| line.to_string())
        })
        .collect::<Vec<_>>()
        .join("\n") + "\n"
}

pub fn finalize_note(
    history_dir: &Path,
    id: &str,
    transcript: &str,
    duration_secs: f64,
) -> Result<(), String> {
    let conn = open_db(history_dir).map_err(|e| e.to_string())?;
    let now = now_millis();
    let wc = count_words_in_transcript(transcript) as i64;
    conn.execute(
        "UPDATE meeting_notes SET transcript = ?1, updated_at = ?2, duration_secs = ?3, is_recording = 0, word_count = ?4 WHERE id = ?5",
        params![transcript, now, duration_secs, wc, id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Count words in a transcript that may be JSONL (new format) or plain text (legacy).
fn count_words_in_transcript(transcript: &str) -> u64 {
    let mut count = 0u64;
    let mut found_jsonl = false;
    for line in transcript.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(seg) = serde_json::from_str::<WalSegment>(line) {
            count += seg.text.unicode_words().count() as u64;
            found_jsonl = true;
        }
    }
    if !found_jsonl {
        // Legacy plain text.
        count = transcript.unicode_words().count() as u64;
    }
    count
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
pub fn recover_stuck_notes(history_dir: &Path, _audio_dir: &Path) {
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
        let wc = count_words_in_transcript(&transcript) as i64;
        let _ = conn.execute(
            "UPDATE meeting_notes SET transcript = ?1, updated_at = ?2, is_recording = 0, word_count = ?3 WHERE id = ?4",
            params![transcript, now, wc, id],
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
#[allow(dead_code)]
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

    fn make_seg(text: &str) -> WalSegment {
        WalSegment {
            speaker: "SPEAKER_00".to_string(),
            start: 0.0,
            end: 1.0,
            text: text.to_string(),
            words: vec![],
        }
    }

    #[test]
    fn wal_multi_append_accumulates() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        append_wal(p, "n1", &make_seg("Hello"));
        append_wal(p, "n1", &make_seg("world"));
        append_wal(p, "n1", &make_seg("foo"));
        let raw = read_wal(p, "n1");
        // Each line is a JSON object; extract texts via transcript_from_wal.
        let text = transcript_from_wal(&raw);
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
        assert!(text.contains("foo"));
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
        append_wal(p, "rec", &make_seg("live transcript data"));
        let fetched = get_note(p, "rec").unwrap();
        // Transcript is raw JSONL; verify the text is inside.
        assert!(fetched.transcript.contains("live transcript data"));
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
        append_wal(p, "stuck", &make_seg("recovered text"));

        recover_stuck_notes(p, p);

        let fetched = get_note(p, "stuck").unwrap();
        assert!(!fetched.is_recording);
        // Transcript is stored as raw JSONL; verify the text is present via transcript_from_wal.
        assert!(transcript_from_wal(&fetched.transcript).contains("recovered text"));
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
        append_wal(p, "1", &make_seg("data"));
        delete_all_notes(p).unwrap();
        assert!(list_notes(p).is_empty());
        assert_eq!(read_wal(p, "1"), "");
    }
}

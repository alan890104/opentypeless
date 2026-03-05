use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use unicode_segmentation::UnicodeSegmentation;

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
            "SELECT id, title, transcript, created_at, updated_at, duration_secs, stt_model, is_recording, word_count, summary
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
        "SELECT id, title, transcript, created_at, updated_at, duration_secs, stt_model, is_recording, word_count, summary
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
    conn.execute("DELETE FROM meeting_notes WHERE id = ?1", params![id])
        .map_err(|e| e.to_string())?;
    // Best-effort cleanup of the WAL file (may not exist for finalized notes).
    remove_wal(history_dir, id);
    Ok(())
}

pub fn delete_all_notes(history_dir: &Path) -> Result<(), String> {
    let conn = open_db(history_dir).map_err(|e| e.to_string())?;
    // Collect IDs before deleting so we can clean up WAL files.
    let mut stmt = conn
        .prepare("SELECT id FROM meeting_notes")
        .map_err(|e| e.to_string())?;
    let ids: Vec<String> = stmt
        .query_map([], |row| row.get(0))
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();
    drop(stmt);
    conn.execute("DELETE FROM meeting_notes", [])
        .map_err(|e| e.to_string())?;
    for id in &ids {
        remove_wal(history_dir, id);
    }
    Ok(())
}

/// On startup, recover notes stuck in is_recording=1 from a previous crash.
/// Reads any WAL file to restore the transcript, then marks the note as finalized.
pub fn recover_stuck_notes(history_dir: &Path) {
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
    })
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

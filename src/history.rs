use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: String,
    pub timestamp: i64,
    pub text: String,
    pub raw_text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    pub stt_model: String,
    pub polish_model: String,
    pub duration_secs: f64,
    pub has_audio: bool,
    pub stt_elapsed_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polish_elapsed_ms: Option<u64>,
    pub total_elapsed_ms: u64,
    #[serde(default)]
    pub app_name: String,
    #[serde(default)]
    pub bundle_id: String,
    #[serde(default)]
    pub chars_per_sec: f64,
    #[serde(default)]
    pub word_count: u64,
}

/// Count "words" using UAX#29 word boundaries.
/// Each CJK character is its own word; Latin text is split by whitespace/punctuation.
pub fn count_words(text: &str) -> usize {
    text.unicode_words().count()
}

fn db_path(history_dir: &Path) -> PathBuf {
    history_dir.join("history.db")
}

fn audio_path(audio_dir: &Path, id: &str) -> PathBuf {
    // Caller must validate id before calling; this is a low-level helper.
    audio_dir.join(format!("{}.wav", id))
}

/// Validate that a history ID contains only safe characters (digits and underscores).
fn validate_id(id: &str) -> Result<(), String> {
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_digit() || c == '_') {
        return Err("Invalid history ID format".to_string());
    }
    Ok(())
}

/// Run schema creation and migrations. Call once at app startup.
pub fn init_db(history_dir: &Path) {
    match open_db_and_migrate(history_dir) {
        Ok(_) => tracing::info!("History DB initialized"),
        Err(e) => tracing::error!("Failed to initialize history DB: {}", e),
    }
}

/// Open + full migration — used only by `init_db` at startup.
fn open_db_and_migrate(history_dir: &Path) -> Result<Connection, rusqlite::Error> {
    let conn = open_db(history_dir)?;
    // Validate schema: if the table exists but is missing expected columns, drop and recreate.
    let has_table: bool = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='history'",
        [],
        |row| row.get::<_, i64>(0),
    )? > 0;
    if has_table {
        let col_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('history') WHERE name IN ('app_name','bundle_id')",
            [],
            |row| row.get(0),
        )?;
        if col_count < 2 {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let backup_name = format!("history_bak_{}", ts);
            tracing::warn!("Schema mismatch — preserving data in '{}', recreating table", backup_name);
            conn.execute_batch(&format!("ALTER TABLE history RENAME TO {};", backup_name))?;
        }
    }
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS history (
            id               TEXT PRIMARY KEY,
            timestamp        INTEGER NOT NULL,
            text             TEXT NOT NULL,
            raw_text         TEXT NOT NULL,
            reasoning        TEXT,
            stt_model        TEXT NOT NULL,
            polish_model     TEXT NOT NULL,
            duration_secs    REAL NOT NULL,
            has_audio        INTEGER NOT NULL DEFAULT 0,
            stt_elapsed_ms   INTEGER NOT NULL DEFAULT 0,
            polish_elapsed_ms INTEGER,
            total_elapsed_ms INTEGER NOT NULL DEFAULT 0,
            app_name         TEXT NOT NULL DEFAULT '',
            bundle_id        TEXT NOT NULL DEFAULT '',
            chars_per_sec    REAL NOT NULL DEFAULT 0.0
        );
        CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history(timestamp DESC);",
    )?;
    // Migrate: add chars_per_sec column if missing (non-destructive)
    let has_cps: bool = conn.query_row(
        "SELECT COUNT(*) FROM pragma_table_info('history') WHERE name = 'chars_per_sec'",
        [],
        |row| row.get::<_, i64>(0),
    )? > 0;
    if !has_cps {
        conn.execute_batch("ALTER TABLE history ADD COLUMN chars_per_sec REAL NOT NULL DEFAULT 0.0;")?;
    }
    // Migrate: add word_count column if missing (non-destructive)
    let has_wc: bool = conn.query_row(
        "SELECT COUNT(*) FROM pragma_table_info('history') WHERE name = 'word_count'",
        [],
        |row| row.get::<_, i64>(0),
    )? > 0;
    if !has_wc {
        conn.execute_batch("ALTER TABLE history ADD COLUMN word_count INTEGER NOT NULL DEFAULT 0;")?;
    }
    // Backfill word_count for existing rows that have 0
    {
        let mut stmt = conn.prepare("SELECT id, raw_text FROM history WHERE word_count = 0")?;
        let rows: Vec<(String, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();
        for (id, raw_text) in rows {
            conn.execute(
                "UPDATE history SET word_count = ?1 WHERE id = ?2",
                params![count_words(&raw_text) as i64, id],
            )?;
        }
    }
    Ok(conn)
}

/// Lightweight connection opener — no migrations, just WAL pragma.
fn open_db(history_dir: &Path) -> Result<Connection, rusqlite::Error> {
    let _ = std::fs::create_dir_all(history_dir);
    let conn = Connection::open(db_path(history_dir))?;
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;
    Ok(conn)
}

/// Delete old `history.json` and clear leftover audio from the JSON era.
/// Idempotent — safe to call on every startup.
pub fn migrate_from_json(history_dir: &Path, audio_dir: &Path) {
    let json_path = history_dir.join("history.json");
    if json_path.exists() {
        tracing::info!("Migrating: removing legacy history.json");
        let _ = std::fs::remove_file(&json_path);
        if audio_dir.exists() {
            let _ = std::fs::remove_dir_all(audio_dir);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryStats {
    pub total_entries: u64,
    pub total_duration_secs: f64,
    pub total_chars: u64,
    pub local_entries: u64,
    pub local_duration_secs: f64,
    pub total_words: u64,
}

pub fn get_stats(history_dir: &Path) -> HistoryStats {
    let zero = HistoryStats { total_entries: 0, total_duration_secs: 0.0, total_chars: 0, local_entries: 0, local_duration_secs: 0.0, total_words: 0 };
    let conn = match open_db(history_dir) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Failed to open history DB for stats: {}", e);
            return zero;
        }
    };
    conn.query_row(
        "SELECT COUNT(*), COALESCE(SUM(duration_secs), 0), COALESCE(SUM(LENGTH(raw_text)), 0),
                SUM(CASE WHEN stt_model NOT LIKE '%(Cloud/%)%' THEN 1 ELSE 0 END),
                COALESCE(SUM(CASE WHEN stt_model NOT LIKE '%(Cloud/%)%' THEN duration_secs ELSE 0 END), 0),
                COALESCE(SUM(word_count), 0)
         FROM history",
        [],
        |row| {
            Ok(HistoryStats {
                total_entries: row.get::<_, i64>(0).unwrap_or(0) as u64,
                total_duration_secs: row.get::<_, f64>(1).unwrap_or(0.0),
                total_chars: row.get::<_, i64>(2).unwrap_or(0) as u64,
                local_entries: row.get::<_, i64>(3).unwrap_or(0) as u64,
                local_duration_secs: row.get::<_, f64>(4).unwrap_or(0.0),
                total_words: row.get::<_, i64>(5).unwrap_or(0) as u64,
            })
        },
    )
    .unwrap_or(zero)
}

/// Shared row mapper for HistoryEntry — used by all query functions.
fn map_row(row: &rusqlite::Row) -> Result<HistoryEntry, rusqlite::Error> {
    Ok(HistoryEntry {
        id: row.get(0)?,
        timestamp: row.get(1)?,
        text: row.get(2)?,
        raw_text: row.get(3)?,
        reasoning: row.get(4)?,
        stt_model: row.get(5)?,
        polish_model: row.get(6)?,
        duration_secs: row.get(7)?,
        has_audio: row.get::<_, i32>(8)? != 0,
        stt_elapsed_ms: row.get::<_, i64>(9).unwrap_or(0) as u64,
        polish_elapsed_ms: row.get::<_, Option<i64>>(10).ok().flatten().map(|v| v as u64),
        total_elapsed_ms: row.get::<_, i64>(11).unwrap_or(0) as u64,
        app_name: row.get::<_, String>(12).unwrap_or_default(),
        bundle_id: row.get::<_, String>(13).unwrap_or_default(),
        chars_per_sec: row.get::<_, f64>(14).unwrap_or(0.0),
        word_count: row.get::<_, i64>(15).unwrap_or(0) as u64,
    })
}

pub fn load_history_page(
    history_dir: &Path,
    before_timestamp: Option<i64>,
    limit: u32,
) -> (Vec<HistoryEntry>, bool) {
    let conn = match open_db(history_dir) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Failed to open history DB: {}", e);
            return (Vec::new(), false);
        }
    };
    let fetch_limit = limit as i64 + 1;
    let mut entries: Vec<HistoryEntry> = if let Some(ts) = before_timestamp {
        let mut stmt = match conn.prepare(
            "SELECT id, timestamp, text, raw_text, reasoning, stt_model, polish_model,
                    duration_secs, has_audio, stt_elapsed_ms, polish_elapsed_ms, total_elapsed_ms,
                    app_name, bundle_id, chars_per_sec, word_count
             FROM history WHERE timestamp < ?1 ORDER BY timestamp DESC LIMIT ?2",
        ) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("Failed to prepare history page query: {}", e);
                return (Vec::new(), false);
            }
        };
        let result = stmt.query_map(params![ts, fetch_limit], map_row);
        match result {
            Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
            Err(e) => {
                tracing::error!("Failed to query history page: {}", e);
                return (Vec::new(), false);
            }
        }
    } else {
        let mut stmt = match conn.prepare(
            "SELECT id, timestamp, text, raw_text, reasoning, stt_model, polish_model,
                    duration_secs, has_audio, stt_elapsed_ms, polish_elapsed_ms, total_elapsed_ms,
                    app_name, bundle_id, chars_per_sec, word_count
             FROM history ORDER BY timestamp DESC LIMIT ?1",
        ) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("Failed to prepare history page query: {}", e);
                return (Vec::new(), false);
            }
        };
        let result = stmt.query_map(params![fetch_limit], map_row);
        match result {
            Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
            Err(e) => {
                tracing::error!("Failed to query history page: {}", e);
                return (Vec::new(), false);
            }
        }
    };
    let has_more = entries.len() > limit as usize;
    if has_more {
        entries.truncate(limit as usize);
    }
    (entries, has_more)
}

pub fn load_history(history_dir: &Path) -> Vec<HistoryEntry> {
    let conn = match open_db(history_dir) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Failed to open history DB: {}", e);
            return Vec::new();
        }
    };
    let mut stmt = match conn.prepare(
        "SELECT id, timestamp, text, raw_text, reasoning, stt_model, polish_model,
                duration_secs, has_audio, stt_elapsed_ms, polish_elapsed_ms, total_elapsed_ms,
                app_name, bundle_id, chars_per_sec, word_count
         FROM history ORDER BY timestamp DESC LIMIT 200",
    ) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("Failed to prepare history query: {}", e);
            return Vec::new();
        }
    };
    let rows = stmt.query_map([], map_row);
    match rows {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            tracing::error!("Failed to query history: {}", e);
            Vec::new()
        }
    }
}

pub fn add_entry(history_dir: &Path, audio_dir: &Path, entry: HistoryEntry, retention_days: u32) {
    let conn = match open_db(history_dir) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Failed to open history DB for insert: {}", e);
            return;
        }
    };
    let has_audio_int: i32 = if entry.has_audio { 1 } else { 0 };
    let polish_ms: Option<i64> = entry.polish_elapsed_ms.map(|v| v as i64);
    if let Err(e) = conn.execute(
        "INSERT OR REPLACE INTO history
            (id, timestamp, text, raw_text, reasoning, stt_model, polish_model,
             duration_secs, has_audio, stt_elapsed_ms, polish_elapsed_ms, total_elapsed_ms,
             app_name, bundle_id, chars_per_sec, word_count)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
        params![
            entry.id,
            entry.timestamp,
            entry.text,
            entry.raw_text,
            entry.reasoning,
            entry.stt_model,
            entry.polish_model,
            entry.duration_secs,
            has_audio_int,
            entry.stt_elapsed_ms as i64,
            polish_ms,
            entry.total_elapsed_ms as i64,
            entry.app_name,
            entry.bundle_id,
            entry.chars_per_sec,
            entry.word_count as i64,
        ],
    ) {
        tracing::error!("Failed to insert history entry: {}", e);
    }
    if retention_days > 0 {
        cleanup_expired(&conn, audio_dir, retention_days);
    }
}

fn cleanup_expired(conn: &Connection, audio_dir: &Path, retention_days: u32) {
    let now_millis = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    let cutoff = now_millis - (retention_days as i64) * 86_400_000;

    // Collect IDs of expired entries that have audio, so we can delete their WAV files.
    let ids: Vec<String> = {
        let mut stmt = match conn.prepare(
            "SELECT id FROM history WHERE timestamp < ?1 AND has_audio = 1",
        ) {
            Ok(s) => s,
            Err(_) => return,
        };
        stmt.query_map(params![cutoff], |row| row.get(0))
            .ok()
            .map(|iter| iter.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    };
    for id in &ids {
        let wav = audio_path(audio_dir, id);
        if wav.exists() {
            let _ = std::fs::remove_file(&wav);
        }
    }
    let _ = conn.execute("DELETE FROM history WHERE timestamp < ?1", params![cutoff]);
}

pub fn delete_entry(history_dir: &Path, audio_dir: &Path, id: &str) {
    if validate_id(id).is_err() { return; }
    if let Ok(conn) = open_db(history_dir) {
        let _ = conn.execute("DELETE FROM history WHERE id = ?1", params![id]);
    }
    let wav = audio_path(audio_dir, id);
    if wav.exists() {
        let _ = std::fs::remove_file(&wav);
    }
}

pub fn clear_all(history_dir: &Path, audio_dir: &Path) {
    if let Ok(conn) = open_db(history_dir) {
        let _ = conn.execute("DELETE FROM history", []);
    }
    if audio_dir.exists() {
        let _ = std::fs::remove_dir_all(audio_dir);
    }
}

pub fn save_audio_wav(audio_dir: &Path, id: &str, samples_16k: &[f32]) -> bool {
    if validate_id(id).is_err() { return false; }
    if std::fs::create_dir_all(audio_dir).is_err() {
        return false;
    }
    let path = audio_path(audio_dir, id);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    match hound::WavWriter::create(&path, spec) {
        Ok(mut writer) => {
            for &s in samples_16k {
                let clamped = s.clamp(-1.0, 1.0);
                let val = (clamped * 32767.0) as i16;
                if writer.write_sample(val).is_err() {
                    return false;
                }
            }
            writer.finalize().is_ok()
        }
        Err(_) => false,
    }
}

pub fn export_audio(audio_dir: &Path, id: &str) -> Result<PathBuf, String> {
    validate_id(id)?;
    let src = audio_path(audio_dir, id);
    if !src.exists() {
        return Err("Audio file not found".to_string());
    }
    let downloads = dirs::download_dir().unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("Downloads")
    });
    let _ = std::fs::create_dir_all(&downloads);
    let dest = downloads.join(format!("{}.wav", id));
    std::fs::copy(&src, &dest).map_err(|e| format!("Failed to copy audio: {}", e))?;
    Ok(dest)
}

pub fn generate_id() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let millis = now.subsec_millis();

    let ts = chrono_free_format(secs);
    format!("{}_{:03}", ts, millis)
}

/// Format seconds-since-epoch as YYYYMMDD_HHMMSS without chrono dependency.
fn chrono_free_format(epoch_secs: u64) -> String {
    #[cfg(unix)]
    {
        let t = epoch_secs as libc::time_t;
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        unsafe {
            libc::localtime_r(&t, &mut tm);
        }
        format!(
            "{:04}{:02}{:02}_{:02}{:02}{:02}",
            tm.tm_year + 1900,
            tm.tm_mon + 1,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min,
            tm.tm_sec,
        )
    }
    #[cfg(target_os = "windows")]
    {
        // Use Windows CRT _localtime64_s via extern C
        #[repr(C)]
        struct Tm {
            tm_sec: i32,
            tm_min: i32,
            tm_hour: i32,
            tm_mday: i32,
            tm_mon: i32,
            tm_year: i32,
            tm_wday: i32,
            tm_yday: i32,
            tm_isdst: i32,
        }
        extern "C" {
            fn _localtime64_s(result: *mut Tm, time: *const i64) -> i32;
        }
        let time = epoch_secs as i64;
        let mut tm: Tm = unsafe { std::mem::zeroed() };
        let err = unsafe { _localtime64_s(&mut tm, &time) };
        if err == 0 {
            format!(
                "{:04}{:02}{:02}_{:02}{:02}{:02}",
                tm.tm_year + 1900,
                tm.tm_mon + 1,
                tm.tm_mday,
                tm.tm_hour,
                tm.tm_min,
                tm.tm_sec,
            )
        } else {
            format!("{}", epoch_secs)
        }
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        format!("{}", epoch_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── count_words: CJK segmentation is the non-obvious part ──

    #[test]
    fn cjk_each_char_is_one_word() {
        assert_eq!(count_words("你好世界"), 4);
    }

    #[test]
    fn mixed_cjk_latin() {
        // "Hello" = 1, "你好" = 2, "world" = 1
        let count = count_words("Hello 你好 world");
        assert!(count >= 4, "expected at least 4, got {}", count);
    }

    // ── validate_id: path traversal prevention ──

    #[test]
    fn path_traversal_rejected() {
        assert!(validate_id("../etc/passwd").is_err());
        assert!(validate_id("foo/bar").is_err());
        assert!(validate_id("").is_err());
    }

    #[test]
    fn generated_id_passes_validation() {
        let id = generate_id();
        assert!(validate_id(&id).is_ok(), "generated id '{}' failed validation", id);
    }

    // ── Retention cleanup: the cutoff math and cascading audio delete ──

    fn make_entry(id: &str, timestamp_ms: i64) -> HistoryEntry {
        HistoryEntry {
            id: id.to_string(),
            timestamp: timestamp_ms,
            text: "polished".to_string(),
            raw_text: "raw".to_string(),
            reasoning: None,
            stt_model: "test".to_string(),
            polish_model: "test".to_string(),
            duration_secs: 1.0,
            has_audio: true,
            stt_elapsed_ms: 100,
            polish_elapsed_ms: Some(50),
            total_elapsed_ms: 150,
            app_name: "".to_string(),
            bundle_id: "".to_string(),
            chars_per_sec: 10.0,
            word_count: 1,
        }
    }

    fn now_ms() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
    }

    #[test]
    fn retention_deletes_old_entries_and_audio_files() {
        let hist_dir = tempfile::tempdir().unwrap();
        let audio_dir = tempfile::tempdir().unwrap();
        let hp = hist_dir.path();
        let ap = audio_dir.path();
        init_db(hp);

        let now = now_ms();
        let two_days_ago = now - 2 * 86_400_000;
        let fresh = now - 3_600_000; // 1 hour ago

        // Insert an old entry and a fresh entry.
        add_entry(hp, ap, make_entry("111_111_111", two_days_ago), 0);
        add_entry(hp, ap, make_entry("222_222_222", fresh), 0);

        // Create fake audio files for both.
        std::fs::write(ap.join("111_111_111.wav"), b"old").unwrap();
        std::fs::write(ap.join("222_222_222.wav"), b"new").unwrap();

        // Now add a third entry with retention_days=1 — triggers cleanup.
        add_entry(hp, ap, make_entry("333_333_333", now), 1);

        let entries = load_history(hp);
        let ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
        assert!(!ids.contains(&"111_111_111"), "old entry should be deleted");
        assert!(ids.contains(&"222_222_222"), "fresh entry should remain");
        assert!(ids.contains(&"333_333_333"), "new entry should exist");

        // Audio for old entry should be cleaned up.
        assert!(!ap.join("111_111_111.wav").exists(), "old audio should be deleted");
        assert!(ap.join("222_222_222.wav").exists(), "fresh audio should remain");
    }

    #[test]
    fn retention_zero_keeps_everything() {
        let hist_dir = tempfile::tempdir().unwrap();
        let audio_dir = tempfile::tempdir().unwrap();
        let hp = hist_dir.path();
        let ap = audio_dir.path();
        init_db(hp);

        let old_ts = now_ms() - 365 * 86_400_000; // 1 year ago
        add_entry(hp, ap, make_entry("111_111_111", old_ts), 0);

        let entries = load_history(hp);
        assert_eq!(entries.len(), 1, "retention_days=0 should keep all entries");
    }

    // ── History stats aggregation ──

    #[test]
    fn stats_aggregate_correctly() {
        let hist_dir = tempfile::tempdir().unwrap();
        let audio_dir = tempfile::tempdir().unwrap();
        let hp = hist_dir.path();
        let ap = audio_dir.path();
        init_db(hp);

        let now = now_ms();
        let mut e1 = make_entry("111_111_111", now);
        e1.duration_secs = 10.5;
        e1.word_count = 20;
        let mut e2 = make_entry("222_222_222", now - 1000);
        e2.duration_secs = 5.0;
        e2.word_count = 10;
        add_entry(hp, ap, e1, 0);
        add_entry(hp, ap, e2, 0);

        let stats = get_stats(hp);
        assert_eq!(stats.total_entries, 2);
        assert!((stats.total_duration_secs - 15.5).abs() < 0.01);
        assert_eq!(stats.total_words, 30);
    }
}

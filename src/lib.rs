mod audio;
mod commands;
mod context_detect;
mod credentials;
mod history;
mod hotkey;
mod permissions;
pub mod platform;
pub mod models;
mod polisher;
mod qwen3_asr;
pub mod settings;
pub mod stt;
mod transcribe;
pub mod whisper_models;

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Instant;
use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    AppHandle, Emitter, Manager,
};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

use unicode_segmentation::UnicodeSegmentation;

use commands::get_cached_api_key;
use hotkey::{hotkey_display_label, parse_hotkey_string};
use settings::{load_settings, models_dir, history_dir, audio_dir, logs_dir, Settings};
use stt::{SttConfig, SttMode};

const MAX_RECORDING_SECS: u64 = 120;

// ── App State ───────────────────────────────────────────────────────────────

pub struct AppState {
    pub is_recording: Arc<AtomicBool>,
    pub is_processing: AtomicBool,
    pub buffer: Arc<Mutex<Vec<f32>>>,
    pub sample_rate: Mutex<Option<u32>>,
    pub settings: Mutex<Settings>,
    pub mic_available: AtomicBool,
    pub whisper_ctx: Mutex<Option<transcribe::WhisperContextCache>>,
    pub llm_model: Mutex<Option<polisher::LlmModelCache>>,
    pub captured_context: Mutex<Option<context_detect::AppContext>>,
    pub context_override: Mutex<Option<context_detect::AppContext>>,
    pub test_mode: AtomicBool,
    pub voice_rule_mode: AtomicBool,
    pub last_hotkey_time: Mutex<Instant>,
    pub http_client: reqwest::blocking::Client,
    pub api_key_cache: Mutex<HashMap<String, String>>,
    pub edit_mode: AtomicBool,
    pub edit_selected_text: Mutex<Option<String>>,
    pub edit_text_override: Mutex<Option<String>>,
    pub saved_clipboard: Mutex<Option<String>>,
    pub vad_ctx: Mutex<Option<transcribe::VadContextCache>>,
    pub downloading: AtomicBool,
    pub audio_thread: Mutex<Option<audio::AudioThreadControl>>,
    pub qwen3_asr_ctx: Mutex<Option<qwen3_asr::Qwen3AsrCache>>,
    pub model_switching: AtomicBool,
    pub reconnecting: AtomicBool,
    pub streaming_active: AtomicBool,
    pub streaming_result: Mutex<Option<String>>,
    pub meeting_active: AtomicBool,
    pub meeting_transcript: Mutex<String>,
    pub meeting_start_time: Mutex<Option<std::time::Instant>>,
}

/// Restore original clipboard content from saved_clipboard.
fn restore_clipboard(state: &AppState) {
    if let Ok(mut saved) = state.saved_clipboard.lock() {
        if let Some(original) = saved.take() {
            std::thread::sleep(std::time::Duration::from_millis(200));
            if let Ok(mut clipboard) = arboard::Clipboard::new() {
                let _ = clipboard.set_text(&original);
            }
        }
    }
}

/// Hide overlay after a delay (in ms). 0 means hide immediately.
fn hide_overlay_delayed(app: &AppHandle, delay_ms: u64) {
    let app_handle = app.clone();
    std::thread::spawn(move || {
        if delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        }
        let app_for_hide = app_handle.clone();
        let _ = app_handle.run_on_main_thread(move || {
            if let Some(overlay) = app_for_hide.get_webview_window("overlay") {
                platform::hide_overlay(&overlay);
            }
        });
    });
}

/// Shared logic: stop recording, transcribe, copy/paste, and hide the overlay.
fn stop_transcribe_and_paste(app: &AppHandle) {
    let state = app.state::<AppState>();
    if state
        .is_processing
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        tracing::info!("stop_transcribe_and_paste: already processing, skipping");
        return;
    }

    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.emit("recording-status", "transcribing");
    }
    {
        let vrm = app.state::<AppState>();
        if vrm.voice_rule_mode.load(Ordering::SeqCst) {
            if let Some(main_win) = app.get_webview_window("main") {
                let _ = main_win.emit("voice-rule-status", "transcribing");
            }
        }
    }

    tracing::info!("⏹️ Stopping recording...");

    let app_handle = app.clone();
    std::thread::spawn(move || {
        let pipeline_start = Instant::now();
        let state = app_handle.state::<AppState>();

        let (auto_paste, polish_config, retention_days, mut stt_config) = state
            .settings
            .lock()
            .map(|s| (s.auto_paste, s.polish.clone(), s.history_retention_days, s.stt.clone()))
            .unwrap_or((true, polisher::PolishConfig::default(), 0, SttConfig::default()));

        if stt_config.mode == SttMode::Cloud {
            let key = get_cached_api_key(&state.api_key_cache, stt_config.cloud.provider.as_key());
            if !key.is_empty() {
                stt_config.cloud.api_key = key;
            }
        }

        let stt_language = stt_config.language.clone();
        let stt_app_name = state
            .captured_context
            .lock()
            .ok()
            .and_then(|ctx| ctx.as_ref().map(|c| c.app_name.clone()))
            .unwrap_or_default();
        let dictionary_terms = polish_config.dictionary.enabled_terms();

        match audio::do_stop_recording(
            &state.is_recording,
            &state.sample_rate,
            &state.buffer,
            &state.whisper_ctx,
            &state.qwen3_asr_ctx,
            &state.http_client,
            &stt_config,
            &stt_language,
            &stt_app_name,
            &dictionary_terms,
            &state.vad_ctx,
            stt_config.vad_enabled,
            &state.streaming_active,
            &state.streaming_result,
        ) {
            Ok((text, samples_16k)) => {
                let transcribe_elapsed = pipeline_start.elapsed();
                tracing::info!("[timing] stop→transcribed: {:.0?} | len: {} graphemes", transcribe_elapsed, text.graphemes(true).count());

                // Voice Rule Mode
                if state.voice_rule_mode.compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    tracing::info!("Voice rule mode: emitting transcript to main window");
                    if let Some(main_win) = app_handle.get_webview_window("main") {
                        let _ = main_win.emit("voice-rule-transcript", &text);
                    }
                    state.is_processing.store(false, Ordering::SeqCst);
                    let app_for_hide = app_handle.clone();
                    let _ = app_handle.run_on_main_thread(move || {
                        if let Some(overlay) = app_for_hide.get_webview_window("overlay") {
                            platform::hide_overlay(&overlay);
                        }
                    });
                    return;
                }

                let raw_text = text.clone();
                let audio_duration_secs = samples_16k.len() as f64 / 16000.0;

                let grapheme_count = text.graphemes(true).count();
                let stt_secs = transcribe_elapsed.as_secs_f64();
                let chars_per_sec = if stt_secs > 0.0 {
                    grapheme_count as f64 / stt_secs
                } else {
                    0.0
                };
                tracing::info!(
                    "[stats] STT output: {} graphemes in {:.2}s = {:.1} graphemes/sec",
                    grapheme_count, stt_secs, chars_per_sec
                );

                // AI Polishing
                let mut polish_config = polish_config;
                if polish_config.enabled && polish_config.mode == polisher::PolishMode::Cloud {
                    let key = get_cached_api_key(&state.api_key_cache, polish_config.cloud.provider.as_key());
                    if !key.is_empty() {
                        polish_config.cloud.api_key = key;
                    }
                }
                let stt_elapsed_ms = transcribe_elapsed.as_millis() as u64;

                let history_context = state
                    .captured_context
                    .lock()
                    .ok()
                    .and_then(|c| c.clone())
                    .unwrap_or_default();

                let (final_text, reasoning, polish_elapsed_ms) = if polish_config.enabled {
                    let model_dir = models_dir();
                    if polisher::is_polish_ready(&model_dir, &polish_config) {
                        if let Some(overlay) = app_handle.get_webview_window("overlay") {
                            let _ = overlay.emit("recording-status", "polishing");
                        }
                        let mode_label = match polish_config.mode {
                            polisher::PolishMode::Cloud => format!("Cloud ({})", polish_config.cloud.model_id),
                            polisher::PolishMode::Local => format!("Local ({})", polish_config.model.display_name()),
                        };
                        let context = state
                            .captured_context
                            .lock()
                            .ok()
                            .and_then(|mut c| c.take())
                            .unwrap_or_default();

                        let polish_start = Instant::now();
                        let result = polisher::polish_text(
                            &state.llm_model,
                            &model_dir,
                            &polish_config,
                            &context,
                            &text,
                            &state.http_client,
                        );
                        let p_elapsed = polish_start.elapsed().as_millis() as u64;
                        tracing::info!("[timing] polish ({}): {:.0?} | len: {} graphemes", mode_label, polish_start.elapsed(), result.text.graphemes(true).count());
                        (result.text, result.reasoning, Some(p_elapsed))
                    } else {
                        tracing::warn!("Polish enabled but not ready (model missing or no API key), skipping");
                        (text, None, None)
                    }
                } else {
                    (text, None, None)
                };
                let text = final_text;

                if let Some(main_win) = app_handle.get_webview_window("main") {
                    let _ = main_win.emit("transcription-result", &text);
                }

                let clipboard_ok = match arboard::Clipboard::new() {
                    Ok(mut clipboard) => {
                        if let Err(e) = clipboard.set_text(&text) {
                            tracing::error!("Clipboard error: {}", e);
                            false
                        } else {
                            true
                        }
                    }
                    Err(e) => {
                        tracing::error!("Clipboard init error: {}", e);
                        false
                    }
                };

                if clipboard_ok {
                    std::thread::sleep(std::time::Duration::from_millis(100));

                    if auto_paste {
                        let pasted = platform::simulate_paste();
                        if pasted {
                            tracing::info!("📋 Auto-pasted at cursor");
                            if let Some(overlay) = app_handle.get_webview_window("overlay") {
                                let _ = overlay.emit("recording-status", "pasted");
                            }
                        } else {
                            tracing::info!("📋 Copied to clipboard (paste simulation failed)");
                            if let Some(overlay) = app_handle.get_webview_window("overlay") {
                                let _ = overlay.emit("recording-status", "copied");
                            }
                        }
                    } else {
                        tracing::info!("📋 Copied to clipboard (auto-paste disabled)");
                        if let Some(overlay) = app_handle.get_webview_window("overlay") {
                            let _ = overlay.emit("recording-status", "copied");
                        }
                    }
                }

                let total_elapsed_ms = pipeline_start.elapsed().as_millis() as u64;
                tracing::info!("[timing] total pipeline: {:.0?}", pipeline_start.elapsed());

                // Save to history
                {
                    let entry_id = history::generate_id();
                    let stt_model = match stt_config.mode {
                        SttMode::Cloud => {
                            format!("{} (Cloud/{})", stt_config.cloud.model_id, stt_config.cloud.provider.as_key())
                        }
                        SttMode::Local => match stt_config.local_engine {
                            stt::LocalSttEngine::Whisper => stt_config.whisper_model.display_name().to_string(),
                            stt::LocalSttEngine::Qwen3Asr => stt_config.qwen3_asr_model.display_name().to_string(),
                        },
                    };
                    let polish_model_name = if polish_elapsed_ms.is_some() {
                        match polish_config.mode {
                            polisher::PolishMode::Cloud => {
                                format!("{} (Cloud/{})", polish_config.cloud.model_id, polish_config.cloud.provider.as_key())
                            }
                            polisher::PolishMode::Local => {
                                format!("{} (Local)", polish_config.model.display_name())
                            }
                        }
                    } else {
                        "None".to_string()
                    };
                    let has_audio = history::save_audio_wav(&audio_dir(), &entry_id, &samples_16k);
                    let word_count = history::count_words(&text) as u64;
                    let entry = history::HistoryEntry {
                        id: entry_id,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::SystemTime::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as i64,
                        text: text.clone(),
                        raw_text,
                        reasoning,
                        stt_model,
                        polish_model: polish_model_name,
                        duration_secs: audio_duration_secs,
                        has_audio,
                        stt_elapsed_ms,
                        polish_elapsed_ms,
                        total_elapsed_ms,
                        app_name: history_context.app_name.clone(),
                        bundle_id: history_context.bundle_id.clone(),
                        chars_per_sec,
                        word_count,
                    };
                    history::add_entry(&history_dir(), &audio_dir(), entry, retention_days);
                    tracing::info!("📝 History entry saved (audio={})", has_audio);
                }
            }
            Err(ref e) if e == "no_speech" => {
                tracing::info!("No speech detected, skipping (took {:.0?})", pipeline_start.elapsed());
                if state.voice_rule_mode.compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    if let Some(main_win) = app_handle.get_webview_window("main") {
                        let _ = main_win.emit("voice-rule-transcript", "");
                    }
                }
                // Release immediately and hide overlay — nothing to display
                state.is_processing.store(false, Ordering::SeqCst);
                let app_for_hide = app_handle.clone();
                let _ = app_handle.run_on_main_thread(move || {
                    if let Some(overlay) = app_for_hide.get_webview_window("overlay") {
                        platform::hide_overlay(&overlay);
                    }
                });
                return;
            }
            Err(e) => {
                tracing::error!("Transcription error: {} (after {:.0?})", e, pipeline_start.elapsed());
                if let Some(overlay) = app_handle.get_webview_window("overlay") {
                    let _ = overlay.emit("recording-status", "error");
                }
                state.voice_rule_mode.store(false, Ordering::SeqCst);
            }
        }

        state.is_processing.store(false, Ordering::SeqCst);

        std::thread::sleep(std::time::Duration::from_millis(1500));
        let app_for_hide = app_handle.clone();
        let _ = app_handle.run_on_main_thread(move || {
            if let Some(overlay) = app_for_hide.get_webview_window("overlay") {
                platform::hide_overlay(&overlay);
            }
        });
    });
}

/// Edit-by-voice pipeline: stop recording, transcribe instruction, edit text, replace.
fn stop_edit_and_replace(app: &AppHandle) {
    let state = app.state::<AppState>();
    if state
        .is_processing
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        tracing::info!("stop_edit_and_replace: already processing, skipping");
        return;
    }

    state.edit_mode.store(false, Ordering::SeqCst);

    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.emit("recording-status", "transcribing");
    }

    tracing::info!("⏹️ Stopping edit-by-voice recording...");

    let app_handle = app.clone();
    std::thread::spawn(move || {
        let pipeline_start = Instant::now();
        let state = app_handle.state::<AppState>();

        let (polish_config, mut stt_config) = state
            .settings
            .lock()
            .map(|s| (s.polish.clone(), s.stt.clone()))
            .unwrap_or((polisher::PolishConfig::default(), SttConfig::default()));

        if stt_config.mode == SttMode::Cloud {
            let key = get_cached_api_key(&state.api_key_cache, stt_config.cloud.provider.as_key());
            if !key.is_empty() {
                stt_config.cloud.api_key = key;
            }
        }

        let selected_text = state
            .edit_selected_text
            .lock()
            .ok()
            .and_then(|mut t| t.take())
            .unwrap_or_default();

        if selected_text.is_empty() {
            tracing::warn!("Edit-by-voice: no selected text");
            if let Some(overlay) = app_handle.get_webview_window("overlay") {
                let _ = overlay.emit("recording-status", "error");
            }
            state.is_processing.store(false, Ordering::SeqCst);
            restore_clipboard(&state);
            hide_overlay_delayed(&app_handle, 1500);
            return;
        }

        let edit_stt_language = stt_config.language.clone();
        let edit_app_name = state
            .captured_context
            .lock()
            .ok()
            .and_then(|ctx| ctx.as_ref().map(|c| c.app_name.clone()))
            .unwrap_or_default();
        let edit_dict_terms = polish_config.dictionary.enabled_terms();

        match audio::do_stop_recording(
            &state.is_recording,
            &state.sample_rate,
            &state.buffer,
            &state.whisper_ctx,
            &state.qwen3_asr_ctx,
            &state.http_client,
            &stt_config,
            &edit_stt_language,
            &edit_app_name,
            &edit_dict_terms,
            &state.vad_ctx,
            stt_config.vad_enabled,
            &state.streaming_active,
            &state.streaming_result,
        ) {
            Ok((instruction, _samples)) => {
                tracing::info!("Edit instruction received: {} graphemes", instruction.graphemes(true).count());

                if let Some(overlay) = app_handle.get_webview_window("overlay") {
                    let _ = overlay.emit("recording-status", "polishing");
                }

                let mut polish_config = polish_config;
                if polish_config.mode == polisher::PolishMode::Cloud {
                    let key = get_cached_api_key(
                        &state.api_key_cache,
                        polish_config.cloud.provider.as_key(),
                    );
                    if !key.is_empty() {
                        polish_config.cloud.api_key = key;
                    }
                }

                let model_dir = models_dir();
                if !polisher::is_polish_ready(&model_dir, &polish_config) {
                    tracing::warn!("Edit-by-voice: LLM not configured");
                    if let Some(overlay) = app_handle.get_webview_window("overlay") {
                        let _ = overlay.emit("recording-status", "error");
                    }
                    state.is_processing.store(false, Ordering::SeqCst);
                    restore_clipboard(&state);
                    hide_overlay_delayed(&app_handle, 1500);
                    return;
                }

                match polisher::edit_text_by_instruction(
                    &state.llm_model,
                    &model_dir,
                    &polish_config,
                    &selected_text,
                    &instruction,
                    &state.http_client,
                ) {
                    Ok(edited_text) => {
                        tracing::info!(
                            "Edit result: {} graphemes (took {:.0?})",
                            edited_text.graphemes(true).count(),
                            pipeline_start.elapsed()
                        );

                        let clipboard_ok = match arboard::Clipboard::new() {
                            Ok(mut clipboard) => clipboard.set_text(&edited_text).is_ok(),
                            Err(_) => false,
                        };

                        if clipboard_ok {
                            std::thread::sleep(std::time::Duration::from_millis(100));
                            platform::simulate_paste();
                            tracing::info!("✏️ Edited text pasted");
                        }

                        restore_clipboard(&state);

                        if let Some(overlay) = app_handle.get_webview_window("overlay") {
                            let _ = overlay.emit("recording-status", "edited");
                        }

                        state.is_processing.store(false, Ordering::SeqCst);
                        hide_overlay_delayed(&app_handle, 5500);
                    }
                    Err(e) => {
                        tracing::error!("Edit-by-voice LLM error: {}", e);
                        if let Some(overlay) = app_handle.get_webview_window("overlay") {
                            let _ = overlay.emit("recording-status", "error");
                        }
                        state.is_processing.store(false, Ordering::SeqCst);
                        restore_clipboard(&state);
                        hide_overlay_delayed(&app_handle, 1500);
                    }
                }
            }
            Err(ref e) if e == "no_speech" => {
                tracing::info!("Edit-by-voice: no speech detected");
                state.is_processing.store(false, Ordering::SeqCst);
                restore_clipboard(&state);
                hide_overlay_delayed(&app_handle, 0);
            }
            Err(e) => {
                tracing::error!("Edit-by-voice transcription error: {}", e);
                if let Some(overlay) = app_handle.get_webview_window("overlay") {
                    let _ = overlay.emit("recording-status", "error");
                }
                state.is_processing.store(false, Ordering::SeqCst);
                restore_clipboard(&state);
                hide_overlay_delayed(&app_handle, 1500);
            }
        }
    });
}

// ── Logging helpers ──────────────────────────────────────────────────────────

/// Holds the WorkerGuard for the non-blocking file appender so it lives until process exit.
/// Dropping the guard shuts down the background writer thread and flushes any buffered logs.
#[cfg(not(debug_assertions))]
static LOG_GUARD: std::sync::Mutex<Option<tracing_appender::non_blocking::WorkerGuard>> =
    std::sync::Mutex::new(None);

/// Delete `sumi.log*` files in `log_dir` that have not been written to within `keep_days` days.
/// Also removes the legacy non-rotating `sumi.log` (written by older app versions).
#[cfg(not(debug_assertions))]
fn cleanup_old_logs(log_dir: &std::path::Path, keep_days: u64) {
    let cutoff = std::time::SystemTime::now()
        .checked_sub(std::time::Duration::from_secs(keep_days * 24 * 3600))
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    let Ok(entries) = std::fs::read_dir(log_dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_log = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with("sumi.log"))
            .unwrap_or(false);
        if !is_log { continue; }
        let mtime = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        if mtime < cutoff {
            let _ = std::fs::remove_file(&path);
        }
    }
}

// ── App Entry ───────────────────────────────────────────────────────────────

pub fn run() {
    // Silence whisper.cpp and GGML logs (no log_backend/tracing_backend → noop)
    whisper_rs::install_logging_hooks();

    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            show_settings_window(app);
        }))
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_os::init())
        .invoke_handler(tauri::generate_handler![
            commands::start_recording,
            commands::stop_recording,
            commands::cancel_recording,
            commands::set_test_mode,
            commands::set_voice_rule_mode,
            commands::set_context_override,
            commands::set_edit_text_override,
            commands::get_settings,
            commands::save_settings,
            commands::update_hotkey,
            commands::reset_settings,
            commands::get_default_prompt,
            commands::get_default_prompt_rules,
            commands::test_polish,
            commands::get_mic_status,
            commands::check_model_status,
            commands::download_model,
            commands::check_llm_model_status,
            commands::download_llm_model,
            commands::save_api_key,
            commands::get_api_key,
            commands::get_history_stats,
            commands::get_history,
            commands::get_history_page,
            commands::delete_history_entry,
            commands::clear_all_history,
            commands::export_history_audio,
            commands::get_history_storage_path,
            commands::get_app_icon,
            permissions::check_permissions,
            permissions::open_permission_settings,
            commands::generate_rule_from_description,
            commands::update_edit_hotkey,
            commands::trigger_undo,
            commands::list_polish_models,
            commands::switch_polish_model,
            commands::download_polish_model,
            commands::list_whisper_models,
            commands::get_system_info,
            commands::get_whisper_model_recommendation,
            commands::switch_whisper_model,
            commands::download_whisper_model,
            commands::check_vad_model_status,
            commands::download_vad_model,
            commands::copy_image_to_clipboard,
            commands::is_dev_mode,
            commands::set_mic_device,
            commands::export_diagnostic_log,
            commands::list_qwen3_asr_models,
            commands::switch_qwen3_asr_model,
            commands::download_qwen3_asr_model,
            commands::update_meeting_hotkey,
        ])
        .setup(|app| {
            // Initialize logger
            {
                let log_dir = logs_dir();
                let _ = std::fs::create_dir_all(&log_dir);

                #[cfg(debug_assertions)]
                {
                    // Dev: write to stderr so `cargo tauri dev` shows logs in the terminal.
                    if let Err(e) = tracing_subscriber::fmt()
                        .with_writer(std::io::stderr)
                        .with_ansi(true)
                        .with_target(false)
                        .try_init()
                    {
                        eprintln!("[Sumi] Logger init failed: {}", e);
                    }
                }

                #[cfg(not(debug_assertions))]
                {
                    // Release: daily-rotating file, keep 7 days, non-blocking writer.
                    cleanup_old_logs(&log_dir, 7);
                    let file_appender = tracing_appender::rolling::daily(&log_dir, "sumi.log");
                    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
                    match tracing_subscriber::fmt()
                        .with_writer(non_blocking)
                        .with_ansi(false)
                        .with_target(false)
                        .try_init()
                    {
                        Ok(()) => {
                            // Guard must stay alive until process exit so the background
                            // writer thread keeps running and flushes all logs on drop.
                            *LOG_GUARD.lock().unwrap_or_else(|e| e.into_inner()) = Some(guard);
                        }
                        Err(e) => {
                            eprintln!("[Sumi] File logger init failed: {}", e);
                            // guard dropped here — background thread exits cleanly.
                        }
                    }
                }

                std::panic::set_hook(Box::new(|info| {
                    tracing::error!("PANIC: {}", info);
                    eprintln!("[Sumi] PANIC: {}", info);
                }));
            }

            // Hide Dock icon (macOS) / equivalent
            platform::set_app_accessory_mode();

            // Load settings, then apply locale defaults.
            let mut settings = load_settings();
            settings::apply_locale_defaults(&mut settings);
            let hotkey_str = settings.hotkey.clone();

            // Migrate legacy JSON history to SQLite
            history::migrate_from_json(&history_dir(), &audio_dir());

            // Pre-initialise audio pipeline
            let is_recording = Arc::new(AtomicBool::new(false));
            let buffer = Arc::new(Mutex::new(Vec::new()));
            let (mic_available, sample_rate, audio_thread_init) =
                match audio::spawn_audio_thread(Arc::clone(&buffer), Arc::clone(&is_recording), settings.mic_device.clone()) {
                    Ok((sr, control)) => (true, Some(sr), Some(control)),
                    Err(e) => {
                        tracing::error!("Audio init failed: {}", e);
                        (false, None, None)
                    }
                };

            let http_client = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .unwrap_or_else(|_| reqwest::blocking::Client::new());

            app.manage(AppState {
                is_recording,
                is_processing: AtomicBool::new(false),
                buffer,
                sample_rate: Mutex::new(sample_rate),
                settings: Mutex::new(settings.clone()),
                mic_available: AtomicBool::new(mic_available),
                whisper_ctx: Mutex::new(None),
                llm_model: Mutex::new(None),
                captured_context: Mutex::new(None),
                context_override: Mutex::new(None),
                test_mode: AtomicBool::new(false),
                voice_rule_mode: AtomicBool::new(false),
                last_hotkey_time: Mutex::new(Instant::now() - std::time::Duration::from_secs(1)),
                http_client,
                api_key_cache: Mutex::new(HashMap::new()),
                edit_mode: AtomicBool::new(false),
                edit_selected_text: Mutex::new(None),
                edit_text_override: Mutex::new(None),
                saved_clipboard: Mutex::new(None),
                vad_ctx: Mutex::new(None),
                downloading: AtomicBool::new(false),
                audio_thread: Mutex::new(audio_thread_init),
                qwen3_asr_ctx: Mutex::new(None),
                model_switching: AtomicBool::new(false),
                reconnecting: AtomicBool::new(false),
                streaming_active: AtomicBool::new(false),
                streaming_result: Mutex::new(None),
                meeting_active: AtomicBool::new(false),
                meeting_transcript: Mutex::new(String::new()),
                meeting_start_time: Mutex::new(None),
            });

            // Register a CoreAudio listener for default-input-device changes.
            // When the user connects Bluetooth headphones, the system default input
            // may switch to them. The listener reconnects the cpal stream to the
            // built-in mic (via resolve_input_device), preventing an A2DP → HFP switch.
            {
                let app_for_listener = app.handle().clone();
                platform::add_default_input_listener(move || {
                    let state = app_for_listener.state::<AppState>();

                    // If the user chose an explicit mic device, never interfere.
                    let explicit = state.settings.lock()
                        .ok()
                        .and_then(|s| s.mic_device.clone());
                    if explicit.is_some() { return; }

                    // Don't interrupt an active recording.
                    if state.is_recording.load(Ordering::SeqCst) {
                        tracing::info!("Default input changed while recording — will apply on next start");
                        return;
                    }

                    // Only reconnect when the new default IS Bluetooth.
                    // If BT just disconnected and the default reverted to built-in,
                    // our cpal stream is already on the built-in mic — no action needed.
                    // (If the BT stream dies, the error callback will set stream_alive=false
                    //  and do_start_recording will trigger a lazy reconnect.)
                    if !platform::is_default_input_bluetooth() {
                        tracing::info!("Default input changed to non-BT device — stream already correct, skipping reconnect");
                        return;
                    }

                    tracing::info!("Default audio input device changed to Bluetooth — reconnecting to built-in mic");

                    // Guard against multiple concurrent reconnects: CoreAudio can fire
                    // 2-3 property-change notifications per physical hotplug event.
                    // If two threads both reach spawn_audio_thread, the second one leaks
                    // the first cpal stream (it keeps running, consuming resources forever).
                    if state.reconnecting.swap(true, Ordering::SeqCst) {
                        tracing::info!("Reconnect already in progress — skipping duplicate CoreAudio notification");
                        return;
                    }

                    // Tear down the old thread so try_reconnect_audio will spawn a fresh one.
                    if let Ok(mut at) = state.audio_thread.lock() {
                        if let Some(ctrl) = at.take() { ctrl.stop(); }
                    }
                    state.mic_available.store(false, Ordering::SeqCst);

                    // Dispatch to a background thread — CoreAudio HAL callbacks must return
                    // quickly; try_reconnect_audio blocks for up to 5 s (recv_timeout).
                    let app2 = app_for_listener.clone();
                    std::thread::spawn(move || {
                        let state = app2.state::<AppState>();
                        let result = audio::try_reconnect_audio(
                            &state.mic_available,
                            &state.sample_rate,
                            &state.buffer,
                            &state.is_recording,
                            &state.audio_thread,
                            None,
                        );
                        // Always reset the guard so future events can trigger reconnect.
                        state.reconnecting.store(false, Ordering::SeqCst);
                        match result {
                            Ok(()) => tracing::info!("Mic stream reconnected after input device change"),
                            Err(e) => tracing::error!("Mic stream reconnect failed: {}", e),
                        }
                    });
                });
            }

            // Migration: if old zh-TW model exists but settings use default (LargeV3Turbo)
            // and the LargeV3Turbo model file doesn't exist, switch to LargeV3TurboZhTw
            {
                let state = app.state::<AppState>();
                let current_model = state.settings.lock()
                    .map(|s| s.stt.whisper_model.clone())
                    .unwrap_or_default();
                if current_model == whisper_models::WhisperModel::LargeV3Turbo {
                    let default_path = models_dir().join(whisper_models::WhisperModel::LargeV3Turbo.filename());
                    let legacy_path = models_dir().join("ggml-large-v3-turbo-zh-TW.bin");
                    if !default_path.exists() && legacy_path.exists() {
                        tracing::info!("Migrating whisper model setting: LargeV3Turbo → LargeV3TurboZhTw (legacy file exists)");
                        if let Ok(mut guard) = state.settings.lock() {
                            guard.stt.whisper_model = whisper_models::WhisperModel::LargeV3TurboZhTw;
                            settings::save_settings_to_disk(&guard);
                        }
                    }
                }
            }

            // Auto-show settings when active whisper model is missing
            {
                let active_model = app.state::<AppState>().settings.lock()
                    .map(|s| s.stt.whisper_model.clone())
                    .unwrap_or_default();
                if !models_dir().join(active_model.filename()).exists() {
                    show_settings_window(app.handle());
                }
            }

            // Pre-warm models in background
            {
                let app_handle = app.handle().clone();
                std::thread::spawn(move || {
                    let warmup_start = Instant::now();
                    let state = app_handle.state::<AppState>();

                    let (stt_mode, whisper_model, local_engine, qwen3_model) = state.settings.lock()
                        .map(|s| (s.stt.mode.clone(), s.stt.whisper_model.clone(), s.stt.local_engine.clone(), s.stt.qwen3_asr_model.clone()))
                        .unwrap_or_default();
                    if stt_mode == SttMode::Local {
                        match local_engine {
                            stt::LocalSttEngine::Whisper => {
                                if transcribe::whisper_model_path_for(&whisper_model).is_ok() {
                                    if let Err(e) = transcribe::warm_whisper_cache(&state.whisper_ctx, &whisper_model) {
                                        tracing::error!("Whisper pre-warm failed: {}", e);
                                    }
                                }
                            }
                            stt::LocalSttEngine::Qwen3Asr => {
                                if stt::is_qwen3_asr_downloaded(&qwen3_model) {
                                    if let Err(e) = qwen3_asr::warm_qwen3_asr(&state.qwen3_asr_ctx, &qwen3_model) {
                                        tracing::error!("Qwen3-ASR pre-warm failed: {}", e);
                                    }
                                }
                            }
                        }
                    }

                    // Pre-warm LLM if polish is local and model exists.
                    // validate_gguf_file() catches corrupted files before loading,
                    // mitigating the SIGSEGV concern that previously motivated lazy loading.
                    let (polish_mode, polish_model) = state.settings.lock()
                        .map(|s| (s.polish.mode.clone(), s.polish.model.clone()))
                        .unwrap_or_default();
                    if polish_mode == polisher::PolishMode::Local {
                        let model_dir = models_dir();
                        if model_dir.join(polish_model.filename()).exists() {
                            if let Err(e) = polisher::warm_llm_cache(&state.llm_model, &model_dir, &polish_model) {
                                tracing::error!("LLM pre-warm failed: {}", e);
                            }
                        }
                    }

                    tracing::info!("All models pre-warmed ({:.0?} total)", warmup_start.elapsed());
                });
            }

            // System Tray
            let settings_i =
                MenuItem::with_id(app, "settings", "Settings...", true, None::<&str>)?;
            let quit_label = if settings::is_debug() { "Quit Sumi (Dev)" } else { "Quit Sumi" };
            let quit_i =
                MenuItem::with_id(app, "quit", quit_label, true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&settings_i, &quit_i])?;

            let tooltip_label = hotkey_display_label(&hotkey_str);
            let _tray = TrayIconBuilder::with_id("main-tray")
                .icon(tauri::image::Image::from_bytes(include_bytes!("../icons/tray-icon.png"))
                    .expect("tray-icon.png is compile-time embedded and must be a valid PNG"))
                .menu(&menu)
                .show_menu_on_left_click(false)
                .tooltip(if settings::is_debug() {
                    format!("Sumi [Dev] – {} to record", tooltip_label)
                } else {
                    format!("Sumi – {} to record", tooltip_label)
                })
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "settings" => {
                        show_settings_window(app);
                    }
                    "quit" => {
                        app.exit(0);
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray: &tauri::tray::TrayIcon, event| {
                    if let tauri::tray::TrayIconEvent::Click {
                        button: tauri::tray::MouseButton::Left,
                        ..
                    } = event
                    {
                        show_settings_window(tray.app_handle());
                    }
                })
                .build(app)?;

            // Window close → hide; enable drag-by-background for overlay title bar
            if let Some(main_window) = app.get_webview_window("main") {
                platform::set_main_window_movable(&main_window);
                let win = main_window.clone();
                main_window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        let _ = win.hide();
                    }
                });
            }

            // Configure overlay
            if let Some(overlay) = app.get_webview_window("overlay") {
                platform::setup_overlay_window(&overlay);
            }

            // Global Shortcut
            #[cfg(desktop)]
            {
                let fallback_shortcut = if settings::is_debug() {
                    Shortcut::new(Some(Modifiers::ALT | Modifiers::SUPER), Code::KeyZ)
                } else {
                    Shortcut::new(Some(Modifiers::ALT | Modifiers::SUPER), Code::KeyR)
                };
                let primary_shortcut = parse_hotkey_string(&hotkey_str)
                    .unwrap_or(fallback_shortcut);
                let edit_shortcut = settings.edit_hotkey.as_deref().and_then(parse_hotkey_string);
                let meeting_shortcut = settings.meeting_hotkey.as_deref().and_then(parse_hotkey_string);

                app.handle().plugin(
                    tauri_plugin_global_shortcut::Builder::new()
                        .with_handler(move |app, shortcut, event| {
                            if event.state() != ShortcutState::Pressed {
                                return;
                            }

                            let state = app.state::<AppState>();

                            let is_meeting_hotkey = state.settings.lock()
                                .ok()
                                .and_then(|s| s.meeting_hotkey.as_deref().and_then(parse_hotkey_string))
                                .is_some_and(|ms| *shortcut == ms);

                            let is_edit_hotkey = state.settings.lock()
                                .ok()
                                .and_then(|s| s.edit_hotkey.as_deref().and_then(parse_hotkey_string))
                                .is_some_and(|es| *shortcut == es);

                            if state.test_mode.load(Ordering::SeqCst) {
                                if let Some(main_win) = app.get_webview_window("main") {
                                    let _ = main_win.emit("hotkey-activated", true);
                                }
                                return;
                            }

                            // Debounce
                            {
                                let now = Instant::now();
                                if let Ok(mut last) = state.last_hotkey_time.lock() {
                                    if now.duration_since(*last) < std::time::Duration::from_millis(300) {
                                        return;
                                    }
                                    *last = now;
                                }
                            }

                            if state.is_processing.load(Ordering::SeqCst) {
                                return;
                            }

                            if state.model_switching.load(Ordering::SeqCst) {
                                return;
                            }

                            // Meeting hotkey: toggle meeting mode independently.
                            if is_meeting_hotkey {
                                if state.meeting_active.load(Ordering::SeqCst) {
                                    let app_clone = app.clone();
                                    std::thread::spawn(move || stop_meeting_mode(&app_clone));
                                } else {
                                    start_meeting_mode(app);
                                }
                                return;
                            }

                            // Block regular hotkeys while meeting is active.
                            if state.meeting_active.load(Ordering::SeqCst) {
                                return;
                            }

                            let is_recording = state.is_recording.load(Ordering::SeqCst);

                            if !is_recording {
                                // Start Recording

                                // For edit hotkey: check polish readiness before anything else
                                if is_edit_hotkey {
                                    let mut polish_config = state.settings.lock()
                                        .map(|s| s.polish.clone())
                                        .unwrap_or_default();
                                    if polish_config.mode == polisher::PolishMode::Cloud {
                                        let key = get_cached_api_key(
                                            &state.api_key_cache,
                                            polish_config.cloud.provider.as_key(),
                                        );
                                        if !key.is_empty() {
                                            polish_config.cloud.api_key = key;
                                        }
                                    }
                                    let model_dir = models_dir();
                                    if !polish_config.enabled || !polisher::is_polish_ready(&model_dir, &polish_config) {
                                        tracing::info!("Edit-by-voice: polish not ready, showing overlay hint");
                                        if let Some(overlay) = app.get_webview_window("overlay") {
                                            let _ = overlay.emit("recording-status", "edit_requires_polish");
                                            if let Ok(Some(monitor)) = overlay.current_monitor() {
                                                let screen = monitor.size();
                                                let scale = monitor.scale_factor();
                                                let win_w = 300.0;
                                                let win_h = 40.0;
                                                let x = (screen.width as f64 / scale - win_w) / 2.0;
                                                let y = screen.height as f64 / scale - win_h - 80.0;
                                                let _ = overlay.set_position(
                                                    tauri::PhysicalPosition::new(
                                                        (x * scale) as i32,
                                                        (y * scale) as i32,
                                                    ),
                                                );
                                            }
                                            platform::show_overlay(&overlay);
                                        }
                                        hide_overlay_delayed(app, 2000);
                                        return;
                                    }

                                    let override_text = state.edit_text_override.lock()
                                        .ok()
                                        .and_then(|mut ov| ov.take());

                                    if let Some(text) = override_text {
                                        if text.is_empty() {
                                            tracing::warn!("Edit-by-voice: override text is empty, aborting");
                                            return;
                                        }
                                        let grapheme_count = text.graphemes(true).count();
                                        if let Ok(mut et) = state.edit_selected_text.lock() {
                                            *et = Some(text);
                                        }
                                        state.edit_mode.store(true, Ordering::SeqCst);
                                        tracing::info!("✏️ Edit-by-voice (override): captured {} graphemes", grapheme_count);
                                    } else {
                                        // Save original clipboard for later restoration
                                        let original_clipboard = arboard::Clipboard::new()
                                            .ok()
                                            .and_then(|mut cb| cb.get_text().ok());
                                        if let Ok(mut saved) = state.saved_clipboard.lock() {
                                            *saved = original_clipboard;
                                        }

                                        // Record change count before copy (macOS/Windows)
                                        let change_count_before = platform::clipboard_change_count();

                                        // On platforms without change count (Linux), write a sentinel
                                        // so we can detect whether Ctrl+C actually fired
                                        let sentinel_str: Option<String> = if change_count_before.is_none() {
                                            let s = format!("__sumi_sentinel_{}__",
                                                std::time::SystemTime::now()
                                                    .duration_since(std::time::UNIX_EPOCH)
                                                    .unwrap_or_default()
                                                    .as_nanos());
                                            if let Ok(mut cb) = arboard::Clipboard::new() {
                                                let _ = cb.set_text(&s);
                                            }
                                            std::thread::sleep(std::time::Duration::from_millis(30));
                                            Some(s)
                                        } else {
                                            None
                                        };

                                        platform::simulate_copy();
                                        std::thread::sleep(std::time::Duration::from_millis(100));

                                        // Determine whether the clipboard was actually updated
                                        let clipboard_changed = match change_count_before {
                                            Some(before) => {
                                                // macOS / Windows: compare sequence numbers
                                                platform::clipboard_change_count()
                                                    .map(|after| after != before)
                                                    .unwrap_or(false)
                                            }
                                            None => {
                                                // Linux / fallback: check the clipboard differs from sentinel
                                                let current = arboard::Clipboard::new()
                                                    .ok()
                                                    .and_then(|mut cb| cb.get_text().ok())
                                                    .unwrap_or_default();
                                                let sentinel = sentinel_str.as_deref().unwrap_or("");
                                                !current.is_empty() && current != sentinel
                                            }
                                        };

                                        if !clipboard_changed {
                                            tracing::info!("Edit-by-voice: no text selected, aborting");
                                            restore_clipboard(&state);
                                            return;
                                        }

                                        let selected = arboard::Clipboard::new()
                                            .ok()
                                            .and_then(|mut cb| cb.get_text().ok())
                                            .unwrap_or_default();

                                        if selected.is_empty() {
                                            tracing::warn!("Edit-by-voice: clipboard empty after copy, aborting");
                                            restore_clipboard(&state);
                                            return;
                                        }

                                        let grapheme_count = selected.graphemes(true).count();
                                        if let Ok(mut et) = state.edit_selected_text.lock() {
                                            *et = Some(selected);
                                        }
                                        state.edit_mode.store(true, Ordering::SeqCst);
                                        tracing::info!("✏️ Edit-by-voice: captured {} graphemes", grapheme_count);
                                    }
                                }

                                let captured_ctx = state.context_override.lock()
                                    .ok()
                                    .and_then(|ctx| ctx.clone())
                                    .unwrap_or_else(context_detect::detect_frontmost_app);

                                let preferred_device = state.settings.lock()
                                    .ok()
                                    .and_then(|s| s.mic_device.clone());
                                match audio::do_start_recording(
                                    &state.is_recording,
                                    &state.mic_available,
                                    &state.sample_rate,
                                    &state.buffer,
                                    &state.is_recording,
                                    &state.audio_thread,
                                    preferred_device,
                                ) {
                                    Ok(()) => {
                                        tracing::info!("🎙️ Recording started (app: {:?}, bundle: {:?}, url: {:?})",
                                            captured_ctx.app_name, captured_ctx.bundle_id, captured_ctx.url);

                                        // Recording-start warm: load models in parallel with the user speaking.
                                        // If startup pre-warm already finished, the guard in each warm function
                                        // makes this a no-op. If the startup warm is still running, the mutex
                                        // serialises the two threads so no double-load occurs.
                                        {
                                            let warm_app = app.clone();
                                            let (stt_mode, whisper_model, local_engine, qwen3_model, polish_mode, polish_model) =
                                                state.settings.lock()
                                                    .map(|s| (
                                                        s.stt.mode.clone(),
                                                        s.stt.whisper_model.clone(),
                                                        s.stt.local_engine.clone(),
                                                        s.stt.qwen3_asr_model.clone(),
                                                        s.polish.mode.clone(),
                                                        s.polish.model.clone(),
                                                    ))
                                                    .unwrap_or_default();
                                            std::thread::spawn(move || {
                                                let state = warm_app.state::<AppState>();
                                                if stt_mode == SttMode::Local {
                                                    match local_engine {
                                                        stt::LocalSttEngine::Whisper => {
                                                            if let Err(e) = transcribe::warm_whisper_cache(
                                                                &state.whisper_ctx, &whisper_model,
                                                            ) {
                                                                tracing::warn!("Recording-start warm (Whisper) skipped: {}", e);
                                                            }
                                                        }
                                                        stt::LocalSttEngine::Qwen3Asr => {
                                                            if let Err(e) = qwen3_asr::warm_qwen3_asr(
                                                                &state.qwen3_asr_ctx, &qwen3_model,
                                                            ) {
                                                                tracing::warn!("Recording-start warm (Qwen3-ASR) skipped: {}", e);
                                                            }
                                                        }
                                                    }
                                                }
                                                if polish_mode == polisher::PolishMode::Local {
                                                    let model_dir = models_dir();
                                                    if model_dir.join(polish_model.filename()).exists() {
                                                        if let Err(e) = polisher::warm_llm_cache(
                                                            &state.llm_model, &model_dir, &polish_model,
                                                        ) {
                                                            tracing::warn!("Recording-start warm (LLM) skipped: {}", e);
                                                        }
                                                    }
                                                }
                                            });
                                        }

                                        // ── Live-preview feeder (Qwen3-ASR non-edit mode only) ──
                                        let should_stream = !is_edit_hotkey && {
                                            state.settings.lock()
                                                .map(|s| s.stt.mode == SttMode::Local
                                                    && s.stt.local_engine == stt::LocalSttEngine::Qwen3Asr)
                                                .unwrap_or(false)
                                        };
                                        if should_stream {
                                            state.streaming_active.store(true, Ordering::SeqCst);
                                            if let Ok(mut r) = state.streaming_result.lock() {
                                                *r = None;
                                            }
                                            let feeder_app = app.clone();
                                            let (feeder_model, feeder_lang) = state.settings.lock()
                                                .map(|s| (s.stt.qwen3_asr_model.clone(), s.stt.language.clone()))
                                                .unwrap_or_default();
                                            std::thread::spawn(move || {
                                                let fstate = feeder_app.state::<AppState>();
                                                // Wait for engine warm (max 8s)
                                                let mut waited = 0u64;
                                                while waited < 8000 {
                                                    let ready = fstate.qwen3_asr_ctx.lock().ok()
                                                        .and_then(|g| g.as_ref().map(|c| c.model == feeder_model))
                                                        .unwrap_or(false);
                                                    if ready { break; }
                                                    std::thread::sleep(std::time::Duration::from_millis(200));
                                                    waited += 200;
                                                }
                                                let ready = fstate.qwen3_asr_ctx.lock().ok()
                                                    .and_then(|g| g.as_ref().map(|c| c.model == feeder_model))
                                                    .unwrap_or(false);
                                                if !ready {
                                                    fstate.streaming_active.store(false, Ordering::SeqCst);
                                                    return;
                                                }
                                                qwen3_asr::run_feeder_loop(feeder_app, feeder_lang);
                                            });
                                        }

                                        if let Ok(mut ctx) = state.captured_context.lock() {
                                            *ctx = Some(captured_ctx);
                                        }

                                        if let Some(main_win) = app.get_webview_window("main") {
                                            let _ = main_win.emit("hotkey-activated", true);
                                            if state.voice_rule_mode.load(Ordering::SeqCst) {
                                                let _ = main_win.emit("voice-rule-status", "recording");
                                            }
                                        }

                                        if let Some(overlay) = app.get_webview_window("overlay") {
                                            let rec_status = if is_edit_hotkey { "edit_recording" } else { "recording" };
                                            let _ = overlay.emit("recording-status", rec_status);
                                            let _ = overlay.emit("recording-max-duration", MAX_RECORDING_SECS);
                                            if let Ok(Some(monitor)) = overlay.current_monitor() {
                                                let screen = monitor.size();
                                                let scale = monitor.scale_factor();
                                                let win_w = 300.0;
                                                let win_h = 40.0;
                                                let x = (screen.width as f64 / scale - win_w) / 2.0;
                                                let y = screen.height as f64 / scale - win_h - 80.0;
                                                let _ = overlay.set_position(
                                                    tauri::PhysicalPosition::new(
                                                        (x * scale) as i32,
                                                        (y * scale) as i32,
                                                    ),
                                                );
                                            }
                                            platform::show_overlay(&overlay);
                                        }

                                        // Audio level monitoring thread
                                        let app_for_monitor = app.clone();
                                        std::thread::spawn(move || {
                                            let state = app_for_monitor.state::<AppState>();
                                            let sr = state.sample_rate.lock().ok().and_then(|v| *v).unwrap_or(44100) as usize;
                                            let recording_start = Instant::now();

                                            const NUM_BARS: usize = 20;
                                            let samples_per_bar = sr / 20;
                                            // Adaptive gain: track recent peak RMS so the waveform
                                            // fills the full visual range regardless of platform
                                            // mic level (macOS Core Audio vs Windows WASAPI).
                                            let mut peak_rms: f32 = 0.01; // initial floor

                                            while state.is_recording.load(Ordering::SeqCst) {
                                                // Dead-stream guard: if buffer is still empty after
                                                // 1.5 s the cpal callback is not running at all.
                                                if recording_start.elapsed().as_millis() >= 1500 {
                                                    let buf_empty = state.buffer.lock()
                                                        .map(|b| b.is_empty())
                                                        .unwrap_or(false);
                                                    if buf_empty {
                                                        tracing::warn!("⚠️ No audio data after 1.5s — stream dead, aborting recording");
                                                        state.is_recording.store(false, Ordering::SeqCst);
                                                        state.mic_available.store(false, Ordering::SeqCst);
                                                        if let Some(ov) = app_for_monitor.get_webview_window("overlay") {
                                                            let _ = ov.emit("recording-status", "error");
                                                        }
                                                        return;
                                                    }
                                                }

                                                if recording_start.elapsed().as_secs() >= MAX_RECORDING_SECS {
                                                    tracing::info!("⏱️ Max recording duration reached ({}s)", MAX_RECORDING_SECS);
                                                    if state.edit_mode.load(Ordering::SeqCst) {
                                                        stop_edit_and_replace(&app_for_monitor);
                                                    } else {
                                                        stop_transcribe_and_paste(&app_for_monitor);
                                                    }
                                                    return;
                                                }
                                                let levels: Vec<f32> = if let Ok(buf) = state.buffer.lock() {
                                                    if buf.is_empty() {
                                                        vec![0.0; NUM_BARS]
                                                    } else {
                                                        let total = NUM_BARS * samples_per_bar;
                                                        let start = buf.len().saturating_sub(total);
                                                        let raw_rms: Vec<f32> = buf[start..]
                                                            .chunks(samples_per_bar)
                                                            .map(|chunk| {
                                                                (chunk.iter().map(|&s| s * s).sum::<f32>()
                                                                    / chunk.len() as f32)
                                                                    .sqrt()
                                                            })
                                                            .collect();
                                                        // Fast attack, slow decay (~3.5s to halve at 50ms tick)
                                                        let frame_max = raw_rms.iter().cloned().fold(0.0f32, f32::max);
                                                        if frame_max > peak_rms {
                                                            peak_rms = frame_max;
                                                        } else {
                                                            peak_rms *= 0.99;
                                                        }
                                                        peak_rms = peak_rms.max(0.001);
                                                        let mut bars: Vec<f32> = raw_rms.iter()
                                                            .map(|&rms| (rms / peak_rms).min(1.0))
                                                            .collect();
                                                        while bars.len() < NUM_BARS {
                                                            bars.insert(0, 0.0);
                                                        }
                                                        bars
                                                    }
                                                } else {
                                                    vec![0.0; NUM_BARS]
                                                };

                                                if let Some(ov) = app_for_monitor.get_webview_window("overlay") {
                                                    let _ = ov.emit("audio-levels", &levels);
                                                }
                                                if state.voice_rule_mode.load(Ordering::SeqCst) {
                                                    if let Some(main_win) = app_for_monitor.get_webview_window("main") {
                                                        let _ = main_win.emit("voice-rule-levels", &levels);
                                                    }
                                                }
                                                std::thread::sleep(std::time::Duration::from_millis(50));
                                            }
                                        });
                                    }
                                    Err(e) => {
                                        tracing::error!("Failed to start recording: {}", e);
                                        if is_edit_hotkey {
                                            state.edit_mode.store(false, Ordering::SeqCst);
                                            restore_clipboard(&state);
                                        }
                                        if let Some(overlay) = app.get_webview_window("overlay") {
                                            platform::hide_overlay(&overlay);
                                        }
                                    }
                                }
                            } else {
                                // Stop Recording
                                if state.edit_mode.load(Ordering::SeqCst) {
                                    stop_edit_and_replace(app);
                                } else {
                                    stop_transcribe_and_paste(app);
                                }
                            }
                        })
                        .build(),
                )?;

                app.global_shortcut().register(primary_shortcut)?;
                let label = hotkey_display_label(&hotkey_str);
                tracing::info!("{} global shortcut registered", label);

                if let Some(edit_sc) = edit_shortcut {
                    app.global_shortcut().register(edit_sc)?;
                    if let Some(ref edit_hk) = settings.edit_hotkey {
                        tracing::info!("{} edit shortcut registered", hotkey_display_label(edit_hk));
                    }
                }

                if let Some(meeting_sc) = meeting_shortcut {
                    app.global_shortcut().register(meeting_sc)?;
                    if let Some(ref meeting_hk) = settings.meeting_hotkey {
                        tracing::info!("{} meeting shortcut registered", hotkey_display_label(meeting_hk));
                    }
                }
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn show_settings_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.set_size(tauri::LogicalSize::new(960.0, 720.0));
        let _ = window.center();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

fn start_meeting_mode(app: &AppHandle) {
    let state = app.state::<AppState>();

    // Capture frontmost app context before recording starts (same as normal recording).
    let captured_ctx = state
        .context_override
        .lock()
        .ok()
        .and_then(|ctx| ctx.clone())
        .unwrap_or_else(context_detect::detect_frontmost_app);
    if let Ok(mut ctx) = state.captured_context.lock() {
        *ctx = Some(captured_ctx);
    }

    // Only Qwen3-ASR local mode is supported.
    let (stt_mode, local_engine, qwen3_model, lang) = state
        .settings
        .lock()
        .map(|s| (
            s.stt.mode.clone(),
            s.stt.local_engine.clone(),
            s.stt.qwen3_asr_model.clone(),
            s.stt.language.clone(),
        ))
        .unwrap_or_default();

    if stt_mode != SttMode::Local || local_engine != stt::LocalSttEngine::Qwen3Asr {
        tracing::warn!("Meeting mode requires Qwen3-ASR local engine");
        if let Some(overlay) = app.get_webview_window("overlay") {
            let _ = overlay.emit("recording-status", "error");
            if let Ok(Some(monitor)) = overlay.current_monitor() {
                let screen = monitor.size();
                let scale = monitor.scale_factor();
                let win_w = 300.0_f64;
                let win_h = 40.0_f64;
                let x = (screen.width as f64 / scale - win_w) / 2.0;
                let y = screen.height as f64 / scale - win_h - 80.0;
                let _ = overlay.set_position(tauri::PhysicalPosition::new(
                    (x * scale) as i32,
                    (y * scale) as i32,
                ));
            }
            platform::show_overlay(&overlay);
        }
        hide_overlay_delayed(app, 2000);
        return;
    }

    let preferred_device = state.settings.lock().ok().and_then(|s| s.mic_device.clone());
    if let Err(e) = audio::do_start_recording(
        &state.is_recording,
        &state.mic_available,
        &state.sample_rate,
        &state.buffer,
        &state.is_recording,
        &state.audio_thread,
        preferred_device,
    ) {
        tracing::error!("Meeting mode start failed: {}", e);
        return;
    }

    state.meeting_active.store(true, Ordering::SeqCst);
    if let Ok(mut t) = state.meeting_transcript.lock() {
        t.clear();
    }
    if let Ok(mut st) = state.meeting_start_time.lock() {
        *st = Some(std::time::Instant::now());
    }

    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.emit("recording-status", "meeting_recording");
        let _ = overlay.emit("recording-max-duration", 0u64); // 0 = unlimited
        if let Ok(Some(monitor)) = overlay.current_monitor() {
            let screen = monitor.size();
            let scale = monitor.scale_factor();
            let win_w = 300.0_f64;
            let win_h = 40.0_f64;
            let x = (screen.width as f64 / scale - win_w) / 2.0;
            let y = screen.height as f64 / scale - win_h - 80.0;
            let _ = overlay.set_position(tauri::PhysicalPosition::new(
                (x * scale) as i32,
                (y * scale) as i32,
            ));
        }
        platform::show_overlay(&overlay);
    }

    // Audio level monitor thread (same as normal recording, but checks meeting_active).
    {
        let app_for_monitor = app.clone();
        std::thread::spawn(move || {
            let state = app_for_monitor.state::<AppState>();
            let sr = state.sample_rate.lock().ok().and_then(|v| *v).unwrap_or(44100) as usize;

            const NUM_BARS: usize = 20;
            let samples_per_bar = sr / 20;
            let mut peak_rms: f32 = 0.01;

            while state.is_recording.load(Ordering::SeqCst) {
                let levels: Vec<f32> = if let Ok(buf) = state.buffer.lock() {
                    if buf.is_empty() {
                        vec![0.0; NUM_BARS]
                    } else {
                        let total = NUM_BARS * samples_per_bar;
                        let start = buf.len().saturating_sub(total);
                        let raw_rms: Vec<f32> = buf[start..]
                            .chunks(samples_per_bar)
                            .map(|chunk| {
                                (chunk.iter().map(|&s| s * s).sum::<f32>()
                                    / chunk.len() as f32)
                                    .sqrt()
                            })
                            .collect();
                        let frame_max = raw_rms.iter().cloned().fold(0.0f32, f32::max);
                        if frame_max > peak_rms {
                            peak_rms = frame_max;
                        } else {
                            peak_rms *= 0.99;
                        }
                        peak_rms = peak_rms.max(0.001);
                        let mut bars: Vec<f32> = raw_rms
                            .iter()
                            .map(|&rms| (rms / peak_rms).min(1.0))
                            .collect();
                        while bars.len() < NUM_BARS {
                            bars.insert(0, 0.0);
                        }
                        bars
                    }
                } else {
                    vec![0.0; NUM_BARS]
                };

                if let Some(ov) = app_for_monitor.get_webview_window("overlay") {
                    let _ = ov.emit("audio-levels", &levels);
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });
    }

    // Spawn meeting feeder after engine is warm (max 8 s wait).
    // Also trigger warm proactively so the engine starts loading in parallel.
    let feeder_app = app.clone();
    std::thread::spawn(move || {
        let fstate = feeder_app.state::<AppState>();

        // Proactively trigger model warm-up (no-op if already loaded).
        if let Err(e) = qwen3_asr::warm_qwen3_asr(&fstate.qwen3_asr_ctx, &qwen3_model) {
            tracing::warn!("Meeting mode: engine warm failed: {}", e);
        }

        // Wait for engine to be ready (up to 8 s total, warm already started above).
        let mut waited = 0u64;
        while waited < 8000 {
            let ready = fstate
                .qwen3_asr_ctx
                .lock()
                .ok()
                .and_then(|g| g.as_ref().map(|c| c.model == qwen3_model))
                .unwrap_or(false);
            if ready {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
            waited += 200;
        }
        let ready = fstate
            .qwen3_asr_ctx
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|c| c.model == qwen3_model))
            .unwrap_or(false);
        if !ready {
            tracing::warn!("Meeting mode: engine not ready after 8s, aborting");
            fstate.meeting_active.store(false, Ordering::SeqCst);
            fstate.is_recording.store(false, Ordering::SeqCst);
            if let Some(ov) = feeder_app.get_webview_window("overlay") {
                let _ = ov.emit("recording-status", "error");
            }
            hide_overlay_delayed(&feeder_app, 2000);
            return;
        }
        qwen3_asr::run_meeting_feeder_loop(feeder_app, lang);
    });
}

fn stop_meeting_mode(app: &AppHandle) {
    let state = app.state::<AppState>();

    // Signal feeder to stop.
    state.is_recording.store(false, Ordering::SeqCst);

    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.emit("recording-status", "processing");
    }

    // Wait for feeder to finish (max 8 s).
    let deadline =
        std::time::Instant::now() + std::time::Duration::from_millis(8000);
    while state.meeting_active.load(Ordering::SeqCst) {
        if std::time::Instant::now() >= deadline {
            tracing::warn!("Meeting feeder timeout — using partial transcript");
            // Mark inactive immediately so the hotkey is not locked out.
            state.meeting_active.store(false, Ordering::SeqCst);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    let transcript = state
        .meeting_transcript
        .lock()
        .map(|t| t.clone())
        .unwrap_or_default();

    let duration_secs = state
        .meeting_start_time
        .lock()
        .ok()
        .and_then(|st| *st)
        .map(|t| t.elapsed().as_secs_f64())
        .unwrap_or(0.0);

    if transcript.is_empty() {
        tracing::info!("Meeting mode stopped — no transcript");
        if let Some(overlay) = app.get_webview_window("overlay") {
            let _ = overlay.emit("recording-status", "error");
        }
        hide_overlay_delayed(app, 1500);
        state.meeting_active.store(false, Ordering::SeqCst);
        return;
    }

    // Copy to clipboard.
    if let Ok(mut clipboard) = arboard::Clipboard::new() {
        if let Err(e) = clipboard.set_text(&transcript) {
            tracing::error!("Meeting clipboard error: {}", e);
        }
    }

    // Save history entry (no audio file).
    {
        let retention_days = state
            .settings
            .lock()
            .map(|s| s.history_retention_days)
            .unwrap_or(0);
        let ctx = state
            .captured_context
            .lock()
            .ok()
            .and_then(|c| c.clone())
            .unwrap_or_default();
        let word_count = history::count_words(&transcript) as u64;
        let entry_id = history::generate_id();
        let entry = history::HistoryEntry {
            id: entry_id,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64,
            text: transcript.clone(),
            raw_text: transcript.clone(),
            reasoning: None,
            stt_model: "Qwen3-ASR (Meeting)".to_string(),
            polish_model: "None".to_string(),
            duration_secs,
            has_audio: false,
            stt_elapsed_ms: 0, // streaming mode — no single STT duration
            polish_elapsed_ms: None,
            total_elapsed_ms: (duration_secs * 1000.0) as u64,
            app_name: ctx.app_name.clone(),
            bundle_id: ctx.bundle_id.clone(),
            chars_per_sec: 0.0,
            word_count,
        };
        history::add_entry(&history_dir(), &audio_dir(), entry, retention_days);
    }

    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.emit("recording-status", "copied");
    }
    hide_overlay_delayed(app, 2000);
    state.meeting_active.store(false, Ordering::SeqCst);
}

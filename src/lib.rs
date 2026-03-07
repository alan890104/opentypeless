mod audio;
mod audio_devices;
mod audio_import;
mod commands;
pub mod diarization;
mod context_detect;
mod credentials;
mod history;
mod hotkey;
mod meeting_feeder;
mod meeting_notes;
mod permissions;
pub mod platform;
pub mod models;
mod polisher;
mod qwen3_asr;
pub mod settings;
mod whisper_streaming;
pub mod stt;
mod transcribe;
pub mod whisper_models;

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Condvar, Mutex,
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
    /// Condvar/flag pair used by `warm_qwen3_asr` (producer) and
    /// `wait_engine_ready` (consumer) to avoid busy-polling.  The flag and
    /// the Condvar share the same lock (`qwen3_ready_mu`) so there is no
    /// lost-wakeup window.  Scoped to Qwen3-ASR only; Whisper uses try_lock
    /// + skip semantics instead.
    pub qwen3_ready_cv: Condvar,
    pub qwen3_ready_mu: Mutex<bool>,
    pub model_switching: AtomicBool,
    pub reconnecting: AtomicBool,
    pub streaming_active: AtomicBool,
    pub streaming_cancelled: AtomicBool,
    pub streaming_result: Mutex<Option<String>>,
    /// Condvar used to wake feeder threads early when `is_recording` is set to
    /// false. Replaces a fixed `thread::sleep(2000 ms)` with an interruptible
    /// `wait_timeout(2000 ms)`, eliminating up to 2 s of unnecessary latency
    /// for short recordings (especially noticeable on 1–2 s utterances).
    pub feeder_stop_cv: Condvar,
    pub feeder_stop_mu: Mutex<()>,
    pub meeting_active: AtomicBool,
    pub meeting_cancelled: AtomicBool,
    /// Guards `stop_meeting_mode` against re-entrant/concurrent invocations.
    /// Set to `true` at entry, cleared to `false` on exit. Using a dedicated
    /// flag (instead of `is_recording.swap`) ensures we handle the case where
    /// `is_recording` was already set to `false` by the dead-stream guard before
    /// `stop_meeting_mode` is called — without this, the transcript would never
    /// be delivered.
    pub meeting_stopping: AtomicBool,
    /// Set by the meeting feeder after its final WAL write completes.
    /// Allows `stop_meeting_mode` to wait for the complete transcript even when
    /// the final-chunk inference takes longer than the hotkey-unlock deadline.
    pub meeting_feeder_done: AtomicBool,
    /// Notified by `run_meeting_feeder` when `meeting_feeder_done` becomes true.
    /// Allows `stop_meeting_mode` to block without busy-polling.
    pub meeting_done_cv: Condvar,
    /// Mutex associated with `meeting_done_cv`.
    pub meeting_done_mu: Mutex<()>,
    /// Monotonically increasing session counter. Incremented each time a new
    /// meeting session starts. The feeder thread captures this value at launch
    /// and aborts post-loop work if the counter has advanced, preventing a
    /// zombie feeder from the previous timed-out session from corrupting the
    /// new session's transcript or forcing `meeting_active` to false.
    pub meeting_session: AtomicU64,
    pub meeting_start_time: Mutex<Option<std::time::Instant>>,
    pub active_meeting_note_id: Mutex<Option<String>>,
    /// Monotonically increasing session counter for the normal (non-meeting)
    /// Qwen3-ASR streaming feeder. Prevents a zombie feeder from a previous
    /// session from overwriting `streaming_result` for a later recording.
    pub streaming_session: AtomicU64,
    /// True while the Whisper live-preview feeder thread is running.
    pub whisper_preview_active: AtomicBool,
    /// Monotonically increasing session counter for the Whisper preview feeder.
    /// Prevents a zombie feeder from a previous recording from emitting stale
    /// partial results into the current session's overlay.
    pub whisper_preview_session: AtomicU64,
    /// Cached `Shortcut` for the edit-by-voice hotkey, set at registration time.
    /// The handler compares directly against this instead of re-parsing
    /// `settings.edit_hotkey` on every keypress, eliminating the TOCTOU race
    /// where settings and the registered shortcut could briefly diverge.
    pub registered_edit_shortcut: Mutex<Option<Shortcut>>,
    /// Cached `Shortcut` for the meeting hotkey. Same rationale as above.
    pub registered_meeting_shortcut: Mutex<Option<Shortcut>>,
    /// Set to `true` when we sent a Play/Pause media key at recording start.
    /// Cleared (and play/pause key sent again) when recording ends, so the
    /// user's music resumes automatically.  Guards against spuriously resuming
    /// music that was already paused before recording started.
    pub media_paused_by_sumi: AtomicBool,
    /// Timestamp of the last recording end. Used by the idle mic watcher to
    /// determine when to close the mic stream.
    pub last_recording_end: Mutex<Option<Instant>>,
    /// True while an audio file import is running.
    pub import_active: AtomicBool,
    /// Set to true to cancel a running audio file import.
    pub import_cancelled: AtomicBool,
    /// Optional speaker diarization engine (WeSpeaker ONNX).
    /// Loaded at meeting start when diarization model files are present.
    pub diarization_ctx: Mutex<Option<diarization::DiarizationEngine>>,
}

/// Emit a `"transcription-partial"` event to the overlay window.
///
/// Used by all feeder loops (Qwen3-ASR normal/meeting, Whisper preview/meeting)
/// to send incremental transcription results to the overlay.
pub(crate) fn emit_transcription_partial(app: &AppHandle, text: &str) {
    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.emit(
            "transcription-partial",
            serde_json::json!({ "text": text }),
        );
    }
}

/// Position the overlay window centered horizontally near the bottom of the focused screen.
///
/// Uses `NSScreen.mainScreen` (the screen with the active keyboard focus) so the
/// capsule always appears on the same screen as the user's frontmost app.
/// Falls back to the overlay's current monitor if the platform call is unavailable.
fn center_overlay_bottom(overlay: &tauri::WebviewWindow) {
    const WIN_W: f64 = 300.0;
    const WIN_H: f64 = 40.0;
    const MARGIN_BOTTOM: f64 = 80.0;

    if let Some((sx, sy, sw, sh, scale)) = platform::focused_screen_logical_frame() {
        let x = sx + (sw - WIN_W) / 2.0;
        let y = sy + sh - WIN_H - MARGIN_BOTTOM;
        let _ = overlay.set_position(tauri::PhysicalPosition::new(
            (x * scale) as i32,
            (y * scale) as i32,
        ));
    } else if let Ok(Some(monitor)) = overlay.current_monitor() {
        let screen = monitor.size();
        let scale = monitor.scale_factor();
        let x = (screen.width as f64 / scale - WIN_W) / 2.0;
        let y = screen.height as f64 / scale - WIN_H - MARGIN_BOTTOM;
        let _ = overlay.set_position(tauri::PhysicalPosition::new(
            (x * scale) as i32,
            (y * scale) as i32,
        ));
    }
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
        let dictionary_terms = polish_config.dictionary.enabled_terms();

        let stop_result = audio::do_stop_recording(
            &state,
            &stt_config,
            &stt_language,
            &dictionary_terms,
        );
        if let Ok(mut t) = state.last_recording_end.lock() {
            *t = Some(Instant::now());
        }
        // Resume media paused at recording start.
        if state.media_paused_by_sumi.swap(false, Ordering::SeqCst) {
            platform::resume_now_playing();
        }
        match stop_result {
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
        let edit_dict_terms = polish_config.dictionary.enabled_terms();

        // The edit path never spawns a live-preview feeder. Defensively clear
        // any residual streaming state from a previous normal recording so a
        // stale result cannot be consumed as the edit instruction.
        state.streaming_active.store(false, Ordering::SeqCst);
        if let Ok(mut r) = state.streaming_result.lock() {
            *r = None;
        }

        let stop_result = audio::do_stop_recording(
            &state,
            &stt_config,
            &edit_stt_language,
            &edit_dict_terms,
        );
        if let Ok(mut t) = state.last_recording_end.lock() {
            *t = Some(Instant::now());
        }
        // Resume media paused at recording start.
        if state.media_paused_by_sumi.swap(false, Ordering::SeqCst) {
            platform::resume_now_playing();
        }
        match stop_result {
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

/// Remove model files / directories that are no longer part of the current
/// model catalogue.  Runs in a background thread at startup so it never
/// blocks the UI.
///
/// Strategy: build an allowlist of every filename / directory name that
/// any currently-supported model variant can produce, then delete everything
/// else found inside `models_dir`.
fn cleanup_obsolete_models(models_dir: &std::path::Path) {
    use std::collections::HashSet;

    let mut known_files: HashSet<&str> = HashSet::new();
    let mut known_dirs: HashSet<&str> = HashSet::new();

    // Whisper — all 6 variants (including Medium/Small which are valid but
    // unmanaged; we still want to keep them if the user downloaded them).
    for model in &[
        whisper_models::WhisperModel::LargeV3Turbo,
        whisper_models::WhisperModel::LargeV3TurboQ5,
        whisper_models::WhisperModel::Medium,
        whisper_models::WhisperModel::Small,
        whisper_models::WhisperModel::Base,
        whisper_models::WhisperModel::LargeV3TurboZhTw,
    ] {
        known_files.insert(model.filename());
    }

    // Polish models — GGUF + optional tokenizer JSON.
    for model in polisher::PolishModel::all() {
        known_files.insert(model.filename());
        if let Some(tok) = model.tokenizer_filename() {
            known_files.insert(tok);
        }
    }

    // VAD model — derive filename from the canonical path so it stays in sync
    // automatically when the version is bumped in transcribe::vad_model_path().
    let vad_filename_owned;
    if let Some(name) = transcribe::vad_model_path()
        .file_name()
        .and_then(|n| n.to_str())
    {
        vad_filename_owned = name.to_owned();
        known_files.insert(&vad_filename_owned);
    }

    // Qwen3-ASR — stored as subdirectories.
    for model in &[stt::Qwen3AsrModel::Qwen3Asr1_7B, stt::Qwen3AsrModel::Qwen3Asr0_6B] {
        known_dirs.insert(model.model_dir_name());
    }

    // Diarization infra models (speaker embedding + segmentation).
    let diar_filenames: Vec<String> = [
        settings::diarization_model_path(),
        settings::segmentation_model_path(),
    ]
    .iter()
    .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(str::to_owned))
    .collect();
    for name in &diar_filenames {
        known_files.insert(name.as_str());
    }

    let Ok(entries) = std::fs::read_dir(models_dir) else {
        tracing::warn!("[model-cleanup] Cannot read models dir");
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()).map(str::to_owned) else {
            continue;
        };

        if path.is_dir() {
            if !known_dirs.contains(name.as_str()) {
                match std::fs::remove_dir_all(&path) {
                    Ok(()) => tracing::info!("[model-cleanup] Removed obsolete model dir: {name}"),
                    Err(e) => tracing::warn!("[model-cleanup] Failed to remove dir {name}: {e}"),
                }
            }
        } else if !known_files.contains(name.as_str()) {
            match std::fs::remove_file(&path) {
                Ok(()) => tracing::info!("[model-cleanup] Removed obsolete model file: {name}"),
                Err(e) => tracing::warn!("[model-cleanup] Failed to remove file {name}: {e}"),
            }
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
        .plugin(tauri_plugin_dialog::init())
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
            commands::delete_whisper_model,
            commands::delete_polish_model,
            commands::delete_qwen3_asr_model,
            commands::delete_vad_model,
            commands::check_diarization_model_status,
            commands::download_diarization_model,
            commands::delete_diarization_model,
            commands::check_segmentation_model_status,
            commands::download_segmentation_model,
            commands::delete_segmentation_model,
            commands::update_meeting_hotkey,
            commands::list_meeting_notes,
            commands::get_meeting_note,
            commands::rename_meeting_note,
            commands::delete_meeting_note,
            commands::delete_all_meeting_notes,
            commands::get_active_meeting_note_id,
            commands::polish_meeting_note,
            commands::export_meeting_audio,
            commands::delete_meeting_audio,
            commands::import_meeting_audio,
            commands::cancel_import,
            commands::start_infra_downloads,
            commands::check_infra_models_ready,
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

            // Migrate legacy JSON history to SQLite, then run schema migrations
            history::migrate_from_json(&history_dir(), &audio_dir());
            history::init_db(&history_dir());

            // Init meeting notes schema & recover notes stuck from a previous crash
            meeting_notes::init_db(&history_dir());
            meeting_notes::recover_stuck_notes(&history_dir(), &audio_dir());

            // Remove obsolete model files in the background (non-blocking).
            {
                let dir = models_dir();
                std::thread::spawn(move || cleanup_obsolete_models(&dir));
            }

            // The mic stream is pre-opened in a background thread shortly after
            // startup (see below) so the first hotkey press has zero latency.
            // Opening the stream activates CoreAudio DSP (echo-cancellation,
            // noise-reduction), which is the accepted trade-off for zero first-word
            // latency.  If pre-open fails, do_start_recording retries on the first
            // hotkey press.
            let is_recording = Arc::new(AtomicBool::new(false));
            let buffer = Arc::new(Mutex::new(Vec::new()));
            let mic_available = false;
            let sample_rate: Option<u32> = None;
            let audio_thread_init: Option<audio::AudioThreadControl> = None;

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
                qwen3_ready_cv: Condvar::new(),
                qwen3_ready_mu: Mutex::new(false),
                model_switching: AtomicBool::new(false),
                reconnecting: AtomicBool::new(false),
                streaming_active: AtomicBool::new(false),
                streaming_cancelled: AtomicBool::new(false),
                streaming_result: Mutex::new(None),
                feeder_stop_cv: Condvar::new(),
                feeder_stop_mu: Mutex::new(()),
                meeting_active: AtomicBool::new(false),
                meeting_cancelled: AtomicBool::new(false),
                meeting_stopping: AtomicBool::new(false),
                meeting_feeder_done: AtomicBool::new(false),
                meeting_done_cv: Condvar::new(),
                meeting_done_mu: Mutex::new(()),
                meeting_session: AtomicU64::new(0),
                meeting_start_time: Mutex::new(None),
                active_meeting_note_id: Mutex::new(None),
                streaming_session: AtomicU64::new(0),
                whisper_preview_active: AtomicBool::new(false),
                whisper_preview_session: AtomicU64::new(0),
                registered_edit_shortcut: Mutex::new(
                    settings.edit_hotkey.as_deref().and_then(parse_hotkey_string),
                ),
                registered_meeting_shortcut: Mutex::new(
                    settings.meeting_hotkey.as_deref().and_then(parse_hotkey_string),
                ),
                media_paused_by_sumi: AtomicBool::new(false),
                last_recording_end: Mutex::new(None),
                import_active: AtomicBool::new(false),
                import_cancelled: AtomicBool::new(false),
                diarization_ctx: Mutex::new(None),
            });

            // Register a CoreAudio listener for default-input-device changes.
            // When the user connects Bluetooth headphones, the system default input
            // may switch to them. The listener reconnects the cpal stream to the
            // built-in mic (via resolve_input_device), preventing an A2DP → HFP switch.
            {
                let app_for_listener = app.handle().clone();
                audio_devices::add_default_input_listener(move || {
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
                    if !crate::audio_devices::is_default_input_bluetooth() {
                        tracing::info!("Default input changed to non-BT device — stream already correct, skipping reconnect");
                        return;
                    }

                    tracing::info!("Default audio input device changed to Bluetooth — reconnecting to built-in mic");

                    // On-demand model: stream is closed between recordings.
                    // resolve_input_device will avoid BT automatically on the next
                    // recording start, so no reconnect is needed while idle.
                    if !state.is_recording.load(Ordering::SeqCst) {
                        tracing::info!("Not recording — skipping BT reconnect; built-in will be used on next hotkey press");
                        return;
                    }

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
                            Ok(()) => {
                                tracing::info!("Mic stream reconnected after input device change");
                                // Reset idle clock so the watcher doesn't immediately
                                // close the freshly reconnected stream.
                                if let Ok(mut t) = state.last_recording_end.lock() {
                                    *t = None;
                                }
                            }
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
                                    if let Err(e) = qwen3_asr::warm_qwen3_asr(&state.qwen3_asr_ctx, &qwen3_model, Some((&state.qwen3_ready_cv, &state.qwen3_ready_mu))) {
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

            // Pre-open the mic stream in the background so the first hotkey press
            // has zero latency.  Without this, do_start_recording would call
            // try_reconnect_audio on the first press, blocking for ~100–300 ms
            // while CoreAudio initialises, causing the first words to be dropped.
            //
            // `reconnecting` is set to true *before* spawning so that any hotkey
            // press arriving in the first few hundred ms sees the flag and waits
            // (via do_start_recording's spin-wait) instead of racing into
            // spawn_audio_thread and leaking one cpal stream.
            {
                let state = app.state::<AppState>();
                state.reconnecting.store(true, Ordering::SeqCst);
                let app_handle = app.handle().clone();
                std::thread::spawn(move || {
                    let state = app_handle.state::<AppState>();
                    let device_name = state.settings.lock().ok().and_then(|s| s.mic_device.clone());
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        audio::try_reconnect_audio(
                            &state.mic_available,
                            &state.sample_rate,
                            &state.buffer,
                            &state.is_recording,
                            &state.audio_thread,
                            device_name,
                        )
                    }));
                    // Always release the guard, even if try_reconnect_audio panics.
                    state.reconnecting.store(false, Ordering::SeqCst);
                    match result {
                        Ok(Ok(())) => tracing::info!("Mic stream pre-opened at startup"),
                        Ok(Err(e)) => tracing::warn!("Mic pre-open failed (will retry on first hotkey): {}", e),
                        Err(_) => tracing::error!("Mic pre-open thread panicked; reconnecting flag reset"),
                    }
                });
            }

            // Idle mic watcher: closes the mic stream after a configurable idle
            // period to avoid CoreAudio DSP (echo cancellation, AGC, audio
            // ducking) from affecting other apps when Sumi is not in use.
            {
                let app_handle = app.handle().clone();
                std::thread::spawn(move || {
                    tracing::info!("Idle mic watcher started (poll interval: 5s)");
                    loop {
                        std::thread::sleep(std::time::Duration::from_secs(5));
                        let state = app_handle.state::<AppState>();

                        let timeout_secs = state
                            .settings
                            .lock()
                            .map(|s| s.idle_mic_timeout_secs)
                            .unwrap_or(0);
                        if timeout_secs == 0 {
                            continue;
                        }

                        // Don't close while recording, meeting, or reconnecting.
                        if state.is_recording.load(Ordering::SeqCst)
                            || state.meeting_active.load(Ordering::SeqCst)
                            || state.reconnecting.load(Ordering::SeqCst)
                            || !state.mic_available.load(Ordering::SeqCst)
                        {
                            continue;
                        }

                        let timeout = std::time::Duration::from_secs(timeout_secs as u64);

                        // Pre-check elapsed outside the lock as a fast path to
                        // avoid contending on audio_thread every 5 seconds.
                        let elapsed = state
                            .last_recording_end
                            .lock()
                            .ok()
                            .and_then(|t| t.map(|i| i.elapsed()));

                        match elapsed {
                            None | Some(std::time::Duration::ZERO) => continue, // never recorded
                            Some(e) if e < timeout => continue,
                            _ => {}
                        }

                        // Acquire audio_thread lock and re-check all guards
                        // atomically — do_start_recording Step 3 also holds
                        // this lock when setting is_recording=true, so the two
                        // operations are mutually exclusive (no TOCTOU window).
                        if let Ok(mut at) = state.audio_thread.lock() {
                            if state.is_recording.load(Ordering::SeqCst)
                                || state.meeting_active.load(Ordering::SeqCst)
                                || state.reconnecting.load(Ordering::SeqCst)
                            {
                                continue;
                            }
                            // Re-read elapsed inside the lock to close the
                            // TOCTOU window: a recording may have completed
                            // between our pre-check and acquiring this lock.
                            let fresh = state
                                .last_recording_end
                                .lock()
                                .ok()
                                .and_then(|t| t.map(|i| i.elapsed()));
                            match fresh {
                                None | Some(std::time::Duration::ZERO) => continue,
                                Some(e) if e < timeout => continue,
                                _ => {}
                            }
                            tracing::info!(
                                "Idle mic timeout ({}s) — closing mic stream",
                                timeout_secs
                            );
                            // Set mic_available=false BEFORE stopping the
                            // stream so a concurrent do_start_recording
                            // sees the unavailable flag immediately.
                            state.mic_available.store(false, Ordering::SeqCst);
                            if let Some(ctrl) = at.take() {
                                ctrl.stop();
                            }
                            // Reset idle clock so a hot-plug reconnect
                            // doesn't immediately re-trigger a close.
                            if let Ok(mut t) = state.last_recording_end.lock() {
                                *t = None;
                            }
                        };
                    }
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

            // Window close → hide (drag handled by data-tauri-drag-region in HTML)
            if let Some(main_window) = app.get_webview_window("main") {
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

                            let is_edit_hotkey = state.registered_edit_shortcut
                                .lock()
                                .ok()
                                .and_then(|g| g.as_ref().map(|s| s == shortcut))
                                .unwrap_or(false);
                            let is_meeting_hotkey = state.registered_meeting_shortcut
                                .lock()
                                .ok()
                                .and_then(|g| g.as_ref().map(|s| s == shortcut))
                                .unwrap_or(false);

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
                                } else if !state.meeting_stopping.load(Ordering::SeqCst) {
                                    // Guard: don't start a new meeting while stop_meeting_mode is
                                    // still running (meeting_active=false but worker not done yet).
                                    // Starting here would reset meeting_stopping=false via
                                    // start_meeting_mode, breaking the CAS idempotency gate.
                                    start_meeting_mode(app);
                                }
                                return;
                            }

                            // Block regular recording while meeting is finalizing (worker still running).
                            if state.meeting_stopping.load(Ordering::SeqCst) {
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
                                            center_overlay_bottom(&overlay);
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

                                // Show overlay in 'preparing' state before opening the mic
                                // stream (on-demand model).  CoreAudio init takes ~70 ms;
                                // the spinner gives the user immediate visual feedback.
                                if let Some(overlay) = app.get_webview_window("overlay") {
                                    let _ = overlay.emit("recording-status", "preparing");
                                    center_overlay_bottom(&overlay);
                                    platform::show_overlay(&overlay);
                                }

                                // Pause background music only if a media player is open.
                                // Skipping when none is running avoids launching Apple Music.
                                if platform::is_now_playing() {
                                    platform::pause_now_playing();
                                    state.media_paused_by_sumi.store(true, Ordering::SeqCst);
                                }

                                match audio::do_start_recording(
                                    &state.is_recording,
                                    &state.mic_available,
                                    &state.reconnecting,
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
                                                                &state.qwen3_asr_ctx, &qwen3_model, Some((&state.qwen3_ready_cv, &state.qwen3_ready_mu)),
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
                                        // Read all feeder-relevant settings in a single lock acquisition
                                        // so that no race can occur between computing should_stream and
                                        // reading feeder_model/feeder_lang.
                                        let stream_config = if !is_edit_hotkey {
                                            state.settings.lock().ok().and_then(|s| {
                                                if s.stt.mode == SttMode::Local
                                                    && s.stt.local_engine == stt::LocalSttEngine::Qwen3Asr
                                                {
                                                    Some((s.stt.qwen3_asr_model.clone(), s.stt.language.clone()))
                                                } else {
                                                    None
                                                }
                                            })
                                        } else {
                                            None
                                        };
                                        if let Some((feeder_model, feeder_lang)) = stream_config {
                                            state.streaming_active.store(true, Ordering::SeqCst);
                                            // Advance session counter BEFORE resetting streaming_cancelled.
                                            // This closes a TOCTOU window: any zombie feeder that sees
                                            // cancelled=false (the reset value) is guaranteed to also see
                                            // the new session_id (SeqCst total order), so its session guard
                                            // will reject it before it can write a stale streaming_result.
                                            let feeder_session_id = state.streaming_session.fetch_add(1, Ordering::SeqCst) + 1;
                                            state.streaming_cancelled.store(false, Ordering::SeqCst);
                                            if let Ok(mut r) = state.streaming_result.lock() {
                                                *r = None;
                                            }
                                            let feeder_app = app.clone();
                                            std::thread::spawn(move || {
                                                let fstate = feeder_app.state::<AppState>();
                                                if !qwen3_asr::wait_engine_ready(&fstate.qwen3_asr_ctx, &feeder_model, &fstate.qwen3_ready_cv, &fstate.qwen3_ready_mu, 8000) {
                                                    tracing::warn!("[streaming] engine warm-up timed out after 8s — no live preview");
                                                    fstate.streaming_active.store(false, Ordering::SeqCst);
                                                    return;
                                                }
                                                qwen3_asr::run_feeder_loop(feeder_app, feeder_lang, feeder_session_id);
                                            });
                                        }

                                        // ── Live-preview feeder (Whisper non-edit mode only) ──
                                        let whisper_preview_config = if !is_edit_hotkey {
                                            state.settings.lock().ok().and_then(|s| {
                                                if s.stt.mode == SttMode::Local
                                                    && s.stt.local_engine == stt::LocalSttEngine::Whisper
                                                {
                                                    Some(s.stt.language.clone())
                                                } else {
                                                    None
                                                }
                                            })
                                        } else {
                                            None
                                        };
                                        if let Some(preview_lang) = whisper_preview_config {
                                            let preview_session_id = state.whisper_preview_session.fetch_add(1, Ordering::SeqCst) + 1;
                                            state.whisper_preview_active.store(true, Ordering::SeqCst);
                                            let preview_app = app.clone();
                                            std::thread::spawn(move || {
                                                whisper_streaming::run_whisper_preview_loop(preview_app, preview_lang, preview_session_id);
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
                                            // overlay already shown in 'preparing' state above
                                        }

                                        // Audio level monitoring thread
                                        spawn_audio_level_monitor(app.clone(), AudioMonitorMode::Normal);
                                    }
                                    Err(e) => {
                                        tracing::error!("Failed to start recording: {}", e);
                                        if is_edit_hotkey {
                                            state.edit_mode.store(false, Ordering::SeqCst);
                                            restore_clipboard(&state);
                                        }
                                        // Resume music if we paused it before do_start_recording.
                                        if state.media_paused_by_sumi.swap(false, Ordering::SeqCst) {
                                            platform::resume_now_playing();
                                        }
                                        // Hide overlay — stream failed to open.
                                        if let Some(overlay) = app.get_webview_window("overlay") {
                                            let _ = overlay.emit("recording-status", "error");
                                        }
                                        hide_overlay_delayed(&app, 1500);
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

    let (stt_mode, local_engine, qwen3_model, whisper_model, lang, cloud_config) = state
        .settings
        .lock()
        .map(|s| (
            s.stt.mode.clone(),
            s.stt.local_engine.clone(),
            s.stt.qwen3_asr_model.clone(),
            s.stt.whisper_model.clone(),
            s.stt.language.clone(),
            s.stt.cloud.clone(),
        ))
        .unwrap_or_default();

    // Guard: if a normal recording is already in progress, refuse to start meeting mode
    // to avoid two feeders sharing the same buffer and is_recording flag.
    if state.is_recording.load(Ordering::SeqCst) {
        tracing::warn!("Meeting mode: a normal recording is already in progress, ignoring");
        return;
    }

    // Mark meeting_active BEFORE starting recording so the hotkey handler cannot
    // race: if is_recording becomes true before meeting_active is set, a concurrent
    // hotkey press would see is_recording=true, meeting_active=false and behave
    // incorrectly. Reset all flags to a clean state first.
    state.meeting_active.store(true, Ordering::SeqCst);
    state.meeting_cancelled.store(false, Ordering::SeqCst);
    state.meeting_stopping.store(false, Ordering::SeqCst);
    state.meeting_feeder_done.store(false, Ordering::SeqCst);

    // Load the speaker diarization engine *outside* the lock so model I/O
    // (2–5 s) does not block concurrent diarization_ctx readers.
    // Load diarization engine if models are present; no explicit enable toggle.
    let new_diar_engine: Option<diarization::DiarizationEngine> = {
        let model_path = settings::diarization_model_path();
        let seg_path = settings::segmentation_model_path();
        if model_path.exists() && seg_path.exists() {
            match diarization::DiarizationEngine::new(&model_path, Some(&seg_path)) {
                Ok(engine) => Some(engine),
                Err(e) => {
                    tracing::warn!("[diarization] failed to load model: {e}");
                    None
                }
            }
        } else {
            tracing::info!("[diarization] models not found, running meeting without speaker labels");
            None
        }
    };
    // Swap in atomically; lock held only for the pointer swap, not model I/O.
    *state.diarization_ctx.lock().unwrap_or_else(|e| e.into_inner()) = new_diar_engine;

    let preferred_device = state.settings.lock().ok().and_then(|s| s.mic_device.clone());

    // Show overlay in 'preparing' state before opening the mic stream.
    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.emit("recording-status", "preparing");
        center_overlay_bottom(&overlay);
        platform::show_overlay(&overlay);
    }

    // Pause background music for the duration of the meeting.
    if platform::is_now_playing() {
        platform::pause_now_playing();
        state.media_paused_by_sumi.store(true, Ordering::SeqCst);
    }

    if let Err(e) = audio::do_start_recording(
        &state.is_recording,
        &state.mic_available,
        &state.reconnecting,
        &state.sample_rate,
        &state.buffer,
        &state.is_recording,
        &state.audio_thread,
        preferred_device,
    ) {
        tracing::error!("Meeting mode start failed: {}", e);
        state.meeting_active.store(false, Ordering::SeqCst);
        // Resume music if we paused it before do_start_recording.
        if state.media_paused_by_sumi.swap(false, Ordering::SeqCst) {
            platform::resume_now_playing();
        }
        if let Some(overlay) = app.get_webview_window("overlay") {
            let _ = overlay.emit("recording-status", "error");
            let app_for_hide = app.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(1500));
                let app_inner = app_for_hide.clone();
                let _ = app_for_hide.run_on_main_thread(move || {
                    if let Some(ov) = app_inner.get_webview_window("overlay") {
                        platform::hide_overlay(&ov);
                    }
                });
            });
        }
        return;
    }
    tracing::info!("🎙️ Meeting mode started (engine: {:?}, lang: {:?})", stt_mode, lang);
    // Advance the session generation counter. The feeder captures this value
    // and aborts post-loop work if the counter has advanced past it, preventing
    // a zombie feeder from a timed-out previous session from corrupting state.
    let session_id = state.meeting_session.fetch_add(1, Ordering::SeqCst) + 1;
    // Reset normal-recording streaming state in case a previous session left it dirty.
    state.streaming_active.store(false, Ordering::SeqCst);
    if let Ok(mut r) = state.streaming_result.lock() {
        *r = None;
    }
    if let Ok(mut st) = state.meeting_start_time.lock() {
        *st = Some(std::time::Instant::now());
    }

    // Create a meeting note in SQLite and store the note_id.
    {
        let model_label = match stt_mode {
            SttMode::Cloud => format!("Cloud (Meeting) – {}", cloud_config.provider.as_key()),
            SttMode::Local => match local_engine {
                stt::LocalSttEngine::Qwen3Asr => {
                    format!("Qwen3-ASR (Meeting) – {}", qwen3_model.display_name())
                }
                stt::LocalSttEngine::Whisper => {
                    format!("Whisper (Meeting) – {}", whisper_model.display_name())
                }
            },
        };
        let note_id = history::generate_id();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let note = meeting_notes::MeetingNote {
            id: note_id.clone(),
            title: String::new(),
            transcript: String::new(),
            created_at: now,
            updated_at: now,
            duration_secs: 0.0,
            stt_model: model_label,
            is_recording: true,
            word_count: 0,
            summary: String::new(),
            audio_path: None,
        };
        if let Err(e) = meeting_notes::create_note(&settings::history_dir(), &note) {
            tracing::error!("Failed to create meeting note: {}", e);
        }
        if let Ok(mut nid) = state.active_meeting_note_id.lock() {
            *nid = Some(note_id.clone());
        }
        let _ = app.emit("meeting-note-created", serde_json::json!({
            "id": note_id,
            "note": note,
        }));
    }

    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.emit("recording-status", "meeting_recording");
        let _ = overlay.emit("recording-max-duration", 0u64); // 0 = unlimited
        // overlay already shown in 'preparing' state above
    }

    // Audio level monitor thread.
    spawn_audio_level_monitor(app.clone(), AudioMonitorMode::Meeting);

    // Spawn meeting feeder based on STT mode.
    let feeder_app = app.clone();
    match stt_mode {
        SttMode::Local => match local_engine {
            stt::LocalSttEngine::Qwen3Asr => {
                std::thread::spawn(move || {
                    let fstate = feeder_app.state::<AppState>();

                    // Proactively trigger model warm-up (no-op if already loaded).
                    if let Err(e) = qwen3_asr::warm_qwen3_asr(&fstate.qwen3_asr_ctx, &qwen3_model, Some((&fstate.qwen3_ready_cv, &fstate.qwen3_ready_mu))) {
                        tracing::warn!("Meeting mode: engine warm failed: {}", e);
                    }

                    if !qwen3_asr::wait_engine_ready(&fstate.qwen3_asr_ctx, &qwen3_model, &fstate.qwen3_ready_cv, &fstate.qwen3_ready_mu, 8000) {
                        tracing::warn!("Meeting mode: engine not ready after 8s, aborting");
                        fstate.meeting_active.store(false, Ordering::SeqCst);
                        fstate.is_recording.store(false, Ordering::SeqCst);
                        // Clear audio captured during the failed warm-up window to prevent
                        // stale samples from being prepended to the next recording session.
                        if let Ok(mut buf) = fstate.buffer.lock() {
                            buf.clear();
                        }
                        if let Some(ov) = feeder_app.get_webview_window("overlay") {
                            let _ = ov.emit("recording-status", "error");
                        }
                        hide_overlay_delayed(&feeder_app, 2000);
                        return;
                    }
                    qwen3_asr::run_meeting_feeder_loop(feeder_app, lang, session_id);
                });
            }
            stt::LocalSttEngine::Whisper => {
                // Check model is downloaded before spawning the feeder thread.
                if transcribe::whisper_model_path_for(&whisper_model).is_err() {
                    tracing::warn!("Meeting mode: Whisper model not downloaded");
                    state.meeting_active.store(false, Ordering::SeqCst);
                    state.is_recording.store(false, Ordering::SeqCst);
                    if let Ok(mut buf) = state.buffer.lock() {
                        buf.clear();
                    }
                    if let Some(ov) = app.get_webview_window("overlay") {
                        let _ = ov.emit("recording-status", "error");
                    }
                    hide_overlay_delayed(app, 2000);
                    return;
                }
                std::thread::spawn(move || {
                    let fstate = feeder_app.state::<AppState>();
                    // Warm the model (fast for Whisper, typically < 2 s).
                    if let Err(e) = transcribe::warm_whisper_cache(&fstate.whisper_ctx, &whisper_model) {
                        tracing::warn!("[whisper-meeting] warm failed: {}", e);
                        fstate.meeting_active.store(false, Ordering::SeqCst);
                        fstate.is_recording.store(false, Ordering::SeqCst);
                        if let Ok(mut buf) = fstate.buffer.lock() {
                            buf.clear();
                        }
                        if let Some(ov) = feeder_app.get_webview_window("overlay") {
                            let _ = ov.emit("recording-status", "error");
                        }
                        hide_overlay_delayed(&feeder_app, 2000);
                        return;
                    }
                    whisper_streaming::run_whisper_meeting_feeder_loop(feeder_app, lang, session_id);
                });
            }
        },
        SttMode::Cloud => {
            // Populate API key from cache for the cloud feeder thread.
            let mut cloud = cloud_config;
            let key = get_cached_api_key(&state.api_key_cache, cloud.provider.as_key());
            if !key.is_empty() {
                cloud.api_key = key;
            }
            std::thread::spawn(move || {
                stt::run_cloud_meeting_feeder_loop(feeder_app, cloud, lang, session_id);
            });
        }
    }
}

fn stop_meeting_mode(app: &AppHandle) {
    let state = app.state::<AppState>();

    // Use meeting_stopping as the idempotency gate. is_recording may already be
    // false if the cpal dead-stream guard fired before this function ran — using
    // is_recording.swap would silently return without delivering the transcript.
    if state
        .meeting_stopping
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        tracing::warn!("stop_meeting_mode: already in progress, skipping re-entrant call");
        return;
    }

    // Capture session_id at entry. If a new meeting starts between the timeout
    // and the clipboard/history section, this guard prevents a zombie stop from
    // corrupting the new session's state.
    let our_session = state.meeting_session.load(Ordering::SeqCst);

    // Ensure the feeder sees is_recording=false (it may already be false if
    // triggered by dead-stream, which is fine — the store is idempotent).
    state.is_recording.store(false, Ordering::SeqCst);
    if let Ok(mut t) = state.last_recording_end.lock() {
        *t = Some(Instant::now());
    }
    // Wake the meeting feeder immediately so it exits its 2 s sleep and starts
    // post-loop work (trailing feed + finish_streaming) right away.
    state.feeder_stop_cv.notify_all();

    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.emit("recording-status", "processing");
    }

    // Immediately mark meeting inactive — meeting_stopping=true prevents new
    // regular recordings from starting while the worker thread finishes.
    // This removes the 8s two-phase complexity: the segmenter exits within ≤2s
    // and the worker drains the queue independently.
    state.meeting_active.store(false, Ordering::SeqCst);

    // Wait for the worker to finish all segments (up to 5 min for very long
    // final segments recorded without any silence pause). Uses a Condvar so
    // stop_meeting_mode wakes the instant the worker sets meeting_feeder_done.
    let guard = state.meeting_done_mu.lock().unwrap_or_else(|e| e.into_inner());
    let (mu_guard, timed_out) = state
        .meeting_done_cv
        .wait_timeout_while(
            guard,
            std::time::Duration::from_millis(300_000),
            |_| !state.meeting_feeder_done.load(Ordering::SeqCst),
        )
        .unwrap_or_else(|e| e.into_inner());
    drop(mu_guard); // release meeting_done_mu; nothing below needs it
    if timed_out.timed_out() {
        tracing::warn!("Meeting feeder timeout after 5 min — using partial transcript");
        state.meeting_cancelled.store(true, Ordering::SeqCst);
    }

    // Abort if a new meeting session has started while we were waiting for the
    // feeder — another stop_meeting_mode will handle that session's transcript.
    if state.meeting_session.load(Ordering::SeqCst) != our_session {
        tracing::warn!("stop_meeting_mode: session advanced during wait — aborting");
        state.meeting_stopping.store(false, Ordering::SeqCst);
        return;
    }

    let duration_secs = state
        .meeting_start_time
        .lock()
        .ok()
        .and_then(|st| *st)
        .map(|t| t.elapsed().as_secs_f64())
        .unwrap_or(0.0);

    // Finalize the meeting note in SQLite (mark is_recording=0).
    // The transcript is read from the WAL file (the sole source of truth during recording).
    let note_id = state
        .active_meeting_note_id
        .lock()
        .ok()
        .and_then(|nid| nid.clone());
    let hdir = settings::history_dir();
    let raw_transcript = note_id
        .as_deref()
        .map(|id| meeting_notes::read_wal(&hdir, id))
        .unwrap_or_default();

    // Run agglomerative clustering over all buffered embeddings for optimal
    // speaker labels, then update the WAL transcript before writing to SQLite.
    let final_labels: Vec<(f64, f64, String)> = {
        let mut diar = state.diarization_ctx.lock().unwrap_or_else(|e| e.into_inner());
        diar.as_mut()
            .map(|engine| engine.finalize_labels())
            .unwrap_or_default()
    };
    let transcript = if final_labels.is_empty() {
        raw_transcript
    } else {
        tracing::info!(
            "[diarization] applying {} agglomerative labels to WAL",
            final_labels.len()
        );
        meeting_notes::update_wal_speakers(&raw_transcript, &final_labels)
    };

    if let Some(ref id) = note_id {
        if let Err(e) = meeting_notes::finalize_note(&hdir, id, &transcript, duration_secs) {
            tracing::error!("Failed to finalize meeting note: {}", e);
        }
        meeting_notes::remove_wal(&hdir, id);
        // Finalize audio WAL → WAV (no-op if record_meeting_audio was off).
        let audio_path = meeting_notes::finalize_audio(&hdir, id, &settings::audio_dir());
        if let Some(ref ap) = audio_path {
            if let Err(e) = meeting_notes::update_audio_path(&hdir, id, ap) {
                tracing::warn!("Failed to update audio_path for meeting note {}: {}", id, e);
            }
        }
        let _ = app.emit(
            "meeting-note-finalized",
            serde_json::json!({ "id": id }),
        );
    }
    // Clear the active note id.
    if let Ok(mut nid) = state.active_meeting_note_id.lock() {
        *nid = None;
    }

    // Resume media paused when meeting started.
    if state.media_paused_by_sumi.swap(false, Ordering::SeqCst) {
        platform::resume_now_playing();
    }

    if transcript.is_empty() {
        tracing::info!("Meeting mode stopped — no transcript");
        if let Some(overlay) = app.get_webview_window("overlay") {
            let _ = overlay.emit("recording-status", "error");
        }
        hide_overlay_delayed(app, 1500);
        state.meeting_stopping.store(false, Ordering::SeqCst);
        return;
    }

    let word_count = history::count_words(&transcript);
    tracing::info!(
        "Meeting mode stopped ({:.0}s, {} words)",
        duration_secs, word_count
    );
    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.emit("recording-status", "meeting_stopped");
    }
    hide_overlay_delayed(app, 2000);
    // meeting_active is already false: stop_meeting_mode set it unconditionally
    // at line ~1856 before waiting for the feeder to finish.

    // Allow future stop_meeting_mode calls (e.g. for the next meeting session).
    state.meeting_stopping.store(false, Ordering::SeqCst);
}

// ---------------------------------------------------------------------------
// Audio level monitor helpers
// ---------------------------------------------------------------------------

enum AudioMonitorMode {
    Normal,
    Meeting,
}

/// Compute per-bar RMS levels from the current audio buffer, applying adaptive
/// gain (fast attack, slow decay) so the waveform fills the visual range on
/// any platform or mic sensitivity.
///
/// The buffer lock is held only while copying the tail slice; all computation
/// runs after the lock is released to minimise contention with the cpal callback.
fn compute_audio_levels(
    buffer: &std::sync::Mutex<Vec<f32>>,
    num_bars: usize,
    samples_per_bar: usize,
    peak_rms: &mut f32,
) -> Vec<f32> {
    // Copy only the tail we need, then release the lock before computing.
    let tail: Vec<f32> = {
        let Ok(buf) = buffer.lock() else {
            return vec![0.0; num_bars];
        };
        if buf.is_empty() {
            return vec![0.0; num_bars];
        }
        let total = num_bars * samples_per_bar;
        let start = buf.len().saturating_sub(total);
        buf[start..].to_vec()
    };

    let raw_rms: Vec<f32> = tail.chunks(samples_per_bar).map(audio::rms).collect();

    // Fast attack, slow decay (~3.5 s to halve at 50 ms tick).
    let frame_max = raw_rms.iter().cloned().fold(0.0f32, f32::max);
    if frame_max > *peak_rms {
        *peak_rms = frame_max;
    } else {
        *peak_rms *= 0.99;
    }
    *peak_rms = peak_rms.max(0.001);

    // Left-pad with zeros so the waveform always has exactly `num_bars` bars,
    // with the most-recent audio on the right.
    let mut bars = vec![0.0f32; num_bars.saturating_sub(raw_rms.len())];
    bars.extend(raw_rms.iter().map(|&rms| (rms / *peak_rms).min(1.0)));
    bars
}

/// Spawn the 50 ms audio-level monitor thread.
/// `Normal` mode applies max-duration auto-stop and voice-rule-mode forwarding.
/// `Meeting` mode calls `stop_meeting_mode` on dead-stream detection.
fn spawn_audio_level_monitor(app: AppHandle, mode: AudioMonitorMode) {
    std::thread::spawn(move || {
        let state = app.state::<AppState>();
        let sr = state.sample_rate.lock().ok().and_then(|v| *v).unwrap_or(44100) as usize;
        let recording_start = Instant::now();

        const NUM_BARS: usize = 20;
        let samples_per_bar = sr / 20;
        // Adaptive gain: track recent peak RMS so the waveform fills the full
        // visual range regardless of platform mic level.
        let mut peak_rms: f32 = 0.01;
        let is_normal = matches!(mode, AudioMonitorMode::Normal);

        while state.is_recording.load(Ordering::SeqCst) {
            let elapsed = recording_start.elapsed();

            // Dead-stream guard: if the buffer is still empty after 1.5 s the
            // cpal callback is not running at all.
            if elapsed.as_millis() >= 1500 {
                let buf_empty = state.buffer.lock().map(|b| b.is_empty()).unwrap_or(false);
                if buf_empty {
                    state.mic_available.store(false, Ordering::SeqCst);
                    match mode {
                        AudioMonitorMode::Normal => {
                            tracing::warn!(
                                "No audio data after 1.5s — stream dead, aborting recording"
                            );
                            state.is_recording.store(false, Ordering::SeqCst);
                            if let Some(ov) = app.get_webview_window("overlay") {
                                let _ = ov.emit("recording-status", "error");
                            }
                        }
                        AudioMonitorMode::Meeting => {
                            tracing::warn!(
                                "No audio data after 1.5s in meeting mode — stream dead, stopping meeting"
                            );
                            // Spawn a new thread: stop_meeting_mode blocks for up
                            // to 8s waiting for the feeder, which would stall this
                            // 50ms monitor loop for the entire duration.
                            let app_clone = app.clone();
                            std::thread::spawn(move || stop_meeting_mode(&app_clone));
                        }
                    }
                    return;
                }
            }

            // Normal mode only: enforce max recording duration.
            if is_normal && elapsed.as_secs() >= MAX_RECORDING_SECS {
                tracing::info!("Max recording duration reached ({}s)", MAX_RECORDING_SECS);
                if state.edit_mode.load(Ordering::SeqCst) {
                    stop_edit_and_replace(&app);
                } else {
                    stop_transcribe_and_paste(&app);
                }
                return;
            }

            let levels = compute_audio_levels(&state.buffer, NUM_BARS, samples_per_bar, &mut peak_rms);

            if let Some(ov) = app.get_webview_window("overlay") {
                let _ = ov.emit("audio-levels", &levels);
            }
            // Normal mode only: forward levels to the main window for voice-rule visualisation.
            if is_normal && state.voice_rule_mode.load(Ordering::SeqCst) {
                if let Some(main_win) = app.get_webview_window("main") {
                    let _ = main_win.emit("voice-rule-levels", &levels);
                }
            }

            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    });
}

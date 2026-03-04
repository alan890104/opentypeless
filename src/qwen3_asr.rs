use std::sync::{atomic::Ordering, Mutex};
use std::time::Duration;

use tauri::{AppHandle, Emitter, Manager};

use crate::stt::{qwen3_asr_model_dir, is_qwen3_asr_downloaded, Qwen3AsrModel};

// ── Cache ─────────────────────────────────────────────────────────────────────

pub struct Qwen3AsrCache {
    pub engine: qwen3_asr::AsrInference,
    pub model: Qwen3AsrModel,
}

// SAFETY: AsrInference holds candle tensors that are not Send; we guard all
// access with the Mutex in AppState, so only one thread touches it at a time.
unsafe impl Send for Qwen3AsrCache {}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Load (or reuse) the Qwen3-ASR engine for `model`.
///
/// Returns an error string if the model files are missing or loading fails.
pub fn warm_qwen3_asr(
    cache: &Mutex<Option<Qwen3AsrCache>>,
    model: &Qwen3AsrModel,
) -> Result<(), String> {
    let mut guard = cache.lock().unwrap_or_else(|e| {
        tracing::warn!("Qwen3-ASR cache mutex was poisoned; recovering from potentially inconsistent state");
        e.into_inner()
    });

    if let Some(ref c) = *guard {
        if &c.model == model {
            return Ok(());
        }
    }

    let model_dir = qwen3_asr_model_dir(model);
    if !is_qwen3_asr_downloaded(model) {
        return Err(format!(
            "Qwen3-ASR model files not found in {}",
            model_dir.display()
        ));
    }

    tracing::info!("Loading Qwen3-ASR {}...", model.display_name());
    let t0 = std::time::Instant::now();

    let device = qwen3_asr::best_device();
    let engine = qwen3_asr::AsrInference::load(&model_dir, device)
        .map_err(|e| format!("Qwen3-ASR load failed: {}", e))?;

    tracing::info!("Qwen3-ASR {} loaded in {:.1?}", model.display_name(), t0.elapsed());
    *guard = Some(Qwen3AsrCache { engine, model: model.clone() });
    Ok(())
}

/// Transcribe `samples` (16 kHz f32) using the cached Qwen3-ASR engine.
pub fn transcribe_with_cached_qwen3_asr(
    cache: &Mutex<Option<Qwen3AsrCache>>,
    samples: &[f32],
    model: &Qwen3AsrModel,
    language: &str,
) -> Result<String, String> {
    warm_qwen3_asr(cache, model)?;

    let guard = cache.lock().unwrap_or_else(|e| {
        tracing::warn!("Qwen3-ASR cache mutex was poisoned; recovering from potentially inconsistent state");
        e.into_inner()
    });
    let c = guard.as_ref().ok_or("Qwen3-ASR cache empty after warm")?;

    let lang_opt = if language == "auto" || language.is_empty() {
        None
    } else {
        Some(language.to_string())
    };

    let mut opts = qwen3_asr::TranscribeOptions::default();
    if let Some(lang) = lang_opt {
        opts = opts.with_language(lang);
    }
    let result = c
        .engine
        .transcribe_samples(samples, opts)
        .map_err(|e| format!("Qwen3-ASR transcription failed: {}", e))?;

    Ok(result.text)
}

/// Drop the cached engine so a new model can be loaded on the next call.
pub fn invalidate_qwen3_asr_cache(cache: &Mutex<Option<Qwen3AsrCache>>) {
    if let Ok(mut guard) = cache.lock() {
        *guard = None;
    }
}

/// Feeder loop for live-preview streaming transcription.
///
/// Runs in a dedicated thread during recording. Every 2 seconds, reads the
/// new audio delta from `AppState.buffer`, feeds it to the Qwen3-ASR streaming
/// engine, and emits a `"transcription-partial"` event to the overlay window.
///
/// When `is_recording` becomes false, exits the loop, calls `finish_streaming`
/// to flush remaining audio, stores the final text in `AppState.streaming_result`,
/// and clears `AppState.streaming_active`.
///
/// # Safety
/// `StreamingState` contains candle `Tensor` objects and is NOT `Send`. It is
/// created and used entirely within this function (i.e. within the feeder
/// thread), so no cross-thread transfer occurs.
pub(crate) fn run_feeder_loop(app: AppHandle, language: String) {
    let state = app.state::<crate::AppState>();

    // Read the native sample rate once (won't change during recording).
    let sr = state.sample_rate.lock().ok().and_then(|v| *v).unwrap_or(44100);

    // Initialise streaming session while holding the engine lock briefly.
    // SAFETY: `sstate` is only used in this function / this thread.
    let mut sstate = {
        let guard = state.qwen3_asr_ctx.lock().unwrap_or_else(|e| e.into_inner());
        let c = match guard.as_ref() {
            Some(c) => c,
            None => {
                state.streaming_active.store(false, Ordering::SeqCst);
                return;
            }
        };
        let opts = if !language.is_empty() && language != "auto" {
            qwen3_asr::StreamingOptions::default().with_language(&language)
        } else {
            qwen3_asr::StreamingOptions::default()
        };
        c.engine.init_streaming(opts)
        // lock released here
    };

    let mut last_tail: usize = 0;

    // Main loop: every 2 s, feed new audio to the engine.
    loop {
        std::thread::sleep(Duration::from_millis(2000));
        if !state.is_recording.load(Ordering::SeqCst) {
            break;
        }

        // Read only the new delta since the last iteration.
        let delta_raw: Vec<f32> = {
            let buf = state.buffer.lock().unwrap_or_else(|e| e.into_inner());
            let delta = buf[last_tail..].to_vec();
            last_tail = buf.len();
            delta
        };
        if delta_raw.is_empty() {
            continue;
        }

        // Resample to 16 kHz if needed.
        let delta_16k = if sr != 16000 {
            crate::audio::resample(&delta_raw, sr, 16000)
        } else {
            delta_raw
        };

        // Run incremental inference (engine lock held only during this call).
        let partial = {
            let guard = state.qwen3_asr_ctx.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().map(|c| c.engine.feed_audio(&mut sstate, &delta_16k))
        };

        if let Some(Ok(Some(result))) = partial {
            if !result.text.is_empty() {
                tracing::debug!("[streaming] partial: {:?}", result.text);
                if let Some(overlay) = app.get_webview_window("overlay") {
                    let _ = overlay.emit(
                        "transcription-partial",
                        serde_json::json!({ "text": result.text }),
                    );
                }
            }
        }
    }

    // Flush remaining audio and store the final result.
    let final_text = {
        let guard = state.qwen3_asr_ctx.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .as_ref()
            .and_then(|c| c.engine.finish_streaming(&mut sstate).ok())
            .map(|r| r.text)
            .unwrap_or_default()
    };
    tracing::info!("[streaming] finish: {:?}", final_text);

    if let Ok(mut r) = state.streaming_result.lock() {
        *r = if final_text.is_empty() { None } else { Some(final_text) };
    }
    // Store result before clearing active flag (SeqCst ensures visibility ordering).
    state.streaming_active.store(false, Ordering::SeqCst);
}

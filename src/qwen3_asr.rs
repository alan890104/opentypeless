use std::sync::{atomic::Ordering, Condvar, Mutex};
use std::time::Duration;

use tauri::{AppHandle, Manager};

use crate::stt::{qwen3_asr_model_dir, is_qwen3_asr_downloaded, Qwen3AsrModel};

// ── Cache ─────────────────────────────────────────────────────────────────────

pub struct Qwen3AsrCache {
    pub engine: qwen3_asr::AsrInference,
    pub model: Qwen3AsrModel,
}

// Qwen3AsrCache inherits Send+Sync from AsrInference automatically.

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Load (or reuse) the Qwen3-ASR engine for `model`.
///
/// If `ready` is provided as `Some((cv, mu))`, the boolean flag inside `mu`
/// is set to `true` and `cv.notify_all()` is called after the cache is
/// populated.  The flag and the Condvar share the same lock, which eliminates
/// the lost-wakeup race that would exist if they were independent locks.
///
/// Returns an error string if the model files are missing or loading fails.
pub fn warm_qwen3_asr(
    cache: &Mutex<Option<Qwen3AsrCache>>,
    model: &Qwen3AsrModel,
    ready: Option<(&Condvar, &Mutex<bool>)>,
) -> Result<(), String> {
    let mut guard = cache.lock().unwrap_or_else(|e| {
        tracing::warn!("Qwen3-ASR cache mutex was poisoned; recovering from potentially inconsistent state");
        e.into_inner()
    });

    if let Some(ref c) = *guard {
        if &c.model == model {
            // Cache already hot — still notify so any concurrent wait_engine_ready
            // wakes up even if this warm was triggered after the flag was reset
            // (e.g., a future caller that resets the flag without invalidating).
            drop(guard);
            if let Some((cv, mu)) = ready {
                let mut flag = mu.lock().unwrap_or_else(|e| e.into_inner());
                *flag = true;
                drop(flag);
                cv.notify_all();
            }
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
    drop(guard); // release cache lock before touching ready_mu to preserve lock ordering
    if let Some((cv, mu)) = ready {
        let mut flag = mu.lock().unwrap_or_else(|e| e.into_inner());
        *flag = true;
        drop(flag);
        cv.notify_all();
    }
    Ok(())
}

/// Transcribe `samples` (16 kHz f32) using the cached Qwen3-ASR engine.
pub fn transcribe_with_cached_qwen3_asr(
    cache: &Mutex<Option<Qwen3AsrCache>>,
    samples: &[f32],
    model: &Qwen3AsrModel,
    language: &str,
) -> Result<String, String> {
    warm_qwen3_asr(cache, model, None /* batch path: no waiter to notify */)?;

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

/// Wait until the cached Qwen3-ASR engine matches `model`, or `timeout_ms` elapses.
///
/// `ready_mu: &Mutex<bool>` is the **same** mutex that `warm_qwen3_asr` holds
/// when it sets the readiness flag and calls `notify_all`.  Sharing the lock
/// eliminates the lost-wakeup race: `wait_timeout` releases the lock atomically,
/// so there is no window where a notification can fire between the predicate
/// check and the sleep.
///
/// Returns `true` if the engine is ready, `false` on timeout.
pub(crate) fn wait_engine_ready(
    ctx: &Mutex<Option<Qwen3AsrCache>>,
    model: &Qwen3AsrModel,
    ready_cv: &Condvar,
    ready_mu: &Mutex<bool>,
    timeout_ms: u64,
) -> bool {
    let deadline = std::time::Instant::now() + Duration::from_millis(timeout_ms);
    let mut flag = ready_mu.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if *flag {
            // Flag was set — verify the specific model variant is in the cache.
            drop(flag);
            return ctx
                .lock()
                .ok()
                .and_then(|g| g.as_ref().map(|c| c.model == *model))
                .unwrap_or(false);
        }
        let remaining = match deadline.checked_duration_since(std::time::Instant::now()) {
            Some(d) => d,
            None => break,
        };
        // wait_timeout atomically releases the lock and sleeps; no lost-wakeup window.
        let (g, _) = ready_cv.wait_timeout(flag, remaining).unwrap_or_else(|e| e.into_inner());
        flag = g;
    }
    drop(flag);
    // Final check after timeout.
    ctx.lock()
        .ok()
        .and_then(|g| g.as_ref().map(|c| c.model == *model))
        .unwrap_or(false)
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
/// `sstate` is created and used entirely within this function (i.e. within the
/// feeder thread); it is never transferred to another thread.
pub(crate) fn run_feeder_loop(app: AppHandle, language: String, session_id: u64) {
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
        // Lock released here. Safety: qwen3-asr StreamingState is fully owned
        // (contains tensor buffers + KV cache, no references to AsrInference).
        // A concurrent model switch will drop the old engine, but sstate remains
        // valid because it does not borrow from the engine.
    };

    let mut last_tail: usize = 0;

    // Main loop: every 2 s (interruptible), feed new audio to the engine.
    loop {
        {
            let guard = state.feeder_stop_mu.lock().unwrap_or_else(|e| e.into_inner());
            let _ = state.feeder_stop_cv.wait_timeout(guard, Duration::from_millis(2000));
        }
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

        // Skip silence — avoid feeding non-speech frames to the streaming engine,
        // which can cause hallucinations and wastes GPU inference time.
        if !crate::transcribe::has_speech_vad(&state.vad_ctx, &delta_16k, 0.003) {
            continue;
        }

        // Run incremental inference (engine lock held only during this call).
        let partial = {
            let guard = state.qwen3_asr_ctx.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().map(|c| c.engine.feed_audio(&mut sstate, &delta_16k))
        };

        if let Some(Ok(Some(result))) = partial {
            if !result.text.is_empty() {
                tracing::debug!("[streaming] partial: {:?}", result.text);
                crate::emit_transcription_partial(&app, &result.text);
            }
        }
    }

    // If do_stop_recording timed out and signalled a cancel, skip the post-loop
    // engine work so the batch fallback can acquire qwen3_asr_ctx without contention.
    if state.streaming_cancelled.load(Ordering::SeqCst) {
        tracing::info!("[streaming] cancelled — skipping trailing feed, batch fallback will handle it");
        state.streaming_active.store(false, Ordering::SeqCst);
        return;
    }

    // Session guard: if streaming_session has advanced past our ID, a new
    // recording already started and this is a zombie feeder. Discard the result
    // so we don't overwrite the new session's (possibly already stored) result.
    if state.streaming_session.load(Ordering::SeqCst) != session_id {
        tracing::warn!("[streaming] stale feeder (session {} vs current {}) — discarding result", session_id, state.streaming_session.load(Ordering::SeqCst));
        state.streaming_active.store(false, Ordering::SeqCst);
        return;
    }

    // Feed samples that arrived since the last tick (up to 2 s may be unread).
    // IMPORTANT: do_stop_recording drains the buffer with std::mem::take *before*
    // entering the feeder wait, so buf.len() may be 0 by the time we reach here.
    // Clamping last_tail prevents an out-of-bounds panic.
    {
        let trailing_raw: Vec<f32> = {
            let buf = state.buffer.lock().unwrap_or_else(|e| e.into_inner());
            buf[last_tail.min(buf.len())..].to_vec()
        };
        if !trailing_raw.is_empty() {
            let trailing_16k = if sr != 16000 {
                crate::audio::resample(&trailing_raw, sr, 16000)
            } else {
                trailing_raw
            };
            let guard = state.qwen3_asr_ctx.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(c) = guard.as_ref() {
                let _ = c.engine.feed_audio(&mut sstate, &trailing_16k);
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

    // Emit the final complete text as a last partial event so the overlay shows
    // the full transcript (including the last 0–2 s of speech) before the
    // "transcribing" status clears partialText.
    if !final_text.is_empty() {
        crate::emit_transcription_partial(&app, &final_text);
    }

    if let Ok(mut r) = state.streaming_result.lock() {
        *r = if final_text.is_empty() { None } else { Some(final_text) };
    }
    // Store result before clearing active flag (SeqCst ensures visibility ordering).
    state.streaming_active.store(false, Ordering::SeqCst);
}

/// Meeting mode feeder loop for continuous long-form transcription with Qwen3-ASR.
///
/// Delegates to `meeting_feeder::run_meeting_feeder` with a Qwen3-ASR transcription
/// closure.  Force-flushes segments at 120 s to bound per-segment inference cost.
pub(crate) fn run_meeting_feeder_loop(app: tauri::AppHandle, language: String, session_id: u64) {
    let state = app.state::<crate::AppState>();

    let model = {
        let settings = state.settings.lock().unwrap_or_else(|e| e.into_inner());
        settings.stt.qwen3_asr_model.clone()
    };

    // Verify engine is available before entering the loop.
    {
        let guard = state.qwen3_asr_ctx.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            state.meeting_active.store(false, Ordering::SeqCst);
            return;
        }
    }

    let app_for_closure = app.clone();
    let transcribe: Box<dyn FnMut(&[f32], &str) -> String + Send + 'static> =
        Box::new(move |samples, _prev_text| {
            let state = app_for_closure.state::<crate::AppState>();
            transcribe_with_cached_qwen3_asr(&state.qwen3_asr_ctx, samples, &model, &language)
                .unwrap_or_default()
        });
    crate::meeting_feeder::run_meeting_feeder(
        app,
        session_id,
        "qwen3-meeting",
        Some(120 * 16_000),
        transcribe,
    );
}


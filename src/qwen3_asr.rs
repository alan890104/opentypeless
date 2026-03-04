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

    // If do_stop_recording timed out and signalled a cancel, skip the post-loop
    // engine work so the batch fallback can acquire qwen3_asr_ctx without contention.
    if state.streaming_cancelled.load(Ordering::SeqCst) {
        tracing::info!("[streaming] cancelled — skipping trailing feed, batch fallback will handle it");
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

    if let Ok(mut r) = state.streaming_result.lock() {
        *r = if final_text.is_empty() { None } else { Some(final_text) };
    }
    // Store result before clearing active flag (SeqCst ensures visibility ordering).
    state.streaming_active.store(false, Ordering::SeqCst);
}

/// Meeting mode feeder loop for continuous long-form transcription.
///
/// Runs in a dedicated thread during meeting recording. Every 2 seconds it drains
/// the entire buffer (preventing the 10M sample safety cap), resamples to 16 kHz,
/// feeds audio to the Qwen3-ASR streaming engine, and emits `"transcription-partial"`
/// events with the accumulated transcript to the overlay window.
///
/// When continuous silence (RMS < 0.003) is detected for ≥ 4 seconds (2 ticks),
/// the current streaming session is finished and a new one is started, keeping each
/// inference segment short (~30–60 s) without requiring an additional VAD model.
///
/// When `is_recording` becomes false, the loop exits, the final segment is flushed,
/// and the full transcript is written to `AppState.meeting_transcript` before
/// `meeting_active` is set to false.
pub(crate) fn run_meeting_feeder_loop(app: tauri::AppHandle, language: String) {
    let state = app.state::<crate::AppState>();

    let sr = state.sample_rate.lock().ok().and_then(|v| *v).unwrap_or(44100);

    let make_opts = || -> qwen3_asr::StreamingOptions {
        if !language.is_empty() && language != "auto" {
            qwen3_asr::StreamingOptions::default().with_language(&language)
        } else {
            qwen3_asr::StreamingOptions::default()
        }
    };

    // Initialise first streaming session.
    let mut sstate = {
        let guard = state.qwen3_asr_ctx.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some(c) => c.engine.init_streaming(make_opts()),
            None => {
                state.meeting_active.store(false, Ordering::SeqCst);
                return;
            }
        }
    };

    let mut accumulated = String::new();
    let mut silence_count: u32 = 0;
    // Only trigger a silence-reset after real speech has been received since
    // the last reset, preventing repeated resets on empty/silent sessions.
    let mut had_speech_since_reset = false;
    const RMS_THRESHOLD: f32 = 0.003;

    // Tracks how far into the shared buffer we have consumed.
    // We do a partial front-trim every tick to prevent the 10M sample safety cap
    // while keeping the last ~2 s of audio visible to the waveform monitor thread.
    let mut last_tail: usize = 0;
    // Keep 2 s of native-rate samples for the waveform level monitor.
    let waveform_keep: usize = sr as usize * 2;

    loop {
        std::thread::sleep(Duration::from_millis(2000));

        if !state.is_recording.load(Ordering::SeqCst) {
            break;
        }

        // Read new audio since last tick, then trim the front of the buffer so it
        // never exceeds waveform_keep + (one tick of audio) ≈ 4 s at 44.1 kHz.
        // This prevents the 10M-sample safety cap without starving the waveform
        // monitor (which always sees the most-recent waveform_keep samples).
        let delta_raw: Vec<f32> = {
            let mut buf = state.buffer.lock().unwrap_or_else(|e| e.into_inner());
            let delta = buf[last_tail..].to_vec();
            last_tail = buf.len();
            if last_tail > waveform_keep {
                let trim = last_tail - waveform_keep;
                buf.drain(..trim);
                last_tail -= trim; // = waveform_keep
            }
            delta
        };

        if delta_raw.is_empty() {
            // No new audio arrived (should be rare); treat as silence only
            // after real speech has been seen since the last session reset.
            if had_speech_since_reset {
                silence_count += 1;
            }
        } else {
            // Resample to 16 kHz if needed.
            let delta_16k = if sr != 16000 {
                crate::audio::resample(&delta_raw, sr, 16000)
            } else {
                delta_raw
            };

            // RMS-based silence detection over the new chunk.
            let rms = (delta_16k.iter().map(|&s| s * s).sum::<f32>()
                / delta_16k.len() as f32)
                .sqrt();
            if rms < RMS_THRESHOLD {
                if had_speech_since_reset {
                    silence_count += 1;
                }
            } else {
                silence_count = 0;
                had_speech_since_reset = true;
            }

            // Feed audio to the streaming engine.
            let partial = {
                let guard = state.qwen3_asr_ctx.lock().unwrap_or_else(|e| e.into_inner());
                guard.as_ref().map(|c| c.engine.feed_audio(&mut sstate, &delta_16k))
            };

            if let Some(Ok(Some(result))) = partial {
                if !result.text.is_empty() {
                    accumulated.push_str(&result.text);
                    emit_meeting_partial(&app, &accumulated);
                }
            }
        }

        // Reset session on prolonged silence (≥ 2 ticks = 4 s), but only
        // if real speech was received since the last reset.
        if silence_count >= 2 && had_speech_since_reset {
            let mut should_abort = false;
            {
                let guard = state.qwen3_asr_ctx.lock().unwrap_or_else(|e| e.into_inner());

                // Flush remaining tokens from the current segment.
                let seg_text = guard
                    .as_ref()
                    .and_then(|c| c.engine.finish_streaming(&mut sstate).ok())
                    .map(|r| r.text)
                    .unwrap_or_default();
                if !seg_text.is_empty() {
                    accumulated.push_str(&seg_text);
                    emit_meeting_partial(&app, &accumulated);
                }

                // Re-initialise a fresh session for the next speech segment.
                match guard.as_ref() {
                    Some(c) => {
                        sstate = c.engine.init_streaming(make_opts());
                        tracing::debug!("[meeting] Silence reset — new streaming session started");
                    }
                    None => {
                        // Engine was unloaded (e.g. model download started) — abort gracefully.
                        tracing::error!("[meeting] Engine unavailable during silence reset — aborting");
                        should_abort = true;
                    }
                }
            }
            silence_count = 0;
            had_speech_since_reset = false;
            if should_abort {
                break;
            }
        }
    }

    // Feed samples that arrived since the last tick (up to 2 s may be unread).
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

    // If stop_meeting_mode timed out and cancelled us, discard results so this
    // zombie thread cannot overwrite a subsequent meeting session's transcript.
    if state.meeting_cancelled.load(Ordering::SeqCst) {
        tracing::warn!("[meeting] Feeder was cancelled (timeout) — discarding result");
        return;
    }

    // Flush remaining audio from the last segment.
    let final_seg = {
        let guard = state.qwen3_asr_ctx.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .as_ref()
            .and_then(|c| c.engine.finish_streaming(&mut sstate).ok())
            .map(|r| r.text)
            .unwrap_or_default()
    };
    if !final_seg.is_empty() {
        accumulated.push_str(&final_seg);
    }

    tracing::info!("[meeting] Feeder finished. Total transcript length: {} chars", accumulated.len());

    // Store transcript and signal completion.
    if let Ok(mut t) = state.meeting_transcript.lock() {
        *t = accumulated;
    }
    state.meeting_active.store(false, Ordering::SeqCst);
}

fn emit_meeting_partial(app: &tauri::AppHandle, text: &str) {
    if let Some(ov) = app.get_webview_window("overlay") {
        let _ = ov.emit(
            "transcription-partial",
            serde_json::json!({ "text": text }),
        );
    }
}

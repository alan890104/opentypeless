use std::sync::{atomic::Ordering, Mutex};
use std::time::Duration;

use tauri::{AppHandle, Emitter, Manager};

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

/// Poll until the cached Qwen3-ASR engine matches `model`, or `timeout_ms` elapses.
///
/// Returns `true` if the engine is ready, `false` on timeout.
pub(crate) fn wait_engine_ready(
    ctx: &Mutex<Option<Qwen3AsrCache>>,
    model: &Qwen3AsrModel,
    timeout_ms: u64,
) -> bool {
    let mut waited = 0u64;
    while waited < timeout_ms {
        let ready = ctx
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|c| c.model == *model))
            .unwrap_or(false);
        if ready {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
        waited += 200;
    }
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
        // lock released here
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

    if let Ok(mut r) = state.streaming_result.lock() {
        *r = if final_text.is_empty() { None } else { Some(final_text) };
    }
    // Store result before clearing active flag (SeqCst ensures visibility ordering).
    state.streaming_active.store(false, Ordering::SeqCst);
}

/// Meeting mode feeder loop for continuous long-form transcription with Qwen3-ASR.
///
/// Uses **batch transcription per silence-separated segment** (same architecture as
/// Whisper and Cloud meeting feeders). This avoids the streaming API's O(n²)
/// cumulative cost and eliminates snowball risk where inference falls behind audio.
///
/// * Every 2 s, drains new audio into a per-segment `chunk_buf`.
/// * When VAD silence ≥ 2 s (1 tick) AND the chunk is non-empty: batch-transcribes
///   the chunk via `transcribe_with_cached_qwen3_asr`, appends result to WAL file.
/// * When `chunk_buf` exceeds `MAX_SEGMENT_SAMPLES` (120 s), forces a transcription
///   even without silence to bound per-segment cost.
/// * After `is_recording` becomes false, flushes the remaining chunk.
pub(crate) fn run_meeting_feeder_loop(app: tauri::AppHandle, language: String, session_id: u64) {
    let state = app.state::<crate::AppState>();

    let sr = state.sample_rate.lock().ok().and_then(|v| *v).unwrap_or(44100);
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

    // File-based transcript: each segment is appended to a WAL file on disk
    // so the backend never holds a growing String in memory.
    let history_dir = crate::settings::history_dir();
    let note_id: Option<String> = state
        .active_meeting_note_id
        .lock()
        .ok()
        .and_then(|nid| nid.clone());

    let mut chunk_buf: Vec<f32> = Vec::new(); // 16 kHz samples for current segment
    let mut silence_count: u32 = 0;
    let mut had_speech_since_reset = false;
    const RMS_FALLBACK: f32 = 0.003;
    // Force-flush segments longer than 120 s to bound per-segment inference cost.
    const MAX_SEGMENT_SAMPLES: usize = 120 * 16_000;

    // O(1) spacing state.
    let mut has_content = false;
    let mut ends_with_space = false;

    let mut last_tail: usize = 0;
    let waveform_keep: usize = sr as usize * 2;

    loop {
        {
            let guard = state.feeder_stop_mu.lock().unwrap_or_else(|e| e.into_inner());
            let _ = state.feeder_stop_cv.wait_timeout(guard, Duration::from_millis(2000));
        }

        if !state.is_recording.load(Ordering::SeqCst) {
            break;
        }

        // Session guard.
        let current_session = state.meeting_session.load(Ordering::SeqCst);
        if current_session != session_id {
            tracing::warn!(
                "[qwen3-meeting] stale feeder (session {} vs current {}) — aborting",
                session_id,
                current_session
            );
            return;
        }

        // Drain new audio delta; partial-drain keeping waveform_keep for monitor.
        let delta_raw: Vec<f32> = {
            let mut buf = state.buffer.lock().unwrap_or_else(|e| e.into_inner());
            let delta = buf[last_tail..].to_vec();
            last_tail = buf.len();
            if last_tail > waveform_keep {
                let trim = last_tail - waveform_keep;
                buf.drain(..trim);
                last_tail -= trim;
            }
            delta
        };

        if delta_raw.is_empty() {
            if had_speech_since_reset {
                silence_count += 1;
            }
        } else {
            let delta_16k = if sr != 16000 {
                crate::audio::resample(&delta_raw, sr, 16000)
            } else {
                delta_raw
            };

            // Use Silero VAD (falls back to RMS if model unavailable).
            if crate::transcribe::has_speech_vad(&state.vad_ctx, &delta_16k, RMS_FALLBACK) {
                silence_count = 0;
                had_speech_since_reset = true;
            } else if had_speech_since_reset {
                silence_count += 1;
            }

            chunk_buf.extend_from_slice(&delta_16k);
        }

        // Transcribe segment on silence (≥ 2 s) or max segment length exceeded.
        let force_flush = chunk_buf.len() >= MAX_SEGMENT_SAMPLES;
        if ((silence_count >= 1 && had_speech_since_reset) || force_flush) && !chunk_buf.is_empty() {
            if force_flush {
                tracing::info!(
                    "[qwen3-meeting] segment reached {}s — force-flushing",
                    chunk_buf.len() / 16_000
                );
            }

            // Filter chunk through VAD before inference.
            let stt_samples = crate::transcribe::filter_with_vad(&state.vad_ctx, &chunk_buf)
                .unwrap_or_else(|_| chunk_buf.clone());

            let seg_text = if stt_samples.is_empty() {
                String::new()
            } else {
                transcribe_with_cached_qwen3_asr(
                    &state.qwen3_asr_ctx,
                    &stt_samples,
                    &model,
                    &language,
                )
                .unwrap_or_default()
            };

            let mut tick_delta = String::new();
            if !seg_text.is_empty() {
                if has_content && !ends_with_space && !seg_text.starts_with(' ') {
                    tick_delta.push(' ');
                }
                tick_delta.push_str(&seg_text);
            }

            // Add trailing space so the next segment does not fuse with this one.
            if (has_content || !tick_delta.is_empty()) && !tick_delta.ends_with(' ') {
                tick_delta.push(' ');
            }

            // Persist delta to WAL file and emit event.
            if !tick_delta.is_empty() {
                if let Some(ref id) = note_id {
                    crate::meeting_notes::append_wal(&history_dir, id, &tick_delta);
                    let duration = state
                        .meeting_start_time
                        .lock()
                        .ok()
                        .and_then(|st| *st)
                        .map(|t| t.elapsed().as_secs_f64())
                        .unwrap_or(0.0);
                    let _ = app.emit(
                        "meeting-note-updated",
                        serde_json::json!({
                            "id": id,
                            "delta": &tick_delta,
                            "duration_secs": duration,
                        }),
                    );
                }
                ends_with_space = tick_delta.ends_with(' ');
                has_content = true;
            }

            chunk_buf.clear();
            silence_count = 0;
            had_speech_since_reset = false;
        }
    }

    // ── Post-loop guards (mirrors Whisper/Cloud meeting feeders) ───────────────

    let current_session = state.meeting_session.load(Ordering::SeqCst);
    if current_session != session_id {
        tracing::warn!(
            "[qwen3-meeting] stale feeder (session {} vs current {}) — aborting post-loop",
            session_id,
            current_session
        );
        return;
    }
    if state.meeting_cancelled.load(Ordering::SeqCst) {
        tracing::warn!("[qwen3-meeting] feeder cancelled — partial transcript already persisted to file");
        return;
    }

    // Flush remaining chunk (audio received after the last silence window).
    if !chunk_buf.is_empty() {
        let stt_samples = crate::transcribe::filter_with_vad(&state.vad_ctx, &chunk_buf)
            .unwrap_or_else(|_| chunk_buf.clone());
        let seg_text = if stt_samples.is_empty() {
            String::new()
        } else {
            transcribe_with_cached_qwen3_asr(
                &state.qwen3_asr_ctx,
                &stt_samples,
                &model,
                &language,
            )
            .unwrap_or_default()
        };
        if !seg_text.is_empty() {
            let mut final_delta = String::new();
            if has_content && !ends_with_space && !seg_text.starts_with(' ') {
                final_delta.push(' ');
            }
            final_delta.push_str(&seg_text);
            if let Some(ref id) = note_id {
                crate::meeting_notes::append_wal(&history_dir, id, &final_delta);
            }
        }
    }

    tracing::info!("[qwen3-meeting] feeder finished — transcript persisted to WAL file");

    // Signal completion. stop_meeting_mode reads the transcript from the WAL file.
    state.meeting_active.store(false, Ordering::SeqCst);
}


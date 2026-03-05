use std::sync::atomic::Ordering;
use std::time::Duration;

use tauri::{AppHandle, Emitter, Manager};
use whisper_rs::{FullParams, SamplingStrategy};

// ══ WhisperPreviewFeeder ══════════════════════════════════════════════════════

const MAX_BUFFER_SAMPLES: usize = 30 * 16_000; // 30 s at 16 kHz
const MIN_NEW_SAMPLES: usize = 3 * 16_000; // 3 s trigger threshold
const MAX_BACKOFF: u32 = 16;

/// Sliding-window live-preview accumulator inspired by scribble's
/// `BufferedSegmentTranscriber`.
///
/// Buffers 16 kHz audio up to 30 s in a rolling window and applies exponential
/// back-off to reduce inference frequency when the output stabilises (i.e. the
/// speaker has paused or the model has "caught up").
pub struct WhisperPreviewFeeder {
    /// Rolling 16 kHz audio buffer, capped at `MAX_BUFFER_SAMPLES`.
    buffer: Vec<f32>,
    /// Samples added since the last inference call.
    new_samples_since_last_infer: usize,
    /// Exponential back-off multiplier (1..=MAX_BACKOFF).
    /// Doubles on unchanged output; resets to 1 when new text appears.
    backoff: u32,
    /// Last emitted text for back-off comparison.
    last_text: String,
}

impl WhisperPreviewFeeder {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            new_samples_since_last_infer: 0,
            backoff: 1,
            last_text: String::new(),
        }
    }

    /// Append new 16 kHz samples into the rolling buffer.
    pub fn push_samples(&mut self, samples: &[f32]) {
        self.buffer.extend_from_slice(samples);
        self.new_samples_since_last_infer += samples.len();
    }

    /// Whether enough new audio has accumulated to justify another inference pass.
    pub fn should_infer(&self) -> bool {
        self.new_samples_since_last_infer >= MIN_NEW_SAMPLES * self.backoff as usize
    }

    /// Run inference on the current rolling buffer (last 30 s of audio).
    ///
    /// `ctx_guard` must be obtained via `try_lock` (see `run_whisper_preview_loop`)
    /// so the preview feeder never blocks the final batch transcription in
    /// `do_stop_recording`.
    pub fn infer(
        &mut self,
        ctx_guard: &std::sync::MutexGuard<'_, Option<crate::transcribe::WhisperContextCache>>,
        language: &str,
    ) -> Result<String, String> {
        // Trim buffer to the last MAX_BUFFER_SAMPLES before inference.
        if self.buffer.len() > MAX_BUFFER_SAMPLES {
            let drop_n = self.buffer.len() - MAX_BUFFER_SAMPLES;
            self.buffer.drain(..drop_n);
        }

        let c = ctx_guard.as_ref().ok_or("Whisper context not loaded")?;
        let mut wh_state = c
            .ctx
            .create_state()
            .map_err(|e| format!("Failed to create whisper state: {}", e))?;

        // Build params in a nested block so `lang_hint` (which borrows `language`)
        // and `params` (which borrows `lang_hint`) are consumed together before
        // `wh_state` is borrowed again for result extraction.
        let infer_ok = {
            let lang_hint = if language.is_empty() || language == "auto" {
                None
            } else {
                Some(language.split('-').next().unwrap_or(language))
            };
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_language(lang_hint);
            params.set_print_special(false);
            params.set_print_realtime(false);
            params.set_print_progress(false);
            params.set_single_segment(true);
            params.set_no_timestamps(true);
            params.set_no_context(true);
            params.set_temperature_inc(0.6);
            params.set_no_speech_thold(0.5);
            params.set_n_threads(crate::transcribe::num_cpus() as _);
            wh_state
                .full(params, &self.buffer)
                .map_err(|e| format!("Whisper preview inference failed: {}", e))
        };
        infer_ok?;

        let num_segments = wh_state.full_n_segments();
        let mut text = String::new();
        for i in 0..num_segments {
            if let Some(seg) = wh_state.get_segment(i) {
                if seg.no_speech_probability() > 0.5 {
                    continue;
                }
                if let Ok(s) = seg.to_str_lossy() {
                    text.push_str(&s);
                }
            }
        }
        let text = text.trim().to_string();

        // Reset new-sample counter; update back-off.
        self.new_samples_since_last_infer = 0;
        if text == self.last_text {
            self.backoff = (self.backoff * 2).min(MAX_BACKOFF);
        } else {
            self.backoff = 1;
            self.last_text.clone_from(&text);
        }

        Ok(text)
    }
}

// ══ pub(crate) entry points ═══════════════════════════════════════════════════

/// Live-preview feeder for normal (non-meeting) Whisper recordings.
///
/// Spawned at recording start when `stt.local_engine == Whisper`.  Runs every
/// 1 s (interruptible via `feeder_stop_cv`), feeds new audio into a
/// `WhisperPreviewFeeder`, and emits `"transcription-partial"` to the overlay
/// window when enough audio has accumulated.
///
/// Uses `try_lock` on `whisper_ctx` — if the lock is contended (e.g. the final
/// batch transcription has already started), the feeder skips that tick and exits
/// on the next loop check (`is_recording == false`).
pub(crate) fn run_whisper_preview_loop(app: AppHandle, language: String, session_id: u64) {
    let state = app.state::<crate::AppState>();
    let sr = state.sample_rate.lock().ok().and_then(|v| *v).unwrap_or(44100);

    let mut feeder = WhisperPreviewFeeder::new();
    let mut last_tail: usize = 0;

    loop {
        // Interruptible 1 s sleep — woken immediately by `feeder_stop_cv.notify_all()`.
        {
            let guard = state.feeder_stop_mu.lock().unwrap_or_else(|e| e.into_inner());
            let _ = state
                .feeder_stop_cv
                .wait_timeout(guard, Duration::from_millis(1000));
        }

        if !state.is_recording.load(Ordering::SeqCst) {
            break;
        }

        // Zombie-feeder guard: abort if a newer recording session has started.
        if state.whisper_preview_session.load(Ordering::SeqCst) != session_id {
            tracing::warn!(
                "[whisper-preview] stale feeder (session {} vs current {}) — exiting",
                session_id,
                state.whisper_preview_session.load(Ordering::SeqCst)
            );
            break;
        }

        // Read new audio delta since last tick.
        let delta_raw: Vec<f32> = {
            let buf = state.buffer.lock().unwrap_or_else(|e| e.into_inner());
            let tail = last_tail.min(buf.len());
            let delta = buf[tail..].to_vec();
            last_tail = buf.len();
            delta
        };

        if delta_raw.is_empty() {
            continue;
        }

        let delta_16k = if sr != 16000 {
            crate::audio::resample(&delta_raw, sr, 16000)
        } else {
            delta_raw
        };

        feeder.push_samples(&delta_16k);

        if !feeder.should_infer() {
            continue;
        }

        // try_lock: skip this tick rather than blocking the final batch transcription.
        let ctx_guard = match state.whisper_ctx.try_lock() {
            Ok(g) => g,
            Err(_) => {
                tracing::debug!("[whisper-preview] whisper_ctx busy — skipping tick");
                continue;
            }
        };

        if ctx_guard.is_none() {
            continue;
        }

        let result = feeder.infer(&ctx_guard, &language);
        drop(ctx_guard); // release the lock as early as possible

        // Session guard again after inference (may have taken ~0.5–2 s).
        if state.whisper_preview_session.load(Ordering::SeqCst) != session_id {
            break;
        }

        match result {
            Ok(text) if !text.is_empty() => {
                tracing::debug!("[whisper-preview] partial: {:?}", text);
                crate::emit_transcription_partial(&app, &text);
            }
            Err(e) => {
                tracing::warn!("[whisper-preview] inference error: {}", e);
            }
            _ => {}
        }
    }

    state
        .whisper_preview_active
        .store(false, Ordering::SeqCst);
    tracing::info!("[whisper-preview] feeder exited (session {})", session_id);
}

// ── Meeting feeder helpers ────────────────────────────────────────────────────

/// Transcribe `samples` (16 kHz) from the already-loaded `WhisperContextCache`.
/// Uses `initial_prompt` for previous-segment context biasing.
/// Returns an empty string (never an error) if the context is not loaded.
fn transcribe_meeting_chunk<'a>(
    ctx_guard: &std::sync::MutexGuard<'_, Option<crate::transcribe::WhisperContextCache>>,
    samples: &[f32],
    language: &'a str,
    initial_prompt: Option<&'a str>,
) -> Result<String, String> {
    let c = ctx_guard.as_ref().ok_or("Whisper context not loaded")?;
    let mut wh_state = c
        .ctx
        .create_state()
        .map_err(|e| format!("Failed to create whisper state: {}", e))?;

    let infer_ok = {
        let lang_hint = if language.is_empty() || language == "auto" {
            None
        } else {
            Some(language.split('-').next().unwrap_or(language))
        };
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(lang_hint);
        if let Some(prompt) = initial_prompt {
            if !prompt.is_empty() {
                params.set_initial_prompt(prompt);
            }
        }
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_progress(false);
        // Allow multiple segments for longer chunks.
        params.set_single_segment(false);
        params.set_no_timestamps(true);
        params.set_no_context(true);
        params.set_temperature_inc(0.6);
        params.set_no_speech_thold(0.5);
        params.set_n_threads(crate::transcribe::num_cpus() as _);
        wh_state
            .full(params, samples)
            .map_err(|e| format!("Whisper meeting inference failed: {}", e))
    };
    infer_ok?;

    let num_segments = wh_state.full_n_segments();
    let mut text = String::new();
    for i in 0..num_segments {
        if let Some(seg) = wh_state.get_segment(i) {
            if seg.no_speech_probability() > 0.5 {
                continue;
            }
            if let Ok(s) = seg.to_str_lossy() {
                text.push_str(&s);
            }
        }
    }
    Ok(text.trim().to_string())
}

/// Meeting-mode feeder for continuous long-form transcription with Whisper.
///
/// Analogous to `qwen3_asr::run_meeting_feeder_loop`, but uses batch Whisper
/// inference per silence-separated segment instead of incremental streaming.
///
/// * Every 2 s, drains new audio into a per-segment chunk buffer.
/// * When RMS silence ≥ 4 s (2 ticks) AND the chunk is non-empty: transcribes
///   the chunk with the previous accumulated text as `initial_prompt`, appends
///   to `accumulated`, emits `"transcription-partial"`.
/// * After `is_recording` becomes false, flushes the remaining chunk and writes
///   the transcript to SQLite via `finalize_note` before clearing `meeting_active`.
pub(crate) fn run_whisper_meeting_feeder_loop(app: AppHandle, language: String, session_id: u64) {
    let state = app.state::<crate::AppState>();
    let sr = state.sample_rate.lock().ok().and_then(|v| *v).unwrap_or(44100);

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
    // Only trigger a silence-reset after real speech has been received since
    // the last reset, preventing repeated resets on empty/silent sessions.
    let mut had_speech_since_reset = false;
    const RMS_FALLBACK: f32 = 0.003;

    // O(1) spacing state.
    let mut has_content = false;
    let mut ends_with_space = false;

    let mut last_tail: usize = 0;
    // Keep 2 s of native-rate samples for the waveform level monitor.
    let waveform_keep: usize = sr as usize * 2;

    loop {
        {
            let guard = state.feeder_stop_mu.lock().unwrap_or_else(|e| e.into_inner());
            let _ = state
                .feeder_stop_cv
                .wait_timeout(guard, Duration::from_millis(2000));
        }

        if !state.is_recording.load(Ordering::SeqCst) {
            break;
        }

        // Session guard.
        let current_session = state.meeting_session.load(Ordering::SeqCst);
        if current_session != session_id {
            tracing::warn!(
                "[whisper-meeting] stale feeder (session {} vs current {}) — aborting",
                session_id,
                current_session
            );
            return;
        }

        // Drain new audio delta; partial-drain the shared buffer to prevent the
        // 10M-sample safety cap while keeping `waveform_keep` samples for the
        // waveform level monitor.
        let delta_raw: Vec<f32> = {
            let mut buf = state.buffer.lock().unwrap_or_else(|e| e.into_inner());
            let tail = last_tail.min(buf.len());
            let delta = buf[tail..].to_vec();
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

        // Silence reset: ≥ 2 s of silence after speech → transcribe and reset chunk.
        if silence_count >= 1 && had_speech_since_reset && !chunk_buf.is_empty() {
            // Read last ~200 chars from WAL file for Whisper context prompt.
            let prev_text = if let Some(ref id) = note_id {
                let full = crate::meeting_notes::read_wal(&history_dir, id);
                let trimmed = full.trim_end();
                if trimmed.len() > 200 {
                    trimmed[trimmed.len().saturating_sub(200)..].to_string()
                } else {
                    trimmed.to_string()
                }
            } else {
                String::new()
            };

            // Filter chunk through VAD before Whisper inference.
            let stt_samples = crate::transcribe::filter_with_vad(&state.vad_ctx, &chunk_buf)
                .unwrap_or_else(|_| chunk_buf.clone());

            let seg_text = if stt_samples.is_empty() {
                String::new()
            } else {
                let ctx_guard = state.whisper_ctx.lock().unwrap_or_else(|e| e.into_inner());
                transcribe_meeting_chunk(
                    &ctx_guard,
                    &stt_samples,
                    &language,
                    if prev_text.is_empty() {
                        None
                    } else {
                        Some(&prev_text)
                    },
                )
                .unwrap_or_default()
            };

            let mut tick_delta = String::new();
            if !seg_text.is_empty() {
                // Insert a space to prevent segment fusion.
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

    // ── Post-loop guards (mirrors qwen3_asr::run_meeting_feeder_loop) ────────

    let current_session = state.meeting_session.load(Ordering::SeqCst);
    if current_session != session_id {
        tracing::warn!(
            "[whisper-meeting] stale feeder (session {} vs current {}) — aborting post-loop",
            session_id,
            current_session
        );
        return;
    }
    if state.meeting_cancelled.load(Ordering::SeqCst) {
        tracing::warn!("[whisper-meeting] feeder cancelled — partial transcript already persisted to file");
        return;
    }

    // Flush remaining chunk (audio received after the last silence window).
    if !chunk_buf.is_empty() {
        // Read last ~200 chars from WAL file for Whisper context prompt.
        let prev_text = if let Some(ref id) = note_id {
            let full = crate::meeting_notes::read_wal(&history_dir, id);
            let trimmed = full.trim_end();
            if trimmed.len() > 200 {
                trimmed[trimmed.len().saturating_sub(200)..].to_string()
            } else {
                trimmed.to_string()
            }
        } else {
            String::new()
        };

        let stt_samples = crate::transcribe::filter_with_vad(&state.vad_ctx, &chunk_buf)
            .unwrap_or_else(|_| chunk_buf.clone());
        let seg_text = if stt_samples.is_empty() {
            String::new()
        } else {
            let ctx_guard = state.whisper_ctx.lock().unwrap_or_else(|e| e.into_inner());
            transcribe_meeting_chunk(
                &ctx_guard,
                &stt_samples,
                &language,
                if prev_text.is_empty() {
                    None
                } else {
                    Some(&prev_text)
                },
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

    tracing::info!("[whisper-meeting] feeder finished — transcript persisted to WAL file");

    // Signal completion. stop_meeting_mode reads the transcript from the WAL file.
    state.meeting_active.store(false, Ordering::SeqCst);
}

use std::sync::atomic::Ordering;
use std::time::Duration;

use tauri::{AppHandle, Manager};
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

    // Final inference pass: run one last inference on whatever audio has
    // accumulated since the last tick so the overlay shows the complete
    // transcript before the "transcribing" status clears partialText.
    // Skip if session has already advanced (zombie feeder).
    //
    // Trailing audio is read BEFORE the is_empty check: if the feeder buffer
    // is empty but audio arrived in state.buffer after the last tick, we still
    // want to run inference on it.
    if state.whisper_preview_session.load(Ordering::SeqCst) == session_id {
        let trailing_raw: Vec<f32> = {
            let buf = state.buffer.lock().unwrap_or_else(|e| e.into_inner());
            let tail = last_tail.min(buf.len());
            buf[tail..].to_vec()
        };
        if !trailing_raw.is_empty() {
            let trailing_16k = if sr != 16000 {
                crate::audio::resample(&trailing_raw, sr, 16000)
            } else {
                trailing_raw
            };
            feeder.push_samples(&trailing_16k);
        }

        if !feeder.buffer.is_empty() {
            if let Ok(ctx_guard) = state.whisper_ctx.try_lock() {
                if ctx_guard.is_some() {
                    if let Ok(text) = feeder.infer(&ctx_guard, &language) {
                        if !text.is_empty() {
                            tracing::debug!("[whisper-preview] final partial: {:?}", text);
                            crate::emit_transcription_partial(&app, &text);
                        }
                    }
                }
            }
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
/// `audio_start_secs` is used to anchor DTW word timestamps to meeting time.
/// Returns `(text, words)`. On error or missing context, returns `("", vec![])`.
pub(crate) fn transcribe_meeting_chunk<'a>(
    ctx_guard: &std::sync::MutexGuard<'_, Option<crate::transcribe::WhisperContextCache>>,
    samples: &[f32],
    language: &'a str,
    initial_prompt: Option<&'a str>,
    audio_start_secs: f64,
) -> Result<(String, Vec<crate::meeting_notes::WordTs>), String> {
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

    let words = crate::transcribe::extract_dtw_words(&wh_state, audio_start_secs);
    Ok((text.trim().to_string(), words))
}

/// Meeting-mode feeder for continuous long-form transcription with Whisper.
///
/// Two-phase diarization:
///   Real-time: segmentation model splits each VAD chunk at speaker boundaries;
///              WeSpeaker + online clustering assigns immediate labels.
///   Finalization: agglomerative clustering re-labels all segments optimally.
///
/// Word timestamps from DTW allow precise text assignment to each sub-segment.
pub(crate) fn run_whisper_meeting_feeder_loop(app: AppHandle, language: String, session_id: u64) {
    use crate::meeting_notes::WalSegment;

    let app_for_closure = app.clone();
    let transcribe: Box<
        dyn FnMut(&[f32], f64, f64, &str) -> Vec<WalSegment> + Send + 'static,
    > = Box::new(move |samples, start_secs, end_secs, prev_text| {
        let state = app_for_closure.state::<crate::AppState>();

        // Phase 1: diarization sub-segmentation (segmentation model + online cluster).
        // Returns (start_abs, end_abs, speaker_label) per sub-segment within this chunk.
        // guard_model_op rejects delete_diarization_model while meeting_active=true, so
        // holding the lock for the full inference duration is safe: no contention with delete.
        let sub_segs: Vec<(f64, f64, String)> = {
            let mut ctx = state.diarization_ctx.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(ref mut engine) = *ctx {
                let segs = engine.process_vad_chunk(samples, start_secs);
                if segs.is_empty() {
                    // Diarization engine found no speech — fall back to whole chunk.
                    vec![(start_secs, end_secs, String::new())]
                } else {
                    segs
                }
            } else {
                // No diarization engine — whole chunk as one segment.
                vec![(start_secs, end_secs, String::new())]
            }
        };

        // STT on full chunk (better quality than per-sub-segment due to context).
        let ctx_guard = state.whisper_ctx.lock().unwrap_or_else(|e| e.into_inner());
        let (text, words) = transcribe_meeting_chunk(
            &ctx_guard,
            samples,
            &language,
            if prev_text.is_empty() { None } else { Some(prev_text) },
            start_secs,
        )
        .unwrap_or_default();
        drop(ctx_guard);

        if sub_segs.len() == 1 {
            // Fast path: no sub-segmentation.
            let (s, e, speaker) = sub_segs.into_iter().next().unwrap();
            return if text.is_empty() {
                vec![]
            } else {
                vec![WalSegment { speaker, start: s, end: e, text, words }]
            };
        }

        // Assign words to sub-segments by their absolute timestamp.
        // Words not covered by any sub-segment are assigned to the last sub-segment.
        let mut result: Vec<WalSegment> = sub_segs
            .iter()
            .map(|(s, e, speaker)| WalSegment {
                speaker: speaker.clone(),
                start: *s,
                end: *e,
                text: String::new(),
                words: Vec::new(),
            })
            .collect();

        for word in &words {
            // Find the sub-segment whose time range contains this word.
            let idx = sub_segs
                .iter()
                .position(|(s, e, _)| word.s >= *s && word.s < *e)
                .unwrap_or(result.len() - 1);
            result[idx].words.push(word.clone());
        }

        // Build text from assigned words; fall back to full text for the
        // longest sub-segment if words are missing (e.g. DTW disabled).
        let any_words = result.iter().any(|s| !s.words.is_empty());
        if any_words {
            for seg in &mut result {
                seg.text = seg
                    .words
                    .iter()
                    .map(|w| w.w.trim())
                    .filter(|w| !w.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");
            }
        } else {
            // No word timestamps — assign full text to the longest sub-segment.
            let longest = result
                .iter()
                .enumerate()
                .max_by_key(|(_, s)| ((s.end - s.start) * 1000.0) as u64)
                .map(|(i, _)| i)
                .unwrap_or(0);
            result[longest].text = text;
        }

        // Do NOT retain only non-empty segments: persist_and_emit writes all sub-segments
        // to WAL (suppressing the frontend event for empty-text ones) so that
        // update_wal_speakers can find them by (start, end) timestamp at meeting stop.
        result
    });

    // Cap each segment at 120 s so the Final segment never exceeds ~12 s of
    // Whisper inference time, keeping stop_meeting_mode well within the 5-min timeout.
    crate::meeting_feeder::run_meeting_feeder(
        app,
        session_id,
        "whisper-meeting",
        Some(120 * 16_000),
        transcribe,
    );
}

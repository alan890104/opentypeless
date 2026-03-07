//! Shared meeting feeder loop.
//!
//! All three meeting feeders (Qwen3-ASR, Whisper, Cloud) follow the same
//! structure: 2 s tick → audio drain → VAD silence detection → transcribe
//! on silence → WAL append → emit event.  The only difference is the actual
//! STT call, which is injected via a closure.
//!
//! Architecture: Two-thread producer-consumer pipeline.
//!
//! Segmenter thread (fast, 2s tick):
//!   drain audio → VAD → accumulate chunk_buf → on silence/force-flush →
//!   send Segment::Tick(chunk); on loop exit → send Segment::Final(remaining)
//!
//! Worker thread (blocking, any duration):
//!   recv segment → filter_with_vad → transcribe → session check →
//!   append_wal → emit event
//!
//! Key benefit: segmenter exits within ≤2s after is_recording=false, so
//! meeting_active can be set false immediately (guarded by meeting_stopping).
//! Worker drains the queue independently without losing any audio.

use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Duration;

use tauri::{AppHandle, Emitter, Manager};

use crate::segment_spacing::SpacingState;

/// Segment types passed from segmenter to worker.
enum Segment {
    Tick(Vec<f32>),
    Final(Vec<f32>),
}

/// Read the last ~200 characters from the WAL file for use as a context prompt.
fn read_wal_context(history_dir: &std::path::Path, note_id: &Option<String>) -> String {
    if let Some(ref id) = note_id {
        let full = crate::meeting_notes::read_wal(history_dir, id);
        let trimmed = full.trim_end();
        // Use char-aware slicing to avoid panics on multi-byte UTF-8 (Chinese, etc.)
        let char_count = trimmed.chars().count();
        if char_count > 200 {
            trimmed
                .char_indices()
                .nth(char_count - 200)
                .map(|(i, _)| &trimmed[i..])
                .unwrap_or(trimmed)
                .to_string()
        } else {
            trimmed.to_string()
        }
    } else {
        String::new()
    }
}

/// Persist a delta to the WAL file and emit a `meeting-note-updated` event.
fn persist_and_emit(
    app: &AppHandle,
    state: &crate::AppState,
    history_dir: &std::path::Path,
    note_id: &Option<String>,
    delta: &str,
) {
    if delta.is_empty() {
        return;
    }
    if let Some(ref id) = note_id {
        crate::meeting_notes::append_wal(history_dir, id, delta);
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
                "delta": delta,
                "duration_secs": duration,
            }),
        );
    }
}

/// Worker thread: receives segments from the segmenter, transcribes them,
/// and persists/emits each result.
///
/// Owns `SpacingState` so spacing is consistent across all segments regardless
/// of how long each transcription takes.
fn run_meeting_worker(
    app: AppHandle,
    session_id: u64,
    label: &str,
    rx: mpsc::Receiver<Segment>,
    mut transcribe: Box<dyn FnMut(&[f32], &str) -> String + Send + 'static>,
) {
    let state = app.state::<crate::AppState>();
    let history_dir = crate::settings::history_dir();
    let note_id: Option<String> = state
        .active_meeting_note_id
        .lock()
        .ok()
        .and_then(|nid| nid.clone());

    let mut spacing = SpacingState::new();

    for segment in rx {
        // Pre-transcription session check.
        let current_session = state.meeting_session.load(Ordering::SeqCst);
        if current_session != session_id {
            tracing::warn!(
                "[{label}] worker: stale session ({session_id} vs {current_session}) — aborting, remaining segments will be dropped",
            );
            break;
        }

        // Check cancellation — skip remaining segments.
        if state.meeting_cancelled.load(Ordering::SeqCst) {
            tracing::warn!("[{label}] worker: cancelled — skipping remaining segments");
            break;
        }

        let (samples, is_final) = match segment {
            Segment::Tick(s) => (s, false),
            Segment::Final(s) => (s, true),
        };

        if samples.is_empty() {
            if is_final {
                // Empty Final = clean stop with no trailing audio; nothing more will arrive.
                tracing::debug!("[{label}] worker: empty final segment, nothing to transcribe");
                break;
            }
            continue;
        }

        let prev_text = read_wal_context(&history_dir, &note_id);

        let stt_samples = crate::transcribe::filter_with_vad(&state.vad_ctx, &samples)
            .unwrap_or(samples);

        let seg_text = if stt_samples.is_empty() {
            String::new()
        } else {
            transcribe(&stt_samples, &prev_text)
        };

        // Post-transcription session check (transcription may take 10–60s).
        let current_session = state.meeting_session.load(Ordering::SeqCst);
        if current_session != session_id {
            tracing::warn!(
                "[{label}] worker: session advanced during transcription ({session_id} vs {current_session}) — discarding result",
            );
            break;
        }

        if state.meeting_cancelled.load(Ordering::SeqCst) {
            tracing::warn!("[{label}] worker: cancelled after transcription — discarding result");
            break;
        }

        let delta = if is_final {
            spacing.build_final_delta(&seg_text)
        } else {
            spacing.build_tick_delta(&seg_text)
        };
        persist_and_emit(&app, &state, &history_dir, &note_id, &delta);
    }

    tracing::info!("[{label}] worker: finished processing all segments");
}

/// Segmenter thread: accumulates audio in 2s ticks, sends completed segments
/// to the worker via channel. Exits within ≤2s after `is_recording=false`.
fn run_meeting_segmenter(
    app: &AppHandle,
    session_id: u64,
    label: &str,
    max_segment_samples: Option<usize>,
    tx: mpsc::Sender<Segment>,
    record_audio: bool,
    audio_note_id: &Option<String>,
) {
    let state = app.state::<crate::AppState>();
    let sr = state
        .sample_rate
        .lock()
        .ok()
        .and_then(|v| *v)
        .unwrap_or(44100);

    let mut chunk_buf: Vec<f32> = Vec::new();
    let mut silence_count: u32 = 0;
    let mut had_speech_since_reset = false;
    const RMS_FALLBACK: f32 = 0.003;

    let mut last_tail: usize = 0;
    let waveform_keep: usize = sr as usize * 2;

    loop {
        {
            let guard = state
                .feeder_stop_mu
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let _ = state
                .feeder_stop_cv
                .wait_timeout(guard, Duration::from_millis(2000));
        }

        if !state.is_recording.load(Ordering::SeqCst) {
            // Final drain: pick up audio recorded between the last 2s tick and
            // the stop signal.  Without this, up to ~2s of speech at the end
            // of the recording is silently discarded.
            let remaining_raw: Vec<f32> = {
                let buf = state.buffer.lock().unwrap_or_else(|e| e.into_inner());
                let tail = last_tail.min(buf.len());
                buf[tail..].to_vec()
            };
            if !remaining_raw.is_empty() {
                let remaining_16k = if sr != 16000 {
                    crate::audio::resample(&remaining_raw, sr, 16000)
                } else {
                    remaining_raw
                };
                chunk_buf.extend_from_slice(&remaining_16k);
            }
            break;
        }

        // Session guard.
        let current_session = state.meeting_session.load(Ordering::SeqCst);
        if current_session != session_id {
            tracing::warn!(
                "[{label}] segmenter: stale session ({session_id} vs {current_session}) — aborting",
            );
            return;
        }

        // Drain new audio delta; partial-drain keeping waveform_keep for the
        // audio-level monitor.
        let delta_raw: Vec<f32> = {
            let mut buf = state
                .buffer
                .lock()
                .unwrap_or_else(|e| e.into_inner());
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

            if crate::transcribe::has_speech_vad(&state.vad_ctx, &delta_16k, RMS_FALLBACK) {
                silence_count = 0;
                had_speech_since_reset = true;
            } else if had_speech_since_reset {
                silence_count += 1;
            }

            chunk_buf.extend_from_slice(&delta_16k);
        }

        // Send segment on silence (≥ 2 s) or max segment length exceeded.
        let force_flush = max_segment_samples.map_or(false, |max| chunk_buf.len() >= max);
        if ((silence_count >= 1 && had_speech_since_reset) || force_flush)
            && !chunk_buf.is_empty()
        {
            if force_flush {
                tracing::info!(
                    "[{label}] segment reached {}s — force-flushing",
                    chunk_buf.len() / 16_000
                );
            }
            if record_audio {
                if let Some(ref id) = *audio_note_id {
                    crate::meeting_notes::append_audio_wal(
                        &crate::settings::history_dir(),
                        id,
                        &chunk_buf,
                    );
                }
            }
            let chunk = std::mem::take(&mut chunk_buf);
            // If send fails (worker exited early due to session change), stop.
            if tx.send(Segment::Tick(chunk)).is_err() {
                tracing::warn!("[{label}] segmenter: worker channel closed — stopping");
                return;
            }
            silence_count = 0;
            had_speech_since_reset = false;
        }
    }

    // Send remaining audio as Final segment (empty = clean stop with no trailing audio).
    tracing::info!(
        "[{label}] segmenter: loop exited, sending Final segment ({} samples)",
        chunk_buf.len()
    );
    if record_audio && !chunk_buf.is_empty() {
        if let Some(ref id) = *audio_note_id {
            crate::meeting_notes::append_audio_wal(
                &crate::settings::history_dir(),
                id,
                &chunk_buf,
            );
        }
    }
    let _ = tx.send(Segment::Final(chunk_buf));
    // Normal path: tx dropped here → channel closes → worker's for-loop exits.
    // Failure path: if the worker already exited early (session change or cancellation),
    // send returns Err and the trailing audio is intentionally discarded.
}

/// Shared meeting feeder — two-thread producer-consumer pipeline.
///
/// `label` is used for log messages (e.g. `"qwen3-meeting"`).
///
/// `max_segment_samples` enables force-flushing segments that exceed a
/// duration cap (e.g. `Some(120 * 16_000)` for 120 s).  Pass `None` to
/// disable.
///
/// `transcribe` receives VAD-filtered 16 kHz samples and the previous WAL
/// context string (~200 chars).  It should return the transcribed text
/// segment, or an empty string on failure / no speech.
///
/// `record_audio` — when true, each segment's raw samples are appended to an
/// audio WAL file alongside the transcript WAL.  `stop_meeting_mode` converts
/// this to a WAV file after the feeder exits.
pub(crate) fn run_meeting_feeder(
    app: AppHandle,
    session_id: u64,
    label: &str,
    max_segment_samples: Option<usize>,
    transcribe: Box<dyn FnMut(&[f32], &str) -> String + Send + 'static>,
    record_audio: bool,
) {
    let (tx, rx) = mpsc::channel::<Segment>();

    // Read note_id once so the segmenter and worker both see the same value.
    let audio_note_id: Option<String> = {
        let state = app.state::<crate::AppState>();
        state.active_meeting_note_id.lock().ok().and_then(|n| n.clone())
    };

    let label_owned = label.to_string();
    let worker_app = app.clone();
    let worker = std::thread::spawn(move || {
        run_meeting_worker(worker_app, session_id, &label_owned, rx, transcribe);
    });

    run_meeting_segmenter(&app, session_id, label, max_segment_samples, tx, record_audio, &audio_note_id);
    // tx dropped when run_meeting_segmenter returns → channel closes → worker exits.

    tracing::info!("[{label}] segmenter done, waiting for worker to finish");
    match worker.join() {
        Ok(()) => {}
        Err(_) => tracing::error!("[{label}] worker thread panicked — transcript may be incomplete"),
    }
    tracing::info!("[{label}] feeder finished — transcript persisted to WAL file");

    // Signal completion only if we still own the session. A stale feeder from a
    // timed-out session (>5 min segment) must not set meeting_feeder_done=true
    // or notify stop_meeting_mode for a newer session, which would cause it to
    // finalize the WAL before session N+1's worker has written anything.
    // In the normal path meeting_session always equals session_id here.
    let state = app.state::<crate::AppState>();
    if state.meeting_session.load(Ordering::SeqCst) == session_id {
        // Hold meeting_done_mu while setting meeting_feeder_done so that
        // stop_meeting_mode cannot observe the flag-false → wait path between
        // its predicate check and the wait_timeout_while sleep entry.  Without
        // this, a missed-wakeup is possible:
        //   1. stop_meeting_mode holds the mutex, checks predicate → false (must wait)
        //   2. we store meeting_feeder_done=true + notify_all without the mutex
        //      → no thread is sleeping yet, signal is lost
        //   3. wait_timeout_while atomically releases the mutex and blocks
        //      → nobody wakes it; sleeps for the full 5-min timeout
        // Holding the mutex here ensures step 2 can only execute after step 3
        // has atomically moved stop_meeting_mode into the waiting state.
        {
            let _guard = state.meeting_done_mu.lock().unwrap_or_else(|e| e.into_inner());
            state.meeting_feeder_done.store(true, Ordering::SeqCst);
        }
        // Notify after releasing the mutex (standard condvar pattern).
        state.meeting_done_cv.notify_all();
    }
}

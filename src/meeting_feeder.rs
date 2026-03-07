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

/// Segment types passed from segmenter to worker.
enum Segment {
    Tick { samples: Vec<f32>, start_secs: f64, end_secs: f64 },
    Final { samples: Vec<f32>, start_secs: f64, end_secs: f64 },
}

/// Read the last ~200 plain-text characters from the WAL file for STT context biasing.
fn read_wal_context(history_dir: &std::path::Path, note_id: &Option<String>) -> String {
    if let Some(ref id) = note_id {
        let full = crate::meeting_notes::read_wal(history_dir, id);
        crate::meeting_notes::wal_text_for_context(&full, 200)
    } else {
        String::new()
    }
}

/// Persist a WAL segment and emit a `meeting-note-updated` event.
fn persist_and_emit(
    app: &AppHandle,
    state: &crate::AppState,
    history_dir: &std::path::Path,
    note_id: &Option<String>,
    segment: &crate::meeting_notes::WalSegment,
) {
    if let Some(ref id) = note_id {
        // Always write to WAL so agglomerative relabeling can find this segment
        // by (start, end) timestamp even when text is empty (e.g. non-primary
        // Qwen3-ASR diarization sub-segments).
        crate::meeting_notes::append_wal(history_dir, id, segment);
        // Only emit the frontend delta event when there is actual text.
        if !segment.text.is_empty() {
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
                    "delta": segment.text,
                    "speaker": segment.speaker,
                    "duration_secs": duration,
                }),
            );
        }
    }
}

/// Worker thread: receives segments from the segmenter, transcribes them,
/// and persists/emits each result.
///
/// The `transcribe` closure returns `Vec<WalSegment>` — one entry per
/// diarization sub-segment within the VAD chunk (may be >1 when the
/// segmentation model detects an intra-utterance speaker change).
fn run_meeting_worker(
    app: AppHandle,
    session_id: u64,
    label: &str,
    rx: mpsc::Receiver<Segment>,
    mut transcribe: Box<
        dyn FnMut(&[f32], f64, f64, &str) -> Vec<crate::meeting_notes::WalSegment>
            + Send
            + 'static,
    >,
) {
    let state = app.state::<crate::AppState>();
    let history_dir = crate::settings::history_dir();
    let note_id: Option<String> = state
        .active_meeting_note_id
        .lock()
        .ok()
        .and_then(|nid| nid.clone());

    for segment in rx {
        // Pre-transcription session check.
        let current_session = state.meeting_session.load(Ordering::SeqCst);
        if current_session != session_id {
            tracing::warn!(
                "[{label}] worker: stale session ({session_id} vs {current_session}) — aborting",
            );
            break;
        }

        if state.meeting_cancelled.load(Ordering::SeqCst) {
            tracing::warn!("[{label}] worker: cancelled — skipping remaining segments");
            break;
        }

        let (samples, start_secs, end_secs, is_final) = match segment {
            Segment::Tick { samples, start_secs, end_secs } => {
                (samples, start_secs, end_secs, false)
            }
            Segment::Final { samples, start_secs, end_secs } => {
                (samples, start_secs, end_secs, true)
            }
        };

        if samples.is_empty() {
            if is_final {
                tracing::debug!("[{label}] worker: empty final segment, nothing to transcribe");
                break;
            }
            continue;
        }

        let prev_text = read_wal_context(&history_dir, &note_id);

        // Pass raw (unfiltered) samples to the transcribe closure so that both
        // diarization (which maps sample offsets → absolute timestamps) and STT
        // operate on the same audio, keeping timestamps consistent.
        // Trade-off: applying filter_with_vad inside the Whisper closure for STT
        // would re-introduce the timestamp mismatch (DTW word offsets would be in
        // VAD-filtered time; diarization boundaries in raw time). The segmenter's
        // coarse VAD gate ensures chunks are speech-dominant; any residual silence
        // is handled gracefully by all three STT backends.
        let wal_segments: Vec<crate::meeting_notes::WalSegment> =
            transcribe(&samples, start_secs, end_secs, &prev_text);

        // Post-transcription session check (transcription may take 10–60s).
        let current_session = state.meeting_session.load(Ordering::SeqCst);
        if current_session != session_id {
            tracing::warn!(
                "[{label}] worker: session advanced during transcription ({session_id} vs {current_session}) — discarding",
            );
            break;
        }

        if state.meeting_cancelled.load(Ordering::SeqCst) {
            tracing::warn!("[{label}] worker: cancelled after transcription — discarding");
            break;
        }

        for seg in &wal_segments {
            persist_and_emit(&app, &state, &history_dir, &note_id, seg);
        }
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
    _audio_note_id: &Option<String>,
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
    let mut total_samples_sent: usize = 0; // cumulative 16 kHz samples sent to worker
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
            let chunk = std::mem::take(&mut chunk_buf);
            let start_secs = total_samples_sent as f64 / 16_000.0;
            let end_secs = (total_samples_sent + chunk.len()) as f64 / 16_000.0;
            total_samples_sent += chunk.len();
            // If send fails (worker exited early due to session change), stop.
            if tx.send(Segment::Tick { samples: chunk, start_secs, end_secs }).is_err() {
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
    let start_secs = total_samples_sent as f64 / 16_000.0;
    let end_secs = (total_samples_sent + chunk_buf.len()) as f64 / 16_000.0;
    let _ = tx.send(Segment::Final { samples: chunk_buf, start_secs, end_secs });
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
/// `transcribe` receives raw (unfiltered) 16 kHz samples, the segment start/end times
/// in seconds (relative to meeting start), and the previous WAL context string.
/// Raw samples are passed so that diarization timestamps (sample offset ÷ 16 kHz)
/// and STT output are in the same time space; each closure may apply VAD filtering
/// internally for STT quality if desired.
/// It returns `Vec<WalSegment>` — one per diarization sub-segment (may be >1
/// when the segmentation model detects an intra-utterance speaker change).
pub(crate) fn run_meeting_feeder(
    app: AppHandle,
    session_id: u64,
    label: &str,
    max_segment_samples: Option<usize>,
    transcribe: Box<
        dyn FnMut(&[f32], f64, f64, &str) -> Vec<crate::meeting_notes::WalSegment>
            + Send
            + 'static,
    >,
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

    run_meeting_segmenter(&app, session_id, label, max_segment_samples, tx, &audio_note_id);
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

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::time::Instant;

use crate::stt::{LocalSttEngine, SttConfig, SttMode};
use crate::transcribe::transcribe_with_cached_whisper;

/// Handle to control (stop) a running audio thread.
pub struct AudioThreadControl {
    thread: std::thread::Thread,
    stop_signal: Arc<AtomicBool>,
    /// False when the cpal stream has emitted an error and is no longer delivering data.
    pub stream_alive: Arc<AtomicBool>,
    /// The resolved device name this stream was opened on (after BT-avoidance).
    /// `None` means cpal's system default was used at open time.
    pub device_name: Option<String>,
}

impl AudioThreadControl {
    /// Signal the audio thread to stop and wake it up.
    pub fn stop(&self) {
        self.stop_signal.store(true, Ordering::Relaxed);
        self.thread.unpark();
    }

    pub fn is_alive(&self) -> bool {
        self.stream_alive.load(Ordering::Relaxed)
    }
}

/// Spawn a persistent audio thread that builds and immediately starts the cpal
/// input stream.  The stream runs for the entire app lifetime — the callback
/// checks `is_recording` atomically and discards samples when false.
///
/// If `device_name` is Some, the named device is used; falls back to the
/// system default if not found.  When `device_name` is None (Auto mode),
/// `resolve_input_device` is applied first so Bluetooth inputs are avoided.
pub fn spawn_audio_thread(
    buffer: Arc<Mutex<Vec<f32>>>,
    is_recording: Arc<AtomicBool>,
    device_name: Option<String>,
) -> Result<(u32, AudioThreadControl), String> {
    // Apply Bluetooth avoidance when in Auto mode (device_name == None).
    let device_name = crate::audio_devices::resolve_input_device(device_name);
    // Clone the resolved name before it is moved into the spawned thread,
    // so we can record it on AudioThreadControl for the mismatch check.
    let resolved_device_name = device_name.clone();

    let stop = Arc::new(AtomicBool::new(false));
    let stop_for_thread = Arc::clone(&stop);

    // Shared flag: set to false by the error callback when the stream dies.
    let stream_alive = Arc::new(AtomicBool::new(true));
    let alive_for_thread = Arc::clone(&stream_alive);

    let (init_tx, init_rx) = mpsc::channel::<Result<u32, String>>();

    let buf_for_thread = Arc::clone(&buffer);
    let rec_for_thread = Arc::clone(&is_recording);

    let join_handle = std::thread::spawn(move || {
        let host = cpal::default_host();

        let device = if let Some(ref name) = device_name {
            let found = host
                .input_devices()
                .ok()
                .and_then(|mut devs| devs.find(|d| d.name().ok().as_deref() == Some(name.as_str())));
            match found {
                Some(d) => d,
                None => {
                    tracing::warn!("Device '{}' not found, falling back to default", name);
                    match host.default_input_device() {
                        Some(d) => d,
                        None => {
                            let _ = init_tx.send(Err("No microphone device found".to_string()));
                            return;
                        }
                    }
                }
            }
        } else {
            match host.default_input_device() {
                Some(d) => d,
                None => {
                    let _ = init_tx.send(Err("No microphone device found".to_string()));
                    return;
                }
            }
        };

        let config = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => {
                let _ = init_tx.send(Err(format!("Failed to get input config: {}", e)));
                return;
            }
        };

        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;

        let stream = {
            let buf = Arc::clone(&buf_for_thread);
            let rec = Arc::clone(&rec_for_thread);
            match config.sample_format() {
                cpal::SampleFormat::F32 => {
                    let rec_err = Arc::clone(&rec_for_thread);
                    let alive_err = Arc::clone(&alive_for_thread);
                    device.build_input_stream(
                        &config.into(),
                        move |data: &[f32], _: &cpal::InputCallbackInfo| {
                            if !rec.load(Ordering::Relaxed) {
                                return;
                            }
                            let mut buf = match buf.lock() {
                                Ok(b) => b,
                                Err(_) => return,
                            };
                            // S-08: safety cap — ~40 MB / 4 bytes = 10M samples
                            if buf.len() > 10_000_000 {
                                rec.store(false, Ordering::Relaxed);
                                return;
                            }
                            if channels == 1 {
                                buf.extend_from_slice(data);
                            } else {
                                for chunk in data.chunks(channels) {
                                    buf.push(chunk.iter().sum::<f32>() / channels as f32);
                                }
                            }
                        },
                        move |err| {
                            tracing::error!("audio stream error: {}", err);
                            alive_err.store(false, Ordering::Relaxed);
                            rec_err.store(false, Ordering::Relaxed);
                        },
                        None,
                    )
                }
                cpal::SampleFormat::I16 => {
                    let buf = Arc::clone(&buf_for_thread);
                    let rec = Arc::clone(&rec_for_thread);
                    let rec_err = Arc::clone(&rec_for_thread);
                    let alive_err = Arc::clone(&alive_for_thread);
                    device.build_input_stream(
                        &config.into(),
                        move |data: &[i16], _: &cpal::InputCallbackInfo| {
                            if !rec.load(Ordering::Relaxed) {
                                return;
                            }
                            let mut buf = match buf.lock() {
                                Ok(b) => b,
                                Err(_) => return,
                            };
                            // S-08: safety cap
                            if buf.len() > 2_000_000 {
                                rec.store(false, Ordering::Relaxed);
                                return;
                            }
                            if channels == 1 {
                                buf.extend(data.iter().map(|&s| s as f32 / i16::MAX as f32));
                            } else {
                                for chunk in data.chunks(channels) {
                                    buf.push(
                                        chunk
                                            .iter()
                                            .map(|&s| s as f32 / i16::MAX as f32)
                                            .sum::<f32>()
                                            / channels as f32,
                                    );
                                }
                            }
                        },
                        move |err| {
                            tracing::error!("audio stream error: {}", err);
                            alive_err.store(false, Ordering::Relaxed);
                            rec_err.store(false, Ordering::Relaxed);
                        },
                        None,
                    )
                }
                other => {
                    let _ = init_tx.send(Err(format!("Unsupported audio format: {:?}", other)));
                    return;
                }
            }
        };

        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                let _ = init_tx.send(Err(format!("Failed to build input stream: {}", e)));
                return;
            }
        };

        if let Err(e) = stream.play() {
            let _ = init_tx.send(Err(format!("Failed to start audio stream: {}", e)));
            return;
        }

        tracing::info!(
            "Audio stream always-on: {} Hz, {} ch",
            sample_rate, channels
        );
        let _ = init_tx.send(Ok(sample_rate));

        // Park the thread until signalled to stop, keeping `stream` alive.
        loop {
            if stop_for_thread.load(Ordering::Relaxed) {
                tracing::info!("Audio thread stopping");
                break;
            }
            std::thread::park();
        }
    });

    // Grab the Thread handle before blocking on the init channel.
    let thread = join_handle.thread().clone();

    let sample_rate = init_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .map_err(|_| "Audio thread init timed out".to_string())??;

    Ok((sample_rate, AudioThreadControl { thread, stop_signal: stop, stream_alive, device_name: resolved_device_name }))
}

/// Attempt to reconnect the microphone when `mic_available` is false.
pub fn try_reconnect_audio(
    mic_available: &AtomicBool,
    sample_rate: &Mutex<Option<u32>>,
    buffer: &Arc<Mutex<Vec<f32>>>,
    is_recording: &Arc<AtomicBool>,
    audio_thread: &Mutex<Option<AudioThreadControl>>,
    device_name: Option<String>,
) -> Result<(), String> {
    if mic_available.load(Ordering::SeqCst) {
        return Ok(());
    }
    let (sr, control) = spawn_audio_thread(Arc::clone(buffer), Arc::clone(is_recording), device_name)?;
    *sample_rate.lock().map_err(|e| e.to_string())? = Some(sr);
    if let Ok(mut at) = audio_thread.lock() {
        *at = Some(control);
    }
    mic_available.store(true, Ordering::SeqCst);
    tracing::info!("Microphone reconnected: {} Hz", sr);
    Ok(())
}

/// Close the audio input stream (on-demand model).
///
/// Setting `mic_available=false` causes `do_start_recording` to reopen the
/// stream via `try_reconnect_audio` on the next hotkey press.  The caller
/// shows the overlay in the `preparing` state before calling
/// `do_start_recording` so the user sees visual feedback during the ~70 ms
/// CoreAudio re-initialization.
pub fn close_audio_stream(
    audio_thread: &Mutex<Option<AudioThreadControl>>,
    mic_available: &AtomicBool,
) {
    if let Ok(mut at) = audio_thread.lock() {
        if let Some(ctrl) = at.take() {
            ctrl.stop();
        }
    }
    mic_available.store(false, Ordering::SeqCst);
    tracing::info!("Mic stream closed (on-demand)");
}

/// Start recording — opens the mic stream if not already running (on-demand model).
pub fn do_start_recording(
    is_recording: &AtomicBool,
    mic_available: &AtomicBool,
    sample_rate: &Mutex<Option<u32>>,
    buffer: &Arc<Mutex<Vec<f32>>>,
    is_recording_arc: &Arc<AtomicBool>,
    audio_thread: &Mutex<Option<AudioThreadControl>>,
    device_name: Option<String>,
) -> Result<(), String> {
    // ── Step 1: ensure stream is alive ───────────────────────────────────
    // Clone device_name here so it remains available for the mismatch check below.
    let stream_dead = audio_thread.lock().ok()
        .and_then(|at| at.as_ref().map(|c| !c.is_alive()))
        .unwrap_or(false);

    if !mic_available.load(Ordering::SeqCst) || stream_dead {
        if stream_dead {
            tracing::warn!("Audio stream was dead, reconnecting before recording");
            mic_available.store(false, Ordering::SeqCst);
        }
        try_reconnect_audio(mic_available, sample_rate, buffer, is_recording_arc, audio_thread, device_name.clone())?;
        // Freshly reconnected stream is already on the correct device;
        // the mismatch check below will be a no-op.
    }

    // ── Step 2: device-mismatch guard (BT hotplug race defence) ─────────
    // resolve_input_device reflects the *current* system state.  If BT
    // headphones connected while the stream was open as "system default"
    // (device_name = None → recorded as None on AudioThreadControl), the
    // wanted device is now Some("MacBook Pro Microphone") — a mismatch.
    // Reconnect *before* setting is_recording so the race window is zero.
    let wanted = crate::audio_devices::resolve_input_device(device_name.clone());
    if wanted.is_some() {
        let current = audio_thread.lock().ok()
            .and_then(|g| g.as_ref().map(|c| c.device_name.clone()))
            .flatten();
        if current.as_deref() != wanted.as_deref() {
            tracing::info!(
                "Stream device mismatch (stream={:?}, wanted={:?}) — reconnecting before recording",
                current, wanted
            );
            {
                let mut at = audio_thread.lock().map_err(|e| e.to_string())?;
                if let Some(ctrl) = at.take() { ctrl.stop(); }
            }
            mic_available.store(false, Ordering::SeqCst);
            try_reconnect_audio(mic_available, sample_rate, buffer, is_recording_arc, audio_thread, device_name)?;
        }
    }

    // ── Step 3: flip the recording flag ──────────────────────────────────
    if is_recording.load(Ordering::SeqCst) {
        return Err("Already recording".to_string());
    }

    {
        let mut buf = buffer.lock().map_err(|e| e.to_string())?;
        buf.clear();
    }

    is_recording.store(true, Ordering::SeqCst);

    Ok(())
}

/// Stop recording, transcribe, and return the text + 16 kHz samples for history.
pub fn do_stop_recording(
    state: &crate::AppState,
    stt_config: &SttConfig,
    language: &str,
    dictionary_terms: &[String],
) -> Result<(String, Vec<f32>), String> {
    let sample_rate = state.sample_rate
        .lock()
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "No microphone available".to_string())?;

    if state.is_recording
        .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("Not currently recording".to_string());
    }
    // Wake the streaming feeder immediately so it exits its 2 s sleep and
    // starts post-loop work (trailing feed + finish_streaming) right away,
    // reducing transcription latency by up to 2 s on short recordings.
    state.feeder_stop_cv.notify_all();

    // ── Wait for feeders BEFORE taking the buffer ────────────────────────
    // The Qwen3-ASR feeder reads trailing audio from the shared buffer in
    // its post-loop code. If we `take` the buffer first, the feeder sees
    // an empty buffer and the last 0–2 s of audio is silently dropped.
    // Waiting here ensures the feeder finishes its trailing feed before we
    // drain the buffer for the batch fallback / history audio.
    let mut qwen3_streaming_result: Option<String> = None;
    if stt_config.mode == SttMode::Local && stt_config.local_engine == LocalSttEngine::Qwen3Asr
        && state.streaming_active.load(Ordering::SeqCst)
    {
        let deadline = Instant::now() + std::time::Duration::from_millis(3000);
        while state.streaming_active.load(Ordering::SeqCst) {
            if Instant::now() >= deadline {
                tracing::warn!("[streaming] feeder timeout — signalling cancel, falling back to batch");
                state.streaming_cancelled.store(true, Ordering::SeqCst);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        // Grab the streaming result now (feeder stored it before clearing streaming_active).
        if let Ok(mut guard) = state.streaming_result.lock() {
            qwen3_streaming_result = guard.take();
        }
    }
    if stt_config.mode == SttMode::Local && stt_config.local_engine == LocalSttEngine::Whisper
        && state.whisper_preview_active.load(Ordering::SeqCst)
    {
        let deadline = Instant::now() + std::time::Duration::from_millis(2000);
        while state.whisper_preview_active.load(Ordering::SeqCst) && Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        if state.whisper_preview_active.load(Ordering::SeqCst) {
            tracing::warn!("[whisper-preview] feeder still active after 2s wait — proceeding anyway");
        }
    }

    let samples: Vec<f32> = {
        let mut buf = state.buffer.lock().map_err(|e| e.to_string())?;
        std::mem::take(&mut *buf)
    };

    if samples.is_empty() {
        return Err("No audio captured".to_string());
    }

    tracing::info!(
        "[timing] recording: {:.2}s ({} samples @ {} Hz)",
        samples.len() as f64 / sample_rate as f64,
        samples.len(),
        sample_rate,
    );

    let t0 = Instant::now();
    let mut samples_16k = if sample_rate != 16000 {
        let resampled = resample(&samples, sample_rate, 16000);
        tracing::info!("[timing] resample {} Hz → 16 kHz: {:.0?}", sample_rate, t0.elapsed());
        resampled
    } else {
        samples
    };

    // ── VAD or RMS trimming ─────────────────────────────────────────────
    let vad_enabled = stt_config.vad_enabled;
    let vad_model_exists = crate::transcribe::vad_model_path().exists();
    if vad_enabled && vad_model_exists {
        // Use Silero VAD to extract speech segments
        match crate::transcribe::filter_with_vad(&state.vad_ctx, &samples_16k) {
            Ok(speech) if speech.is_empty() => {
                tracing::info!("VAD: no speech segments found");
                return Err("no_speech".to_string());
            }
            Ok(speech) => {
                tracing::info!(
                    "VAD filtered: {:.2}s → {:.2}s",
                    samples_16k.len() as f64 / 16000.0,
                    speech.len() as f64 / 16000.0,
                );
                samples_16k = speech;
            }
            Err(e) => {
                tracing::warn!("VAD failed ({}), falling back to RMS trimming", e);
                rms_trim_silence(&mut samples_16k)?;
            }
        }
    } else {
        if vad_enabled && !vad_model_exists {
            tracing::warn!("VAD enabled but model not downloaded, using RMS trimming");
        }
        rms_trim_silence(&mut samples_16k)?;
    }

    let stt_start = Instant::now();
    let text = match stt_config.mode {
        SttMode::Local => match stt_config.local_engine {
            LocalSttEngine::Whisper => {
                let result = transcribe_with_cached_whisper(&state.whisper_ctx, &samples_16k, &stt_config.whisper_model, language, dictionary_terms)?;
                tracing::info!("[timing] STT (local whisper): {:.0?}", stt_start.elapsed());
                result
            }
            LocalSttEngine::Qwen3Asr => {
                // Use the streaming result if the feeder finished in time.
                if let Some(text) = qwen3_streaming_result {
                    tracing::info!("[timing] STT (local qwen3-asr streaming): {:.0?}", stt_start.elapsed());
                    return if text.is_empty() {
                        Err("no_speech".to_string())
                    } else {
                        Ok((text, samples_16k))
                    };
                }

                // Batch fallback. The feeder has been signalled to cancel so it
                // will skip post-loop engine work once it wakes. It may still
                // be holding qwen3_asr_ctx mid-inference; this call will block
                // briefly until the feeder releases the lock.
                tracing::info!("[streaming] using batch fallback");
                let result = crate::qwen3_asr::transcribe_with_cached_qwen3_asr(
                    &state.qwen3_asr_ctx,
                    &samples_16k,
                    &stt_config.qwen3_asr_model,
                    language,
                )?;
                tracing::info!("[timing] STT (local qwen3-asr batch): {:.0?}", stt_start.elapsed());
                result
            }
        },
        SttMode::Cloud => {
            let result = crate::stt::run_cloud_stt(&stt_config.cloud, &samples_16k, &state.http_client, None)?;
            tracing::info!("[timing] STT (cloud {}): {:.0?}", stt_config.cloud.provider.as_key(), stt_start.elapsed());
            result
        }
    };

    if text.is_empty() {
        Err("no_speech".to_string())
    } else {
        Ok((text, samples_16k))
    }
}

/// RMS (root mean square) energy of an audio slice.
#[inline]
pub(crate) fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Strip leading/trailing silence using RMS, and reject near-silent audio.
fn rms_trim_silence(samples_16k: &mut Vec<f32>) -> Result<(), String> {
    const SILENCE_RMS_THRESHOLD: f32 = 0.01;
    const WINDOW: usize = 160;
    const LOOKBACK: usize = 1600;

    let speech_onset = samples_16k
        .windows(WINDOW)
        .position(|w| rms(w) > SILENCE_RMS_THRESHOLD)
        .unwrap_or(0);

    let trim_start = speech_onset.saturating_sub(LOOKBACK);
    if trim_start > 0 {
        tracing::info!(
            "Trimmed {:.0} ms of leading silence (onset at {:.0} ms)",
            trim_start as f64 / 16.0,
            speech_onset as f64 / 16.0
        );
        samples_16k.drain(0..trim_start);
    }

    if samples_16k.len() > WINDOW {
        let total = samples_16k.len();
        let last_speech = samples_16k
            .windows(WINDOW)
            .rposition(|w| rms(w) > SILENCE_RMS_THRESHOLD)
            .map(|pos| pos + WINDOW)
            .unwrap_or(total);

        let trim_end = (last_speech + LOOKBACK).min(total);
        if trim_end < total {
            tracing::info!(
                "Trimmed {:.0} ms of trailing silence",
                (total - trim_end) as f64 / 16.0
            );
            samples_16k.truncate(trim_end);
        }
    }

    // Pre-check: if the entire audio is near-silent, skip Whisper entirely
    let overall_rms = rms(samples_16k);
    if overall_rms < 0.005 {
        tracing::info!("Audio RMS {:.5} below threshold — no speech detected", overall_rms);
        return Err("no_speech".to_string());
    }
    tracing::info!("Audio RMS: {:.5}", overall_rms);

    Ok(())
}

/// Simple linear interpolation resampler.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;
        let sample = if idx + 1 < samples.len() {
            samples[idx] as f64 * (1.0 - frac) + samples[idx + 1] as f64 * frac
        } else {
            samples[idx.min(samples.len() - 1)] as f64
        };
        output.push(sample as f32);
    }
    output
}

//! Audio file import for Meeting Notes.
//!
//! Decodes audio files (WAV, MP3, M4A, OGG, FLAC) using symphonia,
//! resamples to 16 kHz mono, and runs chunked STT to produce a
//! meeting note transcript.

use std::path::Path;
use std::sync::atomic::Ordering;

use tauri::{AppHandle, Emitter, Manager};

use crate::meeting_notes;
use crate::segment_spacing::SpacingState;
use crate::settings;

/// Decode an audio file to mono f32 samples.
/// Returns (samples, sample_rate, duration_secs).
fn decode_audio_file(path: &str) -> Result<(Vec<f32>, u32, f64), String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let path = Path::new(path);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "wav" | "mp3" | "m4a" | "aac" | "ogg" | "flac" => {}
        "mp4" | "mov" | "mkv" | "avi" | "webm" => {
            return Err("video_not_supported".to_string());
        }
        _ => {
            return Err(format!("unsupported_format:{ext}"));
        }
    }

    let file =
        std::fs::File::open(path).map_err(|e| format!("Failed to open file: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension(&ext);

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("Failed to probe audio format: {e}"))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or("No audio track found in file")?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or("Could not determine sample rate")?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Failed to create decoder: {e}"))?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(symphonia::core::errors::Error::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => return Err(format!("Failed to read packet: {e}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let num_frames = decoded.frames();
                // Read channel count from the decoded buffer spec, NOT
                // track metadata. M4A files often omit channel info in
                // codec params, causing unwrap_or(1) to treat stereo
                // interleaved [L,R,L,R,...] data as mono → garbled audio.
                let ch = spec.channels.count();
                let mut sample_buf =
                    SampleBuffer::<f32>::new(num_frames as u64, spec);
                sample_buf.copy_interleaved_ref(decoded);

                let samples = sample_buf.samples();
                if ch <= 1 {
                    all_samples.extend_from_slice(samples);
                } else {
                    for chunk in samples.chunks(ch) {
                        all_samples
                            .push(chunk.iter().sum::<f32>() / ch as f32);
                    }
                }
            }
            Err(symphonia::core::errors::Error::DecodeError(e)) => {
                tracing::warn!("[import] decode error (skipping packet): {e}");
            }
            Err(e) => return Err(format!("Failed to decode audio: {e}")),
        }
    }

    if all_samples.is_empty() {
        return Err("No audio data found in file".to_string());
    }

    let duration_secs = all_samples.len() as f64 / sample_rate as f64;
    Ok((all_samples, sample_rate, duration_secs))
}

/// Run the import pipeline: decode → resample → chunk → STT → WAL → finalize.
///
/// Called on a blocking thread from the Tauri command.
pub fn run_import(app: &AppHandle, file_path: &str) -> Result<String, String> {
    let state = app.state::<crate::AppState>();

    // Prevent concurrent imports.
    if state
        .import_active
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("import_already_active".to_string());
    }
    state.import_cancelled.store(false, Ordering::SeqCst);

    let result = run_import_inner(app, file_path);

    state.import_active.store(false, Ordering::SeqCst);
    result
}

fn run_import_inner(app: &AppHandle, file_path: &str) -> Result<String, String> {
    let state = app.state::<crate::AppState>();

    // ── Step 1: Decode audio file ──
    tracing::info!("[import] decoding: {}", file_path);
    let _ = app.emit(
        "import-progress",
        serde_json::json!({ "status": "decoding", "progress": 0.0 }),
    );

    let (samples, sample_rate, duration_secs) = decode_audio_file(file_path)?;
    let decoded_rms = crate::audio::rms(&samples);
    tracing::info!(
        "[import] decoded: {:.1}s @ {} Hz, {} samples (mono), RMS={:.5}",
        duration_secs,
        sample_rate,
        samples.len(),
        decoded_rms
    );

    // ── Step 2: Resample to 16 kHz ──
    let samples_16k = if sample_rate != 16000 {
        crate::audio::resample(&samples, sample_rate, 16000)
    } else {
        samples
    };

    let total_samples = samples_16k.len();
    if total_samples == 0 {
        return Err("No audio data after resampling".to_string());
    }

    // ── Step 3: Determine STT engine ──
    let stt_config = state
        .settings
        .lock()
        .map_err(|e| e.to_string())?
        .stt
        .clone();
    let language = stt_config.language.clone();

    let model_label = match stt_config.mode {
        crate::stt::SttMode::Cloud => {
            format!(
                "Cloud (Import) – {}",
                stt_config.cloud.provider.as_key()
            )
        }
        crate::stt::SttMode::Local => match stt_config.local_engine {
            crate::stt::LocalSttEngine::Qwen3Asr => {
                format!(
                    "Qwen3-ASR (Import) – {}",
                    stt_config.qwen3_asr_model.display_name()
                )
            }
            crate::stt::LocalSttEngine::Whisper => {
                format!(
                    "Whisper (Import) – {}",
                    stt_config.whisper_model.display_name()
                )
            }
        },
    };

    // ── Step 4: Create meeting note ──
    let note_id = crate::history::generate_id();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    let title = Path::new(file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Imported Audio")
        .to_string();

    let note = meeting_notes::MeetingNote {
        id: note_id.clone(),
        title,
        transcript: String::new(),
        created_at: now,
        updated_at: now,
        duration_secs,
        stt_model: model_label,
        is_recording: true, // Use WAL-based flow
        word_count: 0,
        summary: String::new(),
        audio_path: None,
    };

    let history_dir = settings::history_dir();
    meeting_notes::create_note(&history_dir, &note)
        .map_err(|e| format!("Failed to create note: {e}"))?;

    let _ = app.emit(
        "meeting-note-created",
        serde_json::json!({ "id": note_id, "note": note }),
    );

    // ── Step 5: Build transcribe closure ──
    let mut transcribe: Box<dyn FnMut(&[f32], &str) -> String + Send> =
        match stt_config.mode {
            crate::stt::SttMode::Local => match stt_config.local_engine {
                crate::stt::LocalSttEngine::Qwen3Asr => {
                    let model = stt_config.qwen3_asr_model.clone();
                    let lang = language.clone();
                    let app_c = app.clone();
                    Box::new(move |samples, _prev| {
                        let st = app_c.state::<crate::AppState>();
                        crate::qwen3_asr::transcribe_with_cached_qwen3_asr(
                            &st.qwen3_asr_ctx,
                            samples,
                            &model,
                            &lang,
                        )
                        .unwrap_or_default()
                    })
                }
                crate::stt::LocalSttEngine::Whisper => {
                    let lang = language.clone();
                    let app_c = app.clone();
                    Box::new(move |samples, prev_text| {
                        let st = app_c.state::<crate::AppState>();
                        let ctx_guard = st
                            .whisper_ctx
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        crate::whisper_streaming::transcribe_meeting_chunk(
                            &ctx_guard,
                            samples,
                            &lang,
                            if prev_text.is_empty() {
                                None
                            } else {
                                Some(prev_text)
                            },
                        )
                        .unwrap_or_default()
                    })
                }
            },
            crate::stt::SttMode::Cloud => {
                let mut cloud = stt_config.cloud.clone();
                cloud.language = language;
                let key = crate::commands::get_cached_api_key(
                    &state.api_key_cache,
                    cloud.provider.as_key(),
                );
                if !key.is_empty() {
                    cloud.api_key = key;
                }
                let app_c = app.clone();
                Box::new(move |samples, prev_text| {
                    let st = app_c.state::<crate::AppState>();
                    let prompt = if prev_text.is_empty() {
                        None
                    } else {
                        Some(prev_text)
                    };
                    crate::stt::run_cloud_stt(
                        &cloud,
                        samples,
                        &st.http_client,
                        prompt,
                    )
                    .unwrap_or_else(|e| {
                        tracing::warn!("[import] cloud STT failed: {e}");
                        String::new()
                    })
                })
            }
        };

    // ── Step 6: Chunk and transcribe ──
    let chunk_size = 30 * 16_000; // 30 seconds per chunk
    let mut spacing = SpacingState::new();

    for chunk_start in (0..total_samples).step_by(chunk_size) {
        if state.import_cancelled.load(Ordering::SeqCst) {
            tracing::info!("[import] cancelled by user");
            let _ = meeting_notes::delete_note(&history_dir, &note_id);
            let _ = app.emit(
                "meeting-note-finalized",
                serde_json::json!({ "id": note_id }),
            );
            return Err("cancelled".to_string());
        }

        let chunk_end = (chunk_start + chunk_size).min(total_samples);
        let chunk = &samples_16k[chunk_start..chunk_end];

        // VAD filter
        let stt_samples =
            crate::transcribe::filter_with_vad(&state.vad_ctx, chunk)
                .unwrap_or_else(|_| chunk.to_vec());

        // Read previous context from WAL
        let prev_text = {
            let full = meeting_notes::read_wal(&history_dir, &note_id);
            let trimmed = full.trim_end();
            let cc = trimmed.chars().count();
            if cc > 200 {
                trimmed
                    .char_indices()
                    .nth(cc - 200)
                    .map(|(i, _)| &trimmed[i..])
                    .unwrap_or(trimmed)
                    .to_string()
            } else {
                trimmed.to_string()
            }
        };

        let seg_text = if stt_samples.is_empty() {
            String::new()
        } else {
            transcribe(&stt_samples, &prev_text)
        };

        let is_last = chunk_end >= total_samples;
        let delta = if is_last {
            spacing.build_final_delta(&seg_text)
        } else {
            spacing.build_tick_delta(&seg_text)
        };

        if !delta.is_empty() {
            meeting_notes::append_wal(&history_dir, &note_id, &delta);
            let _ = app.emit(
                "meeting-note-updated",
                serde_json::json!({
                    "id": note_id,
                    "delta": delta,
                    "duration_secs": duration_secs,
                }),
            );
        }

        let progress = chunk_end as f64 / total_samples as f64;
        let _ = app.emit(
            "import-progress",
            serde_json::json!({
                "id": note_id,
                "progress": progress,
                "status": "transcribing",
            }),
        );
    }

    // ── Step 7: Finalize ──
    let transcript = meeting_notes::read_wal(&history_dir, &note_id);
    meeting_notes::finalize_note(&history_dir, &note_id, &transcript, duration_secs)
        .map_err(|e| format!("Failed to finalize note: {e}"))?;
    meeting_notes::remove_wal(&history_dir, &note_id);

    let _ = app.emit(
        "meeting-note-finalized",
        serde_json::json!({ "id": note_id }),
    );
    let _ = app.emit(
        "import-progress",
        serde_json::json!({
            "id": note_id,
            "progress": 1.0,
            "status": "complete",
        }),
    );

    tracing::info!(
        "[import] complete: {} chars, {:.1}s audio",
        transcript.len(),
        duration_secs
    );
    Ok(note_id)
}

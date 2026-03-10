use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;
use whisper_rs::{DtwMode, DtwModelPreset, DtwParameters, WhisperContext, WhisperContextParameters, WhisperVadContext, WhisperVadContextParams, WhisperVadParams};

use crate::settings::models_dir;
use crate::whisper_models::WhisperModel;

/// Cross-attention cache for Whisper DTW word-timestamp alignment.
/// 128 MiB is sufficient for LargeV3Turbo; allocated once per context.
const DTW_MEM_SIZE: usize = 128 * 1024 * 1024;

/// Cached whisper context that tracks which model file is loaded.
/// When the requested model path differs from the loaded one, the context
/// is automatically reloaded.
pub struct WhisperContextCache {
    pub ctx: WhisperContext,
    pub loaded_path: PathBuf,
}

// WhisperContext is Send but not Sync by default; we guard it with a Mutex.
unsafe impl Send for WhisperContextCache {}

/// Cached Silero VAD context, lazily initialised on first use.
pub struct VadContextCache {
    pub ctx: WhisperVadContext,
    pub model_path: PathBuf,
}

// WhisperVadContext is !Send (raw pointer); safe because we guard with Mutex
// and is_processing AtomicBool ensures single-threaded access.
unsafe impl Send for VadContextCache {}

/// Filter audio samples through Silero VAD, returning only speech segments.
/// The VAD context is lazily loaded on first call.
pub fn filter_with_vad(
    vad_cache: &Mutex<Option<VadContextCache>>,
    samples_16k: &[f32],
) -> Result<Vec<f32>, String> {
    let model_path = crate::settings::vad_model_path();
    if !model_path.exists() {
        return Err("VAD model not downloaded".to_string());
    }

    let mut cache_guard = vad_cache
        .lock()
        .map_err(|e| format!("Failed to lock VAD context: {}", e))?;

    // Lazy-init or reload if model path changed
    let needs_reload = match cache_guard.as_ref() {
        Some(c) => c.model_path != model_path,
        None => true,
    };

    if needs_reload {
        let load_start = Instant::now();
        tracing::info!("Loading Silero VAD model...");
        let mut ctx_params = WhisperVadContextParams::new();
        ctx_params.set_use_gpu(cfg!(target_os = "macos"));
        // Silero VAD processes 512-sample chunks sequentially (LSTM is not
        // parallelisable). Using multiple threads only adds barrier-sync
        // overhead on every chunk — set to 1 for best throughput.
        ctx_params.set_n_threads(1);
        let ctx = WhisperVadContext::new(
            model_path.to_str().ok_or("Invalid VAD model path")?,
            ctx_params,
        )
        .map_err(|e| format!("Failed to load VAD model: {:?}", e))?;

        *cache_guard = Some(VadContextCache {
            ctx,
            model_path: model_path.clone(),
        });
        tracing::info!(
            "Silero VAD model loaded (took {:.0?})",
            load_start.elapsed()
        );
    }

    let cache = cache_guard.as_mut().expect("VAD context was just initialized above");

    let vad_start = Instant::now();
    // Use generous VAD params (matching faster-whisper defaults) so that
    // natural pauses are preserved in the output audio.  Whisper relies on
    // silence cues to infer punctuation — aggressive trimming strips them.
    let mut params = WhisperVadParams::default();
    params.set_speech_pad(400);            // 400 ms real-audio padding around each segment
    params.set_min_silence_duration(2000); // only split on silences > 2 s
    let segments = cache
        .ctx
        .segments_from_samples(params, samples_16k)
        .map_err(|e| format!("VAD segmentation failed: {:?}", e))?;

    let n = segments.num_segments();
    tracing::info!("VAD found {} speech segment(s) (took {:.0?})", n, vad_start.elapsed());

    let mut speech_samples = Vec::new();
    for seg in segments {
        // Timestamps are in centiseconds (1cs = 10ms)
        let start_sample = ((seg.start / 100.0) * 16000.0) as usize;
        let end_sample = (((seg.end / 100.0) * 16000.0) as usize).min(samples_16k.len());
        if start_sample < end_sample {
            tracing::info!(
                "  segment: {:.2}s – {:.2}s ({} samples)",
                seg.start / 100.0,
                seg.end / 100.0,
                end_sample - start_sample,
            );
            speech_samples.extend_from_slice(&samples_16k[start_sample..end_sample]);
        }
    }

    Ok(speech_samples)
}

/// Check whether a 16 kHz audio chunk contains speech according to Silero VAD.
///
/// Returns `true` if at least one speech segment is detected.
/// Falls back to RMS-based detection (`rms >= threshold`) if the VAD model is
/// unavailable or cannot be loaded.
pub fn has_speech_vad(
    vad_cache: &Mutex<Option<VadContextCache>>,
    samples_16k: &[f32],
    rms_fallback_threshold: f32,
) -> bool {
    let model_path = crate::settings::vad_model_path();
    if !model_path.exists() {
        return crate::audio::rms(samples_16k) >= rms_fallback_threshold;
    }

    let mut cache_guard = match vad_cache.lock() {
        Ok(g) => g,
        Err(_) => return crate::audio::rms(samples_16k) >= rms_fallback_threshold,
    };

    // Lazy-init or reload if model path changed.
    let needs_reload = match cache_guard.as_ref() {
        Some(c) => c.model_path != model_path,
        None => true,
    };
    if needs_reload {
        let mut ctx_params = WhisperVadContextParams::new();
        ctx_params.set_use_gpu(cfg!(target_os = "macos"));
        // Silero VAD processes 512-sample chunks sequentially (LSTM is not
        // parallelisable). Using multiple threads only adds barrier-sync
        // overhead on every chunk — set to 1 for best throughput.
        ctx_params.set_n_threads(1);
        match WhisperVadContext::new(
            model_path.to_str().unwrap_or(""),
            ctx_params,
        ) {
            Ok(ctx) => {
                *cache_guard = Some(VadContextCache {
                    ctx,
                    model_path: model_path.clone(),
                });
            }
            Err(e) => {
                tracing::warn!("Failed to load VAD for speech detection: {:?}", e);
                return crate::audio::rms(samples_16k) >= rms_fallback_threshold;
            }
        }
    }

    let cache = cache_guard.as_mut().expect("VAD context was just initialized");
    // Use tight params for short-chunk speech detection.
    let mut params = WhisperVadParams::default();
    params.set_speech_pad(100);            // 100 ms padding
    params.set_min_silence_duration(500);  // 500 ms min silence
    match cache.ctx.segments_from_samples(params, samples_16k) {
        Ok(segments) => segments.num_segments() > 0,
        Err(e) => {
            tracing::warn!("VAD speech detection failed: {:?}", e);
            crate::audio::rms(samples_16k) >= rms_fallback_threshold
        }
    }
}

/// Resolve the path to a whisper GGML model file.
/// Returns an error if the model hasn't been downloaded yet.
pub fn whisper_model_path_for(model: &WhisperModel) -> Result<PathBuf, String> {
    let model_path = models_dir().join(model.filename());
    if model_path.exists() {
        Ok(model_path)
    } else {
        Err(format!(
            "Whisper model '{}' not downloaded. Please download it from Settings.",
            model.display_name()
        ))
    }
}

/// Pre-warm the Whisper context cache by loading the given model.
/// Called after model switch so the first transcription doesn't pay the load cost.
pub fn warm_whisper_cache(
    whisper_cache: &Mutex<Option<WhisperContextCache>>,
    model: &WhisperModel,
) -> Result<(), String> {
    let model_path = whisper_model_path_for(model)?;

    // Recover from a poisoned mutex (caused by a panic in a prior warm/transcribe call).
    let mut cache_guard = whisper_cache
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    // Already loaded with the right model? No-op.
    if let Some(ref c) = *cache_guard {
        if c.loaded_path == model_path {
            return Ok(());
        }
    }

    let load_start = Instant::now();
    tracing::info!("Pre-warming Whisper model: {} ...", model.display_name());

    let mut ctx_params = WhisperContextParameters::new();
    ctx_params.use_gpu(true);
    // DTW is always enabled so the loaded context is ready for meeting-mode word
    // timestamps without a model reload (a reload costs 5–10 s on first use).
    // Cost: ~128 MiB of Metal/GPU memory per loaded model, even for non-meeting sessions.
    ctx_params.dtw_parameters(DtwParameters {
        mode: dtw_mode_for(model),
        dtw_mem_size: DTW_MEM_SIZE,
    });
    let ctx = WhisperContext::new_with_params(
        model_path.to_str().ok_or("Invalid model path")?,
        ctx_params,
    )
    .map_err(|e| format!("Failed to load whisper model: {}", e))?;

    *cache_guard = Some(WhisperContextCache {
        ctx,
        loaded_path: model_path,
    });
    tracing::info!(
        "Whisper model pre-warmed with GPU enabled (took {:.0?})",
        load_start.elapsed()
    );

    Ok(())
}

/// Transcribe 16 kHz mono f32 samples using the cached WhisperContext.
/// The context is lazily loaded on first use, and automatically reloaded
/// when the requested model differs from the currently loaded one.
pub fn transcribe_with_cached_whisper(
    whisper_cache: &Mutex<Option<WhisperContextCache>>,
    samples_16k: &[f32],
    model: &WhisperModel,
    language: &str,
    dictionary_terms: &[String],
) -> Result<String, String> {
    use whisper_rs::{FullParams, SamplingStrategy};

    let model_path = whisper_model_path_for(model)?;

    // Recover from a poisoned mutex (caused by a panic in a prior warm/transcribe call).
    let mut cache_guard = whisper_cache
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    // Check if we need to (re)load the model
    let needs_reload = match cache_guard.as_ref() {
        Some(c) => c.loaded_path != model_path,
        None => true,
    };

    if needs_reload {
        let load_start = Instant::now();
        tracing::info!(
            "Loading Whisper model: {} ...",
            model.display_name()
        );
        let mut ctx_params = WhisperContextParameters::new();
        ctx_params.use_gpu(true);
        ctx_params.dtw_parameters(DtwParameters {
            mode: dtw_mode_for(model),
            dtw_mem_size: DTW_MEM_SIZE,
        });
        let ctx = WhisperContext::new_with_params(
            model_path.to_str().ok_or("Invalid model path")?,
            ctx_params,
        )
        .map_err(|e| format!("Failed to load whisper model: {}", e))?;

        *cache_guard = Some(WhisperContextCache {
            ctx,
            loaded_path: model_path.clone(),
        });
        tracing::info!(
            "Whisper model loaded with GPU enabled (took {:.0?})",
            load_start.elapsed()
        );
    }

    let cache = cache_guard.as_ref().expect("Whisper context was just initialized above");

    let state_start = Instant::now();
    let mut wh_state = cache
        .ctx
        .create_state()
        .map_err(|e| format!("Failed to create whisper state: {}", e))?;
    tracing::info!(
        "Whisper state created: {:.0?}",
        state_start.elapsed()
    );

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    // Set language hint from STT config (BCP-47 → ISO 639-1 base code)
    // "auto" or empty means let Whisper auto-detect.
    let lang_hint = if language.is_empty() || language == "auto" {
        None
    } else {
        Some(language.split('-').next().unwrap_or(language))
    };
    params.set_language(lang_hint);

    // Build initial prompt for Whisper token biasing.
    //
    // Whisper treats initial_prompt as "previous transcription output", NOT as
    // instructions.  We use it for two purposes:
    //
    // 1. Script anchor — a short phrase in the target script so Whisper is
    //    biased toward the correct writing system (critical for CJK where
    //    set_language alone is insufficient to distinguish e.g. 繁體 vs 简体).
    //    When language is "auto", omit the anchor to let Whisper decide freely.
    //
    // 2. Dictionary terms — proper nouns the user wants recognized correctly.
    //    Placed at the tail where token bias is strongest.
    let mut prompt_parts: Vec<String> = Vec::new();

    // Script anchor: a short target-script phrase for CJK languages.
    // Only needed where multiple scripts share characters (CJK);
    // other non-Latin scripts (Arabic, Thai, Cyrillic…) are unambiguous
    // and set_language() alone is sufficient.
    match language {
        "zh-TW" | "zh" | "yue" => prompt_parts.push("繁體中文語音轉錄。".to_string()),
        "zh-CN"                 => prompt_parts.push("简体中文语音转录。".to_string()),
        "ja"                    => prompt_parts.push("日本語の音声書き起こし。".to_string()),
        "ko"                    => prompt_parts.push("한국어 음성 전사.".to_string()),
        _                       => {}
    }

    // Dictionary terms at the tail (strongest bias position).
    // Budget: ~350 chars total; reserve what's already used.
    if !dictionary_terms.is_empty() {
        let used_chars: usize = prompt_parts.iter().map(|p| p.len()).sum();
        let budget = 350usize.saturating_sub(used_chars);
        let terms_str = dictionary_terms.join(", ");
        if terms_str.len() <= budget {
            prompt_parts.push(terms_str);
        } else {
            let mut remaining = budget;
            let mut picked: Vec<&str> = Vec::new();
            for term in dictionary_terms {
                let cost = term.len() + if picked.is_empty() { 0 } else { 2 };
                if cost > remaining {
                    break;
                }
                picked.push(term);
                remaining -= cost;
            }
            if !picked.is_empty() {
                prompt_parts.push(picked.join(", "));
            }
        }
    }

    let prompt = prompt_parts.join(" ");
    if !prompt.is_empty() {
        params.set_initial_prompt(&prompt);
    }

    tracing::info!(
        "[whisper] language={:?} (config: {:?}), prompt={:?}",
        lang_hint, language, prompt
    );

    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_progress(false);
    params.set_single_segment(true);
    params.set_no_timestamps(true);
    params.set_no_context(true);
    // Re-enable whisper.cpp quality fallback: compression-ratio, logprob, and
    // no-speech checks can trigger ONE retry at temperature 0.6.  Without this,
    // all quality gates are bypassed and hallucinations on silence pass through.
    params.set_temperature_inc(0.6);
    params.set_no_speech_thold(0.5);
    params.set_n_threads(num_cpus() as _);

    let infer_start = Instant::now();
    wh_state
        .full(params, samples_16k)
        .map_err(|e| format!("Whisper inference failed: {}", e))?;
    tracing::info!(
        "Whisper wh_state.full() done: {:.0?}",
        infer_start.elapsed()
    );

    let num_segments = wh_state.full_n_segments();

    let mut text = String::new();
    for i in 0..num_segments {
        if let Some(seg) = wh_state.get_segment(i) {
            let no_speech_prob = seg.no_speech_probability();
            tracing::info!(
                "segment {} no_speech_prob={:.3}",
                i, no_speech_prob
            );
            if no_speech_prob > 0.5 {
                tracing::info!(
                    "Skipping segment {} (no_speech_prob={:.3} > 0.5)",
                    i, no_speech_prob
                );
                continue;
            }
            if let Ok(s) = seg.to_str_lossy() {
                text.push_str(&s);
            }
        }
    }

    Ok(text.trim().to_string())
}

/// Return the number of available CPU cores.
pub fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Map a WhisperModel to its DTW cross-attention preset.
/// Fine-tuned variants of LargeV3Turbo share the same architecture and use the same preset.
fn dtw_mode_for(model: &WhisperModel) -> DtwMode<'static> {
    match model {
        WhisperModel::LargeV3Turbo
        | WhisperModel::LargeV3TurboQ5
        | WhisperModel::LargeV3TurboZhTw => DtwMode::ModelPreset {
            model_preset: DtwModelPreset::LargeV3Turbo,
        },
        WhisperModel::Medium => DtwMode::ModelPreset {
            model_preset: DtwModelPreset::Medium,
        },
        WhisperModel::Small => DtwMode::ModelPreset {
            model_preset: DtwModelPreset::Small,
        },
        WhisperModel::Base => DtwMode::ModelPreset {
            model_preset: DtwModelPreset::Base,
        },
    }
}

/// Extract token-level timestamps from DTW alignment after Whisper inference.
///
/// `audio_start_secs` is the start time of this audio chunk relative to the meeting start.
///
/// Each Whisper segment's text (correctly decoded via the segment-level API) is
/// proportionally sliced across its valid-DTW tokens so that each token gets a
/// character substring sized proportionally to `1/n_valid_tokens`.  This approach:
///
/// * Preserves correct text even for rare CJK characters encoded as byte-level BPE
///   tokens (individual token strings from whisper-rs may be garbled for those).
/// * Provides token-level timestamps so that diarization can assign sub-slices of
///   text to the correct speaker sub-segment.
/// * Works for both space-separated scripts (English) and CJK scripts (Chinese,
///   Japanese, Korean) — no special-casing required.
///
/// Special tokens (`t_dtw == -1`) are skipped.
pub fn extract_dtw_words(
    wh_state: &whisper_rs::WhisperState,
    audio_start_secs: f64,
) -> Vec<crate::meeting_notes::WordTs> {
    use crate::meeting_notes::WordTs;

    let mut words = Vec::new();
    let num_segments = wh_state.full_n_segments();

    for seg_idx in 0..num_segments {
        let Some(seg) = wh_state.get_segment(seg_idx) else {
            continue;
        };

        // Collect DTW timestamps for all valid tokens (t_dtw >= 0).
        let n_tokens = seg.n_tokens();
        let valid_ts: Vec<i64> = (0..n_tokens)
            .filter_map(|i| {
                let tok = seg.get_token(i)?;
                let td = tok.token_data();
                if td.t_dtw < 0 { None } else { Some(td.t_dtw) }
            })
            .collect();

        if valid_ts.is_empty() {
            continue;
        }

        // Use the segment-level text (correctly decoded, byte-BPE tokens reassembled).
        let seg_text = match seg.to_str_lossy() {
            Ok(s) => s.into_owned(),
            Err(_) => continue,
        };
        let chars: Vec<char> = seg_text.chars().collect();
        let n_chars = chars.len();
        let n_valid = valid_ts.len();

        if n_chars == 0 {
            continue;
        }

        // Distribute character slices proportionally across tokens.
        // Token i gets chars[i*n_chars/n_valid .. (i+1)*n_chars/n_valid].
        // The word text is only used internally for speaker assignment
        // (find_best_speaker matches on timestamps, not text content),
        // so character-level splitting is fine for both CJK and Latin scripts.
        for (pos, t_dtw) in valid_ts.iter().enumerate() {
            let char_start = pos * n_chars / n_valid;
            let char_end = ((pos + 1) * n_chars / n_valid).min(n_chars);
            if char_start >= n_chars {
                break;
            }
            let w: String = chars[char_start..char_end].iter().collect();
            if w.trim().is_empty() {
                continue;
            }
            let t = audio_start_secs + *t_dtw as f64 / 100.0;
            words.push(WordTs { w, s: t, e: t });
        }
    }

    words
}

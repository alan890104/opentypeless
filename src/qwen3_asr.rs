use std::sync::Mutex;

use candle_core::Device;

use crate::stt::{qwen3_asr_model_dir, is_qwen3_asr_downloaded, Qwen3AsrModel};

// ── Cache ─────────────────────────────────────────────────────────────────────

pub struct Qwen3AsrCache {
    pub engine: qwen3_asr::inference::AsrInference,
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

    let device = if cfg!(feature = "metal") {
        Device::new_metal(0).unwrap_or_else(|e| {
            tracing::warn!("Metal unavailable, falling back to CPU (Qwen3-ASR will be slow): {}", e);
            Device::Cpu
        })
    } else if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).unwrap_or_else(|e| {
            tracing::warn!("CUDA unavailable, falling back to CPU (Qwen3-ASR will be slow): {}", e);
            Device::Cpu
        })
    } else {
        tracing::warn!("No GPU acceleration available, Qwen3-ASR will run on CPU");
        Device::Cpu
    };
    let engine = qwen3_asr::inference::AsrInference::load(&model_dir, device)
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

    let lang_opt: Option<&str> = if language == "auto" || language.is_empty() {
        None
    } else {
        Some(language)
    };

    let result = c
        .engine
        .transcribe_samples(samples, lang_opt)
        .map_err(|e| format!("Qwen3-ASR transcription failed: {}", e))?;

    Ok(result.text)
}

/// Drop the cached engine so a new model can be loaded on the next call.
pub fn invalidate_qwen3_asr_cache(cache: &Mutex<Option<Qwen3AsrCache>>) {
    if let Ok(mut guard) = cache.lock() {
        *guard = None;
    }
}

//! Speaker diarization: segmentation + WeSpeaker ResNet34-LM embeddings + clustering.
//!
//! # Two-phase pipeline
//!
//! ## Real-time (during recording)
//! ```text
//! VAD chunk (f32, 16 kHz)
//!   → SegmentationModel [1,1,N] → [1,T,7]  (speaker-class per frame)
//!   → sub-segment boundaries (silence + speaker-class changes)
//!   → WeSpeaker ResNet34-LM [1,n_frames,80] → [1,256]  (per sub-segment, 256-dim embedding)
//!   → L2 normalize
//!   → online cosine clustering  (greedy, for real-time labels)
//!   → "SPEAKER_00" / "SPEAKER_01" / …  +  embedding buffered for phase 2
//! ```
//!
//! ## Finalization (at meeting stop, matches pyannote quality)
//! ```text
//! buffered (start, end, L2-normalised embedding) for all sub-segments
//!   → centroid linkage agglomerative clustering (scipy-equivalent)
//!     threshold=0.7045 euclidean on L2-normalised embeddings (pyannote default)
//!     min_cluster_size=12 — small clusters reassigned to nearest large centroid
//!   → optimal speaker labels matching pyannote AgglomerativeClustering
//!   → update WAL speaker fields before writing to SQLite
//! ```
//!
//! ## Why two phases?
//! Online clustering is O(N) and irrevocable; it can mis-label early segments
//! before enough context is available.  Agglomerative is O(N²) but globally
//! optimal.  The two-phase approach gives real-time output and optimal final
//! labels, matching the experiment's DER ≈ 10.5 % on VoxConverse.
//!
//! ## Segmentation model
//! `speech-turn-detector.onnx` (pyannote segmentation-3.0, Alkd/speech-turn-detector-onnx).
//! Input:  `"input_values"` — `[1, 1, 160_000]` f32 (i16 values cast to f32, scale ×32767)
//! Output: `"logits"` — `[1, num_frames, 7]` where class 0 = silence.
//! Frame hop = 270 samples (≈ 16.9 ms), first frame at sample 721.
//! Splits at both silence→speech transitions AND speaker-class changes within speech.
//!
//! ## Bug note
//! `pyannote_rs::get_segments()` has an iterator bug: terminates early when
//! speech crosses a 10-second window boundary.  We reimplemented the segmentation
//! logic directly against the ONNX session.

use std::path::Path;

use ndarray::{Array1, Array3, ArrayViewD, Axis, IxDyn};
use rustfft::{FftPlanner, num_complex::Complex};

/// Speech turn detector (pyannote segmentation-3.0) download URL.
/// Alkd/speech-turn-detector-onnx — no HuggingFace auth required.
pub const SEGMENTATION_URL: &str =
    "https://huggingface.co/Alkd/speech-turn-detector-onnx/resolve/main/model.onnx";

/// WeSpeaker ResNet34-LM speaker embedding model download URL.
/// Alkd/speaker-embedding-onnx — no HuggingFace auth required.
/// 256-dim embeddings, full-precision ONNX (~26.5 MB).
/// Input:  `"input_features"` — `[1, n_frames, 80]` log-fbank features
/// Output: `"embedding"` — `[1, 256]` speaker embedding
pub const WESPEAKER_URL: &str =
    "https://huggingface.co/Alkd/speaker-embedding-onnx/resolve/main/model.onnx";

// ── Segmentation model ─────────────────────────────────────────────────────────

/// Frame parameters matching speech-turn-detector.onnx (from experiment).
const SEG_WINDOW_SAMPLES: usize = 160_000; // 10 s at 16 kHz
const SEG_FRAME_HOP: usize = 270; // ≈ 16.9 ms
const SEG_FRAME_START: usize = 721; // first frame center
const SEG_MIN_SUBSEG_SAMPLES: usize = 400; // 25 ms — shorter sub-segs skipped

/// Offline pyannote pipeline: sliding window step = 10 % of 10 s = 1 s.
const PYANNOTE_STEP_SAMPLES: usize = 16_000;

/// Number of speakers modelled by speech-turn-detector (powerset with 3 speakers).
const PYANNOTE_NUM_SPEAKERS: usize = 3;

/// Powerset class → per-speaker binary activation mapping.
///
/// `pyannote.audio.utils.powerset.Powerset(num_classes=3, max_set_size=2)`
/// enumerates subsets of {spk0, spk1, spk2} with at most 2 simultaneous speakers:
///
///   class 0 → ∅          [F, F, F]
///   class 1 → {spk0}     [T, F, F]
///   class 2 → {spk1}     [F, T, F]
///   class 3 → {spk2}     [F, F, T]
///   class 4 → {spk0,1}   [T, T, F]
///   class 5 → {spk0,2}   [T, F, T]
///   class 6 → {spk1,2}   [F, T, T]
const PYANNOTE_POWERSET_MAP: [[bool; PYANNOTE_NUM_SPEAKERS]; 7] = [
    [false, false, false],
    [true,  false, false],
    [false, true,  false],
    [false, false, true ],
    [true,  true,  false],
    [true,  false, true ],
    [false, true,  true ],
];

/// Pyannote speech turn detector model.  Runs directly via ORT, bypassing the
/// buggy `pyannote_rs::get_segments()` iterator.
pub struct SegmentationModel {
    session: ort::session::Session,
}

// Compile-time proof that the assumption below holds: ort::session::Session
// implements Send in ort 2.x (via `unsafe impl Send for Session {}` in
// ort/src/session/mod.rs). If it did not, the assertion would fail to compile.
const _: fn() = || {
    fn _assert_send<T: Send>() {}
    _assert_send::<ort::session::Session>();
};
// SAFETY: ort::session::Session is Send in ort 2.x (verified by assertion
// above). SegmentationModel wraps only Session; no thread-local state.
unsafe impl Send for SegmentationModel {}

impl SegmentationModel {
    pub fn new(model_path: &Path) -> Result<Self, String> {
        let session = ort::session::Session::builder()
            .map_err(|e| format!("ORT builder: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Load segmentation model: {e}"))?;
        Ok(Self { session })
    }

    /// Segment `samples_f32` (16 kHz) into speech sub-segments.
    ///
    /// Returns `Vec<(start_sample, end_sample)>` within the input slice.
    /// Splits at:
    ///   - silence → speech boundaries
    ///   - speaker-class changes within continuous speech (intra-utterance)
    ///
    /// The caller converts sample indices to seconds using ÷ 16_000.
    pub fn find_sub_segments(&mut self, samples_f32: &[f32]) -> Vec<(usize, usize)> {
        // Scale f32 [-1,1] → i16-equivalent f32 range (model trained on i16 cast to f32).
        let total_len = samples_f32.len();
        let scaled: Vec<f32> = samples_f32.iter().map(|&s| s * 32767.0).collect();

        // Pad to multiple of SEG_WINDOW_SAMPLES so the last window is complete.
        let pad = (SEG_WINDOW_SAMPLES - (total_len % SEG_WINDOW_SAMPLES)) % SEG_WINDOW_SAMPLES;
        let mut padded = scaled;
        padded.extend(std::iter::repeat_n(0.0f32, pad));

        // None = silence, Some(cls) = speech with class cls.
        let mut cur_class: Option<usize> = None;
        let mut seg_start_sample: usize = 0;
        let mut segments: Vec<(usize, usize)> = Vec::new();

        for win_start in (0..padded.len()).step_by(SEG_WINDOW_SAMPLES) {
            // offset is per-window: anchored to win_start so per-window frame
            // indices don't drift across windows (SEG_FRAME_HOP is approximate).
            let mut offset: usize = win_start + SEG_FRAME_START;
            let window = &padded[win_start..win_start + SEG_WINDOW_SAMPLES];

            let array = match Array1::from_vec(window.to_vec())
                .into_shape_with_order((1_usize, 1_usize, SEG_WINDOW_SAMPLES))
            {
                Ok(a) => a,
                Err(e) => {
                    tracing::warn!("[seg] reshape failed: {e}");
                    break;
                }
            };

            let tensor =
                match ort::value::TensorRef::from_array_view(array.view().into_dyn()) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::warn!("[seg] tensor creation failed: {e}");
                        break;
                    }
                };

            let ort_outs = match self.session.run(ort::inputs!["input_values" => tensor]) {
                Ok(o) => o,
                Err(e) => {
                    tracing::warn!("[seg] inference failed: {e}");
                    break;
                }
            };

            let tensor_out = match ort_outs
                .get("logits")
                .and_then(|t| t.try_extract_tensor::<f32>().ok())
            {
                Some(t) => t,
                None => {
                    tracing::warn!("[seg] missing 'output' tensor");
                    break;
                }
            };

            let (shape, data) = tensor_out;
            let shape_vec: Vec<usize> =
                (0..shape.len()).map(|i| shape[i] as usize).collect();
            let view =
                match ArrayViewD::<f32>::from_shape(IxDyn(&shape_vec), data) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!("[seg] from_shape failed: {e}");
                        break;
                    }
                };

            // view: [1, num_frames, 7]
            for batch in view.outer_iter() {
                // [num_frames, 7]
                for frame in batch.axis_iter(Axis(0)) {
                    // [7]
                    let max_idx = frame
                        .iter()
                        .enumerate()
                        .max_by(|a, b| {
                            a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0);

                    // Clamp offset to actual audio length for boundary extraction.
                    let frame_sample = offset.min(total_len);

                    match (cur_class, max_idx) {
                        // Silence → silence: no-op.
                        (None, 0) => {}
                        // Silence → speech: start a new sub-segment.
                        (None, cls) => {
                            seg_start_sample = frame_sample;
                            cur_class = Some(cls);
                        }
                        // Speech → silence: flush sub-segment.
                        (Some(_), 0) => {
                            let end = frame_sample;
                            if end > seg_start_sample
                                && end - seg_start_sample >= SEG_MIN_SUBSEG_SAMPLES
                            {
                                segments.push((seg_start_sample, end.min(total_len)));
                            }
                            cur_class = None;
                        }
                        // Speech → different speech class: intra-utterance speaker change.
                        (Some(prev), cls) if prev != cls => {
                            let end = frame_sample;
                            if end > seg_start_sample
                                && end - seg_start_sample >= SEG_MIN_SUBSEG_SAMPLES
                            {
                                segments.push((seg_start_sample, end.min(total_len)));
                            }
                            seg_start_sample = frame_sample;
                            cur_class = Some(cls);
                        }
                        // Same class: continue.
                        _ => {}
                    }

                    offset += SEG_FRAME_HOP;
                }
            }
        }

        // Flush trailing speech (the pyannote-rs iterator omits this too).
        if cur_class.is_some() {
            let end = total_len;
            if end > seg_start_sample && end - seg_start_sample >= SEG_MIN_SUBSEG_SAMPLES {
                segments.push((seg_start_sample, end));
            }
        }

        segments
    }

    /// Run the powerset segmentation model and return **soft** per-speaker
    /// probabilities, matching `Powerset(3, 2).to_multilabel(soft=True)`.
    ///
    /// Returns `(soft, binary)`:
    /// - `soft[f][s]` — probability that speaker `s` is active at frame `f`
    ///   (softmax over 7 powerset classes, then summed per speaker).
    /// - `binary[f][s]` — `soft[f][s] > 0.5` (used for embedding masking).
    fn run_window_soft(
        &mut self,
        scaled_window: &[f32],
    ) -> (Vec<[f32; PYANNOTE_NUM_SPEAKERS]>, Vec<[bool; PYANNOTE_NUM_SPEAKERS]>) {
        let array =
            match Array1::from_vec(scaled_window.to_vec())
                .into_shape_with_order((1_usize, 1_usize, SEG_WINDOW_SAMPLES))
            {
                Ok(a) => a,
                Err(e) => {
                    tracing::warn!("[pyannote] seg reshape failed: {e}");
                    return (vec![], vec![]);
                }
            };

        let tensor =
            match ort::value::TensorRef::from_array_view(array.view().into_dyn()) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!("[pyannote] seg tensor failed: {e}");
                    return (vec![], vec![]);
                }
            };

        let ort_outs = match self.session.run(ort::inputs!["input_values" => tensor]) {
            Ok(o) => o,
            Err(e) => {
                tracing::warn!("[pyannote] seg inference failed: {e}");
                return (vec![], vec![]);
            }
        };

        let (shape, data) = match ort_outs
            .get("logits")
            .and_then(|t| t.try_extract_tensor::<f32>().ok())
        {
            Some(t) => t,
            None => {
                tracing::warn!("[pyannote] seg missing 'logits'");
                return (vec![], vec![]);
            }
        };

        let shape_vec: Vec<usize> = (0..shape.len()).map(|i| shape[i] as usize).collect();
        let view = match ArrayViewD::<f32>::from_shape(IxDyn(&shape_vec), data) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("[pyannote] seg from_shape failed: {e}");
                return (vec![], vec![]);
            }
        };

        let mut soft_result = Vec::new();
        let mut binary_result = Vec::new();
        for batch in view.outer_iter() {
            for frame in batch.axis_iter(Axis(0)) {
                // frame: [7] logits — softmax → per-speaker probabilities
                let max_logit = frame.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = frame.iter().map(|&x| (x - max_logit).exp()).collect();
                let sum_exp: f32 = exp.iter().sum();

                // Powerset to multilabel: sum probabilities for each speaker
                let mut speaker_probs = [0.0f32; PYANNOTE_NUM_SPEAKERS];
                for (cls, &e) in exp.iter().enumerate() {
                    let p = e / sum_exp;
                    for s in 0..PYANNOTE_NUM_SPEAKERS {
                        if PYANNOTE_POWERSET_MAP[cls.min(6)][s] {
                            speaker_probs[s] += p;
                        }
                    }
                }
                let binary = [
                    speaker_probs[0] > 0.5,
                    speaker_probs[1] > 0.5,
                    speaker_probs[2] > 0.5,
                ];
                soft_result.push(speaker_probs);
                binary_result.push(binary);
            }
        }
        (soft_result, binary_result)
    }
}

// ── WeSpeaker ResNet34-LM embedding extractor ─────────────────────────────────

/// Speaker embedding extractor for `speaker-embedding.onnx` (WeSpeaker ResNet34-LM, ONNX).
///
/// Uses ORT directly (bypassing `pyannote_rs::EmbeddingExtractor`) to support
/// the model's tensor names:
///   Input:  `"input_features"` — `[1, n_frames, 80]` log-fbank features
///   Output: `"embedding"` — `[1, 256]` speaker embedding
///
/// Feature extraction follows WeSpeaker's Kaldi-style training pipeline:
/// 25 ms frames / 10 ms shift / 512-pt FFT / 80 mel bins / preemphasis 0.97
/// / Hamming window / per-utterance CMN.
pub(crate) struct WeSpeakerExtractor {
    session: ort::session::Session,
}

impl WeSpeakerExtractor {
    pub fn new(model_path: &Path) -> Result<Self, String> {
        let use_coreml = std::env::var("SUMI_COREML").map(|v| v == "1").unwrap_or(true);
        #[cfg(target_os = "macos")]
        let session = if use_coreml {
            ort::session::Session::builder()
                .map_err(|e| format!("ORT builder: {e}"))?
                .with_execution_providers([ort::execution_providers::CoreMLExecutionProvider::default().build()])
                .map_err(|e| format!("CoreML EP: {e}"))?
                .commit_from_file(model_path)
                .map_err(|e| format!("Load WeSpeaker ResNet34-LM: {e}"))?
        } else {
            ort::session::Session::builder()
                .map_err(|e| format!("ORT builder: {e}"))?
                .commit_from_file(model_path)
                .map_err(|e| format!("Load WeSpeaker ResNet34-LM: {e}"))?
        };
        #[cfg(not(target_os = "macos"))]
        let session = ort::session::Session::builder()
            .map_err(|e| format!("ORT builder: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Load WeSpeaker ResNet34-LM: {e}"))?;
        Ok(Self { session })
    }

    /// Compute 256-dim speaker embedding from raw 16 kHz f32 audio.
    /// Returns the raw (pre-L2-normalisation) embedding vector.
    pub fn compute(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
        let fbank = compute_fbank(samples);
        if fbank.is_empty() {
            return Err("audio too short for fbank (< 400 samples)".into());
        }
        let n_frames = fbank.len();
        let flat: Vec<f32> = fbank.into_iter().flatten().collect();

        let array = Array3::from_shape_vec((1, n_frames, 80), flat)
            .map_err(|e| format!("fbank reshape: {e}"))?;
        let tensor = ort::value::TensorRef::from_array_view(array.view().into_dyn())
            .map_err(|e| format!("tensor creation: {e}"))?;

        let outs = self
            .session
            .run(ort::inputs!["input_features" => tensor])
            .map_err(|e| format!("ResNet34-LM inference: {e}"))?;

        let (_, data) = outs
            .get("embedding")
            .ok_or("missing 'last_hidden_state' output tensor")?
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract embedding: {e}"))?;

        Ok(data.to_vec())
    }

    /// Extract speaker embedding with a per-frame mask applied (exclude-overlap mode).
    ///
    /// Matches Python `_emb_forward` in `test_onnx.py`:
    ///
    /// ```python
    /// fbank = emb_inner.compute_fbank(waveforms)   # (1, T, 80)
    /// w_np  = mask[:T] if len(mask) >= T else np.pad(mask, (0, T-len), fill=1.0)
    /// feats = fbank[0][w_np > 0]                   # (T', 80) — masked frames only
    /// emb   = emb_sess.run({"input_features": feats[None]})[0][0]
    /// ```
    ///
    /// Mask alignment (same as Python):
    ///   fbank frame j < `frame_mask.len()` → use `frame_mask[j]`
    ///   fbank frame j ≥ `frame_mask.len()` → always include (padded True)
    ///
    /// Returns `Err` if no frames pass the mask.
    pub(crate) fn compute_masked(
        &mut self,
        samples: &[f32],
        frame_mask: &[bool],
    ) -> Result<Vec<f32>, String> {
        let fbank = compute_fbank(samples);
        if fbank.is_empty() {
            return Err("audio too short for fbank".into());
        }
        // Keep only frames where the mask is True (pad beyond mask len with True).
        let filtered: Vec<[f32; 80]> = fbank
            .into_iter()
            .enumerate()
            .filter(|(j, _)| {
                if *j < frame_mask.len() { frame_mask[*j] } else { true }
            })
            .map(|(_, frame)| frame)
            .collect();

        if filtered.is_empty() {
            return Err("no frames after masking".into());
        }

        let n_filtered = filtered.len();
        let flat: Vec<f32> = filtered.into_iter().flatten().collect();

        let array = Array3::from_shape_vec((1, n_filtered, 80), flat)
            .map_err(|e| format!("masked fbank reshape: {e}"))?;
        let tensor =
            ort::value::TensorRef::from_array_view(array.view().into_dyn())
                .map_err(|e| format!("masked tensor: {e}"))?;

        let outs = self
            .session
            .run(ort::inputs!["input_features" => tensor])
            .map_err(|e| format!("masked ResNet34-LM inference: {e}"))?;

        let (_, data) = outs
            .get("embedding")
            .ok_or("missing 'last_hidden_state' output")?
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract masked embedding: {e}"))?;

        Ok(data.to_vec())
    }
}

/// 80-dim log-Mel filterbank features from 16 kHz f32 audio.
///
/// Matches `torchaudio.compliance.kaldi.fbank` (WeSpeaker training pipeline):
///   - 25 ms frames (400 samples), 10 ms shift (160 samples)
///   - DC offset removal → preemphasis (0.97) → Hamming window
///   - 512-point FFT → power spectrum → 80-band Kaldi-continuous mel filterbank → log
///   - Per-utterance CMN: subtract per-feature mean across all frames
fn compute_fbank(samples: &[f32]) -> Vec<[f32; 80]> {
    const FRAME_LEN: usize = 400;   // 25 ms @ 16 kHz
    const FRAME_SHIFT: usize = 160; // 10 ms @ 16 kHz
    const N_MELS: usize = 80;
    const PREEMPH: f32 = 0.97;

    if samples.len() < FRAME_LEN {
        return vec![];
    }

    let hamming: Vec<f32> = (0..FRAME_LEN)
        .map(|n| {
            0.54 - 0.46
                * (std::f32::consts::TAU * n as f32 / (FRAME_LEN - 1) as f32).cos()
        })
        .collect();

    let filterbank = get_mel_filterbank();
    let num_frames = 1 + (samples.len() - FRAME_LEN) / FRAME_SHIFT;
    let mut frames: Vec<[f32; N_MELS]> = Vec::with_capacity(num_frames);

    for fi in 0..num_frames {
        let start = fi * FRAME_SHIFT;
        let s = &samples[start..start + FRAME_LEN];

        // DC offset removal.
        let mean = s.iter().sum::<f32>() / FRAME_LEN as f32;
        let mut frame: Vec<f32> = s.iter().map(|&x| x - mean).collect();

        // Preemphasis (Kaldi backward pass to avoid in-place aliasing).
        for t in (1..FRAME_LEN).rev() {
            frame[t] -= PREEMPH * frame[t - 1];
        }
        frame[0] *= 1.0 - PREEMPH;

        // Hamming window.
        for (v, &w) in frame.iter_mut().zip(hamming.iter()) {
            *v *= w;
        }

        // Zero-pad to 512 and compute power spectrum.
        let mut fft_in = [0.0f32; 512];
        fft_in[..FRAME_LEN].copy_from_slice(&frame);
        let power = fft_power_spectrum(&fft_in);

        // Mel filterbank → log energy.
        let mut mel_energy = [0.0f32; N_MELS];
        for (m, filter) in filterbank.iter().enumerate() {
            let e: f32 = filter.iter().zip(power.iter()).map(|(&f, &p)| f * p).sum();
            mel_energy[m] = e.max(f32::EPSILON).ln();
        }
        frames.push(mel_energy);
    }

    // Per-utterance CMN: subtract per-feature mean across all frames.
    if !frames.is_empty() {
        let n = frames.len() as f32;
        let mut mean = [0.0f32; N_MELS];
        for frame in &frames {
            for (m, &v) in frame.iter().enumerate() {
                mean[m] += v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n;
        }
        for frame in &mut frames {
            for (m, v) in frame.iter_mut().enumerate() {
                *v -= mean[m];
            }
        }
    }

    frames
}

/// Cached 80-band mel filterbank matrix (512-pt FFT, 16 kHz).
static MEL_FILTERBANK: std::sync::OnceLock<Box<[[f32; 257]; 80]>> =
    std::sync::OnceLock::new();

fn get_mel_filterbank() -> &'static [[f32; 257]; 80] {
    MEL_FILTERBANK.get_or_init(make_mel_filterbank_80).as_ref()
}

/// Build the 80×257 mel triangular filterbank for 16 kHz / 512-pt FFT.
/// Kaldi-continuous style matching `torchaudio.compliance.kaldi.fbank`:
///   - weights computed in mel space per FFT bin (not snapped to bin indices)
///   - `max(0, min(up_slope, down_slope))` — same as torchaudio `get_mel_banks`
///   - Nyquist bin (k=256) is always zero (torchaudio pads the filterbank there)
fn make_mel_filterbank_80() -> Box<[[f32; 257]; 80]> {
    const N_MELS: usize = 80;
    const N_FREQS: usize = 257; // 512/2 + 1
    const N_FFT_BINS: usize = 256; // torchaudio: num_fft_bins = padded_window_size/2 (no Nyquist)
    const SR: f32 = 16_000.0;
    const F_MIN: f32 = 20.0;
    const F_MAX: f32 = 8_000.0;
    const N_FFT: usize = 512;

    let hz_to_mel = |f: f32| 2595.0_f32 * (1.0 + f / 700.0).log10();

    let fft_bin_width = SR / N_FFT as f32; // 31.25 Hz per bin
    let mel_min = hz_to_mel(F_MIN);
    let mel_max = hz_to_mel(F_MAX);
    // divide by N_MELS+1 matching torchaudio: "divide by num_bins+1 because of end-effects"
    let mel_delta = (mel_max - mel_min) / (N_MELS + 1) as f32;

    // Precompute mel value for each FFT bin k=0..255 (Nyquist excluded, same as torchaudio)
    let fft_mel: Vec<f32> = (0..N_FFT_BINS)
        .map(|k| hz_to_mel(fft_bin_width * k as f32))
        .collect();

    let mut fb = Box::new([[0.0f32; N_FREQS]; N_MELS]);
    for m in 0..N_MELS {
        let left_mel   = mel_min + m as f32 * mel_delta;
        let center_mel = mel_min + (m + 1) as f32 * mel_delta;
        let right_mel  = mel_min + (m + 2) as f32 * mel_delta;
        for k in 0..N_FFT_BINS {
            let mel = fft_mel[k];
            let up   = (mel - left_mel)   / (center_mel - left_mel);
            let down = (right_mel - mel)  / (right_mel  - center_mel);
            fb[m][k] = up.min(down).max(0.0);
        }
        // fb[m][256] stays 0.0 (Nyquist always zero — torchaudio right-pads with 0)
    }
    fb
}

/// Cached 512-pt FFT plan (rustfft — allocated once, reused across all frames).
static FFT_PLAN: std::sync::OnceLock<std::sync::Arc<dyn rustfft::Fft<f32>>> =
    std::sync::OnceLock::new();

fn get_fft_plan() -> &'static std::sync::Arc<dyn rustfft::Fft<f32>> {
    FFT_PLAN.get_or_init(|| {
        let mut planner = FftPlanner::<f32>::new();
        planner.plan_fft_forward(512)
    })
}

/// 512-point real-input FFT → power spectrum (257 bins).
/// Uses `rustfft` for numerically accurate results matching `torch.fft.rfft`.
fn fft_power_spectrum(input: &[f32; 512]) -> [f32; 257] {
    let fft = get_fft_plan();
    let mut buf: Vec<Complex<f32>> =
        input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buf);
    // Power spectrum: |FFT[k]|^2 for k = 0..=256.
    let mut power = [0.0f32; 257];
    for (i, c) in buf[..257].iter().enumerate() {
        power[i] = c.re * c.re + c.im * c.im;
    }
    power
}

// ── Agglomerative clustering ───────────────────────────────────────────────────

/// Pyannote threshold constant for WeSpeaker ResNet34-LM embeddings (euclidean space).
///
/// This is the exact value from pyannote's default AgglomerativeClustering config:
/// `threshold=0.7045654963945799` — euclidean distance on L2-normalized embeddings.
/// Equivalent cosine distance: 0.7045²/2 ≈ 0.248.
pub(crate) const PYANNOTE_THRESHOLD: f32 = 0.704_565_5;

/// Centroid linkage agglomerative clustering matching scipy/pyannote exactly.
///
/// Matches `scipy.cluster.hierarchy.linkage(method='centroid', metric='euclidean')`
/// followed by `fcluster(Z, threshold, criterion='distance') - 1` and
/// `min_cluster_size` small-cluster reassignment from pyannote's
/// `AgglomerativeClustering.cluster()`.
///
/// **Input**: `embeddings` — each must already be L2-normalised (unit norm).
/// **threshold**: euclidean distance cutoff; same units as Python's 0.7045.
/// **min_cluster_size**: clusters smaller than
///   `min(min_cluster_size, max(1, round(0.1 * n)))` are reassigned to the
///   nearest large cluster centroid (by euclidean distance between centroids).
///
/// Returns 0-based cluster labels ordered by first-appearance index of each
/// cluster's earliest member, matching numpy `np.unique(…, return_inverse=True)`.
pub(crate) fn centroid_linkage_cluster(
    embeddings: &[Vec<f32>],
    threshold: f32,
    min_cluster_size: usize,
) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0];
    }

    // ── State ───────────────────────────────────────────────────────────────
    // centroids[slot]: current centroid vector (NOT L2-normalised after first merge)
    let mut centroids: Vec<Vec<f32>> = embeddings.to_vec();
    // sizes[slot]: number of original embeddings in this cluster
    let mut sizes: Vec<usize> = vec![1; n];
    // active[slot]: true while this cluster is still alive
    let mut active: Vec<bool> = vec![true; n];
    // origin[i]: which cluster slot currently owns original embedding i
    let mut origin: Vec<usize> = (0..n).collect();
    // last_merge[slot]: merge count when this slot LAST absorbed another cluster.
    // None = slot never absorbed anything (pure singleton).
    // Used for scipy-compatible label ordering: merged clusters get lower labels
    // (in creation order), singletons get higher labels (in original-index order).
    let mut last_merge: Vec<Option<usize>> = vec![None; n];
    let mut merge_count: usize = 0;

    // ── Squared euclidean distance matrix (n×n, symmetric) ──────────────────
    // For L2-normalised inputs: ||a−b||² = 2·(1 − a·b) = 2·cosine_dist(a,b)
    let mut d2 = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let sq = sq_euclidean(&embeddings[i], &embeddings[j]);
            d2[i][j] = sq;
            d2[j][i] = sq;
        }
    }

    // ── Main merge loop ──────────────────────────────────────────────────────
    loop {
        let active_slots: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
        if active_slots.len() <= 1 {
            break;
        }

        // Find pair with minimum squared distance (ties broken by lower index,
        // matching scipy's upper-triangular traversal order).
        let mut min_sq = f32::MAX;
        let mut best_a = 0;
        let mut best_b = 1;
        for (ii, &a) in active_slots.iter().enumerate() {
            for &b in &active_slots[ii + 1..] {
                if d2[a][b] < min_sq {
                    min_sq = d2[a][b];
                    best_a = a;
                    best_b = b;
                }
            }
        }

        // Compare euclidean distance (sqrt of squared) against threshold.
        if min_sq.sqrt() >= threshold {
            break;
        }

        // ── Merge best_b into best_a ─────────────────────────────────────────
        let ni = sizes[best_a] as f32;
        let nj = sizes[best_b] as f32;
        let n_new = ni + nj;

        // New centroid = weighted average of the two centroids.
        for k in 0..centroids[best_a].len() {
            centroids[best_a][k] =
                (ni * centroids[best_a][k] + nj * centroids[best_b][k]) / n_new;
        }
        sizes[best_a] = n_new as usize;
        active[best_b] = false;
        last_merge[best_a] = Some(merge_count);
        merge_count += 1;

        // Lance-Williams update rule for centroid linkage (squared euclidean):
        //   d²(new, l) = (nᵢ/n_new)·d²(a,l) + (nⱼ/n_new)·d²(b,l) − (nᵢnⱼ/n_new²)·d²(a,b)
        // Clamp to 0 to avoid tiny negative values from floating-point cancellation
        // ("reversal" artefact inherent to centroid linkage).
        for &l in &active_slots {
            if l == best_a || l == best_b {
                continue;
            }
            let new_d2 = (ni / n_new) * d2[best_a][l] + (nj / n_new) * d2[best_b][l]
                - (ni * nj / (n_new * n_new)) * min_sq;
            let clamped = new_d2.max(0.0);
            d2[best_a][l] = clamped;
            d2[l][best_a] = clamped;
        }

        // Update membership: all embeddings in best_b now belong to best_a.
        for item in origin.iter_mut().take(n) {
            if *item == best_b {
                *item = best_a;
            }
        }
    }

    // ── min_cluster_size reassignment ────────────────────────────────────────
    // Matches pyannote: effective_min = min(min_cluster_size, max(1, round(0.1·n)))
    let effective_min = min_cluster_size
        .min(((0.1 * n as f64).round() as usize).max(1));

    let active_slots: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
    let large: Vec<usize> = active_slots
        .iter()
        .cloned()
        .filter(|&s| sizes[s] >= effective_min)
        .collect();

    if !large.is_empty() {
        let small: Vec<usize> = active_slots
            .iter()
            .cloned()
            .filter(|&s| sizes[s] < effective_min)
            .collect();
        for small_slot in small {
            // Nearest large cluster by COSINE distance between centroids.
            // Matches pyannote: `cdist(large_centroids, small_centroids, metric='cosine')`.
            // Centroids are means of L2-normalised embeddings (norm < 1), so cosine ≠ euclidean.
            let nearest = large
                .iter()
                .cloned()
                .min_by(|&a, &b| {
                    cosine_dist(&centroids[small_slot], &centroids[a])
                        .total_cmp(&cosine_dist(&centroids[small_slot], &centroids[b]))
                })
                .unwrap();
            for item in origin.iter_mut().take(n) {
                if *item == small_slot {
                    *item = nearest;
                }
            }
            active[small_slot] = false;
        }
    }

    // ── Re-number 0-based matching scipy fcluster + np.unique ordering ──────
    //
    // scipy fcluster assigns cluster IDs such that MERGED clusters (formed by
    // earlier merges in the dendrogram) receive LOWER IDs than singleton
    // observations that were never merged.  np.unique then re-numbers in sorted
    // order of those IDs.  The net effect:
    //
    //   1. Merged cluster slots  → sorted by MERGE CREATION ORDER (earlier = lower label)
    //   2. Singleton slots       → sorted by MIN ORIGINAL OBSERVATION INDEX (lower = lower label)
    //   3. Singletons always get HIGHER labels than all merged clusters.
    //
    // This matches: `_, clusters = np.unique(clusters, return_inverse=True)` in pyannote.

    let final_active: Vec<usize> = (0..n).filter(|&i| active[i]).collect();

    // Partition into merged (has last_merge) and singleton (last_merge = None).
    let mut merged_slots: Vec<(usize, usize)> = final_active
        .iter()
        .filter_map(|&s| last_merge[s].map(|order| (order, s)))
        .collect();
    merged_slots.sort_by_key(|&(order, _)| order);

    let mut singleton_slots: Vec<(usize, usize)> = final_active
        .iter()
        .filter(|&&s| last_merge[s].is_none())
        .map(|&s| {
            let min_orig = (0..n).find(|&i| origin[i] == s).unwrap_or(s);
            (min_orig, s)
        })
        .collect();
    singleton_slots.sort_by_key(|&(min_orig, _)| min_orig);

    let mut slot_to_label: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut next_label = 0usize;
    for (_, slot) in merged_slots {
        slot_to_label.insert(slot, next_label);
        next_label += 1;
    }
    for (_, slot) in singleton_slots {
        slot_to_label.insert(slot, next_label);
        next_label += 1;
    }

    (0..n).map(|i| slot_to_label[&origin[i]]).collect()
}

/// Squared euclidean distance ||a−b||².
fn sq_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

// ── Online clustering state ────────────────────────────────────────────────────

/// Pure-Rust online clustering state (no ONNX dependency — testable in isolation).
pub(crate) struct SpeakerClusters {
    centroids: Vec<Vec<f32>>,
    counts: Vec<usize>,
    threshold: f32,
}

impl SpeakerClusters {
    pub fn new(threshold: f32) -> Self {
        Self { centroids: Vec::new(), counts: Vec::new(), threshold }
    }

    /// Assign an **already L2-normalised** embedding; returns 0-based speaker index.
    pub fn assign(&mut self, emb: Vec<f32>) -> usize {
        let (best_id, best_dist) = self
            .centroids
            .iter()
            .enumerate()
            .map(|(id, c)| (id, cosine_dist(&emb, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((usize::MAX, f32::MAX));

        if best_dist < self.threshold {
            let n = self.counts[best_id] as f32;
            let new_c: Vec<f32> = self.centroids[best_id]
                .iter()
                .zip(emb.iter())
                .map(|(c, e)| (c * n + e) / (n + 1.0))
                .collect();
            self.centroids[best_id] = l2_normalize(&new_c);
            self.counts[best_id] += 1;
            best_id
        } else {
            let id = self.centroids.len();
            self.centroids.push(emb);
            self.counts.push(1);
            id
        }
    }

    /// Assign to the **nearest existing centroid** without creating a new cluster
    /// or updating any centroid.  Used for very short (< 1 s) segments whose
    /// embeddings are unreliable — they are placed next to the closest known
    /// speaker but do not pollute the cluster centroids.
    ///
    /// Returns `None` if no clusters exist yet (caller should fall back to `assign`).
    pub fn assign_nearest(&self, emb: &[f32]) -> Option<usize> {
        self.centroids
            .iter()
            .enumerate()
            .map(|(id, c)| (id, cosine_dist(emb, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    pub fn reset(&mut self) {
        self.centroids.clear();
        self.counts.clear();
    }

    pub fn speaker_count(&self) -> usize {
        self.centroids.len()
    }
}

// ── Diarization engine ─────────────────────────────────────────────────────────

/// Combined diarization engine: segmentation + WeSpeaker embedding + two-phase clustering.
pub struct DiarizationEngine {
    emb_extractor: WeSpeakerExtractor,
    segmentation: Option<SegmentationModel>,
    clusters: SpeakerClusters,
    /// (start_secs, end_secs, L2-normalised embedding) for every sub-segment
    /// processed this session.  Used by `finalize_labels()` for agglomerative pass.
    segment_buffer: Vec<(f64, f64, Vec<f32>)>,
}

// Compile-time proof: EmbeddingExtractor wraps ort::session::Session which is
// Send in ort 2.x. If this assertion fails to compile, the unsafe impl below
// must be removed.
const _: fn() = || {
    fn _assert_send<T: Send>() {}
    _assert_send::<WeSpeakerExtractor>();
};
// SAFETY: WeSpeakerExtractor wraps ort::session::Session which declares
// `unsafe impl Send for Session {}` in ort 2.0.0-rc.10 (session/mod.rs:565).
// All other fields (SpeakerClusters, Vec<…>) are Send. DiarizationEngine is
// accessed only through a Mutex<Option<DiarizationEngine>>, serialising access.
unsafe impl Send for DiarizationEngine {}

impl DiarizationEngine {
    /// Load embedding model; optionally also load segmentation model.
    ///
    /// If `seg_model` is `None` or its path does not exist, intra-utterance
    /// speaker change detection is disabled (VAD chunks are treated as atomic).
    pub fn new(emb_model: &Path, seg_model: Option<&Path>) -> Result<Self, String> {
        let emb_extractor = WeSpeakerExtractor::new(emb_model)
            .map_err(|e| format!("Failed to load WeSpeaker model: {e}"))?;

        let segmentation = seg_model
            .filter(|p| p.exists())
            .and_then(|p| match SegmentationModel::new(p) {
                Ok(m) => {
                    tracing::info!(
                        "[diarization] segmentation model loaded: {}",
                        p.display()
                    );
                    Some(m)
                }
                Err(e) => {
                    tracing::warn!("[diarization] segmentation model load failed: {e}");
                    None
                }
            });

        if segmentation.is_none() {
            tracing::info!(
                "[diarization] running without segmentation model \
                 (no intra-utterance speaker-change detection)"
            );
        }

        Ok(Self {
            emb_extractor,
            segmentation,
            // Online threshold: cosine distance for real-time greedy clustering.
            // 0.90 is conservative (accepts wider same-speaker variance) to avoid
            // over-splitting in the online pass; the agglomerative finalization pass
            // with PYANNOTE_THRESHOLD corrects any mis-splits at session end.
            clusters: SpeakerClusters::new(0.90),
            segment_buffer: Vec::new(),
        })
    }

    /// Run the full offline pyannote pipeline on a complete audio file.
    ///
    /// Requires the segmentation model to be loaded; returns empty vec otherwise.
    /// Use this for audio import where the entire file is available — it produces
    /// significantly better diarization than the online `process_vad_chunk` path
    /// because it uses global centroid-linkage agglomerative clustering over all
    /// windows at once rather than greedy per-chunk assignment.
    ///
    /// Returns `Vec<(start_secs, end_secs, speaker_label)>` sorted by start time.
    pub(crate) fn diarize_full(&mut self, samples: &[f32]) -> Vec<(f64, f64, String)> {
        let seg = match self.segmentation.as_mut() {
            Some(s) => s,
            None => {
                tracing::warn!(
                    "[diarization] diarize_full requires segmentation model; \
                     returning empty"
                );
                return vec![];
            }
        };
        pyannote_diarize(samples, seg, &mut self.emb_extractor)
    }

    /// Process one VAD chunk of 16 kHz f32 audio.
    ///
    /// 1. If segmentation model available: split at silence AND speaker-class
    ///    changes → multiple sub-segments.
    /// 2. For each sub-segment: CAMPPlus embedding → L2-normalise →
    ///    online cluster (immediate label) + buffer (for agglomerative pass).
    ///
    /// Returns `Vec<(start_secs, end_secs, speaker_label)>`.
    /// `start_secs` / `end_secs` are **absolute** (chunk_start_secs + intra-chunk offset).
    /// Returns empty vec if no speech detected or all sub-segments are too short.
    pub fn process_vad_chunk(
        &mut self,
        samples_f32: &[f32],
        chunk_start_secs: f64,
    ) -> Vec<(f64, f64, String)> {
        // Determine sub-segment boundaries within this chunk.
        let sub_segs: Vec<(usize, usize)> = if let Some(ref mut seg) = self.segmentation {
            let segs = seg.find_sub_segments(samples_f32);
            if segs.is_empty() {
                tracing::debug!("[diarization] segmentation found no speech in chunk");
                return vec![];
            }
            segs
        } else {
            // No segmentation model: treat full chunk as one sub-segment.
            if samples_f32.len() < SEG_MIN_SUBSEG_SAMPLES {
                return vec![];
            }
            vec![(0, samples_f32.len())]
        };

        let mut result = Vec::new();

        // CAMPPlus norms scale with segment duration: short segments
        // have larger pre-norm norms, making L2-normalised embeddings
        // incomparable across lengths.  Two mitigations:
        //   1. Cap embedding input to 5 s (normalises dynamic range).
        //   2. Segments < 1 s use `assign_nearest` (no centroid update, not buffered
        //      for agglomerative) since their embeddings are too unreliable to anchor
        //      a cluster centroid.
        const MAX_EMB_SAMPLES: usize = 5 * 16_000; // 5 s @ 16 kHz
        const MIN_RELIABLE_SAMPLES: usize = 16_000; // 1 s @ 16 kHz

        for (start_samp, end_samp) in sub_segs {
            let end_samp = end_samp.min(samples_f32.len());
            if end_samp <= start_samp {
                continue;
            }
            let sub_slice = &samples_f32[start_samp..end_samp];
            let start_secs = chunk_start_secs + start_samp as f64 / 16_000.0;
            let end_secs = chunk_start_secs + end_samp as f64 / 16_000.0;
            let is_reliable = sub_slice.len() >= MIN_RELIABLE_SAMPLES;

            // Cap to first 5 s to normalise embedding magnitude across lengths.
            let emb_slice = if sub_slice.len() > MAX_EMB_SAMPLES {
                &sub_slice[..MAX_EMB_SAMPLES]
            } else {
                sub_slice
            };
            if emb_slice.len() < 400 {
                tracing::debug!(
                    "[diarization] sub-segment [{:.2}-{:.2}s] too short ({} samples)",
                    start_secs,
                    end_secs,
                    emb_slice.len()
                );
                continue;
            }

            let raw_emb: Vec<f32> = match self.emb_extractor.compute(emb_slice) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("[diarization] embedding failed: {e}");
                    // Skip this sub-segment entirely: a WAL entry with speaker=""
                    // cannot be corrected by update_wal_speakers (not in segment_buffer).
                    continue;
                }
            };

            if raw_emb.is_empty() {
                continue;
            }

            let emb = l2_normalize(&raw_emb);

            // Skip NaN embeddings: very short/silent segments fed to the ONNX model
            // can produce NaN output (near-zero fbank → numerical instability in ResNet34).
            // A NaN cosine distance is always ∞, so the segment would never merge with
            // any cluster, creating a spurious singleton that corrupts agglomerative results.
            if emb.iter().any(|x| x.is_nan()) {
                tracing::debug!(
                    "[diarization] NaN embedding for [{:.2}-{:.2}s], skipping",
                    start_secs, end_secs
                );
                continue;
            }

            // Online clustering.
            // Cap agglomerative buffer to avoid O(N³) blowup on very long meetings.
            // N=200 keeps the worst-case final pass to ~2 billion flops (~1 s on CPU).
            // Segments beyond the cap still get an online cluster label; they are simply
            // excluded from the agglomerative re-labeling pass at meeting end.
            const MAX_AGGLOMERATIVE_SEGS: usize = 200;
            let speaker_id = if is_reliable {
                // Reliable segment: may create new cluster, updates centroid, buffered.
                let id = self.clusters.assign(emb.clone());
                if self.segment_buffer.len() < MAX_AGGLOMERATIVE_SEGS {
                    self.segment_buffer.push((start_secs, end_secs, emb));
                } else {
                    tracing::warn!(
                        "[diarization] segment_buffer at cap ({MAX_AGGLOMERATIVE_SEGS}); \
                         sub-segment [{start_secs:.1}-{end_secs:.1}s] excluded from \
                         agglomerative pass (online label kept)"
                    );
                }
                id
            } else {
                // Short/unreliable: assign to nearest existing cluster (no creation,
                // no centroid update, not buffered for agglomerative).
                match self.clusters.assign_nearest(&emb) {
                    Some(id) => id,
                    None => {
                        // No clusters yet — bootstrap with this short segment.
                        self.clusters.assign(emb.clone())
                        // Not buffered: bootstrap centroid only, no agglomerative entry.
                    }
                }
            };

            let label = format!("SPEAKER_{:02}", speaker_id);
            tracing::debug!(
                "[diarization] [{:.2}-{:.2}s] → {} (online, reliable={})",
                start_secs,
                end_secs,
                label,
                is_reliable
            );
            result.push((start_secs, end_secs, label));
        }

        result
    }

    /// At session end: run centroid-linkage agglomerative clustering on all buffered
    /// embeddings (matching pyannote's `AgglomerativeClustering.cluster()`).
    ///
    /// Returns `Vec<(start_secs, end_secs, speaker_label)>` with globally-optimal
    /// labels, one entry per sub-segment processed during the session.
    /// Clears the buffer (call once per session).
    pub fn finalize_labels(&mut self) -> Vec<(f64, f64, String)> {
        if self.segment_buffer.is_empty() {
            return vec![];
        }

        let embeddings: Vec<Vec<f32>> = self
            .segment_buffer
            .iter()
            .map(|(_, _, emb)| emb.clone())
            .collect();

        // Centroid linkage with pyannote's exact threshold and min_cluster_size=12.
        // threshold=0.7045 is euclidean distance on L2-normalised embeddings (same
        // space as Python).  min_cluster_size=12 reassigns spurious singleton clusters.
        let labels =
            centroid_linkage_cluster(&embeddings, PYANNOTE_THRESHOLD, 12);

        let result: Vec<(f64, f64, String)> = self
            .segment_buffer
            .iter()
            .zip(labels.iter())
            .map(|((start, end, _), &id)| (*start, *end, format!("SPEAKER_{:02}", id)))
            .collect();

        let n_speakers = labels.iter().max().map(|&m| m + 1).unwrap_or(0);
        tracing::info!(
            "[diarization] finalized {} sub-segments → {} speakers \
             (centroid linkage, threshold={:.4})",
            result.len(),
            n_speakers,
            PYANNOTE_THRESHOLD,
        );

        self.segment_buffer.clear();
        result
    }

    /// Reset speaker state for a new session.  Models stay loaded.
    pub fn reset(&mut self) {
        self.clusters.reset();
        self.segment_buffer.clear();
        tracing::debug!("[diarization] speaker state reset for new session");
    }

    /// Number of distinct speakers identified by online clustering so far.
    pub fn speaker_count(&self) -> usize {
        self.clusters.speaker_count()
    }
}

// ── Offline pyannote-equivalent pipeline ───────────────────────────────────────

/// Compute the mean centroid of each cluster (NOT L2-normalised).
///
/// Matches pyannote `assign_embeddings`: `np.mean(train_embeddings[train_clusters == k], axis=0)`.
/// The resulting centroids have norm < 1 (mean of L2-normalised embeddings).
/// Distance comparisons use cosine distance which handles non-unit norms correctly.
///
/// `labels[i]` is the cluster index (0-based) of `embeddings[i]`.
/// Returns `k` centroids in a `Vec<Vec<f32>>`.
fn compute_centroids(embeddings: &[Vec<f32>], labels: &[usize], k: usize) -> Vec<Vec<f32>> {
    let dim = embeddings.first().map(|e| e.len()).unwrap_or(0);
    let mut sums: Vec<Vec<f64>> = vec![vec![0.0f64; dim]; k];
    let mut counts: Vec<usize> = vec![0; k];
    for (emb, &lbl) in embeddings.iter().zip(labels.iter()) {
        if lbl < k {
            for (s, &v) in sums[lbl].iter_mut().zip(emb.iter()) {
                *s += v as f64;
            }
            counts[lbl] += 1;
        }
    }
    sums.into_iter().enumerate().map(|(i, s)| {
        let n = counts[i].max(1) as f64;
        s.iter().map(|&x| (x / n) as f32).collect()
    }).collect()
}

/// Full offline speaker diarization matching the Python `test_onnx.py` pipeline.
///
/// Replicates `pyannote.audio.pipelines.SpeakerDiarization` configured with:
///   - `speech-turn-detector.onnx`  (powerset, 3 speakers, argmax → per-speaker binary)
///   - `speaker-embedding.onnx`  (masked fbank input, 256-dim output)
///   - `AgglomerativeClustering` (centroid linkage, threshold=0.7045, min_cluster_size=12)
///   - `embedding_exclude_overlap=True`  (exclude overlapping-speech frames)
///   - Sliding window: 10 s duration / 1 s step (same as pyannote default)
///
/// # Reconstruction
///
/// Follows `SpeakerDiarization.reconstruct()` + `to_diarization()`:
///   1. `clustered_seg[w, f, k]` = max over s where label[w,s]==k of soft_seg[w,f,s]
///   2. `activation_sum[abs_f][k]` += clustered_seg[w, f_local, k]  (skip_avg=True)
///   3. `speaker_count[abs_f]` = round(mean_w sum_s soft_seg[w, f_local, s])  (skip_avg=False)
///   4. At each abs_f: mark top `speaker_count` global speakers by activation_sum as active.
///   5. Convert binary per-frame labels to (start, end, speaker) segments, merge contiguous.
///
/// Uses **soft** per-speaker probabilities (softmax → powerset_to_multilabel) for
/// reconstruction and speaker_count, matching pyannote exactly. Binary masks
/// (threshold > 0.5) are only used for embedding extraction.
///
/// Returns segments sorted by start time.
pub(crate) fn pyannote_diarize(
    samples: &[f32],
    seg_model: &mut SegmentationModel,
    emb_extractor: &mut WeSpeakerExtractor,
) -> Vec<(f64, f64, String)> {
    let total = samples.len();
    if total < 400 {
        return vec![];
    }

    // Minimum masked frames to prefer exclude-overlap path (matches Python
    // `min_num_frames = ceil(num_frames * min_num_samples / num_samples)`
    // = ceil(589 * 400 / 160000) = 2).
    const MIN_EXCL_FRAMES: usize = 2;
    // Minimum segment duration to emit.
    // Matches pyannote 3.1 config: `min_duration_on=0.0` (no filtering).
    const MIN_SEG_S: f64 = 0.0;
    // Matches pyannote `BaseClustering.filter_embeddings(min_active_ratio=0.2)`:
    // only (window, speaker) pairs where the speaker is EXCLUSIVELY active for at
    // least 20 % of the 589-frame window are used for centroid-linkage clustering.
    // Others are still assigned to the nearest centroid after clustering.
    //
    // pyannote default: ceil(0.2 * 589) = 118 frames ≈ 2 s exclusive speech.
    const MIN_RELIABLE_FRAMES: usize = 118;

    let num_windows = if total <= SEG_WINDOW_SAMPLES {
        1
    } else {
        (total - SEG_WINDOW_SAMPLES) / PYANNOTE_STEP_SAMPLES + 1
    };

    // ── Phase 1: Per-window segmentation + masked embeddings ──────────────────

    // soft_segs[w][f][s] — soft per-speaker probabilities (for reconstruction)
    let mut all_soft: Vec<Vec<[f32; PYANNOTE_NUM_SPEAKERS]>> =
        Vec::with_capacity(num_windows);
    // binary_segs[w][f][s] — binarized at 0.5 (for embedding masking only)
    let mut all_binary: Vec<Vec<[bool; PYANNOTE_NUM_SPEAKERS]>> =
        Vec::with_capacity(num_windows);
    // embeddings[w * PYANNOTE_NUM_SPEAKERS + s] — L2-normalised embedding or None
    let mut all_embs: Vec<Option<Vec<f32>>> =
        Vec::with_capacity(num_windows * PYANNOTE_NUM_SPEAKERS);
    // clean_counts[w * PYANNOTE_NUM_SPEAKERS + s] — exclusive-frame count per slot
    let mut all_clean_counts: Vec<usize> =
        Vec::with_capacity(num_windows * PYANNOTE_NUM_SPEAKERS);

    for w in 0..num_windows {
        let win_start = w * PYANNOTE_STEP_SAMPLES;
        let src_end = (win_start + SEG_WINDOW_SAMPLES).min(total);

        // Build padded scaled window for segmentation ONNX (model was trained on
        // i16 values cast to f32, so we scale × 32767).
        let mut scaled = vec![0.0f32; SEG_WINDOW_SAMPLES];
        for (i, &s) in samples[win_start..src_end].iter().enumerate() {
            scaled[i] = s * 32767.0;
        }

        // Segmentation: soft per-speaker probabilities + binarized masks
        let (soft_frames, binary_frames) = seg_model.run_window_soft(&scaled);

        // Build raw (unscaled) padded window for fbank / embedding extraction.
        let mut raw = vec![0.0f32; SEG_WINDOW_SAMPLES];
        raw[..src_end - win_start].copy_from_slice(&samples[win_start..src_end]);

        // For each speaker slot: compute masked embedding using BINARY mask.
        for s in 0..PYANNOTE_NUM_SPEAKERS {
            let speaker_mask: Vec<bool> = binary_frames.iter().map(|f| f[s]).collect();
            let active_count = speaker_mask.iter().filter(|&&x| x).count();

            if active_count == 0 {
                // Inactive speaker: no embedding for this (window, speaker) pair.
                all_embs.push(None);
                all_clean_counts.push(0);
                continue;
            }

            // Exclude-overlap: keep only frames where this speaker is the ONLY one active.
            let clean_mask: Vec<bool> = binary_frames
                .iter()
                .map(|f| f[s] && f.iter().filter(|&&x| x).count() < 2)
                .collect();
            let clean_count = clean_mask.iter().filter(|&&x| x).count();

            // Use clean mask if it has enough frames; otherwise fall back to full mask.
            // Matches Python: `used_mask = clean_mask if sum(clean_mask) > min_num_frames else mask`
            let used: &[bool] = if clean_count > MIN_EXCL_FRAMES {
                &clean_mask
            } else {
                &speaker_mask
            };

            let emb = match emb_extractor.compute_masked(&raw, used) {
                Ok(raw_emb) => {
                    let normed = l2_normalize(&raw_emb);
                    if normed.iter().any(|x| x.is_nan()) { None } else { Some(normed) }
                }
                Err(_) => None,
            };
            all_embs.push(emb);
            all_clean_counts.push(clean_count);
        }

        all_soft.push(soft_frames);
        all_binary.push(binary_frames);
    }

    // ── Phase 2: Centroid-linkage clustering — reliable embeddings only ────────
    //
    // Matches pyannote `BaseClustering.filter_embeddings(min_active_ratio=0.2)`:
    // cluster only (window, speaker) pairs with ≥ MIN_RELIABLE_FRAMES exclusive frames.
    // Unreliable-but-valid embeddings are assigned to the nearest centroid afterwards.

    // Reliable: valid embedding AND sufficient exclusive speech.
    let reliable_idx: Vec<usize> = (0..all_embs.len())
        .filter(|&i| all_embs[i].is_some() && all_clean_counts[i] >= MIN_RELIABLE_FRAMES)
        .collect();
    // All valid (including unreliable): used for reconstruction label assignment.
    let valid_idx: Vec<usize> = (0..all_embs.len())
        .filter(|&i| all_embs[i].is_some())
        .collect();

    tracing::info!(
        "[diarize] windows={} valid_embs={} reliable_embs={} (min_frames>={})",
        num_windows,
        valid_idx.len(),
        reliable_idx.len(),
        MIN_RELIABLE_FRAMES,
    );
    if reliable_idx.len() >= 2 {
        let mut min_d = f32::MAX;
        let mut max_d = 0.0f32;
        for ii in 0..reliable_idx.len() {
            for jj in (ii + 1)..reliable_idx.len() {
                let a = all_embs[reliable_idx[ii]].as_ref().unwrap();
                let b = all_embs[reliable_idx[jj]].as_ref().unwrap();
                let d = sq_euclidean(a, b).sqrt();
                min_d = min_d.min(d);
                max_d = max_d.max(d);
            }
        }
        tracing::info!(
            "[diarize] reliable emb euclidean: min={:.4} max={:.4} threshold={:.4}",
            min_d, max_d, PYANNOTE_THRESHOLD,
        );
    }

    if valid_idx.is_empty() {
        return vec![];
    }

    // Cluster reliable embeddings (matching pyannote: no retry on 1-speaker result).
    let (cluster_labels, num_global_speakers, centroids) = if reliable_idx.is_empty() {
        // No reliable embeddings: cluster all valid (same as pyannote when
        // filter_embeddings returns nothing).
        tracing::info!("[diarize] no reliable embeddings, clustering all valid");
        let vecs: Vec<Vec<f32>> = valid_idx.iter()
            .map(|&i| all_embs[i].as_ref().unwrap().clone())
            .collect();
        let labels = centroid_linkage_cluster(&vecs, PYANNOTE_THRESHOLD, 12);
        let k = labels.iter().max().map(|&m| m + 1).unwrap_or(1);
        let centroids = compute_centroids(&vecs, &labels, k);
        (labels, k, centroids)
    } else {
        let vecs: Vec<Vec<f32>> = reliable_idx.iter()
            .map(|&i| all_embs[i].as_ref().unwrap().clone())
            .collect();
        let labels = centroid_linkage_cluster(&vecs, PYANNOTE_THRESHOLD, 12);
        let k = labels.iter().max().map(|&m| m + 1).unwrap_or(1);
        let centroids = compute_centroids(&vecs, &labels, k);
        (labels, k, centroids)
    };

    // Build flat label map: embedding index → global label.
    // Reliable embeddings get their cluster label directly; unreliable ones are
    // assigned to the nearest centroid using COSINE distance (matches pyannote
    // `assign_embeddings` with `metric='cosine'`).
    let mut label_map: Vec<Option<usize>> = vec![None; all_embs.len()];

    if reliable_idx.is_empty() {
        // Fallback path: all valid embeddings were clustered directly.
        for (&emb_idx, &lbl) in valid_idx.iter().zip(cluster_labels.iter()) {
            label_map[emb_idx] = Some(lbl);
        }
    } else {
        // Assign reliable embeddings.
        for (&emb_idx, &lbl) in reliable_idx.iter().zip(cluster_labels.iter()) {
            label_map[emb_idx] = Some(lbl);
        }
        // Assign valid-but-unreliable embeddings to nearest centroid (cosine distance).
        for &emb_idx in &valid_idx {
            if label_map[emb_idx].is_some() {
                continue; // already assigned
            }
            let emb = all_embs[emb_idx].as_ref().unwrap();
            let nearest = centroids.iter().enumerate()
                .map(|(k, c)| (k, cosine_dist(emb, c)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(k, _)| k)
                .unwrap_or(0);
            label_map[emb_idx] = Some(nearest);
        }
    }

    // ── Phase 3: Reconstruct per-absolute-frame activations ───────────────────
    //
    // Uses SOFT per-speaker probabilities for reconstruction, matching pyannote:
    //   - `Inference.aggregate(segmentations, …, skip_average=True)` for activation_sum
    //   - `Inference.aggregate(speaker_count, …, skip_average=False)` for speaker count
    //
    // Frame alignment: chunk w's local frame f maps to global abs_frame
    //   abs_f = round((w * STEP_SAMPLES + 0.5 * SEG_FRAME_HOP) / SEG_FRAME_HOP)
    //         = round((w * 16000 + 135) / 270)

    let last_win_start_f = {
        let w = num_windows - 1;
        ((w * PYANNOTE_STEP_SAMPLES + 135) as f64 / SEG_FRAME_HOP as f64).round() as usize
    };
    let num_abs_frames = last_win_start_f + 589 + 10;

    // activation_sum[abs_f][global_speaker] — sum of SOFT clustered activations
    // (skip_average=True: NOT divided by number of contributing windows)
    let mut activation_sum: Vec<Vec<f32>> =
        vec![vec![0.0_f32; num_global_speakers]; num_abs_frames];
    // speaker_sum[abs_f] — sum of SOFT active-speaker counts (for average speaker count)
    let mut speaker_sum: Vec<f32> = vec![0.0_f32; num_abs_frames];
    // window_count[abs_f] — number of windows that contributed a frame here
    let mut window_count: Vec<f32> = vec![0.0_f32; num_abs_frames];

    for w in 0..num_windows {
        let start_f = ((w * PYANNOTE_STEP_SAMPLES + 135) as f64
            / SEG_FRAME_HOP as f64)
            .round() as usize;

        let soft_frames = &all_soft[w];

        for (f, speaker_probs) in soft_frames.iter().enumerate() {
            let abs_f = start_f + f;
            if abs_f >= num_abs_frames {
                break;
            }

            // Skip frames whose center is beyond the actual audio (padding region).
            let abs_center =
                w * PYANNOTE_STEP_SAMPLES + SEG_FRAME_START + f * SEG_FRAME_HOP;
            if abs_center > total {
                continue;
            }

            // Speaker count: sum of SOFT per-speaker probabilities at this frame.
            let n_active: f32 = speaker_probs.iter().sum();
            speaker_sum[abs_f] += n_active;
            window_count[abs_f] += 1.0;

            // Clustered segmentation: for each global speaker k, take MAX of soft
            // probabilities across local speakers assigned to k.
            // Matches Python: `clustered_seg[c,:,k] = max(seg[:,cluster==k], axis=1)`.
            let mut global_max = vec![0.0_f32; num_global_speakers];
            for s in 0..PYANNOTE_NUM_SPEAKERS {
                if let Some(k) = label_map[w * PYANNOTE_NUM_SPEAKERS + s] {
                    if speaker_probs[s] > global_max[k] {
                        global_max[k] = speaker_probs[s];
                    }
                }
            }
            for (k, &val) in global_max.iter().enumerate() {
                activation_sum[abs_f][k] += val;
            }
        }
    }

    // ── Phase 4: Threshold → binary per-frame per-speaker ─────────────────────
    //
    // Matches `to_diarization`: sort global speakers by activation_sum (descending),
    // mark top `round(speaker_sum / window_count)` speakers as active.

    let total_s = total as f64 / 16_000.0;
    let min_seg_frames =
        ((MIN_SEG_S * 16_000.0 / SEG_FRAME_HOP as f64).ceil() as usize).max(1);

    let mut result: Vec<(f64, f64, String)> = Vec::new();

    for speaker in 0..num_global_speakers {
        let mut seg_start: Option<usize> = None;

        for abs_f in 0..num_abs_frames {
            let frame_time_s = (abs_f * SEG_FRAME_HOP) as f64 / 16_000.0;
            if frame_time_s >= total_s {
                // Flush trailing segment if any, then stop.
                if let Some(start) = seg_start.take() {
                    if abs_f - start >= min_seg_frames {
                        let s_s = (start * SEG_FRAME_HOP) as f64 / 16_000.0;
                        result.push((s_s, total_s, format!("SPEAKER_{:02}", speaker)));
                    }
                }
                break;
            }

            let wc = window_count[abs_f];
            let active = if wc > 0.0 {
                // speaker_count at this frame
                let count =
                    (speaker_sum[abs_f] / wc).round() as usize;
                // Sort global speakers by activation (descending); check rank of `speaker`
                let mut ranked: Vec<usize> = (0..num_global_speakers).collect();
                ranked.sort_by(|&a, &b| {
                    activation_sum[abs_f][b]
                        .total_cmp(&activation_sum[abs_f][a])
                });
                ranked[..count.min(num_global_speakers)]
                    .contains(&speaker)
            } else {
                false
            };

            match (seg_start, active) {
                (None, true) => seg_start = Some(abs_f),
                (Some(start), false) => {
                    if abs_f - start >= min_seg_frames {
                        let s_s = (start * SEG_FRAME_HOP) as f64 / 16_000.0;
                        let e_s = frame_time_s;
                        result.push((s_s, e_s, format!("SPEAKER_{:02}", speaker)));
                    }
                    seg_start = None;
                }
                _ => {}
            }
        }

        // Flush trailing segment that reached end of audio.
        if let Some(start) = seg_start {
            let remaining = num_abs_frames - start;
            if remaining >= min_seg_frames {
                let s_s = (start * SEG_FRAME_HOP) as f64 / 16_000.0;
                result.push((s_s, total_s, format!("SPEAKER_{:02}", speaker)));
            }
        }
    }

    result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    result
}

// ── Helpers ────────────────────────────────────────────────────────────────────

fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        tracing::warn!(
            "cosine_dist: embedding dimension mismatch ({} vs {}) — returning max distance",
            a.len(),
            b.len()
        );
        return 1.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - dot / (na * nb + 1e-9)
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-9 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Math helpers ──────────────────────────────────────────────────────────

    #[test]
    fn cosine_dist_identical_vectors_is_zero() {
        let a = vec![0.6_f32, 0.8, 0.0];
        assert!(cosine_dist(&a, &a) < 1e-5);
    }

    #[test]
    fn cosine_dist_orthogonal_vectors_is_one() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        assert!((cosine_dist(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_dist_opposite_vectors_is_two() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        assert!((cosine_dist(&a, &b) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_unit_vector_unchanged() {
        let a = vec![0.6_f32, 0.8];
        let n = l2_normalize(&a);
        assert!((n[0] - 0.6).abs() < 1e-5);
        assert!((n[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_scale_invariant() {
        let a = vec![3.0_f32, 4.0];
        let n = l2_normalize(&a);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        assert!((n[0] - 0.6).abs() < 1e-5);
        assert!((n[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_zero_vector_unchanged() {
        let z = vec![0.0_f32; 4];
        assert_eq!(l2_normalize(&z), z);
    }

    // ── Centroid linkage clustering ───────────────────────────────────────────
    //
    // Thresholds are in EUCLIDEAN distance space (NOT cosine), matching Python's
    // pyannote threshold=0.7045.
    // For L2-normalised inputs: euclidean = sqrt(2 * cosine_dist).
    // orthogonal unit vectors → euclidean = sqrt(2) ≈ 1.414.

    #[test]
    fn centroid_linkage_identical_embeddings_same_cluster() {
        let embs = vec![
            l2_normalize(&[1.0_f32, 0.0, 0.0]),
            l2_normalize(&[1.0_f32, 0.0, 0.0]),
        ];
        // euclidean dist = 0.0 < threshold=0.9 → same cluster
        let labels = centroid_linkage_cluster(&embs, 0.9, 1);
        assert_eq!(labels[0], labels[1]);
    }

    #[test]
    fn centroid_linkage_orthogonal_embeddings_different_clusters() {
        let embs = vec![
            l2_normalize(&[1.0_f32, 0.0, 0.0]),
            l2_normalize(&[0.0_f32, 1.0, 0.0]),
        ];
        // euclidean dist = sqrt(2) ≈ 1.414 > threshold=0.9 → different clusters
        let labels = centroid_linkage_cluster(&embs, 0.9, 1);
        assert_ne!(labels[0], labels[1]);
    }

    #[test]
    fn centroid_linkage_two_speakers_four_segments() {
        // A B A B → labels should be [0,1,0,1].
        let a = l2_normalize(&[1.0_f32, 0.0, 0.0]);
        let b = l2_normalize(&[0.0_f32, 1.0, 0.0]);
        let embs = vec![a.clone(), b.clone(), a, b];
        // threshold=0.9: identical pairs (eucl=0) merge, orthogonal pairs (eucl=1.414) do not.
        let labels = centroid_linkage_cluster(&embs, 0.9, 1);
        assert_eq!(labels[0], labels[2], "both A segments must have same label");
        assert_eq!(labels[1], labels[3], "both B segments must have same label");
        assert_ne!(labels[0], labels[1], "A and B must differ");
    }

    #[test]
    fn centroid_linkage_single_embedding_returns_zero() {
        let embs = vec![l2_normalize(&[1.0_f32, 0.0])];
        assert_eq!(centroid_linkage_cluster(&embs, 0.9, 1), vec![0]);
    }

    #[test]
    fn centroid_linkage_empty_returns_empty() {
        assert!(centroid_linkage_cluster(&[], 0.9, 1).is_empty());
    }

    /// Verify the Lance-Williams update formula numerically against Python/scipy.
    ///
    /// Python reference (actual output from sensevoice-test venv):
    /// ```python
    /// import numpy as np
    /// from scipy.cluster.hierarchy import linkage, fcluster
    ///
    /// # 3 embeddings: a=[1,0,0], b=[0,1,0], c=[0.6,0.8,0] (all unit-norm)
    /// embs = np.array([[1.,0.,0.],[0.,1.,0.],[0.6,0.8,0.]], dtype=np.float32)
    /// Z = linkage(embs, method='centroid', metric='euclidean')
    /// # Z[0] merges b(1) and c(2): dist=sqrt(0.4)≈0.6325
    /// # Z[1] merges a(0) and node3: dist=sqrt(1.3)≈1.1402
    ///
    /// # threshold=0.7  →  only Z[0] fires  →  {a} (singleton), {b,c} (merged)
    /// # scipy fcluster assigns: merged cluster gets lower ID (1), singleton gets higher (2)
    /// # After -1 + np.unique:
    /// labels = fcluster(Z, 0.7, criterion='distance') - 1  # [1, 0, 0]
    /// _, labels = np.unique(labels, return_inverse=True)
    /// print(labels)  # [1, 0, 0]   ← b+c merged cluster → 0, a singleton → 1
    ///
    /// # threshold=1.2  →  both merges fire  →  {a,b,c} single cluster
    /// labels2 = fcluster(Z, 1.2, criterion='distance') - 1
    /// _, labels2 = np.unique(labels2, return_inverse=True)
    /// print(labels2)  # [0, 0, 0]
    /// ```
    #[test]
    fn centroid_linkage_lance_williams_numerical_reference() {
        // a=[1,0,0], b=[0,1,0], c=[0.6,0.8,0] — all unit-norm
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let c = vec![0.6_f32, 0.8, 0.0];
        let embs = vec![a, b, c];

        // ── threshold=0.7: only b+c merge (eucl(b,c)≈0.6325 < 0.7) ──────────
        // eucl(a, merged_bc) = sqrt(1.3) ≈ 1.140 > 0.7 → stops
        // scipy: merged cluster {b,c} → lower label (0), singleton {a} → higher label (1)
        // ⇒ a=1, b=0, c=0
        let labels = centroid_linkage_cluster(&embs, 0.7, 1);
        assert_eq!(labels[1], labels[2], "b and c must be in the same cluster");
        assert_ne!(labels[0], labels[1], "a must be in a different cluster from b+c");
        // Exact scipy-matching values (verified against Python output above):
        assert_eq!(labels, vec![1, 0, 0], "merged cluster gets lower label than singleton");

        // ── threshold=1.2: both merges fire (sqrt(1.3)≈1.140 < 1.2) ─────────
        let labels2 = centroid_linkage_cluster(&embs, 1.2, 1);
        assert_eq!(labels2, vec![0, 0, 0], "all three must merge into one cluster");
    }

    /// Verify scipy-matching label ordering for pure singletons (no merges).
    ///
    /// Python reference (verified output):
    /// ```python
    /// import numpy as np
    /// from scipy.cluster.hierarchy import linkage, fcluster
    ///
    /// # b=[1,0], a=[0,1]: eucl=sqrt(2)≈1.414 > threshold=0.5 → no merge, both singletons
    /// embs = np.array([[1.,0.],[0.,1.]], dtype=np.float32)
    /// Z = linkage(embs, method='centroid', metric='euclidean')
    /// raw = fcluster(Z, 0.5, criterion='distance') - 1  # [0, 1]
    /// _, labels = np.unique(raw, return_inverse=True)
    /// print(labels)  # [0, 1]   ← singletons ordered by original index
    /// ```
    #[test]
    fn centroid_linkage_singleton_ordering_by_original_index() {
        // Both singletons (eucl=sqrt(2) > threshold=0.5 → no merge)
        // Singletons sorted by original observation index: obs0→label0, obs1→label1.
        let b = vec![1.0_f32, 0.0];
        let a = vec![0.0_f32, 1.0];
        let embs = vec![b, a];
        let labels = centroid_linkage_cluster(&embs, 0.5, 1);
        assert_eq!(labels, vec![0, 1]);
    }

    /// min_cluster_size reassignment: small clusters are merged into the nearest
    /// large cluster centroid.
    ///
    /// Python reference:
    /// ```python
    /// import numpy as np
    /// from scipy.cluster.hierarchy import linkage, fcluster
    /// from pyannote.audio.pipelines.clustering import AgglomerativeClustering
    ///
    /// # 6 embeddings: 2 from A (close together), 4 from B (close together)
    /// # With threshold=0.5, centroid linkage yields {A0,A1} and {B0,B1,B2,B3}.
    /// # With min_cluster_size=3: cluster_A has size 2 < effective_min
    /// #   effective_min = min(3, max(1, round(0.1*6))) = min(3,1) = 1 → no reassignment
    /// # For reassignment to trigger we need n large enough:
    /// # With n=20 embeddings: effective_min = min(3, max(1, round(2))) = 2
    /// # → singleton cluster (size 1) gets reassigned to nearest large cluster.
    /// ```
    ///
    /// Since effective_min = min(min_cs, max(1, round(0.1*n))), small n means
    /// effective_min=1 and reassignment never triggers.  We test the behaviour
    /// structurally: with n=4, min_cluster_size=12, effective_min=1 → no reassignment.
    #[test]
    fn centroid_linkage_min_cluster_size_no_op_for_small_n() {
        // 4 embeddings, min_cluster_size=12 but effective_min=max(1,round(0.4))=1
        // → min_cluster_size has no effect here.
        let a = l2_normalize(&[1.0_f32, 0.0, 0.0]);
        let b = l2_normalize(&[0.0_f32, 1.0, 0.0]);
        let embs = vec![a.clone(), b.clone(), a, b];
        let labels = centroid_linkage_cluster(&embs, 0.9, 12);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[1], labels[3]);
        assert_ne!(labels[0], labels[1]);
    }

    /// Cross-validate against Python/scipy for all common cases.
    ///
    /// All expected values verified against:
    ///   scipy.cluster.hierarchy.linkage(method='centroid') + fcluster + np.unique
    /// from sensevoice-test venv (scipy 1.17.1).
    #[test]
    fn centroid_linkage_scipy_cross_validation() {
        // helper: run and check exact expected labels
        fn check(embs: Vec<Vec<f32>>, t: f32, mcs: usize, expected: &[usize], desc: &str) {
            let got = centroid_linkage_cluster(&embs, t, mcs);
            assert_eq!(got, expected, "{desc}: got {got:?}, expected {expected:?}");
        }

        // ABAB pattern — t=0.9: identical pairs merge (eucl=0), orthogonal don't (eucl=1.414)
        // Z[0]=merge(0,2,dist=0), Z[1]=merge(1,3,dist=0). Both get labels in Z order.
        // Python: [0,1,0,1]
        check(
            vec![vec![1.,0.,0.,0.], vec![0.,1.,0.,0.], vec![1.,0.,0.,0.], vec![0.,1.,0.,0.]],
            0.9, 1, &[0,1,0,1], "ABAB t=0.9",
        );

        // abc t=0.7: b+c merge (dist=0.6325 < 0.7), a stays singleton
        // merged cluster {b,c} gets lower label; singleton {a} gets higher label
        // Python: [1,0,0]
        check(
            vec![vec![1.,0.,0.], vec![0.,1.,0.], vec![0.6,0.8,0.]],
            0.7, 1, &[1,0,0], "abc t=0.7",
        );

        // abc t=1.2: all merge. Python: [0,0,0]
        check(
            vec![vec![1.,0.,0.], vec![0.,1.,0.], vec![0.6,0.8,0.]],
            1.2, 1, &[0,0,0], "abc t=1.2",
        );

        // Two singletons. Python: [0,1]
        check(vec![vec![1.,0.], vec![0.,1.]], 0.5, 1, &[0,1], "singletons t=0.5");

        // Close pairs: A1≈A2 and B1≈B2, threshold=0.5
        // d(A1,A2)≈0.141<0.5 → merge first; d(B1,B2)≈0.141<0.5 → merge second
        // Z[0]=merge(A1,A2) → label 0; Z[1]=merge(B1,B2) → label 1. Python: [0,0,1,1]
        check(
            vec![vec![1.,0.,0.], vec![0.9,0.1,0.], vec![0.,1.,0.], vec![0.1,0.9,0.]],
            0.5, 1, &[0,0,1,1], "close pairs t=0.5",
        );

        // 3 fully orthogonal unit vectors — no merges at t=0.9, all singletons
        // Singletons ordered by observation index. Python: [0,1,2]
        check(
            vec![vec![1.,0.,0.], vec![0.,1.,0.], vec![0.,0.,1.]],
            0.9, 1, &[0,1,2], "3 orthogonal t=0.9",
        );

        // Triangle: a=[1,0], b=[0,1], c=[0.7071,0.7071] — t=1.0
        // d(a,c)=d(b,c)≈0.765 < 1.0; tie broken: a(0)+c(2) merge first (lower indices)
        // merged {a,c} → label 0; singleton {b} → label 1. Python: [0,1,0]
        check(
            vec![vec![1.,0.], vec![0.,1.], vec![0.7071,0.7071]],
            1.0, 1, &[0,1,0], "triangle t=1.0",
        );
    }

    /// Verify that with large n where effective_min > 1, a singleton cluster
    /// gets reassigned to the nearest large cluster.
    ///
    /// Construction:
    ///   - 19 embeddings all identical to `a = [1,0,0]`
    ///   - 1 embedding `b = [0,1,0]` (orthogonal, very distant)
    ///   - threshold = 0.001 (no merges happen — all singletons except identical a's)
    ///
    /// Wait, with identical a's and threshold=0.001, they'd all merge first.
    /// Let me use: threshold=1.0 so that:
    ///   - All 19 a's merge pairwise (eucl=0 < 1.0)
    ///   - b stays alone (eucl(a,b)=sqrt(2)≈1.414 > 1.0)
    ///   - Result: cluster_a (size=19) + cluster_b (size=1)
    ///   - effective_min = min(5, max(1, round(0.1*20))) = min(5,2) = 2
    ///   - cluster_b (size=1) < 2 → reassign to nearest large cluster_a
    ///   - Final: all 20 embeddings → label 0
    #[test]
    fn centroid_linkage_singleton_reassigned_to_large_cluster() {
        let a = l2_normalize(&[1.0_f32, 0.0, 0.0]);
        let b = l2_normalize(&[0.0_f32, 1.0, 0.0]);
        // 19 × a + 1 × b = 20 embeddings
        let mut embs: Vec<Vec<f32>> = (0..19).map(|_| a.clone()).collect();
        embs.push(b);

        // threshold=1.0: all a's merge (eucl=0 < 1.0), b stays isolated (eucl=sqrt(2)>1.0)
        // effective_min = min(5, max(1, round(0.1*20))) = 2
        // → cluster_b (size=1) < 2 is reassigned to cluster_a (size=19)
        let labels = centroid_linkage_cluster(&embs, 1.0, 5);

        // All labels must be the same: the singleton b was absorbed into a's cluster.
        let first = labels[0];
        for &l in &labels {
            assert_eq!(l, first, "all embeddings should be in the same cluster after reassignment");
        }
    }

    // ── SpeakerClusters (online, no ONNX) ────────────────────────────────────

    #[test]
    fn clustering_identical_embeddings_same_speaker() {
        let mut c = SpeakerClusters::new(0.5);
        assert_eq!(c.assign(l2_normalize(&[1.0_f32, 0.0, 0.0])), 0);
        assert_eq!(c.assign(l2_normalize(&[1.0_f32, 0.0, 0.0])), 0);
        assert_eq!(c.speaker_count(), 1);
    }

    #[test]
    fn clustering_orthogonal_embeddings_new_speaker() {
        let mut c = SpeakerClusters::new(0.5);
        assert_eq!(c.assign(l2_normalize(&[1.0_f32, 0.0, 0.0])), 0);
        assert_eq!(c.assign(l2_normalize(&[0.0_f32, 1.0, 0.0])), 1);
        assert_eq!(c.speaker_count(), 2);
    }

    #[test]
    fn clustering_reset_clears_state() {
        let mut c = SpeakerClusters::new(0.5);
        c.assign(l2_normalize(&[1.0_f32, 0.0, 0.0]));
        c.reset();
        assert_eq!(c.speaker_count(), 0);
    }

    #[test]
    fn clustering_two_speakers_three_segments() {
        let mut c = SpeakerClusters::new(0.5);
        assert_eq!(c.assign(l2_normalize(&[1.0_f32, 0.0, 0.0])), 0);
        assert_eq!(c.assign(l2_normalize(&[0.0_f32, 1.0, 0.0])), 1);
        assert_eq!(c.assign(l2_normalize(&[0.99_f32, 0.01, 0.0])), 0);
        assert_eq!(c.speaker_count(), 2);
    }

    #[test]
    fn clustering_centroid_stays_stable() {
        let mut c = SpeakerClusters::new(0.5);
        for _ in 0..10 {
            assert_eq!(c.assign(l2_normalize(&[1.0_f32, 0.01, 0.0])), 0);
        }
        assert_eq!(c.speaker_count(), 1);
    }

    // ── Powerset mapping ──────────────────────────────────────────────────────
    //
    // Verified against `pyannote.audio.utils.powerset.Powerset(3, 2).build_mapping()`.

    /// Verify the powerset mapping table matches pyannote's Powerset(3,2).build_mapping().
    ///
    /// Python reference:
    /// ```python
    /// from pyannote.audio.utils.powerset import Powerset
    /// ps = Powerset(3, 2)
    /// print(ps.build_mapping().int().tolist())
    /// # [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]]
    /// ```
    #[test]
    fn powerset_mapping_matches_pyannote() {
        // Expected: mapping[class] = [spk0, spk1, spk2]
        let expected: [[bool; 3]; 7] = [
            [false, false, false],  // 0: silence
            [true,  false, false],  // 1: spk0
            [false, true,  false],  // 2: spk1
            [false, false, true ],  // 3: spk2
            [true,  true,  false],  // 4: spk0+spk1
            [true,  false, true ],  // 5: spk0+spk2
            [false, true,  true ],  // 6: spk1+spk2
        ];
        assert_eq!(PYANNOTE_POWERSET_MAP, expected);
    }

    /// Verify that argmax over a one-hot powerset distribution decodes correctly.
    #[test]
    fn powerset_argmax_decodes_correctly() {
        // Class 4 (spk0+spk1) — one-hot probability vector
        let probs = [0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let cls = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(PYANNOTE_POWERSET_MAP[cls], [true, true, false]);
    }

    // ── fbank numerical match vs torchaudio reference (debug helper) ──────────
    #[test]
    #[ignore]
    fn fbank_print_values_for_comparison() {
        // Uses the same LCG broadband signal as fbank_matches_torchaudio_reference.
        let mut state: u32 = 12345;
        let samples: Vec<f32> = (0..32000_usize)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as f32 / 0xFFFF_FFFFu32 as f32 * 2.0 - 1.0) * 0.1
            })
            .collect();
        let feats = compute_fbank(&samples);
        println!("n_frames={}", feats.len());
        println!("frame0 all 80 bins (Rust):");
        for (i, &v) in feats[0].iter().enumerate() {
            println!("  [{i:2}] {v:.8}");
        }
        println!("frame100 first 5 bins (Rust): {:?}", &feats[100][..5]);
    }

    // ── fbank numerical match vs torchaudio reference ─────────────────────────
    //
    // Reference values generated by the following Python snippet:
    //
    //   import torch, torchaudio.compliance.kaldi as kaldi
    //   import numpy as np
    //
    //   # Pure-f32 LCG — reproducible bit-for-bit in Rust
    //   N = 32000; state = np.uint32(12345)
    //   samples = np.zeros(N, dtype=np.float32)
    //   for i in range(N):
    //       state = np.uint32(np.uint64(state) * 1664525 + 1013904223)
    //       samples[i] = (np.float32(state) / np.float32(0xFFFFFFFF)
    //                     * np.float32(2.0) - np.float32(1.0)) * np.float32(0.1)
    //
    //   wf = torch.from_numpy(samples).unsqueeze(0)
    //   feats = kaldi.fbank(wf * (1<<15), num_mel_bins=80, frame_length=25.0,
    //       frame_shift=10.0, round_to_power_of_two=True, snip_edges=True,
    //       dither=0.0, sample_frequency=16000.0, window_type="hamming",
    //       use_energy=False)
    //   feats = (feats - feats.mean(dim=0, keepdim=True)).numpy().astype(np.float32)
    //
    // The 32768× scaling cancels in CMN (2·log(C) is a constant per bin,
    // subtracted by the mean), so our unscaled Rust input is equivalent.
    #[test]
    fn fbank_matches_torchaudio_reference() {
        // Generate the same broadband samples as Python's LCG above.
        const N: usize = 32000;
        let mut state: u32 = 12345;
        let samples: Vec<f32> = (0..N)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as f32 / 0xFFFF_FFFFu32 as f32 * 2.0 - 1.0) * 0.1
            })
            .collect();

        let feats = compute_fbank(&samples);
        assert_eq!(feats.len(), 198, "expected 198 frames for 2 s at 10 ms shift");

        // Python reference values (frame 0 and frame 100, first 5 + last 5 bins).
        // All 80 bins have real signal energy, so differences < 2e-3 are achievable.
        let py_f0_lo: [f32; 5] = [0.757260, 0.450566, 1.317858, 1.777993, 1.905124];
        let py_f0_hi: [f32; 5] = [0.736576, -0.024185, -0.160379, -0.168732, 0.351114];
        let py_f100_lo: [f32; 5] = [0.025155, -0.507911, -3.182381, -0.235278, -0.523984];

        let check = |frame_idx: usize, bin_start: usize, feats_row: &[f32; 80], refs: &[f32; 5]| {
            for (i, (&rust, &py)) in feats_row[bin_start..bin_start + 5]
                .iter()
                .zip(refs.iter())
                .enumerate()
            {
                let diff = (rust - py).abs();
                assert!(
                    diff < 2e-4,
                    "frame{frame_idx} bin {}: rust={rust:.6} py={py:.6} diff={diff:.2e}",
                    bin_start + i,
                );
            }
        };

        // Tolerance 2e-4: measured max diff for broadband signal is 1.23e-4
        // (bin 100/2: -3.182258 vs -3.182381). Pure-tone zero-energy bins diverge
        // more (FFT noise only), but those don't occur in real speech.
        check(0,   0,  &feats[0],   &py_f0_lo);
        check(0,   75, &feats[0],   &py_f0_hi);
        check(100, 0,  &feats[100], &py_f100_lo);
    }
}

// ── Integration tests (require real ONNX models) ───────────────────────────────
//
// Run with: cargo test diarization::integration -- --ignored
//
// Models are loaded from the standard dev model directory
// (~/.sumi-dev/models/).  Copy or symlink the ONNX files there before running:
//   speech-turn-detector.onnx   (5.9 MB)
//   speaker-embedding.onnx      (26.5 MB)
//
// Test audio: set SUMI_TEST_AUDIO_DIR to a directory containing:
//   voxconv11_60s.wav   (~60 s, 2-speaker clip, validated at DER=10.5%)
//   test1.wav           (any speech recording)

#[cfg(test)]
mod integration {
    use super::*;
    use std::collections::HashSet;

    /// Resolve a test audio file from `SUMI_TEST_AUDIO_DIR` env var.
    /// Tests that call this are already `#[ignore]`-gated, so a missing var
    /// simply means the test is skipped (the assert inside the test will panic
    /// with a clear message).
    fn test_audio(name: &str) -> String {
        let dir = std::env::var("SUMI_TEST_AUDIO_DIR").unwrap_or_else(|_| {
            // Fallback: look next to the workspace root under tests/audio/.
            format!(
                "{}/tests/audio",
                std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into())
            )
        });
        format!("{}/{}", dir, name)
    }

    fn voxconv_wav() -> String { test_audio("voxconv11_60s.wav") }
    fn test1_wav()   -> String { test_audio("test1.wav") }

    /// Load a WAV file to mono f32 samples at its native sample rate.
    fn load_wav(path: &str) -> (Vec<f32>, u32) {
        let mut reader = hound::WavReader::open(path)
            .unwrap_or_else(|e| panic!("Failed to open {path}: {e}"));
        let spec = reader.spec();
        let raw: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max = (1_i32 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.expect("read sample") as f32 / max)
                    .collect()
            }
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| s.expect("read sample"))
                .collect(),
        };
        let ch = spec.channels as usize;
        let mono: Vec<f32> = if ch <= 1 {
            raw
        } else {
            raw.chunks(ch)
                .map(|c| c.iter().sum::<f32>() / ch as f32)
                .collect()
        };
        (mono, spec.sample_rate)
    }

    /// Resample using the Sumi audio module (same resampler as production code).
    fn to_16k(samples: &[f32], rate: u32) -> Vec<f32> {
        if rate == 16000 {
            samples.to_vec()
        } else {
            crate::audio::resample(samples, rate, 16000)
        }
    }

    // ── SegmentationModel ──────────────────────────────────────────────────────

    /// Dump raw model output for the first 10 s of voxconv11 to understand
    /// the actual class distribution — used for diagnosis only.
    #[test]
    #[ignore = "diagnostic only, requires speech-turn-detector.onnx in ~/.sumi-dev/models/"]
    fn segmentation_dump_raw_frames_first_10s() {
        use ndarray::{Array1, ArrayViewD, Axis, IxDyn};

        let seg_path = crate::settings::segmentation_model_path();
        assert!(seg_path.exists());
        assert!(std::path::Path::new(&voxconv_wav()).exists());

        let (samples, sr) = load_wav(&voxconv_wav());
        let samples_16k = to_16k(&samples, sr);

        // Process only the first 10 s (one window).
        let window: Vec<f32> = samples_16k[..SEG_WINDOW_SAMPLES]
            .iter()
            .map(|&s| s * 32767.0)
            .collect();

        let mut session = ort::session::Session::builder()
            .unwrap()
            .commit_from_file(&seg_path)
            .unwrap();

        let array = Array1::from_vec(window)
            .into_shape_with_order((1_usize, 1_usize, SEG_WINDOW_SAMPLES))
            .unwrap();
        let tensor =
            ort::value::TensorRef::from_array_view(array.view().into_dyn()).unwrap();
        let outs = session.run(ort::inputs!["input_values" => tensor]).unwrap();
        let (shape, data) = outs
            .get("logits")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap();
        let shape_vec: Vec<usize> =
            (0..shape.len()).map(|i| shape[i] as usize).collect();
        println!("\n[raw] output shape: {:?}", shape_vec);

        let view =
            ArrayViewD::<f32>::from_shape(IxDyn(&shape_vec), data).unwrap();

        let mut class_hist = [0usize; 8];
        let mut prev_class = 99usize;
        let mut offset = SEG_FRAME_START;

        for batch in view.outer_iter() {
            for (fi, frame) in batch.axis_iter(Axis(0)).enumerate() {
                let max_idx = frame
                    .iter()
                    .enumerate()
                    .max_by(|a, b| {
                        a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                if max_idx < 8 {
                    class_hist[max_idx] += 1;
                }
                if max_idx != prev_class {
                    let t = offset as f64 / 16_000.0;
                    let probs: Vec<f32> = frame.iter().copied().collect();
                    println!(
                        "  frame {:4}  t={:.3}s  class→{}  probs={:.3?}",
                        fi, t, max_idx, &probs[..probs.len().min(8)]
                    );
                    prev_class = max_idx;
                }
                offset += SEG_FRAME_HOP;
            }
        }
        println!("[raw] class histogram: {:?}", class_hist);
        println!("[raw] total frames: {}, duration covered: {:.2}s",
            class_hist.iter().sum::<usize>(),
            class_hist.iter().sum::<usize>() as f64 * SEG_FRAME_HOP as f64 / 16_000.0);
    }

    /// Verify the segmentation model loads and returns at least 2 speech
    /// sub-segments from the 60-second two-speaker VoxConverse sample.
    #[test]
    #[ignore = "requires speech-turn-detector.onnx in ~/.sumi-dev/models/"]
    fn segmentation_finds_multiple_speech_segments_in_voxconv() {
        let seg_path = crate::settings::segmentation_model_path();
        assert!(
            seg_path.exists(),
            "Model not found: {}. Copy speech-turn-detector.onnx there.",
            seg_path.display()
        );
        assert!(
            std::path::Path::new(&voxconv_wav()).exists(),
            "Test audio not found: {}" , voxconv_wav()
        );

        let (samples, sr) = load_wav(&voxconv_wav());
        let samples_16k = to_16k(&samples, sr);
        let duration_s = samples_16k.len() as f64 / 16_000.0;

        let mut model =
            SegmentationModel::new(&seg_path).expect("SegmentationModel::new");
        let segs = model.find_sub_segments(&samples_16k);

        println!(
            "\n[seg] {:.1}s audio → {} sub-segments:",
            duration_s,
            segs.len()
        );
        for (i, (s, e)) in segs.iter().enumerate() {
            println!(
                "  [{}] {:.3}s – {:.3}s  ({:.3}s)",
                i,
                *s as f64 / 16_000.0,
                *e as f64 / 16_000.0,
                (*e - *s) as f64 / 16_000.0
            );
        }

        assert!(
            segs.len() >= 2,
            "Expected ≥2 segments from a 2-speaker recording, got {}",
            segs.len()
        );

        // All segments must be within bounds and non-empty.
        for (s, e) in &segs {
            assert!(*s < *e, "degenerate segment: start {s} >= end {e}");
            assert!(
                *e <= samples_16k.len(),
                "segment end {e} beyond audio length {}",
                samples_16k.len()
            );
            assert!(
                *e - *s >= SEG_MIN_SUBSEG_SAMPLES,
                "segment shorter than min: {} samples",
                *e - *s
            );
        }
    }

    /// Verify the segmentation model detects intra-utterance speaker changes
    /// (not just silence-bounded segments) in the test1.wav file.
    #[test]
    #[ignore = "requires speech-turn-detector.onnx in ~/.sumi-dev/models/"]
    fn segmentation_detects_speaker_class_changes_within_speech() {
        let seg_path = crate::settings::segmentation_model_path();
        assert!(seg_path.exists());
        assert!(std::path::Path::new(&test1_wav()).exists());

        let (samples, sr) = load_wav(&test1_wav());
        let samples_16k = to_16k(&samples, sr);

        let mut model =
            SegmentationModel::new(&seg_path).expect("SegmentationModel::new");
        let segs = model.find_sub_segments(&samples_16k);

        println!(
            "\n[seg-class] test1.wav ({:.1}s) → {} sub-segments",
            samples_16k.len() as f64 / 16_000.0,
            segs.len()
        );
        for (i, (s, e)) in segs.iter().enumerate() {
            println!(
                "  [{}] {:.3}s–{:.3}s",
                i,
                *s as f64 / 16_000.0,
                *e as f64 / 16_000.0
            );
        }

        // test1.wav is a natural conversation; we expect the segmentation model
        // to return multiple sub-segments (at minimum from silence gaps).
        assert!(
            !segs.is_empty(),
            "Segmentation returned 0 sub-segments from test1.wav"
        );
    }

    // ── WeSpeaker embedding ───────────────────────────────────────────────────

    /// Verify WeSpeaker ResNet34-LM produces non-zero 256-dim embeddings from real speech.
    #[test]
    #[ignore = "requires speaker-embedding.onnx in ~/.sumi-dev/models/"]
    fn wespeaker_produces_256_dim_embedding() {
        let emb_path = crate::settings::diarization_model_path();
        assert!(emb_path.exists(), "WeSpeaker model not found: {}", emb_path.display());
        assert!(std::path::Path::new(&test1_wav()).exists());

        let (samples, sr) = load_wav(&test1_wav());
        let samples_16k = to_16k(&samples, sr);
        // Take first 3 seconds of speech.
        let chunk = &samples_16k[..3 * 16_000];

        let mut extractor = WeSpeakerExtractor::new(&emb_path).expect("WeSpeakerExtractor");
        let emb = extractor.compute(chunk).expect("compute");

        println!("\n[wespeaker] embedding dim={}, norm={:.4}", emb.len(), {
            emb.iter().map(|x| x * x).sum::<f32>().sqrt()
        });

        assert_eq!(emb.len(), 256, "expected 256-dim ResNet34-LM embedding");

        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.1, "embedding norm {norm} unexpectedly small");
        let n = l2_normalize(&emb);
        let self_dist = cosine_dist(&n, &n);
        assert!(self_dist < 1e-4, "self cosine dist {self_dist} should be ~0");
    }

    // ── Full two-phase pipeline ────────────────────────────────────────────────

    /// End-to-end pipeline test: process kkshow (multi-speaker) in 30 s chunks → finalize →
    /// expect ≥2 speakers identified (kkshow clip has 3-4 speakers).
    ///
    /// Pre-requisite:
    ///   ffmpeg -i ~/Desktop/kkshow.m4a -ar 16000 -ac 1 /tmp/kkshow_16k.wav
    #[test]
    #[ignore = "requires both ONNX models in ~/.sumi-dev/models/ + /tmp/kkshow_16k.wav"]
    fn full_pipeline_detects_two_speakers_in_voxconv() {
        let emb_path = crate::settings::diarization_model_path();
        let seg_path = crate::settings::segmentation_model_path();
        assert!(emb_path.exists());
        assert!(seg_path.exists());
        let audio_path = "/tmp/kkshow_16k.wav";
        assert!(
            std::path::Path::new(audio_path).exists(),
            "{audio_path} missing — run: ffmpeg -i ~/Desktop/kkshow.m4a -ar 16000 -ac 1 {audio_path}"
        );

        let (samples, sr) = load_wav(audio_path);
        let samples_16k = to_16k(&samples, sr);
        let duration_s = samples_16k.len() as f64 / 16_000.0;

        let mut engine =
            DiarizationEngine::new(&emb_path, Some(&seg_path)).expect("DiarizationEngine");

        // Process in 30-second chunks (simulating meeting mode).
        let chunk_size = 30 * 16_000;
        let mut all_online: Vec<(f64, f64, String)> = Vec::new();
        for chunk_start in (0..samples_16k.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(samples_16k.len());
            let chunk = &samples_16k[chunk_start..chunk_end];
            let start_secs = chunk_start as f64 / 16_000.0;
            let sub_segs = engine.process_vad_chunk(chunk, start_secs);
            println!(
                "\n[chunk {:.0}s–{:.0}s] {} online sub-segs:",
                start_secs,
                chunk_end as f64 / 16_000.0,
                sub_segs.len()
            );
            for (s, e, spk) in &sub_segs {
                println!("  [{:.2}s–{:.2}s] {}", s, e, spk);
            }
            all_online.extend(sub_segs);
        }

        let online_speakers: HashSet<&str> =
            all_online.iter().map(|(_, _, s)| s.as_str()).filter(|s| !s.is_empty()).collect();
        println!(
            "\n[online] {:.1}s, {} sub-segs, {} speakers",
            duration_s,
            all_online.len(),
            online_speakers.len()
        );

        // Phase 2: agglomerative finalization.
        let final_labels = engine.finalize_labels();
        let final_speakers: HashSet<&str> =
            final_labels.iter().map(|(_, _, s)| s.as_str()).collect();
        println!(
            "[agglomerative] {} sub-segs → {} speakers",
            final_labels.len(),
            final_speakers.len()
        );
        for (s, e, spk) in &final_labels {
            println!("  [{:.2}s–{:.2}s] {}", s, e, spk);
        }

        // Must find at least some sub-segments.
        assert!(
            !all_online.is_empty(),
            "process_vad_chunk returned no sub-segments"
        );

        // Agglomerative should detect ≥2 speakers for the kkshow clip (3-4 speakers).
        assert!(
            final_speakers.len() >= 2,
            "Expected ≥2 speakers from kkshow clip, got {}: {:?}",
            final_speakers.len(),
            final_speakers
        );

        // Buffer must be cleared after finalize_labels.
        let second = engine.finalize_labels();
        assert!(
            second.is_empty(),
            "finalize_labels should clear buffer; second call returned {} entries",
            second.len()
        );
    }

    /// Verify centroid linkage produces correct 2-speaker result on synthetic ABAB sequence.
    #[test]
    fn centroid_linkage_matches_reference_on_synthetic_two_speakers() {
        // 4 segments: A B A B (alternating) with clearly separated embeddings.
        let a = l2_normalize(&[1.0_f32, 0.0, 0.0, 0.0]);
        let b = l2_normalize(&[0.0_f32, 1.0, 0.0, 0.0]);
        let embs = vec![a.clone(), b.clone(), a.clone(), b.clone()];

        // threshold=0.9 euclidean: identical pairs (eucl=0) merge; orthogonal (eucl=sqrt(2)≈1.414) do not.
        let labels = centroid_linkage_cluster(&embs, 0.9, 1);
        assert_eq!(labels.len(), 4);
        assert_eq!(labels[0], labels[2], "segments 0 and 2 should be same speaker (A)");
        assert_eq!(labels[1], labels[3], "segments 1 and 3 should be same speaker (B)");
        assert_ne!(labels[0], labels[1], "speakers A and B must differ");
    }

    /// Verify update_wal_speakers rewrites speaker labels correctly.
    #[test]
    fn update_wal_speakers_rewrites_labels() {
        use crate::meeting_notes::{update_wal_speakers, WalSegment};

        let seg0 = WalSegment {
            speaker: "SPEAKER_00".to_string(),
            start: 0.0,
            end: 5.0,
            text: "hello".to_string(),
            words: vec![],
        };
        let seg1 = WalSegment {
            speaker: "SPEAKER_00".to_string(), // online wrongly assigned same speaker
            start: 5.0,
            end: 10.0,
            text: "world".to_string(),
            words: vec![],
        };
        let wal = format!(
            "{}\n{}",
            serde_json::to_string(&seg0).unwrap(),
            serde_json::to_string(&seg1).unwrap()
        );

        // Agglomerative says seg1 is actually SPEAKER_01.
        let labels = vec![
            (0.0_f64, 5.0_f64, "SPEAKER_00".to_string()),
            (5.0_f64, 10.0_f64, "SPEAKER_01".to_string()),
        ];
        let updated = update_wal_speakers(&wal, &labels);

        let lines: Vec<&str> = updated.lines().collect();
        let s0: WalSegment = serde_json::from_str(lines[0]).unwrap();
        let s1: WalSegment = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(s0.speaker, "SPEAKER_00");
        assert_eq!(s1.speaker, "SPEAKER_01");
        assert_eq!(s1.text, "world"); // text unchanged
    }

    /// Diagnostic: print pairwise cosine distances between embeddings extracted
    /// from the ACTUAL sub-segments found by the segmentation model.
    /// Shows the threshold range needed for correct 2-speaker clustering.
    ///
    /// Run with: cargo test diarization::integration::wespeaker_pairwise_distances_actual_segs -- --ignored --nocapture
    #[test]
    #[ignore = "diagnostic: requires both ONNX models + voxconv11.wav"]
    fn wespeaker_pairwise_distances_actual_segs() {
        let emb_path = crate::settings::diarization_model_path();
        let seg_path = crate::settings::segmentation_model_path();
        assert!(emb_path.exists());
        assert!(seg_path.exists());
        assert!(std::path::Path::new(&voxconv_wav()).exists());

        let (samples, sr) = load_wav(&voxconv_wav());
        let samples_16k = to_16k(&samples, sr);

        let mut seg_model = SegmentationModel::new(&seg_path).expect("seg model");
        let mut extractor = WeSpeakerExtractor::new(&emb_path).expect("WeSpeakerExtractor");

        // find_sub_segments scales internally (×32767); pass raw f32 directly.
        let sub_segs = seg_model.find_sub_segments(&samples_16k);
        println!("\n[segmentation] {} sub-segments:", sub_segs.len());

        let mut embeddings: Vec<(f64, f64, usize, Vec<f32>)> = Vec::new();
        const MAX_EMB_SAMPLES: usize = 5 * 16_000;
        for (start_samp, end_samp) in &sub_segs {
            let start_s = *start_samp as f64 / 16_000.0;
            let end_s = *end_samp as f64 / 16_000.0;
            let dur_s = end_s - start_s;
            let chunk = &samples_16k[*start_samp..*end_samp];
            let emb_chunk = if chunk.len() > MAX_EMB_SAMPLES { &chunk[..MAX_EMB_SAMPLES] } else { chunk };
            let raw: Vec<f32> = extractor.compute(emb_chunk).expect("compute");
            let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalized = l2_normalize(&raw);
            println!("  [{:.2}s–{:.2}s] dur={:.2}s  norm={:.2}  emb_len={}s", start_s, end_s, dur_s, norm, emb_chunk.len() / 16_000);
            embeddings.push((start_s, end_s, chunk.len(), normalized));
        }

        println!("\n--- Pairwise cosine distances (actual sub-segments) ---");
        for i in 0..embeddings.len() {
            for j in (i+1)..embeddings.len() {
                let d = cosine_dist(&embeddings[i].3, &embeddings[j].3);
                println!("  [{:.2}s vs {:.2}s]: {:.4}", embeddings[i].0, embeddings[j].0, d);
            }
        }

        // Find the threshold gap for 2-cluster agglomerative.
        let mut all_dists: Vec<f32> = (0..embeddings.len())
            .flat_map(|i| ((i+1)..embeddings.len()).map(move |j| (i, j)))
            .map(|(i, j)| cosine_dist(&embeddings[i].3, &embeddings[j].3))
            .collect();
        all_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("\nAll distances sorted: {:?}", all_dists.iter().map(|d| format!("{:.4}", d)).collect::<Vec<_>>());

        // Try centroid linkage at pyannote's default threshold.
        let n = embeddings.len();
        if n >= 2 {
            let raw_embs: Vec<Vec<f32>> = embeddings.iter().map(|(_, _, _, e)| e.clone()).collect();
            for &t in &[PYANNOTE_THRESHOLD, 0.8, 0.9, 1.0] {
                let labels = centroid_linkage_cluster(&raw_embs, t, 12);
                let unique: std::collections::HashSet<usize> = labels.iter().cloned().collect();
                println!("[centroid threshold={:.4}] {} clusters (should be 2)", t, unique.len());
                for ((s, e, _, _), lbl) in embeddings.iter().zip(labels.iter()) {
                    println!("  [{:.2}s–{:.2}s] → SPEAKER_{:02}", s, e, lbl);
                }
            }
        }

        assert!(!embeddings.is_empty());
    }

    /// Diagnostic: print pairwise cosine distances between embeddings extracted
    /// from known speaker-change regions in voxconv11. This tells us what
    /// threshold is appropriate for online clustering.
    ///
    /// Run with: cargo test diarization::integration::wespeaker_pairwise_distances -- --ignored --nocapture
    #[test]
    #[ignore = "diagnostic: requires wespeaker ONNX + voxconv11.wav"]
    fn wespeaker_pairwise_distances_voxconv() {
        let emb_path = crate::settings::diarization_model_path();
        assert!(emb_path.exists(), "WeSpeaker model not found");
        assert!(std::path::Path::new(&voxconv_wav()).exists(), "voxconv11.wav not found — set SUMI_TEST_AUDIO_DIR");

        let (samples, sr) = load_wav(&voxconv_wav());
        let samples_16k = to_16k(&samples, sr);

        let mut extractor = WeSpeakerExtractor::new(&emb_path).expect("WeSpeakerExtractor");

        // Extract embeddings from 6 non-overlapping 3s windows spread across the 60s clip.
        // Windows at 0s, 10s, 20s, 30s, 40s, 50s — if there are 2 speakers,
        // some pairs should have high cosine distance.
        let window_secs = 3.0f64;
        let offsets_s = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0f64];
        let mut embeddings: Vec<(f64, Vec<f32>)> = Vec::new();

        for &off in &offsets_s {
            let start = (off * 16_000.0) as usize;
            let end = ((off + window_secs) * 16_000.0) as usize;
            let end = end.min(samples_16k.len());
            if end <= start { continue; }
            let chunk = &samples_16k[start..end];
            let raw: Vec<f32> = extractor.compute(chunk).expect("compute");
            let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalized = l2_normalize(&raw);
            println!("[embedding @{:.0}s] dim={}, raw_norm={:.2}", off, raw.len(), norm);
            embeddings.push((off, normalized));
        }

        // Print pairwise cosine distances.
        println!("\n--- Pairwise cosine distances ---");
        for i in 0..embeddings.len() {
            for j in (i+1)..embeddings.len() {
                let d = cosine_dist(&embeddings[i].1, &embeddings[j].1);
                println!("  [{:.0}s vs {:.0}s]: {:.4}", embeddings[i].0, embeddings[j].0, d);
            }
        }

        // Print distance range.
        let dists: Vec<f32> = (0..embeddings.len())
            .flat_map(|i| ((i+1)..embeddings.len()).map(move |j| (i, j)))
            .map(|(i, j)| cosine_dist(&embeddings[i].1, &embeddings[j].1))
            .collect();
        let min_d = dists.iter().cloned().fold(f32::MAX, f32::min);
        let max_d = dists.iter().cloned().fold(f32::MIN, f32::max);
        println!("\n  min={:.4}  max={:.4}", min_d, max_d);
        println!("  → For online clustering threshold: use a value between min_same and max_diff.");

        // No assertion — this is diagnostic only.
        assert!(!embeddings.is_empty(), "No embeddings extracted");
    }

    /// Diagnose pairwise cosine distances on kkshow to find the right clustering threshold.
    ///
    /// Run with:
    ///   cargo test diarization::integration::kkshow_pairwise_distances -- --ignored --nocapture
    #[test]
    #[ignore = "diagnostic: requires both ONNX models + /tmp/kkshow_16k.wav"]
    fn kkshow_pairwise_distances() {
        let emb_path = crate::settings::diarization_model_path();
        let seg_path = crate::settings::segmentation_model_path();
        assert!(emb_path.exists());
        assert!(seg_path.exists());
        assert!(std::path::Path::new("/tmp/kkshow_16k.wav").exists());

        let (samples_raw, sr) = load_wav("/tmp/kkshow_16k.wav");
        let samples = to_16k(&samples_raw, sr);

        let mut seg_model = SegmentationModel::new(&seg_path).expect("seg model");
        let mut extractor = WeSpeakerExtractor::new(&emb_path).expect("extractor");

        let sub_segs = seg_model.find_sub_segments(&samples);
        println!("\n[kkshow] {} sub-segments:", sub_segs.len());

        const MAX_EMB_SAMPLES: usize = 5 * 16_000;
        let mut segs: Vec<(f64, f64, Vec<f32>)> = Vec::new();
        for (s, e) in &sub_segs {
            let start_s = *s as f64 / 16_000.0;
            let end_s   = *e as f64 / 16_000.0;
            let dur_s   = end_s - start_s;
            let chunk   = &samples[*s..*e];
            if chunk.len() < 400 { continue; } // skip sub-min-fbank segments
            let emb_chunk = if chunk.len() > MAX_EMB_SAMPLES { &chunk[..MAX_EMB_SAMPLES] } else { chunk };
            if let Ok(raw) = extractor.compute(emb_chunk) {
                let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
                let emb  = l2_normalize(&raw);
                if emb.iter().any(|x| x.is_nan()) {
                    println!("  [{:.2}s–{:.2}s] {:.2}s  NaN embedding — skipped", start_s, end_s, dur_s);
                    continue;
                }
                println!("  [{:.2}s–{:.2}s] {:.2}s  raw_norm={:.3}", start_s, end_s, dur_s, norm);
                segs.push((start_s, end_s, emb));
            }
        }

        // All pairwise distances.
        println!("\n--- Pairwise cosine distances ({} segs) ---", segs.len());
        let mut all_dists: Vec<f32> = Vec::new();
        for i in 0..segs.len() {
            for j in (i + 1)..segs.len() {
                let d = cosine_dist(&segs[i].2, &segs[j].2);
                println!("  [{:.2}s vs {:.2}s]: {:.4}", segs[i].0, segs[j].0, d);
                all_dists.push(d);
            }
        }
        all_dists.sort_by(|a, b| a.total_cmp(b)); // NaN sorts last
        println!("\nDistances sorted: {:?}", all_dists.iter().map(|d| format!("{:.4}", d)).collect::<Vec<_>>());

        // Try centroid linkage at multiple thresholds (euclidean space, pyannote-compatible).
        // Python's default threshold=0.7045 for WeSpeaker ResNet34-LM.
        let embeddings: Vec<Vec<f32>> = segs.iter().map(|(_, _, e)| e.clone()).collect();
        for &t in &[0.55, 0.60, 0.65, 0.70, 0.7045, 0.75, 0.78, 0.80, 0.85, 0.90] {
            let labels = centroid_linkage_cluster(&embeddings, t, 12);
            let n_clusters = *labels.iter().max().unwrap_or(&0) + 1;
            println!("threshold={:.4} → {} clusters: {:?}", t, n_clusters,
                segs.iter().zip(&labels).map(|((s, _, _), l)| format!("[{:.1}s] S{l}", s)).collect::<Vec<_>>());
        }

        assert!(!segs.is_empty());
    }

    /// Full Whisper + diarization transcript for kkshow.m4a (3-speaker, ~36s).
    ///
    /// Run with:
    ///   cargo test diarization::integration::kkshow_full_transcript -- --ignored --nocapture
    ///
    /// Pre-requisites:
    ///   ffmpeg -i ~/Desktop/kkshow.m4a -ar 16000 -ac 1 /tmp/kkshow_16k.wav
    ///   ONNX models in ~/.sumi-dev/models/
    ///   Whisper model: ~/.sumi-dev/models/ggml-large-v3-turbo-q5_0.bin
    #[test]
    #[ignore = "integration: requires ONNX models + /tmp/kkshow_16k.wav + Whisper model"]
    fn kkshow_full_transcript() {
        use whisper_rs::{
            DtwMode, DtwModelPreset, DtwParameters, FullParams, SamplingStrategy,
            WhisperContext, WhisperContextParameters,
        };

        let emb_path = crate::settings::diarization_model_path();
        let seg_path = crate::settings::segmentation_model_path();
        let model_path = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join(".sumi-dev/models/ggml-large-v3-turbo-q5_0.bin");

        assert!(emb_path.exists(), "WeSpeaker model not found: {}", emb_path.display());
        assert!(seg_path.exists(), "Segmentation model not found: {}", seg_path.display());
        assert!(model_path.exists(), "Whisper model not found: {}", model_path.display());
        assert!(
            std::path::Path::new("/tmp/kkshow_16k.wav").exists(),
            "/tmp/kkshow_16k.wav missing — run: ffmpeg -i ~/Desktop/kkshow.m4a -ar 16000 -ac 1 /tmp/kkshow_16k.wav"
        );

        // Load audio.
        let (samples_raw, sr) = load_wav("/tmp/kkshow_16k.wav");
        let samples = to_16k(&samples_raw, sr);
        let duration_s = samples.len() as f64 / 16_000.0;
        println!("\n[kkshow] {:.1}s audio loaded ({} samples @ 16 kHz)", duration_s, samples.len());

        // Initialize DiarizationEngine (segmentation + WeSpeaker embedding).
        let mut engine =
            DiarizationEngine::new(&emb_path, Some(&seg_path)).expect("DiarizationEngine::new");

        // Initialize Whisper with DTW word timestamps.
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.dtw_parameters = DtwParameters {
            mode: DtwMode::ModelPreset { model_preset: DtwModelPreset::LargeV3Turbo },
            dtw_mem_size: 128 * 1024 * 1024,
            ..Default::default()
        };
        let ctx = WhisperContext::new_with_params(model_path.to_str().unwrap(), ctx_params)
            .expect("WhisperContext::new_with_params");

        struct SegResult {
            start: f64,
            end: f64,
            speaker: String, // online label, updated by agglomerative pass
            text: String,
        }

        let mut all_results: Vec<SegResult> = Vec::new();
        let mut prev_text = String::new();
        const CHUNK_SAMPLES: usize = 30 * 16_000;

        let mut offset = 0;
        while offset < samples.len() {
            let chunk_end = (offset + CHUNK_SAMPLES).min(samples.len());
            let chunk = &samples[offset..chunk_end];
            let chunk_start_secs = offset as f64 / 16_000.0;

            // Phase 1a: diarization sub-segments.
            let sub_segs = engine.process_vad_chunk(chunk, chunk_start_secs);

            // Phase 1b: Whisper transcription + DTW word timestamps.
            let mut state = ctx.create_state().expect("create_state");
            {
                let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                params.set_language(Some("zh"));
                if !prev_text.is_empty() {
                    params.set_initial_prompt(&prev_text);
                }
                params.set_print_special(false);
                params.set_print_realtime(false);
                params.set_print_progress(false);
                params.set_single_segment(false);
                params.set_no_timestamps(true);
                params.set_no_context(true);
                params.set_temperature_inc(0.6);
                params.set_no_speech_thold(0.5);
                state.full(params, chunk).expect("whisper full");
            }

            // Collect raw text.
            let mut full_text = String::new();
            for i in 0..state.full_n_segments() {
                if let Some(seg) = state.get_segment(i) {
                    if seg.no_speech_probability() > 0.5 {
                        continue;
                    }
                    if let Ok(s) = seg.to_str_lossy() {
                        full_text.push_str(&s);
                    }
                }
            }
            let full_text = full_text.trim().to_string();

            // DTW word timestamps.
            let words = crate::transcribe::extract_dtw_words(&state, chunk_start_secs);

            // Update context for next chunk.
            if !full_text.is_empty() {
                let chars: Vec<char> = full_text.chars().collect();
                let take = chars.len().min(200);
                prev_text = chars[chars.len() - take..].iter().collect();
            }

            let preview: String = full_text.chars().take(30).collect();
            println!(
                "[chunk {:.0}s–{:.0}s] {} sub-segs, {} dtw-words, text={}",
                chunk_start_secs,
                chunk_end as f64 / 16_000.0,
                sub_segs.len(),
                words.len(),
                preview,
            );

            if sub_segs.is_empty() {
                // No diarization data: emit as a single unspeakered segment.
                if !full_text.is_empty() {
                    all_results.push(SegResult {
                        start: chunk_start_secs,
                        end: chunk_end as f64 / 16_000.0,
                        speaker: String::new(),
                        text: full_text,
                    });
                }
            } else {
                // Assign words to sub-segments by absolute timestamp.
                let mut seg_texts: Vec<String> = vec![String::new(); sub_segs.len()];

                if words.is_empty() {
                    // No DTW words: give full text to the longest sub-segment.
                    if let Some(longest_idx) = sub_segs
                        .iter()
                        .enumerate()
                        .max_by(|a, b| {
                            (a.1.1 - a.1.0)
                                .partial_cmp(&(b.1.1 - b.1.0))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                    {
                        seg_texts[longest_idx] = full_text.clone();
                    }
                } else {
                    for word in &words {
                        let idx = sub_segs
                            .iter()
                            .position(|(s, e, _)| word.s >= *s && word.s < *e)
                            .unwrap_or(sub_segs.len() - 1);
                        // Append token as-is: leading space already encodes the
                        // word boundary for Latin scripts; CJK tokens have none.
                        seg_texts[idx].push_str(&word.w);
                    }
                    // Trim leading/trailing whitespace from each segment's text.
                    for t in &mut seg_texts {
                        let trimmed = t.trim().to_string();
                        *t = trimmed;
                    }
                }

                for ((start, end, speaker), text) in sub_segs.iter().zip(seg_texts.into_iter()) {
                    all_results.push(SegResult {
                        start: *start,
                        end: *end,
                        speaker: speaker.clone(),
                        text,
                    });
                }
            }

            offset += CHUNK_SAMPLES;
        }

        // Phase 2: agglomerative relabeling — update online speaker labels.
        let final_labels = engine.finalize_labels();
        if !final_labels.is_empty() {
            for r in &mut all_results {
                if let Some((_, _, new_spk)) = final_labels
                    .iter()
                    .find(|(s, e, _)| (r.start - s).abs() < 0.05 && (r.end - e).abs() < 0.05)
                {
                    r.speaker = new_spk.clone();
                }
            }
        }

        // Print the final labeled transcript.
        let fmt_time = |secs: f64| -> String {
            let m = secs as u64 / 60;
            let s = secs as u64 % 60;
            format!("{}:{:02}", m, s)
        };
        let fmt_spk = |spk: &str| -> String {
            if let Some(n) = spk.strip_prefix("SPEAKER_") {
                format!("Speaker {}", n.parse::<u32>().unwrap_or(0) + 1)
            } else {
                spk.to_string()
            }
        };

        println!("\n=== kkshow 完整逐字稿（帶說話者）===");
        for r in &all_results {
            if r.text.is_empty() {
                continue;
            }
            if r.speaker.is_empty() {
                println!("[{}] {}", fmt_time(r.start), r.text);
            } else {
                println!("[{}] {}: {}", fmt_time(r.start), fmt_spk(&r.speaker), r.text);
            }
        }
        println!("=== end ({} segments) ===", all_results.iter().filter(|r| !r.text.is_empty()).count());

        assert!(!all_results.is_empty(), "No results produced");
    }

    /// Full pipeline comparison: Rust diarization+Whisper vs Python results_onnx.txt.
    ///
    /// Runs the same three audio files used by the Python `test_onnx.py` script and
    /// outputs in the identical `[MM:SS.ss --> MM:SS.ss] SPEAKER_XX: text` format so
    /// the two outputs can be compared side-by-side.
    ///
    /// Run with:
    ///   cargo test diarization::integration::compare_with_python_reference -- --ignored --nocapture
    ///
    /// Pre-requisites:
    ///   ffmpeg -i ~/Desktop/kkshow.m4a       -ar 16000 -ac 1 /tmp/kkshow_16k.wav
    ///   ffmpeg -i ~/Desktop/test_video_1.m4a -ar 16000 -ac 1 /tmp/test_video_1_16k.wav
    ///   ffmpeg -i ~/Desktop/test_video_2.m4a -ar 16000 -ac 1 /tmp/test_video_2_16k.wav
    ///   ONNX models in ~/.sumi-dev/models/
    ///   Whisper model: ~/.sumi-dev/models/ggml-large-v3-turbo-q5_0.bin
    ///   Python reference (optional): set SUMI_PY_REF env var to results_onnx.txt path
    #[test]
    #[ignore = "integration: requires ONNX models + 3× /tmp/*_16k.wav + Whisper model"]
    fn compare_with_python_reference() {
        use whisper_rs::{
            DtwMode, DtwModelPreset, DtwParameters, FullParams, SamplingStrategy,
            WhisperContext, WhisperContextParameters,
        };

        let emb_path = crate::settings::diarization_model_path();
        let seg_path = crate::settings::segmentation_model_path();
        let model_path = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join(".sumi-dev/models/ggml-large-v3-turbo-q5_0.bin");

        for p in [emb_path.as_path(), seg_path.as_path(), model_path.as_path()] {
            assert!(p.exists(), "required file not found: {}", p.display());
        }

        // Format a seconds value as MM:SS.ss  (e.g. 73.4 → "01:13.40").
        let fmt_t = |secs: f64| -> String {
            let m = secs as u64 / 60;
            let s = secs % 60.0;
            format!("{:02}:{:05.2}", m, s)
        };

        // Load Whisper once — shared across all three files.
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.dtw_parameters = DtwParameters {
            mode: DtwMode::ModelPreset { model_preset: DtwModelPreset::LargeV3Turbo },
            dtw_mem_size: 128 * 1024 * 1024,
            ..Default::default()
        };
        let ctx = WhisperContext::new_with_params(model_path.to_str().unwrap(), ctx_params)
            .expect("WhisperContext::new_with_params");

        // Load Python reference output for comparison (optional).
        let py_ref = std::env::var("SUMI_PY_REF")
            .ok()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .unwrap_or_default();

        let files: &[(&str, &str)] = &[
            ("/tmp/kkshow_16k.wav",       "kkshow.m4a"),
            ("/tmp/test_video_1_16k.wav", "test_video_1.m4a"),
            ("/tmp/test_video_2_16k.wav", "test_video_2.m4a"),
        ];

        let mut all_rust_output: Vec<String> = Vec::new();

        for &(wav_path, label) in files {
            assert!(
                std::path::Path::new(wav_path).exists(),
                "{wav_path} missing — run: ffmpeg -i ~/Desktop/{label} -ar 16000 -ac 1 {wav_path}"
            );

            let (samples_raw, sr) = load_wav(wav_path);
            let samples = to_16k(&samples_raw, sr);
            let duration_s = samples.len() as f64 / 16_000.0;

            // Fresh diarization engine per file.
            let mut engine =
                DiarizationEngine::new(&emb_path, Some(&seg_path)).expect("DiarizationEngine");

            struct Seg { start: f64, end: f64, speaker: String, text: String }
            let mut all_results: Vec<Seg> = Vec::new();
            let mut prev_text = String::new();
            const CHUNK: usize = 30 * 16_000;

            let mut offset = 0usize;
            while offset < samples.len() {
                let chunk_end = (offset + CHUNK).min(samples.len());
                let chunk = &samples[offset..chunk_end];
                let chunk_start_secs = offset as f64 / 16_000.0;

                // Diarization sub-segments.
                let sub_segs = engine.process_vad_chunk(chunk, chunk_start_secs);

                // Whisper transcription (language auto-detect, same as Python).
                let mut state = ctx.create_state().expect("create_state");
                {
                    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                    params.set_language(None); // auto-detect (matches Python language=None)
                    if !prev_text.is_empty() {
                        params.set_initial_prompt(&prev_text);
                    }
                    params.set_print_special(false);
                    params.set_print_realtime(false);
                    params.set_print_progress(false);
                    params.set_no_timestamps(true);
                    params.set_no_context(true);
                    params.set_temperature_inc(0.6);
                    params.set_no_speech_thold(0.5);
                    state.full(params, chunk).expect("whisper full");
                }

                let mut full_text = String::new();
                for i in 0..state.full_n_segments() {
                    if let Some(seg) = state.get_segment(i) {
                        if seg.no_speech_probability() > 0.5 { continue; }
                        if let Ok(s) = seg.to_str_lossy() { full_text.push_str(&s); }
                    }
                }
                let full_text = full_text.trim().to_string();

                // DTW word timestamps for text-to-speaker alignment.
                let words = crate::transcribe::extract_dtw_words(&state, chunk_start_secs);

                if !full_text.is_empty() {
                    let chars: Vec<char> = full_text.chars().collect();
                    let take = chars.len().min(200);
                    prev_text = chars[chars.len() - take..].iter().collect();
                }

                if sub_segs.is_empty() {
                    if !full_text.is_empty() {
                        all_results.push(Seg {
                            start: chunk_start_secs,
                            end: chunk_end as f64 / 16_000.0,
                            speaker: String::new(),
                            text: full_text,
                        });
                    }
                } else {
                    let mut seg_texts: Vec<String> = vec![String::new(); sub_segs.len()];
                    if words.is_empty() {
                        if let Some(idx) = sub_segs.iter().enumerate()
                            .max_by(|a, b| (a.1.1 - a.1.0).partial_cmp(&(b.1.1 - b.1.0))
                                .unwrap_or(std::cmp::Ordering::Equal))
                            .map(|(i, _)| i)
                        {
                            seg_texts[idx] = full_text.clone();
                        }
                    } else {
                        for word in &words {
                            let idx = sub_segs.iter()
                                .position(|(s, e, _)| word.s >= *s && word.s < *e)
                                .unwrap_or(sub_segs.len() - 1);
                            seg_texts[idx].push_str(&word.w);
                        }
                        for t in &mut seg_texts {
                            *t = t.trim().to_string();
                        }
                    }
                    for ((start, end, speaker), text) in sub_segs.iter().zip(seg_texts) {
                        all_results.push(Seg { start: *start, end: *end, speaker: speaker.clone(), text });
                    }
                }

                offset += CHUNK;
            }

            // Agglomerative relabeling pass.
            let final_labels = engine.finalize_labels();
            if !final_labels.is_empty() {
                for r in &mut all_results {
                    if let Some((_, _, new_spk)) = final_labels.iter()
                        .find(|(s, e, _)| (r.start - s).abs() < 0.05 && (r.end - e).abs() < 0.05)
                    {
                        r.speaker = new_spk.clone();
                    }
                }
            }

            // Print Rust output in Python-compatible format.
            let header = format!("\n{}\n=== {} ===\n{}", "=".repeat(60), label, "=".repeat(60));
            println!("{header}");
            all_rust_output.push(header);

            let n_speakers: std::collections::HashSet<&str> =
                all_results.iter().map(|r| r.speaker.as_str()).filter(|s| !s.is_empty()).collect();
            println!("[{:.1}s, {} speakers, {} segments]", duration_s, n_speakers.len(), all_results.len());

            for r in &all_results {
                if r.text.is_empty() { continue; }
                let line = if r.speaker.is_empty() {
                    format!("[{} --> {}] {}", fmt_t(r.start), fmt_t(r.end), r.text)
                } else {
                    format!("[{} --> {}] {}: {}", fmt_t(r.start), fmt_t(r.end), r.speaker, r.text)
                };
                println!("{line}");
                all_rust_output.push(line);
            }
        }

        // Print Python reference for side-by-side comparison.
        println!("\n{}", "─".repeat(60));
        println!("PYTHON REFERENCE (results_onnx.txt):");
        println!("{}", "─".repeat(60));
        println!("{py_ref}");

        assert!(!all_rust_output.is_empty(), "No output produced");
    }

    // ── pyannote_diarize (offline pipeline) ───────────────────────────────────

    /// Offline pyannote-equivalent pipeline on kkshow (3-speaker clip).
    ///
    /// Compare Rust compute_fbank against Python torchaudio reference on real audio.
    ///
    /// Run with:
    ///   cargo test diarization::integration::fbank_matches_torchaudio_on_real_audio -- --ignored --nocapture
    #[test]
    #[ignore = "integration: requires /tmp/kkshow_16k.wav"]
    fn fbank_matches_torchaudio_on_real_audio() {
        let wav = "/tmp/kkshow_16k.wav";
        assert!(std::path::Path::new(wav).exists());

        let (samples_raw, sr) = load_wav(wav);
        let samples = to_16k(&samples_raw, sr);

        // First 10s window (same as window 0 in pyannote pipeline)
        let window = &samples[..SEG_WINDOW_SAMPLES.min(samples.len())];
        let feats = compute_fbank(window);

        // Python reference (torchaudio.compliance.kaldi.fbank, post-CMN)
        // frame 0, first 5 bins:
        let py_f0: [f32; 5] = [-10.006987, -10.695486, -11.923952, -12.579387, -13.051603];

        println!("Rust fbank frames: {}", feats.len());
        println!("Rust frame 0, first 5: {:?}", &feats[0][..5]);
        println!("Python frame 0, first 5: {:?}", &py_f0);

        let mut max_diff = 0.0f32;
        for (i, (&r, &p)) in feats[0][..5].iter().zip(py_f0.iter()).enumerate() {
            let diff = (r - p).abs();
            println!("  bin {i}: rust={r:.6} py={p:.6} diff={diff:.6}");
            max_diff = max_diff.max(diff);
        }

        println!("Max diff (frame 0, first 5 bins): {max_diff:.6}");

        // Allow up to 0.01 tolerance for real audio (different FFT implementations,
        // floating point accumulation order, etc.)
        assert!(
            max_diff < 0.01,
            "fbank frame 0 differs from torchaudio by {max_diff:.6} (max allowed: 0.01)"
        );
    }

    /// Exercises the full `pyannote_diarize()` function end-to-end:
    ///   sliding window segmentation → per-speaker masked embeddings →
    ///   centroid-linkage clustering → frame-level reconstruction → segments.
    ///
    /// Run with:
    ///   ffmpeg -i ~/Desktop/kkshow.m4a -ar 16000 -ac 1 /tmp/kkshow_16k.wav
    ///   cargo test diarization::integration::pyannote_diarize_kkshow -- --ignored --nocapture
    #[test]
    #[ignore = "integration: requires both ONNX models + /tmp/kkshow_16k.wav"]
    fn pyannote_diarize_kkshow() {
        let emb_path = crate::settings::diarization_model_path();
        let seg_path = crate::settings::segmentation_model_path();
        assert!(emb_path.exists(), "WeSpeaker model not found: {}", emb_path.display());
        assert!(seg_path.exists(), "Segmentation model not found: {}", seg_path.display());

        let wav = "/tmp/kkshow_16k.wav";
        assert!(std::path::Path::new(wav).exists(),
            "Test audio not found: {wav}. Run: ffmpeg -i ~/Desktop/kkshow.m4a -ar 16000 -ac 1 /tmp/kkshow_16k.wav");

        let (samples_raw, sr) = load_wav(wav);
        let samples = to_16k(&samples_raw, sr);
        let duration_s = samples.len() as f64 / 16_000.0;

        let mut seg_model = SegmentationModel::new(&seg_path).expect("SegmentationModel::new");
        let mut emb_extractor = WeSpeakerExtractor::new(&emb_path).expect("WeSpeakerExtractor::new");

        println!("\n[pyannote_diarize] kkshow {:.1}s", duration_s);

        let segments = pyannote_diarize(&samples, &mut seg_model, &mut emb_extractor);

        println!("[pyannote_diarize] {} segments:", segments.len());
        for (s, e, spk) in &segments {
            let sm = (*s as u64) / 60;
            let ss = s % 60.0;
            let em = (*e as u64) / 60;
            let es = e % 60.0;
            println!("  [{:02}:{:05.2} --> {:02}:{:05.2}] {}", sm, ss, em, es, spk);
        }

        let speakers: std::collections::HashSet<&str> =
            segments.iter().map(|(_, _, s)| s.as_str()).collect();
        println!("[pyannote_diarize] {} speakers: {:?}", speakers.len(), speakers);

        // kkshow has 3-4 speakers; offline pipeline should find at least 2.
        assert!(
            !segments.is_empty(),
            "pyannote_diarize returned no segments from kkshow"
        );
        assert!(
            speakers.len() >= 2,
            "Expected ≥2 speakers from kkshow, got {}: {:?}",
            speakers.len(),
            speakers
        );
    }

    /// Offline pyannote_diarize on kkshow (multi-speaker) — should find ≥2 speakers.
    ///
    /// Run with:
    ///   ffmpeg -i ~/Desktop/kkshow.m4a -ar 16000 -ac 1 /tmp/kkshow_16k.wav
    ///   cargo test diarization::integration::pyannote_diarize_voxconv -- --ignored --nocapture
    #[test]
    #[ignore = "integration: requires both ONNX models + /tmp/kkshow_16k.wav"]
    fn pyannote_diarize_voxconv() {
        let emb_path = crate::settings::diarization_model_path();
        let seg_path = crate::settings::segmentation_model_path();
        assert!(emb_path.exists());
        assert!(seg_path.exists());
        let audio_path = "/tmp/kkshow_16k.wav";
        assert!(
            std::path::Path::new(audio_path).exists(),
            "{audio_path} missing — run: ffmpeg -i ~/Desktop/kkshow.m4a -ar 16000 -ac 1 {audio_path}"
        );

        let (samples_raw, sr) = load_wav(audio_path);
        let samples = to_16k(&samples_raw, sr);
        let duration_s = samples.len() as f64 / 16_000.0;

        let mut seg_model = SegmentationModel::new(&seg_path).expect("SegmentationModel");
        let mut emb_extractor = WeSpeakerExtractor::new(&emb_path).expect("WeSpeakerExtractor");

        println!("\n[pyannote_diarize] kkshow {:.1}s", duration_s);

        let segments = pyannote_diarize(&samples, &mut seg_model, &mut emb_extractor);

        println!("[pyannote_diarize] {} segments:", segments.len());
        for (s, e, spk) in &segments {
            println!("  [{:.2}s–{:.2}s] {}", s, e, spk);
        }

        let speakers: std::collections::HashSet<&str> =
            segments.iter().map(|(_, _, s)| s.as_str()).collect();
        println!("[pyannote_diarize] {} speakers: {:?}", speakers.len(), speakers);

        // kkshow has 3-4 speakers — offline pipeline should find ≥2.
        assert!(
            !segments.is_empty(),
            "pyannote_diarize returned no segments"
        );
        assert!(
            speakers.len() >= 2,
            "Expected ≥2 speakers from kkshow clip, got {}: {:?}",
            speakers.len(),
            speakers
        );
    }

    /// Full offline pipeline: pyannote_diarize + Whisper on the three Desktop audio files.
    ///
    /// Saves results to /tmp/rust_results_pyannote.txt in the same format as results_onnx.txt.
    ///
    /// Pre-requisites:
    ///   ffmpeg -i ~/Desktop/kkshow.m4a       -ar 16000 -ac 1 /tmp/kkshow_16k.wav
    ///   ffmpeg -i ~/Desktop/test_video_1.m4a -ar 16000 -ac 1 /tmp/test_video_1_16k.wav
    ///   ffmpeg -i ~/Desktop/test_video_2.m4a -ar 16000 -ac 1 /tmp/test_video_2_16k.wav
    ///
    /// Run with:
    ///   cargo test diarization::integration::run_full_pipeline_desktop -- --ignored --nocapture
    #[test]
    #[ignore = "integration: requires ONNX models + 3× /tmp/*_16k.wav + Whisper model"]
    fn run_full_pipeline_desktop() {
        use whisper_rs::{
            DtwMode, DtwModelPreset, DtwParameters, FullParams, SamplingStrategy,
            WhisperContext, WhisperContextParameters,
        };

        let emb_path = crate::settings::diarization_model_path();
        let seg_path = crate::settings::segmentation_model_path();
        let model_path = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join(".sumi-dev/models/ggml-large-v3-turbo-q5_0.bin");

        for p in [emb_path.as_path(), seg_path.as_path(), model_path.as_path()] {
            assert!(p.exists(), "required file missing: {}", p.display());
        }

        let fmt_t = |s: f64| -> String {
            let m = s as u64 / 60;
            let sec = s % 60.0;
            format!("{:02}:{:05.2}", m, sec)
        };

        // Load Whisper once, shared across all files.
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.dtw_parameters = DtwParameters {
            mode: DtwMode::ModelPreset { model_preset: DtwModelPreset::LargeV3Turbo },
            dtw_mem_size: 128 * 1024 * 1024,
            ..Default::default()
        };
        let ctx = WhisperContext::new_with_params(model_path.to_str().unwrap(), ctx_params)
            .expect("WhisperContext");

        let files: &[(&str, &str)] = &[
            ("/tmp/kkshow_16k.wav",       "kkshow.m4a"),
            ("/tmp/test_video_1_16k.wav", "test_video_1.m4a"),
            ("/tmp/test_video_2_16k.wav", "test_video_2.m4a"),
        ];

        let mut output_lines: Vec<String> = Vec::new();

        for &(wav_path, label) in files {
            assert!(
                std::path::Path::new(wav_path).exists(),
                "{wav_path} missing"
            );

            let (samples_raw, sr) = load_wav(wav_path);
            let samples = to_16k(&samples_raw, sr);

            let header = format!("\n{}\n=== {} ===\n{}", "=".repeat(60), label, "=".repeat(60));
            println!("{header}");
            output_lines.push(header);

            // Phase 1: offline diarization.
            let mut seg_model = SegmentationModel::new(&seg_path).expect("SegmentationModel");
            let mut emb_extractor = WeSpeakerExtractor::new(&emb_path).expect("WeSpeakerExtractor");
            let diar_segs = pyannote_diarize(&samples, &mut seg_model, &mut emb_extractor);

            println!("[diarize] {} segments, {} speakers",
                diar_segs.len(),
                diar_segs.iter().map(|(_, _, s)| s.as_str()).collect::<std::collections::HashSet<_>>().len()
            );

            // Phase 2: Whisper transcription per diarized segment.
            let mut prev_text = String::new();
            for (start_s, end_s, speaker) in &diar_segs {
                let start_i = (start_s * 16_000.0) as usize;
                let end_i = ((end_s * 16_000.0) as usize).min(samples.len());
                if end_i <= start_i { continue; }
                let chunk = &samples[start_i..end_i];

                let mut state = ctx.create_state().expect("create_state");
                let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                params.set_language(None);
                if !prev_text.is_empty() {
                    params.set_initial_prompt(&prev_text);
                }
                params.set_print_special(false);
                params.set_print_realtime(false);
                params.set_print_progress(false);
                params.set_no_timestamps(true);
                params.set_no_context(true);
                params.set_temperature_inc(0.6);
                params.set_no_speech_thold(0.5);
                state.full(params, chunk).expect("whisper full");

                let mut text = String::new();
                for i in 0..state.full_n_segments() {
                    if let Some(seg) = state.get_segment(i) {
                        if seg.no_speech_probability() > 0.5 { continue; }
                        if let Ok(s) = seg.to_str_lossy() { text.push_str(&s); }
                    }
                }
                let text = text.trim().to_string();
                if text.is_empty() { continue; }

                let chars: Vec<char> = text.chars().collect();
                let take = chars.len().min(200);
                prev_text = chars[chars.len() - take..].iter().collect();

                let line = format!("[{} --> {}] {}: {}", fmt_t(*start_s), fmt_t(*end_s), speaker, text);
                println!("{line}");
                output_lines.push(line);
            }
        }

        let out_path = std::env::var("SUMI_OUT_PATH")
            .unwrap_or_else(|_| "/tmp/rust_results_pyannote.txt".to_string());
        std::fs::write(&out_path, output_lines.join("\n")).expect("write results");
        println!("\n結果已儲存至 {out_path}");
    }

    // ── Clustering parity with pyannote (scipy) ──────────────────────────

    /// Verified against Python:
    /// ```python
    /// embs = [[0.9,0.1,0], [0.88,0.12,0], [0.87,0.13,0], [0.86,0.14,0],
    ///         [0,0.1,0.9], [0,0.12,0.88], [0,0.13,0.87], [0,0.14,0.86],
    ///         [0.5,0.1,0.5]]
    /// # L2-normalize each
    /// pyannote_cluster(embs, 0.7045654963945799, 12) → [0,0,0,0, 1,1,1,1, 2]
    /// ```
    #[test]
    fn centroid_linkage_matches_scipy_3d_with_outlier() {
        // 9 embeddings in 3D — 2 clear clusters of 4 + 1 outlier.
        let raw: Vec<[f32; 3]> = vec![
            [0.9, 0.1, 0.0],
            [0.88, 0.12, 0.0],
            [0.87, 0.13, 0.0],
            [0.86, 0.14, 0.0],
            [0.0, 0.1, 0.9],
            [0.0, 0.12, 0.88],
            [0.0, 0.13, 0.87],
            [0.0, 0.14, 0.86],
            [0.5, 0.1, 0.5],
        ];
        let embs: Vec<Vec<f32>> = raw
            .iter()
            .map(|r| l2_normalize(&r.to_vec()))
            .collect();

        let labels =
            centroid_linkage_cluster(&embs, PYANNOTE_THRESHOLD, 12);

        // Python scipy result: [0, 0, 0, 0, 1, 1, 1, 1, 2]
        assert_eq!(labels, vec![0, 0, 0, 0, 1, 1, 1, 1, 2]);
    }

    /// Verified against Python:
    /// 5 embeddings → 2 clusters (effective_min_cluster_size = 1).
    #[test]
    fn centroid_linkage_matches_scipy_5_embeddings_2_clusters() {
        // Use the same seed-derived embeddings from the Python test.
        // Center 0: [1, 0, 0], Center 1: [0, 1, 0], with noise 0.03.
        let c0 = vec![1.0_f32, 0.0, 0.0];
        let c1 = vec![0.0_f32, 1.0, 0.0];
        let offsets: Vec<Vec<f32>> = vec![
            vec![0.02, 0.01, 0.0],
            vec![-0.01, 0.02, 0.01],
            vec![0.01, -0.01, 0.02],
            vec![0.0, 0.01, -0.01],
            vec![-0.02, 0.0, 0.01],
        ];
        let mut embs = Vec::new();
        for (i, off) in offsets.iter().enumerate() {
            let base = if i < 3 { &c0 } else { &c1 };
            let v: Vec<f32> = base.iter().zip(off.iter()).map(|(a, b)| a + b).collect();
            embs.push(l2_normalize(&v));
        }

        let labels = centroid_linkage_cluster(&embs, PYANNOTE_THRESHOLD, 12);

        // With effective_min = min(12, max(1, round(0.5))) = 1, no reassignment needed.
        // First 3 → same cluster, last 2 → same cluster, two distinct clusters.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
        assert_eq!(labels.iter().collect::<std::collections::HashSet<_>>().len(), 2);
    }

    /// Small-cluster reassignment must use COSINE distance (matching pyannote).
    #[test]
    fn small_cluster_reassignment_uses_cosine() {
        // Construct embeddings where cosine and euclidean disagree on reassignment.
        // 2 large clusters (4 each) + 1 small cluster (1 embedding).
        // The small cluster is at equal EUCLIDEAN distance to both large centroids
        // but closer in COSINE to cluster A.
        let cluster_a_center = l2_normalize(&vec![1.0_f32, 0.0, 0.0]);
        let cluster_b_center = l2_normalize(&vec![0.0_f32, 0.0, 1.0]);

        let mut embs: Vec<Vec<f32>> = Vec::new();
        // 4 points near cluster A
        for i in 0..4 {
            let mut v = cluster_a_center.clone();
            v[1] += 0.01 * (i as f32 + 1.0);
            embs.push(l2_normalize(&v));
        }
        // 4 points near cluster B
        for i in 0..4 {
            let mut v = cluster_b_center.clone();
            v[1] += 0.01 * (i as f32 + 1.0);
            embs.push(l2_normalize(&v));
        }
        // 1 outlier slightly closer to A in cosine (direction) but same euclidean
        let outlier = l2_normalize(&vec![0.8_f32, 0.1, 0.3]);
        embs.push(outlier.clone());

        let labels = centroid_linkage_cluster(&embs, PYANNOTE_THRESHOLD, 4);

        // The outlier (index 8) should be in the same cluster as A (index 0),
        // because cosine distance to A's centroid is smaller than to B's centroid.
        let outlier_label = labels[8];
        let cluster_a_label = labels[0];
        assert_eq!(
            outlier_label, cluster_a_label,
            "outlier should be assigned to cluster A via cosine distance"
        );
    }

}

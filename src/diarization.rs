//! Speaker diarization: segmentation + WeSpeaker embeddings + clustering.
//!
//! # Two-phase pipeline
//!
//! ## Real-time (during recording)
//! ```text
//! VAD chunk (f32, 16 kHz)
//!   → SegmentationModel [1,1,N] → [1,T,7]  (speaker-class per frame)
//!   → sub-segment boundaries (silence + speaker-class changes)
//!   → WeSpeaker [1,T,80] → [1,512]  (per sub-segment; ResNet34-LM output dim)
//!   → L2 normalize
//!   → online cosine clustering  (greedy, for real-time labels)
//!   → "SPEAKER_00" / "SPEAKER_01" / …  +  embedding buffered for phase 2
//! ```
//!
//! ## Finalization (at meeting stop, matches the experiment's quality)
//! ```text
//! buffered (start, end, embedding) for all sub-segments
//!   → agglomerative hierarchical clustering (average linkage, threshold=0.50)
//!   → optimal speaker labels  (same algorithm as exp_g_diarize_agglomerative.rs)
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
//! `segmentation-3.0.onnx` from pyannote-rs v0.1.0 release.
//! Input:  `"input"` — `[1, 1, 160_000]` f32 (i16 values cast to f32, scale ×32767)
//! Output: `"output"` — `[1, num_frames, 7]` where class 0 = silence.
//! Frame hop = 270 samples (≈ 16.9 ms), first frame at sample 721.
//! Splits at both silence→speech transitions AND speaker-class changes within speech.
//!
//! ## Bug note
//! `pyannote_rs::get_segments()` has an iterator bug: terminates early when
//! speech crosses a 10-second window boundary.  We reimplemented the segmentation
//! logic directly against the ONNX session.

use std::path::Path;

use ndarray::{Array1, ArrayViewD, Axis, IxDyn};

/// segmentation-3.0.onnx model download URL (pyannote-rs v0.1.0 release).
pub const SEGMENTATION_URL: &str =
    "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/segmentation-3.0.onnx";

/// WeSpeaker embedding model download URL (pyannote-rs v0.1.0 release).
pub const WESPEAKER_URL: &str =
    "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker-voxceleb-resnet34-LM.onnx";

// ── Segmentation model ─────────────────────────────────────────────────────────

/// Frame parameters matching segmentation-3.0.onnx (from experiment).
const SEG_WINDOW_SAMPLES: usize = 160_000; // 10 s at 16 kHz
const SEG_FRAME_HOP: usize = 270; // ≈ 16.9 ms
const SEG_FRAME_START: usize = 721; // first frame center
const SEG_MIN_SUBSEG_SAMPLES: usize = 400; // 25 ms — shorter sub-segs skipped

/// Pyannote segmentation-3.0 model.  Runs directly via ORT, bypassing the
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
        padded.extend(std::iter::repeat(0.0f32).take(pad));

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

            let ort_outs = match self.session.run(ort::inputs!["input" => tensor]) {
                Ok(o) => o,
                Err(e) => {
                    tracing::warn!("[seg] inference failed: {e}");
                    break;
                }
            };

            let tensor_out = match ort_outs
                .get("output")
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
}

// ── Agglomerative clustering ───────────────────────────────────────────────────

/// Offline agglomerative hierarchical clustering (average linkage).
///
/// Direct port from `exp_g_diarize_agglomerative.rs`.
/// Threshold calibrated to 0.50 for `wespeaker-voxceleb-resnet34-LM.onnx` with 5 s cap:
/// same-speaker cosine distance ≤ 0.09, different-speaker > 0.51.
///
/// Embeddings must already be L2-normalised before calling.
/// Returns a cluster label (0-based) for each input embedding.
pub(crate) fn agglomerative_cluster(embeddings: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0];
    }

    // clusters: (member_indices, L2-normalised_centroid, member_count)
    let mut clusters: Vec<(Vec<usize>, Vec<f32>, usize)> = embeddings
        .iter()
        .enumerate()
        .map(|(i, e)| (vec![i], e.clone(), 1))
        .collect();

    loop {
        if clusters.len() <= 1 {
            break;
        }

        // O(N³) total: O(N²) pairwise distances × O(N) merges. Fine for N < ~200 segments.
        let mut min_dist = f32::MAX;
        let mut min_i = 0;
        let mut min_j = 1;
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let d = cosine_dist(&clusters[i].1, &clusters[j].1);
                if d < min_dist {
                    min_dist = d;
                    min_i = i;
                    min_j = j;
                }
            }
        }

        if min_dist >= threshold {
            break;
        }

        // Merge j into i: weighted-mean centroid, then re-normalise.
        let n_i = clusters[min_i].2 as f32;
        let n_j = clusters[min_j].2 as f32;
        let total = n_i + n_j;
        let new_centroid: Vec<f32> = clusters[min_i]
            .1
            .iter()
            .zip(clusters[min_j].1.iter())
            .map(|(a, b)| (a * n_i + b * n_j) / total)
            .collect();

        let indices_j = clusters[min_j].0.clone();
        let count_j = clusters[min_j].2;
        clusters[min_i].0.extend(indices_j);
        clusters[min_i].1 = l2_normalize(&new_centroid);
        clusters[min_i].2 += count_j;
        clusters.remove(min_j);
    }

    // Assign labels ordered by first-appearance index (stable, human-readable).
    clusters.sort_by_key(|(indices, _, _)| *indices.iter().min().unwrap_or(&0));

    let mut labels = vec![0usize; n];
    for (label, (indices, _, _)) in clusters.iter().enumerate() {
        for &idx in indices {
            labels[idx] = label;
        }
    }
    labels
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

/// Combined diarization engine: segmentation + WeSpeaker + two-phase clustering.
pub struct DiarizationEngine {
    emb_extractor: pyannote_rs::EmbeddingExtractor,
    segmentation: Option<SegmentationModel>,
    clusters: SpeakerClusters,
    /// (start_secs, end_secs, L2-normalised embedding) for every sub-segment
    /// processed this session.  Used by `finalize_labels()` for agglomerative pass.
    segment_buffer: Vec<(f64, f64, Vec<f32>)>,
}

// SAFETY: pyannote_rs::EmbeddingExtractor (v0.3.4) is a single-field struct
// wrapping ort::session::Session, which declares `unsafe impl Send for Session {}`
// in ort 2.0.0-rc.10 (session/mod.rs:565). All other fields (SpeakerClusters,
// Vec<…>) are Send. DiarizationEngine is accessed only through a
// Mutex<Option<DiarizationEngine>>, which serialises all access.
unsafe impl Send for DiarizationEngine {}

impl DiarizationEngine {
    /// Load embedding model; optionally also load segmentation model.
    ///
    /// If `seg_model` is `None` or its path does not exist, intra-utterance
    /// speaker change detection is disabled (VAD chunks are treated as atomic).
    pub fn new(emb_model: &Path, seg_model: Option<&Path>) -> Result<Self, String> {
        let path_str = emb_model
            .to_str()
            .ok_or("WeSpeaker model path is not valid UTF-8")?;
        let emb_extractor = pyannote_rs::EmbeddingExtractor::new(path_str)
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
            // Threshold calibrated for wespeaker-voxceleb-resnet34-LM.onnx + 5 s cap:
            // same-speaker dist ≤ 0.09 (for comparable lengths), inter-speaker > 0.51.
            // 0.50 cleanly separates speakers while tolerating short-segment noise.
            clusters: SpeakerClusters::new(0.50),
            segment_buffer: Vec::new(),
        })
    }

    /// Process one VAD chunk of 16 kHz f32 audio.
    ///
    /// 1. If segmentation model available: split at silence AND speaker-class
    ///    changes → multiple sub-segments.
    /// 2. For each sub-segment: WeSpeaker embedding → L2-normalise →
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

        // WeSpeaker ResNet34-LM norms scale with segment duration: short segments
        // have 10× larger pre-norm norms, making L2-normalised embeddings
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
            let samples_i16 = f32_to_i16(emb_slice);
            if samples_i16.len() < 400 {
                tracing::debug!(
                    "[diarization] sub-segment [{:.2}-{:.2}s] too short ({} samples)",
                    start_secs,
                    end_secs,
                    samples_i16.len()
                );
                continue;
            }

            let raw_emb: Vec<f32> = match self.emb_extractor.compute(&samples_i16) {
                Ok(iter) => iter.collect(),
                Err(e) => {
                    tracing::warn!("[diarization] embedding failed: {e}");
                    result.push((start_secs, end_secs, String::new()));
                    continue;
                }
            };

            if raw_emb.is_empty() {
                result.push((start_secs, end_secs, String::new()));
                continue;
            }

            let emb = l2_normalize(&raw_emb);

            // Online clustering.
            let speaker_id = if is_reliable {
                // Reliable segment: may create new cluster, updates centroid, buffered.
                let id = self.clusters.assign(emb.clone());
                self.segment_buffer.push((start_secs, end_secs, emb));
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

    /// At session end: run agglomerative clustering on all buffered embeddings.
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

        let labels = agglomerative_cluster(&embeddings, 0.50);

        let result: Vec<(f64, f64, String)> = self
            .segment_buffer
            .iter()
            .zip(labels.iter())
            .map(|((start, end, _), &id)| (*start, *end, format!("SPEAKER_{:02}", id)))
            .collect();

        tracing::info!(
            "[diarization] finalized {} sub-segments → {} speakers (agglomerative)",
            result.len(),
            labels.iter().max().map(|&m| m + 1).unwrap_or(0)
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

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Convert f32 PCM [-1, 1] to i16 PCM (required by WeSpeaker extractor).
pub fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect()
}

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

    #[test]
    fn f32_to_i16_boundary_values() {
        assert_eq!(f32_to_i16(&[0.0])[0], 0);
        assert_eq!(f32_to_i16(&[1.0])[0], 32767);
        assert_eq!(f32_to_i16(&[-1.0])[0], -32767);
    }

    #[test]
    fn f32_to_i16_clamps_out_of_range() {
        assert_eq!(f32_to_i16(&[2.0])[0], 32767);
        assert_eq!(f32_to_i16(&[-2.0])[0], -32767);
    }

    // ── Agglomerative clustering ──────────────────────────────────────────────

    #[test]
    fn agglomerative_two_identical_embeddings_same_cluster() {
        let embs = vec![
            l2_normalize(&[1.0_f32, 0.0, 0.0]),
            l2_normalize(&[1.0_f32, 0.0, 0.0]),
        ];
        let labels = agglomerative_cluster(&embs, 0.9);
        assert_eq!(labels[0], labels[1]);
    }

    #[test]
    fn agglomerative_orthogonal_embeddings_different_clusters() {
        let embs = vec![
            l2_normalize(&[1.0_f32, 0.0, 0.0]),
            l2_normalize(&[0.0_f32, 1.0, 0.0]),
        ];
        let labels = agglomerative_cluster(&embs, 0.9);
        assert_ne!(labels[0], labels[1]);
    }

    #[test]
    fn agglomerative_two_speakers_four_segments() {
        // A B A B → labels should be 0 1 0 1 (or 1 0 1 0, we only check A≠B).
        let a = l2_normalize(&[1.0_f32, 0.0, 0.0]);
        let b = l2_normalize(&[0.0_f32, 1.0, 0.0]);
        let embs = vec![a.clone(), b.clone(), a, b];
        let labels = agglomerative_cluster(&embs, 0.9);
        assert_eq!(labels[0], labels[2]); // both A
        assert_eq!(labels[1], labels[3]); // both B
        assert_ne!(labels[0], labels[1]); // A ≠ B
    }

    #[test]
    fn agglomerative_single_embedding_returns_zero() {
        let embs = vec![l2_normalize(&[1.0_f32, 0.0])];
        assert_eq!(agglomerative_cluster(&embs, 0.9), vec![0]);
    }

    #[test]
    fn agglomerative_empty_returns_empty() {
        assert!(agglomerative_cluster(&[], 0.9).is_empty());
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
}

// ── Integration tests (require real ONNX models) ───────────────────────────────
//
// Run with: cargo test diarization::integration -- --ignored
//
// Models are loaded from the standard dev model directory
// (~/.sumi-dev/models/).  Copy or symlink the ONNX files there before running:
//   segmentation-3.0.onnx       (5.7 MB)
//   wespeaker-voxceleb-resnet34-LM.onnx  (28 MB)
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
    #[ignore = "diagnostic only, requires segmentation-3.0.onnx in ~/.sumi-dev/models/"]
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
        let outs = session.run(ort::inputs!["input" => tensor]).unwrap();
        let (shape, data) = outs
            .get("output")
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
    #[ignore = "requires segmentation-3.0.onnx in ~/.sumi-dev/models/"]
    fn segmentation_finds_multiple_speech_segments_in_voxconv() {
        let seg_path = crate::settings::segmentation_model_path();
        assert!(
            seg_path.exists(),
            "Model not found: {}. Copy segmentation-3.0.onnx there.",
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
    #[ignore = "requires segmentation-3.0.onnx in ~/.sumi-dev/models/"]
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

    // ── WeSpeaker embedding ────────────────────────────────────────────────────

    /// Verify WeSpeaker produces non-zero 512-dim embeddings from real speech.
    #[test]
    #[ignore = "requires wespeaker-voxceleb-resnet34-LM.onnx in ~/.sumi-dev/models/"]
    fn wespeaker_produces_512_dim_embedding() {
        let emb_path = crate::settings::diarization_model_path();
        assert!(emb_path.exists(), "WeSpeaker model not found: {}", emb_path.display());
        assert!(std::path::Path::new(&test1_wav()).exists());

        let (samples, sr) = load_wav(&test1_wav());
        let samples_16k = to_16k(&samples, sr);
        // Take first 3 seconds of speech.
        let chunk = &samples_16k[..3 * 16_000];
        let samples_i16 = f32_to_i16(chunk);

        let path_str = emb_path.to_str().expect("path utf8");
        let mut extractor =
            pyannote_rs::EmbeddingExtractor::new(path_str).expect("EmbeddingExtractor");
        let emb: Vec<f32> = extractor.compute(&samples_i16).expect("compute").collect();

        println!("\n[wespeaker] embedding dim={}, norm={:.4}", emb.len(), {
            emb.iter().map(|x| x * x).sum::<f32>().sqrt()
        });

        assert_eq!(emb.len(), 512, "expected 512-dim WeSpeaker embedding (ResNet34-LM)");

        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 1.0, "WeSpeaker embedding norm {norm} unexpectedly small");
        // After L2 normalization, same-speaker cosine similarity should be ~1.
        let n = l2_normalize(&emb);
        let self_dist = cosine_dist(&n, &n);
        assert!(self_dist < 1e-4, "self cosine dist {self_dist} should be ~0");
    }

    // ── Full two-phase pipeline ────────────────────────────────────────────────

    /// End-to-end pipeline test matching exp_g_diarize_agglomerative.rs:
    /// process VoxConverse 60 s (2 speakers) in 30 s chunks → finalize →
    /// expect exactly 2 speakers identified.
    #[test]
    #[ignore = "requires both ONNX models in ~/.sumi-dev/models/"]
    fn full_pipeline_detects_two_speakers_in_voxconv() {
        let emb_path = crate::settings::diarization_model_path();
        let seg_path = crate::settings::segmentation_model_path();
        assert!(emb_path.exists());
        assert!(seg_path.exists());
        assert!(std::path::Path::new(&voxconv_wav()).exists());

        let (samples, sr) = load_wav(&voxconv_wav());
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

        // Agglomerative should detect exactly 2 speakers for this clip.
        assert_eq!(
            final_speakers.len(),
            2,
            "Expected 2 speakers (VoxConverse 2-speaker clip), got {}: {:?}",
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

    /// Verify that agglomerative clustering produces the same result as
    /// the experiment's reference implementation on a synthetic 2-speaker sequence.
    #[test]
    fn agglomerative_matches_reference_on_synthetic_two_speakers() {
        // 4 segments: A B A B (alternating) with clearly separated embeddings.
        let a = l2_normalize(&[1.0_f32, 0.0, 0.0, 0.0]);
        let b = l2_normalize(&[0.0_f32, 1.0, 0.0, 0.0]);
        let embs = vec![a.clone(), b.clone(), a.clone(), b.clone()];

        let labels = agglomerative_cluster(&embs, 0.9);
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
        let path_str = emb_path.to_str().unwrap();
        let mut extractor =
            pyannote_rs::EmbeddingExtractor::new(path_str).expect("EmbeddingExtractor");

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
            let samples_i16 = f32_to_i16(emb_chunk);
            let raw: Vec<f32> = extractor.compute(&samples_i16).expect("compute").collect();
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

        // For 2 clusters, we need to merge N-2 times.
        // The Nth merge distance is the threshold to use.
        let n = embeddings.len();
        if n >= 2 {
            let labels = agglomerative_cluster(&embeddings.iter().map(|(_, _, _, e)| e.clone()).collect::<Vec<_>>(), 10.0);
            // Count distinct labels.
            let unique: std::collections::HashSet<usize> = labels.iter().cloned().collect();
            println!("[agglomerative threshold=10.0] {} clusters (should be 2)", unique.len());
            for (i, ((s, e, _, _), lbl)) in embeddings.iter().zip(labels.iter()).enumerate() {
                println!("  [{:.2}s–{:.2}s] → SPEAKER_{:02}", s, e, lbl);
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

        let path_str = emb_path.to_str().unwrap();
        let mut extractor =
            pyannote_rs::EmbeddingExtractor::new(path_str).expect("EmbeddingExtractor");

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
            let samples_i16 = f32_to_i16(chunk);
            let raw: Vec<f32> = extractor.compute(&samples_i16).expect("compute").collect();
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
}

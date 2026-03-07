//! Speaker diarization: segmentation + WeSpeaker embeddings + clustering.
//!
//! # Two-phase pipeline
//!
//! ## Real-time (during recording)
//! ```text
//! VAD chunk (f32, 16 kHz)
//!   → SegmentationModel [1,1,N] → [1,T,7]  (speaker-class per frame)
//!   → sub-segment boundaries (silence + speaker-class changes)
//!   → WeSpeaker [1,T,80] → [1,256]  (per sub-segment)
//!   → L2 normalize
//!   → online cosine clustering  (greedy, for real-time labels)
//!   → "SPEAKER_00" / "SPEAKER_01" / …  +  embedding buffered for phase 2
//! ```
//!
//! ## Finalization (at meeting stop, matches the experiment's quality)
//! ```text
//! buffered (start, end, embedding) for all sub-segments
//!   → agglomerative hierarchical clustering (average linkage, threshold=0.9)
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

// ORT Session is Send.
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

        let mut offset: usize = SEG_FRAME_START;
        // None = silence, Some(cls) = speech with class cls.
        let mut cur_class: Option<usize> = None;
        let mut seg_start_sample: usize = 0;
        let mut segments: Vec<(usize, usize)> = Vec::new();

        for win_start in (0..padded.len()).step_by(SEG_WINDOW_SAMPLES) {
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
/// Direct port from `exp_g_diarize_agglomerative.rs`, achieving DER ≈ 10.5 %
/// on VoxConverse sample 11 (60 s, 2 speakers) at threshold = 0.9.
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

        // O(N²) — fine for N < ~200 segments per meeting.
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

// Both ORT sessions (EmbeddingExtractor + SegmentationModel) are Send.
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
            clusters: SpeakerClusters::new(0.9),
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

        for (start_samp, end_samp) in sub_segs {
            let end_samp = end_samp.min(samples_f32.len());
            if end_samp <= start_samp {
                continue;
            }
            let sub_slice = &samples_f32[start_samp..end_samp];
            let start_secs = chunk_start_secs + start_samp as f64 / 16_000.0;
            let end_secs = chunk_start_secs + end_samp as f64 / 16_000.0;

            let samples_i16 = f32_to_i16(sub_slice);
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

            // Buffer for agglomerative finalization pass.
            self.segment_buffer.push((start_secs, end_secs, emb.clone()));

            // Online clustering for real-time label.
            let speaker_id = self.clusters.assign(emb);
            let label = format!("SPEAKER_{:02}", speaker_id);
            tracing::debug!(
                "[diarization] [{:.2}-{:.2}s] → {} (online)",
                start_secs,
                end_secs,
                label
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

        let labels = agglomerative_cluster(&embeddings, 0.9);

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

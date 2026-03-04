//! Phi-4 Mini (text decoder) via GGUF.
//!
//! Uses `phi3.*` GGUF metadata keys — same prefix as all Phi-3/4 family.
//!
//! Verified against:
//!   - candle-transformers quantized_phi3.rs (tensor names)
//!   - llama.cpp tensor_mapping.py (HF → GGUF name mapping)
//!   - microsoft/Phi-4-mini-instruct/config.json (dimensions)
//!
//! ## Tensor layout in GGUF (phi3 architecture tag)
//!
//! Unlike llama/mistral which store separate q/k/v and gate/up projections,
//! Phi-3/4 uses fused projections in GGUF:
//!
//!   blk.{i}.attn_qkv.weight   — Q+K+V concatenated along dim 0
//!                                shape: [(n_head + 2*n_kv_head) * head_dim, hidden]
//!                                split after forward: Q=[0..n_head*head_dim],
//!                                  K=[..+n_kv_head*head_dim], V=[rest]
//!
//!   blk.{i}.attn_output.weight — output projection
//!
//!   blk.{i}.ffn_up.weight     — gate + up projections concatenated
//!                                shape: [2 * feed_forward_length, hidden]
//!                                split after forward: gate=[0..i_size], up=[i_size..]
//!                                SwiGLU: silu(gate) * up
//!
//!   blk.{i}.ffn_down.weight   — down projection
//!
//!   blk.{i}.attn_norm.weight  — pre-attention RMSNorm
//!   blk.{i}.ffn_norm.weight   — pre-FFN RMSNorm
//!
//! ## Partial RoPE
//!
//! Phi-4 Multimodal: partial_rotary_factor = 0.75, so rope_dim = 96 < head_dim = 128.
//! The first `rope_dim` elements of each head are rotated; the rest pass through unchanged.
//! `phi3.rope.dimension_count` in the GGUF holds the pre-computed rope_dim value.

use std::collections::HashMap;

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;

const MAX_SEQ_LEN: usize = 4096;

// ── RoPE precomputation ───────────────────────────────────────────────────────

/// Precompute cos/sin tables for RoPE over `rope_dim` dimensions.
/// Produces tensors of shape [MAX_SEQ_LEN, rope_dim/2].
/// `candle_nn::rotary_emb::rope(xs, cos, sin)` requires `cos.dim(-1) * 2 == xs.dim(-1)`,
/// so when rope_dim < head_dim we narrow `xs` to `rope_dim` before calling rope().
fn precompute_freqs_cis(
    rope_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<f32> = (0..rope_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / rope_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0u32, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx_theta.cos()?, idx_theta.sin()?))
}

// ── Layer ─────────────────────────────────────────────────────────────────────

struct LayerWeights {
    /// Fused Q+K+V projection. Output has size (n_head + 2*n_kv_head)*head_dim.
    attn_qkv: QMatMul,
    attn_o: QMatMul,
    attn_norm: RmsNorm,

    /// Fused gate+up projection. Output has size 2*i_size.
    /// First half = gate (silu-activated); second half = up.
    ffn_gate_up: QMatMul,
    ffn_down: QMatMul,
    ffn_norm: RmsNorm,

    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    i_size: usize,  // feed_forward_length, needed to split fused gate_up
    /// Number of head dims that participate in RoPE. May be < head_dim (partial RoPE).
    rope_dim: usize,

    cos: Tensor, // [MAX_SEQ_LEN, rope_dim/2]
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl LayerWeights {
    /// Apply RoPE to `x` ([b, h, seq, head_dim]).
    /// When rope_dim < head_dim, only the first rope_dim elements are rotated.
    fn apply_rope(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, _h, seq_len, _d) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;

        if self.rope_dim == self.head_dim {
            // Full RoPE: rope() checks cos.dim(-1)*2 == head_dim ✓
            candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
        } else {
            // Partial RoPE: rotate first rope_dim dims, pass through the rest.
            // After narrow: x_rope.dim(-1) == rope_dim; cos.dim(-1) == rope_dim/2
            // rope() checks: rope_dim/2 * 2 == rope_dim ✓
            let x_rope = x.narrow(D::Minus1, 0, self.rope_dim)?;
            let x_pass = x.narrow(D::Minus1, self.rope_dim, self.head_dim - self.rope_dim)?;
            let x_rope = candle_nn::rotary_emb::rope(&x_rope.contiguous()?, &cos, &sin)?;
            Tensor::cat(&[&x_rope, &x_pass.contiguous()?], D::Minus1)
        }
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let n_groups = self.n_head / self.n_kv_head;
        let q_size = self.n_head * self.head_dim;
        let kv_size = self.n_kv_head * self.head_dim;

        // Fused QKV → split into Q, K, V
        let qkv = self.attn_qkv.forward(x)?; // [b, seq, (n_head + 2*n_kv_head) * head_dim]
        let q = qkv.narrow(D::Minus1, 0, q_size)?;
        let k = qkv.narrow(D::Minus1, q_size, kv_size)?;
        let v = qkv.narrow(D::Minus1, q_size + kv_size, kv_size)?;

        let q = q
            .reshape((b, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?; // [b, n_head, seq, head_dim]
        let k = k
            .reshape((b, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.apply_rope(&q, index_pos)?;
        let k = self.apply_rope(&k, index_pos)?;

        // KV cache
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((kc, vc)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    (Tensor::cat(&[kc, &k], 2)?, Tensor::cat(&[vc, &v], 2)?)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let scale = 1.0 / (self.head_dim as f64).sqrt();

        let y = if q.device().is_metal() && seq_len == 1 {
            // Metal SDPA handles GQA natively
            candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?
        } else {
            let k = repeat_kv(k, n_groups)?.contiguous()?;
            let v = repeat_kv(v, n_groups)?.contiguous()?;
            let att = (q.matmul(&k.t()?)? * scale)?;
            let att = match mask {
                None => att,
                Some(m) => {
                    let m = m.broadcast_as(att.shape())?;
                    m.where_cond(&self.neg_inf.broadcast_as(att.shape())?, &att)?
                }
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?
        };

        // [b, n_head, seq, head_dim] → [b, seq, n_head * head_dim]
        let y = y
            .transpose(1, 2)?
            .reshape((b, seq_len, self.n_head * self.head_dim))?;
        self.attn_o.forward(&y)
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    masks: HashMap<usize, Tensor>,
}

// All candle tensors are Send; we only access this behind a Mutex.
unsafe impl Send for ModelWeights {}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {} in phi3 GGUF metadata", s),
            Some(v) => Ok(v),
        };

        let head_count = md_get("phi3.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("phi3.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("phi3.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("phi3.embedding_length")?.to_u32()? as usize;
        let feed_forward_length = md_get("phi3.feed_forward_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("phi3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_dim = md_get("phi3.rope.dimension_count")?.to_u32()? as usize;
        // phi3.rope.freq_base is present in most modern GGUFs; default to 10000
        // (Phi-4 Mini uses rope_theta=10000 in config.json)
        let rope_freq_base = ct
            .metadata
            .get("phi3.rope.freq_base")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(10_000f32);

        let head_dim = embedding_length / head_count;

        let (cos, sin) = precompute_freqs_cis(rope_dim, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        // LM head — fall back to tied embeddings if "output.weight" is absent
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(t) => t,
            Err(_) => tok_embeddings_q,
        };

        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let p = format!("blk.{i}");
            layers.push(LayerWeights {
                // Fused QKV: one tensor of shape [(n_head+2*n_kv_head)*head_dim, hidden]
                attn_qkv: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_qkv.weight"), device)?,
                )?,
                attn_o: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_output.weight"), device)?,
                )?,
                attn_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_norm.weight"), device)?,
                    rms_norm_eps,
                )?,
                // Fused gate+up: one tensor of shape [2*feed_forward_length, hidden]
                ffn_gate_up: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{p}.ffn_up.weight"), device)?,
                )?,
                ffn_down: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{p}.ffn_down.weight"), device)?,
                )?,
                ffn_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.ffn_norm.weight"), device)?,
                    rms_norm_eps,
                )?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                i_size: feed_forward_length,
                rope_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
        })
    }

    fn mask(&mut self, t: usize, device: &Device) -> Result<Tensor> {
        if let Some(m) = self.masks.get(&t) {
            return Ok(m.clone());
        }
        let mask: Vec<u8> = (0..t)
            .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask, (t, t), device)?;
        self.masks.insert(t, mask.clone());
        Ok(mask)
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, x.device())?)
        };

        let mut h = self.tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            // Attention sub-layer
            let residual = h.clone();
            let normed = layer.attn_norm.forward(&h)?;
            let attn = layer.forward_attn(&normed, mask.as_ref(), index_pos)?;
            h = (attn + residual)?;

            // FFN sub-layer: SwiGLU via fused gate_up
            // ffn_up.weight contains gate (first i_size) and up (second i_size) concatenated
            let residual = h.clone();
            let normed = layer.ffn_norm.forward(&h)?;
            let gate_up = layer.ffn_gate_up.forward(&normed)?; // [b, seq, 2*i_size]
            let gate = gate_up.narrow(D::Minus1, 0, layer.i_size)?;
            let up = gate_up.narrow(D::Minus1, layer.i_size, layer.i_size)?;
            let ffn_out = layer
                .ffn_down
                .forward(&(candle_nn::ops::silu(&gate)? * up)?)?;
            h = (ffn_out + residual)?;
        }

        let h = self.norm.forward(&h)?;
        let h = h.i((.., seq_len - 1, ..))?;
        self.output.forward(&h)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.kv_cache = None;
        }
    }
}

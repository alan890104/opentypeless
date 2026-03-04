//! Ministral-3B 2512 (GGUF architecture tag: "mistral3") model implementation.
//!
//! Architecture is a standard causal transformer decoder with:
//! - GQA: 32 Q heads / 8 KV heads, head_dim = 128 (from mistral3.attention.key_length)
//! - SwiGLU FFN (gate * silu(up), then down)
//! - RMSNorm pre-norm
//! - RoPE with freq_base = 1 000 000
//! - Custom attention scale via mistral3.attention.temperature_scale (= 0.1)
//!
//! Differs from quantized_llama only in:
//!   1. Metadata prefix "mistral3.*" instead of "llama.*"
//!   2. head_dim taken from mistral3.attention.key_length, not embedding_length/head_count
//!   3. Attention scale taken from temperature_scale instead of 1/√head_dim
//!   4. No MoE support

use std::collections::HashMap;

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;

const MAX_SEQ_LEN: usize = 4096;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
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
    attn_q: QMatMul,
    attn_k: QMatMul,
    attn_v: QMatMul,
    attn_o: QMatMul,
    attn_norm: RmsNorm,

    ffn_gate: QMatMul, // SwiGLU gate  (w1)
    ffn_up: QMatMul,   // SwiGLU up    (w3)
    ffn_down: QMatMul, // SwiGLU down  (w2)
    ffn_norm: RmsNorm,

    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    scale: f64, // attention temperature scale (= mistral3.attention.temperature_scale)

    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl LayerWeights {
    fn apply_rope(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, _h, seq_len, _d) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let n_groups = self.n_head / self.n_kv_head;

        // Project
        let q = self
            .attn_q
            .forward(x)?
            .reshape((b, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?; // [b, n_head, seq, head_dim]
        let k = self
            .attn_k
            .forward(x)?
            .reshape((b, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .attn_v
            .forward(x)?
            .reshape((b, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // RoPE
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

        // Attention
        let y = if q.device().is_metal() && seq_len == 1 {
            // Metal SDPA handles GQA natively
            candle_nn::ops::sdpa(&q, &k, &v, None, false, self.scale as f32, 1.)?
        } else {
            let k = repeat_kv(k, n_groups)?.contiguous()?;
            let v = repeat_kv(v, n_groups)?.contiguous()?;
            let att = (q.matmul(&k.t()?)? * self.scale)?;
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

        // [b, n_head, seq, head_dim] -> [b, seq, n_head * head_dim]
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
            None => candle_core::bail!("cannot find {} in mistral3 GGUF metadata", s),
            Some(v) => Ok(v),
        };

        let head_count = md_get("mistral3.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("mistral3.attention.head_count_kv")?.to_u32()? as usize;
        // key_length gives the actual head_dim (may differ from embedding/n_heads)
        let head_dim = md_get("mistral3.attention.key_length")?.to_u32()? as usize;
        let block_count = md_get("mistral3.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("mistral3.embedding_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("mistral3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let temperature_scale =
            md_get("mistral3.attention.temperature_scale")?.to_f32()? as f64;
        let rope_freq_base = md_get("mistral3.rope.freq_base")
            .and_then(|v| v.to_f32())
            .unwrap_or(1_000_000f32);

        let (cos, sin) = precompute_freqs_cis(head_dim, rope_freq_base, device)?;
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
                attn_q: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_q.weight"), device)?,
                )?,
                attn_k: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_k.weight"), device)?,
                )?,
                attn_v: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_v.weight"), device)?,
                )?,
                attn_o: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_output.weight"), device)?,
                )?,
                attn_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_norm.weight"), device)?,
                    rms_norm_eps,
                )?,
                ffn_gate: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{p}.ffn_gate.weight"), device)?,
                )?,
                ffn_up: QMatMul::from_qtensor(
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
                scale: temperature_scale,
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

            // FFN sub-layer (SwiGLU)
            let residual = h.clone();
            let normed = layer.ffn_norm.forward(&h)?;
            let gate = candle_nn::ops::silu(&layer.ffn_gate.forward(&normed)?)?;
            let up = layer.ffn_up.forward(&normed)?;
            let ffn_out = layer.ffn_down.forward(&(gate * up)?)?;
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

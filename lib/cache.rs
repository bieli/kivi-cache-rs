// Streaming KIVI cache.
//
// Keys and values share a timeline but not a quantization boundary.
// Keys are flushed in blocks of `residual_length` (a multiple of the group
// size) because per-channel scales need a full group of tokens. Values are a
// sliding window: only the oldest tokens past `residual_length` are quantized,
// one token group at a time. After any append:
//
// * key residual length is `seq % residual_length`
// * value residual length is `min(seq, residual_length)`

use crate::attention::{self, HeadView};
use crate::budget::{self, AgentMemoryBudget, MemoryBudgetSnapshot};
use crate::config::KiviConfig;
use crate::error::KiviError;
use crate::quant::{self, PackedStream};

// Logical footprint of one cache, compared with a dense baseline.
//
// Byte counts describe stored payload. The residual buffers are preallocated
// to `residual_length` tokens; [`MemoryReport::residual_capacity_bytes`]
// reports that allocation, while the residual fields above count occupied tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryReport {
    pub tokens: usize,
    pub key_code_bytes: usize,
    pub key_metadata_bytes: usize,
    pub value_code_bytes: usize,
    pub value_metadata_bytes: usize,
    pub key_residual_bytes: usize,
    pub value_residual_bytes: usize,
    pub residual_capacity_bytes: usize,
    pub fp16_baseline_bytes: usize,
    pub fp32_baseline_bytes: usize,
}

impl MemoryReport {
    // Codes, `(scale, zero)` pairs, and the occupied full-precision tails.
    pub fn compressed_bytes(&self) -> usize {
        self.key_code_bytes
            + self.key_metadata_bytes
            + self.value_code_bytes
            + self.value_metadata_bytes
            + self.key_residual_bytes
            + self.value_residual_bytes
    }

    // `fp16_baseline / compressed`. Returns 1 when the cache is empty.
    pub fn compression_ratio_vs_fp16(&self) -> f64 {
        let compressed = self.compressed_bytes();
        if compressed == 0 {
            1.0
        } else {
            self.fp16_baseline_bytes as f64 / compressed as f64
        }
    }
}

// Asymmetric quantized KV cache for one attention layer.
//
// Layout of every dense buffer passed in or out is row-major
// `[batch, num_kv_heads, seq, head_dim]`.
//
// # Call sequence
//
// Prefill attention should use the original full-precision K and V. Afterwards
// store them with [`KiviCache::append`]. On each decode step, append the new
// token's K and V *before* calling [`KiviCache::attention`], so the new token
// is already inside the residual window. That matches Algorithm 1 of the paper.
pub struct KiviCache {
    config: KiviConfig,
    batch: usize,
    heads: usize,
    head_dim: usize,
    key_streams: Vec<PackedStream>,
    value_streams: Vec<PackedStream>,
    key_residual: Vec<f32>,
    value_residual: Vec<f32>,
    key_res_len: usize,
    value_res_len: usize,
    seq_len: usize,
}

impl KiviCache {
    // Allocate an empty cache. `head_dim` must be a multiple of `config.group_size`.
    pub fn new(
        config: KiviConfig,
        batch: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, KiviError> {
        if batch == 0 {
            return Err(KiviError::ZeroDimension("batch"));
        }
        if num_kv_heads == 0 {
            return Err(KiviError::ZeroDimension("num_kv_heads"));
        }
        if head_dim == 0 || head_dim % config.group_size != 0 {
            return Err(KiviError::HeadDim {
                head_dim,
                group_size: config.group_size,
            });
        }
        let n = batch * num_kv_heads;
        let residual_floats = n * config.residual_length * head_dim;
        Ok(Self {
            config,
            batch,
            heads: num_kv_heads,
            head_dim,
            key_streams: (0..n).map(|_| PackedStream::default()).collect(),
            value_streams: (0..n).map(|_| PackedStream::default()).collect(),
            key_residual: vec![0.0; residual_floats],
            value_residual: vec![0.0; residual_floats],
            key_res_len: 0,
            value_res_len: 0,
            seq_len: 0,
        })
    }

    pub fn config(&self) -> KiviConfig {
        self.config
    }

    pub fn batch_size(&self) -> usize {
        self.batch
    }

    pub fn num_kv_heads(&self) -> usize {
        self.heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn sequence_length(&self) -> usize {
        self.seq_len
    }

    pub fn key_residual_len(&self) -> usize {
        self.key_res_len
    }

    pub fn value_residual_len(&self) -> usize {
        self.value_res_len
    }

    pub fn quantized_key_tokens(&self) -> usize {
        self.key_streams.first().map(|s| s.tokens).unwrap_or(0)
    }

    pub fn quantized_value_tokens(&self) -> usize {
        self.value_streams.first().map(|s| s.tokens).unwrap_or(0)
    }

    // Append `seq_len` tokens. Both buffers use `[batch, num_kv_heads, seq_len, head_dim]`.
    //
    // A multi-token append has the same groups as appending those tokens one
    // at a time. Pass `seq_len == 0` to leave the cache unchanged.
    pub fn append(
        &mut self,
        keys: &[f32],
        values: &[f32],
        seq_len: usize,
    ) -> Result<(), KiviError> {
        let expected = self.batch * self.heads * seq_len * self.head_dim;
        if keys.len() != expected {
            return Err(KiviError::Shape {
                expected,
                actual: keys.len(),
            });
        }
        if values.len() != expected {
            return Err(KiviError::Shape {
                expected,
                actual: values.len(),
            });
        }
        self.append_keys(keys, seq_len);
        self.append_values(values, seq_len);
        self.seq_len += seq_len;
        debug_assert_eq!(self.quantized_key_tokens() + self.key_res_len, self.seq_len);
        debug_assert_eq!(
            self.quantized_value_tokens() + self.value_res_len,
            self.seq_len
        );
        Ok(())
    }

    // Decode attention for one query per batch element.
    //
    // `query` is `[batch, num_q_heads, head_dim]`. `num_q_heads` may be a
    // multiple of the KV head count (grouped-query attention). `scale` is the
    // attention multiplier, usually `1 / sqrt(head_dim)`.
    //
    // The result is `[batch, num_q_heads, head_dim]`. Every cached token is
    // visible; this is the decode step, not a causal prefill.
    pub fn attention(
        &self,
        query: &[f32],
        num_q_heads: usize,
        scale: f32,
    ) -> Result<Vec<f32>, KiviError> {
        if num_q_heads == 0 || num_q_heads % self.heads != 0 {
            return Err(KiviError::QueryHeads {
                num_q_heads,
                num_kv_heads: self.heads,
            });
        }
        let expected = self.batch * num_q_heads * self.head_dim;
        if query.len() != expected {
            return Err(KiviError::Shape {
                expected,
                actual: query.len(),
            });
        }
        let mut out = vec![0.0; expected];
        if self.seq_len == 0 {
            return Ok(out);
        }
        let mut logits = vec![0.0; self.seq_len];
        let group = num_q_heads / self.heads;
        let cap = self.config.residual_length;
        for b in 0..self.batch {
            for qh in 0..num_q_heads {
                let kvh = qh / group;
                let hi = b * self.heads + kvh;
                let slot = hi * cap * self.head_dim;
                let q = &query[(b * num_q_heads + qh) * self.head_dim..][..self.head_dim];
                let y = &mut out[(b * num_q_heads + qh) * self.head_dim..][..self.head_dim];
                attention::attend_head(
                    HeadView {
                        keys: &self.key_streams[hi],
                        values: &self.value_streams[hi],
                        key_residual: &self.key_residual
                            [slot..slot + self.key_res_len * self.head_dim],
                        value_residual: &self.value_residual
                            [slot..slot + self.value_res_len * self.head_dim],
                        key_res_len: self.key_res_len,
                        value_res_len: self.value_res_len,
                    },
                    q,
                    self.config.bits,
                    self.config.group_size,
                    self.head_dim,
                    scale,
                    &mut logits,
                    y,
                );
            }
        }
        Ok(out)
    }

    // Dense reconstruction. Quantized tokens are dequantized; residual tokens are copied.
    pub fn dequantize(&self) -> (Vec<f32>, Vec<f32>) {
        let n = self.batch * self.heads * self.seq_len * self.head_dim;
        let mut keys = vec![0.0; n];
        let mut values = vec![0.0; n];
        let kq = self.quantized_key_tokens();
        let vq = self.quantized_value_tokens();
        let cap = self.config.residual_length;
        for b in 0..self.batch {
            for h in 0..self.heads {
                let hi = b * self.heads + h;
                let dst = hi * self.seq_len * self.head_dim;
                let slot = hi * cap * self.head_dim;
                if kq > 0 {
                    quant::dequant_keys(
                        &self.key_streams[hi],
                        self.config.bits,
                        self.config.group_size,
                        self.head_dim,
                        &mut keys[dst..dst + kq * self.head_dim],
                    );
                }
                if self.key_res_len > 0 {
                    let n_res = self.key_res_len * self.head_dim;
                    keys[dst + kq * self.head_dim..dst + kq * self.head_dim + n_res]
                        .copy_from_slice(&self.key_residual[slot..slot + n_res]);
                }
                if vq > 0 {
                    quant::dequant_values(
                        &self.value_streams[hi],
                        self.config.bits,
                        self.config.group_size,
                        self.head_dim,
                        &mut values[dst..dst + vq * self.head_dim],
                    );
                }
                if self.value_res_len > 0 {
                    let n_res = self.value_res_len * self.head_dim;
                    values[dst + vq * self.head_dim..dst + vq * self.head_dim + n_res]
                        .copy_from_slice(&self.value_residual[slot..slot + n_res]);
                }
            }
        }
        (keys, values)
    }

    // Tokens until the key residual buffer fills and the next append triggers a flush.
    pub fn tokens_until_key_flush(&self) -> usize {
        self.config.residual_length.saturating_sub(self.key_res_len)
    }

    // Free slots in the value fp32 sliding window (`0` when the window is full).
    pub fn value_residual_slots_remaining(&self) -> usize {
        self.config
            .residual_length
            .saturating_sub(self.value_res_len)
    }

    // Live byte accounting plus flush horizon for this layer.
    pub fn budget_snapshot(&self) -> MemoryBudgetSnapshot {
        let report = self.memory();
        MemoryBudgetSnapshot {
            tokens_until_key_flush: self.tokens_until_key_flush(),
            key_residual_tokens: self.key_res_len,
            value_residual_tokens: self.value_res_len,
            sequence_tokens: self.seq_len,
            current_layer_bytes: report.compressed_bytes(),
            marginal_bytes_per_token: budget::marginal_bytes_per_layer(
                &self.config,
                self.heads,
                self.head_dim,
            ),
            residual_capacity_bytes: report.residual_capacity_bytes,
        }
    }

    // Build a multi-layer planner from this cache's shape and config.
    pub fn memory_budget(&self, num_layers: usize) -> Result<AgentMemoryBudget, KiviError> {
        AgentMemoryBudget::from_cache(self, num_layers)
    }

    pub fn memory(&self) -> MemoryReport {
        let key_code_bytes = self.key_streams.iter().map(|s| s.codes.len() * 4).sum();
        let key_metadata_bytes = self.key_streams.iter().map(|s| s.params.len() * 4).sum();
        let value_code_bytes = self.value_streams.iter().map(|s| s.codes.len() * 4).sum();
        let value_metadata_bytes = self.value_streams.iter().map(|s| s.params.len() * 4).sum();
        let elements = self.batch * self.heads * self.head_dim;
        MemoryReport {
            tokens: self.seq_len,
            key_code_bytes,
            key_metadata_bytes,
            value_code_bytes,
            value_metadata_bytes,
            key_residual_bytes: self.key_res_len * elements * 4,
            value_residual_bytes: self.value_res_len * elements * 4,
            residual_capacity_bytes: 2 * self.config.residual_length * elements * 4,
            fp16_baseline_bytes: 2 * self.seq_len * elements * 2,
            fp32_baseline_bytes: 2 * self.seq_len * elements * 4,
        }
    }

    fn append_keys(&mut self, keys: &[f32], seq: usize) {
        let mut offset = 0;
        while offset < seq {
            let space = self.config.residual_length - self.key_res_len;
            let take = space.min(seq - offset);
            copy_tokens(
                &mut self.key_residual,
                self.config.residual_length,
                self.key_res_len,
                keys,
                seq,
                offset,
                take,
                self.batch,
                self.heads,
                self.head_dim,
            );
            self.key_res_len += take;
            offset += take;
            if self.key_res_len == self.config.residual_length {
                self.flush_keys();
            }
        }
    }

    fn flush_keys(&mut self) {
        let r = self.config.residual_length;
        debug_assert_eq!(self.key_res_len, r);
        let bits = self.config.bits;
        let group = self.config.group_size;
        let dim = self.head_dim;
        let batch = self.batch;
        let heads = self.heads;
        for b in 0..batch {
            for h in 0..heads {
                let hi = b * heads + h;
                let start = hi * r * dim;
                quant::append_key_block(
                    &mut self.key_streams[hi],
                    &self.key_residual[start..start + r * dim],
                    r,
                    dim,
                    bits,
                    group,
                );
            }
        }
        self.key_res_len = 0;
    }

    fn append_values(&mut self, values: &[f32], seq: usize) {
        let r = self.config.residual_length;
        if seq >= r {
            self.quantize_value_residual_prefix(self.value_res_len);
            let quant_n = seq - r;
            if quant_n > 0 {
                self.quantize_value_input(values, seq, 0, quant_n);
            }
            copy_tokens(
                &mut self.value_residual,
                r,
                0,
                values,
                seq,
                quant_n,
                r,
                self.batch,
                self.heads,
                self.head_dim,
            );
            self.value_res_len = r;
            return;
        }
        let combined = self.value_res_len + seq;
        let drop = combined.saturating_sub(r);
        if drop > 0 {
            self.quantize_value_residual_prefix(drop);
            let keep = self.value_res_len - drop;
            shift_residual(
                &mut self.value_residual,
                r,
                drop,
                keep,
                self.batch,
                self.heads,
                self.head_dim,
            );
            self.value_res_len = keep;
        }
        copy_tokens(
            &mut self.value_residual,
            r,
            self.value_res_len,
            values,
            seq,
            0,
            seq,
            self.batch,
            self.heads,
            self.head_dim,
        );
        self.value_res_len += seq;
    }

    fn quantize_value_residual_prefix(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        let cap = self.config.residual_length;
        let dim = self.head_dim;
        let bits = self.config.bits;
        let group = self.config.group_size;
        let batch = self.batch;
        let heads = self.heads;
        for b in 0..batch {
            for h in 0..heads {
                let hi = b * heads + h;
                let start = hi * cap * dim;
                quant::append_value_block(
                    &mut self.value_streams[hi],
                    &self.value_residual[start..start + n * dim],
                    n,
                    dim,
                    bits,
                    group,
                );
            }
        }
    }

    fn quantize_value_input(&mut self, values: &[f32], seq: usize, offset: usize, count: usize) {
        let dim = self.head_dim;
        let bits = self.config.bits;
        let group = self.config.group_size;
        for b in 0..self.batch {
            for h in 0..self.heads {
                let hi = b * self.heads + h;
                let start = hi * seq * dim + offset * dim;
                quant::append_value_block(
                    &mut self.value_streams[hi],
                    &values[start..start + count * dim],
                    count,
                    dim,
                    bits,
                    group,
                );
            }
        }
    }
}

fn copy_tokens(
    dst: &mut [f32],
    dst_cap: usize,
    dst_off: usize,
    src: &[f32],
    src_seq: usize,
    src_off: usize,
    count: usize,
    batch: usize,
    heads: usize,
    dim: usize,
) {
    if count == 0 {
        return;
    }
    for b in 0..batch {
        for h in 0..heads {
            let hi = b * heads + h;
            let dst_i = hi * dst_cap * dim + dst_off * dim;
            let src_i = hi * src_seq * dim + src_off * dim;
            let n = count * dim;
            dst[dst_i..dst_i + n].copy_from_slice(&src[src_i..src_i + n]);
        }
    }
}

fn shift_residual(
    residual: &mut [f32],
    cap: usize,
    from: usize,
    keep: usize,
    batch: usize,
    heads: usize,
    dim: usize,
) {
    if keep == 0 || from == 0 {
        return;
    }
    for b in 0..batch {
        for h in 0..heads {
            let base = (b * heads + h) * cap * dim;
            let src = base + from * dim;
            residual.copy_within(src..src + keep * dim, base);
        }
    }
}

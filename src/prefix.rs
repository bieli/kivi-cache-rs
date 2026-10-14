// Shared RAG prefix.
//
// One long document is quantized once. Each question is a short full-precision
// suffix on its own session. Sessions share the prefix allocation.

use std::sync::Arc;

use crate::attention::{self, HeadView};
use crate::cache::KiviCache;
use crate::error::KiviError;
use crate::quant::PackedStream;
use crate::KiviConfig;

struct PrefixState {
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
    prefix_bytes: usize,
}

// Read-only KIVI snapshot of a document prefill.
//
// `Clone` only bumps a reference count. The packed codes stay one copy.
#[derive(Clone)]
pub struct SharedPrefix {
    state: Arc<PrefixState>,
}

// One question attached to a [`SharedPrefix`].
//
// The suffix is full precision and private to this session. It does not change
// the shared document, and it is not quantized, so short questions stay exact.
pub struct RagSession {
    prefix: SharedPrefix,
    local_keys: Vec<f32>,
    local_values: Vec<f32>,
    local_len: usize,
}

// Bytes for `sessions` questions that share one prefix, versus storing the
// document in fp16 once per user.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharedPrefixFootprint {
    pub prefix_tokens: usize,
    pub question_tokens: usize,
    pub sessions: usize,
    pub prefix_bytes: usize,
    pub question_bytes: usize,
    pub shared_total_bytes: usize,
    pub fp16_per_user_bytes: usize,
    pub fp16_all_users_bytes: usize,
}

impl SharedPrefixFootprint {
    // `fp16_all_users / shared_total`. Returns 1 when there is nothing to store.
    pub fn savings_vs_fp16(&self) -> f64 {
        if self.shared_total_bytes == 0 {
            1.0
        } else {
            self.fp16_all_users_bytes as f64 / self.shared_total_bytes as f64
        }
    }
}

impl SharedPrefix {
    pub(crate) fn from_cache(cache: &KiviCache) -> Self {
        let report = cache.memory();
        Self {
            state: Arc::new(PrefixState {
                config: cache.config(),
                batch: cache.batch_size(),
                heads: cache.num_kv_heads(),
                head_dim: cache.head_dim(),
                key_streams: cache.key_streams().to_vec(),
                value_streams: cache.value_streams().to_vec(),
                key_residual: cache.key_residual().to_vec(),
                value_residual: cache.value_residual().to_vec(),
                key_res_len: cache.key_residual_len(),
                value_res_len: cache.value_residual_len(),
                seq_len: cache.sequence_length(),
                prefix_bytes: report.compressed_bytes(),
            }),
        }
    }

    pub fn config(&self) -> KiviConfig {
        self.state.config
    }

    pub fn batch_size(&self) -> usize {
        self.state.batch
    }

    pub fn num_kv_heads(&self) -> usize {
        self.state.heads
    }

    pub fn head_dim(&self) -> usize {
        self.state.head_dim
    }

    pub fn prefix_tokens(&self) -> usize {
        self.state.seq_len
    }

    // Packed codes, metadata, and the occupied fp32 tail of the document.
    pub fn prefix_bytes(&self) -> usize {
        self.state.prefix_bytes
    }

    // True when both prefixes point at the same allocation.
    pub fn shares_allocation_with(&self, other: &SharedPrefix) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }

    pub fn session(&self) -> RagSession {
        RagSession {
            prefix: self.clone(),
            local_keys: Vec::new(),
            local_values: Vec::new(),
            local_len: 0,
        }
    }

    // Footprint of `sessions` questions of `question_tokens` each.
    //
    // Question K and V stay fp32. The document is counted once.
    pub fn footprint(&self, sessions: usize, question_tokens: usize) -> SharedPrefixFootprint {
        let elements = self.state.batch * self.state.heads * self.state.head_dim;
        let question_bytes = sessions * question_tokens * elements * 2 * 4;
        let prefix_bytes = self.state.prefix_bytes;
        let tokens = self.state.seq_len + question_tokens;
        let fp16_per_user = 2 * tokens * elements * 2;
        SharedPrefixFootprint {
            prefix_tokens: self.state.seq_len,
            question_tokens,
            sessions,
            prefix_bytes,
            question_bytes,
            shared_total_bytes: prefix_bytes + question_bytes,
            fp16_per_user_bytes: fp16_per_user,
            fp16_all_users_bytes: sessions * fp16_per_user,
        }
    }
}

impl RagSession {
    pub fn prefix(&self) -> &SharedPrefix {
        &self.prefix
    }

    pub fn prefix_tokens(&self) -> usize {
        self.prefix.prefix_tokens()
    }

    pub fn question_tokens(&self) -> usize {
        self.local_len
    }

    pub fn sequence_length(&self) -> usize {
        self.prefix_tokens() + self.local_len
    }

    // Append question tokens. Layout is `[batch, num_kv_heads, seq_len, head_dim]`.
    pub fn append(
        &mut self,
        keys: &[f32],
        values: &[f32],
        seq_len: usize,
    ) -> Result<(), KiviError> {
        let state = &self.prefix.state;
        let expected = state.batch * state.heads * seq_len * state.head_dim;
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
        if seq_len == 0 {
            return Ok(());
        }
        let new_len = self.local_len + seq_len;
        let mut next_keys = vec![0.0; state.batch * state.heads * new_len * state.head_dim];
        let mut next_values = vec![0.0; state.batch * state.heads * new_len * state.head_dim];
        splice_tokens(
            &mut next_keys,
            &self.local_keys,
            keys,
            self.local_len,
            seq_len,
            state.batch,
            state.heads,
            state.head_dim,
        );
        splice_tokens(
            &mut next_values,
            &self.local_values,
            values,
            self.local_len,
            seq_len,
            state.batch,
            state.heads,
            state.head_dim,
        );
        self.local_keys = next_keys;
        self.local_values = next_values;
        self.local_len = new_len;
        Ok(())
    }

    // Decode attention over the shared document and this session's question.
    //
    // `query` is `[batch, num_q_heads, head_dim]`. The result has the same shape.
    pub fn attention(
        &self,
        query: &[f32],
        num_q_heads: usize,
        scale: f32,
    ) -> Result<Vec<f32>, KiviError> {
        let state = &self.prefix.state;
        if num_q_heads == 0 || num_q_heads % state.heads != 0 {
            return Err(KiviError::QueryHeads {
                num_q_heads,
                num_kv_heads: state.heads,
            });
        }
        let expected = state.batch * num_q_heads * state.head_dim;
        if query.len() != expected {
            return Err(KiviError::Shape {
                expected,
                actual: query.len(),
            });
        }
        let total = self.sequence_length();
        let mut out = vec![0.0; expected];
        if total == 0 {
            return Ok(out);
        }
        let mut logits = vec![0.0; total];
        let group = num_q_heads / state.heads;
        let cap = state.config.residual_length;
        let dim = state.head_dim;
        for b in 0..state.batch {
            for qh in 0..num_q_heads {
                let kvh = qh / group;
                let hi = b * state.heads + kvh;
                let slot = hi * cap * dim;
                let local = hi * self.local_len * dim;
                let q = &query[(b * num_q_heads + qh) * dim..][..dim];
                let y = &mut out[(b * num_q_heads + qh) * dim..][..dim];
                attention::attend_head_with_suffix(
                    HeadView {
                        keys: &state.key_streams[hi],
                        values: &state.value_streams[hi],
                        key_residual: &state.key_residual[slot..slot + state.key_res_len * dim],
                        value_residual: &state.value_residual
                            [slot..slot + state.value_res_len * dim],
                        key_res_len: state.key_res_len,
                        value_res_len: state.value_res_len,
                    },
                    &self.local_keys[local..local + self.local_len * dim],
                    &self.local_values[local..local + self.local_len * dim],
                    self.local_len,
                    q,
                    state.config.bits,
                    state.config.group_size,
                    dim,
                    scale,
                    &mut logits,
                    y,
                );
            }
        }
        Ok(out)
    }
}

fn splice_tokens(
    dst: &mut [f32],
    old: &[f32],
    new_tokens: &[f32],
    old_len: usize,
    new_len: usize,
    batch: usize,
    heads: usize,
    dim: usize,
) {
    let total = old_len + new_len;
    for b in 0..batch {
        for h in 0..heads {
            let hi = b * heads + h;
            let dst_i = hi * total * dim;
            if old_len > 0 {
                let old_i = hi * old_len * dim;
                let n = old_len * dim;
                dst[dst_i..dst_i + n].copy_from_slice(&old[old_i..old_i + n]);
            }
            let src_i = hi * new_len * dim;
            let n = new_len * dim;
            dst[dst_i + old_len * dim..dst_i + old_len * dim + n]
                .copy_from_slice(&new_tokens[src_i..src_i + n]);
        }
    }
}

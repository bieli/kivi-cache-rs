// Quality guard for one KIVI layer.
//
// After each key flush the guard scores attention against the original fp32
// K and V. If the cosine falls below the threshold, it rebuilds the resident
// cache with wider codes or a longer residual window. The fp32 tape is the
// reference used for that check and for the rebuild.

use crate::error::KiviError;
use crate::KiviCache;
use crate::KiviConfig;

// What the guard changes when a flush misses the cosine threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityPolicy {
    // 2-bit becomes 4-bit, then 8-bit, up to the configured ceiling.
    RaiseBits,
    // Residual length doubles, up to the configured ceiling.
    LengthenResidual,
}

// One accepted change applied during a flush check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityUpgrade {
    Bits { from: u8, to: u8 },
    Residual { from: usize, to: usize },
}

// Cosine of one probe query measured just after a key flush.
#[derive(Debug, Clone, PartialEq)]
pub struct FlushCheck {
    pub sequence_tokens: usize,
    pub cosine_before: f32,
    pub cosine: f32,
    pub upgrades: Vec<QualityUpgrade>,
}

// Resident quantized cache plus the fp32 reference tape.
pub struct QualityGuard {
    cache: KiviCache,
    keys: Vec<f32>,
    values: Vec<f32>,
    seq: usize,
    threshold: f32,
    max_bits: u8,
    max_residual: usize,
    policy: QualityPolicy,
}

impl QualityGuard {
    pub fn try_new(
        config: KiviConfig,
        batch: usize,
        num_kv_heads: usize,
        head_dim: usize,
        threshold: f32,
        policy: QualityPolicy,
        max_bits: u8,
        max_residual: usize,
    ) -> Result<Self, KiviError> {
        let cache = KiviCache::new(config, batch, num_kv_heads, head_dim)?;
        Ok(Self {
            cache,
            keys: Vec::new(),
            values: Vec::new(),
            seq: 0,
            threshold,
            max_bits,
            max_residual,
            policy,
        })
    }

    pub fn config(&self) -> KiviConfig {
        self.cache.config()
    }

    pub fn cache(&self) -> &KiviCache {
        &self.cache
    }

    pub fn policy(&self) -> QualityPolicy {
        self.policy
    }

    pub fn sequence_length(&self) -> usize {
        self.seq
    }

    // Append tokens, and check quality once per key flush inside this append.
    //
    // `query` is `[batch, num_q_heads, head_dim]`. The same probe is scored
    // after every flush in this call.
    pub fn append(
        &mut self,
        keys: &[f32],
        values: &[f32],
        seq_len: usize,
        query: &[f32],
        num_q_heads: usize,
        scale: f32,
    ) -> Result<Vec<FlushCheck>, KiviError> {
        let batch = self.cache.batch_size();
        let heads = self.cache.num_kv_heads();
        let dim = self.cache.head_dim();
        let expected = batch * heads * seq_len * dim;
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
        let query_elems = batch * num_q_heads * dim;
        if num_q_heads == 0 || num_q_heads % heads != 0 {
            return Err(KiviError::QueryHeads {
                num_q_heads,
                num_kv_heads: heads,
            });
        }
        if query.len() != query_elems {
            return Err(KiviError::Shape {
                expected: query_elems,
                actual: query.len(),
            });
        }

        let mut checks = Vec::new();
        let mut offset = 0;
        while offset < seq_len {
            let room = self.cache.tokens_until_key_flush();
            let take = room.min(seq_len - offset);
            let key_span = slice_tokens(keys, batch, heads, seq_len, offset, take, dim);
            let value_span = slice_tokens(values, batch, heads, seq_len, offset, take, dim);
            self.cache.append(&key_span, &value_span, take)?;
            self.push_history(&key_span, &value_span, take);
            if self.cache.key_residual_len() == 0 {
                checks.push(self.check(query, num_q_heads, scale)?);
            }
            offset += take;
        }
        Ok(checks)
    }

    fn check(
        &mut self,
        query: &[f32],
        num_q_heads: usize,
        scale: f32,
    ) -> Result<FlushCheck, KiviError> {
        let cosine_before = self.cosine(query, num_q_heads, scale)?;
        let mut cosine = cosine_before;
        let mut upgrades = Vec::new();
        while cosine < self.threshold {
            let Some(upgrade) = self.propose() else {
                break;
            };
            self.apply(upgrade)?;
            upgrades.push(upgrade);
            cosine = self.cosine(query, num_q_heads, scale)?;
        }
        Ok(FlushCheck {
            sequence_tokens: self.seq,
            cosine_before,
            cosine,
            upgrades,
        })
    }

    fn propose(&self) -> Option<QualityUpgrade> {
        let config = self.cache.config();
        match self.policy {
            QualityPolicy::RaiseBits => {
                next_bits(config.bits, self.max_bits).map(|to| QualityUpgrade::Bits {
                    from: config.bits,
                    to,
                })
            }
            QualityPolicy::LengthenResidual => {
                next_residual(config.residual_length, config.group_size, self.max_residual).map(
                    |to| QualityUpgrade::Residual {
                        from: config.residual_length,
                        to,
                    },
                )
            }
        }
    }

    fn apply(&mut self, upgrade: QualityUpgrade) -> Result<(), KiviError> {
        let current = self.cache.config();
        let config = match upgrade {
            QualityUpgrade::Bits { to, .. } => {
                KiviConfig::try_new(to, current.group_size, current.residual_length)?
            }
            QualityUpgrade::Residual { to, .. } => {
                KiviConfig::try_new(current.bits, current.group_size, to)?
            }
        };
        let mut cache = KiviCache::new(
            config,
            self.cache.batch_size(),
            self.cache.num_kv_heads(),
            self.cache.head_dim(),
        )?;
        if self.seq > 0 {
            cache.append(&self.keys, &self.values, self.seq)?;
        }
        self.cache = cache;
        Ok(())
    }

    fn cosine(&self, query: &[f32], num_q_heads: usize, scale: f32) -> Result<f32, KiviError> {
        let got = self.cache.attention(query, num_q_heads, scale)?;
        let expect = fp32_attention(
            query,
            &self.keys,
            &self.values,
            self.cache.batch_size(),
            self.cache.num_kv_heads(),
            num_q_heads,
            self.seq,
            self.cache.head_dim(),
            scale,
        );
        Ok(cosine_sim(&got, &expect))
    }

    fn push_history(&mut self, keys: &[f32], values: &[f32], seq_len: usize) {
        let batch = self.cache.batch_size();
        let heads = self.cache.num_kv_heads();
        let dim = self.cache.head_dim();
        self.keys = concat_tokens(&self.keys, keys, self.seq, seq_len, batch, heads, dim);
        self.values = concat_tokens(&self.values, values, self.seq, seq_len, batch, heads, dim);
        self.seq += seq_len;
    }
}

fn next_bits(current: u8, max_bits: u8) -> Option<u8> {
    let next = match current {
        2 => 4,
        4 => 8,
        _ => return None,
    };
    if next <= max_bits {
        Some(next)
    } else {
        None
    }
}

fn next_residual(current: usize, group_size: usize, max_residual: usize) -> Option<usize> {
    if current >= max_residual {
        return None;
    }
    let doubled = current.saturating_mul(2);
    let next = if doubled <= max_residual {
        doubled
    } else {
        max_residual
    };
    if next > current && next % group_size == 0 {
        Some(next)
    } else {
        None
    }
}

fn slice_tokens(
    src: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    start: usize,
    len: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; batch * heads * len * dim];
    for b in 0..batch {
        for h in 0..heads {
            let hi = b * heads + h;
            let from = hi * seq * dim + start * dim;
            let to = hi * len * dim;
            out[to..to + len * dim].copy_from_slice(&src[from..from + len * dim]);
        }
    }
    out
}

fn concat_tokens(
    old: &[f32],
    new_tokens: &[f32],
    old_len: usize,
    new_len: usize,
    batch: usize,
    heads: usize,
    dim: usize,
) -> Vec<f32> {
    let total = old_len + new_len;
    let mut dst = vec![0.0; batch * heads * total * dim];
    for b in 0..batch {
        for h in 0..heads {
            let hi = b * heads + h;
            let dst_i = hi * total * dim;
            if old_len > 0 {
                let n = old_len * dim;
                let old_i = hi * old_len * dim;
                dst[dst_i..dst_i + n].copy_from_slice(&old[old_i..old_i + n]);
            }
            let n = new_len * dim;
            let src_i = hi * new_len * dim;
            dst[dst_i + old_len * dim..dst_i + old_len * dim + n]
                .copy_from_slice(&new_tokens[src_i..src_i + n]);
        }
    }
    dst
}

fn fp32_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    batch: usize,
    kv_heads: usize,
    q_heads: usize,
    seq: usize,
    dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; batch * q_heads * dim];
    let group = q_heads / kv_heads;
    let mut logits = vec![0.0; seq];
    for b in 0..batch {
        for qh in 0..q_heads {
            let kvh = qh / group;
            let q = &query[(b * q_heads + qh) * dim..][..dim];
            let base = (b * kv_heads + kvh) * seq * dim;
            for t in 0..seq {
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += q[d] * keys[base + t * dim + d];
                }
                logits[t] = dot * scale;
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for logit in logits.iter_mut() {
                *logit = (*logit - max).exp();
                sum += *logit;
            }
            let y = &mut out[(b * q_heads + qh) * dim..][..dim];
            y.fill(0.0);
            for t in 0..seq {
                let w = logits[t] / sum;
                for d in 0..dim {
                    y[d] += w * values[base + t * dim + d];
                }
            }
        }
    }
    out
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        1.0
    } else {
        dot / denom
    }
}

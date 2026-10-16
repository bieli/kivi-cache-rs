// KV backend for a Rust runtime.
//
// Candle, Burn, and a custom server already own contiguous `f32` storage.
// They pass that storage here as slices in `[batch, num_kv_heads, seq, head_dim]`
// order. No framework tensor crosses this boundary.

use crate::error::KiviError;
use crate::KiviCache;
use crate::KiviConfig;

// One layer of KIVI owned by the host process.
pub struct RuntimeKv {
    cache: KiviCache,
}

impl RuntimeKv {
    pub fn try_new(
        config: KiviConfig,
        batch: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, KiviError> {
        Ok(Self {
            cache: KiviCache::new(config, batch, num_kv_heads, head_dim)?,
        })
    }

    pub fn cache(&self) -> &KiviCache {
        &self.cache
    }

    // Store a prefill. `keys` and `values` are `[batch, num_kv_heads, seq_len, head_dim]`.
    pub fn prefill(
        &mut self,
        keys: &[f32],
        values: &[f32],
        seq_len: usize,
    ) -> Result<(), KiviError> {
        self.cache.append(keys, values, seq_len)
    }

    // Append one new token, then score `query`.
    //
    // `key` and `value` are `[batch, num_kv_heads, 1, head_dim]`.
    // `query` is `[batch, num_q_heads, head_dim]`. The result has the query's shape.
    pub fn decode(
        &mut self,
        key: &[f32],
        value: &[f32],
        query: &[f32],
        num_q_heads: usize,
        scale: f32,
    ) -> Result<Vec<f32>, KiviError> {
        self.cache.append(key, value, 1)?;
        self.cache.attention(query, num_q_heads, scale)
    }
}

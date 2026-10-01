// Agent-oriented KV memory planning.
//
// Answer questions like “how many parallel agent sessions fit in 24 GiB if each
// holds 200k tokens?” without running a model. Estimates follow the same byte
// layout as [`MemoryReport`](crate::MemoryReport) and [`KiviCache`](crate::KiviCache).

use crate::config::KiviConfig;
use crate::error::KiviError;
use crate::KiviCache;

// Model shape and KIVI settings for capacity planning (no live cache required).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentMemoryBudget {
    pub config: KiviConfig,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

// Live snapshot of one [`KiviCache`] layer plus planning numbers derived from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudgetSnapshot {
    // Tokens until the key residual buffer fills and flushes to packed storage.
    pub tokens_until_key_flush: usize,
    pub key_residual_tokens: usize,
    pub value_residual_tokens: usize,
    pub sequence_tokens: usize,
    // Packed codes + metadata + occupied fp32 residual for this layer.
    pub current_layer_bytes: usize,
    // Amortized packed growth per new token once context length exceeds
    // [`KiviConfig::residual_length`].
    pub marginal_bytes_per_token: usize,
    // Preallocated fp32 residual capacity (matches [`MemoryReport::residual_capacity_bytes`]).
    pub residual_capacity_bytes: usize,
}

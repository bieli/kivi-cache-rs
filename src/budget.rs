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

impl AgentMemoryBudget {
    // Plan from hyperparameters and transformer depth.
    pub fn try_new(
        config: KiviConfig,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, KiviError> {
        if num_layers == 0 {
            return Err(KiviError::ZeroDimension("num_layers"));
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
        Ok(Self {
            config,
            num_layers,
            num_kv_heads,
            head_dim,
        })
    }

    // Derive a planner from an existing cache and the model's layer count.
    pub fn from_cache(cache: &KiviCache, num_layers: usize) -> Result<Self, KiviError> {
        if num_layers == 0 {
            return Err(KiviError::ZeroDimension("num_layers"));
        }
        Ok(Self {
            config: cache.config(),
            num_layers,
            num_kv_heads: cache.num_kv_heads(),
            head_dim: cache.head_dim(),
        })
    }

    // Average KV bytes per new token per layer (steady state, `seq >> R`).
    //
    // Averaged over one key-flush cycle once the value sliding window is full.
    // Individual decode steps can spike higher while keys sit in the fp32
    // residual; [`Self::next_token_bytes`] returns that per-step delta.
    pub fn marginal_bytes_per_token(&self) -> usize {
        marginal_bytes_per_layer(&self.config, self.num_kv_heads, self.head_dim)
    }

    // Exact byte delta for appending one more token at `current_tokens`.
    pub fn next_token_bytes(&self, current_tokens: usize) -> usize {
        let now = layer_kv_bytes(
            &self.config,
            self.num_kv_heads,
            self.head_dim,
            current_tokens,
        );
        let next = layer_kv_bytes(
            &self.config,
            self.num_kv_heads,
            self.head_dim,
            current_tokens + 1,
        );
        next - now
    }

    // fp32 residual buffers allocated by [`KiviCache::new`] for one agent (batch=1).
    pub fn fixed_overhead_bytes_per_agent(&self) -> usize {
        self.num_layers * residual_capacity_bytes(&self.config, self.num_kv_heads, self.head_dim)
    }

    // Estimated KV bytes for **one** agent session at `context_tokens` (all layers).
    pub fn bytes_for_context(&self, context_tokens: usize) -> usize {
        self.num_layers
            * layer_kv_bytes(
                &self.config,
                self.num_kv_heads,
                self.head_dim,
                context_tokens,
            )
    }

    // Estimated KV bytes for `agents` parallel sessions, each with `context_tokens`.
    pub fn bytes_for_agents(&self, agents: usize, context_tokens: usize) -> usize {
        agents * self.bytes_for_context(context_tokens)
    }

    // Maximum parallel agents that fit in `budget_bytes` with `context_tokens` each.
    pub fn max_agents(&self, budget_bytes: u64, context_tokens: usize) -> usize {
        let per_agent = self.bytes_for_context(context_tokens) as u64;
        if per_agent == 0 {
            return usize::MAX;
        }
        (budget_bytes / per_agent) as usize
    }

    // Returns whether `agents` sessions of length `context_tokens` fit in the budget.
    pub fn fits(&self, budget_bytes: u64, agents: usize, context_tokens: usize) -> bool {
        self.bytes_for_agents(agents, context_tokens) as u64 <= budget_bytes
    }

    // Dense fp16 KV baseline for the same agents and context (all layers).
    pub fn fp16_bytes_for_agents(&self, agents: usize, context_tokens: usize) -> usize {
        let elements = self.num_kv_heads * self.head_dim;
        2 * agents * self.num_layers * context_tokens * elements * 2
    }

    // `fp16 / kivi` byte ratio for the given deployment shape.
    pub fn compression_ratio_vs_fp16(&self, agents: usize, context_tokens: usize) -> f64 {
        let kivi = self.bytes_for_agents(agents, context_tokens);
        let fp16 = self.fp16_bytes_for_agents(agents, context_tokens);
        if kivi == 0 {
            1.0
        } else {
            fp16 as f64 / kivi as f64
        }
    }
}

// KV plan where each layer has its own bit width.
//
// Early layers can stay at 4 bits and later layers at 2 bits. Group size,
// residual length, and head shape are shared. Session counts use the same
// per-layer byte layout as [`AgentMemoryBudget`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedPrecisionBudget {
    layers: Vec<KiviConfig>,
    num_kv_heads: usize,
    head_dim: usize,
}

impl MixedPrecisionBudget {
    // One bit width per layer, from the first transformer block to the last.
    pub fn try_from_bits(
        bits_per_layer: &[u8],
        group_size: usize,
        residual_length: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, KiviError> {
        if bits_per_layer.is_empty() {
            return Err(KiviError::ZeroDimension("num_layers"));
        }
        if num_kv_heads == 0 {
            return Err(KiviError::ZeroDimension("num_kv_heads"));
        }
        let mut layers = Vec::with_capacity(bits_per_layer.len());
        for &bits in bits_per_layer {
            let config = KiviConfig::try_new(bits, group_size, residual_length)?;
            if head_dim == 0 || head_dim % config.group_size != 0 {
                return Err(KiviError::HeadDim {
                    head_dim,
                    group_size: config.group_size,
                });
            }
            layers.push(config);
        }
        Ok(Self {
            layers,
            num_kv_heads,
            head_dim,
        })
    }

    // First `early_layers` use `early_bits`. The remaining layers use `late_bits`.
    pub fn early_wide(
        num_layers: usize,
        early_layers: usize,
        early_bits: u8,
        late_bits: u8,
        group_size: usize,
        residual_length: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, KiviError> {
        if num_layers == 0 {
            return Err(KiviError::ZeroDimension("num_layers"));
        }
        if early_layers > num_layers {
            return Err(KiviError::EarlyLayers {
                early_layers,
                num_layers,
            });
        }
        let mut bits = Vec::with_capacity(num_layers);
        bits.extend(std::iter::repeat(early_bits).take(early_layers));
        bits.extend(std::iter::repeat(late_bits).take(num_layers - early_layers));
        Self::try_from_bits(&bits, group_size, residual_length, num_kv_heads, head_dim)
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn group_size(&self) -> usize {
        self.layers[0].group_size
    }

    pub fn residual_length(&self) -> usize {
        self.layers[0].residual_length
    }

    pub fn layer_bits(&self, layer: usize) -> Option<u8> {
        self.layers.get(layer).map(|config| config.bits)
    }

    pub fn layer_config(&self, layer: usize) -> Option<KiviConfig> {
        self.layers.get(layer).copied()
    }

    // Occupied KV bytes of one layer at `context_tokens`.
    pub fn bytes_for_layer(&self, layer: usize, context_tokens: usize) -> Option<usize> {
        self.layers
            .get(layer)
            .map(|config| layer_kv_bytes(config, self.num_kv_heads, self.head_dim, context_tokens))
    }

    // Sum of every layer for one agent.
    pub fn bytes_for_context(&self, context_tokens: usize) -> usize {
        self.layers.iter().fold(0, |sum, config| {
            sum + layer_kv_bytes(config, self.num_kv_heads, self.head_dim, context_tokens)
        })
    }

    pub fn bytes_for_agents(&self, agents: usize, context_tokens: usize) -> usize {
        agents * self.bytes_for_context(context_tokens)
    }

    // Average bytes added to one agent when the context grows by one token.
    //
    // This is the sum across layers, taken over one key-flush cycle once every
    // value window is full.
    pub fn marginal_bytes_per_token(&self) -> usize {
        self.layers.iter().fold(0, |sum, config| {
            sum + marginal_bytes_per_layer(config, self.num_kv_heads, self.head_dim)
        })
    }

    pub fn next_token_bytes(&self, current_tokens: usize) -> usize {
        self.bytes_for_context(current_tokens + 1) - self.bytes_for_context(current_tokens)
    }

    pub fn max_agents(&self, budget_bytes: u64, context_tokens: usize) -> usize {
        let per_agent = self.bytes_for_context(context_tokens) as u64;
        if per_agent == 0 {
            return usize::MAX;
        }
        (budget_bytes / per_agent) as usize
    }

    pub fn fits(&self, budget_bytes: u64, agents: usize, context_tokens: usize) -> bool {
        self.bytes_for_agents(agents, context_tokens) as u64 <= budget_bytes
    }

    pub fn fp16_bytes_for_agents(&self, agents: usize, context_tokens: usize) -> usize {
        let elements = self.num_kv_heads * self.head_dim;
        2 * agents * self.num_layers() * context_tokens * elements * 2
    }

    pub fn compression_ratio_vs_fp16(&self, agents: usize, context_tokens: usize) -> f64 {
        let kivi = self.bytes_for_agents(agents, context_tokens);
        let fp16 = self.fp16_bytes_for_agents(agents, context_tokens);
        if kivi == 0 {
            1.0
        } else {
            fp16 as f64 / kivi as f64
        }
    }
}

// Analytical KV bytes for one transformer layer (batch = 1).
pub(crate) fn layer_kv_bytes(
    config: &KiviConfig,
    num_kv_heads: usize,
    head_dim: usize,
    tokens: usize,
) -> usize {
    if tokens == 0 {
        return 0;
    }
    let r = config.residual_length;
    let g = config.group_size;
    let b = config.bits as usize;
    let d = head_dim;
    let h = num_kv_heads;

    let key_res = tokens % r;
    let key_quant = tokens - key_res;
    let val_res = tokens.min(r);
    let val_quant = tokens.saturating_sub(r);

    let key_codes = h * key_quant * d * b / 8;
    let key_meta = h * (key_quant / g) * d * 8;

    let val_groups = val_groups_per_token(head_dim, g);
    let val_codes = h * val_quant * d * b / 8;
    let val_meta = h * val_quant * val_groups * 8;

    let residual = h * d * 4 * (key_res + val_res);

    key_codes + key_meta + val_codes + val_meta + residual
}

pub(crate) fn marginal_bytes_per_layer(
    config: &KiviConfig,
    num_kv_heads: usize,
    head_dim: usize,
) -> usize {
    let per_head = marginal_bytes_per_kv_head(config, head_dim);
    per_head * num_kv_heads
}

fn marginal_bytes_per_kv_head(config: &KiviConfig, head_dim: usize) -> usize {
    let b = config.bits as usize;
    let g = config.group_size;
    let d = head_dim;
    let val_groups = val_groups_per_token(d, g);
    d * b / 8 + d * 8 / g + d * b / 8 + val_groups * 8
}

fn val_groups_per_token(head_dim: usize, group_size: usize) -> usize {
    (head_dim + group_size - 1) / group_size
}

fn residual_capacity_bytes(config: &KiviConfig, num_kv_heads: usize, head_dim: usize) -> usize {
    2 * config.residual_length * num_kv_heads * head_dim * 4
}

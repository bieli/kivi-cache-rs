// Compact cache snapshot for agent debugging.
//
// Records which token indices are still full precision and which just left
// that window for packed codes. The snapshot stores counts, not tensors.

use std::ops::Range;

use crate::KiviCache;

// Token-index view of one layer after an append.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheSnapshot {
    pub bits: u8,
    pub residual_length: usize,
    pub sequence_tokens: usize,
    pub quantized_key_tokens: usize,
    pub quantized_value_tokens: usize,
    pub key_residual_tokens: usize,
    pub value_residual_tokens: usize,
}

// Inclusive-exclusive token indices that moved from fp32 into packed codes
// between two snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForgottenTokens {
    pub key_start: usize,
    pub key_end: usize,
    pub value_start: usize,
    pub value_end: usize,
}

impl CacheSnapshot {
    pub fn capture(cache: &KiviCache) -> Self {
        let config = cache.config();
        Self {
            bits: config.bits,
            residual_length: config.residual_length,
            sequence_tokens: cache.sequence_length(),
            quantized_key_tokens: cache.quantized_key_tokens(),
            quantized_value_tokens: cache.quantized_value_tokens(),
            key_residual_tokens: cache.key_residual_len(),
            value_residual_tokens: cache.value_residual_len(),
        }
    }

    // Keys still stored in the fp32 residual, in token order.
    pub fn exact_keys(&self) -> Range<usize> {
        self.quantized_key_tokens..self.sequence_tokens
    }

    // Values still stored in the fp32 residual, in token order.
    pub fn exact_values(&self) -> Range<usize> {
        self.quantized_value_tokens..self.sequence_tokens
    }

    // Key indices already packed.
    pub fn packed_keys(&self) -> Range<usize> {
        0..self.quantized_key_tokens
    }

    // Value indices already packed.
    pub fn packed_values(&self) -> Range<usize> {
        0..self.quantized_value_tokens
    }

    // Tokens that left the fp32 window between `earlier` and `self`.
    //
    // `self` is the later snapshot. A reversed pair yields empty ranges.
    pub fn forgotten_since(&self, earlier: &CacheSnapshot) -> ForgottenTokens {
        ForgottenTokens {
            key_start: earlier.quantized_key_tokens,
            key_end: self.quantized_key_tokens.max(earlier.quantized_key_tokens),
            value_start: earlier.quantized_value_tokens,
            value_end: self
                .quantized_value_tokens
                .max(earlier.quantized_value_tokens),
        }
    }
}

impl ForgottenTokens {
    pub fn key_count(&self) -> usize {
        self.key_end.saturating_sub(self.key_start)
    }

    pub fn value_count(&self) -> usize {
        self.value_end.saturating_sub(self.value_start)
    }

    pub fn keys(&self) -> Range<usize> {
        self.key_start..self.key_end
    }

    pub fn values(&self) -> Range<usize> {
        self.value_start..self.value_end
    }

    pub fn any(&self) -> bool {
        self.key_count() > 0 || self.value_count() > 0
    }
}

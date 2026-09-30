use crate::error::KiviError;

// KIVI hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KiviConfig {
    // Quantized element width. Supported values are 2, 4, and 8.
    pub bits: u8,
    // Elements that share one `(scale, zero)` pair.
    // Keys group this many consecutive tokens. Values group this many consecutive channels.
    pub group_size: usize,
    // Full-precision window, in tokens. Keys flush this many tokens at a time.
    // Values keep this many newest tokens exact.
    pub residual_length: usize,
}

impl Default for KiviConfig {
    fn default() -> Self {
        Self {
            bits: 2,
            group_size: 32,
            residual_length: 128,
        }
    }
}

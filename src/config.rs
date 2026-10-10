use crate::error::KiviError;

// KIVI hyperparameters.
//
// The paper's operating point is 2 bits, group size 32, and residual length 128.
// `residual_length` must be a multiple of `group_size` so a key flush always
// contains whole per-channel groups.
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

impl KiviConfig {
    pub fn try_new(bits: u8, group_size: usize, residual_length: usize) -> Result<Self, KiviError> {
        check_bits_group(bits, group_size)?;
        if residual_length == 0 || residual_length % group_size != 0 {
            return Err(KiviError::ResidualLength {
                residual_length,
                group_size,
            });
        }
        Ok(Self {
            bits,
            group_size,
            residual_length,
        })
    }
}

pub(crate) fn check_bits_group(bits: u8, group_size: usize) -> Result<(), KiviError> {
    if !matches!(bits, 2 | 4 | 8) {
        return Err(KiviError::InvalidBits(bits));
    }
    if group_size == 0 || (group_size * bits as usize) % 32 != 0 {
        return Err(KiviError::MisalignedGroup { group_size, bits });
    }
    Ok(())
}

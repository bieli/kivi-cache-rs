use std::fmt;

// Errors returned by configuration, quantization, and cache updates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KiviError {
    // Bit width must be 2, 4, or 8.
    InvalidBits(u8),
    /// `group_size * bits` must be a multiple of 32 so every group fills whole `u32`s.
    MisalignedGroup { group_size: usize, bits: u8 },
    // Residual length must be a positive multiple of the group size.
    ResidualLength {
        residual_length: usize,
        group_size: usize,
    },
    // Per-token value groups run along the head dimension.
    HeadDim { head_dim: usize, group_size: usize },
    // Per-channel key groups run along the token axis.
    TokenGroup { tokens: usize, group_size: usize },
    // A required dimension was zero.
    ZeroDimension(&'static str),
    // A buffer does not match the shape declared by the other arguments.
    Shape { expected: usize, actual: usize },
    // Query heads must be a positive multiple of KV heads (MHA or GQA).
    QueryHeads {
        num_q_heads: usize,
        num_kv_heads: usize,
    },
}

impl fmt::Display for KiviError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KiviError::InvalidBits(bits) => {
                write!(f, "bit width {bits} is not supported; use 2, 4, or 8")
            }
            KiviError::MisalignedGroup { group_size, bits } => write!(
                f,
                "group_size {group_size} * bits {bits} must be a multiple of 32"
            ),
            KiviError::ResidualLength {
                residual_length,
                group_size,
            } => write!(
                f,
                "residual_length {residual_length} must be a positive multiple of group_size {group_size}"
            ),
            KiviError::HeadDim {
                head_dim,
                group_size,
            } => write!(
                f,
                "head_dim {head_dim} must be a positive multiple of group_size {group_size}"
            ),
            KiviError::TokenGroup { tokens, group_size } => write!(
                f,
                "token count {tokens} must be a positive multiple of group_size {group_size} for per-channel quantization"
            ),
            KiviError::ZeroDimension(name) => write!(f, "{name} must be greater than zero"),
            KiviError::Shape { expected, actual } => {
                write!(f, "expected {expected} elements, got {actual}")
            }
            KiviError::QueryHeads {
                num_q_heads,
                num_kv_heads,
            } => write!(
                f,
                "num_q_heads {num_q_heads} must be a positive multiple of num_kv_heads {num_kv_heads}"
            ),
        }
    }
}

impl std::error::Error for KiviError {}

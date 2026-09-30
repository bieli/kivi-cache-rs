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

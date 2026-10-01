// Asymmetric group quantization and bit packing.
//
// A group is summarized by its minimum (`zero`) and scale
// `(max - min) / (2^bits - 1)`. Codes are
// `round_half_even((x - zero) / scale)`, clamped to the unsigned range.
// Dequantization is `code * scale + zero`. A constant group stores scale 0
// and reconstructs as that constant, so no division by zero occurs.
//
// Packing matches the original KIVI integer layout: the first element of a
// group sits in the low bits of a `u32`, and `32 / bits` codes share a word.

use crate::config::check_bits_group;
use crate::error::KiviError;

// Which axis shares a `(scale, zero)` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantAxis {
    // Keys: one scale per channel inside a block of `group_size` tokens.
    PerChannel,
    // Values: one scale per token inside a block of `group_size` channels.
    PerToken,
}

// Packed group-quantized tensor in `[batch, heads, tokens, head_dim]` order.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    pub(crate) streams: Vec<PackedStream>,
    batch: usize,
    heads: usize,
    head_dim: usize,
    tokens: usize,
    bits: u8,
    group_size: usize,
    axis: QuantAxis,
}

// Append-only packed codes for a single KV head.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct PackedStream {
    pub codes: Vec<u32>,
    // Interleaved `[scale, zero]` pairs, in group-major order.
    pub params: Vec<f32>,
    pub tokens: usize,
}

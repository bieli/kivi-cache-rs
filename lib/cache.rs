// Streaming KIVI cache.
//
// Keys and values share a timeline but not a quantization boundary.
// Keys are flushed in blocks of `residual_length` (a multiple of the group
// size) because per-channel scales need a full group of tokens. Values are a
// sliding window: only the oldest tokens past `residual_length` are quantized,
// one token group at a time. After any append:
//
// * key residual length is `seq % residual_length`
// * value residual length is `min(seq, residual_length)`

use crate::config::KiviConfig;
use crate::error::KiviError;
use crate::quant::{self, PackedStream};

// Logical footprint of one cache, compared with a dense baseline.
//
// Byte counts describe stored payload. The residual buffers are preallocated
// to `residual_length` tokens; [`MemoryReport::residual_capacity_bytes`]
// reports that allocation, while the residual fields above count occupied tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryReport {
    pub tokens: usize,
    pub key_code_bytes: usize,
    pub key_metadata_bytes: usize,
    pub value_code_bytes: usize,
    pub value_metadata_bytes: usize,
    pub key_residual_bytes: usize,
    pub value_residual_bytes: usize,
    pub residual_capacity_bytes: usize,
    pub fp16_baseline_bytes: usize,
    pub fp32_baseline_bytes: usize,
}

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

impl QuantizedTensor {
    pub fn axis(&self) -> QuantAxis {
        self.axis
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn num_tokens(&self) -> usize {
        self.tokens
    }

    pub fn code_bytes(&self) -> usize {
        self.streams.iter().map(|s| s.codes.len() * 4).sum()
    }

    pub fn metadata_bytes(&self) -> usize {
        self.streams.iter().map(|s| s.params.len() * 4).sum()
    }

    // Packed `u32` codes in the order of `quant_and_pack_kcache` / `quant_and_pack_vcache`
    // from the PyTorch KIVI repo (`quant/new_pack.py`).
    //
    // Keys: `(batch, heads, tokens / (32/bits), head_dim)`, row-major.
    // Values: `(batch, heads, tokens, head_dim / (32/bits))`, row-major.
    // The first element of each packed word sits in the low bits, matching `pack_tensor`.
    pub fn pytorch_codes(&self) -> Vec<u32> {
        let feat = 32 / self.bits as usize;
        let mut out = Vec::new();
        match self.axis {
            QuantAxis::PerChannel => {
                let t_pack = self.tokens / feat;
                let ints_per_channel = self.group_size * self.bits as usize / 32;
                for stream in &self.streams {
                    for tp in 0..t_pack {
                        let g = tp / ints_per_channel;
                        let w = tp % ints_per_channel;
                        for d in 0..self.head_dim {
                            let idx = (g * self.head_dim + d) * ints_per_channel + w;
                            out.push(stream.codes[idx]);
                        }
                    }
                }
            }
            QuantAxis::PerToken => {
                let n_ch = self.head_dim / self.group_size;
                let ints_per_group = self.group_size * self.bits as usize / 32;
                let d_pack = self.head_dim / feat;
                for stream in &self.streams {
                    for t in 0..self.tokens {
                        for dp in 0..d_pack {
                            let cg = dp / ints_per_group;
                            let w = dp % ints_per_group;
                            let idx = (t * n_ch + cg) * ints_per_group + w;
                            out.push(stream.codes[idx]);
                        }
                    }
                }
            }
        }
        out
    }

    // Interleaved `(scale, zero)` pairs, group-major, one head after another.
    // `zero` is the group minimum. This is the PyTorch `(scale, mn)` pair.
    pub fn scale_zero(&self) -> Vec<f32> {
        self.streams
            .iter()
            .flat_map(|stream| stream.params.iter().copied())
            .collect()
    }

    // Reconstruct a dense `[batch, heads, tokens, head_dim]` tensor.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.batch * self.heads * self.tokens * self.head_dim];
        for b in 0..self.batch {
            for h in 0..self.heads {
                let hi = b * self.heads + h;
                let start = hi * self.tokens * self.head_dim;
                let dst = &mut out[start..start + self.tokens * self.head_dim];
                match self.axis {
                    QuantAxis::PerChannel => dequant_keys(
                        &self.streams[hi],
                        self.bits,
                        self.group_size,
                        self.head_dim,
                        dst,
                    ),
                    QuantAxis::PerToken => dequant_values(
                        &self.streams[hi],
                        self.bits,
                        self.group_size,
                        self.head_dim,
                        dst,
                    ),
                }
            }
        }
        out
    }
}

// Quantize keys per channel. `tokens` must be a multiple of `group_size`.
pub fn quantize_per_channel(
    data: &[f32],
    batch: usize,
    heads: usize,
    tokens: usize,
    head_dim: usize,
    bits: u8,
    group_size: usize,
) -> Result<QuantizedTensor, KiviError> {
    check_quant_args(data, batch, heads, tokens, head_dim, bits, group_size, true)?;
    let mut streams = Vec::with_capacity(batch * heads);
    for b in 0..batch {
        for h in 0..heads {
            let hi = b * heads + h;
            let start = hi * tokens * head_dim;
            let mut stream = PackedStream::default();
            append_key_block(
                &mut stream,
                &data[start..start + tokens * head_dim],
                tokens,
                head_dim,
                bits,
                group_size,
            );
            streams.push(stream);
        }
    }
    Ok(QuantizedTensor {
        streams,
        batch,
        heads,
        head_dim,
        tokens,
        bits,
        group_size,
        axis: QuantAxis::PerChannel,
    })
}

// Quantize values per token. `head_dim` must be a multiple of `group_size`.
pub fn quantize_per_token(
    data: &[f32],
    batch: usize,
    heads: usize,
    tokens: usize,
    head_dim: usize,
    bits: u8,
    group_size: usize,
) -> Result<QuantizedTensor, KiviError> {
    check_quant_args(
        data, batch, heads, tokens, head_dim, bits, group_size, false,
    )?;
    let mut streams = Vec::with_capacity(batch * heads);
    for b in 0..batch {
        for h in 0..heads {
            let hi = b * heads + h;
            let start = hi * tokens * head_dim;
            let mut stream = PackedStream::default();
            append_value_block(
                &mut stream,
                &data[start..start + tokens * head_dim],
                tokens,
                head_dim,
                bits,
                group_size,
            );
            streams.push(stream);
        }
    }
    Ok(QuantizedTensor {
        streams,
        batch,
        heads,
        head_dim,
        tokens,
        bits,
        group_size,
        axis: QuantAxis::PerToken,
    })
}

fn check_quant_args(
    data: &[f32],
    batch: usize,
    heads: usize,
    tokens: usize,
    head_dim: usize,
    bits: u8,
    group_size: usize,
    per_channel: bool,
) -> Result<(), KiviError> {
    check_bits_group(bits, group_size)?;
    if batch == 0 {
        return Err(KiviError::ZeroDimension("batch"));
    }
    if heads == 0 {
        return Err(KiviError::ZeroDimension("heads"));
    }
    if head_dim == 0 {
        return Err(KiviError::ZeroDimension("head_dim"));
    }
    if per_channel {
        if tokens == 0 || tokens % group_size != 0 {
            return Err(KiviError::TokenGroup { tokens, group_size });
        }
    } else if head_dim % group_size != 0 {
        return Err(KiviError::HeadDim {
            head_dim,
            group_size,
        });
    }
    let expected = batch * heads * tokens * head_dim;
    if data.len() != expected {
        return Err(KiviError::Shape {
            expected,
            actual: data.len(),
        });
    }
    Ok(())
}

pub(crate) fn append_key_block(
    stream: &mut PackedStream,
    data: &[f32],
    tokens: usize,
    head_dim: usize,
    bits: u8,
    group_size: usize,
) {
    debug_assert_eq!(data.len(), tokens * head_dim);
    debug_assert!(tokens % group_size == 0);
    if tokens == 0 {
        return;
    }
    let levels = ((1u32 << bits) - 1) as f32;
    let n_groups = tokens / group_size;
    let ints_per_channel = group_size * bits as usize / 32;
    stream.codes.reserve(n_groups * head_dim * ints_per_channel);
    stream.params.reserve(n_groups * head_dim * 2);

    let mut mn = vec![0.0f32; head_dim];
    let mut mx = vec![0.0f32; head_dim];
    for g in 0..n_groups {
        mn.fill(f32::INFINITY);
        mx.fill(f32::NEG_INFINITY);
        let base = g * group_size * head_dim;
        for t in 0..group_size {
            let row = &data[base + t * head_dim..base + (t + 1) * head_dim];
            for d in 0..head_dim {
                mn[d] = mn[d].min(row[d]);
                mx[d] = mx[d].max(row[d]);
            }
        }
        for d in 0..head_dim {
            let zero = mn[d];
            let scale = scale_of(mx[d] - zero, levels);
            stream.params.push(scale);
            stream.params.push(zero);
            let mut word = 0u32;
            let mut filled = 0u32;
            for t in 0..group_size {
                let x = data[base + t * head_dim + d];
                word |= quant_code(x, zero, scale, levels) << filled;
                filled += u32::from(bits);
                if filled == 32 {
                    stream.codes.push(word);
                    word = 0;
                    filled = 0;
                }
            }
            debug_assert_eq!(filled, 0);
        }
    }
    stream.tokens += tokens;
}

pub(crate) fn append_value_block(
    stream: &mut PackedStream,
    data: &[f32],
    tokens: usize,
    head_dim: usize,
    bits: u8,
    group_size: usize,
) {
    debug_assert_eq!(data.len(), tokens * head_dim);
    debug_assert!(head_dim % group_size == 0);
    if tokens == 0 {
        return;
    }
    let levels = ((1u32 << bits) - 1) as f32;
    let n_ch = head_dim / group_size;
    let ints_per_group = group_size * bits as usize / 32;
    stream.codes.reserve(tokens * n_ch * ints_per_group);
    stream.params.reserve(tokens * n_ch * 2);

    for t in 0..tokens {
        let row = &data[t * head_dim..(t + 1) * head_dim];
        for cg in 0..n_ch {
            let group = &row[cg * group_size..(cg + 1) * group_size];
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for &x in group {
                mn = mn.min(x);
                mx = mx.max(x);
            }
            let scale = scale_of(mx - mn, levels);
            stream.params.push(scale);
            stream.params.push(mn);
            let mut word = 0u32;
            let mut filled = 0u32;
            for &x in group {
                word |= quant_code(x, mn, scale, levels) << filled;
                filled += u32::from(bits);
                if filled == 32 {
                    stream.codes.push(word);
                    word = 0;
                    filled = 0;
                }
            }
            debug_assert_eq!(filled, 0);
        }
    }
    stream.tokens += tokens;
}

pub(crate) fn dequant_keys(
    stream: &PackedStream,
    bits: u8,
    group_size: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), stream.tokens * head_dim);
    if stream.tokens == 0 {
        return;
    }
    let ints_per_channel = group_size * bits as usize / 32;
    let mask = (1u32 << bits) - 1;
    let n_groups = stream.tokens / group_size;
    for g in 0..n_groups {
        for d in 0..head_dim {
            let meta = 2 * (g * head_dim + d);
            let scale = stream.params[meta];
            let zero = stream.params[meta + 1];
            let base = (g * head_dim + d) * ints_per_channel;
            let mut local = 0;
            for i in 0..ints_per_channel {
                let mut word = stream.codes[base + i];
                let mut left = 32u32;
                while left > 0 {
                    let code = (word & mask) as f32;
                    let t = g * group_size + local;
                    out[t * head_dim + d] = code.mul_add(scale, zero);
                    word >>= u32::from(bits);
                    left -= u32::from(bits);
                    local += 1;
                }
            }
        }
    }
}

pub(crate) fn dequant_values(
    stream: &PackedStream,
    bits: u8,
    group_size: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), stream.tokens * head_dim);
    if stream.tokens == 0 {
        return;
    }
    let n_ch = head_dim / group_size;
    let ints_per_group = group_size * bits as usize / 32;
    let ints_per_token = n_ch * ints_per_group;
    let mask = (1u32 << bits) - 1;
    for t in 0..stream.tokens {
        for cg in 0..n_ch {
            let meta = 2 * (t * n_ch + cg);
            let scale = stream.params[meta];
            let zero = stream.params[meta + 1];
            let base = t * ints_per_token + cg * ints_per_group;
            let mut local = 0;
            for i in 0..ints_per_group {
                let mut word = stream.codes[base + i];
                let mut left = 32u32;
                while left > 0 {
                    let code = (word & mask) as f32;
                    out[t * head_dim + cg * group_size + local] = code.mul_add(scale, zero);
                    word >>= u32::from(bits);
                    left -= u32::from(bits);
                    local += 1;
                }
            }
        }
    }
}

fn scale_of(span: f32, levels: f32) -> f32 {
    if span == 0.0 {
        0.0
    } else {
        span / levels
    }
}

fn quant_code(x: f32, zero: f32, scale: f32, levels: f32) -> u32 {
    if scale == 0.0 {
        return 0;
    }
    let mut rounded = round_half_even((x - zero) / scale);
    if rounded < 0.0 {
        rounded = 0.0;
    } else if rounded > levels {
        rounded = levels;
    }
    rounded as u32
}

// Round half to even. Quantized codes are non-negative; this matches PyTorch `round`.
pub(crate) fn round_half_even(x: f32) -> f32 {
    let floor = x.floor();
    let frac = x - floor;
    if frac > 0.5 {
        floor + 1.0
    } else if frac < 0.5 || (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

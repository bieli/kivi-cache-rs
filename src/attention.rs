// Fused dequantize-and-attend kernels.
//
// Scores and the value mix never materialize a dense `[seq, head_dim]` cache.
// The 2-bit, group-32 path is the paper's default tile: one channel of a key
// group is two `u32`s, and one value channel-group is the same.

use crate::quant::PackedStream;

pub(crate) struct HeadView<'a> {
    pub keys: &'a PackedStream,
    pub values: &'a PackedStream,
    pub key_residual: &'a [f32],
    pub value_residual: &'a [f32],
    pub key_res_len: usize,
    pub value_res_len: usize,
}

pub(crate) fn attend_head(
    head: HeadView<'_>,
    query: &[f32],
    bits: u8,
    group_size: usize,
    head_dim: usize,
    scale: f32,
    logits: &mut [f32],
    out: &mut [f32],
) {
    attend_head_with_suffix(
        head,
        &[],
        &[],
        0,
        query,
        bits,
        group_size,
        head_dim,
        scale,
        logits,
        out,
    );
}

// Score `head` (shared prefix) and then a session-local suffix.
//
// Logits are in token order: quantized prefix, prefix residual, suffix.
// The suffix is full precision and is not written back into the prefix.
pub(crate) fn attend_head_with_suffix(
    head: HeadView<'_>,
    suffix_keys: &[f32],
    suffix_values: &[f32],
    suffix_len: usize,
    query: &[f32],
    bits: u8,
    group_size: usize,
    head_dim: usize,
    scale: f32,
    logits: &mut [f32],
    out: &mut [f32],
) {
    let kq = head.keys.tokens;
    let prefix = kq + head.key_res_len;
    debug_assert_eq!(logits.len(), prefix + suffix_len);
    debug_assert_eq!(prefix, head.values.tokens + head.value_res_len);
    debug_assert_eq!(suffix_keys.len(), suffix_len * head_dim);
    debug_assert_eq!(suffix_values.len(), suffix_len * head_dim);

    key_logits(
        head.keys,
        bits,
        group_size,
        head_dim,
        query,
        &mut logits[..kq],
    );
    dense_logits(
        head.key_residual,
        head.key_res_len,
        head_dim,
        query,
        &mut logits[kq..prefix],
    );
    dense_logits(
        suffix_keys,
        suffix_len,
        head_dim,
        query,
        &mut logits[prefix..],
    );
    for logit in logits.iter_mut() {
        *logit *= scale;
    }
    softmax(logits);

    out.fill(0.0);
    let vq = head.values.tokens;
    value_mix(head.values, bits, group_size, head_dim, &logits[..vq], out);
    dense_mix(
        head.value_residual,
        head.value_res_len,
        head_dim,
        &logits[vq..prefix],
        out,
    );
    dense_mix(suffix_values, suffix_len, head_dim, &logits[prefix..], out);
}

fn key_logits(
    stream: &PackedStream,
    bits: u8,
    group_size: usize,
    head_dim: usize,
    query: &[f32],
    logits: &mut [f32],
) {
    if stream.tokens == 0 {
        return;
    }
    if bits == 2 && group_size == 32 {
        key_logits_2bit_g32(stream, head_dim, query, logits);
    } else {
        key_logits_generic(stream, bits, group_size, head_dim, query, logits);
    }
}

fn value_mix(
    stream: &PackedStream,
    bits: u8,
    group_size: usize,
    head_dim: usize,
    weights: &[f32],
    out: &mut [f32],
) {
    if stream.tokens == 0 {
        return;
    }
    if bits == 2 && group_size == 32 {
        value_mix_2bit_g32(stream, head_dim, weights, out);
    } else {
        value_mix_generic(stream, bits, group_size, head_dim, weights, out);
    }
}

fn key_logits_generic(
    stream: &PackedStream,
    bits: u8,
    group_size: usize,
    head_dim: usize,
    query: &[f32],
    logits: &mut [f32],
) {
    let ints_per_channel = group_size * bits as usize / 32;
    let mask = (1u32 << bits) - 1;
    let n_groups = stream.tokens / group_size;
    for g in 0..n_groups {
        let logits_g = &mut logits[g * group_size..(g + 1) * group_size];
        logits_g.fill(0.0);
        for d in 0..head_dim {
            let q = query[d];
            let meta = 2 * (g * head_dim + d);
            let scale = stream.params[meta];
            let zero = stream.params[meta + 1];
            let base = (g * head_dim + d) * ints_per_channel;
            let mut local = 0;
            for i in 0..ints_per_channel {
                let mut word = stream.codes[base + i];
                let mut left = 32u32;
                while left > 0 {
                    let recon = ((word & mask) as f32).mul_add(scale, zero);
                    logits_g[local] = recon.mul_add(q, logits_g[local]);
                    word >>= u32::from(bits);
                    left -= u32::from(bits);
                    local += 1;
                }
            }
        }
    }
}

fn key_logits_2bit_g32(stream: &PackedStream, head_dim: usize, query: &[f32], logits: &mut [f32]) {
    let n_groups = stream.tokens / 32;
    for g in 0..n_groups {
        let logits_g = &mut logits[g * 32..(g + 1) * 32];
        logits_g.fill(0.0);
        for d in 0..head_dim {
            let q = query[d];
            let meta = 2 * (g * head_dim + d);
            let scale = stream.params[meta];
            let zero = stream.params[meta + 1];
            let base = (g * head_dim + d) * 2;
            add16(stream.codes[base], scale, zero, q, &mut logits_g[..16]);
            add16(
                stream.codes[base + 1],
                scale,
                zero,
                q,
                &mut logits_g[16..32],
            );
        }
    }
}

fn value_mix_generic(
    stream: &PackedStream,
    bits: u8,
    group_size: usize,
    head_dim: usize,
    weights: &[f32],
    out: &mut [f32],
) {
    let n_ch = head_dim / group_size;
    let ints_per_group = group_size * bits as usize / 32;
    let ints_per_token = n_ch * ints_per_group;
    let mask = (1u32 << bits) - 1;
    for t in 0..stream.tokens {
        let w = weights[t];
        if w == 0.0 {
            continue;
        }
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
                    let recon = ((word & mask) as f32).mul_add(scale, zero);
                    let d = cg * group_size + local;
                    out[d] = recon.mul_add(w, out[d]);
                    word >>= u32::from(bits);
                    left -= u32::from(bits);
                    local += 1;
                }
            }
        }
    }
}

fn value_mix_2bit_g32(stream: &PackedStream, head_dim: usize, weights: &[f32], out: &mut [f32]) {
    let n_ch = head_dim / 32;
    for t in 0..stream.tokens {
        let w = weights[t];
        if w == 0.0 {
            continue;
        }
        for cg in 0..n_ch {
            let meta = 2 * (t * n_ch + cg);
            let scale = stream.params[meta];
            let zero = stream.params[meta + 1];
            let base = (t * n_ch + cg) * 2;
            acc16(
                stream.codes[base],
                scale,
                zero,
                w,
                &mut out[cg * 32..cg * 32 + 16],
            );
            acc16(
                stream.codes[base + 1],
                scale,
                zero,
                w,
                &mut out[cg * 32 + 16..cg * 32 + 32],
            );
        }
    }
}

#[inline]
fn add16(mut word: u32, scale: f32, zero: f32, q: f32, logits: &mut [f32]) {
    for slot in logits {
        let recon = ((word & 3) as f32).mul_add(scale, zero);
        *slot = recon.mul_add(q, *slot);
        word >>= 2;
    }
}

#[inline]
fn acc16(mut word: u32, scale: f32, zero: f32, w: f32, out: &mut [f32]) {
    for slot in out {
        let recon = ((word & 3) as f32).mul_add(scale, zero);
        *slot = recon.mul_add(w, *slot);
        word >>= 2;
    }
}

fn dense_logits(rows: &[f32], n_tokens: usize, dim: usize, query: &[f32], logits: &mut [f32]) {
    debug_assert_eq!(rows.len(), n_tokens * dim);
    debug_assert_eq!(logits.len(), n_tokens);
    for t in 0..n_tokens {
        let row = &rows[t * dim..(t + 1) * dim];
        let mut dot = 0.0f32;
        for d in 0..dim {
            dot = query[d].mul_add(row[d], dot);
        }
        logits[t] = dot;
    }
}

fn dense_mix(rows: &[f32], n_tokens: usize, dim: usize, weights: &[f32], out: &mut [f32]) {
    debug_assert_eq!(rows.len(), n_tokens * dim);
    debug_assert_eq!(weights.len(), n_tokens);
    for t in 0..n_tokens {
        let w = weights[t];
        if w == 0.0 {
            continue;
        }
        let row = &rows[t * dim..(t + 1) * dim];
        for d in 0..dim {
            out[d] = row[d].mul_add(w, out[d]);
        }
    }
}

fn softmax(xs: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &x in xs.iter() {
        if x > max {
            max = x;
        }
    }
    let mut sum = 0.0f32;
    for x in xs.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = 1.0 / sum;
    for x in xs.iter_mut() {
        *x *= inv;
    }
}

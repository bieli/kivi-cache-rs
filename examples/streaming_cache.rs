// Stream a synthetic layer through KIVI and report the footprint.
//
// Keys with one outlier channel are also quantized both ways, which is the
// observation the paper is built on: per-channel scales keep the quiet
// channels, per-token scales do not.

use kivi_cache::{quantize_per_channel, quantize_per_token, KiviCache, KiviConfig};

fn main() {
    let tokens = 32;
    let dim = 32;
    let mut data = vec![0.0f32; tokens * dim];
    for t in 0..tokens {
        data[t * dim] = 100.0;
        for d in 1..dim {
            data[t * dim + d] = d as f32 / dim as f32;
        }
    }
    let per_channel = quantize_per_channel(&data, 1, 1, tokens, dim, 2, 32)
        .unwrap()
        .dequantize();
    let per_token = quantize_per_token(&data, 1, 1, tokens, dim, 2, 32)
        .unwrap()
        .dequantize();
    println!(
        "quiet-channel MAE  per-channel {:>8.4}   per-token {:>8.4}",
        mae_quiet(&data, &per_channel, tokens, dim),
        mae_quiet(&data, &per_token, tokens, dim)
    );

    let cfg = KiviConfig::default();
    let batch = 1;
    let heads = 8;
    let dim = 128;
    let seq = 2048;
    let mut cache = KiviCache::new(cfg, batch, heads, dim).unwrap();
    let n = batch * heads * seq * dim;
    let keys = lcg(n, 1);
    let values = lcg(n, 2);

    // Prefill, then a few decode steps. Same groups as one shot.
    let prompt = seq - 4;
    cache
        .append(
            &slice_tokens(&keys, batch, heads, seq, dim, 0, prompt),
            &slice_tokens(&values, batch, heads, seq, dim, 0, prompt),
            prompt,
        )
        .unwrap();
    for step in 0..4 {
        let i = prompt + step;
        cache
            .append(
                &slice_tokens(&keys, batch, heads, seq, dim, i, 1),
                &slice_tokens(&values, batch, heads, seq, dim, i, 1),
                1,
            )
            .unwrap();
    }

    let report = cache.memory();
    println!("sequence                 {}", report.tokens);
    println!("key residual tokens      {}", cache.key_residual_len());
    println!("value residual tokens    {}", cache.value_residual_len());
    println!("quantized key tokens     {}", cache.quantized_key_tokens());
    println!(
        "quantized value tokens   {}",
        cache.quantized_value_tokens()
    );
    println!(
        "compressed               {} bytes",
        report.compressed_bytes()
    );
    println!(
        "fp16 baseline            {} bytes",
        report.fp16_baseline_bytes
    );
    println!(
        "compression vs fp16      {:.2}x",
        report.compression_ratio_vs_fp16()
    );

    let (k_hat, _) = cache.dequantize();
    let mut err = 0.0f32;
    for (a, b) in k_hat.iter().zip(&keys) {
        err = err.max((a - b).abs());
    }
    println!("max |K - dequant(K)|     {err:.4}");
}

/// Gather `[batch, heads, count, dim]` from a full `[batch, heads, total, dim]` buffer.
fn slice_tokens(
    full: &[f32],
    batch: usize,
    heads: usize,
    total: usize,
    dim: usize,
    offset: usize,
    count: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(batch * heads * count * dim);
    for b in 0..batch {
        for h in 0..heads {
            let base = ((b * heads + h) * total + offset) * dim;
            out.extend_from_slice(&full[base..base + count * dim]);
        }
    }
    out
}

fn mae_quiet(orig: &[f32], recon: &[f32], tokens: usize, dim: usize) -> f32 {
    let mut acc = 0.0;
    let mut n = 0.0;
    for t in 0..tokens {
        for d in 1..dim {
            acc += (orig[t * dim + d] - recon[t * dim + d]).abs();
            n += 1.0;
        }
    }
    acc / n
}

fn lcg(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

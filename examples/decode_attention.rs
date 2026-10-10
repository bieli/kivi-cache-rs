// Compare decode attention against a full-precision cache.
//
// Prefill attention in a real model stays in fp32. This example stores that
// prefill in KIVI, then runs four decode steps. Each step appends the new K
// and V before scoring, which is the order in Algorithm 1.

use kivi_cache::{KiviCache, KiviConfig};

fn main() {
    let cfg = KiviConfig::try_new(2, 32, 64).unwrap();
    let dim = 64;
    let heads = 4;
    let prompt = 96;
    let steps = 4;
    let total = prompt + steps;
    let scale = 1.0 / (dim as f32).sqrt();

    let mut cache = KiviCache::new(cfg, 1, heads, dim).unwrap();
    let keys = lcg(heads * total * dim, 4);
    let values = lcg(heads * total * dim, 5);
    let queries = lcg(heads * steps * dim, 6);

    cache
        .append(
            &slice_tokens(&keys, heads, total, dim, 0, prompt),
            &slice_tokens(&values, heads, total, dim, 0, prompt),
            prompt,
        )
        .unwrap();

    println!(
        "{:>6} {:>12} {:>12} {:>10}",
        "step", "cosine", "max_abs", "k_residual"
    );
    for step in 0..steps {
        let t = prompt + step;
        cache
            .append(
                &slice_tokens(&keys, heads, total, dim, t, 1),
                &slice_tokens(&values, heads, total, dim, t, 1),
                1,
            )
            .unwrap();
        let q = &queries[step * heads * dim..(step + 1) * heads * dim];
        let got = cache.attention(q, heads, scale).unwrap();
        let expect = fp32_attention(
            q,
            &slice_tokens(&keys, heads, total, dim, 0, t + 1),
            &slice_tokens(&values, heads, total, dim, 0, t + 1),
            heads,
            t + 1,
            dim,
            scale,
        );
        println!(
            "{:>6} {:>12.6} {:>12.6} {:>10}",
            step + 1,
            cosine(&got, &expect),
            max_abs(&got, &expect),
            cache.key_residual_len()
        );
    }

    let report = cache.memory();
    println!(
        "\n{} tokens, {:.2}x vs fp16 ({} / {} bytes)",
        report.tokens,
        report.compression_ratio_vs_fp16(),
        report.compressed_bytes(),
        report.fp16_baseline_bytes
    );
}

fn slice_tokens(
    full: &[f32],
    heads: usize,
    total: usize,
    dim: usize,
    offset: usize,
    count: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(heads * count * dim);
    for h in 0..heads {
        let base = (h * total + offset) * dim;
        out.extend_from_slice(&full[base..base + count * dim]);
    }
    out
}

fn fp32_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    heads: usize,
    seq: usize,
    dim: usize,
    scale: f32,
) -> Vec<f32> {
    // keys/values layout: [heads, seq, dim]
    let mut out = vec![0.0; heads * dim];
    for h in 0..heads {
        let q = &query[h * dim..(h + 1) * dim];
        let mut logits = vec![0.0; seq];
        for t in 0..seq {
            let mut dot = 0.0;
            for d in 0..dim {
                dot += q[d] * keys[(h * seq + t) * dim + d];
            }
            logits[t] = dot * scale;
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for x in &mut logits {
            *x = (*x - max).exp();
            sum += *x;
        }
        for d in 0..dim {
            let mut acc = 0.0;
            for t in 0..seq {
                acc += (logits[t] / sum) * values[(h * seq + t) * dim + d];
            }
            out[h * dim + d] = acc;
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / na.sqrt() / nb.sqrt()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
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

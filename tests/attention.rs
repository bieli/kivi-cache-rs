use kivi_cache::{KiviCache, KiviConfig};

fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn reference(q: &[f32], k: &[f32], v: &[f32], seq: usize, dim: usize, scale: f32) -> Vec<f32> {
    let mut logits = vec![0.0; seq];
    for t in 0..seq {
        let mut dot = 0.0;
        for d in 0..dim {
            dot += q[d] * k[t * dim + d];
        }
        logits[t] = dot * scale;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in &mut logits {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in &mut logits {
        *x /= sum;
    }
    let mut out = vec![0.0; dim];
    for t in 0..seq {
        for d in 0..dim {
            out[d] += logits[t] * v[t * dim + d];
        }
    }
    out
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn assert_matches_dequantized(bits: u8, group: usize, residual: usize, dim: usize, seq: usize) {
    let cfg = KiviConfig::try_new(bits, group, residual).unwrap();
    let mut cache = KiviCache::new(cfg, 1, 1, dim).unwrap();
    let keys = fill(seq * dim, 21);
    let values = fill(seq * dim, 22);
    cache.append(&keys, &values, seq).unwrap();
    let (k_hat, v_hat) = cache.dequantize();
    let query = fill(dim, 23);
    let scale = 1.0 / (dim as f32).sqrt();
    let got = cache.attention(&query, 1, scale).unwrap();
    let expect = reference(&query, &k_hat, &v_hat, seq, dim, scale);
    let err = max_abs(&got, &expect);
    assert!(err < 1e-4, "bits={bits} group={group} err={err}");
}

#[test]
fn generic_kernel_matches_dequantized_attention() {
    assert_matches_dequantized(2, 16, 32, 16, 40);
    assert_matches_dequantized(4, 32, 64, 32, 70);
}

#[test]
fn fast_2bit_group32_kernel_matches_dequantized_attention() {
    assert_matches_dequantized(2, 32, 64, 64, 100);
}

#[test]
fn grouped_query_attention_repeats_kv_heads() {
    let dim = 32;
    let cfg = KiviConfig::try_new(2, 32, 64).unwrap();
    let batch = 2;
    let kv = 2;
    let q_heads = 4;
    let seq = 48;
    let mut cache = KiviCache::new(cfg, batch, kv, dim).unwrap();
    let keys = fill(batch * kv * seq * dim, 31);
    let values = fill(batch * kv * seq * dim, 32);
    cache.append(&keys, &values, seq).unwrap();
    let (k_hat, v_hat) = cache.dequantize();
    let query = fill(batch * q_heads * dim, 33);
    let scale = 0.125;
    let got = cache.attention(&query, q_heads, scale).unwrap();

    for b in 0..batch {
        for qh in 0..q_heads {
            let kvh = qh / 2;
            let q = &query[(b * q_heads + qh) * dim..(b * q_heads + qh + 1) * dim];
            let k = &k_hat[(b * kv + kvh) * seq * dim..(b * kv + kvh + 1) * seq * dim];
            let v = &v_hat[(b * kv + kvh) * seq * dim..(b * kv + kvh + 1) * seq * dim];
            let expect = reference(q, k, v, seq, dim, scale);
            let y = &got[(b * q_heads + qh) * dim..(b * q_heads + qh + 1) * dim];
            assert!(max_abs(y, &expect) < 1e-4);
        }
    }
}

#[test]
fn empty_cache_attention_is_zero() {
    let cache = KiviCache::new(KiviConfig::default(), 1, 2, 128).unwrap();
    let y = cache.attention(&[0.2; 2 * 128], 2, 1.0).unwrap();
    assert!(y.iter().all(|v| *v == 0.0));
}

#[test]
fn rejects_query_head_count_that_is_not_a_multiple() {
    let cache = KiviCache::new(KiviConfig::try_new(2, 16, 32).unwrap(), 1, 2, 16).unwrap();
    assert!(cache.attention(&[0.0; 3 * 16], 3, 1.0).is_err());
}

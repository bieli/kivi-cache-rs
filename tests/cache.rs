use kivi_cache::{quantize_per_channel, quantize_per_token, KiviCache, KiviConfig, KiviError};

fn config() -> KiviConfig {
    KiviConfig::try_new(2, 16, 32).unwrap()
}

fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

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

fn append_chunks(
    cache: &mut KiviCache,
    keys: &[f32],
    values: &[f32],
    total: usize,
    chunks: &[usize],
) {
    let heads = cache.num_kv_heads();
    let batch = cache.batch_size();
    let dim = cache.head_dim();
    let mut off = 0;
    for &n in chunks {
        cache
            .append(
                &slice_tokens(keys, batch, heads, total, dim, off, n),
                &slice_tokens(values, batch, heads, total, dim, off, n),
                n,
            )
            .unwrap();
        off += n;
    }
}

#[test]
fn streaming_boundaries_match_algorithm_1() {
    let dim = 16;
    let mut cache = KiviCache::new(config(), 2, 3, dim).unwrap();
    let steps = [10usize, 20, 5, 100];
    let total: usize = steps.iter().sum();
    let n = 2 * 3 * total * dim;
    let keys = fill(n, 1);
    let values = fill(n, 2);
    append_chunks(&mut cache, &keys, &values, total, &steps);

    let r = 32;
    assert_eq!(cache.sequence_length(), total);
    assert_eq!(cache.key_residual_len(), total % r);
    assert_eq!(cache.value_residual_len(), r.min(total));
    assert_eq!(cache.quantized_key_tokens(), total - total % r);
    assert_eq!(cache.quantized_value_tokens(), total - r.min(total));
}

#[test]
fn chunked_appends_match_token_by_token() {
    let dim = 16;
    let batch = 2;
    let heads = 3;
    let total = 70;
    let n = batch * heads * total * dim;
    let keys = fill(n, 7);
    let values = fill(n, 9);

    let mut streamed = KiviCache::new(config(), batch, heads, dim).unwrap();
    append_chunks(&mut streamed, &keys, &values, total, &vec![1; total]);

    let mut chunked = KiviCache::new(config(), batch, heads, dim).unwrap();
    append_chunks(&mut chunked, &keys, &values, total, &[1, 50, 7, 12]);

    let mut once = KiviCache::new(config(), batch, heads, dim).unwrap();
    once.append(&keys, &values, total).unwrap();

    let (sk, sv) = streamed.dequantize();
    let (ck, cv) = chunked.dequantize();
    let (ok, ov) = once.dequantize();
    assert_eq!(sk, ck);
    assert_eq!(sv, cv);
    assert_eq!(sk, ok);
    assert_eq!(sv, ov);
}

#[test]
fn quantized_prefix_matches_one_shot_and_residual_is_exact() {
    let dim = 16;
    let total = 80;
    let keys = fill(total * dim, 3);
    let values = fill(total * dim, 4);
    let mut cache = KiviCache::new(config(), 1, 1, dim).unwrap();
    cache.append(&keys, &values, total).unwrap();

    let kq = cache.quantized_key_tokens();
    let vq = cache.quantized_value_tokens();
    let (k_hat, v_hat) = cache.dequantize();

    let k_ref = quantize_per_channel(&keys[..kq * dim], 1, 1, kq, dim, 2, 16)
        .unwrap()
        .dequantize();
    assert_eq!(&k_hat[..kq * dim], k_ref.as_slice());
    assert_eq!(&k_hat[kq * dim..], &keys[kq * dim..]);

    let v_ref = quantize_per_token(&values[..vq * dim], 1, 1, vq, dim, 2, 16)
        .unwrap()
        .dequantize();
    assert_eq!(&v_hat[..vq * dim], v_ref.as_slice());
    assert_eq!(&v_hat[vq * dim..], &values[vq * dim..]);
}

#[test]
fn empty_append_and_shape_errors() {
    let mut cache = KiviCache::new(config(), 1, 1, 16).unwrap();
    cache.append(&[], &[], 0).unwrap();
    assert_eq!(cache.sequence_length(), 0);
    let err = cache.append(&[0.0; 4], &[0.0; 16], 1).unwrap_err();
    assert_eq!(
        err,
        KiviError::Shape {
            expected: 16,
            actual: 4
        }
    );
}

#[test]
fn config_rejects_residual_that_is_not_a_group_multiple() {
    assert!(matches!(
        KiviConfig::try_new(2, 32, 100),
        Err(KiviError::ResidualLength { .. })
    ));
    assert!(KiviCache::new(KiviConfig::default(), 1, 1, 96).is_ok());
    assert!(matches!(
        KiviCache::new(KiviConfig::default(), 1, 1, 40),
        Err(KiviError::HeadDim { .. })
    ));
}

#[test]
fn same_prompt_builds_the_same_cache_twice() {
    let cfg = KiviConfig::try_new(2, 32, 32).unwrap();
    let mut keys = fill(64 * 32, 4);
    let values = fill(64 * 32, 5);
    keys[3] = 0.5;
    keys[4] = 1.5;
    keys[5] = 2.5;
    let query_a = fill(32, 8);
    let query_b = fill(32, 9);
    let scale = 0.25;

    let packed_k = quantize_per_channel(&keys, 1, 1, 64, 32, 2, 32).unwrap();
    let packed_k_again = quantize_per_channel(&keys, 1, 1, 64, 32, 2, 32).unwrap();
    assert_eq!(packed_k, packed_k_again);
    let packed_v = quantize_per_token(&values, 1, 1, 64, 32, 2, 32).unwrap();
    let packed_v_again = quantize_per_token(&values, 1, 1, 64, 32, 2, 32).unwrap();
    assert_eq!(packed_v, packed_v_again);

    let mut first = KiviCache::new(cfg, 1, 1, 32).unwrap();
    let mut second = KiviCache::new(cfg, 1, 1, 32).unwrap();
    first.append(&keys, &values, 64).unwrap();
    second.append(&keys, &values, 64).unwrap();
    assert_eq!(first.dequantize(), second.dequantize());
    let a = first.attention(&query_a, 1, scale).unwrap();
    let b = first.attention(&query_b, 1, scale).unwrap();
    assert_eq!(a, second.attention(&query_a, 1, scale).unwrap());
    assert_eq!(b, second.attention(&query_b, 1, scale).unwrap());
    assert_ne!(a, b);

    keys[0] = 50.0;
    let mut edited = KiviCache::new(cfg, 1, 1, 32).unwrap();
    edited.append(&keys, &values, 64).unwrap();
    assert_ne!(first.dequantize().0, edited.dequantize().0);
}

#[test]
fn long_sequence_beats_fp16_footprint() {
    let cfg = KiviConfig::try_new(2, 32, 64).unwrap();
    let mut cache = KiviCache::new(cfg, 1, 2, 64).unwrap();
    let seq = 512;
    let n = 2 * seq * 64;
    cache.append(&fill(n, 11), &fill(n, 13), seq).unwrap();
    let report = cache.memory();
    assert!(report.compression_ratio_vs_fp16() > 2.0);
    assert!(report.compressed_bytes() < report.fp16_baseline_bytes);
    assert_eq!(report.key_residual_bytes, 0);
}

use kivi_cache::{KiviCache, KiviConfig, KiviError, RuntimeKv};

fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn runtime_decode_matches_direct_cache() {
    let cfg = KiviConfig::try_new(2, 32, 32).unwrap();
    let batch = 1;
    let kv = 2;
    let q_heads = 4;
    let dim = 32;
    let prompt = 40;
    let steps = 3usize;
    let scale = 1.0 / (dim as f32).sqrt();

    let keys = fill(batch * kv * prompt * dim, 3);
    let values = fill(batch * kv * prompt * dim, 4);
    let mut runtime = RuntimeKv::try_new(cfg, batch, kv, dim).unwrap();
    let mut cache = KiviCache::new(cfg, batch, kv, dim).unwrap();
    runtime.prefill(&keys, &values, prompt).unwrap();
    cache.append(&keys, &values, prompt).unwrap();

    let mut previous = Vec::new();
    for step in 0..steps {
        let key = fill(batch * kv * dim, 10 + step as u32);
        let value = fill(batch * kv * dim, 20 + step as u32);
        let query = fill(batch * q_heads * dim, 30 + step as u32);
        let got = runtime
            .decode(&key, &value, &query, q_heads, scale)
            .unwrap();
        cache.append(&key, &value, 1).unwrap();
        let expect = cache.attention(&query, q_heads, scale).unwrap();
        assert_eq!(got, expect);
        if !previous.is_empty() {
            assert_ne!(got, previous);
        }
        previous = got;
    }
    assert_eq!(runtime.cache().sequence_length(), prompt + steps);
    assert_eq!(
        runtime.cache().memory().compressed_bytes(),
        cache.memory().compressed_bytes()
    );
}

#[test]
fn runtime_rejects_a_short_prefill() {
    let mut runtime =
        RuntimeKv::try_new(KiviConfig::try_new(2, 32, 32).unwrap(), 1, 1, 32).unwrap();
    let err = runtime.prefill(&[0.0; 8], &[0.0; 32], 1).unwrap_err();
    assert_eq!(
        err,
        KiviError::Shape {
            expected: 32,
            actual: 8
        }
    );
}

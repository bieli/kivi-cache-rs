use kivi_cache::{CacheSnapshot, KiviCache, KiviConfig};

fn cache() -> KiviCache {
    KiviCache::new(KiviConfig::try_new(2, 32, 32).unwrap(), 1, 1, 32).unwrap()
}

fn append(cache: &mut KiviCache, n: usize) {
    let floats = n * 32;
    cache
        .append(&vec![0.25; floats], &vec![-0.5; floats], n)
        .unwrap();
}

#[test]
fn snapshot_matches_live_counts() {
    let mut cache = cache();
    append(&mut cache, 40);
    let snap = CacheSnapshot::capture(&cache);
    assert_eq!(snap.bits, 2);
    assert_eq!(snap.residual_length, 32);
    assert_eq!(snap.sequence_tokens, cache.sequence_length());
    assert_eq!(snap.quantized_key_tokens, cache.quantized_key_tokens());
    assert_eq!(snap.quantized_value_tokens, cache.quantized_value_tokens());
    assert_eq!(snap.key_residual_tokens, cache.key_residual_len());
    assert_eq!(snap.value_residual_tokens, cache.value_residual_len());
    assert_eq!(snap.packed_keys(), 0..32);
    assert_eq!(snap.exact_keys(), 32..40);
    assert_eq!(snap.packed_values(), 0..8);
    assert_eq!(snap.exact_values(), 8..40);
}

#[test]
fn nothing_is_forgotten_inside_the_window() {
    let mut cache = cache();
    let before = CacheSnapshot::capture(&cache);
    append(&mut cache, 10);
    let after = CacheSnapshot::capture(&cache);
    let forgotten = after.forgotten_since(&before);
    assert!(!forgotten.any());
    assert_eq!(forgotten.keys(), 0..0);
    assert_eq!(forgotten.values(), 0..0);
    assert_eq!(after.exact_keys(), 0..10);
    assert_eq!(after.exact_values(), 0..10);
}

#[test]
fn key_flush_packs_a_block_and_keeps_values_exact() {
    let mut cache = cache();
    let before = CacheSnapshot::capture(&cache);
    append(&mut cache, 32);
    let after = CacheSnapshot::capture(&cache);
    let forgotten = after.forgotten_since(&before);
    assert_eq!(forgotten.keys(), 0..32);
    assert_eq!(forgotten.value_count(), 0);
    assert_eq!(after.exact_keys(), 32..32);
    assert_eq!(after.exact_values(), 0..32);
    assert_eq!(after.key_residual_tokens, 0);
    assert_eq!(after.value_residual_tokens, 32);
}

#[test]
fn next_token_after_the_window_packs_the_oldest_value() {
    let mut cache = cache();
    append(&mut cache, 32);
    let before = CacheSnapshot::capture(&cache);
    append(&mut cache, 1);
    let after = CacheSnapshot::capture(&cache);
    let forgotten = after.forgotten_since(&before);
    assert_eq!(forgotten.key_count(), 0);
    assert_eq!(forgotten.values(), 0..1);
    assert_eq!(after.exact_keys(), 32..33);
    assert_eq!(after.exact_values(), 1..33);
}

#[test]
fn chunk_append_reports_one_contiguous_span() {
    let mut cache = cache();
    let before = CacheSnapshot::capture(&cache);
    append(&mut cache, 40);
    let forgotten = CacheSnapshot::capture(&cache).forgotten_since(&before);
    assert_eq!(forgotten.keys(), 0..32);
    assert_eq!(forgotten.values(), 0..8);
}

#[test]
fn single_steps_name_each_value_and_the_next_key_flush() {
    let mut cache = cache();
    append(&mut cache, 40);
    let mut prev = CacheSnapshot::capture(&cache);
    let mut value_spans = Vec::new();
    let mut key_spans = Vec::new();
    for _ in 0..24 {
        append(&mut cache, 1);
        let now = CacheSnapshot::capture(&cache);
        let forgotten = now.forgotten_since(&prev);
        if forgotten.key_count() > 0 {
            key_spans.push(forgotten.keys());
        }
        if forgotten.value_count() > 0 {
            value_spans.push(forgotten.values());
        }
        prev = now;
    }
    assert_eq!(key_spans, vec![32..64]);
    assert_eq!(value_spans.first().cloned(), Some(8..9));
    assert_eq!(value_spans.last().cloned(), Some(31..32));
    assert_eq!(value_spans.len(), 24);
}

#[test]
fn reversed_snapshots_forget_nothing() {
    let mut cache = cache();
    let earlier = CacheSnapshot::capture(&cache);
    append(&mut cache, 40);
    let later = CacheSnapshot::capture(&cache);
    let reversed = earlier.forgotten_since(&later);
    assert!(!reversed.any());
}

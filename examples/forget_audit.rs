// Audit when tokens leave the fp32 window.
//
// Each append is snapshotted. The trace lists token indices that just moved
// into 2-bit codes, plus the indices that are still exact. It does not print
// key or value tensors.

use kivi_cache::{CacheSnapshot, ForgottenTokens, KiviCache, KiviConfig};

fn main() {
    let residual = 32;
    let dim = 32;
    let cfg = KiviConfig::try_new(2, 32, residual).unwrap();
    let mut cache = KiviCache::new(cfg, 1, 1, dim).unwrap();
    let mut prev = CacheSnapshot::capture(&cache);

    println!("forget audit   bits=2  R={residual}  one layer, one head");
    println!(
        "{:>6} {:>8}  {:<42} still exact",
        "step", "tokens", "left fp32"
    );

    let mut events = Vec::new();
    report(&mut cache, &mut prev, &mut events, 40, dim);
    for _ in 0..32 {
        report(&mut cache, &mut prev, &mut events, 1, dim);
    }

    let final_snap = CacheSnapshot::capture(&cache);
    println!();
    println!(
        "packed keys {}  exact keys {}",
        fmt_range(&final_snap.packed_keys()),
        fmt_range(&final_snap.exact_keys())
    );
    println!(
        "packed values {}  exact values {}",
        fmt_range(&final_snap.packed_values()),
        fmt_range(&final_snap.exact_values())
    );

    let at = |tokens: usize| events.iter().find(|e: &&Event| e.tokens == tokens).unwrap();
    let prefill = at(40);
    assert_eq!(prefill.forgotten.keys(), 0..32);
    assert_eq!(prefill.forgotten.values(), 0..8);
    let first_value_step = at(41);
    assert_eq!(first_value_step.forgotten.key_count(), 0);
    assert_eq!(first_value_step.forgotten.values(), 8..9);
    let second_flush = at(64);
    assert_eq!(second_flush.forgotten.keys(), 32..64);
    assert_eq!(second_flush.forgotten.values(), 31..32);
}

struct Event {
    tokens: usize,
    forgotten: ForgottenTokens,
}

fn report(
    cache: &mut KiviCache,
    prev: &mut CacheSnapshot,
    events: &mut Vec<Event>,
    count: usize,
    dim: usize,
) {
    let n = count * dim;
    cache.append(&vec![0.1; n], &vec![-0.2; n], count).unwrap();
    let now = CacheSnapshot::capture(cache);
    let forgotten = now.forgotten_since(prev);
    if forgotten.any() {
        println!(
            "{:>6} {:>8}  {:<42} keys {}  values {}",
            events.len() + 1,
            now.sequence_tokens,
            left_fp32(&forgotten, now.bits),
            fmt_range(&now.exact_keys()),
            fmt_range(&now.exact_values()),
        );
        events.push(Event {
            tokens: now.sequence_tokens,
            forgotten,
        });
    }
    *prev = now;
}

fn left_fp32(forgotten: &ForgottenTokens, bits: u8) -> String {
    match (forgotten.key_count() > 0, forgotten.value_count() > 0) {
        (true, true) => format!(
            "keys {}; values {} -> {bits}-bit",
            fmt_range(&forgotten.keys()),
            fmt_range(&forgotten.values())
        ),
        (true, false) => format!("keys {} -> {bits}-bit", fmt_range(&forgotten.keys())),
        (false, true) => format!("values {} -> {bits}-bit", fmt_range(&forgotten.values())),
        (false, false) => String::new(),
    }
}

fn fmt_range(range: &std::ops::Range<usize>) -> String {
    if range.start == range.end {
        "none".to_string()
    } else {
        format!("[{}, {})", range.start, range.end)
    }
}

// Same prompt, same KV cache, twice.
//
// KIVI codes come from group min/max and half-to-even rounding. There is no
// seed in the quantizer. Two policies can be scored against that cache and
// the scores match on a replay, which is what an RL comparison needs.

use kivi_cache::{KiviCache, KiviConfig};

const DIM: usize = 32;
const SEQ: usize = 64;

fn main() {
    let cfg = KiviConfig::try_new(2, 32, 32).unwrap();
    let keys = prompt(SEQ * DIM, 4);
    let values = prompt(SEQ * DIM, 5);
    let policy_a = prompt(DIM, 8);
    let policy_b = prompt(DIM, 9);
    let scale = 1.0 / (DIM as f32).sqrt();

    let first = store(cfg, &keys, &values);
    let second = store(cfg, &keys, &values);
    let (k1, v1) = first.dequantize();
    let (k2, v2) = second.dequantize();
    let a1 = first.attention(&policy_a, 1, scale).unwrap();
    let a2 = second.attention(&policy_a, 1, scale).unwrap();
    let b1 = first.attention(&policy_b, 1, scale).unwrap();
    let b2 = second.attention(&policy_b, 1, scale).unwrap();

    println!("rl cache replay   2-bit min/max   no rng in the quantizer");
    println!("  prompt tokens   {SEQ}");
    println!("  dequant match   {}", k1 == k2 && v1 == v2);
    println!(
        "  bytes match     {}",
        first.memory().compressed_bytes() == second.memory().compressed_bytes()
    );
    println!();
    println!(
        "{:<10} {:>14} {:>14} {:>12}",
        "policy", "run 1", "run 2", "max abs"
    );
    println!(
        "{:<10} {:>14.8} {:>14.8} {:>12.1e}",
        "A",
        score(&a1),
        score(&a2),
        max_abs(&a1, &a2)
    );
    println!(
        "{:<10} {:>14.8} {:>14.8} {:>12.1e}",
        "B",
        score(&b1),
        score(&b2),
        max_abs(&b1, &b2)
    );

    let mut edited = keys.clone();
    edited[0] = 50.0;
    let drifted = store(cfg, &edited, &values);
    let edited_score = score(&drifted.attention(&policy_a, 1, scale).unwrap());
    println!();
    println!(
        "edited prompt, policy A   {:.8}  (replay of the original is {:.8})",
        edited_score,
        score(&a1)
    );

    assert_eq!(k1, k2);
    assert_eq!(v1, v2);
    assert_eq!(a1, a2);
    assert_eq!(b1, b2);
    assert_ne!(a1, b1);
    assert_ne!(edited_score.to_bits(), score(&a1).to_bits());
}

fn store(cfg: KiviConfig, keys: &[f32], values: &[f32]) -> KiviCache {
    let mut cache = KiviCache::new(cfg, 1, 1, DIM).unwrap();
    cache.append(keys, values, SEQ).unwrap();
    cache
}

fn prompt(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|i| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
            // Plant half-ties so the round-to-even step is on the path.
            if i % 17 == 0 {
                (i % 8) as f32 + 0.5
            } else {
                unit * 2.0 - 1.0
            }
        })
        .collect()
}

fn score(output: &[f32]) -> f32 {
    output.iter().sum()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

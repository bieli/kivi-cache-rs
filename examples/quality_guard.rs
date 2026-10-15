// After each key flush, score attention against fp32.
//
// The same 64-token sequence is fed to two guards. One raises the bit width
// when the cosine falls below 0.995. The other doubles the residual window.
// The fp32 tape is only the reference for that check and for the rebuild.

use kivi_cache::{KiviConfig, QualityGuard, QualityPolicy, QualityUpgrade};

const DIM: usize = 32;
const SEQ: usize = 64;
const THRESHOLD: f32 = 0.995;

fn main() {
    let keys = lcg(SEQ * DIM, 21);
    let values = lcg(SEQ * DIM, 22);
    let query = lcg(DIM, 23);
    let scale = 1.0 / (DIM as f32).sqrt();

    println!("quality guard   threshold {THRESHOLD}   dim {DIM}   group 32   start 2-bit R=32");
    println!(
        "{:<20} {:>8} {:>10} {:>10}  upgrade",
        "policy", "tokens", "before", "after"
    );

    let bits = run(
        QualityPolicy::RaiseBits,
        8,
        128,
        &keys,
        &values,
        &query,
        scale,
    );
    let residual = run(
        QualityPolicy::LengthenResidual,
        8,
        128,
        &keys,
        &values,
        &query,
        scale,
    );

    assert!(bits.iter().any(|c| matches!(
        c.upgrades.first(),
        Some(QualityUpgrade::Bits { from: 2, to: 4 })
    )));
    assert!(residual.iter().any(|c| matches!(
        c.upgrades.first(),
        Some(QualityUpgrade::Residual { from: 32, to: 64 })
    )));
    for check in bits.iter().chain(residual.iter()) {
        assert!(
            check.cosine + 1e-5 >= THRESHOLD,
            "cosine {} stayed under {THRESHOLD}",
            check.cosine
        );
    }
}

fn run(
    policy: QualityPolicy,
    max_bits: u8,
    max_residual: usize,
    keys: &[f32],
    values: &[f32],
    query: &[f32],
    scale: f32,
) -> Vec<kivi_cache::FlushCheck> {
    let name = match policy {
        QualityPolicy::RaiseBits => "raise-bits",
        QualityPolicy::LengthenResidual => "lengthen-residual",
    };
    let mut guard = QualityGuard::try_new(
        KiviConfig::try_new(2, 32, 32).unwrap(),
        1,
        1,
        DIM,
        THRESHOLD,
        policy,
        max_bits,
        max_residual,
    )
    .unwrap();
    let checks = guard.append(keys, values, SEQ, query, 1, scale).unwrap();
    for check in &checks {
        println!(
            "{:<20} {:>8} {:>10.6} {:>10.6}  {}",
            name,
            check.sequence_tokens,
            check.cosine_before,
            check.cosine,
            fmt_upgrades(&check.upgrades)
        );
    }
    checks
}

fn fmt_upgrades(upgrades: &[QualityUpgrade]) -> String {
    if upgrades.is_empty() {
        return "none".to_string();
    }
    upgrades
        .iter()
        .map(|upgrade| match upgrade {
            QualityUpgrade::Bits { from, to } => format!("bits {from} -> {to}"),
            QualityUpgrade::Residual { from, to } => format!("R {from} -> {to}"),
        })
        .collect::<Vec<_>>()
        .join(", ")
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

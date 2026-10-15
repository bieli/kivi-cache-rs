use kivi_cache::{KiviConfig, QualityGuard, QualityPolicy, QualityUpgrade};

fn signal(seq: usize, dim: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..seq * dim)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn guard(policy: QualityPolicy, threshold: f32, max_bits: u8, max_residual: usize) -> QualityGuard {
    QualityGuard::try_new(
        KiviConfig::try_new(2, 32, 32).unwrap(),
        1,
        1,
        32,
        threshold,
        policy,
        max_bits,
        max_residual,
    )
    .unwrap()
}

#[test]
fn no_flush_means_no_check() {
    let mut guard = guard(QualityPolicy::RaiseBits, 0.99, 8, 128);
    let keys = signal(16, 32, 1);
    let values = signal(16, 32, 2);
    let query = signal(1, 32, 3);
    let checks = guard.append(&keys, &values, 16, &query, 1, 0.5).unwrap();
    assert!(checks.is_empty());
    assert_eq!(guard.config().bits, 2);
    assert_eq!(guard.sequence_length(), 16);
}

#[test]
fn passing_cosine_does_not_upgrade() {
    let mut guard = guard(QualityPolicy::RaiseBits, 0.0, 8, 128);
    let keys = signal(32, 32, 4);
    let values = signal(32, 32, 5);
    let query = signal(1, 32, 6);
    let checks = guard.append(&keys, &values, 32, &query, 1, 0.5).unwrap();
    assert_eq!(checks.len(), 1);
    assert!(checks[0].upgrades.is_empty());
    assert_eq!(checks[0].cosine_before, checks[0].cosine);
    assert_eq!(checks[0].sequence_tokens, 32);
    assert_eq!(guard.config().bits, 2);
    assert_eq!(guard.config().residual_length, 32);
}

#[test]
fn raise_bits_stops_at_the_ceiling() {
    let mut guard = guard(QualityPolicy::RaiseBits, 1.1, 4, 128);
    let keys = signal(32, 32, 7);
    let values = signal(32, 32, 8);
    let query = signal(1, 32, 9);
    let checks = guard.append(&keys, &values, 32, &query, 1, 0.5).unwrap();
    assert_eq!(
        checks[0].upgrades,
        vec![QualityUpgrade::Bits { from: 2, to: 4 }]
    );
    assert_eq!(guard.config().bits, 4);
    assert_eq!(guard.config().residual_length, 32);
    assert_eq!(guard.cache().sequence_length(), 32);
    assert!(checks[0].cosine < 1.1);
}

#[test]
fn lengthen_residual_puts_the_flushed_block_back_in_fp32() {
    let mut guard = guard(QualityPolicy::LengthenResidual, 1.1, 8, 64);
    let keys = signal(32, 32, 10);
    let values = signal(32, 32, 11);
    let query = signal(1, 32, 12);
    let checks = guard.append(&keys, &values, 32, &query, 1, 0.25).unwrap();
    assert_eq!(
        checks[0].upgrades,
        vec![QualityUpgrade::Residual { from: 32, to: 64 }]
    );
    assert_eq!(guard.config().residual_length, 64);
    assert_eq!(guard.config().bits, 2);
    assert_eq!(guard.cache().quantized_key_tokens(), 0);
    assert_eq!(guard.cache().quantized_value_tokens(), 0);
    assert_eq!(guard.cache().key_residual_len(), 32);
    assert_eq!(guard.cache().value_residual_len(), 32);
    assert!(checks[0].cosine > 0.999, "{}", checks[0].cosine);
}

#[test]
fn one_append_checks_every_key_flush() {
    let mut guard = guard(QualityPolicy::RaiseBits, 0.0, 8, 128);
    let keys = signal(64, 32, 13);
    let values = signal(64, 32, 14);
    let query = signal(1, 32, 15);
    let checks = guard.append(&keys, &values, 64, &query, 1, 0.5).unwrap();
    assert_eq!(checks.len(), 2);
    assert_eq!(checks[0].sequence_tokens, 32);
    assert_eq!(checks[1].sequence_tokens, 64);
    assert!(checks.iter().all(|c| c.upgrades.is_empty()));
}

#[test]
fn four_bit_rebuild_beats_the_two_bit_flush() {
    let keys = signal(32, 32, 21);
    let values = signal(32, 32, 22);
    let query = signal(1, 32, 23);
    let scale = 1.0 / (32.0_f32).sqrt();
    let mut loose = guard(QualityPolicy::RaiseBits, 0.0, 8, 128);
    let baseline = loose.append(&keys, &values, 32, &query, 1, scale).unwrap();
    let before = baseline[0].cosine_before;

    let mut guard = guard(QualityPolicy::RaiseBits, 0.995, 8, 128);
    let checks = guard.append(&keys, &values, 32, &query, 1, scale).unwrap();
    assert!(before < 0.995, "2-bit cosine {before} did not miss 0.995");
    assert_eq!(
        checks[0].upgrades,
        vec![QualityUpgrade::Bits { from: 2, to: 4 }]
    );
    assert!(checks[0].cosine > before);
    assert!(checks[0].cosine >= 0.995, "{}", checks[0].cosine);
    assert_eq!(guard.config().bits, 4);
}

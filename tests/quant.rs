use kivi_cache::{quantize_per_channel, quantize_per_token, KiviError};

fn pattern_token() -> [f32; 16] {
    let mut values = [1.0; 16];
    values[0] = 0.0;
    values[1] = 0.5;
    values[2] = 1.5;
    values[3] = 2.5;
    values[4] = 3.0;
    values
}

#[test]
fn pytorch_pack_matches_the_published_word_layout() {
    let dim = 32;
    let seq = 64;
    let mut data = vec![0.0f32; seq * dim];
    for t in 0..seq {
        for d in 0..dim {
            data[t * dim + d] = ((t + d) % 4) as f32;
        }
    }
    let keys = quantize_per_channel(&data, 1, 1, seq, dim, 2, 32).unwrap();
    let values = quantize_per_token(&data, 1, 1, seq, dim, 2, 32).unwrap();
    let key_codes = keys.pytorch_codes();
    let value_codes = values.pytorch_codes();
    assert_eq!(key_codes.len(), (seq / 16) * dim);
    assert_eq!(value_codes.len(), seq * (dim / 16));
    assert!(key_codes.chunks(dim).all(|row| row == &key_codes[..dim]));
    assert_eq!(key_codes[0], 0xE4E4_E4E4);
    assert_eq!(key_codes[1], 0x3939_3939);
    assert_eq!(key_codes[2], 0x4E4E_4E4E);
    assert_eq!(key_codes[3], 0x9393_9393);
    assert_eq!(value_codes[0], 0xE4E4_E4E4);
    assert_eq!(value_codes[1], 0xE4E4_E4E4);
    assert_eq!(value_codes[2], 0x3939_3939);
    assert!(keys.scale_zero().chunks(2).all(|p| p == [1.0, 0.0]));
    assert!(values.scale_zero().chunks(2).all(|p| p == [1.0, 0.0]));
    assert_eq!(keys.dequantize(), data);
    assert_eq!(values.dequantize(), data);
}

#[test]
fn four_bit_half_ties_round_to_even() {
    // scale = 1, so the stored code is the rounded value.
    let mut data = vec![0.0f32; 16];
    data[0] = 0.0;
    data[1] = 0.5;
    data[2] = 1.5;
    data[3] = 2.5;
    data[4] = 3.5;
    data[5] = 1.2;
    data[6] = 1.6;
    data[15] = 15.0;
    let out = quantize_per_token(&data, 1, 1, 1, 16, 4, 16)
        .unwrap()
        .dequantize();
    assert_eq!(out[0], 0.0);
    assert_eq!(out[1], 0.0);
    assert_eq!(out[2], 2.0);
    assert_eq!(out[3], 2.0);
    assert_eq!(out[4], 4.0);
    assert_eq!(out[5], 1.0);
    assert_eq!(out[6], 2.0);
    assert_eq!(out[15], 15.0);
}

#[test]
fn per_token_half_even_and_endpoints() {
    let data = pattern_token();
    let packed = quantize_per_token(&data, 1, 1, 1, 16, 2, 16).unwrap();
    let out = packed.dequantize();
    let expect = [
        0.0, 0.0, 2.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    assert_eq!(out, expect);
}

#[test]
fn per_channel_half_even_on_one_channel() {
    let pattern = pattern_token();
    let mut data = vec![0.0; 16 * 16];
    for t in 0..16 {
        data[t * 16] = pattern[t];
    }
    let packed = quantize_per_channel(&data, 1, 1, 16, 16, 2, 16).unwrap();
    let out = packed.dequantize();
    let expect_ch0 = [
        0.0, 0.0, 2.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    for t in 0..16 {
        assert_eq!(out[t * 16], expect_ch0[t], "token {t}");
        for d in 1..16 {
            assert_eq!(out[t * 16 + d], 0.0);
        }
    }
}

#[test]
fn negative_channel_endpoints_are_exact() {
    let mut data = vec![0.0; 16 * 16];
    let values = [-2.0, -1.0, 0.0, 1.0];
    for (t, v) in values.iter().enumerate() {
        data[t * 16] = *v;
    }
    let out = quantize_per_channel(&data, 1, 1, 16, 16, 2, 16)
        .unwrap()
        .dequantize();
    assert_eq!(out[0], -2.0);
    assert_eq!(out[16], -1.0);
    assert_eq!(out[32], 0.0);
    assert_eq!(out[48], 1.0);
    for t in 4..16 {
        assert_eq!(out[t * 16], 0.0);
    }
}

#[test]
fn constant_group_reconstructs_exactly() {
    let data = vec![0.25_f32; 2 * 3 * 32 * 16];
    let out = quantize_per_channel(&data, 2, 3, 32, 16, 4, 16)
        .unwrap()
        .dequantize();
    assert_eq!(out, data);
    let out = quantize_per_token(&data, 2, 3, 32, 16, 4, 16)
        .unwrap()
        .dequantize();
    assert_eq!(out, data);
}

#[test]
fn reconstruction_error_is_bounded_by_half_scale() {
    let tokens = 32;
    let dim = 32;
    let mut data = vec![0.0; tokens * dim];
    let mut state = 0x1234_5678u32;
    for x in &mut data {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *x = ((state >> 8) as f32 / (1u32 << 24) as f32) * 6.0 - 3.0;
    }
    let bits = 2u8;
    let group = 32usize;
    let levels = (1u32 << bits) - 1;
    let out = quantize_per_channel(&data, 1, 1, tokens, dim, bits, group)
        .unwrap()
        .dequantize();
    for d in 0..dim {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for t in 0..tokens {
            let v = data[t * dim + d];
            mn = mn.min(v);
            mx = mx.max(v);
        }
        let scale = if mx == mn {
            0.0
        } else {
            (mx - mn) / levels as f32
        };
        for t in 0..tokens {
            let err = (out[t * dim + d] - data[t * dim + d]).abs();
            assert!(
                err <= scale * 0.5 + 1e-4,
                "err {err} scale {scale} at t={t} d={d}"
            );
        }
    }
}

#[test]
fn per_channel_keys_keep_quiet_channels() {
    // Fixed high-magnitude key channel plus quieter channels that vary across
    // the head. Per-token scales are dominated by the outlier; per-channel
    // scales isolate it (the paper's key-cache observation).
    let tokens = 32;
    let dim = 32;
    let mut data = vec![0.0; tokens * dim];
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
    let mae = |xs: &[f32]| {
        let mut acc = 0.0;
        let mut n = 0.0;
        for t in 0..tokens {
            for d in 1..dim {
                acc += (xs[t * dim + d] - data[t * dim + d]).abs();
                n += 1.0;
            }
        }
        acc / n
    };
    let channel_mae = mae(&per_channel);
    let token_mae = mae(&per_token);
    assert!(
        channel_mae < 1e-5,
        "per-channel should keep constant quiet channels exact, mae {channel_mae}"
    );
    assert!(
        token_mae > 0.2,
        "per-token should crush quiet channels, mae {token_mae}"
    );
}

#[test]
fn rejects_bad_shapes() {
    let data = vec![0.0; 8];
    assert!(matches!(
        quantize_per_channel(&data, 1, 1, 8, 1, 3, 32),
        Err(KiviError::InvalidBits(3))
    ));
    assert!(matches!(
        quantize_per_channel(&data, 1, 1, 7, 1, 2, 32),
        Err(KiviError::TokenGroup { .. })
    ));
    assert!(matches!(
        quantize_per_token(&data, 1, 1, 4, 2, 2, 32),
        Err(KiviError::HeadDim { .. })
    ));
}

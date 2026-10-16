// Golden vectors against PyTorch KIVI (`quant/new_pack.py`).
//
// Tensor element [t, d] is `(t + d) % 4`, so every 2-bit group has min 0,
// max 3, and scale 1. `pack_tensor` then puts the first element in the low
// bits of each `int32`. Attention on that cache is the mean of the values,
// which is 1.5 in every channel. The same tensor is what a CUDA serving
// stack should score.

use kivi_cache::{quantize_per_channel, quantize_per_token, KiviCache, KiviConfig};

const DIM: usize = 32;
const SEQ: usize = 64;
const BITS: u8 = 2;
const GROUP: usize = 32;

fn main() {
    let data = fixture();
    let keys = quantize_per_channel(&data, 1, 1, SEQ, DIM, BITS, GROUP).unwrap();
    let values = quantize_per_token(&data, 1, 1, SEQ, DIM, BITS, GROUP).unwrap();
    let key_codes = keys.pytorch_codes();
    let value_codes = values.pytorch_codes();

    println!("pytorch KIVI golden   2-bit  group 32  shape [1, 1, {SEQ}, {DIM}]");
    println!("  element [t, d] = (t + d) % 4");
    println!(
        "  key code shape   [1, 1, {}, {DIM}]   (tokens / 16, dim)",
        SEQ / 16
    );
    println!(
        "  value code shape [1, 1, {SEQ}, {}]   (tokens, dim / 16)",
        DIM / 16
    );
    println!();
    println!("first key words (packed token row 0)");
    for (d, word) in key_codes.iter().enumerate().take(8) {
        println!("  d={d:<2}  {word:#010x}");
    }
    println!("first value words (token 0)");
    println!("  {:#010x}  {:#010x}", value_codes[0], value_codes[1]);

    assert_eq!(key_codes, golden_key_codes());
    assert_eq!(value_codes, golden_value_codes());
    assert!(keys.scale_zero().chunks(2).all(|p| p == [1.0, 0.0]));
    assert!(values.scale_zero().chunks(2).all(|p| p == [1.0, 0.0]));
    assert_eq!(keys.dequantize(), data);
    assert_eq!(values.dequantize(), data);

    let cfg = KiviConfig::try_new(BITS, GROUP, 32).unwrap();
    let mut cache = KiviCache::new(cfg, 1, 1, DIM).unwrap();
    cache.append(&data, &data, SEQ).unwrap();
    let query = vec![1.0; DIM];
    let scale = 1.0 / (DIM as f32).sqrt();
    let got = cache.attention(&query, 1, scale).unwrap();
    let expect = vec![1.5; DIM];
    println!();
    println!(
        "attention vs fp32 mean   max abs {:.1e}",
        got.iter()
            .zip(&expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    );
    assert_eq!(got, expect);
    assert_eq!(cache.quantized_key_tokens(), SEQ);
    assert_eq!(cache.quantized_value_tokens(), SEQ - 32);
    println!("codes match PyTorch pack_tensor, attention is 1.5 on every channel");
}

fn fixture() -> Vec<f32> {
    let mut data = vec![0.0; SEQ * DIM];
    for t in 0..SEQ {
        for d in 0..DIM {
            data[t * DIM + d] = ((t + d) % 4) as f32;
        }
    }
    data
}

// `pack_tensor(..., bits=2, pack_dim=2)` on keys. Each word is 16 tokens.
// Residue 0,1,2,3 repeats, so the word is `0xE4` four times, then rotated.
fn golden_key_codes() -> Vec<u32> {
    let row: Vec<u32> = (0..DIM).map(|d| packed_word((d % 4) as u32)).collect();
    // 64 tokens / 16 = 4 packed rows, and 16 % 4 == 0 so every row matches row 0.
    row.repeat(SEQ / 16)
}

// `pack_tensor(..., bits=2, pack_dim=3)` on values. Two words per token.
fn golden_value_codes() -> Vec<u32> {
    let mut codes = Vec::with_capacity(SEQ * 2);
    for t in 0..SEQ {
        let word = packed_word((t % 4) as u32);
        codes.push(word);
        codes.push(word);
    }
    codes
}

fn packed_word(start: u32) -> u32 {
    let mut word = 0u32;
    for i in 0..16 {
        let code = (start + i) % 4;
        word |= code << (2 * i);
    }
    word
}

// Synthetic passkey retrieval (needle in a haystack).
//
// Stand-in for the 7-digit passkey task used by KIVI (ICML 2024, Appendix B)
// and Landmark Attention. This crate does not run an LLM: the "question" is a
// query aligned with the hidden key token, and the passkey is a one-hot value
// vector planted in the middle of a long haystack.
//
// Prefill attention in a real model would use the original fp32 K/V. Here those
// tensors are stored with [`KiviCache::append`], then a single decode query
// reads them back. The needle sits far outside the fp32 residual window, so a
// hit means the compressed prefix is still addressable.
//
// The table sweeps bit width and residual length.

use kivi_cache::{KiviCache, KiviConfig};

const PASSKEY: [u8; 7] = [2, 7, 1, 8, 2, 8, 1];
const N_DIGITS: usize = 7;
const RADIX: usize = 10;
const DIM: usize = 128;
const HEADS: usize = 1;
const SEQ: usize = 512;
const NEEDLE: usize = 256;

fn main() {
    let mut keys = vec![0.0f32; HEADS * SEQ * DIM];
    let mut values = vec![0.0f32; HEADS * SEQ * DIM];
    plant_haystack(&mut keys, &mut values);
    plant_needle(&mut keys, &mut values);

    let query = needle_query();
    let scale = 1.0 / (DIM as f32).sqrt();
    let passkey_value = needle_value();

    println!("synthetic passkey retrieval");
    println!("  passkey          {}", format_passkey(&PASSKEY));
    println!("  haystack tokens  {SEQ}");
    println!("  needle index     {NEEDLE} (middle; outside every residual window below)");
    println!("  head_dim         {DIM}");
    println!();
    println!(
        "{:>5} {:>9} {:>10} {:>8} {:>10} {:>12}",
        "bits", "residual", "storage", "hit", "cosine", "decoded"
    );

    let configs = [
        (2u8, 32usize),
        (2, 128),
        (4, 32),
        (4, 128),
        (8, 32),
        (8, 128),
    ];
    for (bits, residual) in configs {
        let hit = retrieve(
            bits,
            residual,
            &keys,
            &values,
            &query,
            scale,
            &passkey_value,
        );
        println!(
            "{:>5} {:>9} {:>10} {:>8} {:>10.4} {:>12}",
            bits,
            residual,
            if hit.needle_quantized {
                "quantized"
            } else {
                "fp32 window"
            },
            if hit.recovered { "yes" } else { "no" },
            hit.cosine,
            hit.decoded,
        );
        assert!(
            hit.recovered,
            "passkey lost at {bits}-bit, residual {residual} (decoded {})",
            hit.decoded
        );
    }

    println!();
    println!("Every hit is outside the fp32 residual: the 7-digit key was read from packed KV.");
}

struct Hit {
    recovered: bool,
    needle_quantized: bool,
    cosine: f32,
    decoded: String,
}

fn retrieve(
    bits: u8,
    residual: usize,
    keys: &[f32],
    values: &[f32],
    query: &[f32],
    scale: f32,
    passkey_value: &[f32],
) -> Hit {
    let cfg = KiviConfig::try_new(bits, 32, residual).unwrap();
    let mut cache = KiviCache::new(cfg, 1, HEADS, DIM).unwrap();
    cache.append(keys, values, SEQ).unwrap();

    let needle_quantized = NEEDLE < cache.quantized_value_tokens();
    let out = cache.attention(query, HEADS, scale).unwrap();
    let decoded_digits = decode_passkey(&out);
    Hit {
        recovered: decoded_digits == PASSKEY,
        needle_quantized,
        cosine: cosine(&out, passkey_value),
        decoded: format_passkey(&decoded_digits),
    }
}

fn plant_haystack(keys: &mut [f32], values: &mut [f32]) {
    let mut state = 0x4b49_5649_u32;
    for t in 0..SEQ {
        for d in 0..DIM {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = (state >> 8) as f32 / (1u32 << 24) as f32;
            let noise = u * 0.1 - 0.05;
            keys[t * DIM + d] = noise;
            values[t * DIM + d] = noise * 0.2;
        }
        // Keep the query channel well below the needle so softmax stays peaked.
        keys[t * DIM] = keys[t * DIM].clamp(-0.2, 0.2);
    }
}

fn plant_needle(keys: &mut [f32], values: &mut [f32]) {
    let k = NEEDLE * DIM;
    for d in 0..DIM {
        keys[k + d] = 0.0;
        values[k + d] = 0.0;
    }
    keys[k] = 8.0;
    for (i, digit) in PASSKEY.iter().enumerate() {
        values[k + i * RADIX + *digit as usize] = 1.0;
    }
}

fn needle_query() -> Vec<f32> {
    let mut q = vec![0.0; HEADS * DIM];
    q[0] = 8.0;
    q
}

fn needle_value() -> Vec<f32> {
    let mut v = vec![0.0; DIM];
    for (i, digit) in PASSKEY.iter().enumerate() {
        v[i * RADIX + *digit as usize] = 1.0;
    }
    v
}

fn decode_passkey(out: &[f32]) -> [u8; N_DIGITS] {
    let mut digits = [0u8; N_DIGITS];
    for i in 0..N_DIGITS {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for digit in 0..RADIX {
            let v = out[i * RADIX + digit];
            if v > best_v {
                best_v = v;
                best = digit;
            }
        }
        digits[i] = best as u8;
    }
    digits
}

fn format_passkey(digits: &[u8]) -> String {
    digits.iter().map(|d| char::from(b'0' + d)).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / na.sqrt() / nb.sqrt()
}

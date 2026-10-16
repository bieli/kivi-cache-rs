// One KV backend, three Rust hosts.
//
// Candle, Burn, and a custom server keep their own storage. Each one exposes
// a contiguous `f32` slice in `[batch, kv_heads, seq, dim]` order. `RuntimeKv`
// is the only cache they call. No PyTorch tensor is involved.

use kivi_cache::{KiviConfig, RuntimeKv};

const DIM: usize = 32;
const KV: usize = 2;
const PROMPT: usize = 40;
const STEPS: usize = 3;

fn main() {
    let prompt_k = fill(KV * PROMPT * DIM, 3);
    let prompt_v = fill(KV * PROMPT * DIM, 4);
    let mut steps = Vec::new();
    for step in 0..STEPS {
        steps.push(DecodeTokens {
            key: fill(KV * DIM, 10 + step as u32),
            value: fill(KV * DIM, 20 + step as u32),
            query: fill(KV * DIM, 30 + step as u32),
        });
    }

    let candle = serve(
        &CandleCpu::new(&prompt_k),
        &CandleCpu::new(&prompt_v),
        &steps,
    );
    let burn = serve(
        &BurnCpu::from_f32(&prompt_k),
        &BurnCpu::from_f32(&prompt_v),
        &steps,
    );
    let server = serve(
        &ServerBuf::new(&prompt_k),
        &ServerBuf::new(&prompt_v),
        &steps,
    );

    println!("runtime bridge   contiguous f32 slices   no PyTorch");
    println!("  layout          [batch, kv_heads, seq, dim]");
    println!("  prompt tokens   {PROMPT}");
    println!("  decode steps    {STEPS}");
    println!();
    println!("{:<16} {:>14} {:>12}", "host", "matches", "KV bytes");
    println!(
        "{:<16} {:>14} {:>12}",
        "candle",
        candle.1 == burn.1 && candle.0 == burn.0,
        candle.1
    );
    println!(
        "{:<16} {:>14} {:>12}",
        "burn",
        burn.0 == server.0 && burn.1 == server.1,
        burn.1
    );
    println!(
        "{:<16} {:>14} {:>12}",
        "custom server",
        server.0 == candle.0 && server.1 == candle.1,
        server.1
    );

    assert_eq!(candle.0, burn.0);
    assert_eq!(burn.0, server.0);
    assert_eq!(candle.1, burn.1);
    assert_eq!(burn.1, server.1);
    assert_eq!(candle.2, PROMPT + STEPS);
}

struct DecodeTokens {
    key: Vec<f32>,
    value: Vec<f32>,
    query: Vec<f32>,
}

// What a host must hand the backend: a contiguous `f32` view.
trait HostF32 {
    fn as_f32(&self) -> &[f32];
}

fn serve(
    keys: &impl HostF32,
    values: &impl HostF32,
    steps: &[DecodeTokens],
) -> (Vec<f32>, usize, usize) {
    let mut kv = RuntimeKv::try_new(KiviConfig::try_new(2, 32, 32).unwrap(), 1, KV, DIM).unwrap();
    kv.prefill(keys.as_f32(), values.as_f32(), PROMPT).unwrap();
    let scale = 1.0 / (DIM as f32).sqrt();
    let mut last = Vec::new();
    for step in steps {
        last = kv
            .decode(&step.key, &step.value, &step.query, KV, scale)
            .unwrap();
    }
    let bytes = kv.cache().memory().compressed_bytes();
    let tokens = kv.cache().sequence_length();
    (last, bytes, tokens)
}

// Candle CPU tensor: owned `f32` storage plus the logical shape.
struct CandleCpu {
    data: Vec<f32>,
}

impl CandleCpu {
    fn new(data: &[f32]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }
}

impl HostF32 for CandleCpu {
    fn as_f32(&self) -> &[f32] {
        &self.data
    }
}

// Burn `TensorData`: little-endian `f32` bytes, read back as a slice for the cache.
struct BurnCpu {
    bytes: Vec<u8>,
    floats: Vec<f32>,
}

impl BurnCpu {
    fn from_f32(data: &[f32]) -> Self {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let floats = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Self { bytes, floats }
    }
}

impl HostF32 for BurnCpu {
    fn as_f32(&self) -> &[f32] {
        let _storage = &self.bytes;
        &self.floats
    }
}

// Custom server buffer. Same layout, no framework.
struct ServerBuf {
    data: Vec<f32>,
}

impl ServerBuf {
    fn new(data: &[f32]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }
}

impl HostF32 for ServerBuf {
    fn as_f32(&self) -> &[f32] {
        &self.data
    }
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

// Ablate bit width and residual length on one sequence.
//
// The same prefill is stored four ways: 2-bit and 4-bit, with residual length
// 32 and 128. Reconstruction error and attention cosine come from that one
// layer. The GiB column is the same configuration on a 32-layer agent
// (8 KV heads, dim 128), so the table is a configuration choice, not only a
// kernel demo.

use kivi_cache::{AgentMemoryBudget, KiviCache, KiviConfig};

const LAYERS: usize = 32;
const HEADS: usize = 8;
const DIM: usize = 128;
const SEQ: usize = 560;
const GROUP: usize = 32;

fn main() {
    let keys = lcg(HEADS * SEQ * DIM, 11);
    let values = lcg(HEADS * SEQ * DIM, 12);
    let query = lcg(HEADS * DIM, 13);
    let scale = 1.0 / (DIM as f32).sqrt();
    let fp32 = fp32_attention(&query, &keys, &values, HEADS, SEQ, DIM, scale);
    let fp16_gib = AgentMemoryBudget::try_new(
        KiviConfig::try_new(2, GROUP, 128).unwrap(),
        LAYERS,
        HEADS,
        DIM,
    )
    .unwrap()
    .fp16_bytes_for_agents(1, SEQ) as f64
        / gib();

    println!("config ablation on one {SEQ}-token sequence");
    println!("  layers {LAYERS}, kv heads {HEADS}, dim {DIM}, group {GROUP}");
    println!("  GiB is one agent, all layers, this sequence");
    println!(
        "{:>4} {:>5} {:>10} {:>10} {:>10} {:>10}",
        "bits", "R", "K_mae", "V_mae", "cosine", "GiB"
    );

    let mut rows = Vec::new();
    for &(bits, residual) in &[(2u8, 32usize), (2, 128), (4, 32), (4, 128)] {
        let row = measure(bits, residual, &keys, &values, &query, &fp32, scale);
        println!(
            "{:>4} {:>5} {:>10.5} {:>10.5} {:>10.6} {:>10.4}",
            row.bits, row.residual, row.k_mae, row.v_mae, row.cosine, row.gib
        );
        rows.push(row);
    }

    println!("\nfp16 KV for the same shape   {fp16_gib:.4} GiB");
    let narrow = rows
        .iter()
        .find(|r| r.bits == 2 && r.residual == 32)
        .unwrap();
    let wide = rows
        .iter()
        .find(|r| r.bits == 2 && r.residual == 128)
        .unwrap();
    println!(
        "2-bit R=32 vs R=128: cosine {:+.4}, memory {:.2}x smaller ({:.4} vs {:.4} GiB)",
        narrow.cosine - wide.cosine,
        wide.gib / narrow.gib,
        narrow.gib,
        wide.gib
    );
    assert!(narrow.gib < wide.gib);
    assert!((narrow.cosine - wide.cosine).abs() < 0.02);
    assert!(narrow.k_mae + 1e-5 >= wide.k_mae);
    assert!(narrow.v_mae + 1e-5 >= wide.v_mae);

    let best = rows
        .iter()
        .map(|r| r.cosine)
        .fold(f32::NEG_INFINITY, f32::max);
    let pick = rows
        .iter()
        .filter(|r| best - r.cosine < 0.01)
        .min_by(|a, b| a.gib.partial_cmp(&b.gib).unwrap())
        .unwrap();
    println!(
        "smallest GiB within 0.01 cosine of the best: {}-bit, R={}",
        pick.bits, pick.residual
    );
}

struct Row {
    bits: u8,
    residual: usize,
    k_mae: f32,
    v_mae: f32,
    cosine: f32,
    gib: f64,
}

fn measure(
    bits: u8,
    residual: usize,
    keys: &[f32],
    values: &[f32],
    query: &[f32],
    fp32: &[f32],
    scale: f32,
) -> Row {
    let cfg = KiviConfig::try_new(bits, GROUP, residual).unwrap();
    let mut cache = KiviCache::new(cfg, 1, HEADS, DIM).unwrap();
    cache.append(keys, values, SEQ).unwrap();
    let (k_hat, v_hat) = cache.dequantize();
    let got = cache.attention(query, HEADS, scale).unwrap();
    let plan = AgentMemoryBudget::try_new(cfg, LAYERS, HEADS, DIM).unwrap();
    let layer_bytes = cache.memory().compressed_bytes();
    assert_eq!(plan.bytes_for_context(SEQ), layer_bytes * LAYERS);
    Row {
        bits,
        residual,
        k_mae: mae(keys, &k_hat),
        v_mae: mae(values, &v_hat),
        cosine: cosine(&got, fp32),
        gib: plan.bytes_for_context(SEQ) as f64 / gib(),
    }
}

fn fp32_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    heads: usize,
    seq: usize,
    dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; heads * dim];
    for h in 0..heads {
        let q = &query[h * dim..(h + 1) * dim];
        let mut logits = vec![0.0; seq];
        for t in 0..seq {
            let mut dot = 0.0;
            for d in 0..dim {
                dot += q[d] * keys[(h * seq + t) * dim + d];
            }
            logits[t] = dot * scale;
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for x in &mut logits {
            *x = (*x - max).exp();
            sum += *x;
        }
        for d in 0..dim {
            let mut acc = 0.0;
            for t in 0..seq {
                acc += (logits[t] / sum) * values[(h * seq + t) * dim + d];
            }
            out[h * dim + d] = acc;
        }
    }
    out
}

fn mae(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b) {
        sum += (x - y).abs();
    }
    sum / a.len() as f32
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

fn lcg(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn gib() -> f64 {
    1024.0 * 1024.0 * 1024.0
}

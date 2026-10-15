// Long-context train or eval batch on one GPU.
//
// Llama-3-8B weights take about 16 GiB. On a 24 GiB card the leftover is the
// KV budget. At 32k and 128k tokens that budget, not another GPU, decides the
// batch. KIVI keeps every token and spends the leftover on more sequences.

use kivi_cache::{kv_budget_bytes, AgentMemoryBudget, KiviConfig, ModelPreset};

fn main() {
    let preset = ModelPreset::by_id("llama3-8b").unwrap();
    let plan = AgentMemoryBudget::try_new(
        KiviConfig::default(),
        preset.num_layers,
        preset.num_kv_heads,
        preset.head_dim,
    )
    .unwrap();
    let gpu_gib = 24.0;
    let budget = kv_budget_bytes(gpu_gib, preset.weight_gib);
    let contexts = [32_768usize, 131_072];

    println!("long-context batch   KV only   one GPU, no extra card");
    println!("  model            {}", preset.display_name);
    println!("  GPU              {gpu_gib:.0} GiB");
    println!("  weights          {:.0} GiB reserved", preset.weight_gib);
    println!(
        "  KV budget        {:.0} GiB leftover",
        budget as f64 / gib()
    );
    println!(
        "  bits / group / R {} / {} / {}",
        plan.config.bits, plan.config.group_size, plan.config.residual_length
    );
    println!();
    println!(
        "{:>8} {:>10} {:>10} {:>12} {:>12}",
        "tokens", "fp16 GiB", "KIVI GiB", "fp16 batch", "KIVI batch"
    );

    for tokens in contexts {
        let fp16_one = plan.fp16_bytes_for_agents(1, tokens);
        let kivi_one = plan.bytes_for_context(tokens);
        let fp16_batch = budget / fp16_one as u64;
        let kivi_batch = plan.max_agents(budget, tokens) as u64;
        println!(
            "{:>8} {:>10.2} {:>10.2} {:>12} {:>12}",
            tokens,
            fp16_one as f64 / gib(),
            kivi_one as f64 / gib(),
            fp16_batch,
            kivi_batch
        );
        assert!(
            kivi_batch > fp16_batch,
            "{tokens}: KIVI batch {kivi_batch} vs fp16 {fp16_batch}"
        );
        assert!(plan.fits(budget, kivi_batch as usize, tokens));
        if kivi_batch < u64::MAX {
            assert!(!plan.fits(budget, kivi_batch as usize + 1, tokens));
        }
    }

    let fp16_128k = plan.fp16_bytes_for_agents(1, 131_072) as u64;
    assert!(fp16_128k > budget);
    assert!(plan.max_agents(budget, 131_072) >= 1);
}

fn gib() -> f64 {
    1024.0 * 1024.0 * 1024.0
}

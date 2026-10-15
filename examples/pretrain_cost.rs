// Cost of a fine-tune before the job exists.
//
// The same economics report as `kivi-cache econ`: model shape, context, batch,
// and the hourly price of one GPU. Nothing is loaded and no step is timed.

use kivi_cache::{estimate_pretrain, kv_budget_bytes, KiviConfig, ModelPreset};

fn main() {
    let preset = ModelPreset::by_id("llama3-8b").unwrap();
    let gpu_gib = 24.0;
    let rate = 2.40;
    let hours = 8.0;
    let batch = 4;
    let budget = kv_budget_bytes(gpu_gib, preset.weight_gib);
    let contexts = [32_768usize, 131_072];

    println!("pre-train cost estimate   no job launched");
    println!("  model            {}", preset.display_name);
    println!("  batch            {batch} sequences");
    println!("  GPU              {gpu_gib:.0} GiB at ${rate:.2}/h");
    println!("  weights          {:.0} GiB reserved", preset.weight_gib);
    println!(
        "  KV budget        {:.0} GiB leftover",
        budget as f64 / gib()
    );
    println!("  planned          {hours:.0} h");
    println!("  bits / group / R 2 / 32 / 128");
    println!();
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "tokens", "KIVI GiB", "fp16 GiB", "KIVI GPUs", "fp16 GPUs", "KIVI $", "fp16 $"
    );

    for tokens in contexts {
        let inputs = preset.to_inputs(
            KiviConfig::try_new(2, 32, 128).unwrap(),
            tokens,
            batch,
            budget,
            rate,
            None,
        );
        let estimate = estimate_pretrain(&inputs, hours).unwrap();
        println!(
            "{:>8} {:>10.2} {:>10.2} {:>10} {:>10} {:>10.2} {:>10.2}",
            estimate.context_tokens,
            estimate.kivi_kv_gib,
            estimate.fp16_kv_gib,
            estimate.kivi_gpus,
            estimate.fp16_gpus,
            estimate.kivi_usd,
            estimate.fp16_usd
        );
        assert!(estimate.kivi_gpus < estimate.fp16_gpus);
        assert!(estimate.saved_usd() > 0.0);
        assert!((estimate.kivi_usd - estimate.kivi_gpus as f64 * rate * hours).abs() < 1e-9);
    }
}

fn gib() -> f64 {
    1024.0 * 1024.0 * 1024.0
}

use kivi_cache::{
    compute_report, estimate_pretrain, kv_budget_bytes, parse_context_tokens, EconInputs,
    KiviConfig, ModelPreset,
};

#[test]
fn parse_context_suffixes() {
    assert_eq!(parse_context_tokens("128k").unwrap(), 128_000);
    assert_eq!(parse_context_tokens("1.5m").unwrap(), 1_500_000);
    assert_eq!(parse_context_tokens("4096").unwrap(), 4096);
}

#[test]
fn wider_gpu_fits_more_agents_than_fp16() {
    let preset = ModelPreset::by_id("llama3-8b").unwrap();
    let inputs = preset.to_inputs(
        KiviConfig::default(),
        128_000,
        16,
        kv_budget_bytes(80.0, preset.weight_gib),
        1.10,
        None,
    );
    let report = compute_report(&inputs).unwrap();
    assert!(report.kv_saved_gib > 0.0);
    assert!(report.compression_ratio > 2.0);
    assert!(report.max_agents_on_gpu >= report.max_agents_fp16_on_gpu);
}

#[test]
fn cli_example_llama3_16_agents_128k() {
    let preset = ModelPreset::by_id("llama3-8b").unwrap();
    let inputs = preset.to_inputs(
        KiviConfig::try_new(2, 32, 128).unwrap(),
        parse_context_tokens("128k").unwrap(),
        16,
        kv_budget_bytes(24.0, preset.weight_gib),
        1.10,
        None,
    );
    let report = compute_report(&inputs).unwrap();
    assert!(report.kv_saved_gib > 1.0);
    assert!(report.compression_ratio > 2.0);
    assert_eq!(report.agents, 16);
    assert_eq!(report.context_tokens, 128_000);
}

#[test]
fn pretrain_bill_uses_gpu_count_and_planned_hours() {
    let preset = ModelPreset::by_id("llama3-8b").unwrap();
    let hours = 8.0;
    let rate = 2.40;
    let budget = kv_budget_bytes(24.0, preset.weight_gib);
    let inputs = preset.to_inputs(
        KiviConfig::try_new(2, 32, 128).unwrap(),
        32_768,
        4,
        budget,
        rate,
        None,
    );
    let report = compute_report(&inputs).unwrap();
    let estimate = estimate_pretrain(&inputs, hours).unwrap();
    assert_eq!(estimate.context_tokens, 32_768);
    assert_eq!(estimate.batch, 4);
    assert!((estimate.kivi_kv_gib - report.kivi_kv_gib).abs() < 1e-12);
    assert!((estimate.fp16_kv_gib - report.fp16_kv_gib).abs() < 1e-12);
    assert!(estimate.kivi_gpus < estimate.fp16_gpus);
    assert!((estimate.kivi_usd - estimate.kivi_gpus as f64 * rate * hours).abs() < 1e-9);
    assert!((estimate.fp16_usd - estimate.fp16_gpus as f64 * rate * hours).abs() < 1e-9);
    assert!(estimate.saved_usd() > 0.0);
    assert_eq!(estimate_pretrain(&inputs, 0.0).unwrap().kivi_usd, 0.0);
}

#[test]
fn long_context_pretrain_needs_fewer_gpus_than_fp16() {
    let preset = ModelPreset::by_id("llama3-8b").unwrap();
    let budget = kv_budget_bytes(24.0, preset.weight_gib);
    for tokens in [32_768usize, 131_072] {
        let inputs = preset.to_inputs(KiviConfig::default(), tokens, 4, budget, 2.40, None);
        let estimate = estimate_pretrain(&inputs, 8.0).unwrap();
        assert!(estimate.kivi_kv_gib < estimate.fp16_kv_gib);
        assert!(estimate.kivi_gpus >= 1);
        assert!(estimate.kivi_gpus < estimate.fp16_gpus);
        assert!(estimate.kivi_usd < estimate.fp16_usd);
    }
}

#[test]
fn custom_shape_via_inputs() {
    let inputs = EconInputs {
        model_id: "custom".into(),
        num_layers: 32,
        num_kv_heads: 8,
        head_dim: 128,
        weight_gib: 16.0,
        config: KiviConfig::default(),
        context_tokens: 4096,
        agents: 4,
        kv_budget_bytes: kv_budget_bytes(48.0, 16.0),
        usd_per_hour: 2.0,
    };
    assert!(compute_report(&inputs).is_ok());
}

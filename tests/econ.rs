use kivi_cache::{
    compute_report, kv_budget_bytes, parse_context_tokens, EconInputs, KiviConfig, ModelPreset,
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

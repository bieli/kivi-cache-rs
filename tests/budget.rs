use kivi_cache::{AgentMemoryBudget, KiviCache, KiviConfig, KiviError, MixedPrecisionBudget};

#[test]
fn tokens_until_key_flush_tracks_residual() {
    let mut cache = KiviCache::new(KiviConfig::try_new(2, 16, 32).unwrap(), 1, 1, 16).unwrap();
    assert_eq!(cache.tokens_until_key_flush(), 32);
    cache
        .append(&vec![0.0; 16 * 10], &vec![0.0; 16 * 10], 10)
        .unwrap();
    assert_eq!(cache.tokens_until_key_flush(), 22);
    cache
        .append(&vec![0.0; 16 * 22], &vec![0.0; 16 * 22], 22)
        .unwrap();
    assert_eq!(cache.key_residual_len(), 0);
    assert_eq!(cache.tokens_until_key_flush(), 32);
}

#[test]
fn value_slots_remaining_shrink_as_window_fills() {
    let mut cache = KiviCache::new(KiviConfig::try_new(2, 16, 32).unwrap(), 1, 1, 16).unwrap();
    assert_eq!(cache.value_residual_slots_remaining(), 32);
    cache
        .append(&vec![0.0; 16 * 20], &vec![0.0; 16 * 20], 20)
        .unwrap();
    assert_eq!(cache.value_residual_slots_remaining(), 12);
    cache
        .append(&vec![0.0; 16 * 20], &vec![0.0; 16 * 20], 20)
        .unwrap();
    assert_eq!(cache.value_residual_slots_remaining(), 0);
}

#[test]
fn budget_snapshot_reflects_live_cache() {
    let mut cache = KiviCache::new(KiviConfig::default(), 1, 8, 128).unwrap();
    cache
        .append(&vec![0.1; 8 * 128 * 200], &vec![0.2; 8 * 128 * 200], 200)
        .unwrap();
    let snap = cache.budget_snapshot();
    assert_eq!(snap.sequence_tokens, 200);
    assert_eq!(snap.key_residual_tokens, cache.key_residual_len());
    assert_eq!(snap.marginal_bytes_per_token, 1024);
    assert_eq!(snap.current_layer_bytes, cache.memory().compressed_bytes());
}

#[test]
fn agent_planner_llama_shaped_200k_context() {
    // Llama-3-8B-ish: 32 layers, 8 KV heads (GQA), 128 head dim.
    let plan = AgentMemoryBudget::try_new(KiviConfig::default(), 32, 8, 128).unwrap();
    let tokens = 200_000;
    let agents = 8;
    let budget_gib = 24.0;
    let budget_bytes = (budget_gib * 1024.0 * 1024.0 * 1024.0) as u64;

    let need = plan.bytes_for_agents(agents, tokens);
    let max = plan.max_agents(budget_bytes, tokens);
    let ratio = plan.compression_ratio_vs_fp16(agents, tokens);

    assert!(need < plan.fp16_bytes_for_agents(agents, tokens));
    assert!(ratio > 2.0);
    assert!(plan.fits(budget_bytes, max, tokens));
    if max > 0 {
        assert!(!plan.fits(budget_bytes, max + 1, tokens));
    }
}

#[test]
fn planner_bytes_match_live_cache() {
    let cfg = KiviConfig::default();
    let plan = AgentMemoryBudget::try_new(cfg, 1, 8, 128).unwrap();
    for seq in [1usize, 32, 127, 128, 129, 512, 2048] {
        let mut cache = KiviCache::new(cfg, 1, 8, 128).unwrap();
        let n = 8 * seq * 128;
        cache.append(&vec![0.1; n], &vec![0.1; n], seq).unwrap();
        assert_eq!(
            plan.bytes_for_context(seq),
            cache.memory().compressed_bytes(),
            "seq={seq}"
        );
    }
}

#[test]
fn marginal_rate_matches_one_key_flush_cycle() {
    let plan = AgentMemoryBudget::try_new(KiviConfig::default(), 1, 8, 128).unwrap();
    let cycle = plan.config.residual_length;
    let start = cycle * 40;
    let delta = plan.bytes_for_context(start + cycle) - plan.bytes_for_context(start);
    assert_eq!(delta / cycle, plan.marginal_bytes_per_token());
}

#[test]
fn next_token_costs_more_inside_the_key_residual() {
    let plan = AgentMemoryBudget::try_new(KiviConfig::default(), 1, 8, 128).unwrap();
    assert!(plan.next_token_bytes(5000) > plan.marginal_bytes_per_token());
}

#[test]
fn max_agents_scales_with_budget() {
    let plan = AgentMemoryBudget::try_new(KiviConfig::default(), 32, 8, 128).unwrap();
    let one = plan.bytes_for_context(200_000);
    assert_eq!(plan.bytes_for_agents(8, 200_000), one * 8);
    let budget = one as u64 * 4;
    assert_eq!(plan.max_agents(budget, 200_000), 4);
    assert!(plan.fits(budget, 4, 200_000));
    assert!(!plan.fits(budget, 5, 200_000));
}

#[test]
fn mixed_precision_sums_wide_and_narrow_layers() {
    let tokens = 200_000;
    let heads = 8;
    let dim = 128;
    let mixed = MixedPrecisionBudget::early_wide(32, 8, 4, 2, 32, 128, heads, dim).unwrap();
    assert_eq!(mixed.num_layers(), 32);
    assert_eq!(mixed.layer_bits(0), Some(4));
    assert_eq!(mixed.layer_bits(7), Some(4));
    assert_eq!(mixed.layer_bits(8), Some(2));
    assert_eq!(mixed.layer_bits(31), Some(2));
    assert_eq!(mixed.layer_bits(32), None);

    let wide = AgentMemoryBudget::try_new(KiviConfig::try_new(4, 32, 128).unwrap(), 8, heads, dim)
        .unwrap();
    let narrow =
        AgentMemoryBudget::try_new(KiviConfig::try_new(2, 32, 128).unwrap(), 24, heads, dim)
            .unwrap();
    assert_eq!(
        mixed.bytes_for_context(tokens),
        wide.bytes_for_context(tokens) + narrow.bytes_for_context(tokens)
    );
    assert!(mixed.bytes_for_layer(0, tokens).unwrap() > mixed.bytes_for_layer(31, tokens).unwrap());
}

#[test]
fn mixed_layer_bytes_match_live_caches() {
    let plan = MixedPrecisionBudget::early_wide(2, 1, 4, 2, 32, 128, 4, 64).unwrap();
    let seq = 300;
    let mut total = 0;
    for (layer, bits) in [(0usize, 4u8), (1, 2)] {
        let cfg = KiviConfig::try_new(bits, 32, 128).unwrap();
        let mut cache = KiviCache::new(cfg, 1, 4, 64).unwrap();
        let n = 4 * seq * 64;
        cache.append(&vec![0.15; n], &vec![-0.2; n], seq).unwrap();
        let live = cache.memory().compressed_bytes();
        assert_eq!(plan.bytes_for_layer(layer, seq).unwrap(), live);
        total += live;
    }
    assert_eq!(plan.bytes_for_context(seq), total);
}

#[test]
fn mixed_precision_sits_between_uniform_budgets() {
    let heads = 8;
    let dim = 128;
    let tokens = 8_192;
    let mixed = MixedPrecisionBudget::early_wide(32, 8, 4, 2, 32, 128, heads, dim).unwrap();
    let all2 = AgentMemoryBudget::try_new(KiviConfig::try_new(2, 32, 128).unwrap(), 32, heads, dim)
        .unwrap();
    let all4 = AgentMemoryBudget::try_new(KiviConfig::try_new(4, 32, 128).unwrap(), 32, heads, dim)
        .unwrap();
    let mixed_one = mixed.bytes_for_context(tokens);
    assert!(all4.bytes_for_context(tokens) > mixed_one);
    assert!(mixed_one > all2.bytes_for_context(tokens));

    let budget = mixed_one as u64 * 3;
    assert_eq!(mixed.max_agents(budget, tokens), 3);
    assert!(mixed.fits(budget, 3, tokens));
    assert!(!mixed.fits(budget, 4, tokens));
    assert!(all4.max_agents(budget, tokens) < mixed.max_agents(budget, tokens));
    assert!(all2.max_agents(budget, tokens) >= mixed.max_agents(budget, tokens));
    assert_eq!(mixed.bytes_for_agents(3, tokens), mixed_one * 3);
}

#[test]
fn mixed_marginal_matches_one_flush_cycle() {
    let plan = MixedPrecisionBudget::early_wide(4, 1, 4, 2, 32, 64, 2, 64).unwrap();
    let cycle = plan.residual_length();
    let start = cycle * 40;
    let delta = plan.bytes_for_context(start + cycle) - plan.bytes_for_context(start);
    assert_eq!(delta / cycle, plan.marginal_bytes_per_token());
    assert_eq!(
        plan.next_token_bytes(start),
        plan.bytes_for_context(start + 1) - plan.bytes_for_context(start)
    );
}

#[test]
fn uniform_bit_schedule_matches_agent_budget() {
    let cfg = KiviConfig::try_new(2, 32, 128).unwrap();
    let mixed = MixedPrecisionBudget::early_wide(16, 0, 4, 2, 32, 128, 8, 128).unwrap();
    let plan = AgentMemoryBudget::try_new(cfg, 16, 8, 128).unwrap();
    assert!(mixed.layer_bits(0) == Some(2));
    assert_eq!(mixed.bytes_for_context(4096), plan.bytes_for_context(4096));
    assert_eq!(
        mixed.fp16_bytes_for_agents(2, 4096),
        plan.fp16_bytes_for_agents(2, 4096)
    );
    let ratio =
        mixed.fp16_bytes_for_agents(2, 4096) as f64 / mixed.bytes_for_agents(2, 4096) as f64;
    assert!((mixed.compression_ratio_vs_fp16(2, 4096) - ratio).abs() < 1e-12);
}

#[test]
fn mixed_precision_rejects_bad_schedules() {
    let err = MixedPrecisionBudget::early_wide(8, 9, 4, 2, 32, 128, 8, 128).unwrap_err();
    assert_eq!(
        err,
        KiviError::EarlyLayers {
            early_layers: 9,
            num_layers: 8,
        }
    );
    let err = MixedPrecisionBudget::try_from_bits(&[], 32, 128, 8, 128).unwrap_err();
    assert!(matches!(err, KiviError::ZeroDimension("num_layers")));
    let err = MixedPrecisionBudget::early_wide(4, 2, 4, 2, 8, 32, 1, 32).unwrap_err();
    assert!(matches!(err, KiviError::MisalignedGroup { .. }));
}

#[test]
fn memory_budget_from_cache_matches_manual_plan() {
    let cache = KiviCache::new(KiviConfig::default(), 1, 4, 64).unwrap();
    let from_cache = cache.memory_budget(24).unwrap();
    let manual = AgentMemoryBudget::try_new(KiviConfig::default(), 24, 4, 64).unwrap();
    assert_eq!(
        from_cache.marginal_bytes_per_token(),
        manual.marginal_bytes_per_token()
    );
    assert_eq!(
        from_cache.bytes_for_context(4096),
        manual.bytes_for_context(4096)
    );
}

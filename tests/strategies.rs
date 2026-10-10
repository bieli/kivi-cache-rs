use kivi_cache::{implemented_kv_strategies, kv_strategies, kv_strategy_by_id, StrategyFamily};

#[test]
fn kivi_is_listed_first() {
    let profiles = kv_strategies();
    assert_eq!(profiles[0].id, "kivi");
    assert_eq!(
        profiles[0].status,
        kivi_cache::ImplementationStatus::Implemented
    );
}

#[test]
fn documents_five_strategies() {
    assert_eq!(kv_strategies().len(), 5);
}

#[test]
fn only_kivi_is_implemented() {
    let implemented: Vec<_> = implemented_kv_strategies().collect();
    assert_eq!(implemented.len(), 1);
    assert_eq!(implemented[0].id, "kivi");
}

#[test]
fn eviction_profiles_do_not_retain_context() {
    for id in ["h2o", "streaming-llm"] {
        let p = kv_strategy_by_id(id).unwrap();
        assert_eq!(p.family, StrategyFamily::Eviction);
        assert!(!p.retains_full_context);
    }
}

#[test]
fn quant_profiles_retain_context() {
    for id in ["kivi", "int4-kv", "int2-symmetric"] {
        let p = kv_strategy_by_id(id).unwrap();
        assert_eq!(p.family, StrategyFamily::Quantization);
        assert!(p.retains_full_context);
    }
}

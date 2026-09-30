// KV cache strategy profiles for comparison and agent planning.

// High-level mechanism family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyFamily {
    Quantization,
    Eviction,
}

// Whether this repository ships an implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImplementationStatus {
    Implemented,
    DocumentedOnly,
}

// Static profile for a KV memory strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrategyProfile {
    pub id: &'static str,
    pub name: &'static str,
    pub family: StrategyFamily,
    pub status: ImplementationStatus,
    // If false, tokens may be dropped from the cache entirely.
    pub retains_full_context: bool,
    // No fine-tuning / calibration dataset required at inference setup.
    pub tuning_free: bool,
    // Typical KV size vs fp16, e.g. "~4x smaller".
    pub typical_kv_vs_fp16: &'static str,
    pub best_for: &'static str,
    pub agent_caveat: &'static str,
    pub reference_url: &'static str,
}


const PROFILES: &[StrategyProfile] = &[
    StrategyProfile {
        id: "kivi",
        name: "KIVI (asymmetric 2-bit quant)",
        family: StrategyFamily::Quantization,
        status: ImplementationStatus::Implemented,
        retains_full_context: true,
        tuning_free: true,
        typical_kv_vs_fp16: "~4x smaller at 2-bit (paper ~2.6x peak incl. weights)",
        best_for: "Long-context agents, RAG prefill reuse, parallel sessions, retrieval",
        agent_caveat: "2-bit is lossy — watch reasoning-heavy tasks; use residual window defaults",
        reference_url: "https://arxiv.org/abs/2402.02750",
    },
    StrategyProfile {
        id: "int4-kv",
        name: "Vanilla INT4 KV (per-token K/V)",
        family: StrategyFamily::Quantization,
        status: ImplementationStatus::DocumentedOnly,
        retains_full_context: true,
        tuning_free: true,
        typical_kv_vs_fp16: "~4x smaller",
        best_for: "Simple quant baseline when 4-bit quality is sufficient",
        agent_caveat: "Does not fix 2-bit failure modes; verify packed storage not fake-quant",
        reference_url: "https://huggingface.co/docs/transformers/kv_cache",
    },
];

// All documented strategy profiles (KIVI first)
pub fn all() -> &'static [StrategyProfile] {
    PROFILES
}

// KV cache strategy profiles for comparison and agent planning.
//
// Only **KIVI** is implemented in this crate. Other entries document tradeoffs
// so callers can reason about the design space without pulling in extra deps.
// See `docs/kv-strategies.md` for the full write-up.

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
        agent_caveat: "2-bit is lossy - watch reasoning-heavy tasks; use residual window defaults",
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
    StrategyProfile {
        id: "int2-symmetric",
        name: "Vanilla INT2 KV (symmetric K/V)",
        family: StrategyFamily::Quantization,
        status: ImplementationStatus::DocumentedOnly,
        retains_full_context: true,
        tuning_free: true,
        typical_kv_vs_fp16: "~8x smaller (theoretical)",
        best_for: "Experiments only - not recommended for production agents",
        agent_caveat:
            "Large accuracy drops on reasoning unless using KIVI-style asymmetry + residual",
        reference_url: "https://arxiv.org/abs/2402.02750",
    },
    StrategyProfile {
        id: "h2o",
        name: "H2O (heavy-hitter eviction)",
        family: StrategyFamily::Eviction,
        status: ImplementationStatus::DocumentedOnly,
        retains_full_context: false,
        tuning_free: true,
        typical_kv_vs_fp16: "Fixed token budget (not a fixed ratio)",
        best_for: "Throughput chat with bounded KV RAM when old turns may be forgotten",
        agent_caveat: "Evicted tool output cannot be re-attended - risky for audit / replay",
        reference_url: "https://arxiv.org/abs/2306.14048",
    },
    StrategyProfile {
        id: "streaming-llm",
        name: "StreamingLLM (sink + sliding window)",
        family: StrategyFamily::Eviction,
        status: ImplementationStatus::DocumentedOnly,
        retains_full_context: false,
        tuning_free: true,
        typical_kv_vs_fp16: "Bounded window (constant w.r.t. gen length)",
        best_for: "Very long generation streams where middle context can be discarded",
        agent_caveat: "Middle of long prompts lost - pair with external memory for agents",
        reference_url: "https://arxiv.org/abs/2309.17453",
    },
];

// All documented strategy profiles (KIVI first).
pub fn all() -> &'static [StrategyProfile] {
    PROFILES
}

pub fn by_id(id: &str) -> Option<&'static StrategyProfile> {
    let key = id.to_ascii_lowercase().replace('_', "-");
    PROFILES.iter().find(|p| p.id == key)
}

// Implemented strategies only.
pub fn implemented() -> impl Iterator<Item = &'static StrategyProfile> {
    PROFILES
        .iter()
        .filter(|p| p.status == ImplementationStatus::Implemented)
}

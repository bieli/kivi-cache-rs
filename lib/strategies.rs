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

// Context-economics estimates for agent deployments (KV cache only).

use crate::config::KiviConfig;
use crate::error::KiviError;

// Known model shapes for quick CLI / planning (KV geometry only).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPreset {
    pub id: &'static str,
    pub display_name: &'static str,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    // Approximate fp16 weight footprint (GiB), for whole-GPU budgeting.
    pub weight_gib: f64,
}

// Inputs for a context-economics report.
#[derive(Debug, Clone, PartialEq)]
pub struct EconInputs {
    pub model_id: String,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub weight_gib: f64,
    pub config: KiviConfig,
    pub context_tokens: usize,
    pub agents: usize,
    // VRAM available for KV after optional weight reservation (bytes).
    pub kv_budget_bytes: u64,
    // Cloud/on-prem GPU hourly rate used for rough cost math.
    pub usd_per_hour: f64,
}

// KV memory and rough cost summary.
#[derive(Debug, Clone, PartialEq)]
pub struct EconReport {
    pub model_id: String,
    pub context_tokens: usize,
    pub agents: usize,
    pub bits: u8,
    pub kivi_kv_gib: f64,
    pub fp16_kv_gib: f64,
    pub kv_saved_gib: f64,
    pub compression_ratio: f64,
    pub max_agents_on_gpu: usize,
    pub max_agents_fp16_on_gpu: usize,
    pub fits_requested: bool,
    pub usd_per_hour: f64,
    pub cost_per_agent_hour: f64,
    // Rough $/h freed by KIVI vs fp16 at the same agent count (KV-proportional).
    pub kv_savings_usd_per_hour: f64,
}

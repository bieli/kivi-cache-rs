// Context-economics estimates for agent deployments (KV cache only).

use crate::budget::AgentMemoryBudget;
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

const PRESETS: &[ModelPreset] = &[
    ModelPreset {
        id: "llama2-7b",
        display_name: "Llama-2-7B (MHA)",
        num_layers: 32,
        num_kv_heads: 32,
        head_dim: 128,
        weight_gib: 13.0,
    },
    ModelPreset {
        id: "llama2-13b",
        display_name: "Llama-2-13B (MHA)",
        num_layers: 40,
        num_kv_heads: 40,
        head_dim: 128,
        weight_gib: 24.0,
    },
    ModelPreset {
        id: "llama3-8b",
        display_name: "Llama-3-8B (GQA)",
        num_layers: 32,
        num_kv_heads: 8,
        head_dim: 128,
        weight_gib: 16.0,
    },
    ModelPreset {
        id: "mistral-7b",
        display_name: "Mistral-7B (GQA)",
        num_layers: 32,
        num_kv_heads: 8,
        head_dim: 128,
        weight_gib: 14.0,
    },
    ModelPreset {
        id: "falcon-7b",
        display_name: "Falcon-7B (MQA)",
        num_layers: 32,
        num_kv_heads: 1,
        head_dim: 64,
        weight_gib: 14.0,
    },
    ModelPreset {
        id: "longchat-7b-32k",
        display_name: "LongChat-7B-v1.5-32K",
        num_layers: 32,
        num_kv_heads: 32,
        head_dim: 128,
        weight_gib: 13.0,
    },
];

impl ModelPreset {
    pub fn all() -> &'static [ModelPreset] {
        PRESETS
    }

    pub fn by_id(id: &str) -> Option<ModelPreset> {
        let key = id.to_ascii_lowercase().replace('_', "-");
        PRESETS.iter().copied().find(|p| p.id == key)
    }

    pub fn to_inputs(
        self,
        config: KiviConfig,
        context_tokens: usize,
        agents: usize,
        kv_budget_bytes: u64,
        usd_per_hour: f64,
        weight_gib: Option<f64>,
    ) -> EconInputs {
        EconInputs {
            model_id: self.id.to_string(),
            num_layers: self.num_layers,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            weight_gib: weight_gib.unwrap_or(self.weight_gib),
            config,
            context_tokens,
            agents,
            kv_budget_bytes,
            usd_per_hour,
        }
    }
}

impl EconInputs {
    pub fn plan(&self) -> Result<AgentMemoryBudget, KiviError> {
        AgentMemoryBudget::try_new(
            self.config,
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
        )
    }
}

// Build a report from validated inputs.
pub fn compute_report(inputs: &EconInputs) -> Result<EconReport, KiviError> {
    let plan = inputs.plan()?;
    let agents = inputs.agents;
    let tokens = inputs.context_tokens;

    let kivi_bytes = plan.bytes_for_agents(agents, tokens);
    let fp16_bytes = plan.fp16_bytes_for_agents(agents, tokens);
    let kivi_kv_gib = bytes_to_gib(kivi_bytes);
    let fp16_kv_gib = bytes_to_gib(fp16_bytes);
    let kv_saved_gib = (fp16_bytes.saturating_sub(kivi_bytes)) as f64 / GIB;
    let compression_ratio = plan.compression_ratio_vs_fp16(agents, tokens);

    let max_agents_on_gpu = plan.max_agents(inputs.kv_budget_bytes, tokens);
    let max_agents_fp16_on_gpu = max_agents_fp16(&plan, inputs.kv_budget_bytes, tokens);

    let usd = inputs.usd_per_hour;
    let cost_per_agent_hour = if agents > 0 { usd / agents as f64 } else { 0.0 };

    // If KV is a fixed share of VRAM time, using `kivi/fp16` of KV for the same
    // agent count frees the rest proportionally (very rough).
    let kv_share = if fp16_bytes > 0 {
        kivi_bytes as f64 / fp16_bytes as f64
    } else {
        1.0
    };
    let kv_savings_usd_per_hour = usd * (1.0 - kv_share);

    Ok(EconReport {
        model_id: inputs.model_id.clone(),
        context_tokens: tokens,
        agents,
        bits: inputs.config.bits,
        kivi_kv_gib,
        fp16_kv_gib,
        kv_saved_gib,
        compression_ratio,
        max_agents_on_gpu,
        max_agents_fp16_on_gpu,
        fits_requested: plan.fits(inputs.kv_budget_bytes, agents, tokens),
        usd_per_hour: usd,
        cost_per_agent_hour,
        kv_savings_usd_per_hour,
    })
}

// Default KV budget: `--gpu-gb` minus model weights (floored at zero).
pub fn kv_budget_bytes(gpu_gib: f64, weight_gib: f64) -> u64 {
    let kv_gib = (gpu_gib - weight_gib).max(0.0);
    (kv_gib * GIB) as u64
}

// Parse `128k`, `200K`, `1.5m`, or plain integers as token counts.
pub fn parse_context_tokens(raw: &str) -> Result<usize, String> {
    let s = raw.trim().to_ascii_lowercase().replace('_', "");
    if s.is_empty() {
        return Err("context length must not be empty".into());
    }
    let (num, suffix) = if let Some(stripped) = s.strip_suffix('k') {
        (stripped, 1_000usize)
    } else if let Some(stripped) = s.strip_suffix('m') {
        (stripped, 1_000_000usize)
    } else {
        (s.as_str(), 1usize)
    };
    let base: f64 = num
        .parse()
        .map_err(|_| format!("invalid context token count: {raw}"))?;
    if base < 0.0 {
        return Err(format!("context must be non-negative: {raw}"));
    }
    let value = base * suffix as f64;
    if value > usize::MAX as f64 {
        return Err(format!("context too large: {raw}"));
    }
    Ok(value as usize)
}

fn max_agents_fp16(plan: &AgentMemoryBudget, budget_bytes: u64, tokens: usize) -> usize {
    let per_agent = plan.fp16_bytes_for_agents(1, tokens) as u64;
    if per_agent == 0 {
        return usize::MAX;
    }
    (budget_bytes / per_agent) as usize
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn bytes_to_gib(bytes: usize) -> f64 {
    bytes as f64 / GIB
}

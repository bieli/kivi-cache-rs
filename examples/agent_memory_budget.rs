// Plan parallel agent sessions against a GPU memory budget.
//
// Example: Llama-3-8B-shaped GQA (32 layers, 8 KV heads, 128 head dim) with
// KIVI defaults (2-bit, group 32, residual 128).

use kivi_cache::{AgentMemoryBudget, KiviConfig};

fn main() {
    let plan = AgentMemoryBudget::try_new(KiviConfig::default(), 32, 8, 128).unwrap();

    let context_tokens = 200_000;
    let target_agents = 8;
    let budget_gib = 24.0;
    let budget_bytes = (budget_gib * 1024.0 * 1024.0 * 1024.0) as u64;

    let per_agent = plan.bytes_for_context(context_tokens);
    let total = plan.bytes_for_agents(target_agents, context_tokens);
    let max_fit = plan.max_agents(budget_bytes, context_tokens);
    let fp16 = plan.fp16_bytes_for_agents(target_agents, context_tokens);
    let ratio = plan.compression_ratio_vs_fp16(target_agents, context_tokens);

    println!("KIVI agent memory budget (KV cache only, one layer shape)");
    println!("  layers           {}", plan.num_layers);
    println!("  kv_heads         {}", plan.num_kv_heads);
    println!("  head_dim         {}", plan.head_dim);
    println!(
        "  bits / group / R  {} / {} / {}",
        plan.config.bits, plan.config.group_size, plan.config.residual_length
    );
    println!();
    println!("Context per agent   {context_tokens} tokens");
    println!(
        "Marginal / token    {} bytes/layer (steady state)",
        plan.marginal_bytes_per_token()
    );
    println!(
        "Per agent (all layers) {} MiB",
        per_agent as f64 / (1024.0 * 1024.0)
    );
    println!();
    println!("Target deployment   {target_agents} agents");
    println!("  KIVI KV total     {:.2} GiB", total as f64 / gib());
    println!("  fp16 KV total     {:.2} GiB", fp16 as f64 / gib());
    println!("  compression       {ratio:.2}x vs fp16");
    println!();
    println!("Budget              {budget_gib:.0} GiB");
    println!(
        "  fits {target_agents} agents? {}",
        if plan.fits(budget_bytes, target_agents, context_tokens) {
            "yes"
        } else {
            "no"
        }
    );
    println!("  max agents @ {context_tokens} tok  {max_fit}");
}

fn gib() -> f64 {
    1024.0 * 1024.0 * 1024.0
}

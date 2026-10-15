// Early layers stay at 4 bits, later layers drop to 2 bits.
//
// The planner sums those layers and reports how many parallel sessions fit
// in one KV budget. Shape matches Llama-3-8B (32 layers, 8 KV heads, dim 128).

use kivi_cache::{AgentMemoryBudget, KiviConfig, MixedPrecisionBudget};

fn main() {
    let layers = 32;
    let early = 8;
    let heads = 8;
    let dim = 128;
    let tokens = 200_000;
    let budget_gib = 26.0;
    let budget_bytes = (budget_gib * 1024.0 * 1024.0 * 1024.0) as u64;

    let mixed = MixedPrecisionBudget::early_wide(layers, early, 4, 2, 32, 128, heads, dim).unwrap();
    let all2 =
        AgentMemoryBudget::try_new(KiviConfig::try_new(2, 32, 128).unwrap(), layers, heads, dim)
            .unwrap();
    let all4 =
        AgentMemoryBudget::try_new(KiviConfig::try_new(4, 32, 128).unwrap(), layers, heads, dim)
            .unwrap();

    println!("Mixed-precision KV budget (KV cache only)");
    println!("  layers           {layers}");
    println!("  early 4-bit      {early}");
    println!("  later 2-bit      {}", layers - early);
    println!("  kv_heads         {heads}");
    println!("  head_dim         {dim}");
    println!(
        "  group / R        {} / {}",
        mixed.group_size(),
        mixed.residual_length()
    );
    println!();
    println!("Context per agent   {tokens} tokens");
    println!(
        "Marginal / token    {} bytes/agent (sum of layers)",
        mixed.marginal_bytes_per_token()
    );
    println!(
        "Per agent           {:.2} GiB",
        mixed.bytes_for_context(tokens) as f64 / gib()
    );
    println!(
        "  one 4-bit layer   {:.2} MiB",
        mixed.bytes_for_layer(0, tokens).unwrap() as f64 / mib()
    );
    println!(
        "  one 2-bit layer   {:.2} MiB",
        mixed.bytes_for_layer(early, tokens).unwrap() as f64 / mib()
    );
    println!();
    println!("Budget              {budget_gib:.0} GiB");
    println!(
        "  max agents, first {early} layers at 4-bit   {}",
        mixed.max_agents(budget_bytes, tokens)
    );
    println!(
        "  max agents, every layer at 2-bit   {}",
        all2.max_agents(budget_bytes, tokens)
    );
    println!(
        "  max agents, every layer at 4-bit   {}",
        all4.max_agents(budget_bytes, tokens)
    );
    println!(
        "  compression vs fp16   {:.2}x",
        mixed.compression_ratio_vs_fp16(1, tokens)
    );
}

fn mib() -> f64 {
    1024.0 * 1024.0
}

fn gib() -> f64 {
    mib() * 1024.0
}

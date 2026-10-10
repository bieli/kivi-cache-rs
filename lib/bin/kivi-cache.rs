// `kivi-cache` - context economics and KV planning CLI.

use kivi_cache::{
    compute_report, kv_budget_bytes, kv_strategies, parse_context_tokens, EconInputs,
    ImplementationStatus, KiviConfig, KiviError, ModelPreset, StrategyFamily,
};
use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            eprintln!();
            print_help();
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help" | "help") {
        print_help();
        return Ok(());
    }
    match args[0].as_str() {
        "econ" => run_econ(&args[1..]),
        "models" => {
            list_models();
            Ok(())
        }
        "strategies" => {
            list_strategies(&args[1..]);
            Ok(())
        }
        cmd => Err(format!("unknown command `{cmd}`")),
    }
}

fn run_econ(args: &[String]) -> Result<(), String> {
    let mut model: Option<String> = None;
    let mut agents: usize = 1;
    let mut ctx: Option<String> = None;
    let mut bits: u8 = 2;
    let mut group_size: usize = 32;
    let mut residual: usize = 128;
    let mut gpu_gib: f64 = 24.0;
    let mut weights_gib: Option<f64> = None;
    let mut usd_per_hour: f64 = 1.10;
    let mut layers: Option<usize> = None;
    let mut kv_heads: Option<usize> = None;
    let mut head_dim: Option<usize> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_econ_help();
                return Ok(());
            }
            "--model" => {
                model = Some(next_value(args, &mut i, "--model")?);
            }
            "--agents" => {
                agents = next_value(args, &mut i, "--agents")?
                    .parse()
                    .map_err(|_| "invalid --agents".to_string())?;
            }
            "--ctx" | "--context" => {
                ctx = Some(next_value(args, &mut i, "--ctx")?);
            }
            "--bits" => {
                bits = next_value(args, &mut i, "--bits")?
                    .parse()
                    .map_err(|_| "invalid --bits".to_string())?;
            }
            "--group-size" => {
                group_size = next_value(args, &mut i, "--group-size")?
                    .parse()
                    .map_err(|_| "invalid --group-size".to_string())?;
            }
            "--residual" => {
                residual = next_value(args, &mut i, "--residual")?
                    .parse()
                    .map_err(|_| "invalid --residual".to_string())?;
            }
            "--gpu-gb" => {
                gpu_gib = next_value(args, &mut i, "--gpu-gb")?
                    .parse()
                    .map_err(|_| "invalid --gpu-gb".to_string())?;
            }
            "--weights-gb" => {
                weights_gib = Some(
                    next_value(args, &mut i, "--weights-gb")?
                        .parse()
                        .map_err(|_| "invalid --weights-gb".to_string())?,
                );
            }
            "--usd-per-hour" => {
                usd_per_hour = next_value(args, &mut i, "--usd-per-hour")?
                    .parse()
                    .map_err(|_| "invalid --usd-per-hour".to_string())?;
            }
            "--layers" => {
                layers = Some(
                    next_value(args, &mut i, "--layers")?
                        .parse()
                        .map_err(|_| "invalid --layers".to_string())?,
                );
            }
            "--kv-heads" => {
                kv_heads = Some(
                    next_value(args, &mut i, "--kv-heads")?
                        .parse()
                        .map_err(|_| "invalid --kv-heads".to_string())?,
                );
            }
            "--head-dim" => {
                head_dim = Some(
                    next_value(args, &mut i, "--head-dim")?
                        .parse()
                        .map_err(|_| "invalid --head-dim".to_string())?,
                );
            }
            other => return Err(format!("unknown flag `{other}`")),
        }
        i += 1;
    }

    let model_id = model.ok_or("--model is required (or pass --layers --kv-heads --head-dim)")?;
    let ctx_raw = ctx.ok_or("--ctx is required")?;

    let config =
        KiviConfig::try_new(bits, group_size, residual).map_err(|e: KiviError| e.to_string())?;
    let context_tokens = parse_context_tokens(&ctx_raw)?;
    let inputs = if let Some(preset) = ModelPreset::by_id(&model_id) {
        if layers.is_some() || kv_heads.is_some() || head_dim.is_some() {
            return Err(
                "use either --model preset or --layers/--kv-heads/--head-dim, not both".into(),
            );
        }
        let weight_gib = weights_gib.unwrap_or(preset.weight_gib);
        let kv_budget_bytes = kv_budget_bytes(gpu_gib, weight_gib);
        preset.to_inputs(
            config,
            context_tokens,
            agents,
            kv_budget_bytes,
            usd_per_hour,
            Some(weight_gib),
        )
    } else if let (Some(l), Some(h), Some(d)) = (layers, kv_heads, head_dim) {
        let weight_gib = weights_gib.unwrap_or(0.0);
        let kv_budget_bytes = kv_budget_bytes(gpu_gib, weight_gib);
        EconInputs {
            model_id: model_id.clone(),
            num_layers: l,
            num_kv_heads: h,
            head_dim: d,
            weight_gib,
            config,
            context_tokens,
            agents,
            kv_budget_bytes,
            usd_per_hour,
        }
    } else {
        return Err(format!(
            "unknown model `{model_id}`; run `kivi-cache models` or pass --layers --kv-heads --head-dim"
        ));
    };
    let report = compute_report(&inputs).map_err(|e: KiviError| e.to_string())?;

    print_report(&report, gpu_gib, inputs.weight_gib, inputs.kv_budget_bytes);
    Ok(())
}

fn print_report(report: &kivi_cache::EconReport, gpu_gib: f64, weight_gib: f64, kv_budget: u64) {
    println!("kivi-cache econ (KV cache only - weights excluded from KV totals)");
    println!("  model             {}", report.model_id);
    println!("  KIVI bits         {}", report.bits);
    println!("  context / agent   {} tokens", report.context_tokens);
    println!("  agents requested  {}", report.agents);
    println!();
    println!("KV footprint");
    println!("  KIVI              {:.2} GiB", report.kivi_kv_gib);
    println!("  fp16 baseline     {:.2} GiB", report.fp16_kv_gib);
    println!(
        "  saved             {:.2} GiB ({:.2}x)",
        report.kv_saved_gib, report.compression_ratio
    );
    println!();
    println!(
        "GPU budget          {:.1} GiB total, {:.1} GiB weights -> {:.2} GiB for KV",
        gpu_gib,
        weight_gib,
        kv_budget as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("  max agents (KIVI) {}", report.max_agents_on_gpu);
    println!("  max agents (fp16) {}", report.max_agents_fp16_on_gpu);
    println!(
        "  fits request?     {}",
        if report.fits_requested { "yes" } else { "no" }
    );
    println!();
    println!("Cost (rough, GPU = ${:.2}/hr)", report.usd_per_hour);
    println!("  $/agent/hr @ req   ${:.4}", report.cost_per_agent_hour);
    println!(
        "  KV savings vs fp16  ${:.2}/hr (same agent count, KV-proportional)",
        report.kv_savings_usd_per_hour
    );
}

fn list_strategies(args: &[String]) {
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");
    println!("KV cache strategies (see docs/kv-strategies.md for full comparison)\n");
    println!(
        "{:<18} {:<10} {:<6} {:<6} STATUS",
        "ID", "FAMILY", "FULL?", "TUNING"
    );
    println!("{}", "-".repeat(72));
    for p in kv_strategies() {
        println!(
            "{:<18} {:<10} {:<6} {:<6} {}",
            p.id,
            family_label(p.family),
            yes_no(p.retains_full_context),
            yes_no(p.tuning_free),
            status_label(p.status),
        );
        if verbose {
            println!("  name:     {}", p.name);
            println!("  kv/fp16:  {}", p.typical_kv_vs_fp16);
            println!("  best:     {}", p.best_for);
            println!("  caveat:   {}", p.agent_caveat);
            println!("  ref:      {}", p.reference_url);
            println!();
        }
    }
    if !verbose {
        println!("\nRun `kivi-cache strategies --verbose` for details.");
    }
}

fn family_label(f: StrategyFamily) -> &'static str {
    match f {
        StrategyFamily::Quantization => "quant",
        StrategyFamily::Eviction => "evict",
    }
}

fn status_label(s: ImplementationStatus) -> &'static str {
    match s {
        ImplementationStatus::Implemented => "in crate",
        ImplementationStatus::DocumentedOnly => "docs only",
    }
}

fn yes_no(v: bool) -> &'static str {
    if v {
        "yes"
    } else {
        "no"
    }
}

fn list_models() {
    println!("Model presets (`kivi-cache econ --model <id>`):\n");
    for p in ModelPreset::all() {
        println!(
            "  {:<18} {:>2} layers  {:>2} kv_heads  {:>3} head_dim  ~{:.0} GiB weights",
            p.id, p.num_layers, p.num_kv_heads, p.head_dim, p.weight_gib
        );
    }
}

fn next_value(args: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn print_help() {
    println!(
        "    kivi-cache - KIVI KV-cache tools

USAGE:
    kivi-cache econ [OPTIONS]
    kivi-cache models
    kivi-cache strategies [--verbose]

COMMANDS:
    econ         Context economics: GB saved, max batch, rough $/hr
    models       List built-in model presets
    strategies   Compare KIVI vs H2O, StreamingLLM, INT4 (tradeoffs)

Run `kivi-cache econ --help` for options.
Full guide: docs/kv-strategies.md"
    );
}

fn print_econ_help() {
    println!(
        "kivi-cache econ - context economics (KV cache only)

USAGE:
    kivi-cache econ --model <id> --ctx <tokens> [OPTIONS]

OPTIONS:
    --model <id>          Preset id (see `kivi-cache models`)
    --ctx <n>             Context length (e.g. 128k, 200000, 1.5m)
    --agents <n>          Parallel agent sessions [default: 1]
    --bits <2|4|8>        KIVI bit width [default: 2]
    --group-size <n>      Quantization group size [default: 32]
    --residual <n>        Residual window [default: 128]
    --gpu-gb <gib>        Total GPU memory [default: 24]
    --weights-gb <gib>    Weight reservation (defaults to preset)
    --usd-per-hour <usd>  GPU hourly rate for rough costs [default: 1.10]
    --layers/--kv-heads/--head-dim   Custom shape (use id `custom`)

EXAMPLE:
    kivi-cache econ --model llama3-8b --agents 16 --ctx 128k --bits 2"
    );
}

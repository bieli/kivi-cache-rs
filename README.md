# kivi-cache

Rust library for **KIVI**: tuning-free asymmetric quantization of the LLM key–value cache
([ICML 2024](https://proceedings.mlr.press/v235/liu24bz.html), [arXiv:2402.02750](https://arxiv.org/abs/2402.02750),
[original PyTorch/CUDA repo](https://github.com/jy-yuan/KIVI)).

Keys are quantized **per channel**, values **per token**, into 2-bit (or 4/8-bit) packed integers.
A short full-precision residual of the newest tokens stays exact. Prefill attention uses the
original K/V; only the stored cache is compressed.

## Why this layout in Rust

| Choice | Rationale |
| --- | --- |
| Packed codes as the real store | Memory savings only show up if the cache never stays dense fp16 after a fake quantize–dequantize. |
| Fused attention over tiles | Matches the paper’s Q_MatMul idea: score and mix without materializing a full dequantized history. |
| Fast path for 2-bit / group 32 | Paper defaults; one key channel-group is two `u32`s, which keeps the inner loop tiny. |
| Streaming residual rules | Keys flush every `R` tokens (whole groups). Values keep a sliding window of `R` newest tokens. |
| Flat `[batch, heads, seq, dim]` slices | Zero-copy friendly for candle / burn / custom runtimes; no ndarray dependency. |
| `#![forbid(unsafe_code)]` | Correct packing and soft-max first; GPU/SIMD kernels can plug in later. |

## Algorithm (short)

1. **Per-channel keys.** For each channel, take `group_size` consecutive tokens, store
   `(scale, zero)` and packed unsigned codes.
2. **Per-token values.** For each token, take `group_size` consecutive channels the same way.
3. **Residual.** Newest key tokens wait in fp32 until a flush of length `residual_length`
   (`R % group_size == 0`). Newest value tokens stay in a window of length `R`; older ones
   are quantized as they leave.
4. **Decode attention.** Softmax over `[Q * K_qᵀ , Q * K_rᵀ]`, then
   `A_g V_q + A_r V_r`, with dequantization fused into the matmuls.

Defaults match the paper: `bits = 2`, `group_size = 32`, `residual_length = 128`.

## Usage

```rust
use kivi_cache::{KiviCache, KiviConfig};

let cfg = KiviConfig::default(); // 2-bit, G=32, R=128
let mut cache = KiviCache::new(cfg, /*batch*/ 1, /*kv_heads*/ 8, /*head_dim*/ 128)?;

// After prefill attention on full-precision K/V:
cache.append(&keys, &values, prompt_len)?;

// Each decode step: append first, then attend.
cache.append(&new_k, &new_v, 1)?;
let out = cache.attention(&query, /*num_q_heads*/ 8, 1.0 / (128f32).sqrt())?;

let report = cache.memory();
println!("{:.2}x vs fp16", report.compression_ratio_vs_fp16());
```

Grouped-query attention is supported: `num_q_heads` may be a multiple of `num_kv_heads`.

## Agent memory budget

The Agent memory budget API is in place. Here is what was added:

Answer questions like *“I have 200k tokens of tool logs - can I still run 8 parallel agents on 24 GiB?”* without loading weights or instantiating a model. Byte estimates follow the same layout as [`MemoryReport`](src/cache.rs) and match live cache accounting (validated in tests).

### New types

- **`AgentMemoryBudget`** - plan from KIVI config + model shape (layers, KV heads, head dim).
- **`MemoryBudgetSnapshot`** - live stats from a running [`KiviCache`](src/cache.rs).

### Planning API (`AgentMemoryBudget`)

| Method | Purpose |
| --- | --- |
| `try_new(config, num_layers, kv_heads, head_dim)` | Build a planner |
| `from_cache` / `KiviCache::memory_budget(num_layers)` | Derive planner from an existing cache |
| `marginal_bytes_per_token()` | Average bytes per token per layer (over one key-flush cycle) |
| `next_token_bytes(current_tokens)` | Exact byte delta for appending one more token |
| `bytes_for_context(tokens)` | Total KV bytes for one agent (all layers) |
| `bytes_for_agents(agents, tokens)` | Total KV bytes for parallel sessions |
| `max_agents(budget_bytes, tokens)` | How many agents fit in the budget |
| `fits(budget, agents, tokens)` | Boolean capacity check |
| `fp16_bytes_for_agents` / `compression_ratio_vs_fp16` | Compare against dense fp16 KV |

`marginal_bytes_per_token` is an **average** over a key-flush cycle once the value sliding window is full. Individual decode steps can spike higher while keys sit in the fp32 residual; use `next_token_bytes` for per-step growth.

### Live cache API (`KiviCache`)

| Method | Purpose |
| --- | --- |
| `tokens_until_key_flush()` | Tokens until the key residual buffer flushes to packed storage |
| `value_residual_slots_remaining()` | Free slots in the value fp32 sliding window |
| `budget_snapshot()` | Current footprint, flush horizon, and marginal rate |

### Quick start

```rust
use kivi_cache::{AgentMemoryBudget, KiviConfig};

let plan = AgentMemoryBudget::try_new(
    KiviConfig::default(),
    /*num_layers*/ 32,
    /*num_kv_heads*/ 8,
    /*head_dim*/ 128,
)?;

let tokens = 200_000;
let budget_bytes = 24 * 1024 * 1024 * 1024; // 24 GiB for KV only

println!("max agents: {}", plan.max_agents(budget_bytes, tokens));
println!("avg marginal: {} B/token/layer", plan.marginal_bytes_per_token());
println!("next token at 5000: {} B", plan.next_token_bytes(5000));
println!(
    "8 agents need {:.1} GiB KV",
    plan.bytes_for_agents(8, tokens) as f64 / (1024.0 * 1024.0 * 1024.0)
);
```

On a live cache, inspect flush horizon and current footprint:

```rust
println!("flush in {} tokens", cache.tokens_until_key_flush());
let snap = cache.budget_snapshot();
```

### Example output (Llama-3-8B-ish, 200k tokens, 24 GiB KV budget)

Running `cargo run --example agent_memory_budget --release`:

```
KIVI agent memory budget (KV cache only, one layer shape)
  layers           32
  kv_heads         8
  head_dim         128
  bits / group / R  2 / 32 / 128

Context per agent   200000 tokens
Marginal / token    1024 bytes/layer (steady state)
Per agent (all layers) 6271 MiB

Target deployment   8 agents
  KIVI KV total     48.99 GiB
  fp16 KV total     195.31 GiB
  compression       3.99x vs fp16

Budget              24 GiB
  fits 8 agents? no
  max agents @ 200000 tok  3
```

Eight agents at 200k tokens need ~49 GiB of KIVI-compressed KV alone; a 24 GiB KV budget fits **3** agents at that context length (~4x smaller than fp16).

## Context economics CLI

Rough GB saved, max parallel agents on a GPU, and $/hour estimates (KV cache only):

```bash
cargo run --release --bin kivi-cache -- econ \
  --model llama3-8b \
  --agents 16 \
  --ctx 128k \
  --bits 2 \
  --gpu-gb 24 \
  --usd-per-hour 1.10

cargo run --release --bin kivi-cache -- models
```

Built-in presets: `llama2-7b`, `llama2-13b`, `llama3-8b`, `mistral-7b`, `falcon-7b`, `longchat-7b-32k`.
Custom shapes: `--model custom --layers 32 --kv-heads 8 --head-dim 128 --weights-gb 16`.

## Examples

Each example is a small synthetic program. None of them loads model weights.

**`streaming_cache`** - why keys are quantized per channel, and what a long cache costs.
One channel is a large outlier and the rest are quiet. Per-channel 2-bit keeps those quiet channels; per-token 2-bit does not. Then a 2048-token layer (8 heads, dim 128) is appended as a prefill plus four decode steps, and the program prints residual lengths, packed bytes, and the compression ratio against fp16.

```bash
cargo run --example streaming_cache --release
```

**`decode_attention`** - decode quality against a full-precision cache.
A 96-token prompt is stored in KIVI (2-bit, group 32, residual 64). Each of the next four steps appends one token and runs attention. The printout is cosine similarity and max absolute error versus the same attention on fp32 K/V, plus the key-residual length after that step.

```bash
cargo run --example decode_attention --release
```

**`agent_memory_budget`** - how many parallel sessions fit in a KV budget.
Plans a Llama-3-8B-shaped cache (32 layers, 8 KV heads, dim 128) at the paper defaults. For 8 agents at 200k tokens each it prints KIVI versus fp16 gigabytes and how many of those agents fit in 24 GiB of KV memory.

```bash
cargo run --example agent_memory_budget --release
```

**`passkey_retrieval`** - a needle in the middle of a long context is still readable.
A 7-digit passkey (`2718281`) is planted at token 256 of a 512-token haystack, outside the fp32 residual window. After the haystack is stored, one decode query tries to read the key back. The table sweeps 2/4/8 bits and residual lengths 32 and 128, and reports whether the key was recovered from packed storage.

```bash
cargo run --example passkey_retrieval --release
```

## Tests

```bash
cargo test
```

Coverage includes round-trip quantization (including half-to-even ties), Algorithm 1 residual
boundaries, chunk-invariant streaming, fused attention vs dense reference, and GQA head mapping.

## KV memory strategies (comparison hub)

Choosing a KV policy for agents is not “KIVI or nothing.” This repo documents the tradeoffs
between **quantization** (keep every token, store cheaply) and **eviction** (drop tokens).

| Strategy | In this crate? | Full context? | Agent note |
| --- | --- | --- | --- |
| **KIVI** | **Yes** | Yes (compressed) | Long tool logs, RAG, parallel sessions |
| Vanilla INT4 KV | Docs | Yes | Simpler; use when 4-bit is enough |
| Vanilla INT2 symmetric | Docs | Yes | Avoid - reasoning breaks without asymmetry |
| **H2O** | Docs | No (eviction) | OK if old turns can be forgotten |
| **StreamingLLM** | Docs | No (window) | Streaming gen; bad for buried tool results |

**Full guide:** [docs/kv-strategies.md](docs/kv-strategies.md)

```bash
cargo run --release --bin kivi-cache -- strategies --verbose
```

Rust API: `kivi_cache::kv_strategies()`.

## Scope

This crate is the **KV cache + fused attention** core. It does not load Hugging Face weights or
run Llama/Mistral end to end. Hook it into your Rust inference stack by replacing the layer cache
with `KiviCache` and keeping prefill attention on the full-precision tensors returned by the
projections.

## License

MIT. The KIVI method and reference implementation are from Liu, Yuan, et al. (ICML 2024).

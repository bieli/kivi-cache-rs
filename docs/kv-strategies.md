# KV memory strategies for agent systems

This page compares common **key–value cache compression** approaches for long-running,
multi-turn, and tool-heavy agents. `kivi-cache` implements **KIVI**; the others are
documented here so you can pick (or combine) strategies without treating any single paper
as the whole design space.

**Scope:** all numbers are qualitative unless tied to a cited benchmark. Eviction methods
shrink *what you remember*; quantization shrinks *how you store what you keep*.

---

## At a glance

| Strategy | Mechanism | Full context retained? | Tuning | Typical footprint | Agent sweet spot |
| --- | --- | --- | --- | --- | --- |
| **KIVI** (this crate) | Asymmetric 2–8 bit quant: keys per-channel, values per-token + fp32 residual window | **Yes** (lossy compress, no drop) | **None** | ~4x vs fp16 KV at 2-bit (paper + this crate’s accounting) | Long tool logs, RAG prefill, many parallel sessions, passkey-style retrieval |
| **Vanilla INT4 KV** | Round-to-nearest 4-bit (often per-token for **both** K and V) | Yes | None | ~4x vs fp16 | Simple drop-in when 4-bit is enough; weak at 2-bit |
| **Vanilla INT2 KV** | Same as INT4 but 2-bit symmetric | Yes | None | ~8x vs fp16 (in theory) | **Avoid** for agents - accuracy collapses without KIVI’s asymmetry + residual |
| **H2O** | Evict low-attention tokens; keep “heavy hitters” + recent | **No** - tokens deleted | Usually none at inference | Bounded by cache budget (not fixed ratio) | High-throughput chat when middle context is rarely re-attended |
| **StreamingLLM** | Keep attention-sink tokens + recent sliding window | **No** - middle dropped | None (sink tokens fixed) | Bounded window | “Infinite” streaming decode; **bad** for buried tool results |

---

## How to read this as an agent architect

```text
Need every token still addressable at decode?
  ├─ Yes -> quantization family (KIVI, INT4, ...)
  │         └─ Need 2-bit + quality -> KIVI (asymmetric axes + residual)
  └─ No  -> eviction family (H2O, StreamingLLM, ...)
            └─ OK losing middle context -> StreamingLLM
            └─ Want importance-based retention -> H2O
```

Agents that **append** observations (ReAct, MCP tools, sub-agent transcripts) usually
need **retained** context, not a hard window - that pushes you toward **KIVI or INT4**,
not pure eviction.

---

## KIVI (implemented here)

**Idea:** Key caches have a few high-magnitude **channels**; value caches vary more **per
token**. Quantize keys **per channel** (group tokens) and values **per token** (group
channels). Keep the newest `R` tokens in fp32 (residual) so local decode stays sharp.

| Pros | Cons |
| --- | --- |
| Tuning-free; plug into existing Llama/Mistral/Falcon stacks | 2-bit still lossy - monitor GSM8K-style reasoning |
| **Full sequence** stays in cache (compressed) | Asymmetric layout is harder to hand-roll than uniform INT4 |
| Strong 2-bit vs fake “INT2 both axes” (see KIVI paper Table 3) | Prefill still uses fp16/fp32 K/V; only **stored** cache is packed |
| Good LongBench / passkey results in paper | Extreme bit widths need kernel support for best throughput |
| Enables larger **batch of agents** on same GPU (memory-bound serving) | Does not reduce **weight** memory |

**Where KIVI wins**

- **Long context + retrieval:** needle-in-haystack, RAG prefill, audit logs - nothing is evicted.
- **No calibration dataset:** unlike GPTQ-style weight quant, no offline pass on KV.
- **2-bit agent fleets:** [`AgentMemoryBudget`](https://github.com/jy-yuan/KIVI) / `kivi-cache econ` math applies directly.
- **Asymmetric axes matter at 2-bit:** uniform INT2 (K per-token, V per-token) fails hard on reasoning tasks in the paper; KIVI-2 does not.

**References:** [KIVI ICML 2024](https://arxiv.org/abs/2402.02750), [code](https://github.com/jy-yuan/KIVI).

---

## Vanilla INT4 / INT2 KV quantization

**Idea:** Round-to-nearest with per-group scale/zero-point. In many serving stacks both
K and V are quantized **per token** because it streams naturally along the sequence dimension.

| Pros | Cons |
| --- | --- |
| Simple mental model; easy fake-quant baseline | **INT2 + symmetric axes** destroys quality (KIVI paper) |
| INT4 per-token often near fp16 on many tasks | Ignores channel outliers in keys |
| Supported in several inference frameworks | “Fake quant” wrappers may not save real memory (see crate README) |

**Versus KIVI**

| | Vanilla INT4 (K per-token, V per-token) | KIVI-2 |
| --- | --- | --- |
| GSM8K (Llama-2-7B, paper) | Usable at 4-bit; weak at 2-bit | ~12.74 vs 13.50 fp16 at 2-bit |
| Streaming append | Trivial per-token | Key side needs flush/residual rules |
| Memory at 2-bit | Similar ratio if actually packed | Similar ratio **with** better accuracy |

**Agent takeaway:** use **INT4** when you want the simplest quant path and 4x KV shrink is
enough. Use **KIVI** when you need **2-bit** or want the paper’s asymmetric layout without
tuning.

---

## H2O (Heavy-Hitter Oracle)

**Idea:** During decode, track approximate attention mass per cached token. **Evict** tokens
that are unlikely to be heavy hitters later; keep a small set of high-score tokens plus recency.

| Pros | Cons |
| --- | --- |
| Cache size **cap** independent of sequence length | **Destructive** - evicted tokens are gone |
| Can outperform uniform quant on some long-context tasks when budget is tiny | Policy errors = silent context loss (bad for agents) |
| Complements weight-only quant | Extra book-keeping / scoring during attention |
| No weight fine-tuning in typical use | Retrieval from evicted span is impossible |

**Versus KIVI**

| Dimension | H2O | KIVI |
| --- | --- | --- |
| Middle of 100k tool log | May evict rare but critical lines | Still present (compressed) |
| Memory scaling | Fixed budget | Grows with context, ~4x cheaper than fp16 |
| Parallel agents | Frees RAM like KIVI | Frees RAM via compression |
| Tuning | Inference heuristic | None |

**Agent takeaway:** H2O fits **throughput-first chat** with loose fidelity on old turns.
Avoid for **compliance, debugging, or tool replay** unless you explicitly want forgetting.

**Reference:** [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048).

---

## StreamingLLM

**Idea:** A few initial tokens act as **attention sinks**; keep those plus a **recent**
sliding window. Drop everything in the middle - enables very long *generation* with bounded KV.

| Pros | Cons |
| --- | --- |
| Bounded memory for unbounded *decode* | **Middle context is deleted** |
| Simple window + sink hyperparameters | Terrible for “read 50k tokens, answer from token 12k” |
| No model training for sink phenomenon | Not a replacement for RAG index - it’s cache eviction |
| Works well on streaming perplexity demos | Multi-agent **shared** prefix still needs full prefill store elsewhere |

**Versus KIVI**

| Scenario | StreamingLLM | KIVI |
| --- | --- | --- |
| Infinite generation after long prompt | Strong when middle can be forgotten | Keeps full prompt in compressed form |
| Passkey in middle of haystack | Fails if outside window | Paper NIAH benchmarks |
| Agent tool loop (500 steps) | Recent tools OK; old tools gone | Old tools still in cache (lossy) |

**Agent takeaway:** pair StreamingLLM with **external memory** (vector DB, filesystem) if
agents need long-horizon recall. KIVI keeps recall **inside** the KV tensor.

**Reference:** [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453).

---

## Combining strategies (not implemented here)

Real systems often **stack** ideas:

| Stack | When |
| --- | --- |
| **KIVI + larger batch** | Memory-bound agent serving on one GPU |
| **KIVI + paged KV (vLLM-style)** | Many concurrent sessions, variable length |
| **H2O or window + external RAG** | Bounded GPU cache + disk/cloud retrieval |
| **INT4 weights + KIVI KV** | Paper reports compatibility |

This crate does **not** implement H2O or StreamingLLM eviction - use it as the
**quantization layer** when you choose the left branch of the decision tree above.

---

## Choosing for common agent patterns

| Pattern | Recommended starting point | Why |
| --- | --- | --- |
| ReAct / tool loop, 10k–200k tokens | **KIVI** or INT4 | Full transcript stays addressable |
| RAG: huge static prefill + many queries | **KIVI** on shared prefix | Compress once, reuse; see `bytes_for_context` |
| Slack bot with 2k visible history | INT4 or fp16 | KIVI’s edge shows at long context / 2-bit |
| “Run forever” codegen stream | StreamingLLM (+ sinks) | Middle context intentionally discarded |
| Strict KV byte cap, OK to forget | H2O | Fixed RAM, heuristic retention |
| 8 agents x 128k on one 24 GiB GPU | **KIVI** + `kivi-cache econ` | Quantitative fit in this repo |

---

## Quick command reference (this repo)

```bash
# Compare footprint for a deployment shape
cargo run --release --bin kivi-cache -- econ \
  --model llama3-8b --agents 8 --ctx 128k --bits 2 --gpu-gb 80

# List strategy profiles (includes H2O / StreamingLLM / INT4 summaries)
cargo run --release --bin kivi-cache -- strategies
```

Programmatic profiles: `kivi_cache::strategies::all()`.

---

## Further reading

| Method | Link |
| --- | --- |
| KIVI | https://arxiv.org/abs/2402.02750 |
| H2O | https://arxiv.org/abs/2306.14048 |
| StreamingLLM | https://arxiv.org/abs/2309.17453 |
| Hugging Face KV quant docs (KIVI-inspired) | https://huggingface.co/docs/transformers/kv_cache |
| KV cache survey (2024) | https://arxiv.org/abs/2407.01527 |

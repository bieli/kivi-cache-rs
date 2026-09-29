# kivi-cache-rs

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


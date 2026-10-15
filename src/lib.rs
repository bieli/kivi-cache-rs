// Tuning-free asymmetric KV-cache quantization (KIVI, ICML 2024).
//
// KIVI compresses the key cache **per channel** and the value cache **per token**
// to 2 or 4 bits, and keeps a short full-precision residual of the newest tokens.
// This crate is that cache: packed integer storage, streaming updates, and
// attention that dequantizes tiles while it scores them.
//
// It is not an LLM runtime. Project Q, K, and V in the caller. Run prefill
// attention on the full-precision tensors, then [`KiviCache::append`] them.
// On each decode step, append the new K and V before [`KiviCache::attention`].
//
// ```
// use kivi_cache::{KiviCache, KiviConfig};
//
// let mut cache = KiviCache::new(KiviConfig::default(), 1, 1, 128).unwrap();
// let seq = 160;
// let n = seq * 128;
// let keys = vec![0.25_f32; n];
// let values = vec![-1.5_f32; n];
// cache.append(&keys, &values, seq).unwrap();
// assert_eq!(cache.sequence_length(), 160);
// assert_eq!(cache.key_residual_len(), 32);
// assert_eq!(cache.value_residual_len(), 128);
// let (k_hat, v_hat) = cache.dequantize();
// assert_eq!(k_hat, keys);
// assert_eq!(v_hat, values);
// ```
//
// The algorithm is described in Liu, Yuan, et al., *KIVI: A Tuning-Free
// Asymmetric 2bit Quantization for KV Cache*, ICML 2024,
// [arXiv:2402.02750](https://arxiv.org/abs/2402.02750).

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

mod attention;
mod budget;
mod cache;
mod config;
mod econ;
mod error;
mod guard;
mod prefix;
mod quant;
mod strategies;
mod trace;

pub use budget::{AgentMemoryBudget, MemoryBudgetSnapshot, MixedPrecisionBudget};
pub use cache::{KiviCache, MemoryReport};
pub use config::KiviConfig;
pub use econ::{
    compute_report, estimate_pretrain, gpus_for_kv, kv_budget_bytes, parse_context_tokens,
    EconInputs, EconReport, ModelPreset, PretrainEstimate,
};
pub use error::KiviError;
pub use guard::{FlushCheck, QualityGuard, QualityPolicy, QualityUpgrade};
pub use prefix::{RagSession, SharedPrefix, SharedPrefixFootprint};
pub use quant::{quantize_per_channel, quantize_per_token, QuantAxis, QuantizedTensor};
pub use strategies::{
    all as kv_strategies, by_id as kv_strategy_by_id, implemented as implemented_kv_strategies,
    ImplementationStatus, StrategyFamily, StrategyProfile,
};
pub use trace::{CacheSnapshot, ForgottenTokens};

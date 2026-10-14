use kivi_cache::{KiviCache, KiviConfig, KiviError};

fn cfg() -> KiviConfig {
    KiviConfig::try_new(2, 32, 64).unwrap()
}

fn document(batch: usize, heads: usize, seq: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let n = batch * heads * seq * dim;
    let mut keys = vec![0.0; n];
    let mut values = vec![0.0; n];
    for (i, k) in keys.iter_mut().enumerate() {
        *k = ((i * 17) % 23) as f32 / 11.0 - 0.4;
    }
    for (i, v) in values.iter_mut().enumerate() {
        *v = ((i * 13) % 19) as f32 / 9.0 - 0.2;
    }
    (keys, values)
}

#[allow(clippy::too_many_arguments)]
fn dense_attend(
    keys: &[f32],
    values: &[f32],
    query: &[f32],
    batch: usize,
    kv_heads: usize,
    q_heads: usize,
    seq: usize,
    dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; batch * q_heads * dim];
    let group = q_heads / kv_heads;
    let mut logits = vec![0.0; seq];
    for b in 0..batch {
        for qh in 0..q_heads {
            let kvh = qh / group;
            let hi = b * kv_heads + kvh;
            let q = &query[(b * q_heads + qh) * dim..][..dim];
            let base = hi * seq * dim;
            for t in 0..seq {
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += q[d] * keys[base + t * dim + d];
                }
                logits[t] = dot * scale;
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for logit in logits.iter_mut() {
                *logit = (*logit - max).exp();
                sum += *logit;
            }
            for logit in logits.iter_mut() {
                *logit /= sum;
            }
            let y = &mut out[(b * q_heads + qh) * dim..][..dim];
            y.fill(0.0);
            for t in 0..seq {
                let w = logits[t];
                for d in 0..dim {
                    y[d] += w * values[base + t * dim + d];
                }
            }
        }
    }
    out
}

fn concat_head_major(
    prefix: &[f32],
    extra: &[f32],
    batch: usize,
    heads: usize,
    prefix_len: usize,
    extra_len: usize,
    dim: usize,
) -> Vec<f32> {
    let total = prefix_len + extra_len;
    let mut out = vec![0.0; batch * heads * total * dim];
    for b in 0..batch {
        for h in 0..heads {
            let hi = b * heads + h;
            let dst = hi * total * dim;
            let src = hi * prefix_len * dim;
            let n = prefix_len * dim;
            out[dst..dst + n].copy_from_slice(&prefix[src..src + n]);
            let src = hi * extra_len * dim;
            let n = extra_len * dim;
            out[dst + prefix_len * dim..dst + prefix_len * dim + n]
                .copy_from_slice(&extra[src..src + n]);
        }
    }
    out
}

#[test]
fn sessions_share_one_prefix_allocation() {
    let (keys, values) = document(1, 2, 160, 64);
    let mut cache = KiviCache::new(cfg(), 1, 2, 64).unwrap();
    cache.append(&keys, &values, 160).unwrap();
    let prefix = cache.share_prefix();
    let again = prefix.clone();
    assert!(prefix.shares_allocation_with(&again));
    assert_eq!(prefix.prefix_tokens(), 160);
    let a = prefix.session();
    let b = prefix.session();
    assert!(a.prefix().shares_allocation_with(b.prefix()));
    assert_eq!(a.prefix().prefix_bytes(), prefix.prefix_bytes());
}

#[test]
fn empty_question_matches_cache_attention() {
    let (keys, values) = document(1, 2, 96, 64);
    let mut cache = KiviCache::new(cfg(), 1, 2, 64).unwrap();
    cache.append(&keys, &values, 96).unwrap();
    let prefix = cache.share_prefix();
    let session = prefix.session();
    let query = vec![0.3; 4 * 64];
    let scale = 1.0 / (64.0_f32).sqrt();
    let from_session = session.attention(&query, 4, scale).unwrap();
    let from_cache = cache.attention(&query, 4, scale).unwrap();
    assert_eq!(from_session, from_cache);
}

#[test]
fn question_attention_matches_dense_prefix_plus_suffix() {
    let batch = 1;
    let heads = 2;
    let dim = 64;
    let doc = 96;
    let qlen = 4;
    let (keys, values) = document(batch, heads, doc, dim);
    let mut cache = KiviCache::new(cfg(), batch, heads, dim).unwrap();
    cache.append(&keys, &values, doc).unwrap();
    let (dk, dv) = cache.dequantize();
    let prefix = cache.share_prefix();
    let mut session = prefix.session();
    let (qk, qv) = document(batch, heads, qlen, dim);
    session.append(&qk, &qv, qlen).unwrap();
    let full_k = concat_head_major(&dk, &qk, batch, heads, doc, qlen, dim);
    let full_v = concat_head_major(&dv, &qv, batch, heads, doc, qlen, dim);
    let query = vec![0.15; batch * heads * dim];
    let scale = 0.25;
    let got = session.attention(&query, heads, scale).unwrap();
    let expect = dense_attend(
        &full_k,
        &full_v,
        &query,
        batch,
        heads,
        heads,
        doc + qlen,
        dim,
        scale,
    );
    for (a, b) in got.iter().zip(expect.iter()) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

#[test]
fn one_session_does_not_see_another_question() {
    let (keys, values) = document(1, 1, 80, 64);
    let mut cache = KiviCache::new(cfg(), 1, 1, 64).unwrap();
    cache.append(&keys, &values, 80).unwrap();
    let prefix = cache.share_prefix();
    let mut a = prefix.session();
    let b = prefix.session();
    let (ak, av) = document(1, 1, 3, 64);
    a.append(&ak, &av, 3).unwrap();
    assert_eq!(b.question_tokens(), 0);
    assert_eq!(b.sequence_length(), 80);
    let query = vec![0.2; 64];
    let scale = 1.0;
    let only_prefix = prefix.session().attention(&query, 1, scale).unwrap();
    let from_b = b.attention(&query, 1, scale).unwrap();
    assert_eq!(only_prefix, from_b);
    let from_a = a.attention(&query, 1, scale).unwrap();
    assert_ne!(from_a, from_b);
}

#[test]
fn shared_footprint_counts_the_document_once() {
    let (keys, values) = document(1, 2, 256, 64);
    let mut cache = KiviCache::new(cfg(), 1, 2, 64).unwrap();
    cache.append(&keys, &values, 256).unwrap();
    let prefix = cache.share_prefix();
    let users = 8;
    let question = 8;
    let foot = prefix.footprint(users, question);
    assert_eq!(foot.prefix_bytes, cache.memory().compressed_bytes());
    assert!(foot.shared_total_bytes < foot.fp16_all_users_bytes);
    assert!(foot.savings_vs_fp16() > 1.0);
    let one = prefix.footprint(1, question);
    let two = prefix.footprint(2, question);
    assert_eq!(two.prefix_bytes, one.prefix_bytes);
    assert_eq!(two.question_bytes, 2 * one.question_bytes);
}

#[test]
fn bad_question_shape_is_rejected() {
    let mut cache = KiviCache::new(cfg(), 1, 1, 64).unwrap();
    cache.append(&vec![0.1; 64], &vec![0.2; 64], 1).unwrap();
    let prefix = cache.share_prefix();
    let mut session = prefix.session();
    let err = session.append(&[0.0; 8], &[0.0; 8], 1).unwrap_err();
    assert!(matches!(err, KiviError::Shape { .. }));
    let err = session.append(&[0.0; 64], &[0.0; 8], 1).unwrap_err();
    assert!(matches!(
        err,
        KiviError::Shape {
            expected: 64,
            actual: 8
        }
    ));
}

fn assert_close(got: &[f32], expect: &[f32], tol: f32) {
    assert_eq!(got.len(), expect.len());
    let err = got
        .iter()
        .zip(expect)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(err < tol, "max abs {err} >= {tol}");
}

fn share(
    cfg: KiviConfig,
    batch: usize,
    heads: usize,
    dim: usize,
    seq: usize,
) -> (KiviCache, kivi_cache::SharedPrefix) {
    let (keys, values) = document(batch, heads, seq, dim);
    let mut cache = KiviCache::new(cfg, batch, heads, dim).unwrap();
    cache.append(&keys, &values, seq).unwrap();
    let prefix = cache.share_prefix();
    (cache, prefix)
}

#[test]
fn constant_keys_make_suffix_attention_the_value_mean() {
    let batch = 1;
    let heads = 2;
    let dim = 32;
    let doc = 96;
    let qlen = 4;
    let config = KiviConfig::try_new(2, 32, 64).unwrap();
    let n = batch * heads * doc * dim;
    let keys = vec![0.0; n];
    let values = document(batch, heads, doc, dim).1;
    let mut cache = KiviCache::new(config, batch, heads, dim).unwrap();
    cache.append(&keys, &values, doc).unwrap();
    assert_eq!(cache.key_residual_len(), doc % 64);
    assert_eq!(cache.value_residual_len(), 64);
    assert_ne!(cache.key_residual_len(), cache.value_residual_len());

    let prefix = cache.share_prefix();
    let mut session = prefix.session();
    let qk = vec![0.0; batch * heads * qlen * dim];
    let (_, qv) = document(batch, heads, qlen, dim);
    session.append(&qk, &qv, qlen).unwrap();

    let (dk, dv) = cache.dequantize();
    assert!(dk.iter().all(|x| *x == 0.0));
    let full_v = concat_head_major(&dv, &qv, batch, heads, doc, qlen, dim);
    let expect = token_mean(&full_v, batch, heads, doc + qlen, dim);
    let query = vec![1.0; batch * heads * dim];
    let got = session.attention(&query, heads, 3.5).unwrap();
    assert_close(&got, &expect, 1e-5);
}

#[test]
fn four_bit_suffix_matches_dense_attention() {
    let batch = 1;
    let heads = 1;
    let dim = 64;
    let doc = 90;
    let qlen = 6;
    let config = KiviConfig::try_new(4, 32, 64).unwrap();
    let (cache, prefix) = share(config, batch, heads, dim, doc);
    assert_eq!(cache.key_residual_len(), 26);
    assert_eq!(cache.value_residual_len(), 64);
    let (dk, dv) = cache.dequantize();
    let mut session = prefix.session();
    let (qk, qv) = document(batch, heads, qlen, dim);
    session.append(&qk, &qv, qlen).unwrap();
    let full_k = concat_head_major(&dk, &qk, batch, heads, doc, qlen, dim);
    let full_v = concat_head_major(&dv, &qv, batch, heads, doc, qlen, dim);
    let query = document(batch, heads, 1, dim).0;
    let scale = 1.0 / (dim as f32).sqrt();
    let got = session.attention(&query, heads, scale).unwrap();
    let expect = dense_attend(
        &full_k,
        &full_v,
        &query,
        batch,
        heads,
        heads,
        doc + qlen,
        dim,
        scale,
    );
    assert_close(&got, &expect, 1e-4);
}

#[test]
fn grouped_query_batch_suffix_matches_dense_attention() {
    let batch = 2;
    let kv = 2;
    let q_heads = 6;
    let dim = 32;
    let doc = 48;
    let qlen = 3;
    let config = KiviConfig::try_new(2, 32, 32).unwrap();
    let (cache, prefix) = share(config, batch, kv, dim, doc);
    let (dk, dv) = cache.dequantize();
    let mut session = prefix.session();
    let (qk, qv) = document(batch, kv, qlen, dim);
    session.append(&qk, &qv, qlen).unwrap();
    let full_k = concat_head_major(&dk, &qk, batch, kv, doc, qlen, dim);
    let full_v = concat_head_major(&dv, &qv, batch, kv, doc, qlen, dim);
    let query = document(batch, q_heads, 1, dim).0;
    let scale = 0.5;
    let got = session.attention(&query, q_heads, scale).unwrap();
    let expect = dense_attend(
        &full_k,
        &full_v,
        &query,
        batch,
        kv,
        q_heads,
        doc + qlen,
        dim,
        scale,
    );
    assert_close(&got, &expect, 1e-4);
}

#[test]
fn chunked_question_append_matches_one_shot() {
    let batch = 2;
    let heads = 2;
    let dim = 32;
    let doc = 40;
    let config = KiviConfig::try_new(2, 32, 32).unwrap();
    let (_, prefix) = share(config, batch, heads, dim, doc);
    let (qk, qv) = document(batch, heads, 7, dim);
    let qk_a = slice_tokens(&qk, batch, heads, 7, 0, 2, dim);
    let qv_a = slice_tokens(&qv, batch, heads, 7, 0, 2, dim);
    let qk_b = slice_tokens(&qk, batch, heads, 7, 2, 5, dim);
    let qv_b = slice_tokens(&qv, batch, heads, 7, 2, 5, dim);
    let mut chunked = prefix.session();
    chunked.append(&qk_a, &qv_a, 2).unwrap();
    chunked.append(&qk_b, &qv_b, 5).unwrap();
    let mut once = prefix.session();
    once.append(&qk, &qv, 7).unwrap();
    assert_eq!(chunked.question_tokens(), 7);
    assert_eq!(chunked.sequence_length(), doc + 7);
    let query = vec![0.4; batch * heads * dim];
    let a = chunked.attention(&query, heads, 1.25).unwrap();
    let b = once.attention(&query, heads, 1.25).unwrap();
    assert_eq!(a, b);
}

#[test]
fn matching_suffix_key_dominates_the_softmax() {
    let dim = 32;
    let doc = 40;
    let config = KiviConfig::try_new(2, 32, 32).unwrap();
    let n = doc * dim;
    let keys = vec![0.01; n];
    let values = vec![0.0; n];
    let mut cache = KiviCache::new(config, 1, 1, dim).unwrap();
    cache.append(&keys, &values, doc).unwrap();
    let prefix = cache.share_prefix();
    let mut session = prefix.session();
    let qk = vec![1.0; dim];
    let qv = vec![5.0; dim];
    session.append(&qk, &qv, 1).unwrap();
    let query = vec![1.0; dim];
    let got = session.attention(&query, 1, 20.0).unwrap();
    assert_close(&got, &qv, 1e-3);
}

#[test]
fn question_only_session_matches_dense_attention() {
    let cache = KiviCache::new(cfg(), 1, 2, 64).unwrap();
    let prefix = cache.share_prefix();
    let mut session = prefix.session();
    let (qk, qv) = document(1, 2, 5, 64);
    session.append(&qk, &qv, 5).unwrap();
    let query = vec![0.2; 2 * 64];
    let got = session.attention(&query, 2, 0.7).unwrap();
    let expect = dense_attend(&qk, &qv, &query, 1, 2, 2, 5, 64, 0.7);
    assert_close(&got, &expect, 1e-5);
    assert_eq!(prefix.prefix_tokens(), 0);
    assert_eq!(prefix.prefix_bytes(), 0);
}

#[test]
fn footprint_matches_the_byte_formula() {
    let batch = 2;
    let heads = 2;
    let dim = 32;
    let doc = 70;
    let sessions = 3;
    let question = 5;
    let config = KiviConfig::try_new(2, 32, 64).unwrap();
    let (cache, prefix) = share(config, batch, heads, dim, doc);
    let foot = prefix.footprint(sessions, question);
    let elements = batch * heads * dim;
    let question_bytes = sessions * question * elements * 2 * 4;
    let tokens = doc + question;
    let fp16_per_user = 2 * tokens * elements * 2;
    assert_eq!(foot.prefix_tokens, doc);
    assert_eq!(foot.question_tokens, question);
    assert_eq!(foot.sessions, sessions);
    assert_eq!(foot.prefix_bytes, cache.memory().compressed_bytes());
    assert_eq!(foot.question_bytes, question_bytes);
    assert_eq!(foot.shared_total_bytes, foot.prefix_bytes + question_bytes);
    assert_eq!(foot.fp16_per_user_bytes, fp16_per_user);
    assert_eq!(foot.fp16_all_users_bytes, sessions * fp16_per_user);
    let ratio = foot.fp16_all_users_bytes as f64 / foot.shared_total_bytes as f64;
    assert!((foot.savings_vs_fp16() - ratio).abs() < 1e-12);
}

#[test]
fn empty_store_reports_unit_savings() {
    let cache = KiviCache::new(cfg(), 1, 1, 64).unwrap();
    let foot = cache.share_prefix().footprint(4, 0);
    assert_eq!(foot.prefix_bytes, 0);
    assert_eq!(foot.question_bytes, 0);
    assert_eq!(foot.shared_total_bytes, 0);
    assert_eq!(foot.fp16_all_users_bytes, 0);
    assert_eq!(foot.savings_vs_fp16(), 1.0);
}

#[test]
fn appending_a_question_leaves_the_prefix_unchanged() {
    let (cache, prefix) = share(cfg(), 1, 1, 64, 80);
    let before_tokens = prefix.prefix_tokens();
    let before_bytes = prefix.prefix_bytes();
    let query = vec![0.25; 64];
    let before = prefix.session().attention(&query, 1, 1.0).unwrap();
    let mut session = prefix.session();
    let (qk, qv) = document(1, 1, 4, 64);
    session.append(&qk, &qv, 4).unwrap();
    session.append(&[], &[], 0).unwrap();
    assert_eq!(prefix.prefix_tokens(), before_tokens);
    assert_eq!(prefix.prefix_bytes(), before_bytes);
    assert_eq!(cache.sequence_length(), before_tokens);
    let after = prefix.session().attention(&query, 1, 1.0).unwrap();
    assert_eq!(before, after);
}

#[test]
fn bad_query_shape_and_head_count_are_rejected() {
    let (_, prefix) = share(cfg(), 1, 2, 64, 32);
    let session = prefix.session();
    let heads = session.attention(&[0.0; 64], 3, 1.0).unwrap_err();
    assert_eq!(
        heads,
        KiviError::QueryHeads {
            num_q_heads: 3,
            num_kv_heads: 2,
        }
    );
    let shape = session.attention(&[0.0; 8], 2, 1.0).unwrap_err();
    assert_eq!(
        shape,
        KiviError::Shape {
            expected: 2 * 64,
            actual: 8,
        }
    );
}

fn slice_tokens(
    src: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    start: usize,
    len: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; batch * heads * len * dim];
    for b in 0..batch {
        for h in 0..heads {
            let hi = b * heads + h;
            let from = hi * seq * dim + start * dim;
            let to = hi * len * dim;
            out[to..to + len * dim].copy_from_slice(&src[from..from + len * dim]);
        }
    }
    out
}

fn token_mean(values: &[f32], batch: usize, heads: usize, seq: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; batch * heads * dim];
    for b in 0..batch {
        for h in 0..heads {
            let base = (b * heads + h) * seq * dim;
            let y = &mut out[(b * heads + h) * dim..][..dim];
            for t in 0..seq {
                for d in 0..dim {
                    y[d] += values[base + t * dim + d];
                }
            }
            for y_d in y.iter_mut() {
                *y_d /= seq as f32;
            }
        }
    }
    out
}

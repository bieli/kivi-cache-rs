// One quantized document, many short questions.
//
// The document is stored once with KIVI. Each user session keeps only its
// question tokens in full precision and scores them together with the shared
// prefix.

use kivi_cache::{KiviCache, KiviConfig};

fn main() {
    let dim = 64;
    let heads = 4;
    let doc_len = 512;
    let question_len = 8;
    let users = 16;
    let config = KiviConfig::try_new(2, 32, 128).expect("config");
    let mut cache = KiviCache::new(config, 1, heads, dim).expect("cache");

    let n = heads * doc_len * dim;
    let keys: Vec<f32> = (0..n)
        .map(|i| ((i * 17) % 29) as f32 / 14.0 - 0.5)
        .collect();
    let values: Vec<f32> = (0..n)
        .map(|i| ((i * 11) % 23) as f32 / 12.0 - 0.3)
        .collect();
    cache.append(&keys, &values, doc_len).expect("document");

    let prefix = cache.share_prefix();
    let mut sessions: Vec<_> = (0..users).map(|_| prefix.session()).collect();
    for (user, session) in sessions.iter_mut().enumerate() {
        let qn = heads * question_len * dim;
        let qk = vec![0.05 * (user + 1) as f32; qn];
        let qv = vec![-0.02 * (user + 1) as f32; qn];
        session.append(&qk, &qv, question_len).expect("question");
    }

    let query = vec![0.1; heads * dim];
    let scale = 1.0 / (dim as f32).sqrt();
    let first = sessions[0].attention(&query, heads, scale).expect("attend");
    let second = sessions[1].attention(&query, heads, scale).expect("attend");
    let same_prefix = sessions[0]
        .prefix()
        .shares_allocation_with(sessions[1].prefix());

    let foot = prefix.footprint(users, question_len);
    let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
    println!("shared prefix RAG");
    println!("  document tokens: {}", foot.prefix_tokens);
    println!("  users: {users}, question tokens: {question_len}");
    println!("  one allocation: {same_prefix}");
    println!("  prefix: {:.3} MiB", mib(foot.prefix_bytes));
    println!("  all questions: {:.3} MiB", mib(foot.question_bytes));
    println!("  shared total: {:.3} MiB", mib(foot.shared_total_bytes));
    println!(
        "  fp16 if each user stored the document: {:.3} MiB",
        mib(foot.fp16_all_users_bytes)
    );
    println!("  savings vs per-user fp16: {:.2}x", foot.savings_vs_fp16());
    println!("  answers differ across questions: {}", first != second);
}

use anndists::dist::DistL2;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rayon::prelude::*;
use rust_diskann::{DiskANN, DiskAnnParams};
use std::{collections::HashSet, fs, time::Instant};

mod utils;
use utils::annhdf5::AnnBenchmarkData;

const R: usize = 64;
const BUILD_BEAM: usize = 128;
const SEARCH_BEAMS: [usize; 4] = [64, 128, 256, 512];
const K: usize = 10;
const DELETE_RATE: f64 = 0.001;
const SEED: u64 = 20_260_920;

fn params() -> DiskAnnParams {
    DiskAnnParams {
        max_degree: R,
        build_beam_width: BUILD_BEAM,
        alpha: 1.2,
        extra_seeds: 1,
    }
}

fn ground_truth(data: &AnnBenchmarkData, deleted: &HashSet<u32>) -> Vec<HashSet<u32>> {
    data.test_neighbours
        .rows()
        .into_iter()
        .map(|row| {
            let truth = row
                .iter()
                .map(|id| *id as u32)
                .filter(|id| !deleted.contains(id))
                .take(K)
                .collect::<HashSet<_>>();
            assert_eq!(truth.len(), K, "top-100 truth exhausted after deletion");
            truth
        })
        .collect()
}

fn evaluate(
    index: &DiskANN<f32, DistL2>,
    internal_to_original: &[u32],
    data: &AnnBenchmarkData,
    truth: &[HashSet<u32>],
    beam: usize,
) -> (f64, f64) {
    let started = Instant::now();
    let hits = data
        .test_data
        .par_iter()
        .enumerate()
        .map(|(query_id, query)| {
            index
                .search(query, K, beam)
                .into_iter()
                .filter(|id| truth[query_id].contains(&internal_to_original[*id as usize]))
                .count()
        })
        .sum::<usize>();
    let elapsed = started.elapsed().as_secs_f64();
    (
        hits as f64 / (data.test_data.len() * K) as f64,
        data.test_data.len() as f64 / elapsed,
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./sift-128-euclidean.hdf5".to_owned());
    let data = AnnBenchmarkData::new(dataset)?;
    let vectors = data
        .train_data
        .iter()
        .map(|vector| vector.0.clone())
        .collect::<Vec<_>>();
    let n = vectors.len();
    let delete_count = ((n as f64 * DELETE_RATE).round() as usize).max(1);
    println!(
        "SIFT delete benchmark: n={n}, delete={delete_count} ({:.4}%), R={R}, build_beam={BUILD_BEAM}",
        100.0 * delete_count as f64 / n as f64
    );

    let mut deletion_order = (0..n as u32).collect::<Vec<_>>();
    deletion_order.shuffle(&mut StdRng::seed_from_u64(SEED));
    let deleted_ids = deletion_order[..delete_count].to_vec();
    let deleted = deleted_ids.iter().copied().collect::<HashSet<_>>();
    let surviving_ids = (0..n as u32)
        .filter(|id| !deleted.contains(id))
        .collect::<Vec<_>>();
    let truth = ground_truth(&data, &deleted);

    for path in [
        "sift1m_before.db",
        "sift1m_update.work",
        "sift1m_updated.db",
        "sift1m_rebuilt.db",
    ] {
        let _ = fs::remove_file(path);
    }

    let started = Instant::now();
    let initial = DiskANN::build_index_with_params(&vectors, DistL2, "sift1m_before.db", params())?;
    let initial_build_seconds = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let mut update = DiskANN::begin_updates(&initial, n, 1.2, "sift1m_update.work")?;
    let begin_seconds = started.elapsed().as_secs_f64();
    drop(initial);
    fs::remove_file("sift1m_before.db")?;

    let started = Instant::now();
    let stats = update.delete_batch(&deleted_ids, 2 * R, 2)?;
    let delete_seconds = started.elapsed().as_secs_f64();
    assert_eq!(stats.len(), delete_count);
    let recovered: usize = stats.iter().map(|item| item.recovered_in_neighbors).sum();
    let candidates: usize = stats.iter().map(|item| item.repair_candidates).sum();

    let started = Instant::now();
    let (updated, old_to_new) = update.commit_updates_to_static("sift1m_updated.db")?;
    let commit_seconds = started.elapsed().as_secs_f64();
    let mut updated_map = vec![u32::MAX; updated.num_vectors];
    for (old_id, new_id) in old_to_new.into_iter().enumerate() {
        if let Some(new_id) = new_id {
            updated_map[new_id as usize] = old_id as u32;
        }
    }
    assert!(updated_map.iter().all(|id| *id != u32::MAX));

    let surviving_vectors = surviving_ids
        .iter()
        .map(|id| vectors[*id as usize].clone())
        .collect::<Vec<_>>();
    let started = Instant::now();
    let rebuilt = DiskANN::build_index_with_params(
        &surviving_vectors,
        DistL2,
        "sift1m_rebuilt.db",
        params(),
    )?;
    let rebuild_seconds = started.elapsed().as_secs_f64();

    println!("initial build       : {initial_build_seconds:.3} s");
    println!("begin update session: {begin_seconds:.3} s");
    println!("MERIT delete        : {delete_seconds:.3} s");
    println!(
        "repair averages     : {:.2} recovered in-neighbors, {:.2} candidates/delete",
        recovered as f64 / delete_count as f64,
        candidates as f64 / delete_count as f64
    );
    println!("static commit       : {commit_seconds:.3} s");
    println!(
        "update total        : {:.3} s",
        begin_seconds + delete_seconds + commit_seconds
    );
    println!("fresh rebuild       : {rebuild_seconds:.3} s");
    println!(
        "speedup             : {:.2}x",
        rebuild_seconds / (begin_seconds + delete_seconds + commit_seconds)
    );

    println!("beam,rebuilt_recall,updated_recall,delta,rebuilt_qps,updated_qps");
    for beam in SEARCH_BEAMS {
        let (rebuilt_recall, rebuilt_qps) = evaluate(&rebuilt, &surviving_ids, &data, &truth, beam);
        let (updated_recall, updated_qps) = evaluate(&updated, &updated_map, &data, &truth, beam);
        println!(
            "{beam},{rebuilt_recall:.8},{updated_recall:.8},{:.8},{rebuilt_qps:.2},{updated_qps:.2}",
            updated_recall - rebuilt_recall
        );
    }
    drop(updated);
    drop(rebuilt);
    fs::remove_file("sift1m_updated.db")?;
    fs::remove_file("sift1m_rebuilt.db")?;
    Ok(())
}

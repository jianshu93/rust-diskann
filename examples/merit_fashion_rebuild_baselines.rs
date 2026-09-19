use anndists::dist::DistL2;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rayon::prelude::*;
use rust_diskann::{DiskANN, DiskAnnParams, mmap_dynamic::MmapDynamicDiskANN};
use std::{collections::HashSet, fs, time::Instant};

mod utils;
use utils::annhdf5::AnnBenchmarkData;

const R: usize = 48;
const BUILD_L: usize = 128;
const SEARCH_BEAMS: [usize; 4] = [32, 64, 128, 256];
const K: usize = 10;
const ROUNDS: usize = 5;
const BATCH: usize = 3_000;
const INITIAL_SMALL: usize = 45_000;

fn truths(data: &AnnBenchmarkData, active: &[bool]) -> Vec<HashSet<u32>> {
    data.test_neighbours
        .rows()
        .into_iter()
        .map(|row| {
            let truth = row
                .iter()
                .map(|x| *x as u32)
                .filter(|id| active[*id as usize])
                .take(K)
                .collect::<HashSet<_>>();
            assert_eq!(
                truth.len(),
                K,
                "official top-100 exhausted by active-set filtering"
            );
            truth
        })
        .collect()
}

fn evaluate_static(
    index: &DiskANN<f32, DistL2>,
    id_map: &[u32],
    data: &AnnBenchmarkData,
    truth: &[HashSet<u32>],
    beam: usize,
) -> (f64, f64) {
    let t = Instant::now();
    let hits = data
        .test_data
        .par_iter()
        .enumerate()
        .map(|(i, q)| {
            index
                .search(q, K, beam)
                .into_iter()
                .filter(|id| truth[i].contains(&id_map[*id as usize]))
                .count()
        })
        .sum::<usize>();
    (
        hits as f64 / (data.test_data.len() * K) as f64,
        data.test_data.len() as f64 / t.elapsed().as_secs_f64(),
    )
}

fn evaluate_dynamic(
    index: &MmapDynamicDiskANN<f32, DistL2>,
    id_map: &[Option<u32>],
    data: &AnnBenchmarkData,
    truth: &[HashSet<u32>],
    beam: usize,
) -> (f64, f64) {
    let t = Instant::now();
    let hits = data
        .test_data
        .par_iter()
        .enumerate()
        .map(|(i, q)| {
            index
                .search(q, K, beam)
                .into_iter()
                .filter(|id| id_map[*id as usize].is_some_and(|orig| truth[i].contains(&orig)))
                .count()
        })
        .sum::<usize>();
    (
        hits as f64 / (data.test_data.len() * K) as f64,
        data.test_data.len() as f64 / t.elapsed().as_secs_f64(),
    )
}

fn build_static(
    vectors: &[Vec<f32>],
    original_ids: &[u32],
    tag: &str,
) -> Result<(DiskANN<f32, DistL2>, f64, String), Box<dyn std::error::Error>> {
    let path = format!("rebuild_{tag}.db");
    let _ = fs::remove_file(&path);
    let selected = original_ids
        .iter()
        .map(|id| vectors[*id as usize].clone())
        .collect::<Vec<_>>();
    let t = Instant::now();
    let index = DiskANN::build_index_with_params(
        &selected,
        DistL2,
        &path,
        DiskAnnParams {
            max_degree: R,
            build_beam_width: BUILD_L,
            alpha: 1.2,
            extra_seeds: 1,
        },
    )?;
    Ok((index, t.elapsed().as_secs_f64(), path))
}

#[allow(clippy::too_many_arguments)]
fn record(
    scenario: &str,
    round: usize,
    active: &[bool],
    dynamic: &MmapDynamicDiskANN<f32, DistL2>,
    dynamic_map: &[Option<u32>],
    vectors: &[Vec<f32>],
    data: &AnnBenchmarkData,
    update_seconds: f64,
    csv: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    let ids = active
        .iter()
        .enumerate()
        .filter_map(|(id, on)| on.then_some(id as u32))
        .collect::<Vec<_>>();
    let truth = truths(data, active);
    let (static_index, rebuild_seconds, path) =
        build_static(vectors, &ids, &format!("{scenario}_{round}"))?;
    for beam in SEARCH_BEAMS {
        let (sr, sq) = evaluate_static(&static_index, &ids, data, &truth, beam);
        let (dr, dq) = evaluate_dynamic(dynamic, dynamic_map, data, &truth, beam);
        csv.push_str(&format!("{scenario},{round},{},{beam},{sr:.8},{dr:.8},{:.8},{sq:.3},{dq:.3},{rebuild_seconds:.3},{update_seconds:.3}\n",ids.len(),dr-sr));
        println!(
            "{scenario} round={round} n={} ef={beam}: static={sr:.6} dynamic={dr:.6} delta={:.6}",
            ids.len(),
            dr - sr
        );
    }
    drop(static_index);
    fs::remove_file(path)?;
    Ok(())
}

fn run_delete(
    vectors: &[Vec<f32>],
    data: &AnnBenchmarkData,
    csv: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    let ids = (0..vectors.len() as u32).collect::<Vec<_>>();
    let (base, _, base_path) = build_static(vectors, &ids, "delete_initial")?;
    let path = "fashion_delete_dynamic.db";
    let _ = fs::remove_file(path);
    let mut dynamic = MmapDynamicDiskANN::create_from_static(&base, vectors.len(), 1.2, path)?;
    drop(base);
    fs::remove_file(base_path)?;
    let mut active = vec![true; vectors.len()];
    let mut map = (0..vectors.len() as u32).map(Some).collect::<Vec<_>>();
    let mut order = (0..vectors.len() as u32).collect::<Vec<_>>();
    order.shuffle(&mut StdRng::seed_from_u64(7001));
    for round in 1..=ROUNDS {
        let batch = &order[(round - 1) * BATCH..round * BATCH];
        let t = Instant::now();
        assert_eq!(dynamic.delete_batch(batch, 2 * R, 2).len(), BATCH);
        let update = t.elapsed().as_secs_f64();
        for &id in batch {
            active[id as usize] = false;
            map[id as usize] = None;
        }
        record(
            "delete_only",
            round,
            &active,
            &dynamic,
            &map,
            vectors,
            data,
            update,
            csv,
        )?;
    }
    fs::remove_file(path)?;
    Ok(())
}

fn run_insert(
    vectors: &[Vec<f32>],
    data: &AnnBenchmarkData,
    csv: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    let ids = (0..INITIAL_SMALL as u32).collect::<Vec<_>>();
    let (base, _, base_path) = build_static(vectors, &ids, "insert_initial")?;
    let path = "fashion_insert_dynamic.db";
    let _ = fs::remove_file(path);
    let mut dynamic = MmapDynamicDiskANN::create_from_static(&base, vectors.len(), 1.2, path)?;
    drop(base);
    fs::remove_file(base_path)?;
    let mut active = (0..vectors.len())
        .map(|id| id < INITIAL_SMALL)
        .collect::<Vec<_>>();
    let mut map = (0..vectors.len())
        .map(|id| (id < INITIAL_SMALL).then_some(id as u32))
        .collect::<Vec<_>>();
    for round in 1..=ROUNDS {
        let start = INITIAL_SMALL + (round - 1) * BATCH;
        let originals = (start..start + BATCH)
            .map(|id| id as u32)
            .collect::<Vec<_>>();
        let replacement = originals
            .iter()
            .map(|id| vectors[*id as usize].clone())
            .collect();
        let t = Instant::now();
        let slots = dynamic.insert_batch(replacement, BUILD_L)?;
        let update = t.elapsed().as_secs_f64();
        for (slot, orig) in slots.into_iter().zip(originals) {
            active[orig as usize] = true;
            map[slot as usize] = Some(orig);
        }
        record(
            "insert_only",
            round,
            &active,
            &dynamic,
            &map,
            vectors,
            data,
            update,
            csv,
        )?;
    }
    fs::remove_file(path)?;
    Ok(())
}

fn run_mixed(
    vectors: &[Vec<f32>],
    data: &AnnBenchmarkData,
    csv: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    let ids = (0..INITIAL_SMALL as u32).collect::<Vec<_>>();
    let (base, _, base_path) = build_static(vectors, &ids, "mixed_initial")?;
    let path = "fashion_mixed_dynamic.db";
    let _ = fs::remove_file(path);
    let mut dynamic = MmapDynamicDiskANN::create_from_static(&base, vectors.len(), 1.2, path)?;
    drop(base);
    fs::remove_file(base_path)?;
    let mut active = (0..vectors.len())
        .map(|id| id < INITIAL_SMALL)
        .collect::<Vec<_>>();
    let mut map = (0..vectors.len())
        .map(|id| (id < INITIAL_SMALL).then_some(id as u32))
        .collect::<Vec<_>>();
    let mut slots = (0..INITIAL_SMALL as u32).collect::<Vec<_>>();
    slots.shuffle(&mut StdRng::seed_from_u64(9001));
    for round in 1..=ROUNDS {
        let deleted = slots[(round - 1) * BATCH..round * BATCH]
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let start = INITIAL_SMALL + (round - 1) * BATCH;
        let originals = (start..start + BATCH)
            .map(|id| id as u32)
            .collect::<Vec<_>>();
        let replacement = originals
            .iter()
            .map(|id| vectors[*id as usize].clone())
            .collect();
        let t = Instant::now();
        assert_eq!(dynamic.delete_batch(&deleted, 2 * R, 2).len(), BATCH);
        for &slot in &deleted {
            active[map[slot as usize].take().unwrap() as usize] = false;
        }
        let inserted = dynamic.insert_batch(replacement, BUILD_L)?;
        for (slot, orig) in inserted.into_iter().zip(originals) {
            active[orig as usize] = true;
            map[slot as usize] = Some(orig);
        }
        let update = t.elapsed().as_secs_f64();
        record(
            "mixed", round, &active, &dynamic, &map, vectors, data, update, csv,
        )?;
    }
    fs::remove_file(path)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/fashion-mnist-784-euclidean.hdf5".into());
    let data = AnnBenchmarkData::new(path)?;
    let vectors = data
        .train_data
        .iter()
        .map(|x| x.0.clone())
        .collect::<Vec<_>>();
    let mut csv = String::from(
        "scenario,round,active_n,search_beam,static_recall,dynamic_recall,recall_delta,static_qps,dynamic_qps,rebuild_seconds,update_seconds\n",
    );
    run_delete(&vectors, &data, &mut csv)?;
    run_insert(&vectors, &data, &mut csv)?;
    run_mixed(&vectors, &data, &mut csv)?;
    fs::write("merit_fashion_rebuild_baselines.csv", csv)?;
    Ok(())
}

use anndists::dist::DistL2;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rayon::prelude::*;
use rust_diskann::{DiskANN, DiskAnnParams};
use std::{collections::HashSet, fs, time::Instant};
mod utils;
use utils::annhdf5::AnnBenchmarkData;

const R: usize = 48;
const L: usize = 128;
const BEAMS: [usize; 4] = [32, 64, 128, 256];
const K: usize = 10;
const ROUNDS: usize = 5;
const DEFAULT_BATCH: usize = 3_000;
const INITIAL: usize = 45_000;

fn build(
    v: &[Vec<f32>],
    ids: &[u32],
    path: &str,
) -> Result<DiskANN<f32, DistL2>, Box<dyn std::error::Error>> {
    let selected = ids
        .iter()
        .map(|id| v[*id as usize].clone())
        .collect::<Vec<_>>();
    Ok(DiskANN::build_index_with_params(
        &selected,
        DistL2,
        path,
        DiskAnnParams {
            max_degree: R,
            build_beam_width: L,
            alpha: 1.2,
            extra_seeds: 1,
        },
    )?)
}

fn truth(data: &AnnBenchmarkData, active: &[bool]) -> Vec<HashSet<u32>> {
    data.test_neighbours
        .rows()
        .into_iter()
        .map(|row| {
            let t = row
                .iter()
                .map(|x| *x as u32)
                .filter(|id| active[*id as usize])
                .take(K)
                .collect::<HashSet<_>>();
            assert_eq!(t.len(), K);
            t
        })
        .collect()
}

fn eval(
    index: &DiskANN<f32, DistL2>,
    map: &[u32],
    data: &AnnBenchmarkData,
    truth: &[HashSet<u32>],
    beam: usize,
) -> f64 {
    let hits: usize = data
        .test_data
        .par_iter()
        .enumerate()
        .map(|(i, q)| {
            index
                .search(q, K, beam)
                .into_iter()
                .filter(|id| truth[i].contains(&map[*id as usize]))
                .count()
        })
        .sum();
    hits as f64 / (data.test_data.len() * K) as f64
}

fn remap(work: &[Option<u32>], slots: &[Option<u32>], n: usize) -> Vec<u32> {
    let mut out = vec![u32::MAX; n];
    for (old, original) in work.iter().enumerate() {
        if let (Some(original), Some(new)) = (original, slots[old]) {
            out[new as usize] = *original;
        }
    }
    assert!(out.iter().all(|x| *x != u32::MAX));
    out
}

fn slots(map: &[u32], originals: &[u32], universe: usize) -> Vec<u32> {
    let mut inverse = vec![u32::MAX; universe];
    for (slot, original) in map.iter().enumerate() {
        inverse[*original as usize] = slot as u32;
    }
    originals
        .iter()
        .map(|id| {
            assert_ne!(inverse[*id as usize], u32::MAX);
            inverse[*id as usize]
        })
        .collect()
}

fn compare(
    scenario: &str,
    round: usize,
    index: &DiskANN<f32, DistL2>,
    map: &[u32],
    active: &[bool],
    vectors: &[Vec<f32>],
    data: &AnnBenchmarkData,
    update_s: f64,
    csv: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    let ids = active
        .iter()
        .enumerate()
        .filter_map(|(i, on)| on.then_some(i as u32))
        .collect::<Vec<_>>();
    let path = format!("{scenario}_{round}_fresh.db");
    let started = Instant::now();
    let fresh = build(vectors, &ids, &path)?;
    let rebuild_s = started.elapsed().as_secs_f64();
    let t = truth(data, active);
    for beam in BEAMS {
        let baseline = eval(&fresh, &ids, data, &t, beam);
        let updated = eval(index, map, data, &t, beam);
        println!(
            "{scenario} round={round} n={} beam={beam}: rebuild={baseline:.6} committed={updated:.6} delta={:.6}",
            ids.len(),
            updated - baseline
        );
        csv.push_str(&format!("{scenario},{round},{},{beam},{baseline:.8},{updated:.8},{:.8},{rebuild_s:.3},{update_s:.3}\n", ids.len(), updated-baseline));
    }
    drop(fresh);
    fs::remove_file(path)?;
    Ok(())
}

fn run(
    scenario: &str,
    vectors: &[Vec<f32>],
    data: &AnnBenchmarkData,
    batch: usize,
    csv: &mut String,
) -> Result<(), Box<dyn std::error::Error>> {
    let insert_only = scenario == "insert_only";
    let initial_n = if insert_only || scenario == "mixed" {
        INITIAL
    } else {
        vectors.len()
    };
    let initial = (0..initial_n as u32).collect::<Vec<_>>();
    let index_path = format!("{scenario}_current.db");
    let mut index = build(vectors, &initial, &index_path)?;
    let mut map = initial.clone();
    let mut active = (0..vectors.len())
        .map(|i| i < initial_n)
        .collect::<Vec<_>>();
    let mut order = initial;
    order.shuffle(&mut StdRng::seed_from_u64(if scenario == "delete_only" {
        7001
    } else {
        9001
    }));
    for round in 1..=ROUNDS {
        let deleting = scenario != "insert_only";
        let inserting = scenario != "delete_only";
        let deleted_orig = if deleting {
            order[(round - 1) * batch..round * batch].to_vec()
        } else {
            vec![]
        };
        let deleted_slots = slots(&map, &deleted_orig, vectors.len());
        let start = INITIAL + (round - 1) * batch;
        let inserted_orig = if inserting {
            (start..start + batch).map(|x| x as u32).collect::<Vec<_>>()
        } else {
            vec![]
        };
        let capacity = index.num_vectors + if insert_only { batch } else { 0 };
        let mut work_map = map.iter().copied().map(Some).collect::<Vec<_>>();
        work_map.resize(capacity, None);
        let work_path = format!("{scenario}_{round}.work");
        let started = Instant::now();
        let mut update = DiskANN::begin_updates(&index, capacity, 1.2, &work_path)?;
        if deleting {
            update.delete_batch(&deleted_slots, 2 * R, 2)?;
            for (slot, orig) in deleted_slots.iter().zip(&deleted_orig) {
                work_map[*slot as usize] = None;
                active[*orig as usize] = false;
            }
        }
        if inserting {
            let vv = inserted_orig
                .iter()
                .map(|id| vectors[*id as usize].clone())
                .collect();
            let ss = update.insert_batch(vv, L)?;
            for (slot, orig) in ss.iter().zip(&inserted_orig) {
                work_map[*slot as usize] = Some(*orig);
                active[*orig as usize] = true;
            }
        }
        drop(index);
        let (next, old_new) = update.commit_updates_to_static(&index_path)?;
        index = next;
        map = remap(&work_map, &old_new, index.num_vectors);
        compare(
            scenario,
            round,
            &index,
            &map,
            &active,
            vectors,
            data,
            started.elapsed().as_secs_f64(),
            csv,
        )?;
    }
    drop(index);
    fs::remove_file(index_path)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/fashion-mnist-784-euclidean.hdf5".into());
    let batch = std::env::args()
        .nth(2)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_BATCH);
    let data = AnnBenchmarkData::new(path)?;
    let vectors = data
        .train_data
        .iter()
        .map(|x| x.0.clone())
        .collect::<Vec<_>>();
    let mut csv = String::from(
        "scenario,round,active_n,beam,rebuild_recall,committed_recall,delta,rebuild_seconds,update_commit_seconds\n",
    );
    for scenario in ["delete_only", "insert_only", "mixed"] {
        run(scenario, &vectors, &data, batch, &mut csv)?;
    }
    fs::write("merit_fashion_rebuild_baselines.csv", csv)?;
    Ok(())
}

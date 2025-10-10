// examples/hannoy_sift.rs
#![allow(clippy::needless_range_loop)]

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};


mod utils;
use utils::*;
use annhdf5::AnnBenchmarkData;
use cpu_time::ProcessTime;
use heed::{Env, EnvOpenOptions, RwTxn};
use hannoy::{distances::Euclidean, Database, Reader, Writer};
use ndarray::s;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

/// SIFT-1M (L2) dimensions
const DIM: usize = 128;

// Build/search knobs for hannoy
const M: usize = 48;
const M0: usize = 48;
const EF_CONSTRUCTION: usize = 256;
const EF_SEARCH: usize = 64;

// LMDB map size (adjust upward if you raise M0/ef_c)
const MAP_SIZE_BYTES: usize = 8 * 1024 * 1024 * 1024; // 8 GiB

fn ensure_dir(path: &Path) {
    if !path.exists() {
        fs::create_dir_all(path).expect("failed to create LMDB dir");
    }
}

fn open_env(dir: &Path, map_size_bytes: usize) -> Env {
    unsafe { EnvOpenOptions::new().map_size(map_size_bytes).open(dir) }.expect("open LMDB env")
}

fn slice_to_array<const N: usize>(v: &[f32]) -> [f32; N] {
    assert_eq!(v.len(), N, "vector dim {} != {}", v.len(), N);
    let mut arr = [0.0f32; N];
    arr.copy_from_slice(&v[..N]);
    arr
}

/// Simple L2 helper for recall stats (true Euclidean).
fn euclid(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for j in 0..a.len() {
        let d = a[j] - b[j];
        s += d * d;
    }
    s.sqrt()
}

/// Create the DB and insert all training vectors (on-disk).
fn create_db_and_fill_with_train<'a>(
    env: &'a Env,
    train: &'a [(Vec<f32>, usize)],
) -> hannoy::Result<(Writer<Euclidean>, RwTxn<'a>, Database<Euclidean>)> {
    let mut wtxn = env.write_txn().unwrap();
    let db: Database<Euclidean> = env.create_database(&mut wtxn, None)?;
    let writer: Writer<Euclidean> = Writer::new(db, 0, DIM);

    // Insert all training vectors; ids are 0..n-1
    for (id, (vec, _orig)) in train.iter().enumerate() {
        let arr = slice_to_array::<DIM>(vec);
        writer.add_item(&mut wtxn, id as u32, &arr)?;
    }

    Ok((writer, wtxn, db))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load SIFT-1M (Euclidean)
    // wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
    let fname = "./sift-128-euclidean.hdf5";
    println!("\n\nhannoy × SIFT1M @ L2  -> {:?}", fname);

    let mut anndata =
        AnnBenchmarkData::new(fname.to_string()).expect("Failed to load SIFT1M HDF5 file");

    let knbn_max = anndata.test_distances.dim().1;
    let nb_elem = anndata.train_data.len();
    let nb_search = anndata.test_data.len();

    println!("Train size : {}", nb_elem);
    println!("Test size  : {}", nb_search);
    println!("Ground-truth k per query in file: {}", knbn_max);
    // Open LMDB & (re)build on disk

    let lmdb_dir = PathBuf::from("./hannoy_sift1m_env");
    ensure_dir(&lmdb_dir);
    let env = open_env(&lmdb_dir, MAP_SIZE_BYTES);

    // If you're profiling hot-cache search, you can prefetch:
    // std::env::set_var("HANNOY_READER_PREFETCH_MEMORY", format!("{}", 1_usize << 30));

    let reader = {
        let rtxn = env.read_txn().unwrap();

        // Help heed infer codecs via explicit type
        let db_opt: Option<Database<Euclidean>> = env.open_database(&rtxn, None)?;
        match db_opt {
            Some(db) => match Reader::<Euclidean>::open(&rtxn, 0, db) {
                Ok(reader) => {
                    println!("Found existing hannoy index on disk; using it.");
                    reader
                }
                Err(hannoy::Error::NeedBuild(_)) | Err(hannoy::Error::MissingMetadata(_)) => {
                    drop(rtxn);
                    println!("Database present but needs build; building now…");
                    build_hannoy(&env, &mut anndata)?
                }
                Err(e) => {
                    drop(rtxn);
                    eprintln!("Reader::open failed ({e:?}), rebuilding…");
                    build_hannoy(&env, &mut anndata)?
                }
            },
            None => {
                drop(rtxn);
                println!("No database found; building hannoy on disk…");
                build_hannoy(&env, &mut anndata)?
            }
        }
    };

    // Free training vectors after build/open
    anndata.train_data.clear();
    anndata.train_data.shrink_to_fit();


    // Search; compute recall@k correctly
    //    - ID-overlap recall (robust)
    //    - distance recall using TRUE L2 (recomputed)
    let k = 10.min(knbn_max);
    let ef_search = EF_SEARCH;

    println!(
        "\nSearching {} queries with k={}, ef_search={} …",
        nb_search, k, ef_search
    );

    let start_cpu = ProcessTime::now();
    let start_wall = SystemTime::now();

    // Parallel search: each worker opens its own read transaction.
    // We keep (ids, dists_true_l2) per query.
    let results: Vec<(Vec<u32>, Vec<f32>)> = (0..nb_search)
        .into_par_iter()
        .map(|i| {
            let rtxn = env.read_txn().unwrap();
            let q = &anndata.test_data[i];

            // Top-k from hannoy (distance may be squared-L2 internally)
            let nns = reader
                .nns(k)
                .ef_search(ef_search)
                .by_vector(&rtxn, q)
                .expect("hannoy search");

            let ids: Vec<u32> = nns.iter().map(|(id, _)| *id).collect();

            // Recompute TRUE L2 for these ids (consistent with HDF5 GT distances)
            let mut l2s: Vec<f32> = Vec::with_capacity(ids.len());
            for &id in &ids {
                let v = reader.item_vector(&rtxn, id).unwrap().unwrap();
                l2s.push(euclid(q, &v));
            }
            // Keep result sorted by true L2 so it’s directly comparable to GT
            let mut paired: Vec<(u32, f32)> = ids.iter().copied().zip(l2s.iter().copied()).collect();
            paired.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let (ids_sorted, dists_sorted): (Vec<u32>, Vec<f32>) =
                paired.into_iter().unzip();

            (ids_sorted, dists_sorted)
        })
        .collect();

    let cpu_time = start_cpu.elapsed();
    let wall_time = start_wall.elapsed().unwrap();

    // ID-overlap recall@k
    let mut id_recalls: Vec<usize> = Vec::with_capacity(nb_search);

    // b) distance-based recall using TRUE L2
    let mut dist_recalls: Vec<usize> = Vec::with_capacity(nb_search);
    let mut last_distances_ratio: Vec<f32> = Vec::with_capacity(nb_search);

    for i in 0..nb_search {
        // Ground-truth
        let gt_ids_row = anndata.test_neighbours.row(i);
        let gt_dists_row = anndata.test_distances.row(i);

        let true_k = k.min(gt_ids_row.len());

        // FIX: ndarray slicing must use the `s![]` macro
        let gt_topk_ids: HashSet<usize> = gt_ids_row
            .slice(s![..true_k])
            .iter()
            .map(|&x| x as usize)
            .collect();
        let gt_kth_l2 = gt_dists_row[true_k - 1];

        // From our result (already sorted by true L2)
        let (ref ids, ref dists) = results[i];

        // ID recall
        let hits = ids
            .iter()
            .take(true_k)
            .filter(|&&id| gt_topk_ids.contains(&(id as usize)))
            .count();
        id_recalls.push(hits);

        // distance-based recall (using true L2, consistent with GT)
        let dist_hits = dists.iter().take(true_k).filter(|&&d| d <= gt_kth_l2).count();
        dist_recalls.push(dist_hits);

        // last distance ratio (ours / GT kth)
        let ratio = if !dists.is_empty() {
            dists[dists.len() - 1] / gt_kth_l2
        } else {
            0.0
        };
        last_distances_ratio.push(ratio);
    }

    let mean_id_recall = (id_recalls.iter().sum::<usize>() as f32) / ((k * id_recalls.len()) as f32);
    let mean_dist_recall =
        (dist_recalls.iter().sum::<usize>() as f32) / ((k * dist_recalls.len()) as f32);
    let mean_last_ratio =
        last_distances_ratio.iter().sum::<f32>() / (last_distances_ratio.len() as f32);

    let search_sys_time_us = wall_time.as_micros() as f32;
    let req_per_s = (nb_search as f32) * 1.0e6_f32 / search_sys_time_us;

    //println!("\n id-overlap recall@{}: {:.4}", k, mean_id_recall);
    println!(" distance recall@{}  : {:.4}", k, mean_dist_recall);
    println!(" last distances ratio (ours true L2 / GT kth): {:.4}", mean_last_ratio);
    println!(
        " throughput: {:.0} q/s — cpu: {:?}  wall: {:?}",
        req_per_s, cpu_time, wall_time
    );

    Ok(())
}

fn build_hannoy(
    env: &Env,
    anndata: &mut AnnBenchmarkData,
) -> Result<Reader<Euclidean>, Box<dyn std::error::Error>> {
    println!(
        "Building hannoy index: N={}, dim={}, M={}, M0={}, ef_c={}",
        anndata.train_data.len(),
        DIM,
        M,
        M0,
        EF_CONSTRUCTION
    );

    let (writer, mut wtxn, db) = create_db_and_fill_with_train(env, &anndata.train_data)?;

    let start_cpu = ProcessTime::now();
    let start_wall = SystemTime::now();

    let mut rng = StdRng::seed_from_u64(42);
    let mut builder = writer.builder::<StdRng>(&mut rng);
    builder.ef_construction(EF_CONSTRUCTION).build::<M, M0>(&mut wtxn)?;
    wtxn.commit().unwrap();

    let cpu_time: Duration = start_cpu.elapsed();
    let wall_time = start_wall.elapsed().unwrap();
    println!(
        "hannoy build complete. CPU time: {:?}, wall time: {:?}",
        cpu_time, wall_time
    );

    // Open a Reader on the freshly built DB
    let rtxn = env.read_txn().unwrap();
    let reader = Reader::<Euclidean>::open(&rtxn, 0, db)?;
    Ok(reader)
}
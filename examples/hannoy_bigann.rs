//! hannoy_bigann.rs — Hannoy on BigANN with distance-based recall@k (parallel or single-threaded)
//
// Expected files in repo root (10M GT subset from ANN_SIFT1B):
//   - bigann_base.bvecs
//   - bigann_query.bvecs
//   - idx_10M.ivecs
//   - dis_10M.fvecs   (squared L2; we compare using squared distances)
// Builds (or reuses) an LMDB-backed hannoy index in ./hannoy_bigann.lmdb,
// then evaluates recall@10 and recall@100 on NB_QUERY queries.

use byteorder::{LittleEndian, ReadBytesExt};
use heed::{Env, EnvOpenOptions};
use hannoy::{distances::Euclidean, Database, Error as HannoyError, Reader, Writer};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ------------------------ Config ------------------------

const DIM: usize = 128;

// How many base vectors to ingest from bigann_base.bvecs (use your shard size).
const NB_DATA_POINTS: usize = 10_000_000;

// Number of queries to evaluate.
const NB_QUERY: usize = 10_000;

// LMDB directory for hannoy
const LMDB_DIR: &str = "hannoy_bigann.lmdb";

// LMDB map size (bytes). Plenty of headroom for vectors+links; override via HANNOY_MAP_SIZE_GB.
const DEFAULT_MAP_SIZE_BYTES: usize = 64 * 1024 * 1024 * 1024; // 64 GiB

// Hannoy build/search knobs (align with your SIFT runs)
const M: usize = 16;
const M0: usize = 32;
const EF_CONSTRUCTION: usize = 160;
// Search knob (analogous to DiskANN beam width)
const EF_SEARCH: usize = 512;

fn read_bvecs_block<const SIZE: usize>(
    r: &mut BufReader<File>,
    max_points: usize,
) -> io::Result<Vec<Vec<u8>>> {
    let mut out = Vec::with_capacity(max_points);
    let mut buf = [0u8; SIZE];
    for _ in 0..max_points {
        let dim = match r.read_u32::<LittleEndian>() {
            Ok(v) => v as usize,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        };
        if dim != SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bvecs dim {} != {}", dim, SIZE),
            ));
        }
        r.read_exact(&mut buf)?;
        out.push(buf.to_vec());
    }
    Ok(out)
}

fn read_all_bvecs_prefix<const SIZE: usize>(
    path: &str,
    n_points: usize,
    block: usize,
) -> io::Result<Vec<Vec<u8>>> {
    let f = OpenOptions::new().read(true).open(path)?;
    let mut br = BufReader::new(f);
    let mut all = Vec::with_capacity(n_points.min(1_000_000));
    let mut read_total = 0usize;
    while read_total < n_points {
        let want = block.min(n_points - read_total);
        let mut chunk = read_bvecs_block::<SIZE>(&mut br, want)?;
        if chunk.is_empty() {
            break;
        }
        read_total += chunk.len();
        all.append(&mut chunk);
    }
    Ok(all)
}

fn read_query_bvecs<const SIZE: usize>(path: &str, n_queries: usize) -> io::Result<Vec<Vec<u8>>> {
    let f = OpenOptions::new().read(true).open(path)?;
    let mut br = BufReader::new(f);
    let mut out = Vec::with_capacity(n_queries);
    let mut buf = [0u8; SIZE];
    for _ in 0..n_queries {
        let dim = br.read_u32::<LittleEndian>()? as usize;
        if dim != SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("query bvecs dim {} != {}", dim, SIZE),
            ));
        }
        br.read_exact(&mut buf)?;
        out.push(buf.to_vec());
    }
    Ok(out)
}

fn read_f32_block(r: &mut BufReader<File>) -> io::Result<Vec<f32>> {
    let dim = r.read_u32::<LittleEndian>()? as usize;
    let mut v = vec![0f32; dim];
    for x in &mut v {
        *x = r.read_f32::<LittleEndian>()?;
    }
    Ok(v)
}

fn read_u32_block(r: &mut BufReader<File>) -> io::Result<Vec<u32>> {
    let dim = r.read_u32::<LittleEndian>()? as usize;
    let mut v = vec![0u32; dim];
    for x in &mut v {
        *x = r.read_u32::<LittleEndian>()?;
    }
    Ok(v)
}

/// Ground truth: returns (ids, squared L2 distances), each as Vec[query] -> Vec[..]
fn read_ground_truth(
    i_path: &str,
    f_path: &str,
    n_queries: usize,
) -> io::Result<(Vec<Vec<u32>>, Vec<Vec<f32>>)> {
    let fi = OpenOptions::new().read(true).open(i_path)?;
    let ff = OpenOptions::new().read(true).open(f_path)?;
    let mut ri = BufReader::new(fi);
    let mut rf = BufReader::new(ff);

    let mut ids = Vec::with_capacity(n_queries);
    let mut dists2 = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        ids.push(read_u32_block(&mut ri)?);
        dists2.push(read_f32_block(&mut rf)?);
    }
    Ok((ids, dists2))
}

#[inline]
fn u8s_to_f32(v: &[u8]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

fn ensure_dir(path: &Path) -> io::Result<()> {
    if !path.exists() {
        fs::create_dir_all(path)?;
    }
    Ok(())
}

fn open_env(dir: &Path) -> heed::Result<Env> {
    let map_size_bytes = std::env::var("HANNOY_MAP_SIZE_GB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|gb| gb * 1024 * 1024 * 1024)
        .unwrap_or(DEFAULT_MAP_SIZE_BYTES);

    unsafe {
        EnvOpenOptions::new()
            .map_size(map_size_bytes)
            .max_dbs(8)
            .open(dir)
    }
}

/// Try to open a Reader; if NeedBuild/MissingMetadata, return Ok(None).
fn try_open_reader(env: &Env) -> Result<Option<(Reader<Euclidean>, Database<Euclidean>)>, Box<dyn Error>> {
    let mut wtxn = env.write_txn()?;
    let db: Database<Euclidean> = env.create_database(&mut wtxn, None)?;
    wtxn.commit()?;

    let rtxn = env.read_txn()?;
    match Reader::<Euclidean>::open(&rtxn, 0, db.clone()) {
        Ok(reader) => {
            drop(rtxn);
            Ok(Some((reader, db)))
        }
        Err(HannoyError::NeedBuild(_)) | Err(HannoyError::MissingMetadata(_)) => {
            drop(rtxn);
            Ok(None)
        }
        Err(e) => Err(e.into()),
    }
}

/// Build the index from vectors, then reopen a Reader.
fn build_and_open_reader(
    env: &Env,
    db: &Database<Euclidean>,
    dim: usize,
    vectors: &[Vec<f32>],
) -> Result<Reader<Euclidean>, Box<dyn Error>> {
    let mut wtxn = env.write_txn()?;
    let writer: Writer<Euclidean> = Writer::new(db.clone(), 0, dim);

    for (id, vecf) in vectors.iter().enumerate() {
        writer.add_item(&mut wtxn, id as u32, vecf)?;
    }

    let mut rng = StdRng::seed_from_u64(42);
    let mut builder = writer.builder(&mut rng);
    builder.ef_construction(EF_CONSTRUCTION).build::<M, M0>(&mut wtxn)?;
    wtxn.commit()?;

    let rtxn = env.read_txn()?;
    let reader = Reader::<Euclidean>::open(&rtxn, 0, db.clone())?;
    drop(rtxn);
    Ok(reader)
}

fn eval_distance_recall_parallel(
    env: &Env,
    reader: &Reader<Euclidean>,
    queries_f32: &[Vec<f32>],
    gt_d2: &[Vec<f32>], // squared L2; we compare in squared space
    k: usize,
    ef_search: usize,
) {
    let t0 = Instant::now();

    let (correct, last_ratio_sum, frac_returned_sum) = queries_f32
        .par_iter()
        .enumerate()
        .map(|(qi, q)| {
            // Open a read-txn per worker iteration (RoTxn is !Send/!Sync)
            let rtxn = env.read_txn().expect("failed to open read txn");
            let res = reader
                .nns(k)
                .ef_search(ef_search)
                .by_vector(&rtxn, q)
                .expect("hannoy search failed");

            let kth_sq = gt_d2[qi][k - 1];
            let kth    = kth_sq.sqrt();              // for reporting

            let returned = res.len();
            // hannoy returns squared L2, compare directly to squared GT radius
            let correct_here = res.iter().filter(|(_, d)| *d <= kth_sq).count();

            let last_ratio = if let Some((_, dlast)) = res.last() {
                dlast.sqrt() / kth
            } else {
                0.0
            };
            (correct_here as usize, last_ratio, returned as f32 / k as f32)
        })
        .reduce(
            || (0usize, 0f32, 0f32),
            |(a1, a2, a3), (b1, b2, b3)| (a1 + b1, a2 + b2, a3 + b3),
        );

    let secs = t0.elapsed().as_secs_f32();
    let recall = (correct as f32) / ((k * queries_f32.len()) as f32);
    let qps = (queries_f32.len() as f32) / secs;
    let mean_last_ratio = last_ratio_sum / (queries_f32.len() as f32);
    let mean_frac_returned = frac_returned_sum / (queries_f32.len() as f32);

    println!("\nSearching {} queries with k={}, ef_search={} …\n", queries_f32.len(), k, ef_search);
    println!(" mean fraction nb returned by search {}", mean_frac_returned);
    println!(" last distances ratio {}", mean_last_ratio);
    println!(
        " distance recall@{}: {:.4}  | throughput: {:.0} q/s  — time: {:.3}s",
        k, recall, qps, secs
    );
}

fn eval_distance_recall_single(
    env: &Env,
    reader: &Reader<Euclidean>,
    queries_f32: &[Vec<f32>],
    gt_d2: &[Vec<f32>],
    k: usize,
    ef_search: usize,
) {
    let t0 = Instant::now();
    let mut correct = 0usize;
    let mut last_ratio_sum = 0f32;
    let mut frac_returned_sum = 0f32;

    // Single read-txn is fine on one thread
    let rtxn = env.read_txn().expect("failed to open read txn");

    for (qi, q) in queries_f32.iter().enumerate() {
        let res = reader
            .nns(k)
            .ef_search(ef_search)
            .by_vector(&rtxn, q)
            .expect("hannoy search failed");
        let kth_sq = gt_d2[qi][k - 1];
        let kth    = kth_sq.sqrt();              // for reporting

        // hannoy returns squared L2, compare directly to squared GT radius
        correct += res.iter().filter(|(_, d)| *d <= kth_sq).count();
        if let Some((_, dlast)) = res.last() {
            last_ratio_sum += dlast.sqrt() / kth;
        }
        frac_returned_sum += (res.len() as f32) / (k as f32);
    }

    drop(rtxn);

    let secs = t0.elapsed().as_secs_f32();
    let recall = (correct as f32) / ((k * queries_f32.len()) as f32);
    let qps = (queries_f32.len() as f32) / secs;
    let mean_last_ratio = last_ratio_sum / (queries_f32.len() as f32);
    let mean_frac_returned = frac_returned_sum / (queries_f32.len() as f32);

    println!("\nSearching {} queries with k={}, ef_search={} …\n", queries_f32.len(), k, ef_search);
    println!(" mean fraction nb returned by search {}", mean_frac_returned);
    println!(" last distances ratio {}", mean_last_ratio);
    println!(
        " distance recall@{}: {:.4}  | throughput: {:.0} q/s  — time: {:.3}s",
        k, recall, qps, secs
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    // Toggle parallel vs single like your DiskANN example
    const PARALLEL: bool = true;

    // Files
    let base_path = "bigann_base.bvecs";
    let query_path = "bigann_query.bvecs";
    let gt_i_path = "idx_10M.ivecs";
    let gt_f_path = "dis_10M.fvecs";

    // LMDB env
    let lmdb_dir = PathBuf::from(LMDB_DIR);
    ensure_dir(&lmdb_dir)?;
    let env = open_env(&lmdb_dir)?;

    // Try to open existing index first (no base read).
    println!("Opening hannoy DB at {} …", LMDB_DIR);
    let (reader, db) = match try_open_reader(&env)? {
        Some((reader, db)) => {
            println!("Found existing hannoy index; skipping base load.");
            (reader, db)
        }
        None => {
            // Need to build: load base vectors now.
            println!(
                "No index or needs build; reading {} base vectors from {} …",
                NB_DATA_POINTS, base_path
            );
            let t_load = Instant::now();
            let base_u8 =
                read_all_bvecs_prefix::<DIM>(base_path, NB_DATA_POINTS, 50_000).expect("read base failed");
            assert!(
                base_u8.len() == NB_DATA_POINTS,
                "requested {} base vectors, got {}",
                NB_DATA_POINTS,
                base_u8.len()
            );
            let vectors_f32: Vec<Vec<f32>> = base_u8.iter().map(|v| u8s_to_f32(v)).collect();
            println!(
                "Loaded {} vectors in {:.1}s; building…",
                vectors_f32.len(),
                t_load.elapsed().as_secs_f32()
            );

            // Create/open the database handle for the build
            let mut wtxn = env.write_txn()?;
            let db: Database<Euclidean> = env.create_database(&mut wtxn, None)?;
            wtxn.commit()?;

            let reader = build_and_open_reader(&env, &db, DIM, &vectors_f32)?;
            (reader, db)
        }
    };

    // Queries
    println!("Reading first {} queries from {}…", NB_QUERY, query_path);
    let queries_u8 = read_query_bvecs::<DIM>(query_path, NB_QUERY).expect("failed reading queries");
    let queries_f32: Vec<Vec<f32>> = queries_u8.iter().map(|v| u8s_to_f32(v)).collect();

    // Ground truth (10M): we only need dists^2 for distance recall
    println!("Reading ground truth from {}, {}…", gt_i_path, gt_f_path);
    let (_gt_ids, gt_d2) = read_ground_truth(gt_i_path, gt_f_path, NB_QUERY).expect("failed reading GT");
    let kn = gt_d2[0].len();
    println!("GT loaded: {} queries, GT@{} per query", gt_d2.len(), kn);

    // Distance-based recall
    if PARALLEL {
        eval_distance_recall_parallel(&env, &reader, &queries_f32, &gt_d2, 10, EF_SEARCH);
        eval_distance_recall_parallel(&env, &reader, &queries_f32, &gt_d2, 100, EF_SEARCH);
    } else {
        eval_distance_recall_single(&env, &reader, &queries_f32, &gt_d2, 10, EF_SEARCH);
        eval_distance_recall_single(&env, &reader, &queries_f32, &gt_d2, 100, EF_SEARCH);
    }

    // Prevent unused warning in case you add mutations later
    let _ = db;

    Ok(())
}
// examples/perf_test.rs
use anndists::dist::DistCosine; // swap to DistL2/DistDot/etc. if desired
use rand::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rust_diskann::{DiskANN, DiskAnnError, DiskAnnParams};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), DiskAnnError> {
    const NUM_VECTORS: usize = 1_000_000;
    const DIM: usize = 1536;
    const MAX_DEGREE: usize = 32;
    const BUILD_BEAM_WIDTH: usize = 128;
    const ALPHA: f32 = 1.2;
    const PASSES: usize = 2;
    const EXTRA_SEEDS: usize = 2;

    let singlefile_path = "diskann_large.db";

    // Build if missing
    if !std::path::Path::new(singlefile_path).exists() {
        println!(
            "Building DiskANN index with {} vectors, dim={}, distance={}, \
             max_degree={}, build_beam={}, alpha={}, passes={}, extra_seeds={}",
            NUM_VECTORS,
            DIM,
            std::any::type_name::<DistCosine>(),
            MAX_DEGREE,
            BUILD_BEAM_WIDTH,
            ALPHA,
            PASSES,
            EXTRA_SEEDS
        );

        // Generate vectors
        println!("Generating vectors...");
        let mut rng = thread_rng();
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(NUM_VECTORS);
        for i in 0..NUM_VECTORS {
            if i % 100_000 == 0 {
                println!("  Generated {} vectors...", i);
            }
            // use r#gen() because `gen` is a reserved keyword in newer Rust editions
            let v: Vec<f32> = (0..DIM).map(|_| rng.r#gen::<f32>()).collect();
            vectors.push(v);
        }

        println!("Starting index build...");
        let start = Instant::now();
        let params = DiskAnnParams {
            max_degree: MAX_DEGREE,
            build_beam_width: BUILD_BEAM_WIDTH,
            alpha: ALPHA,
            passes: PASSES,
            extra_seeds: EXTRA_SEEDS,
        };

        // Distance type must match on open; include element type `f32`
        let _index = DiskANN::<f32, DistCosine>::build_index_with_params(
            &vectors,
            DistCosine,
            singlefile_path,
            params,
        )?;
        let elapsed = start.elapsed().as_secs_f32();
        println!("Done building index in {:.2} s", elapsed);
    } else {
        println!(
            "Index file {} already exists, skipping build.",
            singlefile_path
        );
    }

    // Open index (same element + distance types as build)
    let open_start = Instant::now();
    let index = Arc::new(DiskANN::<f32, DistCosine>::open_index_with(
        singlefile_path,
        DistCosine,
    )?);
    let open_time = open_start.elapsed().as_secs_f32();
    println!(
        "Opened index with {} vectors, dim={}, metric={} in {:.2} s",
        index.num_vectors, index.dim, index.distance_name, open_time
    );

    // Query settings
    let num_queries = 100;
    let k = 10;
    let beam_width = 64;

    // Generate query batch
    println!("\nGenerating {} query vectors...", num_queries);
    let mut rng = thread_rng();
    let mut query_batch: Vec<Vec<f32>> = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        let q: Vec<f32> = (0..index.dim).map(|_| rng.r#gen::<f32>()).collect();
        query_batch.push(q);
    }

    // Sequential queries to measure per-query latency
    println!("\nRunning sequential queries to measure performance...");
    let mut times = Vec::new();
    for (i, query) in query_batch.iter().take(10).enumerate() {
        let start = Instant::now();
        let neighbors = index.search(query, k, beam_width);
        let elapsed = start.elapsed();
        times.push(elapsed.as_micros());
        println!(
            "Query {}: found {} neighbors in {:?}",
            i,
            neighbors.len(),
            elapsed
        );
    }

    if !times.is_empty() {
        let avg_time = times.iter().sum::<u128>() as f64 / times.len() as f64;
        println!("Average query time (first 10): {:.2} µs", avg_time);
    }

    // Parallel queries to test throughput
    println!("\nRunning {} queries in parallel...", num_queries);
    let search_start = Instant::now();
    let results: Vec<Vec<u32>> = query_batch
        .par_iter()
        .map(|query| index.search(query, k, beam_width))
        .collect();
    let search_time = search_start.elapsed().as_secs_f32();

    println!("Performed {} queries in {:.2} s", num_queries, search_time);
    println!(
        "Throughput: {:.2} queries/sec",
        num_queries as f32 / search_time
    );

    // Verify all queries returned results
    let all_valid = results.iter().all(|r| r.len() == k.min(index.num_vectors));
    println!("All queries returned valid results: {}", all_valid);

    // Memory footprint note
    println!("\nMemory-mapped index ready; process RSS should stay modest.");
    println!("(Optional) Check memory usage with `ps aux | grep perf_test` in another terminal.");
    println!("Press Enter to exit...");

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    Ok(())
}
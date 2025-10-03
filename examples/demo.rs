// examples/demo.rs
use anndists::dist::DistCosine; // swap to DistL2, DistDot, etc. if desired
use rust_diskann::{DiskANN, DiskAnnError, DiskAnnParams};
use rand::prelude::*;
use std::path::Path;
use std::sync::Arc;

fn main() -> Result<(), DiskAnnError> {
    let singlefile_path = "diskann.db";
    let num_vectors = 1_000_000usize;
    let dim = 1024usize;

    // Build-time knobs (match your DiskAnnParams)
    let max_degree = 32usize;
    let build_beam_width = 128usize;
    let alpha = 1.2f32;

    // Build if missing
    if !Path::new(singlefile_path).exists() {
        println!("Building DiskANN index at {singlefile_path}...");

        // Generate sample vectors (replace with your real dataset)
        println!("Generating {num_vectors} sample vectors of dimension {dim}...");
        let mut rng = thread_rng();
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(num_vectors);
        for _ in 0..num_vectors {
            // NOTE: use r#gen() since `gen` is a reserved keyword in newer Rust editions
            let v: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>()).collect();
            vectors.push(v);
        }

        // Or: use explicit params (shown here)
        let params = DiskAnnParams {
            max_degree,
            build_beam_width,
            alpha,
        };
        // NOTE: DiskANN<T, D> → T = f32, D = DistCosine
        let index = DiskANN::<f32, DistCosine>::build_index_with_params(
            &vectors,
            DistCosine,
            singlefile_path,
            params,
        )?;

        println!(
            "Build done. Index contains {} vectors (dim={}, max_degree={})",
            index.num_vectors, index.dim, index.max_degree
        );
    } else {
        println!("Index file {singlefile_path} already exists, skipping build.");
    }

    // Open the index (distance type must match what you used to build)
    let index = Arc::new(DiskANN::<f32, DistCosine>::open_index_with(
        singlefile_path,
        DistCosine,
    )?);

    println!(
        "Opened index: {} vectors, dimension={}, max_degree={}",
        index.num_vectors, index.dim, index.max_degree
    );

    // Perform a sample query
    let mut rng = thread_rng();
    let query: Vec<f32> = (0..index.dim).map(|_| rng.r#gen::<f32>()).collect();

    let k = 10usize;
    let search_beam_width = 64usize;

    println!(
        "\nSearching for {k} nearest neighbors with beam_width={}...",
        search_beam_width
    );
    let start = std::time::Instant::now();
    let neighbors: Vec<u32> = index.search(&query, k, search_beam_width);
    let elapsed = start.elapsed();

    println!("Search completed in {:?}", elapsed);
    println!("Found {} neighbors:", neighbors.len());
    for (i, &id) in neighbors.iter().enumerate() {
        println!("  {}: node {}", i + 1, id);
    }

    Ok(())
}
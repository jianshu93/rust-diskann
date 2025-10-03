# DiskANN: On-disk graph-based approximate nearest neighbor search 🦀

[![Latest Version](https://img.shields.io/crates/v/rust_diskann?style=for-the-badge&color=mediumpurple&logo=rust)](https://crates.io/crates/rust_diskann)
[![docs.rs](https://img.shields.io/docsrs/rust-diskann?style=for-the-badge&logo=docs.rs&color=mediumseagreen)](https://docs.rs/rust_diskann/latest/rust_diskann/)


A Rust implementation of [DiskANN](https://proceedings.neurips.cc/paper_files/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html) (Disk-based Approximate Nearest Neighbor search) using the Vamana graph algorithm. This project provides an efficient and scalable solution for large-scale vector similarity search with minimal memory footprint, as an alternative to the widely used in-memory [HNSW](https://ieeexplore.ieee.org/abstract/document/8594636) algorithm. 

## Key algorithm

This implementation follows the DiskANN paper's approach:
- Using the Vamana graph algorithm for index construction, pruning and refinement (in parallel)
- Memory-mapping the index file for efficient disk-based access (via memmap2)
- Implementing beam search with medoid entry points (in parallel)
- Supporting Euclidean, Cosine, Hamming and other distance metrics via a generic distance trait
- Maintaining minimal memory footprint during search operations

## Features

- **Single-file storage**: All index data stored in one memory-mapped file in the following way:

        [ metadata_len:u64 ][ metadata (bincode) ][ padding up to vectors_offset ]
        [ vectors (num * dim * f32) ][ adjacency (num * max_degree * u32) ]

        `vectors_offset` is a fixed 1 MiB gap by default.


- **Vamana graph construction**: Efficient graph building with α-pruning, with rayon for concurrent and parallel construction
- **Memory-efficient search**: Uses beam search that visits < 1% of vectors
- **Distance metrics**: Support for Euclidean, Cosine and Hamming similarity et.al. via [anndists](https://crates.io/crates/anndists). A generic distance trait that can be extended to other distance metrics
- **Medoid-based entry points**: Smart starting points for search
- **Parallel query processing**: Using rayon for concurrent searches. Note: this may increase the loaded pages during memory map
- **Minimal memory footprint**: ~330MB RAM for 2GB index (16% of file size)
- **Extensitve benchmarks**: Speed, accuracy and memory consumption benchmark with HNSW (both in-memory and on-disk)

## Usage in Rust 🦀

### Building a New Index

```rust
use anndists::dist::{DistL2, DistCosine}; // or your own Distance types
use diskann_rs::{DiskANN, DiskAnnParams};

// Your vectors to index (all rows must share the same dimension)
let vectors: Vec<Vec<f32>> = vec![
    vec![0.1, 0.2, 0.3],
    vec![0.4, 0.5, 0.6],
];

// Easiest: build with defaults (M=64, L_build=128, alpha=1.2)
let index = DiskANN::<DistL2>::build_index_default(&vectors, DistL2 {}, "index.db")?;

// Or: custom construction parameters
let params = DiskAnnParams {
    max_degree: 48,        // max neighbors per node
    build_beam_width: 128, // construction beam width
    alpha: 1.2,            // α for pruning
};
let index2 = DiskANN::<DistCosine>::build_index_with_params(
    &vectors,
    DistCosine {},
    "index_cos.db",
    params,
)?;
```

### Opening an Existing Index

```rust
use anndists::dist::DistL2;
use diskann_rs::DiskANN;

// If you built with DistL2 and defaults:
let index = DiskANN::<DistL2>::open_index_default_metric("index.db")?;

// Or, explicitly provide the distance you built with:
let index2 = DiskANN::<DistL2>::open_index_with("index.db", DistL2 {})?;
```

### Searching the Index

```rust
use anndists::dist::DistL2;
use diskann_rs::DiskANN;

let index = DiskANN::<DistL2>::open_index_default_metric("index.db")?;
let query: Vec<f32> = vec![0.1, 0.2, 0.4]; // length must match the indexed dim
let k = 10;
let beam = 256; // search beam width

// (IDs, distance)
let hits: Vec<(u32, f32)> = index.search_with_dists(&query, 10, beam);
// `neighbors` are the IDs of the k nearest vectors
let neighbors: Vec<u32> = index.search(&query, k, beam);

```

### Parallel Search

```rust
use anndists::dist::DistL2;
use diskann_rs::DiskANN;
use rayon::prelude::*;

let index = DiskANN::<DistL2>::open_index_default_metric("index.db")?;

// Suppose you have a batch of queries
let query_batch: Vec<Vec<f32>> = /* ... */;

let results: Vec<Vec<u32>> = query_batch
    .par_iter()
    .map(|q| index.search(q, 10, 256))
    .collect();
```

## Space and time complexity analysis

- **Index Build Time**: O(n * max_degree * beam_width)
- **Disk Space**: n * (dimension * 4 + max_degree * 4) bytes
- **Search Time**: O(beam_width * log n) - typically visits < 1% of dataset
- **Memory Usage**: O(beam_width) during search
- **Query Throughput**: Scales linearly with CPU cores

## Parameters Tuning

### Index building Parameters
- `max_degree`: 32-64 for most datasets
- `build_beam_width`: 128-256 for good graph quality
- `alpha`: 1.2-2.0 (higher = more diverse neighbors)

### Index search Parameters
- `beam_width`: 128 or larger (trade-off between speed and recall)
- Higher beam_width = better recall but slower search

### Index memory-mapping

When host RAM is not large enough for mapping the entire database file, it is possible to build the database in several smaller pieces (random split). Then users can search the query againt each piece and collect results from each piece before merging (rank by distance). This is equivalent to a single big database approach but requires a much smaller number of RAM for memory-mapping. 

## Building and Testing

```bash
# Build the library
cargo build --release

# Run tests
cargo test

# Run demo
cargo run --release --example demo
# Run performance test
cargo run --release --example perf_test

# test MNIST fashion dataset
wget http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5
cargo run --release --example diskann_mnist

# test SIFT dataset
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
cargo run --release --example diskann_sift
```

## Examples

See the `examples/` directory for:
- `demo.rs`: Demo with 100k vectors  
- `perf_test.rs`: Performance benchmarking with 1M vectors
- `diskann_mnist.rs`: Performance benchmarking with MNIST fashion dataset (60K)
- `diskann_sift.rs`: Performance benchmarking with SIFT 1M dataset
- `bigann.rs`: Performance benchmarking with SIFT 10M dataset
- `hnsw_sift.rs`: Comparison with in-memory HNSW

## Benchmark against in-memory HNSW ([hnsw_rs](https://crates.io/crates/hnsw_rs) crate)
```bash
### MNIST fashion, diskann, M4 Max
Building DiskANN index: n=60000, dim=784, max_degree=48, build_beam=256, alpha=1.2
Build complete. CPU time: 1726.199372s, wall time: 111.145414s
Searching 10000 queries with k=10, beam_width=384 …
 mean fraction nb returned by search 1.0

 last distances ratio 1.0031366
 recall rate for "./fashion-mnist-784-euclidean.hdf5" is 0.98838 , nb req /s 18067.664

 total cpu time for search requests 8.520862s , system time 553.475ms


### MNIST fashion, hnsw_rs, M4 Max
parallel insertion

 hnsw data insertion cpu time  111.169283s  system time Ok(7.256291s) 
 debug dump of PointIndexation
 layer 0 : length : 59999 
 layer 1 : length : 1 
 debug dump of PointIndexation end
 hnsw data nb point inserted 60000

 searching with ef : 24
 
 parallel search
total cpu time for search requests 3838.7310ms , system time 263.571ms 

 mean fraction nb returned by search 1.0 

 last distances ratio 1.0003573 

 recall rate for "./fashion-mnist-784-euclidean.hdf5" is 0.99054 , nb req /s 37940.44

```


## License
MIT 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References
Jayaram Subramanya, S., Devvrit, F., Simhadri, H.V., Krishnawamy, R. and Kadekodi, R., 2019. Diskann: Fast accurate billion-point nearest neighbor search on a single node. Advances in neural information processing Systems, 32.

## Acknowledgments

This implementation is based on the DiskANN paper and the official Microsoft implementation. It was also largely inspired by the implementation [here](https://github.com/lukaesch/diskann-rs). 
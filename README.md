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
- **Generic over any vector<T>**: suport various types of vector input.
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
let index = DiskANN::<f32, DistL2>::build_index_default(&vectors, DistL2, "index.db")?;

// Or: custom construction parameters
let params = DiskAnnParams {
    max_degree: 48,        // max neighbors per node
    build_beam_width: 128, // construction beam width
    alpha: 1.2,            // α for pruning
};
let index2 = DiskANN::<f32, DistCosine>::build_index_with_params(
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
let index = DiskANN::<f32, DistL2>::open_index_default_metric("index.db")?;

// Or, explicitly provide the distance you built with:
let index2 = DiskANN::<f32, DistL2>::open_index_with("index.db", DistL2)?;
```

### Searching the Index

```rust
use anndists::dist::DistL2;
use diskann_rs::DiskANN;

let index = DiskANN::<f32, DistL2>::open_index_default_metric("index.db")?;
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

let index = DiskANN::<f32, DistL2>::open_index_default_metric("index.db")?;

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

When host RAM is not large enough for mapping the entire database file, it is possible to build the database in several smaller pieces (random split). Then users can search the query againt each piece and collect results from each piece before merging (rank by distance). This is equivalent to a single big database approach (as long as K'>=K) but requires a much smaller number of RAM for memory-mapping. In practice, the Microsoft Azure Cosmos DB found that this database shard idea can improve recall. Intutively, with smaller data points for each piece, we can use large M and build beam width to further improve accuracy. See their paper [here](https://www.vldb.org/pvldb/vol18/p5166-upreti.pdf)

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
- `hannoy_sift.rs`: Comparison with on-disk HNSW

## Benchmark against in-memory HNSW ([hnsw_rs](https://crates.io/crates/hnsw_rs) crate) and on-disk HNSW ([hannoy](https://crates.io/crates/hannoy)) for SIFT 1 million dataset

```bash
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
cargo run --release --example diskann_sift
cargo run --release --example hnsw_sift
cargo run --release --example hannoy_sift

```

Results:
```bash
## DiskANN,  sift1m , M4 Max
DiskANN benchmark on "./sift-128-euclidean.hdf5"
neighbours shape : [10000, 100]

 10 first neighbours for first vector : 
 932085  934876  561813  708177  706771  695756  435345  701258  455537  872728 
 10 first neighbours for second vector : 
 413247  413071  706838  880592  249062  400194  942339  880462  987636  941776  test data, nb element 10000,  dim : 128

 train data shape : [1000000, 128], nbvector 1000000 
 allocating vector for search neighbours answer : 10000
Train size : 1000000
Test size  : 10000
Ground-truth k per query in file: 100

Building DiskANN index: n=1000000, dim=128, max_degree=64, build_beam=128, alpha=1.2
Build complete. CPU time: 4424.408064s, wall time: 294.704059s

Searching 10000 queries with k=10, beam_width=512 …

 mean fraction nb returned by search 1.0

 last distances ratio 1.0002044

 recall rate for "./sift-128-euclidean.hdf5" is 0.99591 , nb req /s 8590.497
 total cpu time for search requests 15.077785s , system time 1.164077s

 

###  sift1m, hnsw_rs, M4 Max
neighbours shape : [10000, 100]

 10 first neighbours for first vector : 
 932085  934876  561813  708177  706771  695756  435345  701258  455537  872728 
 10 first neighbours for second vector : 
 413247  413071  706838  880592  249062  400194  942339  880462  987636  941776  test data, nb element 10000,  dim : 128

 train data shape : [1000000, 128], nbvector 1000000 
 allocating vector for search neighbours answer : 10000
No saved index. Building new one: N=1000000 layers=16 ef_c=256

  Current scale value : 2.58e-1, Scale modification factor asked : 5.00e-1,(modification factor must be between 2.00e-1 and 1.)
 
 parallel insertion
 setting number of points 50000 
 setting number of points 100000 
 setting number of points 150000 
 setting number of points 200000 
 setting number of points 250000 
 setting number of points 300000 
 setting number of points 350000 
 setting number of points 400000 
 setting number of points 450000 
 setting number of points 500000 
 setting number of points 550000 
 setting number of points 600000 
 setting number of points 650000 
 setting number of points 700000 
 setting number of points 750000 
 setting number of points 800000 
 setting number of points 850000 
 setting number of points 900000 
 setting number of points 950000 
 setting number of points 1000000 
HNSW index saved as: sift1m_l2_hnsw.hnsw.graph / .hnsw.data

 hnsw data insertion cpu time  1447.190068s  system time Ok(95.761465s) 
 debug dump of PointIndexation
 layer 0 : length : 999596 
 layer 1 : length : 403 
 layer 2 : length : 1 
 debug dump of PointIndexation end
 hnsw data nb point inserted 1000000
searching with ef = 64


 ef_search : 64 knbn : 10 
searching with ef : 64
 
 parallel search
total cpu time for search requests 6140413.0 , system time Ok(405.2ms) 

 mean fraction nb returned by search 1.0 

 last distances ratio 1.0004181 

 recall rate for "./sift-128-euclidean.hdf5" is 0.98791 , nb req /s 24679.17


### hannoy, on-disk hnsw

hannoy × SIFT1M @ L2  -> "./sift-128-euclidean.hdf5"
neighbours shape : [10000, 100]

 10 first neighbours for first vector : 
 932085  934876  561813  708177  706771  695756  435345  701258  455537  872728 
 10 first neighbours for second vector : 
 413247  413071  706838  880592  249062  400194  942339  880462  987636  941776  test data, nb element 10000,  dim : 128

 train data shape : [1000000, 128], nbvector 1000000 
 allocating vector for search neighbours answer : 10000
Train size : 1000000
Test size  : 10000
Ground-truth k per query in file: 100
Database present but needs build; building now…
Building hannoy index: N=1000000, dim=128, M=48, M0=48, ef_c=256
hannoy build complete. CPU time: 1149.622047s, wall time: 76.101147s

Searching 10000 queries with k=10, ef_search=64 …
 distance recall@10  : 0.9803
 last distances ratio (ours true L2 / GT kth): 1.0007
 throughput: 11000 q/s — cpu: 14.334211s  wall: 909.061ms



```


## License
MIT 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References
Jayaram Subramanya, S., Devvrit, F., Simhadri, H.V., Krishnawamy, R. and Kadekodi, R., 2019. Diskann: Fast accurate billion-point nearest neighbor search on a single node. Advances in neural information processing Systems, 32.

## Acknowledgments

This implementation is based on the DiskANN paper and the official Microsoft implementation. It was also largely inspired by the implementation [here](https://github.com/lukaesch/diskann-rs). 
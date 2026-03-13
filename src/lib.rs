//! # DiskAnn (generic over `anndists::Distance<T>`)
//!
//! An on-disk DiskANN library that:
//! - Builds a Vamana-style graph (greedy + α-pruning) in memory
//! - Writes vectors + fixed-degree adjacency to a single file
//! - Memory-maps the file for low-overhead reads
//! - Is **generic over any Distance<T>** from `anndists` (e.g. L2 on `f32`, Cosine on `f32`,
//!   Hamming on `u64`, …)
//!
//! ## Example (f32 + L2)
//! ```no_run
//! use anndists::dist::DistL2;
//! use rust_diskann::{DiskANN, DiskAnnParams};
//!
//! let vectors: Vec<Vec<f32>> = vec![vec![0.0; 128]; 1000];
//! let index = DiskANN::<f32, DistL2>::build_index_default(&vectors, DistL2, "index.db").unwrap();
//!
//! let q = vec![0.0; 128];
//! let nns = index.search(&q, 10, 64);
//! ```
//!
//! ## Example (u64 + Hamming)
//! ```no_run
//! use anndists::dist::DistHamming;
//! use rust_diskann::{DiskANN, DiskAnnParams};
//! let index: Vec<Vec<u64>> = vec![vec![0u64; 128]; 1000];
//! let idx = DiskANN::<u64, DistHamming>::build_index_default(&index, DistHamming, "mh.db").unwrap();
//! let q = vec![0u64; 128];
//! let _ = idx.search(&q, 10, 64);
//! ```
//!
//! ## File Layout
//! [ metadata_len:u64 ][ metadata (bincode) ][ padding up to vectors_offset ]
//! [ vectors (num * dim * T) ][ adjacency (num * max_degree * u32) ]
//!
//! `vectors_offset` is a fixed 1 MiB gap by default.

use anndists::prelude::Distance;
use memmap2::Mmap;
use rand::{prelude::*, thread_rng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use thiserror::Error;

/// Padding sentinel for adjacency slots (avoid colliding with node 0).
const PAD_U32: u32 = u32::MAX;

/// Defaults for in-memory DiskANN builds
pub const DISKANN_DEFAULT_MAX_DEGREE: usize = 64;
pub const DISKANN_DEFAULT_BUILD_BEAM: usize = 128;
pub const DISKANN_DEFAULT_ALPHA: f32 = 1.2;
/// Default number of refinement passes during graph build
pub const DISKANN_DEFAULT_PASSES: usize = 2;
/// Default number of extra random seeds per node per pass during graph build
pub const DISKANN_DEFAULT_EXTRA_SEEDS: usize = 2;

/// Practical DiskANN-style slack before reverse-neighbor re-pruning.
/// Legacy C++ DiskANN allows reverse lists to grow to about GRAPH_SLACK_FACTOR * R
/// before triggering prune, instead of pruning immediately at R.
const GRAPH_SLACK_FACTOR: f32 = 1.3;

/// Optional bag of knobs if you want to override just a few.
#[derive(Clone, Copy, Debug)]
pub struct DiskAnnParams {
    pub max_degree: usize,
    pub build_beam_width: usize,
    pub alpha: f32,
    /// Number of refinement passes over the graph (>=1).
    pub passes: usize,
    /// Extra random seeds per node during each pass (>=0).
    pub extra_seeds: usize,
}

impl Default for DiskAnnParams {
    fn default() -> Self {
        Self {
            max_degree: DISKANN_DEFAULT_MAX_DEGREE,
            build_beam_width: DISKANN_DEFAULT_BUILD_BEAM,
            alpha: DISKANN_DEFAULT_ALPHA,
            passes: DISKANN_DEFAULT_PASSES,
            extra_seeds: DISKANN_DEFAULT_EXTRA_SEEDS,
        }
    }
}

/// Custom error type for DiskAnn operations
#[derive(Debug, Error)]
pub enum DiskAnnError {
    /// Represents I/O errors during file operations
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Represents serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Bincode(#[from] bincode::Error),

    /// Represents index-specific errors
    #[error("Index error: {0}")]
    IndexError(String),
}

/// Internal metadata structure stored in the index file
#[derive(Serialize, Deserialize, Debug)]
struct Metadata {
    dim: usize,
    num_vectors: usize,
    max_degree: usize,
    medoid_id: u32,
    vectors_offset: u64,
    adjacency_offset: u64,
    elem_size: u8,
    distance_name: String,
}

/// Candidate for search/frontier queues
#[derive(Clone, Copy, Debug)]
struct Candidate {
    dist: f32,
    id: u32,
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.dist.to_bits() == other.dist.to_bits()
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(
            self.dist
                .total_cmp(&other.dist)
                .then_with(|| self.id.cmp(&other.id)),
        )
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Flat contiguous matrix used during build to improve cache locality.
///
/// Rows are stored consecutively in `data`, row-major.
#[derive(Clone, Debug)]
struct FlatVectors<T> {
    data: Vec<T>,
    dim: usize,
    n: usize,
}

impl<T: Copy> FlatVectors<T> {
    fn from_vecs(vectors: &[Vec<T>]) -> Result<Self, DiskAnnError> {
        if vectors.is_empty() {
            return Err(DiskAnnError::IndexError("No vectors provided".to_string()));
        }
        let dim = vectors[0].len();
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(DiskAnnError::IndexError(format!(
                    "Vector {} has dimension {} but expected {}",
                    i,
                    v.len(),
                    dim
                )));
            }
        }

        let n = vectors.len();
        let mut data = Vec::with_capacity(n * dim);
        for v in vectors {
            data.extend_from_slice(v);
        }

        Ok(Self { data, dim, n })
    }

    #[inline]
    fn row(&self, idx: usize) -> &[T] {
        let start = idx * self.dim;
        let end = start + self.dim;
        &self.data[start..end]
    }
}

/// Small ordered beam structure used only during build-time greedy search.
///
/// It keeps elements in **descending** distance order:
/// - index 0 is the worst element
/// - last element is the best element
///
/// This makes:
/// - `best()` cheap via `last()`
/// - `worst()` cheap via `first()`
/// - capped beam maintenance simple
#[derive(Default, Debug)]
struct OrderedBeam {
    items: Vec<Candidate>,
}

impl OrderedBeam {
    #[inline]
    fn clear(&mut self) {
        self.items.clear();
    }

    #[inline]
    fn len(&self) -> usize {
        self.items.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    #[inline]
    fn best(&self) -> Option<Candidate> {
        self.items.last().copied()
    }

    #[inline]
    fn worst(&self) -> Option<Candidate> {
        self.items.first().copied()
    }

    #[inline]
    fn pop_best(&mut self) -> Option<Candidate> {
        self.items.pop()
    }

    #[inline]
    fn reserve(&mut self, cap: usize) {
        if self.items.capacity() < cap {
            self.items.reserve(cap - self.items.capacity());
        }
    }

    #[inline]
    fn insert_unbounded(&mut self, cand: Candidate) {
        let pos = self.items.partition_point(|x| {
            x.dist > cand.dist || (x.dist.to_bits() == cand.dist.to_bits() && x.id > cand.id)
        });
        self.items.insert(pos, cand);
    }

    #[inline]
    fn insert_capped(&mut self, cand: Candidate, cap: usize) {
        if cap == 0 {
            return;
        }

        if self.items.len() < cap {
            self.insert_unbounded(cand);
            return;
        }

        // Since items[0] is the worst, only insert if the new candidate is better.
        let worst = self.items[0];
        if cand.dist >= worst.dist {
            return;
        }

        self.insert_unbounded(cand);

        if self.items.len() > cap {
            self.items.remove(0);
        }
    }
}

/// Reusable scratch buffers for build-time greedy search.
/// One instance is created per Rayon worker via `map_init`, so allocations are reused
/// across many nodes in the build.
#[derive(Debug)]
struct BuildScratch {
    marks: Vec<u32>,
    epoch: u32,

    visited_ids: Vec<u32>,
    visited_dists: Vec<f32>,

    frontier: OrderedBeam,
    work: OrderedBeam,

    seeds: Vec<usize>,
    candidates: Vec<(u32, f32)>,
}

impl BuildScratch {
    fn new(n: usize, beam_width: usize, max_degree: usize, extra_seeds: usize) -> Self {
        Self {
            marks: vec![0u32; n],
            epoch: 1,
            visited_ids: Vec::with_capacity(beam_width * 4),
            visited_dists: Vec::with_capacity(beam_width * 4),
            frontier: {
                let mut b = OrderedBeam::default();
                b.reserve(beam_width * 2);
                b
            },
            work: {
                let mut b = OrderedBeam::default();
                b.reserve(beam_width * 2);
                b
            },
            seeds: Vec::with_capacity(1 + extra_seeds),
            candidates: Vec::with_capacity(beam_width * (4 + extra_seeds) + max_degree * 2),
        }
    }

    #[inline]
    fn reset_search(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
        self.visited_ids.clear();
        self.visited_dists.clear();
        self.frontier.clear();
        self.work.clear();
    }

    #[inline]
    fn is_marked(&self, idx: usize) -> bool {
        self.marks[idx] == self.epoch
    }

    #[inline]
    fn mark_with_dist(&mut self, idx: usize, dist: f32) {
        self.marks[idx] = self.epoch;
        self.visited_ids.push(idx as u32);
        self.visited_dists.push(dist);
    }
}

/// Main struct representing a DiskANN index (generic over vector element `T` and distance `D`)
pub struct DiskANN<T, D>
where
    T: bytemuck::Pod + Copy + Send + Sync + 'static,
    D: Distance<T> + Send + Sync + Copy + Clone + 'static,
{
    /// Dimensionality of vectors in the index
    pub dim: usize,
    /// Number of vectors in the index
    pub num_vectors: usize,
    /// Maximum number of edges per node
    pub max_degree: usize,
    /// Informational: type name of the distance (from metadata)
    pub distance_name: String,

    /// ID of the medoid (used as entry point)
    medoid_id: u32,
    // Offsets
    vectors_offset: u64,
    adjacency_offset: u64,

    /// Memory-mapped file
    mmap: Mmap,

    /// The distance strategy
    dist: D,

    /// keep `T` in the type so the compiler knows about it
    _phantom: PhantomData<T>,
}

// constructors

impl<T, D> DiskANN<T, D>
where
    T: bytemuck::Pod + Copy + Send + Sync + 'static,
    D: Distance<T> + Send + Sync + Copy + Clone + 'static,
{
    /// Build with default parameters: (M=64, L=128, alpha=1.2, passes=2, extra_seeds=2).
    pub fn build_index_default(
        vectors: &[Vec<T>],
        dist: D,
        file_path: &str,
    ) -> Result<Self, DiskAnnError> {
        Self::build_index(
            vectors,
            DISKANN_DEFAULT_MAX_DEGREE,
            DISKANN_DEFAULT_BUILD_BEAM,
            DISKANN_DEFAULT_ALPHA,
            DISKANN_DEFAULT_PASSES,
            DISKANN_DEFAULT_EXTRA_SEEDS,
            dist,
            file_path,
        )
    }

    /// Build with a `DiskAnnParams` bundle.
    pub fn build_index_with_params(
        vectors: &[Vec<T>],
        dist: D,
        file_path: &str,
        p: DiskAnnParams,
    ) -> Result<Self, DiskAnnError> {
        Self::build_index(
            vectors,
            p.max_degree,
            p.build_beam_width,
            p.alpha,
            p.passes,
            p.extra_seeds,
            dist,
            file_path,
        )
    }

    /// Opens an existing index file, supplying the distance strategy explicitly.
    pub fn open_index_with(path: &str, dist: D) -> Result<Self, DiskAnnError> {
        let mut file = OpenOptions::new().read(true).write(false).open(path)?;

        // Read metadata length
        let mut buf8 = [0u8; 8];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut buf8)?;
        let md_len = u64::from_le_bytes(buf8);

        // Read metadata
        let mut md_bytes = vec![0u8; md_len as usize];
        file.read_exact(&mut md_bytes)?;
        let metadata: Metadata = bincode::deserialize(&md_bytes)?;

        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Validate element size vs T
        let want = std::mem::size_of::<T>() as u8;
        if metadata.elem_size != want {
            return Err(DiskAnnError::IndexError(format!(
                "element size mismatch: file has {}B, T is {}B",
                metadata.elem_size, want
            )));
        }

        // Optional sanity/logging: warn if type differs from recorded name
        let expected = std::any::type_name::<D>();
        if metadata.distance_name != expected {
            eprintln!(
                "Warning: index recorded distance `{}` but you opened with `{}`",
                metadata.distance_name, expected
            );
        }

        Ok(Self {
            dim: metadata.dim,
            num_vectors: metadata.num_vectors,
            max_degree: metadata.max_degree,
            distance_name: metadata.distance_name,
            medoid_id: metadata.medoid_id,
            vectors_offset: metadata.vectors_offset,
            adjacency_offset: metadata.adjacency_offset,
            mmap,
            dist,
            _phantom: PhantomData,
        })
    }
}

/// Extra sugar when your distance type implements `Default`.
impl<T, D> DiskANN<T, D>
where
    T: bytemuck::Pod + Copy + Send + Sync + 'static,
    D: Distance<T> + Default + Send + Sync + Copy + Clone + 'static,
{
    /// Build with default params **and** `D::default()` metric.
    pub fn build_index_default_metric(
        vectors: &[Vec<T>],
        file_path: &str,
    ) -> Result<Self, DiskAnnError> {
        Self::build_index_default(vectors, D::default(), file_path)
    }

    /// Open an index using `D::default()` as the distance (matches what you built with).
    pub fn open_index_default_metric(path: &str) -> Result<Self, DiskAnnError> {
        Self::open_index_with(path, D::default())
    }
}

impl<T, D> DiskANN<T, D>
where
    T: bytemuck::Pod + Copy + Send + Sync + 'static,
    D: Distance<T> + Send + Sync + Copy + Clone + 'static,
{
    /// Builds a new index from provided vectors
    ///
    /// # Arguments
    /// * `vectors` - The vectors to index (slice of Vec<T>)
    /// * `max_degree` - Maximum edges per node (M ~ 24-64+)
    /// * `build_beam_width` - Construction L (e.g., 128-400)
    /// * `alpha` - Pruning parameter (1.2–2.0)
    /// * `passes` - Refinement passes over the graph (>=1)
    /// * `extra_seeds` - Extra random seeds per node per pass (>=0)
    /// * `dist` - Any `anndists::Distance<T>`
    /// * `file_path` - Path of index file
    pub fn build_index(
        vectors: &[Vec<T>],
        max_degree: usize,
        build_beam_width: usize,
        alpha: f32,
        passes: usize,
        extra_seeds: usize,
        dist: D,
        file_path: &str,
    ) -> Result<Self, DiskAnnError> {
        let flat = FlatVectors::from_vecs(vectors)?;

        let num_vectors = flat.n;
        let dim = flat.dim;

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(file_path)?;

        // Reserve space for metadata (we'll write it after data)
        let vectors_offset = 1024 * 1024;
        assert_eq!(
            (vectors_offset as usize) % std::mem::align_of::<T>(),
            0,
            "vectors_offset must be aligned for T"
        );

        let elem_sz = std::mem::size_of::<T>() as u64;
        let total_vector_bytes = (num_vectors as u64) * (dim as u64) * elem_sz;

        // Write vectors contiguous
        file.seek(SeekFrom::Start(vectors_offset as u64))?;
        file.write_all(bytemuck::cast_slice::<T, u8>(&flat.data))?;

        // Compute medoid using flat storage
        let medoid_id = calculate_medoid(&flat, dist);

        // Build graph
        let adjacency_offset = vectors_offset as u64 + total_vector_bytes;
        let graph = build_vamana_graph(
            &flat,
            max_degree,
            build_beam_width,
            alpha,
            passes,
            extra_seeds,
            dist,
            medoid_id as u32,
        );

        // Write adjacency lists
        file.seek(SeekFrom::Start(adjacency_offset))?;
        for neighbors in &graph {
            let mut padded = neighbors.clone();
            padded.resize(max_degree, PAD_U32);
            let bytes = bytemuck::cast_slice::<u32, u8>(&padded);
            file.write_all(bytes)?;
        }

        // Write metadata
        let metadata = Metadata {
            dim,
            num_vectors,
            max_degree,
            medoid_id: medoid_id as u32,
            vectors_offset: vectors_offset as u64,
            adjacency_offset,
            elem_size: std::mem::size_of::<T>() as u8,
            distance_name: std::any::type_name::<D>().to_string(),
        };

        let md_bytes = bincode::serialize(&metadata)?;
        file.seek(SeekFrom::Start(0))?;
        let md_len = md_bytes.len() as u64;
        file.write_all(&md_len.to_le_bytes())?;
        file.write_all(&md_bytes)?;
        file.sync_all()?;

        // Memory map the file
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        Ok(Self {
            dim,
            num_vectors,
            max_degree,
            distance_name: metadata.distance_name,
            medoid_id: metadata.medoid_id,
            vectors_offset: metadata.vectors_offset,
            adjacency_offset: metadata.adjacency_offset,
            mmap,
            dist,
            _phantom: PhantomData,
        })
    }

    /// Searches the index for nearest neighbors using a best-first beam search.
    /// Termination rule: continue while the best frontier can still improve the worst in working set.
    pub fn search_with_dists(&self, query: &[T], k: usize, beam_width: usize) -> Vec<(u32, f32)> {
        assert_eq!(
            query.len(),
            self.dim,
            "Query dim {} != index dim {}",
            query.len(),
            self.dim
        );

        let mut visited = HashSet::new();
        let mut frontier: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        let mut w: BinaryHeap<Candidate> = BinaryHeap::new();

        let start_dist = self.distance_to(query, self.medoid_id as usize);
        let start = Candidate {
            dist: start_dist,
            id: self.medoid_id,
        };
        frontier.push(Reverse(start));
        w.push(start);
        visited.insert(self.medoid_id);

        while let Some(Reverse(best)) = frontier.peek().copied() {
            if w.len() >= beam_width {
                if let Some(worst) = w.peek() {
                    if best.dist >= worst.dist {
                        break;
                    }
                }
            }
            let Reverse(current) = frontier.pop().unwrap();

            for &nb in self.get_neighbors(current.id) {
                if nb == PAD_U32 {
                    continue;
                }
                if !visited.insert(nb) {
                    continue;
                }

                let d = self.distance_to(query, nb as usize);
                let cand = Candidate { dist: d, id: nb };

                if w.len() < beam_width {
                    w.push(cand);
                    frontier.push(Reverse(cand));
                } else if d < w.peek().unwrap().dist {
                    w.pop();
                    w.push(cand);
                    frontier.push(Reverse(cand));
                }
            }
        }

        let mut results: Vec<_> = w.into_vec();
        results.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        results.truncate(k);
        results.into_iter().map(|c| (c.id, c.dist)).collect()
    }

    /// search but only return neighbor ids
    pub fn search(&self, query: &[T], k: usize, beam_width: usize) -> Vec<u32> {
        self.search_with_dists(query, k, beam_width)
            .into_iter()
            .map(|(id, _dist)| id)
            .collect()
    }

    /// Gets the neighbors of a node from the (fixed-degree) adjacency region
    fn get_neighbors(&self, node_id: u32) -> &[u32] {
        let offset = self.adjacency_offset + (node_id as u64 * self.max_degree as u64 * 4);
        let start = offset as usize;
        let end = start + (self.max_degree * 4);
        let bytes = &self.mmap[start..end];
        bytemuck::cast_slice(bytes)
    }

    /// Computes distance between `query` and vector `idx`
    fn distance_to(&self, query: &[T], idx: usize) -> f32 {
        let elem_sz = std::mem::size_of::<T>();
        let offset = self.vectors_offset + (idx as u64 * self.dim as u64 * elem_sz as u64);
        let start = offset as usize;
        let end = start + (self.dim * elem_sz);
        let bytes = &self.mmap[start..end];
        let vector: &[T] = bytemuck::cast_slice(bytes);
        self.dist.eval(query, vector)
    }

    /// Gets a vector from the index
    pub fn get_vector(&self, idx: usize) -> Vec<T> {
        let elem_sz = std::mem::size_of::<T>();
        let offset = self.vectors_offset + (idx as u64 * self.dim as u64 * elem_sz as u64);
        let start = offset as usize;
        let end = start + (self.dim * elem_sz);
        let bytes = &self.mmap[start..end];
        let vector: &[T] = bytemuck::cast_slice(bytes);
        vector.to_vec()
    }
}

/// Calculates the medoid using flat contiguous storage.
fn calculate_medoid<T, D>(vectors: &FlatVectors<T>, dist: D) -> usize
where
    T: bytemuck::Pod + Copy + Send + Sync,
    D: Distance<T> + Copy + Sync,
{
    let n = vectors.n;
    let k = 8.min(n);
    let mut rng = thread_rng();
    let pivots: Vec<usize> = (0..k).map(|_| rng.gen_range(0..n)).collect();

    let (best_idx, _best_score) = (0..n)
        .into_par_iter()
        .map(|i| {
            let vi = vectors.row(i);
            let score: f32 = pivots.iter().map(|&p| dist.eval(vi, vectors.row(p))).sum();
            (i, score)
        })
        .reduce(|| (0usize, f32::MAX), |a, b| if a.1 <= b.1 { a } else { b });

    best_idx
}

/// Build Vamana-like graph using:
/// - flat contiguous vectors
/// - per-worker reusable scratch
/// - dense visited marks
/// - ordered beams instead of BinaryHeap for build-time greedy search
/// - batched parallel symmetrization-and-repruning
fn build_vamana_graph<T, D>(
    vectors: &FlatVectors<T>,
    max_degree: usize,
    build_beam_width: usize,
    alpha: f32,
    passes: usize,
    extra_seeds: usize,
    dist: D,
    medoid_id: u32,
) -> Vec<Vec<u32>>
where
    T: bytemuck::Pod + Copy + Send + Sync,
    D: Distance<T> + Copy + Sync,
{
    let n = vectors.n;
    let mut graph = vec![Vec::<u32>::new(); n];

    // Random R-out directed graph bootstrap
    {
        let mut rng = thread_rng();
        let target = max_degree.min(n.saturating_sub(1));

        for i in 0..n {
            let mut s = HashSet::with_capacity(target);
            while s.len() < target {
                let nb = rng.gen_range(0..n);
                if nb != i {
                    s.insert(nb as u32);
                }
            }
            graph[i] = s.into_iter().collect();
        }
    }

    let passes = passes.max(1);
    let mut rng = thread_rng();

    for pass_idx in 0..passes {
        let pass_alpha = if passes == 1 {
            alpha
        } else if pass_idx == 0 {
            1.0
        } else {
            alpha
        };

        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);

        // Important: practical incremental insertion
        for &u in &order {
            let mut scratch = BuildScratch::new(n, build_beam_width, max_degree, extra_seeds);
            scratch.candidates.clear();

            // Legacy DiskANN-style candidate pool starts from search results,
            // not whole V. We also include current adjacency as a cheap prior.
            for &nb in &graph[u] {
                let d = dist.eval(vectors.row(u), vectors.row(nb as usize));
                scratch.candidates.push((nb, d));
            }

            // Seeds: medoid + distinct random starts
            scratch.seeds.clear();
            scratch.seeds.push(medoid_id as usize);
            while scratch.seeds.len() < 1 + extra_seeds {
                let s = rng.gen_range(0..n);
                if !scratch.seeds.contains(&s) {
                    scratch.seeds.push(s);
                }
            }

            let seeds = scratch.seeds.clone();
            for start in seeds {
                greedy_search_visited_collect(
                    vectors.row(u),
                    vectors,
                    &graph,
                    start,
                    build_beam_width,
                    dist,
                    &mut scratch,
                );

                for i in 0..scratch.visited_ids.len() {
                    scratch
                        .candidates
                        .push((scratch.visited_ids[i], scratch.visited_dists[i]));
                }
            }

            // Deduplicate by id, keep best distance
            scratch.candidates.sort_by(|a, b| a.0.cmp(&b.0));
            scratch.candidates.dedup_by(|a, b| {
                if a.0 == b.0 {
                    if a.1 < b.1 {
                        *b = *a;
                    }
                    true
                } else {
                    false
                }
            });

            let pruned = prune_neighbors(
                u,
                &scratch.candidates,
                vectors,
                max_degree,
                pass_alpha,
                dist,
            );

            // Set u's outgoing list
            graph[u] = pruned.clone();

            // Reverse insertion with slack-triggered overflow re-pruning
            inter_insert_with_slack(
                &mut graph,
                u as u32,
                &pruned,
                vectors,
                max_degree,
                pass_alpha,
                dist,
            );
        }
    }

    // Final cleanup: exactly the practical thing to keep.
    graph
        .into_iter()
        .enumerate()
        .map(|(u, neigh)| {
            if neigh.len() <= max_degree {
                return neigh;
            }
            let mut ids = neigh;
            ids.sort_unstable();
            ids.dedup();

            let pool: Vec<(u32, f32)> = ids
                .into_iter()
                .filter(|&id| id as usize != u)
                .map(|id| (id, dist.eval(vectors.row(u), vectors.row(id as usize))))
                .collect();

            prune_neighbors(u, &pool, vectors, max_degree, alpha, dist)
        })
        .collect()
}

/// Insert reverse edge `src -> dst` in the practical DiskANN style:
/// - if dst's degree is still below slack * R, just append
/// - otherwise, gather dst's current neighbors + src, deduplicate, and re-prune dst
fn inter_insert_with_slack<T, D>(
    graph: &mut [Vec<u32>],
    src: u32,
    pruned_list: &[u32],
    vectors: &FlatVectors<T>,
    max_degree: usize,
    alpha: f32,
    dist: D,
) where
    T: bytemuck::Pod + Copy + Send + Sync,
    D: Distance<T> + Copy,
{
    let slack_limit = ((GRAPH_SLACK_FACTOR * max_degree as f32).ceil() as usize).max(max_degree);

    for &dst in pruned_list {
        let dst_usize = dst as usize;
        let src_usize = src as usize;

        if dst_usize == src_usize {
            continue;
        }

        // already present -> nothing to do
        if graph[dst_usize].contains(&src) {
            continue;
        }

        if graph[dst_usize].len() < slack_limit {
            graph[dst_usize].push(src);
            continue;
        }

        // overflow: rebuild candidate pool and prune only this destination node
        let mut ids = graph[dst_usize].clone();
        ids.push(src);
        ids.sort_unstable();
        ids.dedup();

        let pool: Vec<(u32, f32)> = ids
            .into_iter()
            .filter(|&id| id as usize != dst_usize)
            .map(|id| (id, dist.eval(vectors.row(dst_usize), vectors.row(id as usize))))
            .collect();

        graph[dst_usize] = prune_neighbors(dst_usize, &pool, vectors, max_degree, alpha, dist);
    }
}

/// Build-time greedy search:
/// - dense visited marks instead of HashMap/HashSet
/// - visited_ids + visited_dists instead of recomputing distances later
/// - ordered beams instead of BinaryHeap
///
/// Output is written into `scratch.visited_ids` and `scratch.visited_dists`.
fn greedy_search_visited_collect<T, D>(
    query: &[T],
    vectors: &FlatVectors<T>,
    graph: &[Vec<u32>],
    start_id: usize,
    beam_width: usize,
    dist: D,
    scratch: &mut BuildScratch,
) where
    T: bytemuck::Pod + Copy + Send + Sync,
    D: Distance<T> + Copy,
{
    scratch.reset_search();

    let start_dist = dist.eval(query, vectors.row(start_id));
    let start = Candidate {
        dist: start_dist,
        id: start_id as u32,
    };

    scratch.frontier.insert_unbounded(start);
    scratch.work.insert_capped(start, beam_width);
    scratch.mark_with_dist(start_id, start_dist);

    while !scratch.frontier.is_empty() {
        let best = scratch.frontier.best().unwrap();
        if scratch.work.len() >= beam_width {
            if let Some(worst) = scratch.work.worst() {
                if best.dist >= worst.dist {
                    break;
                }
            }
        }

        let cur = scratch.frontier.pop_best().unwrap();

        for &nb in &graph[cur.id as usize] {
            let nb_usize = nb as usize;
            if scratch.is_marked(nb_usize) {
                continue;
            }

            let d = dist.eval(query, vectors.row(nb_usize));
            scratch.mark_with_dist(nb_usize, d);

            let cand = Candidate { dist: d, id: nb };

            if scratch.work.len() < beam_width {
                scratch.work.insert_unbounded(cand);
                scratch.frontier.insert_unbounded(cand);
            } else if let Some(worst) = scratch.work.worst() {
                if d < worst.dist {
                    scratch.work.insert_capped(cand, beam_width);
                    scratch.frontier.insert_unbounded(cand);
                }
            }
        }
    }
}

/// α-pruning
fn prune_neighbors<T, D>(
    node_id: usize,
    candidates: &[(u32, f32)],
    vectors: &FlatVectors<T>,
    max_degree: usize,
    alpha: f32,
    dist: D,
) -> Vec<u32>
where
    T: bytemuck::Pod + Copy + Send + Sync,
    D: Distance<T> + Copy,
{
    if candidates.is_empty() || max_degree == 0 {
        return Vec::new();
    }

    // Sort by distance from node_id, nearest first.
    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));

    // Remove self and duplicate ids while keeping the best (nearest) occurrence.
    let mut uniq = Vec::<(u32, f32)>::with_capacity(sorted.len());
    let mut last_id: Option<u32> = None;
    for &(cand_id, cand_dist) in &sorted {
        if cand_id as usize == node_id {
            continue;
        }
        if last_id == Some(cand_id) {
            continue;
        }
        uniq.push((cand_id, cand_dist));
        last_id = Some(cand_id);
    }

    let mut pruned = Vec::<u32>::with_capacity(max_degree);

    // Pure robust pruning: DO NOT backfill rejected candidates.
    for &(cand_id, cand_dist_to_node) in &uniq {
        let mut occluded = false;

        for &sel_id in &pruned {
            let d_cand_sel = dist.eval(
                vectors.row(cand_id as usize),
                vectors.row(sel_id as usize),
            );

            // If selected neighbor sel_id "covers" cand_id, reject cand_id.
            if alpha * d_cand_sel <= cand_dist_to_node {
                occluded = true;
                break;
            }
        }

        if !occluded {
            pruned.push(cand_id);
            if pruned.len() >= max_degree {
                break;
            }
        }
    }

    pruned
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndists::dist::{DistCosine, DistL2};
    use rand::Rng;
    use std::fs;

    fn euclid(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn test_small_index_l2() {
        let path = "test_small_l2.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];

        let index = DiskANN::<f32, DistL2>::build_index_default(&vectors, DistL2, path).unwrap();

        let q = vec![0.1, 0.1];
        let nns = index.search(&q, 3, 8);
        assert_eq!(nns.len(), 3);

        let v = index.get_vector(nns[0] as usize);
        assert!(euclid(&q, &v) < 1.0);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_cosine() {
        let path = "test_cosine.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
        ];

        let index =
            DiskANN::<f32, DistCosine>::build_index_default(&vectors, DistCosine, path).unwrap();

        let q = vec![2.0, 0.0, 0.0];
        let nns = index.search(&q, 2, 8);
        assert_eq!(nns.len(), 2);

        let v = index.get_vector(nns[0] as usize);
        let dot = v.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>();
        let n1 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n2 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (n1 * n2);
        assert!(cos > 0.7);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_persistence_and_open() {
        let path = "test_persist.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        {
            let _idx =
                DiskANN::<f32, DistL2>::build_index_default(&vectors, DistL2, path).unwrap();
        }

        let idx2 = DiskANN::<f32, DistL2>::open_index_default_metric(path).unwrap();
        assert_eq!(idx2.num_vectors, 4);
        assert_eq!(idx2.dim, 2);

        let q = vec![0.9, 0.9];
        let res = idx2.search(&q, 2, 8);
        assert_eq!(res[0], 3);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_grid_connectivity() {
        let path = "test_grid.db";
        let _ = fs::remove_file(path);

        let mut vectors = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                vectors.push(vec![i as f32, j as f32]);
            }
        }

        let index = DiskANN::<f32, DistL2>::build_index_with_params(
            &vectors,
            DistL2,
            path,
            DiskAnnParams {
                max_degree: 4,
                build_beam_width: 64,
                alpha: 1.5,
                passes: DISKANN_DEFAULT_PASSES,
                extra_seeds: DISKANN_DEFAULT_EXTRA_SEEDS,
            },
        )
        .unwrap();

        for target in 0..vectors.len() {
            let q = &vectors[target];
            let nns = index.search(q, 10, 32);
            if !nns.contains(&(target as u32)) {
                let v = index.get_vector(nns[0] as usize);
                assert!(euclid(q, &v) < 2.0);
            }
            for &nb in nns.iter().take(5) {
                let v = index.get_vector(nb as usize);
                assert!(euclid(q, &v) < 5.0);
            }
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_medium_random() {
        let path = "test_medium.db";
        let _ = fs::remove_file(path);

        let n = 200usize;
        let d = 32usize;
        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..d).map(|_| rng.r#gen::<f32>()).collect())
            .collect();

        let index = DiskANN::<f32, DistL2>::build_index_with_params(
            &vectors,
            DistL2,
            path,
            DiskAnnParams {
                max_degree: 32,
                build_beam_width: 128,
                alpha: 1.2,
                passes: DISKANN_DEFAULT_PASSES,
                extra_seeds: DISKANN_DEFAULT_EXTRA_SEEDS,
            },
        )
        .unwrap();

        let q: Vec<f32> = (0..d).map(|_| rng.r#gen::<f32>()).collect();
        let res = index.search(&q, 10, 64);
        assert_eq!(res.len(), 10);

        let dists: Vec<f32> = res
            .iter()
            .map(|&id| {
                let v = index.get_vector(id as usize);
                euclid(&q, &v)
            })
            .collect();
        let mut sorted = dists.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(dists, sorted);

        let _ = fs::remove_file(path);
    }
}
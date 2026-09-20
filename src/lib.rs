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

mod mmap_dynamic;
pub use mmap_dynamic::DeleteStats;

/// Padding sentinel for adjacency slots (avoid colliding with node 0).
const PAD_U32: u32 = u32::MAX;

/// Defaults for in-memory DiskANN builds
pub const DISKANN_DEFAULT_MAX_DEGREE: usize = 64;
pub const DISKANN_DEFAULT_BUILD_BEAM: usize = 128;
pub const DISKANN_DEFAULT_ALPHA: f32 = 1.2;
/// Default number of extra random seeds per node during graph build
pub const DISKANN_DEFAULT_EXTRA_SEEDS: usize = 1;

/// Practical DiskANN-style slack before reverse-neighbor re-pruning.
/// Legacy C++ DiskANN allows reverse lists to grow to about GRAPH_SLACK_FACTOR * R
/// before triggering prune, instead of pruning immediately at R.
const GRAPH_SLACK_FACTOR: f32 = 1.3;

/// Maximum candidate pool considered by RobustPrune, matching DiskANN's default.
const MAX_OCCLUSION_SIZE: usize = 750;

/// Number of nodes processed together in one micro-batch during graph build.
///
/// Smaller values:
/// - are closer to true incremental insertion
/// - usually improve faithfulness to practical Vamana behavior
/// - but are slower
///
/// Larger values:
/// - are faster
/// - but behave more like a batched graph rebuild
///
/// Recommended starting values:
/// - 128 for better quality
/// - 256 for a balanced tradeoff
/// - 512 for faster builds
const MICRO_BATCH_CHUNK_SIZE: usize = 256;

/// Optional bag of knobs if you want to override just a few.
#[derive(Clone, Copy, Debug)]
pub struct DiskAnnParams {
    pub max_degree: usize,
    pub build_beam_width: usize,
    pub alpha: f32,
    /// Extra random seeds per node during graph construction (>=0).
    pub extra_seeds: usize,
}

impl Default for DiskAnnParams {
    fn default() -> Self {
        Self {
            max_degree: DISKANN_DEFAULT_MAX_DEGREE,
            build_beam_width: DISKANN_DEFAULT_BUILD_BEAM,
            alpha: DISKANN_DEFAULT_ALPHA,
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
pub(crate) struct Candidate {
    dist: f32,
    id: u32,
}

/// One beam-search implementation shared by legacy static and dynamic storage.
/// Storage adapters provide distances and the currently visible neighbor view.
pub(crate) fn graph_search(
    start_id: u32,
    beam_width: usize,
    mut distance: impl FnMut(u32) -> f32,
    mut neighbors: impl FnMut(u32) -> Vec<u32>,
) -> Vec<Candidate> {
    let beam_width = beam_width.max(1);
    let start = Candidate {
        dist: distance(start_id),
        id: start_id,
    };
    let mut visited = HashSet::from([start_id]);
    let mut frontier = BinaryHeap::from([Reverse(start)]);
    let mut work = BinaryHeap::from([start]);

    while let Some(Reverse(best)) = frontier.peek().copied() {
        if work.len() >= beam_width && best.dist >= work.peek().unwrap().dist {
            break;
        }
        let Reverse(current) = frontier.pop().unwrap();
        for neighbor in neighbors(current.id) {
            if !visited.insert(neighbor) {
                continue;
            }
            let candidate = Candidate {
                dist: distance(neighbor),
                id: neighbor,
            };
            if work.len() < beam_width {
                work.push(candidate);
                frontier.push(Reverse(candidate));
            } else if candidate.dist < work.peek().unwrap().dist {
                work.pop();
                work.push(candidate);
                frontier.push(Reverse(candidate));
            }
        }
    }

    let mut results = work.into_vec();
    results.sort_by(|a, b| a.dist.total_cmp(&b.dist));
    results
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

#[derive(Debug)]
struct IncrementalInsertScratch {
    build: BuildScratch,
}

impl IncrementalInsertScratch {
    fn new(n: usize, beam_width: usize, max_degree: usize, extra_seeds: usize) -> Self {
        Self {
            build: BuildScratch::new(n, beam_width, max_degree, extra_seeds),
        }
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
    mmap: Option<Mmap>,

    /// Dynamic-v2 storage when this index was opened or built for updates.
    dynamic: Option<mmap_dynamic::MmapDynamicDiskANN<T, D>>,

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
    fn from_dynamic(dynamic: mmap_dynamic::MmapDynamicDiskANN<T, D>, dist: D) -> Self {
        Self {
            dim: dynamic.dim(),
            num_vectors: dynamic.len(),
            max_degree: dynamic.max_degree(),
            distance_name: dynamic.distance_name().to_string(),
            medoid_id: dynamic.medoid_id(),
            vectors_offset: 0,
            adjacency_offset: 0,
            mmap: None,
            dynamic: Some(dynamic),
            dist,
            _phantom: PhantomData,
        }
    }

    /// Build with default parameters: (M=64, L=128, alpha=1.2, extra_seeds=1).
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
            mmap: Some(mmap),
            dynamic: None,
            dist,
            _phantom: PhantomData,
        })
    }

    /// Starts a transient update session from a static index.
    /// Commit it with `commit_updates_to_static`; the work file is not a
    /// persistent index format and is never accepted by `open_index_with`.
    pub fn begin_updates(
        self,
        capacity: usize,
        alpha: f32,
        path: &str,
    ) -> Result<Self, DiskAnnError> {
        if self.dynamic.is_some() {
            return Err(DiskAnnError::IndexError(
                "source index is already dynamic".into(),
            ));
        }
        let dist = self.dist;
        let dynamic =
            mmap_dynamic::MmapDynamicDiskANN::create_from_static(&self, capacity, alpha, path)?;
        Ok(Self::from_dynamic(dynamic, dist))
    }

    /// True while this handle is a transient update session.
    pub fn is_updating(&self) -> bool {
        self.dynamic.is_some()
    }

    /// Number of addressable slots. Static-v1 capacity equals its vector count.
    pub fn capacity(&self) -> usize {
        self.dynamic
            .as_ref()
            .map_or(self.num_vectors, |dynamic| dynamic.capacity())
    }

    /// Inserts one vector during a transient update session.
    pub fn insert(&mut self, vector: Vec<T>, beam: usize) -> Result<u32, DiskAnnError> {
        let dynamic = self
            .dynamic
            .as_mut()
            .ok_or_else(|| DiskAnnError::IndexError("begin_updates must be called first".into()))?;
        let id = dynamic.insert(vector, beam)?;
        self.num_vectors = dynamic.len();
        Ok(id)
    }

    /// Inserts a batch during a transient update session.
    pub fn insert_batch(
        &mut self,
        vectors: Vec<Vec<T>>,
        beam: usize,
    ) -> Result<Vec<u32>, DiskAnnError> {
        let dynamic = self
            .dynamic
            .as_mut()
            .ok_or_else(|| DiskAnnError::IndexError("begin_updates must be called first".into()))?;
        let ids = dynamic.insert_batch(vectors, beam)?;
        self.num_vectors = dynamic.len();
        Ok(ids)
    }

    /// Deletes one node and repairs its neighborhood with MERIT.
    pub fn delete(&mut self, id: u32) -> Result<Option<DeleteStats>, DiskAnnError> {
        let mut stats = self.delete_batch(&[id])?;
        Ok(stats.pop())
    }

    /// Deletes one node with explicitly configured MERIT repair parameters.
    pub fn delete_with_params(
        &mut self,
        id: u32,
        repair_beam: usize,
        repair_degree: usize,
    ) -> Result<Option<DeleteStats>, DiskAnnError> {
        let mut stats = self.delete_batch_with_params(&[id], repair_beam, repair_degree)?;
        Ok(stats.pop())
    }

    /// Deletes a batch with MERIT defaults: repair beam `2R` and `k_r = 2`.
    pub fn delete_batch(&mut self, ids: &[u32]) -> Result<Vec<DeleteStats>, DiskAnnError> {
        self.delete_batch_with_params(ids, 2 * self.max_degree, 2)
    }

    /// Deletes a batch with explicitly configured MERIT repair parameters.
    pub fn delete_batch_with_params(
        &mut self,
        ids: &[u32],
        repair_beam: usize,
        repair_degree: usize,
    ) -> Result<Vec<DeleteStats>, DiskAnnError> {
        let dynamic = self
            .dynamic
            .as_mut()
            .ok_or_else(|| DiskAnnError::IndexError("begin_updates must be called first".into()))?;
        let stats = dynamic.delete_batch(ids, repair_beam, repair_degree);
        self.num_vectors = dynamic.len();
        Ok(stats)
    }

    /// Flushes pending update-workspace writes. Static handles are read-only.
    pub fn flush(&self) -> Result<(), DiskAnnError> {
        if let Some(dynamic) = &self.dynamic {
            dynamic.flush()?;
        }
        Ok(())
    }

    /// Finishes an update session and commits it as an ordinary static index.
    ///
    /// Only live vectors and live, version-matching edges are written. IDs are
    /// compacted to `0..num_vectors`; the returned tuple contains the ordinary
    /// static search index and the required old-to-new ID mapping.
    pub fn commit_updates_to_static(
        self,
        path: &str,
    ) -> Result<(Self, Vec<Option<u32>>), DiskAnnError> {
        let dynamic = self.dynamic.as_ref().ok_or_else(|| {
            DiskAnnError::IndexError("commit requires a dynamic update session".into())
        })?;
        let (vectors, graph, medoid_id, id_map) = dynamic.static_snapshot();
        let work_path = dynamic.work_path().to_path_buf();
        if vectors.is_empty() {
            return Err(DiskAnnError::IndexError(
                "cannot commit an empty static index".into(),
            ));
        }
        let temporary = format!("{path}.static-commit-{}", std::process::id());
        let dist = self.dist;
        let result = write_static_graph(
            &vectors,
            &graph,
            self.max_degree,
            medoid_id,
            &self.distance_name,
            &temporary,
        );
        drop(self);
        result?;
        std::fs::rename(&temporary, path)?;
        if work_path != std::path::Path::new(path) {
            let _ = std::fs::remove_file(work_path);
        }
        Ok((Self::open_index_with(path, dist)?, id_map))
    }
}

fn write_static_graph<T: bytemuck::Pod>(
    vectors: &[Vec<T>],
    graph: &[Vec<u32>],
    max_degree: usize,
    medoid_id: u32,
    distance_name: &str,
    path: &str,
) -> Result<(), DiskAnnError> {
    let dim = vectors[0].len();
    if vectors.iter().any(|vector| vector.len() != dim) || graph.len() != vectors.len() {
        return Err(DiskAnnError::IndexError(
            "invalid static snapshot dimensions".into(),
        ));
    }
    let vectors_offset = 1024 * 1024u64;
    let adjacency_offset = vectors_offset + (vectors.len() * dim * std::mem::size_of::<T>()) as u64;
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(path)?;
    file.seek(SeekFrom::Start(vectors_offset))?;
    for vector in vectors {
        file.write_all(bytemuck::cast_slice(vector))?;
    }
    file.seek(SeekFrom::Start(adjacency_offset))?;
    for neighbors in graph {
        let mut row = neighbors
            .iter()
            .copied()
            .take(max_degree)
            .collect::<Vec<_>>();
        row.resize(max_degree, PAD_U32);
        file.write_all(bytemuck::cast_slice(&row))?;
    }
    let metadata = Metadata {
        dim,
        num_vectors: vectors.len(),
        max_degree,
        medoid_id,
        vectors_offset,
        adjacency_offset,
        elem_size: std::mem::size_of::<T>() as u8,
        distance_name: distance_name.to_owned(),
    };
    let bytes = bincode::serialize(&metadata)?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&(bytes.len() as u64).to_le_bytes())?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
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
    /// * `extra_seeds` - Extra random seeds per node (>=0)
    /// * `dist` - Any `anndists::Distance<T>`
    /// * `file_path` - Path of index file
    pub fn build_index(
        vectors: &[Vec<T>],
        max_degree: usize,
        build_beam_width: usize,
        alpha: f32,
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
            mmap: Some(mmap),
            dynamic: None,
            dist,
            _phantom: PhantomData,
        })
    }

    /// Searches the index for nearest neighbors using a best-first beam search.
    /// Termination rule: continue while the best frontier can still improve the worst in working set.
    pub fn search_with_dists(&self, query: &[T], k: usize, beam_width: usize) -> Vec<(u32, f32)> {
        if let Some(dynamic) = &self.dynamic {
            return dynamic.search_with_dists(query, k, beam_width);
        }
        assert_eq!(
            query.len(),
            self.dim,
            "Query dim {} != index dim {}",
            query.len(),
            self.dim
        );

        let mut results = graph_search(
            self.medoid_id,
            beam_width,
            |id| self.distance_to(query, id as usize),
            |id| {
                self.get_neighbors(id)
                    .iter()
                    .copied()
                    .filter(|neighbor| *neighbor != PAD_U32)
                    .collect()
            },
        );
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
        let bytes = &self.mmap.as_ref().expect("static mmap missing")[start..end];
        bytemuck::cast_slice(bytes)
    }

    /// Computes distance between `query` and vector `idx`
    fn distance_to(&self, query: &[T], idx: usize) -> f32 {
        let elem_sz = std::mem::size_of::<T>();
        let offset = self.vectors_offset + (idx as u64 * self.dim as u64 * elem_sz as u64);
        let start = offset as usize;
        let end = start + (self.dim * elem_sz);
        let bytes = &self.mmap.as_ref().expect("static mmap missing")[start..end];
        let vector: &[T] = bytemuck::cast_slice(bytes);
        self.dist.eval(query, vector)
    }

    /// Gets a vector from the index
    pub fn get_vector(&self, idx: usize) -> Vec<T> {
        if let Some(dynamic) = &self.dynamic {
            return dynamic.get_vector(idx);
        }
        let elem_sz = std::mem::size_of::<T>();
        let offset = self.vectors_offset + (idx as u64 * self.dim as u64 * elem_sz as u64);
        let start = offset as usize;
        let end = start + (self.dim * elem_sz);
        let bytes = &self.mmap.as_ref().expect("static mmap missing")[start..end];
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

fn dedup_keep_best_by_id_in_place(cands: &mut Vec<(u32, f32)>) {
    if cands.is_empty() {
        return;
    }

    cands.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));

    let mut write = 0usize;
    for read in 0..cands.len() {
        if write == 0 || cands[read].0 != cands[write - 1].0 {
            cands[write] = cands[read];
            write += 1;
        }
    }
    cands.truncate(write);
}

/// Merge one micro-batch of newly computed outgoing neighbor lists back into the graph.
/// Mark all nodes in the chunk as affected.
/// Count reverse-edge insertions implied by the chunk.
/// Build a CSR-style flat incoming buffer.
/// Commit chunk outgoing lists.
/// Re-prune only the affected nodes.
fn merge_chunk_updates_into_graph_reuse<T, D>(
    graph: &mut [Vec<u32>],
    chunk_nodes: &[usize],
    chunk_pruned: &[Vec<u32>],
    vectors: &FlatVectors<T>,
    max_degree: usize,
    slack_limit: usize,
    alpha: f32,
    dist: D,
    merge: &mut MergeScratch,
) where
    T: bytemuck::Pod + Copy + Send + Sync,
    D: Distance<T> + Copy + Sync,
{
    merge.reset();

    // Mark chunk nodes as affected.
    for &u in chunk_nodes {
        merge.mark_affected(u);
    }

    // Count reverse-edge insertions and mark touched destinations.
    let mut total_incoming = 0usize;

    for (local_idx, &u) in chunk_nodes.iter().enumerate() {
        for &dst in &chunk_pruned[local_idx] {
            let dst_usize = dst as usize;
            if dst_usize == u {
                continue;
            }

            merge.mark_affected(dst_usize);
            merge.incoming_counts[dst_usize] += 1;
            total_incoming += 1;
        }
    }

    // Build CSR offsets only for affected nodes.
    merge.affected_nodes.sort_unstable();

    let mut running = 0usize;
    for &u in &merge.affected_nodes {
        merge.incoming_offsets[u] = running;
        running += merge.incoming_counts[u];
        merge.incoming_offsets[u + 1] = running;
    }

    merge.incoming_flat.resize(total_incoming, PAD_U32);

    // Initialize write cursors.
    for &u in &merge.affected_nodes {
        merge.incoming_write[u] = merge.incoming_offsets[u];
    }

    // Fill CSR incoming buffer.
    for (local_idx, &u) in chunk_nodes.iter().enumerate() {
        for &dst in &chunk_pruned[local_idx] {
            let dst_usize = dst as usize;
            if dst_usize == u {
                continue;
            }

            let pos = merge.incoming_write[dst_usize];
            merge.incoming_flat[pos] = u as u32;
            merge.incoming_write[dst_usize] += 1;
        }
    }

    // Commit chunk outgoing lists first.
    for (local_idx, &u) in chunk_nodes.iter().enumerate() {
        graph[u] = chunk_pruned[local_idx].clone();
    }

    // Hybrid slack-aware microbatch merge:
    // keep merged lists under slack, reprune only overflowed ones.
    let affected = merge.affected_nodes.clone();

    let updated_pairs: Vec<(usize, Vec<u32>)> = affected
        .into_par_iter()
        .map(|u| {
            let start = merge.incoming_offsets[u];
            let end = merge.incoming_offsets[u + 1];

            let mut ids: Vec<u32> = Vec::with_capacity(graph[u].len() + (end - start));

            // Current adjacency after chunk commit.
            ids.extend_from_slice(&graph[u]);

            // Reverse insertions from this chunk.
            if start < end {
                ids.extend_from_slice(&merge.incoming_flat[start..end]);
            }

            // Remove self-loops / padding.
            ids.retain(|&id| id != PAD_U32 && id as usize != u);

            // Deduplicate.
            ids.sort_unstable();
            ids.dedup();

            if ids.is_empty() {
                return (u, Vec::new());
            }

            // Under slack: keep as-is.
            if ids.len() <= slack_limit {
                return (u, ids);
            }

            // Overflow: score and prune back to max_degree.
            let mut pool = Vec::<(u32, f32)>::with_capacity(ids.len());
            for id in ids {
                let d = dist.eval(vectors.row(u), vectors.row(id as usize));
                pool.push((id, d));
            }

            let pruned = prune_neighbors(u, &pool, vectors, max_degree, alpha, dist);
            (u, pruned)
        })
        .collect();

    for (u, neigh) in updated_pairs {
        graph[u] = neigh;
    }

    // Cleanup touched metadata.
    for &u in &merge.affected_nodes {
        merge.incoming_counts[u] = 0;
        merge.incoming_offsets[u + 1] = 0;
    }
}

/// Reusable scratch buffers for micro-batch merge.
/// This avoids rebuilding `Vec<Vec<u32>>` for incoming reverse edges on every chunk.
/// Instead, incoming reverse edges are accumulated into a CSR-like flat buffer.
///
/// Layout:
/// - incoming_counts[u] = number of reverse edges targeting node u in this chunk
/// - incoming_offsets[u]..incoming_offsets[u+1] is u's segment in incoming_flat
/// - affected_nodes stores exactly the nodes touched in this chunk, so we do not
///   scan all `n` nodes during merge.
#[derive(Debug)]
struct MergeScratch {
    incoming_counts: Vec<usize>,
    incoming_offsets: Vec<usize>,
    incoming_write: Vec<usize>,
    incoming_flat: Vec<u32>,

    affected_marks: Vec<u32>,
    affected_epoch: u32,
    affected_nodes: Vec<usize>,
}

impl MergeScratch {
    fn new(n: usize) -> Self {
        Self {
            incoming_counts: vec![0usize; n],
            incoming_offsets: vec![0usize; n + 1],
            incoming_write: vec![0usize; n],
            incoming_flat: Vec::new(),
            affected_marks: vec![0u32; n],
            affected_epoch: 1,
            affected_nodes: Vec::new(),
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.affected_epoch = self.affected_epoch.wrapping_add(1);
        if self.affected_epoch == 0 {
            self.affected_marks.fill(0);
            self.affected_epoch = 1;
        }
        self.affected_nodes.clear();
        self.incoming_flat.clear();
    }

    #[inline]
    fn mark_affected(&mut self, u: usize) {
        if self.affected_marks[u] != self.affected_epoch {
            self.affected_marks[u] = self.affected_epoch;
            self.affected_nodes.push(u);
            self.incoming_counts[u] = 0;
        }
    }
}

/// Build a Vamana-like graph using a micro-batched practical DiskANN strategy,
/// with reusable scratch both for per-thread search state and for chunk merge state.
fn build_vamana_graph<T, D>(
    vectors: &FlatVectors<T>,
    max_degree: usize,
    build_beam_width: usize,
    alpha: f32,
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
    // Bootstrap with a random R-out directed graph.
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

    let mut rng = thread_rng();
    let slack_limit = ((GRAPH_SLACK_FACTOR * max_degree as f32).ceil() as usize).max(max_degree);

    // Reused across all chunks in the single Vamana construction pass.
    let mut merge_scratch = MergeScratch::new(n);

    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut rng);

    for chunk in order.chunks(MICRO_BATCH_CHUNK_SIZE) {
        let snapshot = &graph;
        // Compute new outgoing lists for this chunk in parallel.
        let chunk_results: Vec<(usize, Vec<u32>)> = chunk
            .par_iter()
            .map_init(
                || IncrementalInsertScratch::new(n, build_beam_width, max_degree, extra_seeds),
                |scratch, &u| {
                    let bs = &mut scratch.build;
                    bs.candidates.clear();

                    // Start from current adjacency.
                    for &nb in &snapshot[u] {
                        let d = dist.eval(vectors.row(u), vectors.row(nb as usize));
                        bs.candidates.push((nb, d));
                    }

                    // Seed list: medoid + distinct random starts.
                    bs.seeds.clear();
                    bs.seeds.push(medoid_id as usize);

                    let mut local_rng = thread_rng();
                    while bs.seeds.len() < 1 + extra_seeds {
                        let s = local_rng.gen_range(0..n);
                        if !bs.seeds.contains(&s) {
                            bs.seeds.push(s);
                        }
                    }

                    let seeds_len = bs.seeds.len();
                    for si in 0..seeds_len {
                        let start = bs.seeds[si];

                        greedy_search_visited_collect(
                            vectors.row(u),
                            vectors,
                            snapshot,
                            start,
                            build_beam_width,
                            dist,
                            bs,
                        );

                        for i in 0..bs.visited_ids.len() {
                            bs.candidates.push((bs.visited_ids[i], bs.visited_dists[i]));
                        }
                    }

                    dedup_keep_best_by_id_in_place(&mut bs.candidates);

                    let pruned =
                        prune_neighbors(u, &bs.candidates, vectors, max_degree, alpha, dist);

                    (u, pruned)
                },
            )
            .collect();

        let mut chunk_nodes = Vec::<usize>::with_capacity(chunk_results.len());
        let mut chunk_pruned = Vec::<Vec<u32>>::with_capacity(chunk_results.len());

        for (u, pruned) in chunk_results {
            chunk_nodes.push(u);
            chunk_pruned.push(pruned);
        }
        // Merge chunk back into graph using reusable CSR-style scratch.
        merge_chunk_updates_into_graph_reuse(
            &mut graph,
            &chunk_nodes,
            &chunk_pruned,
            vectors,
            max_degree,
            slack_limit,
            alpha,
            dist,
            &mut merge_scratch,
        );
    }

    // Final cleanup: enforce bounded degree and deduplication.
    graph
        .into_par_iter()
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

/// Build-time greedy search:
/// - dense visited marks instead of HashMap/HashSet
/// - visited_ids + visited_dists instead of recomputing distances later
/// - ordered beams instead of BinaryHeap
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

/// Vamana RobustPrune with progressive alpha relaxation.
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
    sorted.truncate(MAX_OCCLUSION_SIZE);

    // Remove self and duplicate ids while keeping the nearest occurrence.
    let mut uniq = Vec::<(u32, f32)>::with_capacity(sorted.len());
    let mut seen = HashSet::with_capacity(sorted.len());
    for &(cand_id, cand_dist) in &sorted {
        if cand_id as usize == node_id || !seen.insert(cand_id) {
            continue;
        }
        uniq.push((cand_id, cand_dist));
    }

    if uniq.is_empty() {
        return Vec::new();
    }

    let mut pruned = Vec::<u32>::with_capacity(max_degree);
    let mut occlude_factors = vec![0.0f32; uniq.len()];
    let target_alpha = alpha.max(1.0);
    let increment = target_alpha.min(1.2);
    let mut current_alpha = 1.0f32;

    loop {
        for i in 0..uniq.len() {
            if pruned.len() >= max_degree {
                return pruned;
            }
            if occlude_factors[i] > current_alpha {
                continue;
            }

            let (selected_id, _) = uniq[i];
            occlude_factors[i] = f32::MAX;
            pruned.push(selected_id);

            for j in (i + 1)..uniq.len() {
                if occlude_factors[j] > target_alpha {
                    continue;
                }

                let (candidate_id, candidate_dist) = uniq[j];
                let pair_dist = dist.eval(
                    vectors.row(candidate_id as usize),
                    vectors.row(selected_id as usize),
                );
                let factor = if pair_dist == 0.0 {
                    f32::MAX
                } else {
                    candidate_dist / pair_dist
                };
                occlude_factors[j] = occlude_factors[j].max(factor);
            }
        }

        if current_alpha >= target_alpha {
            break;
        }
        current_alpha = (current_alpha * increment).min(target_alpha);
    }

    pruned
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndists::dist::{DistCosine, DistJaccard, DistL1, DistL2};
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::{fs, time::Instant};

    fn euclid(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    fn exact_recall<T, D>(
        index: &DiskANN<T, D>,
        vectors: &[Vec<T>],
        queries: &[Vec<T>],
        dist: D,
        k: usize,
        beam: usize,
    ) -> f32
    where
        T: bytemuck::Pod + Copy + Send + Sync + 'static,
        D: Distance<T> + Send + Sync + Copy + Clone + 'static,
    {
        let mut hits = 0usize;
        for query in queries {
            let mut truth: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(id, vector)| (id, dist.eval(query, vector)))
                .collect();
            truth.sort_by(|a, b| a.1.total_cmp(&b.1));
            let truth: HashSet<u32> = truth.into_iter().take(k).map(|(id, _)| id as u32).collect();

            hits += index
                .search(query, k, beam)
                .into_iter()
                .filter(|id| truth.contains(id))
                .count();
        }
        hits as f32 / (queries.len() * k) as f32
    }

    fn exact_recall_mapped<T, D>(
        index: &DiskANN<T, D>,
        internal_to_original: &[u32],
        vectors: &[Vec<T>],
        active: &[bool],
        queries: &[Vec<T>],
        dist: D,
    ) -> f32
    where
        T: bytemuck::Pod + Copy + Send + Sync + 'static,
        D: Distance<T> + Send + Sync + Copy + Clone + 'static,
    {
        let mut hits = 0usize;
        for query in queries {
            let mut truth = vectors
                .iter()
                .enumerate()
                .filter(|(id, _)| active[*id])
                .map(|(id, vector)| (id as u32, dist.eval(query, vector)))
                .collect::<Vec<_>>();
            truth.sort_by(|a, b| a.1.total_cmp(&b.1));
            let truth = truth
                .into_iter()
                .take(10)
                .map(|item| item.0)
                .collect::<HashSet<_>>();
            hits += index
                .search(query, 10, 256)
                .into_iter()
                .filter(|id| truth.contains(&internal_to_original[*id as usize]))
                .count();
        }
        hits as f32 / (queries.len() * 10) as f32
    }

    fn merit_clustered_delete<T, D>(
        vectors: &[Vec<T>],
        queries: &[Vec<T>],
        dist: D,
        tag: &str,
    ) -> (f32, f32, f64, f64)
    where
        T: bytemuck::Pod + Copy + Send + Sync + 'static,
        D: Distance<T> + Send + Sync + Copy + Clone + 'static,
    {
        let static_path = format!("test_{tag}_static.db");
        let work_path = format!("test_{tag}.work");
        let rebuilt_path = format!("test_{tag}_rebuilt.db");
        for path in [&static_path, &work_path, &rebuilt_path] {
            let _ = fs::remove_file(path);
        }
        let mut active = vec![true; vectors.len()];
        let deleted = (0..vectors.len())
            .step_by(100)
            .map(|id| id as u32)
            .collect::<Vec<_>>();
        for id in &deleted {
            active[*id as usize] = false;
        }

        let initial =
            DiskANN::build_index_with_params(vectors, dist, &static_path, quality_params())
                .unwrap();
        let started = Instant::now();
        let mut update = initial
            .begin_updates(vectors.len(), 1.2, &work_path)
            .unwrap();
        update.delete_batch(&deleted).unwrap();
        let (committed, old_to_new) = update.commit_updates_to_static(&static_path).unwrap();
        let update_seconds = started.elapsed().as_secs_f64();
        let mut committed_map = vec![u32::MAX; committed.num_vectors];
        for (old, new) in old_to_new.into_iter().enumerate() {
            if let Some(new) = new {
                committed_map[new as usize] = old as u32;
            }
        }

        let surviving_ids = active
            .iter()
            .enumerate()
            .filter_map(|(id, live)| live.then_some(id as u32))
            .collect::<Vec<_>>();
        let surviving = surviving_ids
            .iter()
            .map(|id| vectors[*id as usize].clone())
            .collect::<Vec<_>>();
        let started = Instant::now();
        let rebuilt =
            DiskANN::build_index_with_params(&surviving, dist, &rebuilt_path, quality_params())
                .unwrap();
        let rebuild_seconds = started.elapsed().as_secs_f64();
        let committed_recall =
            exact_recall_mapped(&committed, &committed_map, vectors, &active, queries, dist);
        let rebuilt_recall =
            exact_recall_mapped(&rebuilt, &surviving_ids, vectors, &active, queries, dist);
        drop(committed);
        drop(rebuilt);
        for path in [&static_path, &work_path, &rebuilt_path] {
            let _ = fs::remove_file(path);
        }
        (
            committed_recall,
            rebuilt_recall,
            update_seconds,
            rebuild_seconds,
        )
    }

    fn quality_params() -> DiskAnnParams {
        DiskAnnParams {
            max_degree: 32,
            build_beam_width: 128,
            alpha: 1.2,
            extra_seeds: 1,
        }
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
    fn update_session_commits_as_ordinary_static_index() {
        let static_path = "test_update_commit_static.db";
        let work_path = "test_update_commit_work.db";
        let _ = fs::remove_file(static_path);
        let _ = fs::remove_file(work_path);
        let vectors = (0..200)
            .map(|id| vec![id as f32, (id % 11) as f32])
            .collect::<Vec<_>>();
        let static_index = DiskANN::build_index_default(&vectors, DistL2, static_path).unwrap();
        let mut update = static_index.begin_updates(220, 1.2, work_path).unwrap();
        assert!(update.is_updating());
        assert!(update.delete(40).unwrap().is_some());
        let replacement = vec![40.25, 7.0];
        update.insert(replacement.clone(), 128).unwrap();
        let (committed, id_map) = update.commit_updates_to_static(static_path).unwrap();
        assert_eq!(id_map.len(), 220);
        assert!(!committed.is_updating());
        assert_eq!(committed.num_vectors, 200);
        assert_eq!(committed.search(&replacement, 1, 128).len(), 1);
        drop(committed);

        let reopened = DiskANN::<f32, DistL2>::open_index_with(static_path, DistL2).unwrap();
        assert!(!reopened.is_updating());
        assert_eq!(reopened.num_vectors, 200);
        assert_eq!(reopened.search(&replacement, 1, 128).len(), 1);
        let _ = fs::remove_file(static_path);
        let _ = fs::remove_file(work_path);
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
            let _idx = DiskANN::<f32, DistL2>::build_index_default(&vectors, DistL2, path).unwrap();
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

    #[test]
    fn test_regular_l2_recall() {
        let path = "test_regular_l2_recall.db";
        let _ = fs::remove_file(path);
        let mut rng = StdRng::seed_from_u64(11);
        let vectors: Vec<Vec<f32>> = (0..500)
            .map(|_| (0..24).map(|_| rng.r#gen::<f32>()).collect())
            .collect();
        let queries: Vec<Vec<f32>> = (0..40)
            .map(|_| (0..24).map(|_| rng.r#gen::<f32>()).collect())
            .collect();

        let index =
            DiskANN::build_index_with_params(&vectors, DistL2, path, quality_params()).unwrap();
        let recall = exact_recall(&index, &vectors, &queries, DistL2, 10, 256);
        assert!(recall >= 0.95, "regular L2 recall was {recall}");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_clustered_l2_recall() {
        let path = "test_clustered_l2_recall.db";
        let _ = fs::remove_file(path);
        let mut rng = StdRng::seed_from_u64(23);
        let dim = 24usize;
        let clusters = 12usize;
        let mut vectors = Vec::with_capacity(clusters * 50);
        let mut queries = Vec::with_capacity(clusters * 3);

        for cluster in 0..clusters {
            let mut center = vec![0.0f32; dim];
            center[cluster] = 40.0;
            center[(cluster + 7) % dim] = -25.0;
            for _ in 0..50 {
                vectors.push(
                    center
                        .iter()
                        .map(|value| value + rng.gen_range(-0.4f32..0.4f32))
                        .collect(),
                );
            }
            for _ in 0..3 {
                queries.push(
                    center
                        .iter()
                        .map(|value| value + rng.gen_range(-0.4f32..0.4f32))
                        .collect(),
                );
            }
        }

        let index =
            DiskANN::build_index_with_params(&vectors, DistL2, path, quality_params()).unwrap();
        let recall = exact_recall(&index, &vectors, &queries, DistL2, 10, 256);
        assert!(recall >= 0.95, "clustered L2 recall was {recall}");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_cosine_recall() {
        let path = "test_cosine_recall.db";
        let _ = fs::remove_file(path);
        let mut rng = StdRng::seed_from_u64(37);
        let vectors: Vec<Vec<f32>> = (0..500)
            .map(|_| (0..32).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
            .collect();
        let queries: Vec<Vec<f32>> = (0..40)
            .map(|_| (0..32).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
            .collect();

        let index =
            DiskANN::build_index_with_params(&vectors, DistCosine, path, quality_params()).unwrap();
        let recall = exact_recall(&index, &vectors, &queries, DistCosine, 10, 256);
        assert!(recall >= 0.95, "cosine recall was {recall}");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_jaccard_recall() {
        let path = "test_jaccard_recall.db";
        let _ = fs::remove_file(path);
        let mut rng = StdRng::seed_from_u64(41);
        let vectors: Vec<Vec<u32>> = (0..400)
            .map(|_| (0..32).map(|_| rng.gen_range(0u32..8u32)).collect())
            .collect();
        let queries: Vec<Vec<u32>> = (0..40)
            .map(|_| (0..32).map(|_| rng.gen_range(0u32..8u32)).collect())
            .collect();

        let index = DiskANN::build_index_with_params(&vectors, DistJaccard, path, quality_params())
            .unwrap();
        let recall = exact_recall(&index, &vectors, &queries, DistJaccard, 10, 256);
        assert!(recall >= 0.95, "Jaccard recall was {recall}");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_merit_clustered_l1_delete() {
        let mut rng = StdRng::seed_from_u64(81);
        let mut vectors = Vec::with_capacity(10_000);
        let mut queries = Vec::with_capacity(100);
        for cluster in 0..50 {
            let center = (0..32)
                .map(|dim| (((cluster * 37 + dim * 13) % 101) as f32 - 50.0) * 8.0)
                .collect::<Vec<_>>();
            for _ in 0..200 {
                vectors.push(
                    center
                        .iter()
                        .map(|value| value + rng.gen_range(-0.15f32..0.15))
                        .collect(),
                );
            }
            for _ in 0..2 {
                queries.push(
                    center
                        .iter()
                        .map(|value| value + rng.gen_range(-0.15f32..0.15))
                        .collect(),
                );
            }
        }
        let (updated, rebuilt, update_s, rebuild_s) =
            merit_clustered_delete(&vectors, &queries, DistL1, "merit_clustered_l1");
        println!(
            "clustered L1 MERIT: updated={updated:.6}, rebuilt={rebuilt:.6}, delta={:.6}, update={update_s:.3}s, rebuild={rebuild_s:.3}s",
            updated - rebuilt
        );
        assert!(updated >= 0.95, "clustered L1 MERIT recall was {updated}");
        assert!(
            updated + 0.02 >= rebuilt,
            "clustered L1 delta was {}",
            updated - rebuilt
        );
    }

    #[test]
    fn test_merit_clustered_jaccard_delete() {
        let mut rng = StdRng::seed_from_u64(82);
        let mut vectors = Vec::with_capacity(10_000);
        let mut queries = Vec::with_capacity(100);
        for cluster in 0..50 {
            let mut center = vec![1u32; 64];
            for marker in 0..8 {
                center[(cluster * 11 + marker * 7) % 64] = 30 + (cluster % 7) as u32;
            }
            for _ in 0..200 {
                vectors.push(
                    center
                        .iter()
                        .map(|value| value + rng.gen_range(0u32..3))
                        .collect(),
                );
            }
            for _ in 0..2 {
                queries.push(
                    center
                        .iter()
                        .map(|value| value + rng.gen_range(0u32..3))
                        .collect(),
                );
            }
        }
        let (updated, rebuilt, update_s, rebuild_s) =
            merit_clustered_delete(&vectors, &queries, DistJaccard, "merit_clustered_jaccard");
        println!(
            "clustered Jaccard MERIT: updated={updated:.6}, rebuilt={rebuilt:.6}, delta={:.6}, update={update_s:.3}s, rebuild={rebuild_s:.3}s",
            updated - rebuilt
        );
        assert!(
            updated >= 0.95,
            "clustered Jaccard MERIT recall was {updated}"
        );
        assert!(
            updated + 0.02 >= rebuilt,
            "clustered Jaccard delta was {}",
            updated - rebuilt
        );
    }
}

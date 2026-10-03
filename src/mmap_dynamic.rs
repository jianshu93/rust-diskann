//! Fixed-layout, mmap-backed dynamic DiskANN with MERIT deletion repair.

use super::{Candidate, DiskANN, DiskAnnError, PAD_U32, graph_search};
use anndists::prelude::Distance;
use log::debug;
use memmap2::{MmapMut, MmapOptions};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DATA_OFFSET: u64 = 1024 * 1024;
const MAGIC: [u8; 8] = *b"DYNANN01";
// Match the static Vamana builder's balanced update granularity. A chunk is
// planned against a consistent graph snapshot, then atomically merged before
// the next chunk can use its new routes.
const INSERT_MICRO_BATCH_SIZE: usize = 256;

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct DynamicMetadata {
    magic: [u8; 8],
    dim: usize,
    capacity: usize,
    max_degree: usize,
    alpha: f32,
    medoid_id: u32,
    vectors_offset: u64,
    adjacency_offset: u64,
    edge_versions_offset: u64,
    node_versions_offset: u64,
    valid_offset: u64,
    elem_size: u8,
    distance_name: String,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DeleteStats {
    pub outgoing_seeds: usize,
    pub recovered_in_neighbors: usize,
    pub repair_candidates: usize,
    pub repair_edges_attempted: usize,
}

/// Configuration for routability-aware deletion admission control.
///
/// The guard keeps a small, dispersed landmark pool in memory. Before a batch
/// is changed, it virtually masks the proposed deleted vertices and confirms
/// that the same residual entry point can still reach the selected landmarks
/// with beam search. It is intentionally a lightweight preflight rather than
/// a graph-wide connectivity oracle. [`Default`] is enabled automatically by
/// [`DiskANN::begin_updates`](crate::DiskANN::begin_updates).
#[derive(Clone, Copy, Debug)]
pub struct RoutabilityGuardConfig {
    /// Number of landmark probes required for each delete batch.
    pub landmark_count: usize,
    /// Number of stored landmarks per required probe. Extra landmarks make an
    /// occasional landmark deletion inexpensive to handle.
    pub landmark_pool_multiplier: usize,
    /// Maximum number of evenly sampled live vertices considered when forming
    /// the farthest-first landmark pool.
    pub candidate_sample_size: usize,
    /// Beam width used for the virtual baseline and residual probes.
    pub beam_width: usize,
}

impl Default for RoutabilityGuardConfig {
    fn default() -> Self {
        Self {
            landmark_count: 6,
            landmark_pool_multiplier: 4,
            candidate_sample_size: 1024,
            beam_width: 128,
        }
    }
}

impl RoutabilityGuardConfig {
    fn normalized(self) -> Self {
        Self {
            landmark_count: self.landmark_count.max(1),
            landmark_pool_multiplier: self.landmark_pool_multiplier.max(1),
            candidate_sample_size: self.candidate_sample_size.max(1),
            beam_width: self.beam_width.max(1),
        }
    }
}

/// Current state of the optional routability guard.
#[derive(Clone, Copy, Debug, Default)]
pub struct RoutabilityGuardStatus {
    pub enabled: bool,
    pub landmark_pool_size: usize,
    pub live_landmarks: usize,
    pub landmark_count: usize,
    pub beam_width: usize,
}

/// Result of a virtual routability check for one proposed delete batch.
///
/// A false `admitted` value leaves the graph untouched when used through
/// [`DiskANN::delete_batch_with_admission_control`](crate::DiskANN::delete_batch_with_admission_control).
/// It is a signal to defer, split, or statically rebuild that batch.
#[derive(Clone, Debug, Default)]
pub struct RoutabilityAdmissionReport {
    pub admitted: bool,
    pub requested: usize,
    pub entry: Option<u32>,
    pub checked_landmarks: Vec<u32>,
    pub baseline_reachable_landmarks: usize,
    pub residual_reachable_landmarks: usize,
    pub lost_landmarks: Vec<u32>,
    /// Number of anchors drawn from a fresh small sample because the retained
    /// pool had fewer than `landmark_count` live, non-deleted members.
    pub supplemental_landmarks: usize,
}

/// Outcome of an atomic guarded deletion request.
#[derive(Clone, Debug)]
pub enum GuardedDeleteResult {
    /// The preflight passed and MERIT repair was applied.
    Applied {
        stats: Vec<DeleteStats>,
        report: RoutabilityAdmissionReport,
    },
    /// The preflight failed and the index was not modified.
    Deferred(RoutabilityAdmissionReport),
}

impl GuardedDeleteResult {
    pub fn report(&self) -> &RoutabilityAdmissionReport {
        match self {
            Self::Applied { report, .. } | Self::Deferred(report) => report,
        }
    }

    pub fn was_applied(&self) -> bool {
        matches!(self, Self::Applied { .. })
    }
}

struct RoutabilityGuard {
    config: RoutabilityGuardConfig,
    landmark_pool: Vec<u32>,
}

#[derive(Clone)]
struct DeletePlan {
    id: u32,
    old_version: u16,
    candidates: Vec<u32>,
    stats: DeleteStats,
}

/// One immutable plan for a batch insertion.  The reverse edges are merged by
/// destination after all of these plans have been computed from one snapshot.
struct InsertPlan {
    id: u32,
    outgoing: Vec<(u32, f32)>,
}

/// A fixed-capacity dynamic index. Vector and `u32` adjacency regions have the
/// same row-major mmap layout as `DiskANN`; version and validity arrays follow.
pub(crate) struct MmapDynamicDiskANN<T, D>
where
    T: bytemuck::Pod + Copy + Send + Sync + 'static,
    D: Distance<T> + Send + Sync + Copy + Clone + 'static,
{
    meta: DynamicMetadata,
    mmap: MmapMut,
    work_path: PathBuf,
    dist: D,
    live_count: usize,
    routability_guard: Option<RoutabilityGuard>,
    _marker: PhantomData<T>,
}

impl<T, D> MmapDynamicDiskANN<T, D>
where
    T: bytemuck::Pod + Copy + Send + Sync + 'static,
    D: Distance<T> + Send + Sync + Copy + Clone + 'static,
{
    pub fn create_from_static(
        index: &DiskANN<T, D>,
        capacity: usize,
        alpha: f32,
        path: impl AsRef<Path>,
    ) -> Result<Self, DiskAnnError> {
        let started = Instant::now();
        if capacity < index.num_vectors {
            return Err(DiskAnnError::IndexError(
                "dynamic capacity is smaller than source index".into(),
            ));
        }
        let elem = std::mem::size_of::<T>() as u64;
        let vectors_offset = DATA_OFFSET;
        let adjacency_offset = vectors_offset + capacity as u64 * index.dim as u64 * elem;
        let edge_versions_offset = adjacency_offset + capacity as u64 * index.max_degree as u64 * 4;
        let node_versions_offset =
            edge_versions_offset + capacity as u64 * index.max_degree as u64 * 2;
        let valid_offset = node_versions_offset + capacity as u64 * 2;
        let file_len = valid_offset + capacity as u64;
        let work_path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)?;
        file.set_len(file_len)?;
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        let meta = DynamicMetadata {
            magic: MAGIC,
            dim: index.dim,
            capacity,
            max_degree: index.max_degree,
            alpha,
            medoid_id: index.medoid_id,
            vectors_offset,
            adjacency_offset,
            edge_versions_offset,
            node_versions_offset,
            valid_offset,
            elem_size: std::mem::size_of::<T>() as u8,
            distance_name: std::any::type_name::<D>().to_string(),
        };
        let mut out = Self {
            meta,
            mmap,
            work_path,
            dist: index.dist,
            live_count: index.num_vectors,
            routability_guard: None,
            _marker: PhantomData,
        };
        out.write_metadata()?;

        // The first `num_vectors` vector and adjacency rows are byte-for-byte
        // compatible with the static layout.  Copying the two contiguous
        // regions avoids one allocation and several small mmap writes per node.
        let source = index
            .mmap
            .as_ref()
            .ok_or_else(|| DiskAnnError::IndexError("source index mmap missing".into()))?;
        let vector_bytes = index.num_vectors * index.dim * std::mem::size_of::<T>();
        let source_vectors = index.vectors_offset as usize;
        let destination_vectors = out.meta.vectors_offset as usize;
        out.mmap[destination_vectors..destination_vectors + vector_bytes]
            .copy_from_slice(&source[source_vectors..source_vectors + vector_bytes]);

        let source_adjacency = index.adjacency_offset as usize;
        let copied_adjacency_bytes =
            index.num_vectors * index.max_degree * std::mem::size_of::<u32>();
        let destination_adjacency = out.meta.adjacency_offset as usize;
        let dynamic_adjacency_bytes = capacity * index.max_degree * std::mem::size_of::<u32>();
        // PAD_U32 is all one bits, so this also works when the adjacency start
        // is not naturally aligned for a `u32` slice.
        out.mmap[destination_adjacency..destination_adjacency + dynamic_adjacency_bytes].fill(0xff);
        out.mmap[destination_adjacency..destination_adjacency + copied_adjacency_bytes]
            .copy_from_slice(&source[source_adjacency..source_adjacency + copied_adjacency_bytes]);

        let valid = out.meta.valid_offset as usize;
        out.mmap[valid..valid + index.num_vectors].fill(1);

        // The workspace is transient.  The later static commit is the durable
        // operation, so forcing a full synchronous write here only delays an
        // insert without improving recovery semantics.
        debug!(
            "dynamic workspace initialized path={} source_vectors={} capacity={} bytes={} copied_vector_bytes={} elapsed_ms={}",
            out.work_path.display(),
            index.num_vectors,
            capacity,
            file_len,
            vector_bytes,
            started.elapsed().as_millis()
        );
        Ok(out)
    }

    pub fn capacity(&self) -> usize {
        self.meta.capacity
    }
    pub(crate) fn dim(&self) -> usize {
        self.meta.dim
    }
    pub(crate) fn max_degree(&self) -> usize {
        self.meta.max_degree
    }
    pub(crate) fn medoid_id(&self) -> u32 {
        self.meta.medoid_id
    }
    pub(crate) fn distance_name(&self) -> &str {
        &self.meta.distance_name
    }
    pub fn len(&self) -> usize {
        self.live_count
    }
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }
    pub fn is_valid(&self, id: u32) -> bool {
        (id as usize) < self.meta.capacity
            && self.mmap[self.meta.valid_offset as usize + id as usize] != 0
    }
    pub fn flush(&self) -> Result<(), DiskAnnError> {
        self.mmap.flush()?;
        Ok(())
    }

    /// Enable a lightweight, in-memory routability guard for later guarded
    /// deletions. The guard is a dynamic-session sidecar and is discarded when
    /// the session commits to a static index.
    pub(crate) fn enable_routability_guard(
        &mut self,
        config: RoutabilityGuardConfig,
    ) -> Result<RoutabilityGuardStatus, DiskAnnError> {
        let config = config.normalized();
        let excluded = HashSet::new();
        let entry = self.residual_entry(&excluded).ok_or_else(|| {
            DiskAnnError::IndexError("cannot enable a routability guard on an empty index".into())
        })?;
        let pool_size = config
            .landmark_count
            .saturating_mul(config.landmark_pool_multiplier)
            .max(config.landmark_count);
        let landmark_pool = self.select_routability_landmarks(
            &excluded,
            entry,
            pool_size,
            config.candidate_sample_size,
            &[],
        );
        if landmark_pool.is_empty() {
            return Err(DiskAnnError::IndexError(
                "routability guard requires at least one live vector".into(),
            ));
        }
        self.routability_guard = Some(RoutabilityGuard {
            config,
            landmark_pool,
        });
        let status = self.routability_guard_status();
        debug!(
            "routability guard enabled landmarks={} pool={} beam={}",
            status.landmark_count, status.landmark_pool_size, status.beam_width
        );
        Ok(status)
    }

    /// Disable the in-memory routability guard without changing graph data.
    pub(crate) fn disable_routability_guard(&mut self) {
        self.routability_guard = None;
    }

    /// Return the status of the optional in-memory routability guard.
    pub(crate) fn routability_guard_status(&self) -> RoutabilityGuardStatus {
        let Some(guard) = &self.routability_guard else {
            return RoutabilityGuardStatus::default();
        };
        RoutabilityGuardStatus {
            enabled: true,
            landmark_pool_size: guard.landmark_pool.len(),
            live_landmarks: guard
                .landmark_pool
                .iter()
                .filter(|&&id| self.is_valid(id))
                .count(),
            landmark_count: guard.config.landmark_count,
            beam_width: guard.config.beam_width,
        }
    }

    /// Assess a proposed delete batch without modifying the graph.
    pub(crate) fn assess_delete_admission(
        &self,
        ids: &[u32],
    ) -> Result<RoutabilityAdmissionReport, DiskAnnError> {
        let guard = self.routability_guard.as_ref().ok_or_else(|| {
            DiskAnnError::IndexError(
                "routability guard is disabled; call enable_routability_guard first".into(),
            )
        })?;
        let deleted = ids
            .iter()
            .copied()
            .filter(|id| self.is_valid(*id))
            .collect::<HashSet<_>>();
        let mut report = RoutabilityAdmissionReport {
            requested: deleted.len(),
            ..RoutabilityAdmissionReport::default()
        };
        if deleted.is_empty() {
            report.admitted = true;
            return Ok(report);
        }
        let Some(entry) = self.residual_entry(&deleted) else {
            return Ok(report);
        };
        report.entry = Some(entry);

        let mut landmarks = guard
            .landmark_pool
            .iter()
            .copied()
            .filter(|id| self.is_valid(*id) && !deleted.contains(id))
            .take(guard.config.landmark_count)
            .collect::<Vec<_>>();
        if landmarks.len() < guard.config.landmark_count {
            let needed = guard.config.landmark_count - landmarks.len();
            let supplemental = self.select_routability_landmarks(
                &deleted,
                entry,
                needed,
                guard.config.candidate_sample_size,
                &landmarks,
            );
            report.supplemental_landmarks = supplemental.len();
            landmarks.extend(supplemental);
        }
        report.checked_landmarks = landmarks.clone();
        if landmarks.is_empty() {
            return Ok(report);
        }

        let no_deletions = HashSet::new();
        let probes = landmarks
            .par_iter()
            .map(|&landmark| {
                let baseline = self
                    .virtual_search_pool(
                        self.vector(landmark),
                        guard.config.beam_width,
                        &no_deletions,
                        entry,
                    )
                    .iter()
                    .any(|candidate| candidate.id == landmark);
                let residual = self
                    .virtual_search_pool(
                        self.vector(landmark),
                        guard.config.beam_width,
                        &deleted,
                        entry,
                    )
                    .iter()
                    .any(|candidate| candidate.id == landmark);
                (landmark, baseline, residual)
            })
            .collect::<Vec<_>>();
        report.baseline_reachable_landmarks =
            probes.iter().filter(|(_, baseline, _)| *baseline).count();
        report.residual_reachable_landmarks =
            probes.iter().filter(|(_, _, residual)| *residual).count();
        report.lost_landmarks = probes
            .into_iter()
            .filter_map(|(landmark, baseline, residual)| {
                (baseline && !residual).then_some(landmark)
            })
            .collect();
        report.admitted =
            report.baseline_reachable_landmarks > 0 && report.lost_landmarks.is_empty();
        debug!(
            "routability preflight requested={} checked={} baseline={} residual={} lost={} admitted={}",
            report.requested,
            report.checked_landmarks.len(),
            report.baseline_reachable_landmarks,
            report.residual_reachable_landmarks,
            report.lost_landmarks.len(),
            report.admitted
        );
        Ok(report)
    }

    /// Atomically run the optional routability preflight and, only on success,
    /// apply MERIT deletion repair. A deferred batch leaves all graph data
    /// unchanged.
    pub(crate) fn delete_batch_with_admission_control(
        &mut self,
        ids: &[u32],
        repair_beam: usize,
        repair_degree: usize,
    ) -> Result<GuardedDeleteResult, DiskAnnError> {
        let report = self.assess_delete_admission(ids)?;
        if report.admitted {
            let stats = self.delete_batch(ids, repair_beam, repair_degree);
            Ok(GuardedDeleteResult::Applied { stats, report })
        } else {
            Ok(GuardedDeleteResult::Deferred(report))
        }
    }

    pub(crate) fn search_with_dists(&self, query: &[T], k: usize, beam: usize) -> Vec<(u32, f32)> {
        self.search_pool(query, beam)
            .into_iter()
            .take(k)
            .map(|x| (x.id, x.dist))
            .collect()
    }

    pub(crate) fn get_vector(&self, id: usize) -> Vec<T> {
        self.vector(id as u32).to_vec()
    }

    pub(crate) fn static_layout(&self) -> (Vec<u32>, Vec<u32>, u32, Vec<Option<u32>>) {
        let live = (0..self.meta.capacity)
            .map(|id| id as u32)
            .filter(|id| self.is_valid(*id))
            .collect::<Vec<_>>();
        let mut remap = vec![PAD_U32; self.meta.capacity];
        for (new_id, old_id) in live.iter().enumerate() {
            remap[*old_id as usize] = new_id as u32;
        }
        let id_map = remap
            .iter()
            .map(|id| (*id != PAD_U32).then_some(*id))
            .collect();
        let medoid = remap[self.meta.medoid_id as usize];
        (live, remap, medoid, id_map)
    }

    pub(crate) fn vector_slice(&self, id: u32) -> &[T] {
        self.vector(id)
    }

    pub(crate) fn work_path(&self) -> &Path {
        &self.work_path
    }

    pub fn insert(&mut self, vector: Vec<T>, beam: usize) -> Result<u32, DiskAnnError> {
        self.insert_batch(vec![vector], beam)
            .map(|mut ids| ids.remove(0))
    }

    /// Inserts a batch as Vamana micro-batches. Each chunk is planned against
    /// one snapshot and merges reverse edges per destination before the next
    /// chunk starts. All expensive distance work is parallel; final mmap row
    /// writes are deterministic and conflict-free.
    pub fn insert_batch(
        &mut self,
        vectors: Vec<Vec<T>>,
        beam: usize,
    ) -> Result<Vec<u32>, DiskAnnError> {
        let started = Instant::now();
        if vectors.iter().any(|v| v.len() != self.meta.dim) {
            return Err(DiskAnnError::IndexError(
                "insert vector dimension mismatch".into(),
            ));
        }
        let slots = (0..self.meta.capacity)
            .filter(|&id| !self.is_valid(id as u32))
            .take(vectors.len())
            .map(|id| id as u32)
            .collect::<Vec<_>>();
        if slots.len() != vectors.len() {
            return Err(DiskAnnError::IndexError(
                "dynamic mmap capacity exhausted".into(),
            ));
        }

        let mut total_outgoing_edges = 0usize;
        let mut total_reverse_rows = 0usize;
        for (chunk_index, (slot_chunk, vector_chunk)) in slots
            .chunks(INSERT_MICRO_BATCH_SIZE)
            .zip(vectors.chunks(INSERT_MICRO_BATCH_SIZE))
            .enumerate()
        {
            let chunk_started = Instant::now();
            for (&id, vector) in slot_chunk.iter().zip(vector_chunk.iter()) {
                self.write_vector(id, vector);
                self.fill_neighbors(id);
                self.set_valid(id, true);
            }
            self.live_count += slot_chunk.len();

            let outgoing_started = Instant::now();
            debug!(
                "dynamic insert chunk={} phase=outgoing_plan start inserted={} beam={} rayon_workers={}",
                chunk_index + 1,
                slot_chunk.len(),
                beam,
                rayon::current_num_threads()
            );
            let plans = slot_chunk
                .par_iter()
                .zip(vector_chunk.par_iter())
                .map(|(&id, vector)| {
                    let pool = self.search_pool(vector, beam.max(self.meta.max_degree));
                    let selected = self.prune_scored(
                        id,
                        pool.iter()
                            .map(|candidate| (candidate.id, candidate.dist))
                            .collect(),
                    );
                    let outgoing = selected
                        .into_iter()
                        .map(|target| {
                            let distance = pool
                                .iter()
                                .find(|candidate| candidate.id == target)
                                .expect("RobustPrune selected a candidate outside its pool")
                                .dist;
                            (target, distance)
                        })
                        .collect();
                    InsertPlan { id, outgoing }
                })
                .collect::<Vec<_>>();
            let outgoing_edges: usize = plans.iter().map(|plan| plan.outgoing.len()).sum();
            debug!(
                "dynamic insert chunk={} phase=outgoing_plan complete outgoing_edges={} elapsed_ms={}",
                chunk_index + 1,
                outgoing_edges,
                outgoing_started.elapsed().as_millis()
            );

            let mut reverse_inputs = HashMap::<u32, Vec<(u32, f32)>>::new();
            for plan in &plans {
                for &(target, distance) in &plan.outgoing {
                    reverse_inputs
                        .entry(target)
                        .or_default()
                        .push((plan.id, distance));
                }
            }
            let mut reverse_inputs = reverse_inputs.into_iter().collect::<Vec<_>>();
            reverse_inputs.sort_unstable_by_key(|(source, _)| *source);
            let reverse_started = Instant::now();
            debug!(
                "dynamic insert chunk={} phase=reverse_prune start reverse_rows={}",
                chunk_index + 1,
                reverse_inputs.len()
            );
            let reverse_updates = reverse_inputs
                .par_iter()
                .map(|(source, incoming)| {
                    let source_vector = self.vector(*source);
                    let mut candidates = self
                        .live_neighbors(*source)
                        .into_iter()
                        .map(|id| (id, self.dist.eval(source_vector, self.vector(id))))
                        .collect::<Vec<_>>();
                    candidates.extend(incoming.iter().copied());
                    (*source, self.prune_scored(*source, candidates))
                })
                .collect::<Vec<_>>();
            debug!(
                "dynamic insert chunk={} phase=reverse_prune complete reverse_rows={} elapsed_ms={}",
                chunk_index + 1,
                reverse_inputs.len(),
                reverse_started.elapsed().as_millis()
            );

            for plan in &plans {
                let outgoing = plan
                    .outgoing
                    .iter()
                    .map(|(target, _)| *target)
                    .collect::<Vec<_>>();
                self.write_neighbors(plan.id, &outgoing);
            }
            for (source, neighbors) in reverse_updates {
                self.write_neighbors(source, &neighbors);
            }

            total_outgoing_edges += outgoing_edges;
            total_reverse_rows += reverse_inputs.len();
            debug!(
                "dynamic insert chunk={} inserted={} outgoing_edges={} reverse_rows={} elapsed_ms={}",
                chunk_index + 1,
                slot_chunk.len(),
                outgoing_edges,
                reverse_inputs.len(),
                chunk_started.elapsed().as_millis()
            );
        }
        debug!(
            "dynamic insert planned_and_committed={} outgoing_edges={} reverse_rows={} beam={} elapsed_ms={}",
            slots.len(),
            total_outgoing_edges,
            total_reverse_rows,
            beam,
            started.elapsed().as_millis()
        );
        Ok(slots)
    }

    /// MERIT batch deletion: invalidate together, build repair plans in parallel,
    /// then commit potentially overlapping adjacency changes deterministically.
    pub fn delete_batch(
        &mut self,
        ids: &[u32],
        repair_beam: usize,
        repair_degree: usize,
    ) -> Vec<DeleteStats> {
        let started = Instant::now();
        let plans = self.prepare_delete_plans(ids, repair_beam);
        if plans.is_empty() {
            return Vec::new();
        }
        let mut waves: Vec<Vec<usize>> = Vec::new();
        let mut last_write_wave = HashMap::<u32, usize>::new();
        for (index, plan) in plans.iter().enumerate() {
            let wave_index = plan
                .candidates
                .iter()
                .filter_map(|id| last_write_wave.get(id).map(|wave| wave + 1))
                .max()
                .unwrap_or(0);
            while waves.len() <= wave_index {
                waves.push(Vec::new());
            }
            waves[wave_index].push(index);
            for id in &plan.candidates {
                last_write_wave.insert(*id, wave_index);
            }
        }

        debug!(
            "MERIT repair planned deletes={} conflict_waves={} repair_beam={} repair_degree={}",
            plans.len(),
            waves.len(),
            repair_beam,
            repair_degree
        );
        let mut attempted = vec![0usize; plans.len()];
        for wave in waves {
            let results = wave
                .par_iter()
                .map(|&index| {
                    (
                        index,
                        self.repair_kr_mst_updates(&plans[index].candidates, repair_degree.max(1)),
                    )
                })
                .collect::<Vec<_>>();
            for (index, (updates, edge_attempts)) in results {
                for (source, neighbors) in updates {
                    self.write_neighbors(source, &neighbors);
                }
                attempted[index] = edge_attempts;
            }
        }

        let stats = plans
            .into_iter()
            .enumerate()
            .map(|(index, mut plan)| {
                plan.stats.repair_edges_attempted = attempted[index];
                self.finish_delete_plan(&plan);
                plan.stats
            })
            .collect::<Vec<_>>();
        debug!(
            "MERIT repair committed deletes={} elapsed_ms={}",
            stats.len(),
            started.elapsed().as_millis()
        );
        stats
    }

    fn prepare_delete_plans(&mut self, ids: &[u32], repair_beam: usize) -> Vec<DeletePlan> {
        let mut unique = ids
            .iter()
            .copied()
            .filter(|id| self.is_valid(*id))
            .collect::<Vec<_>>();
        unique.sort_unstable();
        unique.dedup();
        if unique.is_empty() {
            return Vec::new();
        }
        let deleted: HashSet<u32> = unique.iter().copied().collect();
        let snapshots = unique
            .iter()
            .map(|&id| {
                (
                    id,
                    self.node_version(id),
                    self.live_neighbors(id),
                    self.vector(id).to_vec(),
                )
            })
            .collect::<Vec<_>>();
        for &id in &unique {
            self.set_valid(id, false);
        }
        self.live_count -= unique.len();
        if deleted.contains(&self.meta.medoid_id)
            && let Some(id) = (0..self.meta.capacity).find(|&id| self.is_valid(id as u32))
        {
            self.meta.medoid_id = id as u32;
            let _ = self.write_metadata();
        }
        snapshots
            .par_iter()
            .map(|(id, old_version, outgoing, vector)| {
                let recovered = self
                    .search_pool(vector, repair_beam)
                    .into_iter()
                    .map(|c| c.id)
                    .filter(|source| {
                        self.raw_edges(*source)
                            .any(|(target, version)| target == *id && version == *old_version)
                    })
                    .collect::<Vec<_>>();
                let mut candidates = outgoing.clone();
                candidates.extend(recovered.iter().copied());
                candidates.sort_unstable();
                candidates.dedup();
                candidates.retain(|x| self.is_valid(*x));
                DeletePlan {
                    id: *id,
                    old_version: *old_version,
                    candidates: candidates.clone(),
                    stats: DeleteStats {
                        outgoing_seeds: outgoing.len(),
                        recovered_in_neighbors: recovered.len(),
                        repair_candidates: candidates.len(),
                        repair_edges_attempted: 0,
                    },
                }
            })
            .collect()
    }

    fn finish_delete_plan(&mut self, plan: &DeletePlan) {
        self.set_node_version(plan.id, plan.old_version.wrapping_add(1));
        self.fill_neighbors(plan.id);
    }

    fn write_metadata(&mut self) -> Result<(), DiskAnnError> {
        let bytes = bincode::serialize(&self.meta)?;
        if bytes.len() + 8 > DATA_OFFSET as usize {
            return Err(DiskAnnError::IndexError(
                "dynamic metadata too large".into(),
            ));
        }
        self.mmap[0..8].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
        self.mmap[8..8 + bytes.len()].copy_from_slice(&bytes);
        Ok(())
    }
    fn vector_range(&self, id: u32) -> std::ops::Range<usize> {
        let s = self.meta.vectors_offset as usize
            + id as usize * self.meta.dim * std::mem::size_of::<T>();
        s..s + self.meta.dim * std::mem::size_of::<T>()
    }
    fn vector(&self, id: u32) -> &[T] {
        bytemuck::cast_slice(&self.mmap[self.vector_range(id)])
    }
    fn write_vector(&mut self, id: u32, v: &[T]) {
        let r = self.vector_range(id);
        self.mmap[r].copy_from_slice(bytemuck::cast_slice(v));
    }
    fn adj_pos(&self, id: u32, slot: usize) -> usize {
        self.meta.adjacency_offset as usize + (id as usize * self.meta.max_degree + slot) * 4
    }
    fn edge_ver_pos(&self, id: u32, slot: usize) -> usize {
        self.meta.edge_versions_offset as usize + (id as usize * self.meta.max_degree + slot) * 2
    }
    fn node_ver_pos(&self, id: u32) -> usize {
        self.meta.node_versions_offset as usize + id as usize * 2
    }
    fn node_version(&self, id: u32) -> u16 {
        u16::from_le_bytes(
            self.mmap[self.node_ver_pos(id)..self.node_ver_pos(id) + 2]
                .try_into()
                .unwrap(),
        )
    }
    fn set_node_version(&mut self, id: u32, v: u16) {
        let p = self.node_ver_pos(id);
        self.mmap[p..p + 2].copy_from_slice(&v.to_le_bytes());
    }
    fn set_valid(&mut self, id: u32, valid: bool) {
        self.mmap[self.meta.valid_offset as usize + id as usize] = valid as u8;
    }
    fn raw_edges(&self, id: u32) -> impl Iterator<Item = (u32, u16)> + '_ {
        (0..self.meta.max_degree).filter_map(move |slot| {
            let p = self.adj_pos(id, slot);
            let target = u32::from_le_bytes(self.mmap[p..p + 4].try_into().unwrap());
            if target == PAD_U32 {
                None
            } else {
                let q = self.edge_ver_pos(id, slot);
                Some((
                    target,
                    u16::from_le_bytes(self.mmap[q..q + 2].try_into().unwrap()),
                ))
            }
        })
    }
    pub(crate) fn live_neighbors(&self, id: u32) -> Vec<u32> {
        self.raw_edges(id)
            .filter(|(t, v)| self.is_valid(*t) && self.node_version(*t) == *v)
            .map(|x| x.0)
            .collect()
    }
    fn fill_neighbors(&mut self, id: u32) {
        self.write_neighbors(id, &[]);
    }
    fn write_neighbors(&mut self, id: u32, ids: &[u32]) {
        for slot in 0..self.meta.max_degree {
            let target = ids.get(slot).copied().unwrap_or(PAD_U32);
            let p = self.adj_pos(id, slot);
            self.mmap[p..p + 4].copy_from_slice(&target.to_le_bytes());
            let version = if target == PAD_U32 {
                0
            } else {
                self.node_version(target)
            };
            let q = self.edge_ver_pos(id, slot);
            self.mmap[q..q + 2].copy_from_slice(&version.to_le_bytes());
        }
    }

    fn search_pool(&self, query: &[T], beam: usize) -> Vec<Candidate> {
        if self.is_empty() || !self.is_valid(self.meta.medoid_id) {
            return Vec::new();
        }
        graph_search(
            self.meta.medoid_id,
            beam,
            |id| self.dist.eval(query, self.vector(id)),
            |id| self.live_neighbors(id),
        )
    }

    fn residual_entry(&self, deleted: &HashSet<u32>) -> Option<u32> {
        if self.is_valid(self.meta.medoid_id) && !deleted.contains(&self.meta.medoid_id) {
            return Some(self.meta.medoid_id);
        }
        (0..self.meta.capacity as u32).find(|id| self.is_valid(*id) && !deleted.contains(id))
    }

    fn select_routability_landmarks(
        &self,
        deleted: &HashSet<u32>,
        entry: u32,
        count: usize,
        sample_size: usize,
        existing: &[u32],
    ) -> Vec<u32> {
        if count == 0 {
            return Vec::new();
        }
        let mut references = existing.to_vec();
        if !references.contains(&entry) {
            references.push(entry);
        }
        let mut sample = Vec::new();
        let mut seen = HashSet::new();
        let stride = (self.meta.capacity / sample_size.max(1)).max(1);
        for id in (0..self.meta.capacity as u32).step_by(stride) {
            if id != entry
                && self.is_valid(id)
                && !deleted.contains(&id)
                && !references.contains(&id)
                && seen.insert(id)
            {
                sample.push(id);
            }
        }
        if sample.len() < count {
            for id in 0..self.meta.capacity as u32 {
                if id != entry
                    && self.is_valid(id)
                    && !deleted.contains(&id)
                    && !references.contains(&id)
                    && seen.insert(id)
                {
                    sample.push(id);
                }
            }
        }

        let mut selected = Vec::new();
        while selected.len() < count {
            let next = sample
                .iter()
                .copied()
                .filter(|id| !references.contains(id))
                .max_by(|a, b| {
                    let distance_a = references
                        .iter()
                        .map(|reference| self.dist.eval(self.vector(*a), self.vector(*reference)))
                        .min_by(f32::total_cmp)
                        .unwrap();
                    let distance_b = references
                        .iter()
                        .map(|reference| self.dist.eval(self.vector(*b), self.vector(*reference)))
                        .min_by(f32::total_cmp)
                        .unwrap();
                    distance_a.total_cmp(&distance_b)
                });
            let Some(next) = next else {
                break;
            };
            references.push(next);
            selected.push(next);
        }
        selected
    }

    fn virtual_search_pool(
        &self,
        query: &[T],
        beam: usize,
        deleted: &HashSet<u32>,
        entry: u32,
    ) -> Vec<Candidate> {
        if deleted.contains(&entry) || !self.is_valid(entry) {
            return Vec::new();
        }
        graph_search(
            entry,
            beam,
            |id| self.dist.eval(query, self.vector(id)),
            |id| {
                self.live_neighbors(id)
                    .into_iter()
                    .filter(|neighbor| !deleted.contains(neighbor))
                    .collect()
            },
        )
    }

    fn prune(&self, source: u32, ids: &[u32]) -> Vec<u32> {
        let sv = self.vector(source);
        let candidates = ids
            .iter()
            .copied()
            .filter(|id| *id != source && self.is_valid(*id))
            .map(|id| (id, self.dist.eval(sv, self.vector(id))))
            .collect::<Vec<_>>();
        self.prune_scored(source, candidates)
    }

    /// RobustPrune using distances that were already obtained during a graph
    /// search or a reverse-edge merge.  This avoids immediately recomputing the
    /// query-to-candidate distances for large, high-dimensional insert batches.
    fn prune_scored(&self, source: u32, candidates: Vec<(u32, f32)>) -> Vec<u32> {
        let mut c = candidates
            .into_iter()
            .filter(|(id, _)| *id != source && self.is_valid(*id))
            .collect::<Vec<_>>();
        c.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut seen = HashSet::new();
        c.retain(|x| seen.insert(x.0));
        c.truncate(750);
        let mut selected = Vec::new();
        let mut factors = vec![0f32; c.len()];
        let target = self.meta.alpha.max(1.0);
        let inc = target.min(1.2);
        let mut alpha = 1.;
        loop {
            for i in 0..c.len() {
                if selected.len() == self.meta.max_degree {
                    return selected;
                }
                if factors[i] > alpha {
                    continue;
                }
                let sid = c[i].0;
                factors[i] = f32::MAX;
                selected.push(sid);
                for j in i + 1..c.len() {
                    if factors[j] > target {
                        continue;
                    }
                    let pair = self.dist.eval(self.vector(c[j].0), self.vector(sid));
                    let f = if pair == 0. { f32::MAX } else { c[j].1 / pair };
                    factors[j] = factors[j].max(f);
                }
            }
            if alpha >= target {
                break;
            }
            alpha = (alpha * inc).min(target);
        }
        selected
    }
    // See MERIT paper: Wu et.al., 2026. MERIT: Efficient In-Place Deletion for Dynamic Graph-Based Approximate Nearest Neighbor Indexes. arXiv preprint arXiv:2607.29173.
    fn repair_kr_mst_updates(
        &self,
        candidates: &[u32],
        degree: usize,
    ) -> (HashMap<u32, Vec<u32>>, usize) {
        if candidates.is_empty() {
            return (HashMap::new(), 0);
        }
        // Prim selection and parent ranking revisit the same candidate pairs.
        // Compute each high-dimensional distance once and keep the small local
        // matrix hot in cache for the remainder of this repair plan.
        let count = candidates.len();
        let mut pair_distances = vec![0.0f32; count * count];
        for left in 0..count {
            for right in (left + 1)..count {
                let distance = self.dist.eval(
                    self.vector(candidates[left]),
                    self.vector(candidates[right]),
                );
                pair_distances[left * count + right] = distance;
                pair_distances[right * count + left] = distance;
            }
        }
        let entry = self.vector(self.meta.medoid_id);
        let start = candidates
            .iter()
            .enumerate()
            .min_by(|a, b| {
                self.dist
                    .eval(self.vector(*a.1), entry)
                    .total_cmp(&self.dist.eval(self.vector(*b.1), entry))
            })
            .map(|(index, _)| index)
            .unwrap();
        let mut connected = vec![start];
        let mut remaining = (0..count)
            .filter(|index| *index != start)
            .collect::<Vec<_>>();
        let mut updates = HashMap::<u32, Vec<u32>>::new();
        let mut attempted = 0;
        while !remaining.is_empty() {
            let (ri, next_index) = remaining
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = connected
                        .iter()
                        .map(|x| pair_distances[**a * count + *x])
                        .min_by(f32::total_cmp)
                        .unwrap();
                    let db = connected
                        .iter()
                        .map(|x| pair_distances[**b * count + *x])
                        .min_by(f32::total_cmp)
                        .unwrap();
                    da.total_cmp(&db)
                })
                .map(|(i, x)| (i, *x))
                .unwrap();
            let next = candidates[next_index];
            let mut parents = connected
                .iter()
                .copied()
                .map(|p| (candidates[p], pair_distances[next_index * count + p]))
                .collect::<Vec<_>>();
            parents.sort_by(|a, b| a.1.total_cmp(&b.1));
            for (parent, _) in parents.into_iter().take(degree) {
                for (source, target) in [(next, parent), (parent, next)] {
                    let mut ids = updates
                        .get(&source)
                        .cloned()
                        .unwrap_or_else(|| self.live_neighbors(source));
                    if !ids.contains(&target) {
                        ids.push(target);
                    }
                    updates.insert(source, self.prune(source, &ids));
                }
                attempted += 2;
            }
            connected.push(next_index);
            remaining.swap_remove(ri);
        }
        (updates, attempted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndists::dist::{DistHamming, DistL2};
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::fs;
    #[test]
    fn mmap_parallel_batches_persist_and_hide_stale_edges() {
        let base = "test_mmap_base.db";
        let dynp = "test_mmap_dynamic.db";
        let _ = fs::remove_file(base);
        let _ = fs::remove_file(dynp);
        let mut rng = StdRng::seed_from_u64(44);
        let vectors = (0..400)
            .map(|_| (0..16).map(|_| rng.r#gen::<f32>()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let idx = DiskANN::build_index_default(&vectors, DistL2, base).unwrap();
        let mut d = MmapDynamicDiskANN::create_from_static(&idx, 500, 1.2, dynp).unwrap();
        let ids = (20..60).collect::<Vec<_>>();
        assert_eq!(d.delete_batch(&ids, 128, 2).len(), 40);
        for id in &ids {
            assert!(
                !d.search_with_dists(&vectors[*id as usize], 20, 128)
                    .iter()
                    .any(|result| result.0 == *id)
            );
        }
        let replacements = ids
            .iter()
            .map(|id| vectors[*id as usize].iter().map(|x| x + 0.00001).collect())
            .collect();
        let reused = d.insert_batch(replacements, 128).unwrap();
        assert_eq!(reused.len(), 40);
        d.flush().unwrap();
        assert_eq!(d.len(), 400);
        for id in reused {
            assert_eq!(d.search_with_dists(&vectors[id as usize], 1, 128)[0].0, id);
        }
        let _ = fs::remove_file(base);
        let _ = fs::remove_file(dynp);
    }

    #[test]
    fn mmap_hamming_insert_uses_multiple_micro_batches() {
        let dir = std::env::temp_dir();
        let nonce = std::process::id();
        let base = dir.join(format!("rust_diskann_hamming_insert_{nonce}.db"));
        let dynp = dir.join(format!("rust_diskann_hamming_insert_{nonce}.work"));
        let _ = fs::remove_file(&base);
        let _ = fs::remove_file(&dynp);

        let mut rng = StdRng::seed_from_u64(91);
        let initial = (0..512)
            .map(|_| (0..128).map(|_| rng.r#gen::<u16>()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let inserted = (0..384)
            .map(|_| (0..128).map(|_| rng.r#gen::<u16>()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let index = DiskANN::build_index(
            &initial,
            32,
            64,
            1.2,
            1,
            DistHamming,
            base.to_str().unwrap(),
        )
        .unwrap();
        let mut dynamic = MmapDynamicDiskANN::create_from_static(
            &index,
            initial.len() + inserted.len(),
            1.2,
            &dynp,
        )
        .unwrap();
        let ids = dynamic.insert_batch(inserted.clone(), 64).unwrap();

        assert_eq!(ids.len(), inserted.len());
        assert_eq!(dynamic.len(), initial.len() + inserted.len());
        for (id, vector) in ids.iter().zip(&inserted) {
            assert!(!dynamic.live_neighbors(*id).is_empty());
            assert!(!dynamic.search_with_dists(vector, 1, 512).is_empty());
        }

        drop(dynamic);
        drop(index);
        let _ = fs::remove_file(base);
        let _ = fs::remove_file(dynp);
    }

    #[test]
    fn routability_guard_defers_a_bridge_delete_without_mutating_the_index() {
        let dir = std::env::temp_dir();
        let nonce = std::process::id();
        let base = dir.join(format!("rust_diskann_guard_{nonce}.db"));
        let dynp = dir.join(format!("rust_diskann_guard_{nonce}.work"));
        let _ = fs::remove_file(&base);
        let _ = fs::remove_file(&dynp);

        let vectors = (0..10).map(|id| vec![id as f32]).collect::<Vec<_>>();
        let index = DiskANN::build_index_default(&vectors, DistL2, base.to_str().unwrap()).unwrap();
        let mut dynamic =
            MmapDynamicDiskANN::create_from_static(&index, vectors.len(), 1.2, &dynp).unwrap();

        // Force a directed chain so deleting node 4 removes the only route
        // from entry 0 to the distant landmark(s).
        dynamic.meta.medoid_id = 0;
        for id in 0..vectors.len() as u32 {
            if (id as usize) + 1 < vectors.len() {
                dynamic.write_neighbors(id, &[id + 1]);
            } else {
                dynamic.write_neighbors(id, &[]);
            }
        }
        let status = dynamic
            .enable_routability_guard(RoutabilityGuardConfig {
                landmark_count: 2,
                landmark_pool_multiplier: 1,
                candidate_sample_size: 16,
                beam_width: 16,
            })
            .unwrap();
        assert!(status.enabled);

        let report = dynamic.assess_delete_admission(&[4]).unwrap();
        assert!(report.baseline_reachable_landmarks > 0);
        assert!(!report.admitted);
        assert!(!report.lost_landmarks.is_empty());

        match dynamic
            .delete_batch_with_admission_control(&[4], 16, 2)
            .unwrap()
        {
            GuardedDeleteResult::Deferred(deferred) => {
                assert_eq!(deferred.lost_landmarks, report.lost_landmarks);
            }
            GuardedDeleteResult::Applied { .. } => {
                panic!("bridge delete should be deferred")
            }
        }
        assert!(dynamic.is_valid(4));

        let mut update = DiskANN::from_dynamic(dynamic, DistL2);
        let error = update.delete_batch(&[4]).unwrap_err();
        match error {
            DiskAnnError::IndexError(message) => {
                assert!(message.contains("deferred by routability admission control"));
            }
            other => panic!("expected a routability deferral, got {other:?}"),
        }
        assert_eq!(update.num_vectors, vectors.len());
        assert!(update.dynamic.as_ref().unwrap().is_valid(4));

        drop(update);
        drop(index);
        let _ = fs::remove_file(base);
        let _ = fs::remove_file(dynp);
    }
}

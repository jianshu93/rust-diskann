//! Fixed-layout, mmap-backed dynamic DiskANN with MERIT deletion repair.

use super::{Candidate, DiskANN, DiskAnnError, PAD_U32, graph_search};
use anndists::prelude::Distance;
use memmap2::{MmapMut, MmapOptions};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

const DATA_OFFSET: u64 = 1024 * 1024;
const MAGIC: [u8; 8] = *b"DYNANN01";

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

#[derive(Clone)]
struct DeletePlan {
    id: u32,
    old_version: u16,
    candidates: Vec<u32>,
    stats: DeleteStats,
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
            _marker: PhantomData,
        };
        out.write_metadata()?;
        for id in 0..capacity {
            out.fill_neighbors(id as u32);
        }
        for id in 0..index.num_vectors {
            out.write_vector(id as u32, &index.get_vector(id));
            out.set_valid(id as u32, true);
            let neighbors = index
                .get_neighbors(id as u32)
                .iter()
                .copied()
                .filter(|x| *x != PAD_U32)
                .collect::<Vec<_>>();
            out.write_neighbors(id as u32, &neighbors);
        }
        out.flush()?;
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
        (0..self.meta.capacity)
            .filter(|&id| self.is_valid(id as u32))
            .count()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn is_valid(&self, id: u32) -> bool {
        (id as usize) < self.meta.capacity
            && self.mmap[self.meta.valid_offset as usize + id as usize] != 0
    }
    pub fn flush(&self) -> Result<(), DiskAnnError> {
        self.mmap.flush()?;
        Ok(())
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

    /// Parallel search/planning, followed by deterministic graph commits.
    pub fn insert_batch(
        &mut self,
        vectors: Vec<Vec<T>>,
        beam: usize,
    ) -> Result<Vec<u32>, DiskAnnError> {
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
        let pools = vectors
            .par_iter()
            .map(|v| {
                self.search_pool(v, beam.max(self.meta.max_degree))
                    .into_iter()
                    .map(|c| c.id)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for ((&id, vector), _) in slots.iter().zip(vectors.iter()).zip(pools.iter()) {
            self.write_vector(id, vector);
            self.fill_neighbors(id);
            self.set_valid(id, true);
        }
        for (&id, pool) in slots.iter().zip(pools.iter()) {
            let selected = self.prune(id, pool);
            self.write_neighbors(id, &selected);
            for target in selected {
                self.insert_and_prune(target, id);
            }
        }
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

        plans
            .into_iter()
            .enumerate()
            .map(|(index, mut plan)| {
                plan.stats.repair_edges_attempted = attempted[index];
                self.finish_delete_plan(&plan);
                plan.stats
            })
            .collect()
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
    fn prune(&self, source: u32, ids: &[u32]) -> Vec<u32> {
        let sv = self.vector(source);
        let mut c = ids
            .iter()
            .copied()
            .filter(|id| *id != source && self.is_valid(*id))
            .map(|id| (id, self.dist.eval(sv, self.vector(id))))
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
    fn insert_and_prune(&mut self, source: u32, target: u32) {
        if !self.is_valid(source) || !self.is_valid(target) || source == target {
            return;
        }
        let mut ids = self.live_neighbors(source);
        if !ids.contains(&target) {
            ids.push(target)
        }
        let selected = self.prune(source, &ids);
        self.write_neighbors(source, &selected);
    }
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
    use anndists::dist::DistL2;
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
}

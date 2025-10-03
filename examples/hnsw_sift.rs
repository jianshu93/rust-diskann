#![allow(clippy::needless_range_loop)]

use cpu_time::ProcessTime;
use std::path::Path;
use std::time::{Duration, SystemTime};

use anndists::dist::*;
use log::info;

use hnsw_rs::hnswio::HnswIo;
use hnsw_rs::prelude::*;

mod utils;
use utils::*;

pub fn main() {
    let parallel = true;

    // dataset
    // wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
    let fname = String::from("./sift-128-euclidean.hdf5");
    println!("\n\n test_load_hdf5 {:?}", fname);
    let anndata = annhdf5::AnnBenchmarkData::new(fname).unwrap();

    let knbn_max = anndata.test_distances.dim().1;
    let nb_elem = anndata.train_data.len();
    info!(
        " train size : {}, test size : {}",
        nb_elem,
        anndata.test_data.len()
    );
    info!(" nb neighbours answers for test data : {}", knbn_max);

    // index location/name
    let out_dir = Path::new("./");
    let base = "sift1m_l2_hnsw";
    let graph_p = out_dir.join(format!("{base}.hnsw.graph"));
    let data_p = out_dir.join(format!("{base}.hnsw.data"));
    let have_dump = graph_p.exists() && data_p.exists();

    let mut _reload_guard: Option<HnswIo> = None;

    let mut hnsw: Hnsw<f32, DistL2>;

    if have_dump {
        println!("Found existing index: {:?} and {:?}", graph_p, data_p);

        // Keep the loader alive for as long as `hnsw` is used.
        _reload_guard = Some(HnswIo::new(out_dir, base));

        // Call load_hnsw via a mutable ref to the loader stored in the guard.
        let loaded: Hnsw<f32, DistL2> = _reload_guard
            .as_mut()
            .unwrap()
            .load_hnsw::<f32, DistL2>()
            .expect("reload HNSW from disk");

        hnsw = loaded;

        println!("Reloaded HNSW with {} points", hnsw.get_nb_point());
    } else {
        // build parameters
        let max_nb_connection = 48;
        let nb_layer = 16;
        let ef_c = 1600;

        println!(
            "No saved index. Building new one: N={} layers={} ef_c={}",
            nb_elem, nb_layer, ef_c
        );

        let mut idx =
            Hnsw::<f32, DistL2>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL2 {});

        let extend_flag = false;
        info!("extend flag = {:?} ", extend_flag);
        idx.set_extend_candidates(extend_flag);
        idx.modify_level_scale(0.5);

        // parallel insertion
        let start = ProcessTime::now();
        let now = SystemTime::now();
        let data_for_par_insertion: Vec<(&[f32], usize)> = anndata
            .train_data
            .iter()
            .map(|x| (x.0.as_slice(), x.1))
            .collect();

        if parallel {
            println!(" \n parallel insertion");
            idx.parallel_insert_slice(&data_for_par_insertion);
        } else {
            println!(" \n serial insertion");
            for d in data_for_par_insertion {
                idx.insert_slice(d);
            }
        }
        let cpu_time: Duration = start.elapsed();

        // save to disk
        match idx.file_dump(out_dir, base) {
            Ok(dump_base) => {
                println!("HNSW index saved as: {}.hnsw.graph / .hnsw.data", dump_base);
            }
            Err(e) => eprintln!("HNSW file_dump failed: {e}"),
        }

        println!(
            "\n hnsw data insertion cpu time  {:?}  system time {:?} ",
            cpu_time,
            now.elapsed()
        );

        hnsw = idx;
    }

    // search
    hnsw.set_searching_mode(true);
    hnsw.dump_layer_info();
    println!(" hnsw data nb point inserted {:?}", hnsw.get_nb_point());

    let knbn = 10.min(knbn_max);

    let ef_search = 64;
    println!("searching with ef = {}", ef_search);
    search(&mut hnsw, knbn, ef_search, &anndata);
}

pub fn search<Dist>(
    hnsw: &mut Hnsw<f32, Dist>,
    knbn: usize,
    ef_search: usize,
    anndata: &annhdf5::AnnBenchmarkData,
) where
    Dist: Distance<f32> + Send + Sync,
{
    println!("\n\n ef_search : {:?} knbn : {:?} ", ef_search, knbn);
    let parallel = true;

    let nb_elem = anndata.train_data.len();
    let nb_search = anndata.test_data.len();

    let mut recalls = Vec::<usize>::with_capacity(nb_elem);
    let mut nb_returned = Vec::<usize>::with_capacity(nb_elem);
    let mut last_distances_ratio = Vec::<f32>::with_capacity(nb_elem);
    let mut knn_neighbours_for_tests = Vec::<Vec<Neighbour>>::with_capacity(nb_elem);

    hnsw.set_searching_mode(true);
    println!("searching with ef : {:?}", ef_search);

    let start = ProcessTime::now();
    let now = SystemTime::now();

    if parallel {
        println!(" \n parallel search");
        knn_neighbours_for_tests = hnsw.parallel_search(&anndata.test_data, knbn, ef_search);
    } else {
        println!(" \n serial search");
        for i in 0..anndata.test_data.len() {
            let knn_neighbours: Vec<Neighbour> =
                hnsw.search(&anndata.test_data[i], knbn, ef_search);
            knn_neighbours_for_tests.push(knn_neighbours);
        }
    }

    let cpu_time = start.elapsed();
    let search_cpu_time = cpu_time.as_micros() as f32;
    let search_sys_time = now.elapsed().unwrap().as_micros() as f32;
    println!(
        "total cpu time for search requests {:?} , system time {:?} ",
        search_cpu_time,
        now.elapsed()
    );

    for i in 0..anndata.test_data.len() {
        let max_dist = anndata.test_distances.row(i)[knbn - 1];
        let knn_neighbours_d: Vec<f32> = knn_neighbours_for_tests[i]
            .iter()
            .map(|p| p.distance)
            .collect();
        nb_returned.push(knn_neighbours_d.len());
        let recall = knn_neighbours_d.iter().filter(|d| *d <= &max_dist).count();
        recalls.push(recall);
        let mut ratio = 0.;
        if !knn_neighbours_d.is_empty() {
            ratio = knn_neighbours_d[knn_neighbours_d.len() - 1] / max_dist;
        }
        last_distances_ratio.push(ratio);
    }

    let mean_recall = (recalls.iter().sum::<usize>() as f32) / ((knbn * recalls.len()) as f32);
    println!(
        "\n mean fraction nb returned by search {:?} ",
        (nb_returned.iter().sum::<usize>() as f32) / ((nb_returned.len() * knbn) as f32)
    );
    println!(
        "\n last distances ratio {:?} ",
        last_distances_ratio.iter().sum::<f32>() / last_distances_ratio.len() as f32
    );
    println!(
        "\n recall rate for {:?} is {:?} , nb req /s {:?}",
        anndata.fname,
        mean_recall,
        (nb_search as f32) * 1.0e+6_f32 / search_sys_time
    );
} // end of search

use serde::{Deserialize, Deserializer, Serialize};
use std::fs;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::marker::PhantomData;

pub struct tile_config {
    pub relu_chunks: usize,
    pub relu_chunk_size: usize,
    pub leftovers: usize,
}

pub fn configure_tiling(relus: usize, memory_capacity: usize, tiled:bool) -> tile_config {
    if tiled{
        let relu_chunk_size = std::cmp::min(memory_capacity, relus);
        let mut relu_chunks = relus / relu_chunk_size;
        let leftovers = relus % relu_chunk_size;
        if leftovers != 0 {
            relu_chunks += 1;
        }
        return tile_config {
            relu_chunks: relu_chunks as usize,
            relu_chunk_size: relu_chunk_size as usize,
            leftovers: leftovers as usize,
        };
    }else{
        return tile_config {
            relu_chunks: 1 as usize,
            relu_chunk_size: relus as usize,
            leftovers: 0 as usize,
        };
    }
}

pub fn create_directory(path: &str) -> std::io::Result<()> {
    fs::create_dir_all(path)
}

pub fn get_file_name(
    data_dir: &str,
    mode: &str,
    system: &str,
    batch_id: &u16,
    layer_id: &str,
    chunk_index: &str,
) -> String {
    let file_name = format!(
        "{}/relu_data/batch{}/layer{}/{}/{}/",
        data_dir, batch_id, layer_id, mode, system
    );
    match create_directory(&file_name) {
        Ok(_) => {},
        Err(e) => eprintln!("Error in creating directory: {}", e),
    };

    format!("{}/data_{}.bin", file_name, chunk_index)
}

use csv::Writer;
use serde::Serialize;
use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;

fn create_directory_if_not_exists(path_str: &str) -> io::Result<()> {
    let path = Path::new(path_str);

    if !path.exists() {
        fs::create_dir_all(path)?;
    }

    Ok(())
}

pub fn csv_file_name_network(network: &str, system: &str, phase: &str, batch_id: u64, batch_size: u64, cores: u64, memory: u64) -> String {
    let mut csv_file = format!(
        "/mnt/mohammad/delphi_bench/benchmarking/data/{}/_{}_{}_{}_{}/_{}__batchsz/_{}__batchid/{}/{}/network",
        network, cores, cores, memory, memory, batch_size, batch_id, system, phase
    );
    create_directory_if_not_exists(&csv_file).expect("Error creating directory");
    csv_file.push_str("/");
    csv_file.push_str("total_communication");
    csv_file.push_str(".csv");
    csv_file
}

pub fn csv_file_name_comm(
    network: &str,
    system: &str,
    phase: &str,
    subroutine: &str,
    conv_id: u64,
    batch_id: u64,
    batch_size: u64,
    cores: u64,
    memory: u64,
) -> String {
    let mut csv_file = format!(
        "/mnt/mohammad/delphi_bench/benchmarking/data/{}/_{}_{}_{}_{}/_{}__batchsz/_{}__batchid/{}/{}/{}/network",
        network, cores, cores, memory, memory, batch_size, batch_id, system, phase, subroutine
    );
    create_directory_if_not_exists(&csv_file).expect("Error creating directory");
    csv_file.push_str("/");
    csv_file.push_str("layer_");
    csv_file.push_str(&conv_id.to_string());
    csv_file.push_str(".csv");
    csv_file
}

pub fn csv_file_name(
    network: &str,
    system: &str,
    phase: &str,
    subroutine: &str,
    conv_id: u64,
    batch_id: u64,
    batch_size: u64,
    cores: u64,
    memory: u64,
) -> String {
    let mut csv_file = format!(
        "/mnt/mohammad/delphi_bench/benchmarking/data/{}/_{}_{}_{}_{}/_{}__batchsz/_{}__batchid/{}/{}/{}/latency",
        network, cores, cores, memory, memory, batch_size, batch_id, system, phase, subroutine
    );
    create_directory_if_not_exists(&csv_file).expect("Error creating directory");
    csv_file.push_str("/");
    csv_file.push_str("layer_");
    csv_file.push_str(&conv_id.to_string());
    csv_file.push_str(".csv");
    csv_file
}

pub fn write_to_csv<T: Serialize>(item: &T, csv_file: &str) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(csv_file)?;
    wtr.serialize(item)?;
    wtr.flush()?;
    Ok(())
}

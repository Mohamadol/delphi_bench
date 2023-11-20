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

pub fn csv_file_name(
    network: &str,
    system: &str,
    phase: &str,
    subroutine: &str,
    conv_id: u64,
    batch_id: u64,
) -> String {
    let mut csv_file = String::new();
    csv_file.push_str("/mnt/mohammad/delphi/rust/benchmarking");
    csv_file.push_str("/");
    csv_file.push_str(network);
    csv_file.push_str("/");
    csv_file.push_str(system);
    csv_file.push_str("/");
    csv_file.push_str(phase);
    csv_file.push_str("/");
    csv_file.push_str(subroutine);
    csv_file.push_str("/");
    csv_file.push_str("_batch__");
    csv_file.push_str(&batch_id.to_string());

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

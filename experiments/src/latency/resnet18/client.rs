use clap::{App, Arg, ArgMatches};
use experiments::resnet18::construct_resnet18_model;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use std::env;

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <number>", args[0]);
        std::process::exit(1);
    }
    let batch_id = match args[1].parse::<u16>() {
        Ok(number) => number,
        Err(e) => {
            eprintln!("Error: Argument is not a valid integer - {}", e);
            std::process::exit(1);
        },
    };

    let vs = tch::nn::VarStore::new(tch::Device::cuda_if_available());
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let server_addr = format!("10.128.0.34:{}", 8001 + batch_id);
    let network = construct_resnet18_model(Some(&vs.root()), 8, &mut rng);

    println!(
        "sending client request for batch ID {} and port {}",
        batch_id,
        8001 + batch_id,
    );
    let architecture = (&network).into();
    let network_name = "resnet18";
    experiments::latency::client::nn_client(
        &server_addr,
        architecture,
        &mut rng,
        batch_id,
        network_name,
    );
}

use ::neural_network as nn;
extern crate num_cpus;
extern crate rayon;
use algebra::{fields::near_mersenne_64::F, FixedPoint, FixedPointParameters, Polynomial};
use bench_utils::*;
use io_utils::{counting::CountingIO, imux::IMuxSync};
use nn::{
    layers::{
        average_pooling::AvgPoolParams,
        convolution::{Conv2dParams, Padding},
        fully_connected::FullyConnectedParams,
        Layer, LayerDims, LinearLayer, NonLinearLayer,
    },
    tensors::*,
    NeuralArchitecture, NeuralNetwork,
};
use protocols::{csv_timing::*, neural_network::NNProtocol, AdditiveShare, CommunicationData};
use rand::{CryptoRng, Rng, RngCore};
use std::{
    io::{BufReader, BufWriter},
    net::{TcpListener, TcpStream},
    thread,
    time::Duration,
};

// pub mod inference;
pub mod latency;
pub mod linear_only;
// pub mod minionn;
// pub mod mnist;
pub mod resnet18;
pub mod resnet32;
pub mod resnet50;
pub mod test_3layers;
// pub mod throughput;
// pub mod validation;

pub struct TenBitExpParams {}

impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 3;
    const EXPONENT_CAPACITY: u8 = 8;
}

type TenBitExpFP = FixedPoint<TenBitExpParams>;
type TenBitAS = AdditiveShare<TenBitExpParams>;

pub fn client_connect(
    addr: &str,
) -> (
    IMuxSync<CountingIO<BufReader<TcpStream>>>,
    IMuxSync<CountingIO<BufWriter<TcpStream>>>,
) {
    // TODO: Maybe change to rayon_num_threads
    let mut readers = Vec::with_capacity(16);
    let mut writers = Vec::with_capacity(16);
    for _ in 0..16 {
        // let stream = TcpStream::connect(addr).unwrap();
        // readers.push(CountingIO::new(BufReader::new(stream.try_clone().unwrap())));
        // writers.push(CountingIO::new(BufWriter::new(stream)));
        let mut connected = false;
        while !connected {
            match TcpStream::connect(addr) {
                Ok(stream) => {
                    // println!("Successfully connected to the server for the next ReLU chunk");
                    connected = true;
                    readers.push(CountingIO::new(BufReader::new(stream.try_clone().unwrap())));
                    writers.push(CountingIO::new(BufWriter::new(stream)));
                },
                Err(e) => {
                    // println!("Failed to connect: {}. Retrying in 1 second...", e);
                    thread::sleep(Duration::from_secs(1));
                },
            }
        }
    }
    (IMuxSync::new(readers), IMuxSync::new(writers))
}

pub fn server_connect(
    addr: &str,
) -> (
    IMuxSync<CountingIO<BufReader<TcpStream>>>,
    IMuxSync<CountingIO<BufWriter<TcpStream>>>,
) {
    // let listener = TcpListener::bind(addr).unwrap();
    let listener = loop {
        match TcpListener::bind(addr) {
            Ok(listener) => {
                break listener;
            },
            Err(e) => {
                thread::sleep(Duration::from_secs(1));
            },
        }
    };

    let mut incoming = listener.incoming();
    let mut readers = Vec::with_capacity(16);
    let mut writers = Vec::with_capacity(16);
    for _ in 0..16 {
        // let stream = incoming.next().unwrap().unwrap();
        // readers.push(CountingIO::new(BufReader::new(stream.try_clone().unwrap())));
        // writers.push(CountingIO::new(BufWriter::new(stream)));
        let stream = match incoming.next().unwrap() {
            Ok(stream) => stream,
            Err(e) => {
                // Handle the error, perhaps break or continue depending on the desired behavior
                panic!("Failed to accept connection: {}", e);
            },
        };
        readers.push(CountingIO::new(BufReader::new(match stream.try_clone() {
            Ok(cloned_stream) => cloned_stream,
            Err(e) => {
                // Handle the cloning error
                panic!("Failed to clone the stream: {}", e);
            },
        })));

        writers.push(CountingIO::new(BufWriter::new(stream)));
    }
    (IMuxSync::new(readers), IMuxSync::new(writers))
}

pub fn nn_client<R: RngCore + CryptoRng>(
    server_addr: &str,
    architecture: &NeuralArchitecture<TenBitAS, TenBitExpFP>,
    input: Input<TenBitExpFP>,
    rng: &mut R,
    batch_id: u16,
    batch_size: u16,
    cores: u16,
    memory: u16,
    network_name: &str,
    tiled: bool,
    tile_size: u64,
) -> Input<TenBitExpFP> {
    let (client_state, offline_read, offline_write) = {
        let (mut reader, mut writer) = client_connect(server_addr);
        (
            NNProtocol::offline_client_protocol(
                &mut reader,
                &mut writer,
                &architecture,
                rng,
                batch_id,
                batch_size,
                cores,
                memory,
                network_name,
                tiled,
                tile_size,
            )
            .unwrap(),
            reader.count(),
            writer.count(),
        )
    };

    let (client_output, online_read, online_write) = {
        let (mut reader, mut writer) = client_connect(server_addr);
        (
            NNProtocol::online_client_protocol(
                &mut reader,
                &mut writer,
                &input,
                &architecture,
                &client_state,
                batch_id,
                batch_size,
                cores,
                memory,
                network_name,
                tiled,
                tile_size,
            )
            .unwrap(),
            reader.count(),
            writer.count(),
        )
    };

    let offline_communication = CommunicationData {
        reads: offline_read,
        writes: offline_write,
    };

    let online_communication = CommunicationData {
        reads: online_read,
        writes: online_write,
    };

    let offline_communication_file = csv_file_name_network(
        network_name,
        "client",
        "offline",
        batch_id as u64,
        batch_size as u64,
        cores as u64,
        memory as u64,
    );
    let online_communication_file = csv_file_name_network(
        network_name,
        "client",
        "online",
        batch_id as u64,
        batch_size as u64,
        cores as u64,
        memory as u64,
    );
    write_to_csv(&offline_communication, &offline_communication_file);
    write_to_csv(&online_communication, &online_communication_file);

    add_to_trace!(|| "Offline Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        offline_read, offline_write
    ));
    add_to_trace!(|| "Online Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        online_read, online_write
    ));
    client_output
}

pub fn nn_server<R: RngCore + CryptoRng>(
    server_addr: &str,
    nn: &NeuralNetwork<TenBitAS, TenBitExpFP>,
    rng: &mut R,
    batch_id: u16,
    batch_size: u16,
    cores: u16,
    memory: u16,
    network_name: &str,
    tiled: bool,
    tile_size: u64,
) {
    let (server_state, offline_read, offline_write) = {
        let (mut reader, mut writer) = server_connect(server_addr);
        let res = match NNProtocol::offline_server_protocol(
            &mut reader,
            &mut writer,
            &nn,
            rng,
            batch_id,
            batch_size,
            cores,
            memory,
            network_name,
            tiled,
            tile_size,
        ) {
            Ok(res) => res,
            Err(e) => {
                panic!("there was an error running offline server {}", e);
            },
        };
        (res, reader.count(), writer.count())
    };

    let (_, online_read, online_write) = {
        let (mut reader, mut writer) = server_connect(server_addr);
        (
            NNProtocol::online_server_protocol(
                &mut reader,
                &mut writer,
                &nn,
                &server_state,
                batch_id,
                batch_size,
                cores,
                memory,
                network_name,
                tiled,
                tile_size,
            )
            .unwrap(),
            reader.count(),
            writer.count(),
        )
    };

    let offline_communication = CommunicationData {
        reads: offline_read,
        writes: offline_write,
    };

    let online_communication = CommunicationData {
        reads: online_read,
        writes: online_write,
    };
    let offline_communication_file = csv_file_name_network(
        network_name,
        "server",
        "offline",
        batch_id as u64,
        batch_size as u64,
        cores as u64,
        memory as u64,
    );
    let online_communication_file = csv_file_name_network(
        network_name,
        "server",
        "online",
        batch_id as u64,
        batch_size as u64,
        cores as u64,
        memory as u64,
    );
    write_to_csv(&offline_communication, &offline_communication_file);
    write_to_csv(&online_communication, &online_communication_file);

    add_to_trace!(|| "Offline Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        offline_read, offline_write
    ));
    add_to_trace!(|| "Online Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        online_read, online_write
    ));
}

pub fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -1.0 } else { 1.0 };
    let float: f64 = rng.gen();
    let f = TenBitExpFP::truncate_float(float * mul);
    let n = TenBitExpFP::from(f);
    (f, n)
}

fn sample_conv_layer<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    input_dims: (usize, usize, usize, usize),
    kernel_dims: (usize, usize, usize, usize),
    stride: usize,
    padding: Padding,
    rng: &mut R,
) -> (
    LinearLayer<TenBitAS, TenBitExpFP>,
    LinearLayer<TenBitExpFP, TenBitExpFP>,
) {
    let mut kernel = Kernel::zeros(kernel_dims);
    let mut bias = Kernel::zeros((kernel_dims.0, 1, 1, 1));
    kernel
        .iter_mut()
        .for_each(|ker_i| *ker_i = generate_random_number(rng).1);
    bias.iter_mut()
        .for_each(|bias_i| *bias_i = generate_random_number(rng).1);
    let layer_params = match vs {
        Some(vs) => Conv2dParams::<TenBitAS, _>::new_with_gpu(
            vs,
            padding,
            stride,
            kernel.clone(),
            bias.clone(),
        ),
        None => Conv2dParams::<TenBitAS, _>::new(padding, stride, kernel.clone(), bias.clone()),
    };
    let output_dims = layer_params.calculate_output_size(input_dims);
    let layer_dims = LayerDims {
        input_dims,
        output_dims,
    };
    let layer = LinearLayer::Conv2d {
        dims: layer_dims,
        params: layer_params,
    };

    let pt_layer_params =
        Conv2dParams::<TenBitExpFP, _>::new(padding, stride, kernel.clone(), bias.clone());
    let pt_layer = LinearLayer::Conv2d {
        dims: layer_dims,
        params: pt_layer_params,
    };
    (layer, pt_layer)
}

fn sample_fc_layer<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    input_dims: (usize, usize, usize, usize),
    out_chn: usize,
    rng: &mut R,
) -> (
    LinearLayer<TenBitAS, TenBitExpFP>,
    LinearLayer<TenBitExpFP, TenBitExpFP>,
) {
    let weight_dims = (out_chn, input_dims.1, input_dims.2, input_dims.3);
    let mut weights = Kernel::zeros(weight_dims);
    weights
        .iter_mut()
        .for_each(|w_i| *w_i = generate_random_number(rng).1);

    let bias_dims = (out_chn, 1, 1, 1);
    let mut bias = Kernel::zeros(bias_dims);
    bias.iter_mut()
        .for_each(|w_i| *w_i = generate_random_number(rng).1);

    let pt_weights = weights.clone();
    let pt_bias = bias.clone();
    let params = match vs {
        Some(vs) => FullyConnectedParams::new_with_gpu(vs, weights, bias),
        None => FullyConnectedParams::new(weights, bias),
    };
    let output_dims = params.calculate_output_size(input_dims);
    let dims = LayerDims {
        input_dims,
        output_dims,
    };
    let pt_params = FullyConnectedParams::new(pt_weights, pt_bias);
    let layer = LinearLayer::FullyConnected { dims, params };
    let pt_layer = LinearLayer::FullyConnected {
        dims,
        params: pt_params,
    };
    (layer, pt_layer)
}

#[allow(dead_code)]
fn sample_iden_layer(
    input_dims: (usize, usize, usize, usize),
) -> (
    LinearLayer<TenBitAS, TenBitExpFP>,
    LinearLayer<TenBitExpFP, TenBitExpFP>,
) {
    let output_dims = input_dims;
    let layer_dims = LayerDims {
        input_dims,
        output_dims,
    };
    let layer = LinearLayer::Identity { dims: layer_dims };
    let pt_layer = LinearLayer::Identity { dims: layer_dims };
    (layer, pt_layer)
}

#[allow(dead_code)]
fn sample_avg_pool_layer(
    input_dims: (usize, usize, usize, usize),
    (pool_h, pool_w): (usize, usize),
    stride: usize,
) -> LinearLayer<TenBitAS, TenBitExpFP> {
    let size = (pool_h * pool_w) as f64;
    let avg_pool_params = AvgPoolParams::new(pool_h, pool_w, stride, TenBitExpFP::from(1.0 / size));
    let pool_dims = LayerDims {
        input_dims,
        output_dims: avg_pool_params.calculate_output_size(input_dims),
    };

    LinearLayer::AvgPool {
        dims: pool_dims,
        params: avg_pool_params,
    }
}

fn add_activation_layer(nn: &mut NeuralNetwork<TenBitAS, TenBitExpFP>, relu_layers: &[usize]) {
    let cur_input_dims = nn.layers.last().as_ref().unwrap().output_dimensions();
    let layer_dims = LayerDims {
        input_dims: cur_input_dims,
        output_dims: cur_input_dims,
    };
    let num_layers_so_far = nn.layers.len();
    let is_relu = relu_layers.contains(&num_layers_so_far);
    let layer = if is_relu {
        Layer::NLL(NonLinearLayer::ReLU(layer_dims))
    } else {
        let activation_poly_coefficients = vec![
            TenBitExpFP::from(0.2),
            TenBitExpFP::from(0.5),
            TenBitExpFP::from(0.2),
        ];
        let poly = Polynomial::new(activation_poly_coefficients);
        let poly_layer = NonLinearLayer::PolyApprox {
            dims: layer_dims,
            poly,
            _v: std::marker::PhantomData,
        };
        Layer::NLL(poly_layer)
    };
    nn.layers.push(layer);
}

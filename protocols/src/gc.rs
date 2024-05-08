use crate::{AdditiveShare, InMessage, OutMessage};
use algebra::{
    fields::PrimeField,
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::Fp64Parameters,
    BigInteger64, FpParameters, UniformRandom,
};
use crypto_primitives::{
    gc::{
        fancy_garbling,
        fancy_garbling::{
            circuit::{Circuit, CircuitBuilder},
            Encoder, GarbledCircuit, Wire,
        },
    },
    Share,
};
use io_utils::{counting::CountingIO, imux::IMuxSync};
use ocelot::ot::{AlszReceiver as OTReceiver, AlszSender as OTSender, Receiver, Sender};
use rand::{CryptoRng, RngCore};
use rayon::prelude::*;
use scuttlebutt::Channel;
use std::{
    convert::TryFrom,
    io::{BufReader, BufWriter, Read, Write},
    marker::PhantomData,
    net::{TcpListener, TcpStream},
    thread, time,
};

use crate::csv_timing::*;
use crate::ClientOfflineNonLinear;
use crate::ClientOnlineNonLinear;
use crate::CommClientOfflineNonLinear;
use crate::CommClientOnlineNonLinear;
use crate::CommServerOfflineNonLinear;
use crate::CommServerOnlineNonLinear;
use crate::ServerOfflineNonLinear;
use crate::ServerOnlineNonLinear;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::Duration;

pub mod tiling;
use tiling::tile_config;

#[derive(Default)]
pub struct ReluProtocol<P: FixedPointParameters> {
    _share: PhantomData<P>,
}

pub struct ReluProtocolType;

pub type ServerGcMsgSend<'a> = OutMessage<'a, (&'a [GarbledCircuit], &'a [Wire]), ReluProtocolType>;
pub type ClientGcMsgRcv = InMessage<(Vec<GarbledCircuit>, Vec<Wire>), ReluProtocolType>;

// The message is a slice of (vectors of) input labels;
pub type ServerLabelMsgSend<'a> = OutMessage<'a, [Vec<Wire>], ReluProtocolType>;
pub type ClientLabelMsgRcv = InMessage<Vec<Vec<Wire>>, ReluProtocolType>;

pub fn make_relu<P: FixedPointParameters>() -> Circuit
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    let mut b = CircuitBuilder::new();
    crypto_primitives::gc::relu::<P>(&mut b, 1).unwrap();
    b.finish()
}

pub fn u128_from_share<P: FixedPointParameters>(s: AdditiveShare<P>) -> u128
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = BigInteger64>,
{
    let s: u64 = s.inner.inner.into_repr().into();
    s.into()
}

//--------------------------------- Server State ---------------------------------
#[derive(Serialize, Deserialize)]
pub struct ServerState<P: FixedPointParameters>
where
    P::Field: Serialize, // Ensure P::Field implements Serialize
{
    pub encoders: Vec<Encoder>,
    pub output_randomizers: Vec<P::Field>,
}

// impl<P> ServerState<P>
// where
//     P: FixedPointParameters,
//     P::Field: Serialize,
// {
//     pub fn save(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
//         let encoded = bincode::serialize(self)?;
//         let mut file = File::create(filename)?;
//         file.write_all(&encoded)?;
//         Ok(())
//     }

//     pub fn load(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
//         let mut file = File::open(filename)?;
//         let mut encoded = Vec::new();
//         file.read_to_end(&mut encoded)?;
//         let state = bincode::deserialize(&encoded)?;
//         Ok(state)
//     }
// }

// fn save_server_state<P>(state: &ServerState<P>, filename: &str)
// where
//     P: FixedPointParameters,
//     P::Field: Serialize,
// {
//     let trials = 50;
//     let sleep_time = 100; //ms
//     for _ in 0..trials {
//         match state.save(filename) {
//             Ok(_) => return,
//             Err(_) => {
//                 sleep(Duration::from_millis(sleep_time));
//             },
//         }
//     }
//     panic!("Failed to save server state after {} tries!", trials);
// }

// fn load_server_state<P>(filename: &str) -> ServerState<P>
// where
//     P: FixedPointParameters,
//     P::Field: Serialize + for<'de> Deserialize<'de>,
// {
//     let trials = 50;
//     let sleep_time = 100; //ms
//     for _ in 0..trials {
//         match ServerState::load(filename) {
//             Ok(state) => return state,
//             Err(_) => {
//                 sleep(Duration::from_millis(sleep_time));
//             },
//         }
//     }
//     panic!("Failed to load server state after {} tries!", trials);
// }

//--------------------------------- Server State that will be saved ---------------------------------
#[derive(Serialize, Deserialize)]
pub struct ServerTile {
    pub encoders: Vec<Encoder>,
}

impl ServerTile {
    // Saves the instance to a file as binary
    pub fn save(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(self)?;
        let mut file = File::create(filename)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    // Loads an instance from a binary file
    pub fn load(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let instance: ServerTile = bincode::deserialize(&buffer)?;
        Ok(instance)
    }
}

fn save_server_state(state: &ServerTile, filename: &str) {
    let trials = 50;
    let sleep_time = 100; //ms
    for _ in 0..trials {
        match state.save(filename) {
            Ok(_) => return,
            Err(_) => {
                sleep(Duration::from_millis(sleep_time));
            },
        }
    }
    panic!("Failed to save server state after {} tries!", trials);
}

fn load_server_state(filename: &str) -> ServerTile {
    let trials = 50;
    let sleep_time = 100; //ms
    for _ in 0..trials {
        match ServerTile::load(filename) {
            Ok(state) => return state,
            Err(_) => {
                sleep(Duration::from_millis(sleep_time));
            },
        }
    }
    panic!("Failed to load server state after {} tries!", trials);
}

//--------------------------------- Client State ---------------------------------
#[derive(Serialize, Deserialize)]
pub struct ClientState {
    pub gc_s: Vec<GarbledCircuit>,
    pub server_randomizer_labels: Vec<Wire>,
    pub client_input_labels: Vec<Wire>,
}
impl ClientState {
    pub fn save(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(self)?;
        let mut file = File::create(filename)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    pub fn load(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(filename)?;
        let mut encoded = Vec::new();
        file.read_to_end(&mut encoded)?;
        let state: ClientState = bincode::deserialize(&encoded)?;
        Ok(state)
    }
}
fn save_client_state(state: &ClientState, filename: &str) {
    let trials = 50;
    let sleep_time = 100; //ms
    for _ in 0..trials {
        match state.save(filename) {
            Ok(_) => return,
            Err(_) => {
                sleep(Duration::from_millis(sleep_time));
            },
        }
    }
    panic!("Failed to save client state after {} tries!", trials);
}
fn load_client_state(filename: &str) -> ClientState {
    let trials = 50;
    let sleep_time = 100; //ms
    for _ in 0..trials {
        match ClientState::load(filename) {
            Ok(state) => return state,
            Err(_) => {
                sleep(Duration::from_millis(sleep_time));
            },
        }
    }
    panic!("Failed to load client state after {} tries!", trials);
}

//--------------------------------- Protocols ---------------------------------
impl<P: FixedPointParameters> ReluProtocol<P>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = BigInteger64>,
{
    #[inline]
    pub fn size_of_client_inputs() -> usize {
        make_relu::<P>().num_evaluator_inputs()
    }

    // pub fn offline_server_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
    pub fn offline_server_protocol<RNG: CryptoRng + RngCore>(
        // reader: &mut IMuxSync<R>,
        // writer: &mut IMuxSync<W>,
        reader: &mut IMuxSync<CountingIO<BufReader<TcpStream>>>,
        writer: &mut IMuxSync<CountingIO<BufWriter<TcpStream>>>,
        number_of_relus: usize,
        rng: &mut RNG,
        batch_id: u16,
        batch_size: u16,
        cores: u16,
        memory: u16,
        network_name: &str,
        tiled: bool,
        tile_size: u64,
        relus_per_layer: &Vec<usize>,
    ) -> Result<ServerState<P>, bincode::Error> {
        //--------------------------------- products of this protocol ---------------------------------
        let mut encoders = Vec::new();
        let mut output_randomizers = Vec::new();

        //--------------------------------- layer-wise processing ---------------------------------
        for (layer_index, relus) in relus_per_layer.iter().enumerate() {
            //--------------------------------- profiling ---------------------------------
            let mut timing = ServerOfflineNonLinear {
                garbling: 0,
                encoding: 0,
                OT_communication: 0,
                GC_communication: 0,
                IO_write: 0,
                total_duration: 0,
            };
            let mut comm = CommServerOfflineNonLinear {
                gc_write: 0,
                ot_write: 0,
                ot_read: 0,
            };
            let (mut r_before, mut w_before) = (0, 0);
            let start_time = timer_start!(|| "ReLU offline protocol");
            let total_time = Instant::now();

            //--------------------------------- tiling details ---------------------------------
            let tiling_configuration = tiling::configure_tiling(relus.clone(), tile_size as usize);
            let relu_chunk_size = tiling_configuration.relu_chunk_size;
            let relu_chunks = tiling_configuration.relu_chunks;
            let leftovers = tiling_configuration.leftovers;
            let mut previous_handle: Option<thread::JoinHandle<()>> = None;

            //--------------------------------- tile-wise processing ---------------------------------
            for chunk_index in 0..relu_chunks {
                let current_chunk_size = match chunk_index {
                    x if x == relu_chunks - 1 && leftovers != 0 => leftovers,
                    _ => relu_chunk_size,
                };
                let start_index = relu_chunk_size * chunk_index;
                let end_index = start_index + current_chunk_size;

                //--------------------------------- Garbling ---------------------------------
                let garbling_time = Instant::now();
                let mut gc_s = Vec::with_capacity(current_chunk_size);
                let mut tile_encoders = Vec::with_capacity(current_chunk_size); // this tile's portion of product
                let p = (<<P::Field as PrimeField>::Params>::MODULUS.0).into();
                let c = make_relu::<P>();
                let garble_time = timer_start!(|| "Garbling");
                (0..current_chunk_size)
                    .into_par_iter()
                    .map(|_| {
                        let mut c = c.clone();
                        let (en, gc) = fancy_garbling::garble(&mut c).unwrap();
                        (en, gc)
                    })
                    .unzip_into_vecs(&mut tile_encoders, &mut gc_s);
                timing.garbling += garbling_time.elapsed().as_micros() as u64;
                timer_end!(garble_time);

                //--------------------------------- Encoding ---------------------------------
                let encode_time = timer_start!(|| "Encoding inputs");
                let encoding_time = Instant::now();
                let num_garbler_inputs = c.num_garbler_inputs();
                let num_evaluator_inputs = c.num_evaluator_inputs();

                let zero_inputs = vec![0u16; num_evaluator_inputs]; //used for OT
                let one_inputs = vec![1u16; num_evaluator_inputs]; //used for OT
                let mut labels = Vec::with_capacity(current_chunk_size * num_evaluator_inputs); //used for OT

                let mut randomizer_labels = Vec::with_capacity(current_chunk_size); // gets sent to client
                let mut tile_output_randomizers = Vec::with_capacity(current_chunk_size); // tile's portion of product

                for enc in tile_encoders.iter() {
                    let r = P::Field::uniform(rng);
                    tile_output_randomizers.push(r);
                    let r_bits: u64 = ((-r).into_repr()).into();
                    let r_bits = fancy_garbling::util::u128_to_bits(
                        r_bits.into(),
                        crypto_primitives::gc::num_bits(p),
                    );
                    for w in ((num_garbler_inputs / 2)..num_garbler_inputs)
                        .zip(r_bits)
                        .map(|(i, r_i)| enc.encode_garbler_input(r_i, i))
                    {
                        randomizer_labels.push(w);
                    }

                    //--------------------------------- labels used for OT ---------------------------------
                    let all_zeros = enc.encode_evaluator_inputs(&zero_inputs);
                    let all_ones = enc.encode_evaluator_inputs(&one_inputs);
                    all_zeros
                        .into_iter()
                        .zip(all_ones)
                        .for_each(|(label_0, label_1)| {
                            labels.push((label_0.as_block(), label_1.as_block()))
                        });
                }
                timing.encoding += encoding_time.elapsed().as_micros() as u64;

                timer_end!(encode_time);

                //--------------------------------- GC communication ---------------------------------
                w_before = writer.count(); // communication done so far
                let send_gc_time = timer_start!(|| "Sending GCs");
                let gc_communication_time = Instant::now();
                let randomizer_label_per_relu = if current_chunk_size == 0 {
                    8192
                } else {
                    randomizer_labels.len() / current_chunk_size
                };
                for msg_contents in gc_s
                    .chunks(8192)
                    .zip(randomizer_labels.chunks(randomizer_label_per_relu * 8192))
                {
                    let sent_message = ServerGcMsgSend::new(&msg_contents);
                    crate::bytes::serialize(writer, &sent_message)?;
                }
                timing.GC_communication += gc_communication_time.elapsed().as_micros() as u64;
                timer_end!(send_gc_time);
                comm.gc_write = writer.count() - w_before; // GC writes in bytes

                //--------------------------------- OT communication ---------------------------------
                {
                    let r_const = reader.get_ref().remove(0);
                    let w_const = writer.get_ref().remove(0);
                    r_before = r_const.count();
                    w_before = w_const.count();
                }
                if current_chunk_size != 0 {
                    let r = reader.get_mut_ref().remove(0);
                    let w = writer.get_mut_ref().remove(0);
                    let ot_time = timer_start!(|| "OTs");
                    let ot_communication_time = Instant::now();
                    let mut channel = Channel::new(r, w);
                    let mut ot = OTSender::init(&mut channel, rng).unwrap();
                    ot.send(&mut channel, labels.as_slice(), rng).unwrap();
                    timing.OT_communication += ot_communication_time.elapsed().as_micros() as u64;
                    timer_end!(ot_time);
                }
                {
                    let r_const = reader.get_ref().remove(0);
                    let w_const = writer.get_ref().remove(0);
                    comm.ot_read = r_const.count() - r_before;
                    comm.ot_write = w_const.count() - w_before;
                }

                //--------------------------------- stage for return and/or save the products ---------------------------------
                output_randomizers.extend(tile_output_randomizers);
                if !tiled {
                    encoders.extend(tile_encoders);
                } else {
                    let server_IO_write = Instant::now();
                    let tile_state: ServerTile = ServerTile {
                        encoders: tile_encoders,
                    };
                    // let tile_state: ServerState<P> = ServerState {
                    //     encoders: tile_encoders,
                    //     output_randomizers: tile_output_randomizers,
                    // };

                    //--------------------------------- get a file token for saving offline data tile ---------------------------------
                    let file_name = tiling::get_file_name(
                        "/mnt/mohammad/delphi_bench",
                        "offline",
                        "server",
                        &batch_id,
                        &(layer_index.to_string()),
                        &(chunk_index.to_string()),
                    );
                    if let Some(handle1) = previous_handle.take() {
                        // if there is a thread saving, wait for it
                        handle1.join().unwrap();
                    }
                    let handle = thread::spawn(move || {
                        // spawn a new thread to save
                        save_server_state(&tile_state, &file_name);
                    });
                    previous_handle = Some(handle); // keep the handle for next tile iteration to wait for it
                    let server_IO_write_local = server_IO_write.elapsed().as_micros() as u64;
                    timing.IO_write += server_IO_write_local;
                }
            }
            timing.total_duration += total_time.elapsed().as_micros() as u64;
            timer_end!(start_time);

            //--------------------------------- save profiling data for this layer ---------------------------------
            let file_name = csv_file_name(
                network_name,
                "server",
                "offline",
                "non_linear",
                layer_index as u64,
                batch_id as u64,
                batch_size as u64,
                cores as u64,
                memory as u64,
            );
            write_to_csv(&timing, &file_name);
            let comm_file_name = csv_file_name_comm(
                network_name,
                "server",
                "offline",
                "non_linear",
                layer_index as u64,
                batch_id as u64,
                batch_size as u64,
                cores as u64,
                memory as u64,
            );
            write_to_csv(&comm, &comm_file_name);
        }

        Ok(ServerState {
            encoders,
            output_randomizers,
        })
    }

    // pub fn offline_client_protocol<R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
    pub fn offline_client_protocol<RNG: RngCore + CryptoRng>(
        // reader: &mut IMuxSync<R>,
        // writer: &mut IMuxSync<W>,
        reader: &mut IMuxSync<CountingIO<BufReader<TcpStream>>>,
        writer: &mut IMuxSync<CountingIO<BufWriter<TcpStream>>>,
        number_of_relus: usize,
        shares: &[AdditiveShare<P>],
        rng: &mut RNG,
        batch_id: u16,
        batch_size: u16,
        cores: u16,
        memory: u16,
        network_name: &str,
        tiled: bool,
        tile_size: u64,
        relus_per_layer: &Vec<usize>,
    ) -> Result<ClientState, bincode::Error> {
        //--------------------------------- products of this protocol ---------------------------------
        let mut gc_s = Vec::new();
        let mut r_wires = Vec::new();
        let mut labels = Vec::new();

        let mut relus_processed = 0;
        //--------------------------------- layer-wise processing ---------------------------------
        for (layer_index, relus) in relus_per_layer.iter().enumerate() {
            //--------------------------------- profiling ---------------------------------
            let mut timing = ClientOfflineNonLinear {
                OT_communication: 0,
                GC_communication: 0,
                IO_write: 0,
                total_duration: 0,
            };
            let mut comm = CommClientOfflineNonLinear {
                gc_read: 0,
                ot_write: 0,
                ot_read: 0,
            };
            let (mut r_before, mut w_before) = (0, 0);
            use fancy_garbling::util::*;
            let start_time = timer_start!(|| "ReLU offline protocol");
            let total_time = Instant::now();

            let p = u128::from(<<P::Field as PrimeField>::Params>::MODULUS.0);
            let field_size = crypto_primitives::gc::num_bits(p);

            //--------------------------------- tiling details ---------------------------------
            let tiling_configuration = tiling::configure_tiling(relus.clone(), tile_size as usize);
            let relu_chunk_size = tiling_configuration.relu_chunk_size;
            let relu_chunks = tiling_configuration.relu_chunks;
            let leftovers = tiling_configuration.leftovers;
            let mut previous_handle: Option<thread::JoinHandle<()>> = None;

            //--------------------------------- tile-wise processing ---------------------------------
            for chunk_index in 0..relu_chunks {
                let current_chunk_size = match chunk_index {
                    x if x == relu_chunks - 1 && leftovers != 0 => leftovers,
                    _ => relu_chunk_size,
                };
                let start_index = relu_chunk_size * chunk_index;
                let end_index = start_index + current_chunk_size;

                //--------------------------------- offseting the tile input data ---------------------------------
                let tile_shares = if !tiled {
                    &shares[relus_processed..(relus_processed + relus)]
                } else {
                    &shares[(relus_processed + start_index)..(relus_processed + end_index)]
                };

                //--------------------------------- GC receiving ---------------------------------
                r_before = reader.count(); // communication done so far
                let rcv_gc_time = timer_start!(|| "Receiving GCs");
                let gc_communication_time = Instant::now();
                let mut tile_gc_s = Vec::with_capacity(current_chunk_size); // this tile's portion of product
                let mut tile_r_wires = Vec::with_capacity(current_chunk_size); // this tile's portion of product
                let num_chunks = (current_chunk_size as f64 / 8192.0).ceil() as usize;
                for i in 0..num_chunks {
                    let in_msg: ClientGcMsgRcv = crate::bytes::deserialize(reader)?;
                    let (gc_chunks, r_wire_chunks) = in_msg.msg();
                    if i < (num_chunks - 1) {
                        assert_eq!(gc_chunks.len(), 8192);
                    }
                    tile_gc_s.extend(gc_chunks);
                    tile_r_wires.extend(r_wire_chunks);
                }
                timing.GC_communication += gc_communication_time.elapsed().as_micros() as u64;
                timer_end!(rcv_gc_time);
                comm.gc_read = reader.count() - r_before; // GC writes in bytes

                //--------------------------------- OT communication ---------------------------------
                {
                    let r_const = reader.get_ref().remove(0);
                    let w_const = writer.get_ref().remove(0);
                    r_before = r_const.count();
                    w_before = w_const.count();
                }
                let OT_communication_time = Instant::now();
                assert_eq!(tile_gc_s.len(), current_chunk_size);
                let bs = tile_shares
                    .iter()
                    .flat_map(|s| u128_to_bits(u128_from_share(*s), field_size))
                    .map(|b| b == 1)
                    .collect::<Vec<_>>();

                let tile_labels = if current_chunk_size != 0 {
                    let r = reader.get_mut_ref().remove(0);
                    let w = writer.get_mut_ref().remove(0);

                    let ot_time = timer_start!(|| "OTs");
                    let mut channel = Channel::new(r, w);
                    let mut ot = OTReceiver::init(&mut channel, rng).expect("should work");
                    let tile_labels = ot
                        .receive(&mut channel, bs.as_slice(), rng)
                        .expect("should work");
                    let tile_labels = tile_labels
                        .into_iter()
                        .map(|l| Wire::from_block(l, 2))
                        .collect::<Vec<_>>();
                    timer_end!(ot_time);
                    tile_labels
                } else {
                    Vec::new()
                };
                timing.OT_communication += OT_communication_time.elapsed().as_micros() as u64;
                {
                    let r_const = reader.get_ref().remove(0);
                    let w_const = writer.get_ref().remove(0);
                    comm.ot_read = r_const.count() - r_before;
                    comm.ot_write = w_const.count() - w_before;
                }

                //--------------------------------- stage for return and/or save as the products ---------------------------------
                if !tiled {
                    gc_s.extend(tile_gc_s);
                    r_wires.extend(tile_r_wires);
                    labels.extend(tile_labels);
                } else {
                    let client_IO_write = Instant::now();
                    let tile_state = ClientState {
                        gc_s: tile_gc_s,
                        server_randomizer_labels: tile_r_wires,
                        client_input_labels: tile_labels,
                    };
                    //--------------------------------- get a file token for saving offline data tile ---------------------------------
                    let file_name = tiling::get_file_name(
                        "/mnt/mohammad/delphi_bench",
                        "offline",
                        "client",
                        &batch_id,
                        &(layer_index.to_string()),
                        &(chunk_index.to_string()),
                    );
                    if let Some(handle) = previous_handle.take() {
                        // if there is a tile being saved, wait for it
                        handle.join().unwrap();
                    }
                    let handle = thread::spawn(move || {
                        // spawn a new thread for saving the tile
                        save_client_state(&tile_state, &file_name);
                    });
                    previous_handle = Some(handle);
                    timing.IO_write += client_IO_write.elapsed().as_micros() as u64;
                }
            }

            //--------------------------------- save profiling data for this layer ---------------------------------
            timing.total_duration += total_time.elapsed().as_micros() as u64;
            timer_end!(start_time);
            let file_name = csv_file_name(
                network_name,
                "client",
                "offline",
                "non_linear",
                layer_index as u64,
                batch_id.into(),
                batch_size as u64,
                cores as u64,
                memory as u64,
            );
            write_to_csv(&timing, &file_name);
            let comm_file_name = csv_file_name_comm(
                network_name,
                "client",
                "offline",
                "non_linear",
                layer_index as u64,
                batch_id as u64,
                batch_size as u64,
                cores as u64,
                memory as u64,
            );
            write_to_csv(&comm, &comm_file_name);

            relus_processed += relus;
        }

        Ok(ClientState {
            gc_s,
            server_randomizer_labels: r_wires,
            client_input_labels: labels,
        })
    }

    // pub fn online_server_protocol<'a, W: Write + Send>(
    pub fn online_server_protocol(
        // writer: &mut IMuxSync<W>,
        writer: &mut IMuxSync<CountingIO<BufWriter<TcpStream>>>,
        shares: &[AdditiveShare<P>],
        encoders: &[Encoder],
        batch_id: u16,
        batch_size: u16,
        cores: u16,
        memory: u16,
        conv_id: u16,
        network_name: &str,
        tiled: bool,
        tile_size: u64,
        relu_id: u16,
        relus: usize,
    ) {
        let (tx, rx) = mpsc::channel();

        //--------------------------------- profiling ---------------------------------
        let mut timing = ServerOnlineNonLinear {
            encoding: 0,
            communication: 0,
            IO_read: 0,
            total_duration: 0,
        };
        let mut comm = CommServerOnlineNonLinear {
            encoded_labels_write: 0,
        };
        let (mut r_before, mut w_before) = (0, 0);
        let start_time = timer_start!(|| "ReLU online protocol");
        let total_time = Instant::now();
        let p = u128::from(u64::from(P::Field::characteristic()));

        //--------------------------------- tiling details ---------------------------------
        let tiling_configuration = tiling::configure_tiling(relus, tile_size as usize);
        let relu_chunk_size = tiling_configuration.relu_chunk_size;
        let relu_chunks = tiling_configuration.relu_chunks;
        let leftovers = tiling_configuration.leftovers;

        //--------------------------------- state to be loaded if tiling is done ---------------------------------
        let mut garbler_offline_data: ServerTile = ServerTile {
            encoders: Default::default(),
        };
        let mut previous_handle: Option<thread::JoinHandle<()>> = None;

        //--------------------------------- tile-wise processing ---------------------------------
        for chunk_index in 0..relu_chunks {
            let current_chunk_size = match chunk_index {
                x if x == relu_chunks - 1 && leftovers != 0 => leftovers,
                _ => relu_chunk_size,
            };
            let start_index = relu_chunk_size * chunk_index;
            let end_index = start_index + current_chunk_size;

            //--------------------------------- loading the state tile ---------------------------------
            if tiled {
                let server_IO_read_instance = Instant::now();
                if chunk_index == 0 {
                    // if first tile, load the data
                    let file_name = tiling::get_file_name(
                        "/mnt/mohammad/delphi_bench",
                        "offline",
                        "server",
                        &batch_id,
                        &(relu_id.to_string()),
                        &(chunk_index.to_string()),
                    );
                    garbler_offline_data = load_server_state(&file_name);
                }
                if chunk_index != relu_chunks - 1 {
                    // if not last tile, pre-fetch the next tile
                    let next_file_name = tiling::get_file_name(
                        "/mnt/mohammad/delphi_bench",
                        "offline",
                        "server",
                        &batch_id,
                        &(relu_id.to_string()),
                        &((chunk_index + 1).to_string()),
                    );
                    let tx_clone = tx.clone();
                    thread::spawn(move || {
                        let next_garbler_offline_data: ServerTile =
                            load_server_state(&next_file_name);
                        tx_clone
                            .send(next_garbler_offline_data)
                            .expect("Failed to send garbler offline tile to main thread");
                    });
                }
                timing.IO_read += server_IO_read_instance.elapsed().as_micros() as u64;
            }

            //--------------------------------- offseting the tile input data ---------------------------------
            let tile_encoders = if !tiled {
                &encoders
            } else {
                garbler_offline_data.encoders.as_slice()
            };
            let tile_shares = if !tiled {
                &shares
            } else {
                &shares[start_index..end_index]
            };

            //--------------------------------- encoding ---------------------------------
            let encoding_time = timer_start!(|| "Encoding inputs");
            let server_encoding = Instant::now();
            let field_size = (p.next_power_of_two() * 2).trailing_zeros() as usize;
            let wires = tile_shares
                .iter()
                .map(|share| {
                    let share = u128_from_share(*share);
                    fancy_garbling::util::u128_to_bits(share, field_size)
                })
                .zip(tile_encoders)
                .map(|(share_bits, encoder)| encoder.encode_garbler_inputs(&share_bits))
                .collect::<Vec<Vec<_>>>();
            timing.encoding += server_encoding.elapsed().as_micros() as u64;
            timer_end!(encoding_time);

            //--------------------------------- communication ---------------------------------
            w_before = writer.count(); // communication done so far
            let send_time = timer_start!(|| "Sending inputs");
            let server_communication_time = Instant::now();
            let sent_message = ServerLabelMsgSend::new(wires.as_slice());
            crate::bytes::serialize(writer, &sent_message)
                .expect("sending encrypted labels from online server failed.");
            timing.communication += server_communication_time.elapsed().as_micros() as u64;
            timer_end!(send_time);
            comm.encoded_labels_write = writer.count() - w_before; // GC writes in bytes

            //--------------------------------- wait until offline_data of next iteration is ready ---------------------------------
            let io_duration_instance = Instant::now();
            if chunk_index != relu_chunks - 1 && tiled {
                match rx.recv() {
                    Ok(next_garbler_offline_data) => {
                        garbler_offline_data = next_garbler_offline_data
                    },
                    Err(e) => {
                        panic!("there was an issue when pre-fetching garbler data: {}", e)
                    },
                }
            }
            timing.IO_read += io_duration_instance.elapsed().as_micros() as u64;
        }

        //--------------------------------- save profiling data ---------------------------------
        timing.total_duration += total_time.elapsed().as_micros() as u64;
        timer_end!(start_time);
        let file_name = csv_file_name(
            network_name,
            "server",
            "online",
            "non_linear",
            conv_id as u64,
            batch_id as u64,
            batch_size as u64,
            cores as u64,
            memory as u64,
        );
        write_to_csv(&timing, &file_name);
        let comm_file_name = csv_file_name_comm(
            network_name,
            "server",
            "online",
            "non_linear",
            conv_id as u64,
            batch_id as u64,
            batch_size as u64,
            cores as u64,
            memory as u64,
        );
        write_to_csv(&comm, &comm_file_name);
    }

    /// Outputs shares for the next round's input.
    // pub fn online_client_protocol<R: Read + Send>(
    pub fn online_client_protocol(
        // reader: &mut IMuxSync<R>,
        reader: &mut IMuxSync<CountingIO<BufReader<TcpStream>>>,
        num_relus: usize,
        server_input_wires: &[Wire],
        client_input_wires: &[Wire],
        evaluators: &[GarbledCircuit],
        next_layer_randomizers: &[P::Field],
        batch_id: u16,
        batch_size: u16,
        cores: u16,
        memory: u16,
        conv_id: u16,
        network_name: &str,
        tiled: bool,
        tile_size: u64,
        relu_id: u16,
        relus: usize,
    ) -> Result<Vec<AdditiveShare<P>>, bincode::Error> {
        let (tx, rx) = mpsc::channel();

        //--------------------------------- products of this protocol ---------------------------------
        let mut results = Vec::new();

        //--------------------------------- profiling data ---------------------------------
        let mut timing = ClientOnlineNonLinear {
            GC_eval: 0,
            communication: 0,
            IO_read: 0,
            total_duration: 0,
        };
        let mut comm = CommClientOnlineNonLinear {
            encoded_labels_read: 0,
        };
        let (mut r_before, mut w_before) = (0, 0);
        let start_time = timer_start!(|| "ReLU online protocol");
        let total_time = Instant::now();

        //--------------------------------- tiling details ---------------------------------
        let tiling_configuration = tiling::configure_tiling(relus, tile_size as usize);
        let relu_chunk_size = tiling_configuration.relu_chunk_size;
        let relu_chunks = tiling_configuration.relu_chunks;
        let leftovers = tiling_configuration.leftovers;

        //--------------------------------- state to be loaded if tiling is done ---------------------------------
        let mut evaluator_offline_data: ClientState = ClientState {
            gc_s: Default::default(),
            server_randomizer_labels: Default::default(),
            client_input_labels: Default::default(),
        };

        //--------------------------------- tile-wise processing ---------------------------------
        for chunk_index in 0..relu_chunks {
            let current_chunk_size = match chunk_index {
                x if x == relu_chunks - 1 && leftovers != 0 => leftovers,
                _ => relu_chunk_size,
            };
            let start_index = relu_chunk_size * chunk_index;
            let end_index = start_index + current_chunk_size;

            //--------------------------------- loading the state tile ---------------------------------
            if tiled {
                let server_IO_read_instance = Instant::now();
                if chunk_index == 0 {
                    let file_name = tiling::get_file_name(
                        "/mnt/mohammad/delphi_bench",
                        "offline",
                        "client",
                        &batch_id,
                        &(relu_id.to_string()),
                        &(chunk_index.to_string()),
                    );
                    evaluator_offline_data = load_client_state(&file_name);
                }
                //--------------------------------- pre-fetch offline_data for next iteration ---------------------------------
                if chunk_index != relu_chunks - 1 {
                    let next_file_name = tiling::get_file_name(
                        "/mnt/mohammad/delphi_bench",
                        "offline",
                        "client",
                        &batch_id,
                        &(relu_id.to_string()),
                        &((chunk_index + 1).to_string()),
                    );
                    let tx_clone = tx.clone();
                    thread::spawn(move || {
                        let next_evaluator_offline_data: ClientState =
                            load_client_state(&next_file_name);
                        tx_clone
                            .send(next_evaluator_offline_data)
                            .expect("Failed to send tile to main thread");
                    });
                }
                timing.IO_read += server_IO_read_instance.elapsed().as_micros() as u64;
            }

            //--------------------------------- offseting the tile input data ---------------------------------
            let mut tile_server_input_wires = if !tiled {
                server_input_wires
            } else {
                &evaluator_offline_data.server_randomizer_labels
            };
            let mut tile_client_input_wires = if !tiled {
                client_input_wires
            } else {
                &evaluator_offline_data.client_input_labels
            };
            let mut tile_evaluators = if !tiled {
                evaluators
            } else {
                &evaluator_offline_data.gc_s
            };
            let mut tiled_next_layer_randomizers = if !tiled {
                next_layer_randomizers
            } else {
                &next_layer_randomizers[start_index..end_index]
            };

            //--------------------------------- communication ---------------------------------
            r_before = reader.count(); // communication done so far
            let rcv_time = timer_start!(|| "Receiving inputs");

            let communication_time = Instant::now();
            let in_msg: ClientLabelMsgRcv = crate::bytes::deserialize(reader)?;
            let mut garbler_wires = in_msg.msg();
            timing.communication += communication_time.elapsed().as_micros() as u64;
            timer_end!(rcv_time);
            comm.encoded_labels_read = reader.count() - r_before; // GC writes in bytes

            //--------------------------------- GC evaluation ---------------------------------
            let eval_time = timer_start!(|| "Evaluating GCs");
            let gc_eval_time = Instant::now();
            let c = make_relu::<P>();
            let num_evaluator_inputs = c.num_evaluator_inputs();
            let num_garbler_inputs = c.num_garbler_inputs();
            garbler_wires
                .iter_mut()
                .zip(tile_server_input_wires.chunks(num_garbler_inputs / 2))
                .for_each(|(w1, w2)| w1.extend_from_slice(w2));
            assert_eq!(current_chunk_size, garbler_wires.len());
            assert_eq!(
                num_evaluator_inputs * current_chunk_size,
                tile_client_input_wires.len()
            );
            // We access the input wires in reverse.
            let c = make_relu::<P>();
            let mut tile_results = tile_client_input_wires
                .par_chunks(num_evaluator_inputs)
                .zip(garbler_wires)
                .zip(tile_evaluators)
                .map(|((eval_inps, garbler_inps), gc)| {
                    let mut c = c.clone();
                    let result = gc
                        .eval(&mut c, &garbler_inps, eval_inps)
                        .expect("evaluation failed");
                    let result = fancy_garbling::util::u128_from_bits(result.as_slice());
                    FixedPoint::new(P::Field::from_repr(u64::try_from(result).unwrap().into()))
                        .into()
                })
                .collect::<Vec<AdditiveShare<P>>>();
            tile_results
                .iter_mut()
                .zip(tiled_next_layer_randomizers)
                .for_each(|(s, r)| *s = FixedPoint::<P>::randomize_local_share(s, r));
            timing.GC_eval += gc_eval_time.elapsed().as_micros() as u64;
            timer_end!(eval_time);

            //--------------------------------- stage the result for return as the products ---------------------------------
            results.extend(tile_results);

            //--------------------------------- wait until offline_data of next iteration is ready ---------------------------------
            if tiled && chunk_index != relu_chunks - 1 {
                match rx.recv() {
                    Ok(next_evaluator_offline_data) => {
                        evaluator_offline_data = next_evaluator_offline_data
                    },
                    Err(e) => panic!("there was an issue when pre-fetching client state: {}", e), // Exit the loop if there are no more tiles to process.
                }
            }
        }

        //--------------------------------- save profiling data ---------------------------------
        timing.total_duration += total_time.elapsed().as_micros() as u64;
        timer_end!(start_time);
        let file_name = csv_file_name(
            network_name,
            "client",
            "online",
            "non_linear",
            conv_id as u64,
            batch_id as u64,
            batch_size as u64,
            cores as u64,
            memory as u64,
        );
        write_to_csv(&timing, &file_name);
        let comm_file_name = csv_file_name_comm(
            network_name,
            "client",
            "online",
            "non_linear",
            conv_id as u64,
            batch_id as u64,
            batch_size as u64,
            cores as u64,
            memory as u64,
        );
        write_to_csv(&comm, &comm_file_name);

        Ok(results)
    }
}

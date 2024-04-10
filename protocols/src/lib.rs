use algebra::fixed_point::FixedPoint;
use io_utils::imux::IMuxSync;
use protocols_sys::{ClientFHE, KeyShare, ServerFHE};
use serde::{Deserialize, Serialize};
use std::{
    io::{Read, Write},
    marker::PhantomData,
};

#[macro_use]
extern crate bench_utils;

extern crate ndarray;

pub mod beavers_mul;
pub mod gc;
pub mod linear_layer;
pub mod neural_network;
pub mod quad_approx;

pub mod bytes;

#[cfg(test)]
mod tests;

pub type AdditiveShare<P> = crypto_primitives::AdditiveShare<FixedPoint<P>>;

pub struct KeygenType;
pub type ServerKeyRcv = InMessage<Vec<std::os::raw::c_char>, KeygenType>;
pub type ClientKeySend<'a> = OutMessage<'a, Vec<std::os::raw::c_char>, KeygenType>;

//--------------------------------- added for benchmarking ---------------------------------

//--------------------------------- latency profiling structs ---------------------------------
pub mod csv_timing;
#[derive(Serialize)]
pub struct ServerOfflineLinear {
    random_gen: u64,
    weight_encoding: u64,
    HE_processing: u64,
    input_CT_communication: u64,
    output_CT_communication: u64,
    total_duration: u64,
}

#[derive(Serialize)]
pub struct ClientOfflineLinear {
    random_gen: u64,
    encryption: u64,
    decryption: u64,
    input_CT_communication: u64,
    output_CT_communication: u64,
    total_duration: u64,
}

#[derive(Serialize)]
pub struct ServerOfflineNonLinear {
    garbling: u64,
    encoding: u64,
    OT_communication: u64,
    GC_communication: u64,
    IO_write: u64,
    total_duration: u64,
}

#[derive(Serialize)]
pub struct ClientOfflineNonLinear {
    OT_communication: u64,
    GC_communication: u64,
    IO_write: u64,
    total_duration: u64,
}

#[derive(Serialize)]
pub struct ServerOnlineLinear {
    plain_processing: u64,
    communication: u64,
    total_duration: u64,
}

#[derive(Serialize)]
pub struct ClientOnlineLinear {
    communication: u64,
    total_duration: u64,
}

#[derive(Serialize)]
pub struct ServerOnlineNonLinear {
    encoding: u64,
    communication: u64,
    IO_read: u64,
    total_duration: u64,
}

#[derive(Serialize)]
pub struct ClientOnlineNonLinear {
    GC_eval: u64,
    communication: u64,
    IO_read: u64,
    total_duration: u64,
}

//--------------------------------- communication profiling structs ---------------------------------
#[derive(Serialize)]
pub struct CommServerOfflineLinear {
    ct_write: u64,
    ct_read: u64,
}

#[derive(Serialize)]
pub struct CommClientOfflineLinear {
    ct_write: u64,
    ct_read: u64,
}

#[derive(Serialize)]
pub struct CommServerOfflineNonLinear {
    gc_write: u64,
    ot_write: u64,
    ot_read: u64,
}

#[derive(Serialize)]
pub struct CommClientOfflineNonLinear {
    gc_read: u64,
    ot_write: u64,
    ot_read: u64,
}

#[derive(Serialize)]
pub struct CommServerOnlineLinear {
    read: u64,
}

#[derive(Serialize)]
pub struct CommClientOnlineLinear {
    write: u64,
}

#[derive(Serialize)]
pub struct CommServerOnlineNonLinear {
    encoded_labels_write: u64,
}

#[derive(Serialize)]
pub struct CommClientOnlineNonLinear {
    encoded_labels_read: u64,
}

#[derive(Serialize)]
pub struct CommunicationData {
    pub reads: u64,
    pub writes: u64,
}
//-------------------------------------------------------------------------------------

pub fn client_keygen<W: Write + Send>(
    writer: &mut IMuxSync<W>,
) -> Result<ClientFHE, bincode::Error> {
    let mut key_share = KeyShare::new();
    let gen_time = timer_start!(|| "Generating keys");
    let (cfhe, keys_vec) = key_share.generate();
    timer_end!(gen_time);

    let send_time = timer_start!(|| "Sending keys");
    let sent_message = ClientKeySend::new(&keys_vec);
    crate::bytes::serialize(writer, &sent_message)?;
    timer_end!(send_time);
    Ok(cfhe)
}

pub fn server_keygen<R: Read + Send>(
    reader: &mut IMuxSync<R>,
) -> Result<ServerFHE, bincode::Error> {
    let recv_time = timer_start!(|| "Receiving keys");
    let keys: ServerKeyRcv = crate::bytes::deserialize(reader)?;
    timer_end!(recv_time);
    let mut key_share = KeyShare::new();
    Ok(key_share.receive(keys.msg()))
}

#[derive(Serialize)]
pub struct OutMessage<'a, T: 'a + ?Sized, Type> {
    msg: &'a T,
    protocol_type: PhantomData<Type>,
}

impl<'a, T: 'a + ?Sized, Type> OutMessage<'a, T, Type> {
    pub fn new(msg: &'a T) -> Self {
        Self {
            msg,
            protocol_type: PhantomData,
        }
    }

    pub fn msg(&self) -> &T {
        self.msg
    }
}

#[derive(Deserialize)]
pub struct InMessage<T, Type> {
    msg: T,
    protocol_type: PhantomData<Type>,
}

impl<T, Type> InMessage<T, Type> {
    pub fn msg(self) -> T {
        self.msg
    }
}

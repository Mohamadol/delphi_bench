use crate::{AdditiveShare, InMessage, OutMessage};
use algebra::{
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::Fp64Parameters,
    FpParameters, PrimeField, UniformRandom,
};
use crypto_primitives::additive_share::Share;
use io_utils::imux::IMuxSync;
use neural_network::{
    layers::*,
    tensors::{Input, Output},
    Evaluate,
};
use protocols_sys::{SealClientCG, SealServerCG, *};
use rand::{CryptoRng, RngCore};
use std::{
    io::{Read, Write},
    marker::PhantomData,
    os::raw::c_char,
};

use crate::csv_timing::*;
use crate::ClientOfflineLinear;
use crate::ClientOnlineLinear;
use crate::ServerOfflineLinear;
use crate::ServerOnlineLinear;
use std::time::Instant;

pub struct LinearProtocol<P: FixedPointParameters> {
    _share: PhantomData<P>,
}

pub struct LinearProtocolType;

pub type OfflineServerMsgSend<'a> = OutMessage<'a, Vec<c_char>, LinearProtocolType>;
pub type OfflineServerMsgRcv = InMessage<Vec<c_char>, LinearProtocolType>;
pub type OfflineServerKeyRcv = InMessage<Vec<c_char>, LinearProtocolType>;

pub type OfflineClientMsgSend<'a> = OutMessage<'a, Vec<c_char>, LinearProtocolType>;
pub type OfflineClientMsgRcv = InMessage<Vec<c_char>, LinearProtocolType>;
pub type OfflineClientKeySend<'a> = OutMessage<'a, Vec<c_char>, LinearProtocolType>;

pub type MsgSend<'a, P> = OutMessage<'a, Input<AdditiveShare<P>>, LinearProtocolType>;
pub type MsgRcv<P> = InMessage<Input<AdditiveShare<P>>, LinearProtocolType>;

impl<P: FixedPointParameters> LinearProtocol<P>
where
    P: FixedPointParameters,
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    pub fn offline_server_protocol<R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        _input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        server_cg: &mut SealServerCG,
        rng: &mut RNG,
        layer_id: u16,
        batch_id: u16,
        network_name: &str,
        weight_encoding_time: u64,
    ) -> Result<Output<P::Field>, bincode::Error> {
        // TODO: Add batch size

        let mut timing = ServerOfflineLinear {
            random_gen: 0,
            weight_encoding: weight_encoding_time,
            input_CT_communication: 0,
            HE_processing: 0,
            output_CT_communication: 0,
            total_duration: weight_encoding_time,
        };

        let start_time = timer_start!(|| "Server linear offline protocol");

        let total_time = Instant::now();

        //--------------------------------- random gen ---------------------------------
        let preprocess_time = timer_start!(|| "Preprocessing");

        let server_random_gen = Instant::now();
        // Sample server's randomness `s` for randomizing the i+1-th layer's share.
        let mut server_randomness: Output<P::Field> = Output::zeros(output_dims);
        // TODO
        for r in &mut server_randomness {
            *r = P::Field::uniform(rng);
        }
        // Convert the secret share from P::Field -> u64
        let mut server_randomness_c = Output::zeros(output_dims);
        server_randomness_c
            .iter_mut()
            .zip(&server_randomness)
            .for_each(|(e1, e2)| *e1 = e2.into_repr().0);

        server_cg.preprocess(&server_randomness_c);
        timing.random_gen += server_random_gen.elapsed().as_micros() as u64;


        timer_end!(preprocess_time);

        // Receive client Enc(r_i)
        //--------------------------------- communication receiving input CTs ---------------------------------
        let rcv_time = timer_start!(|| "Receiving Input");
        let input_ct_communication_time = Instant::now();
        let client_share: OfflineServerMsgRcv = crate::bytes::deserialize(reader)?;
        let client_share_i = client_share.msg();
        timing.input_CT_communication += input_ct_communication_time.elapsed().as_micros() as u64;

        timer_end!(rcv_time);

        // Compute client's share for layer `i + 1`.
        // That is, compute -Lr + s
        //--------------------------------- HE processing ---------------------------------
        let processing = timer_start!(|| "Processing Layer");

        let HE_processing_time = Instant::now();
        let enc_result_vec = server_cg.process(client_share_i);
        timing.HE_processing += HE_processing_time.elapsed().as_micros() as u64;

        timer_end!(processing);

        //--------------------------------- communication output CT sending ---------------------------------
        let send_time = timer_start!(|| "Sending result");

        let output_ct_communication_time = Instant::now();
        let sent_message = OfflineServerMsgSend::new(&enc_result_vec);
        crate::bytes::serialize(writer, &sent_message)?;
        timing.output_CT_communication += output_ct_communication_time.elapsed().as_micros() as u64;

        timer_end!(send_time);

        timing.total_duration += total_time.elapsed().as_micros() as u64;
        timer_end!(start_time);

        let file_name = csv_file_name(
            network_name,
            "server",
            "offline",
            "linear",
            layer_id.into(),
            batch_id.into(),
        );
        write_to_csv(&timing, &file_name);

        Ok(server_randomness)
    }

    // Output randomness to share the input in the online phase, and an additive
    // share of the output of after the linear function has been applied.
    // Basically, r and -(Lr + s).
    pub fn offline_client_protocol<
        'a,
        R: Read + Send,
        W: Write + Send,
        RNG: RngCore + CryptoRng,
    >(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        client_cg: &mut SealClientCG,
        rng: &mut RNG,
        layer_id: u16,
        batch_id: u16,
        network_name: &str,
    ) -> Result<(Input<P::Field>, Output<AdditiveShare<P>>), bincode::Error> {
        let mut timing = ClientOfflineLinear {
            random_gen: 0,
            encryption: 0,
            decryption: 0,
            input_CT_communication: 0,
            output_CT_communication: 0,
            total_duration: 0,
        };

        let start_time = timer_start!(|| "Linear offline protocol");
        let total_time = Instant::now();

        // TODO: Add batch size
        let preprocess_time = timer_start!(|| "Client preprocessing");

        //--------------------------------- random generation ---------------------------------
        let client_gen_mask = Instant::now();
        // Generate random share -> r2 = -r1 (because the secret being shared is zero).
        let client_share: Input<FixedPoint<P>> = Input::zeros(input_dims);
        let (r1, r2) = client_share.share(rng);
        timing.random_gen += client_gen_mask.elapsed().as_micros() as u64;

        // Preprocess and encrypt client secret share for sending
        //--------------------------------- encryption ---------------------------------
        let client_encryption = Instant::now();
        let ct_vec = client_cg.preprocess(&r2.to_repr());
        timer_end!(preprocess_time);
        timing.encryption += client_encryption.elapsed().as_micros() as u64;

        // Send layer_i randomness for processing by server.
        //--------------------------------- communication sending input CTs ---------------------------------
        let send_time = timer_start!(|| "Sending input");

        let input_ct_communication_time = Instant::now();
        let sent_message = OfflineClientMsgSend::new(&ct_vec);
        crate::bytes::serialize(writer, &sent_message)?;
        timing.input_CT_communication += input_ct_communication_time.elapsed().as_micros() as u64;

        timer_end!(send_time);

        //--------------------------------- communication receiving output CTs ---------------------------------
        let rcv_time = timer_start!(|| "Receiving Result");

        let output_ct_communication_time = Instant::now();
        let enc_result: OfflineClientMsgRcv = crate::bytes::deserialize(reader)?;
        timing.output_CT_communication += output_ct_communication_time.elapsed().as_micros() as u64;

        timer_end!(rcv_time);

        //--------------------------------- decryption ---------------------------------
        let post_time = timer_start!(|| "Post-processing");
        let client_decryption = Instant::now();
        let mut client_share_next = Input::zeros(output_dims);
        // Decrypt + reshape resulting ciphertext and free C++ allocations
        client_cg.decrypt(enc_result.msg());
        client_cg.postprocess(&mut client_share_next);

        // Should be equal to -(L*r1 - s)
        assert_eq!(client_share_next.dim(), output_dims);
        // Extract the inner field element.
        let layer_randomness = r1
            .iter()
            .map(|r: &AdditiveShare<P>| r.inner.inner)
            .collect::<Vec<_>>();
        let layer_randomness = ndarray::Array1::from_vec(layer_randomness)
            .into_shape(input_dims)
            .unwrap();
        timing.decryption += client_decryption.elapsed().as_micros() as u64;
        timer_end!(post_time);

        timing.total_duration += total_time.elapsed().as_micros() as u64;
        timer_end!(start_time);

        let file_name = csv_file_name(
            network_name,
            "client",
            "offline",
            "linear",
            layer_id.into(),
            batch_id.into(),
        );
        write_to_csv(&timing, &file_name);

        Ok((layer_randomness.into(), client_share_next))
    }

    pub fn online_client_protocol<W: Write + Send>(
        writer: &mut IMuxSync<W>,
        x_s: &Input<AdditiveShare<P>>,
        layer: &LinearLayerInfo<AdditiveShare<P>, FixedPoint<P>>,
        next_layer_input: &mut Output<AdditiveShare<P>>,
        layer_id: u16,
        batch_id: u16,
        network_name: &str,
    ) -> Result<(), bincode::Error> {
        let mut timing = ClientOnlineLinear {
            communication: 0,
            total_duration: 0,
        };

        let start = timer_start!(|| "Linear online protocol");
        let total_time = Instant::now();

        match layer {
            LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                let communication_time = Instant::now();
                let sent_message = MsgSend::new(x_s);
                crate::bytes::serialize(writer, &sent_message)?;
                timing.communication += communication_time.elapsed().as_micros() as u64;
            },
            _ => {
                layer.evaluate_naive(x_s, next_layer_input);
                for elem in next_layer_input.iter_mut() {
                    elem.inner.signed_reduce_in_place();
                }
            },
        }

        timing.total_duration += total_time.elapsed().as_micros() as u64;
        timer_end!(start);

        let file_name = csv_file_name(
            network_name,
            "client",
            "online",
            "linear",
            layer_id.into(),
            batch_id.into(),
        );
        write_to_csv(&timing, &file_name);

        Ok(())
    }

    pub fn online_server_protocol<R: Read + Send>(
        reader: &mut IMuxSync<R>,
        layer: &LinearLayer<AdditiveShare<P>, FixedPoint<P>>,
        output_rerandomizer: &Output<P::Field>,
        input_derandomizer: &Input<P::Field>,
        output: &mut Output<AdditiveShare<P>>,
        layer_id: u16,
        batch_id: u16,
        network_name: &str,
    ) -> Result<(), bincode::Error> {
        let mut timing = ServerOnlineLinear {
            plain_processing: 0,
            communication: 0,
            total_duration: 0,
        };

        let start = timer_start!(|| "Linear online protocol");
        let total_time = Instant::now();

        // Receive client share and compute layer if conv or fc
        let mut input: Input<AdditiveShare<P>> = match &layer {
            LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                let communication_time = Instant::now();
                let recv: MsgRcv<P> = crate::bytes::deserialize(reader).unwrap();
                timing.communication += communication_time.elapsed().as_micros() as u64;

                recv.msg()
            },
            _ => Input::zeros(input_derandomizer.dim()),
        };

        let server_processing = Instant::now();
        input.randomize_local_share(input_derandomizer);
        *output = layer.evaluate(&input);
        output.zip_mut_with(output_rerandomizer, |out, s| {
            *out = FixedPoint::randomize_local_share(out, s)
        });
        timing.plain_processing += server_processing.elapsed().as_micros() as u64;

        timing.total_duration += total_time.elapsed().as_micros() as u64;
        timer_end!(start);

        let file_name = csv_file_name(
            network_name,
            "server",
            "online",
            "linear",
            layer_id.into(),
            batch_id.into(),
        );
        write_to_csv(&timing, &file_name);

        Ok(())
    }
}

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
    ) -> Result<Output<P::Field>, bincode::Error> {
        // TODO: Add batch size
        let mut timing = ServerOfflineLinear {
            server_gen_masks: 0,
            server_processing: 0,
            total_duration: 0,
        };

        let start_time = timer_start!(|| "Server linear offline protocol");
        let preprocess_time = timer_start!(|| "Preprocessing");

        let total_time = Instant::now();
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

        timing.server_gen_masks += server_random_gen.elapsed().as_micros() as u64;

        // Preprocess filter rotations and noise masks
        server_cg.preprocess(&server_randomness_c);

        timer_end!(preprocess_time);

        // Receive client Enc(r_i)
        let rcv_time = timer_start!(|| "Receiving Input");
        let client_share: OfflineServerMsgRcv = crate::bytes::deserialize(reader)?;
        let client_share_i = client_share.msg();
        timer_end!(rcv_time);

        // Compute client's share for layer `i + 1`.
        // That is, compute -Lr + s
        let server_processing = Instant::now();
        let processing = timer_start!(|| "Processing Layer");
        let enc_result_vec = server_cg.process(client_share_i);
        timer_end!(processing);
        timing.server_processing += server_processing.elapsed().as_micros() as u64;

        let send_time = timer_start!(|| "Sending result");
        let sent_message = OfflineServerMsgSend::new(&enc_result_vec);
        crate::bytes::serialize(writer, &sent_message)?;

        timing.total_duration += total_time.elapsed().as_micros() as u64;

        timer_end!(send_time);
        timer_end!(start_time);

        let file_name = csv_file_name(
            "resnet50",
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
    ) -> Result<(Input<P::Field>, Output<AdditiveShare<P>>), bincode::Error> {
        let mut timing = ClientOfflineLinear {
            client_gen_masks: 0,
            client_encryption: 0,
            client_decryption: 0,
            total_duration: 0,
        };
        let total_time = Instant::now();

        // TODO: Add batch size
        let start_time = timer_start!(|| "Linear offline protocol");
        let preprocess_time = timer_start!(|| "Client preprocessing");

        let client_gen_mask = Instant::now();
        // Generate random share -> r2 = -r1 (because the secret being shared is zero).
        let client_share: Input<FixedPoint<P>> = Input::zeros(input_dims);
        let (r1, r2) = client_share.share(rng);
        timing.client_gen_masks += client_gen_mask.elapsed().as_micros() as u64;

        // Preprocess and encrypt client secret share for sending
        let client_encryption = Instant::now();
        let ct_vec = client_cg.preprocess(&r2.to_repr());
        timer_end!(preprocess_time);
        timing.client_encryption += client_encryption.elapsed().as_micros() as u64;

        // Send layer_i randomness for processing by server.
        let send_time = timer_start!(|| "Sending input");
        let sent_message = OfflineClientMsgSend::new(&ct_vec);
        crate::bytes::serialize(writer, &sent_message)?;
        timer_end!(send_time);

        let rcv_time = timer_start!(|| "Receiving Result");
        let enc_result: OfflineClientMsgRcv = crate::bytes::deserialize(reader)?;
        timer_end!(rcv_time);

        let client_decryption = Instant::now();
        let post_time = timer_start!(|| "Post-processing");
        let mut client_share_next = Input::zeros(output_dims);
        // Decrypt + reshape resulting ciphertext and free C++ allocations
        client_cg.decrypt(enc_result.msg());
        client_cg.postprocess(&mut client_share_next);
        timing.client_decryption += client_decryption.elapsed().as_micros() as u64;

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
        timing.total_duration += total_time.elapsed().as_micros() as u64;

        timer_end!(post_time);
        timer_end!(start_time);

        let file_name = csv_file_name(
            "resnet50",
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
    ) -> Result<(), bincode::Error> {
        let start = timer_start!(|| "Linear online protocol");
        match layer {
            LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                let sent_message = MsgSend::new(x_s);
                crate::bytes::serialize(writer, &sent_message)?;
            },
            _ => {
                layer.evaluate_naive(x_s, next_layer_input);
                for elem in next_layer_input.iter_mut() {
                    elem.inner.signed_reduce_in_place();
                }
            },
        }
        timer_end!(start);
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
    ) -> Result<(), bincode::Error> {
        let mut timing = ServerOnlineLinear {
            server_processing_plain: 0,
            total_duration: 0,
        };
        let total_time = Instant::now();

        let start = timer_start!(|| "Linear online protocol");
        // Receive client share and compute layer if conv or fc
        let mut input: Input<AdditiveShare<P>> = match &layer {
            LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                let recv: MsgRcv<P> = crate::bytes::deserialize(reader).unwrap();
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
        timing.server_processing_plain += server_processing.elapsed().as_micros() as u64;
        timing.total_duration += total_time.elapsed().as_micros() as u64;
        timer_end!(start);

        let file_name = csv_file_name(
            "resnet50",
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

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
use io_utils::imux::IMuxSync;
use ocelot::ot::{AlszReceiver as OTReceiver, AlszSender as OTSender, Receiver, Sender};
use rand::{CryptoRng, RngCore};
use rayon::prelude::*;
use scuttlebutt::Channel;
use std::{
    convert::TryFrom,
    io::{Read, Write},
    marker::PhantomData,
};

use crate::csv_timing::*;
use crate::ClientOfflineNonLinear;
use crate::ClientOnlineNonLinear;
use crate::ServerOfflineNonLinear;
use crate::ServerOnlineNonLinear;
use std::time::Instant;

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

pub struct ServerState<P: FixedPointParameters> {
    pub encoders: Vec<Encoder>,
    pub output_randomizers: Vec<P::Field>,
}

pub struct ClientState {
    pub gc_s: Vec<GarbledCircuit>,
    pub server_randomizer_labels: Vec<Wire>,
    pub client_input_labels: Vec<Wire>,
}

impl<P: FixedPointParameters> ReluProtocol<P>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = BigInteger64>,
{
    #[inline]
    pub fn size_of_client_inputs() -> usize {
        make_relu::<P>().num_evaluator_inputs()
    }

    pub fn offline_server_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        number_of_relus: usize,
        rng: &mut RNG,
        batch_id: u16,
    ) -> Result<ServerState<P>, bincode::Error> {
        let mut timing = ServerOfflineNonLinear {
            garbling: 0,
            encoding: 0,
            OT_communication: 0,
            GC_communication: 0,
            total_duration: 0,
        };

        let start_time = timer_start!(|| "ReLU offline protocol");
        let total_time = Instant::now();

        //--------------------------------- Garbling ---------------------------------
        let garbling_time = Instant::now();
        let mut gc_s = Vec::with_capacity(number_of_relus);
        let mut encoders = Vec::with_capacity(number_of_relus);
        let p = (<<P::Field as PrimeField>::Params>::MODULUS.0).into();

        let c = make_relu::<P>();
        let garble_time = timer_start!(|| "Garbling");
        (0..number_of_relus)
            .into_par_iter()
            .map(|_| {
                let mut c = c.clone();
                let (en, gc) = fancy_garbling::garble(&mut c).unwrap();
                (en, gc)
            })
            .unzip_into_vecs(&mut encoders, &mut gc_s);
        timing.garbling += garbling_time.elapsed().as_micros() as u64;
        timer_end!(garble_time);

        //--------------------------------- Encoding ---------------------------------
        let encode_time = timer_start!(|| "Encoding inputs");

        let encoding_time = Instant::now();
        let num_garbler_inputs = c.num_garbler_inputs();
        let num_evaluator_inputs = c.num_evaluator_inputs();

        let zero_inputs = vec![0u16; num_evaluator_inputs];
        let one_inputs = vec![1u16; num_evaluator_inputs];
        let mut labels = Vec::with_capacity(number_of_relus * num_evaluator_inputs);
        let mut randomizer_labels = Vec::with_capacity(number_of_relus);
        let mut output_randomizers = Vec::with_capacity(number_of_relus);
        for enc in encoders.iter() {
            let r = P::Field::uniform(rng);
            output_randomizers.push(r);
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
        let send_gc_time = timer_start!(|| "Sending GCs");

        let gc_communication_time = Instant::now();
        let randomizer_label_per_relu = if number_of_relus == 0 {
            8192
        } else {
            randomizer_labels.len() / number_of_relus
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

        //--------------------------------- OT communication ---------------------------------
        if number_of_relus != 0 {
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

        timing.total_duration += total_time.elapsed().as_micros() as u64;
        timer_end!(start_time);

        let file_name = csv_file_name(
            "resnet18",
            "server",
            "offline",
            "non_linear",
            1 as u64,
            batch_id as u64,
        );
        write_to_csv(&timing, &file_name);

        Ok(ServerState {
            encoders,
            output_randomizers,
        })
    }

    pub fn offline_client_protocol<R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        number_of_relus: usize,
        shares: &[AdditiveShare<P>],
        rng: &mut RNG,
        batch_id: u16,
    ) -> Result<ClientState, bincode::Error> {
        let mut timing = ClientOfflineNonLinear {
            OT_communication: 0,
            GC_communication: 0,
            total_duration: 0,
        };

        use fancy_garbling::util::*;
        let start_time = timer_start!(|| "ReLU offline protocol");

        let total_time = Instant::now();

        let p = u128::from(<<P::Field as PrimeField>::Params>::MODULUS.0);
        let field_size = crypto_primitives::gc::num_bits(p);

        //--------------------------------- GC receiving ---------------------------------
        let rcv_gc_time = timer_start!(|| "Receiving GCs");

        let gc_communication_time = Instant::now();
        let mut gc_s = Vec::with_capacity(number_of_relus);
        let mut r_wires = Vec::with_capacity(number_of_relus);

        let num_chunks = (number_of_relus as f64 / 8192.0).ceil() as usize;
        for i in 0..num_chunks {
            let in_msg: ClientGcMsgRcv = crate::bytes::deserialize(reader)?;
            let (gc_chunks, r_wire_chunks) = in_msg.msg();
            if i < (num_chunks - 1) {
                assert_eq!(gc_chunks.len(), 8192);
            }
            gc_s.extend(gc_chunks);
            r_wires.extend(r_wire_chunks);
        }
        timing.GC_communication += gc_communication_time.elapsed().as_micros() as u64;

        timer_end!(rcv_gc_time);

        //--------------------------------- OT communication ---------------------------------
        let OT_communication_time = Instant::now();
        assert_eq!(gc_s.len(), number_of_relus);
        let bs = shares
            .iter()
            .flat_map(|s| u128_to_bits(u128_from_share(*s), field_size))
            .map(|b| b == 1)
            .collect::<Vec<_>>();

        let labels = if number_of_relus != 0 {
            let r = reader.get_mut_ref().remove(0);
            let w = writer.get_mut_ref().remove(0);

            let ot_time = timer_start!(|| "OTs");
            let mut channel = Channel::new(r, w);
            let mut ot = OTReceiver::init(&mut channel, rng).expect("should work");
            let labels = ot
                .receive(&mut channel, bs.as_slice(), rng)
                .expect("should work");
            let labels = labels
                .into_iter()
                .map(|l| Wire::from_block(l, 2))
                .collect::<Vec<_>>();
            timer_end!(ot_time);
            labels
        } else {
            Vec::new()
        };
        timing.OT_communication += OT_communication_time.elapsed().as_micros() as u64;

        timing.total_duration += total_time.elapsed().as_micros() as u64;
        timer_end!(start_time);

        let file_name = csv_file_name(
            "resnet18",
            "client",
            "offline",
            "non_linear",
            1 as u64,
            batch_id.into(),
        );
        write_to_csv(&timing, &file_name);

        Ok(ClientState {
            gc_s,
            server_randomizer_labels: r_wires,
            client_input_labels: labels,
        })
    }

    pub fn online_server_protocol<'a, W: Write + Send>(
        writer: &mut IMuxSync<W>,
        shares: &[AdditiveShare<P>],
        encoders: &[Encoder],
        batch_id: u16,
        conv_id: u16,
    ) -> Result<(), bincode::Error> {
        let mut timing = ServerOnlineNonLinear {
            encoding: 0,
            communication: 0,
            total_duration: 0,
        };

        let p = u128::from(u64::from(P::Field::characteristic()));

        let start_time = timer_start!(|| "ReLU online protocol");
        let total_time = Instant::now();

        //--------------------------------- encoding ---------------------------------
        let encoding_time = timer_start!(|| "Encoding inputs");

        let server_encoding = Instant::now();
        let field_size = (p.next_power_of_two() * 2).trailing_zeros() as usize;
        let wires = shares
            .iter()
            .map(|share| {
                let share = u128_from_share(*share);
                fancy_garbling::util::u128_to_bits(share, field_size)
            })
            .zip(encoders)
            .map(|(share_bits, encoder)| encoder.encode_garbler_inputs(&share_bits))
            .collect::<Vec<Vec<_>>>();
        timing.encoding += server_encoding.elapsed().as_micros() as u64;

        timer_end!(encoding_time);

        //--------------------------------- communication ---------------------------------
        let send_time = timer_start!(|| "Sending inputs");

        let server_communication_time = Instant::now();
        let sent_message = ServerLabelMsgSend::new(wires.as_slice());
        let res = crate::bytes::serialize(writer, &sent_message);
        timing.communication += server_communication_time.elapsed().as_micros() as u64;

        timer_end!(send_time);

        timing.total_duration += total_time.elapsed().as_micros() as u64;
        timer_end!(start_time);

        let file_name = csv_file_name(
            "resnet18",
            "server",
            "online",
            "non_linear",
            conv_id as u64,
            batch_id as u64,
        );
        write_to_csv(&timing, &file_name);

        res
    }

    /// Outputs shares for the next round's input.
    pub fn online_client_protocol<R: Read + Send>(
        reader: &mut IMuxSync<R>,
        num_relus: usize,
        server_input_wires: &[Wire],
        client_input_wires: &[Wire],
        evaluators: &[GarbledCircuit],
        next_layer_randomizers: &[P::Field],
        batch_id: u16,
        conv_id: u16,
    ) -> Result<Vec<AdditiveShare<P>>, bincode::Error> {
        let mut timing = ClientOnlineNonLinear {
            GC_eval: 0,
            communication: 0,
            total_duration: 0,
        };

        let start_time = timer_start!(|| "ReLU online protocol");
        let total_time = Instant::now();

        //--------------------------------- communication ---------------------------------
        let rcv_time = timer_start!(|| "Receiving inputs");

        let communication_time = Instant::now();
        let in_msg: ClientLabelMsgRcv = crate::bytes::deserialize(reader)?;
        let mut garbler_wires = in_msg.msg();
        timing.communication += communication_time.elapsed().as_micros() as u64;

        timer_end!(rcv_time);

        //--------------------------------- GC evaluation ---------------------------------
        let eval_time = timer_start!(|| "Evaluating GCs");

        let gc_eval_time = Instant::now();
        let c = make_relu::<P>();
        let num_evaluator_inputs = c.num_evaluator_inputs();
        let num_garbler_inputs = c.num_garbler_inputs();
        garbler_wires
            .iter_mut()
            .zip(server_input_wires.chunks(num_garbler_inputs / 2))
            .for_each(|(w1, w2)| w1.extend_from_slice(w2));

        assert_eq!(num_relus, garbler_wires.len());
        assert_eq!(num_evaluator_inputs * num_relus, client_input_wires.len());
        // We access the input wires in reverse.
        let c = make_relu::<P>();
        let mut results = client_input_wires
            .par_chunks(num_evaluator_inputs)
            .zip(garbler_wires)
            .zip(evaluators)
            .map(|((eval_inps, garbler_inps), gc)| {
                let mut c = c.clone();
                let result = gc
                    .eval(&mut c, &garbler_inps, eval_inps)
                    .expect("evaluation failed");
                let result = fancy_garbling::util::u128_from_bits(result.as_slice());
                FixedPoint::new(P::Field::from_repr(u64::try_from(result).unwrap().into())).into()
            })
            .collect::<Vec<AdditiveShare<P>>>();
        results
            .iter_mut()
            .zip(next_layer_randomizers)
            .for_each(|(s, r)| *s = FixedPoint::<P>::randomize_local_share(s, r));

        timing.GC_eval += gc_eval_time.elapsed().as_micros() as u64;
        timer_end!(eval_time);

        timing.total_duration += total_time.elapsed().as_micros() as u64;
        timer_end!(start_time);

        let file_name = csv_file_name(
            "resnet18",
            "client",
            "online",
            "non_linear",
            conv_id as u64,
            batch_id as u64,
        );
        write_to_csv(&timing, &file_name);

        Ok(results)
    }
}

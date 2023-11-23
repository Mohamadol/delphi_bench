use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

use super::*;

pub fn construct_resnet18_model<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    rng: &mut R,
) -> NeuralNetwork<TenBitAS, TenBitExpFP> {
    use std::collections::HashSet;

    // ---------------- these are the ReLU layers id ----------------
    let mut relu_layers = Vec::new();
    // for l in 0..46 {
    for l in 0..16 {
        relu_layers.push(2 * l + 1);
    }

    // ---------------- these are the ReLU layers id ----------------
    let mut network = match &vs {
        Some(vs) => NeuralNetwork {
            layers: vec![],
            eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };

    // ---------------- Conv Layers ----------------
    // for conv_id in 2..50 {
    for conv_id in 1..18 {
        match conv_id {
            1 => {
                let k: usize = 64;
                let c: usize = 3;
                let p: usize = 30;
                let r: usize = 3;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Same, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            2 | 3 | 4 | 5 => {
                let k: usize = 64;
                let c: usize = 64;
                let p: usize = 30;
                let r: usize = 3;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Same, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            6 => {
                let k: usize = 128;
                let c: usize = 64;
                let p: usize = 30;
                let r: usize = 3;
                let stride: usize = 2;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Same, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            7 | 8 | 9 => {
                let k: usize = 128;
                let c: usize = 128;
                let p: usize = 16;
                let r: usize = 3;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Same, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            10 => {
                let k: usize = 256;
                let c: usize = 128;
                let p: usize = 16;
                let r: usize = 3;
                let stride: usize = 2;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Same, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            11 | 12 | 13 => {
                let k: usize = 256;
                let c: usize = 256;
                let p: usize = 9;
                let r: usize = 3;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Same, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            14 => {
                let k: usize = 512;
                let c: usize = 256;
                let p: usize = 9;
                let r: usize = 3;
                let stride: usize = 2;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Same, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            15 | 16 | 17 => {
                let k: usize = 512;
                let c: usize = 512;
                let p: usize = 6;
                let r: usize = 3;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Same, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            _ => {
                panic!("unkown layer {}", conv_id)
            },
        }
    }

    assert!(network.validate());
    network
}

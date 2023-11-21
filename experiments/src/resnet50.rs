use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

use super::*;

pub fn construct_resnet50_model<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    rng: &mut R,
) -> NeuralNetwork<TenBitAS, TenBitExpFP> {
    use std::collections::HashSet;

    // ---------------- these are the ReLU layers id ----------------
    let mut relu_layers = Vec::new();
    for l in 0..47 {
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
    for conv_id in 1..48 {
        match conv_id {
            1 => {
                let k: usize = 64;
                let c: usize = 3;
                let p: usize = 230;
                let r: usize = 7;
                let stride: usize = 2;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Valid, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);

                let avg_pool_input_dims = network.layers.last().unwrap().output_dimensions();
                network.layers.push(Layer::LL(sample_avg_pool_layer(
                    avg_pool_input_dims,
                    (2, 2),
                    2,
                )));
            },

            2 => {
                let k: usize = 64;
                let c: usize = 64;
                let p: usize = 56;
                let r: usize = 1;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Valid, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            3 | 6 | 9 => {
                let k: usize = 64;
                let c: usize = 64;
                let p: usize = 56;
                let r: usize = 3;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Same, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            4 | 7 | 10 => {
                let k: usize = 256;
                let c: usize = 64;
                let p: usize = 56;
                let r: usize = 1;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Valid, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            5 | 8 => {
                let k: usize = 64;
                let c: usize = 256;
                let p: usize = 56;
                let r: usize = 1;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Valid, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            11 => {
                let k: usize = 128;
                let c: usize = 256;
                let p: usize = 56;
                let r: usize = 1;
                let stride: usize = 2;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Valid, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            12 | 15 | 18 | 21 => {
                let k: usize = 128;
                let c: usize = 128;
                let p: usize = 28;
                let r: usize = 3;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Same, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            13 | 16 | 19 | 22 => {
                let k: usize = 512;
                let c: usize = 128;
                let p: usize = 28;
                let r: usize = 1;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Valid, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            14 | 17 | 20 => {
                let k: usize = 128;
                let c: usize = 512;
                let p: usize = 28;
                let r: usize = 1;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Valid, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            23 => {
                let k: usize = 256;
                let c: usize = 512;
                let p: usize = 28;
                let r: usize = 1;
                let stride: usize = 2;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Valid, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            24 | 27 | 30 | 33 | 36 | 39 => {
                let k: usize = 256;
                let c: usize = 256;
                let p: usize = 14;
                let r: usize = 3;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Valid, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            25 | 28 | 31 | 34 | 37 | 40 => {
                let k: usize = 1024;
                let c: usize = 256;
                let p: usize = 14;
                let r: usize = 1;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Valid, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            26 | 29 | 32 | 35 | 38 => {
                let k: usize = 256;
                let c: usize = 1024;
                let p: usize = 14;
                let r: usize = 1;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Valid, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            41 => {
                let k: usize = 512;
                let c: usize = 1024;
                let p: usize = 14;
                let r: usize = 1;
                let stride: usize = 2;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Valid, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            42 | 45 | 48 => {
                let k: usize = 512;
                let c: usize = 512;
                let p: usize = 7;
                let r: usize = 3;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Valid, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            43 | 46 | 49 => {
                let k: usize = 2048;
                let c: usize = 512;
                let p: usize = 7;
                let r: usize = 1;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Valid, rng);
                network.layers.push(Layer::LL(conv_1));
                add_activation_layer(&mut network, &relu_layers);
            },

            44 | 47 => {
                let k: usize = 512;
                let c: usize = 2048;
                let p: usize = 7;
                let r: usize = 1;
                let stride: usize = 1;
                let input_dims: (usize, usize, usize, usize) = (1, c, p, p);
                let kernel_dims: (usize, usize, usize, usize) = (k, c, r, r);
                let (conv_1, _) =
                    sample_conv_layer(vs, input_dims, kernel_dims, stride, Padding::Valid, rng);
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

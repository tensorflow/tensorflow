# Test models for testing quantization

This directory contains test models for testing quantization.

## Models

* `single_conv_weights_min_0_max_plus_10.tflite` \
   A floating point model with single convolution where all weights are
   integers between [0, 10] weights are randomly distributed. It is not
   guaranteed that min max for weights are going to appear in each channel.
   All activations have min maxes and activations are in range [0,10].
* `single_conv_weights_min_minus_127_max_plus_127.tflite` \
   A floating point model with a single convolution where weights of the model
   are all integers that lie in range[-127, 127]. The weights have been put in
   such a way that each channel has at least one weight as -127 and one weight
   as 127. The activations are all in range: [-128, 127].
   This means all bias computations should result in 1.0 scale.
* `single_softmax_min_minus_5_max_5.tflite` \
   A floating point model with a single softmax. The input tensor has min
   and max in range [-5, 5], not necessarily -5 or +5.
* `single_avg_pool_input_min_minus_5_max_5.tflite` \
   A floating point model with a single average pool. The input tensor has min
   and max in range [-5, 5], not necessarily -5 or +5.
* `weight_shared_between_convs.tflite` \
   A floating point model with two convs that have a use the same weight tensor.

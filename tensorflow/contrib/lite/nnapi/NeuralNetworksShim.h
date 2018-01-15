/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef NN_API_SHIM_H0
#define NN_API_SHIM_H0

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// helpers

#define NNAPI_LOG(format, ...) printf(format "\n", __VA_ARGS__);
#define LOAD_FUNCTION(name) \
  static name##_fn fn = reinterpret_cast<name##_fn>(loadFunction(#name));
#define EXECUTE_FUNCTION(...) \
  if (fn != nullptr) {        \
    fn(__VA_ARGS__);          \
  }
#define EXECUTE_FUNCTION_RETURN(...) return fn != nullptr ? fn(__VA_ARGS__) : 0;

inline void* loadLibrary(const char* name) {
  // TODO: change RTLD_LOCAL? Assumes there can be multiple instances of nn
  // api RT
  void* handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    NNAPI_LOG("nnapi error: unable to open library %s", name);
  }
  return handle;
}

inline void* getLibraryHandle() {
  static void* handle = loadLibrary("libneuralnetworks.so");
  return handle;
}

inline void* loadFunction(const char* name) {
  void* fn = nullptr;
  if (getLibraryHandle() != nullptr) {
    fn = dlsym(getLibraryHandle(), name);
  }
  if (fn == nullptr) {
    NNAPI_LOG("nnapi error: unable to open function %s", name);
  }
  return fn;
}

inline bool NNAPIExists() {
  static bool nnapi_is_available = getLibraryHandle();
  return nnapi_is_available;
}

// nn api types

/**
 * Operand types.
 *
 * The type of operands that can be added to a model.
 *
 * Although we define many types, most operators accept just a few
 * types.  Most used are ANEURALNETWORKS_TENSOR_FLOAT32,
 * ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, and ANEURALNETWORKS_INT32.
 */
enum {
  /** The following entries are used to declare scalars. */

  /** A 32 bit floating point scalar value. */
  ANEURALNETWORKS_FLOAT32 = 0,
  /** A signed 32 bit integer scalar value. */
  ANEURALNETWORKS_INT32 = 1,
  /** An unsigned 32 bit integer scalar value. */
  ANEURALNETWORKS_UINT32 = 2,

  /** The following entries are used to declare tensors. */

  /** A tensor of 32 bit floating point values. */
  ANEURALNETWORKS_TENSOR_FLOAT32 = 3,
  /** A tensor of 32 bit integer values. */
  ANEURALNETWORKS_TENSOR_INT32 = 4,
  /** A tensor of 8 bit integers that represent real numbers.
   *
   * Attached to this tensor are two numbers that can be used to convert
   * the 8 bit integer to the real value and vice versa.  These two numbers are:
   * - scale: a 32 bit floating point value
   * - zero_value: an 32 bit integer
   *
   * The formula is:
   * real_value = (integer_value - zero_value) * scale.
   */
  ANEURALNETWORKS_TENSOR_QUANT8_ASYMM = 5,
};

/**
 * Operation types.
 *
 * The type of operations that can be added to a model.
 */
enum {
  /** Adds two tensors, element-wise.
   *
   * Takes two input tensors of identical type and compatible dimensions. The
   * output is the sum of both input tensors, optionally modified by an
   * activation function.
   *
   * Two dimensions are compatible when:
   *     1. they are equal, or
   *     2. one of them is 1
   *
   * The size of the output is the maximum size along each dimension of the
   * input operands. It starts with the trailing dimensions, and works its way
   * forward.
   *
   * Example:
   *
   *     input1.dimension = {4, 1, 2}
   *     input2.dimension = {5, 4, 3, 1}
   *     output.dimension = {5, 4, 3, 2}
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A tensor.
   * * 1: A tensor of the same type, and compatible dimensions as input0.
   * * 2: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The sum, a tensor of the same type as input0.
   */
  ANEURALNETWORKS_ADD = 0,
  /** Performs a 2-D average pooling operation.
   *
   * The output dimensions are functions of the filter dimensions, stride, and
   * padding.
   *
   * The values in the output tensor are computed as:
   *
   *     output[batch, row, col, channel] =
   *         sum_{i, j}(input[batch, row + i, col + j, channel]) / sum(1)
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Inputs:
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the
   * input.
   * * 1: An INT32 value, specifying the padding on the left, in the ‘width’
   * dimension.
   * * 2: An INT32 value, specifying the padding on the right,in the ‘width’
   * dimension.
   * * 3: An INT32 value, specifying the padding on the top, in the ‘height’
   * dimension.
   * * 4: An INT32 value, specifying the padding on the bottom, in the ‘height’
   * dimension.
   * * 5: An INT32 value, specifying the output stride in the ‘width’ dimension.
   * * 6: An INT32 value, specifying the output stride in the ‘height’
   * dimension.
   * * 7: An INT32 value, specifying the filter width.
   * * 8: An INT32 value, specifying the filter height.
   * * 9: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, out_height, out_width,
   * depth].
   */
  ANEURALNETWORKS_AVERAGE_POOL_2D = 1,
  /** Concatenates the input tensors along the given dimension.
   *
   * The input tensors must have identical type and the same dimensions except
   * the dimension along the concatenation axis.
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * 0 ~ n: The list on n input tensors, of shape [D0, D1, ..., Daxis(i), ...,
   * Dm] n+1: An INT32 value, specifying the concatenation axis. n+2: An INT32
   * value, and has to be one of the {@link FuseCode} values. Specifies the
   * activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The output, a tensor of the same type as the input tensors.
   *      The output shape is [D0, D1, ..., sum(Daxis(i)), ..., Dm].
   */
  ANEURALNETWORKS_CONCATENATION = 2,
  /** Performs an 2-D convolution operation.
   *
   * The CONV_2D op sweeps a 2-D filter that can mix channels together over a
   * batch of images, applying the filter to each window of each image of the
   * appropriate size.
   *
   * The output dimensions are functions of the filter dimensions, stride, and
   * padding.
   *
   * The values in the output tensor are computed as:
   *
   *     output[batch, row, col, channel] =
   *         sum_{i, j} (
   *             input[batch, row + i, col + j, k] *
   *             filter[channel, row + i, col + j, k] +
   *             bias[channel]
   *         )
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Inputs:
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying
   * the input.
   * * 1: A 4-D tensor, of shape [depth_out, filter_height, filter_width,
   * depth_in], specifying the filter.
   * * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
   *      For input tensor of {@link ANEURALNETWORKS_TENSOR_FLOAT32} type, the
   * bias should also be of {@link ANEURALNETWORKS_TENSOR_FLOAT32}. For input
   * tensor of {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} type, the bias should
   * be of {@link ANEURALNETWORKS_TENSOR_INT32}.
   * * 3: An INT32 value, specifying the padding on the left, in the ‘width’
   * dimension.
   * * 4: An INT32 value, specifying the padding on the right,in the ‘width’
   * dimension.
   * * 5: An INT32 value, specifying the padding on the top, in the ‘height’
   * dimension.
   * * 6: An INT32 value, specifying the padding on the bottom, in the ‘height’
   * dimension.
   * * 7: An INT32 value, specifying the output stride in the ‘width’ dimension.
   * * 8: An INT32 value, specifying the output stride in the ‘height’
   * dimension.
   * * 9: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, out_height, out_width,
   * depth_out].
   */
  ANEURALNETWORKS_CONV_2D = 3,
  /** Performs a depthwise 2-D convolution operation.
   *
   * Given an input tensor of shape [batches, height, width, depth_in] and a
   * filter tensor of shape [depth_out, filter_height, filter_width, depth_in]
   * containing in_channels convolutional filters of depth 1, DEPTHWISE_CONV
   * applies a different filter to each input channel (expanding from 1 channel
   * to channel_multiplier channels for each), then concatenates the results
   * together.
   *
   * The output has depth_out = depth_in * depth_multiplier channels.
   * The output dimensions are functions of the filter dimensions, stride, and
   * padding.
   *
   * The values in the output tensor are computed as:
   *
   *     output[b, i, j, k * channel_multiplier + q] =
   *         sum_{di, dj} (
   *             input[b, strides[1] * i + di, strides[2] * j + dj, k] *
   *             filter[di, dj, k, q]
   *         )
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Inputs:
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying
   * the input.
   * * 1: A 4-D tensor, of shape [depth_out, filter_height, filter_width,
   * depth_in], specifying the filter.
   * * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
   *      For input tensor of {@link ANEURALNETWORKS_TENSOR_FLOAT32} type, the
   * bias should also be of {@link ANEURALNETWORKS_TENSOR_FLOAT32}. For input
   * tensor of {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} type, the bias should
   * be of {@link ANEURALNETWORKS_TENSOR_INT32}.
   * * 3: An INT32 value, specifying the padding on the left, in the ‘width’
   * dimension.
   * * 4: An INT32 value, specifying the padding on the right,in the ‘width’
   * dimension.
   * * 5: An INT32 value, specifying the padding on the top, in the ‘height’
   * dimension.
   * * 6: An INT32 value, specifying the padding on the bottom, in the ‘height’
   * dimension.
   * * 7: An INT32 value, specifying the output stride in the ‘width’ dimension.
   * * 8: An INT32 value, specifying the output stride in the ‘height’
   * dimension.
   * * 9: An INT32 value, specifying the depthwise multiplier.
   * * 10: An INT32 value, and has to be one of the {@link FuseCode} values.
   *       Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, out_height, out_width,
   * depth_out].
   */
  ANEURALNETWORKS_DEPTHWISE_CONV_2D = 4,
  /** Rearranges data from depth into blocks of spatial data.
   *
   * More specifically, this op outputs a copy of the input tensor where values
   * from the depth dimension are moved in spatial blocks to the height and
   * width dimensions. The value block_size indicates the input block size and
   * how the data is moved.
   *
   * Chunks of data of size block_size * block_size from depth are rearranged
   * into non-overlapping blocks of size block_size x block_size.
   *
   * The width of the output tensor is input_depth * block_size, whereas the
   * height is input_height * block_size. The depth of the input tensor must be
   * divisible by block_size * block_size
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Inputs:
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying
   * the input.
   * * 1: An INT32 value, specifying the block_size. block_size must be >=1 and
   *      block_size * block_size must be a divisor of the input depth.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batch, height*block_size,
   * width*block_size, depth/(block_size*block_size)].
   */
  ANEURALNETWORKS_DEPTH_TO_SPACE = 5,
  /** Dequantizes the input tensor.
   *
   * The formula is:
   *
   *     output = (input - zero_value) * scale.
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0, but with type
   *      {@link ANEURALNETWORKS_TENSOR_FLOAT32}.
   */
  ANEURALNETWORKS_DEQUANTIZE = 6,

  /**
   * Looks up items from a given tensor.
   *
   * Each item in the output is a raw copy of the corresponding item in
   * the input “values”. If the the given “lookup” indices are out of bounds,
   * the op will fail and an error will be reported.
   *
   * Inputs:
   * * 0: Values. An n-D tensor of any type X (where n >= 2). E.g., if n is 2,
   *      then the shape would be [lookup_dimension, values_dimension], where
   *      “lookup_dimension” corresponds to the indexing dimension in the lookup
   *      table, and “values_dimension” to the contents.
   * * 1: Lookups. An 1-D tensor of type T, of shape [lookup_size], where
   *      “lookup_size” is the number of elements to look for, and each entry
   *      corresponds to the first dimension of the “values” tensor.
   *
   * Output:
   * * 0: A n-D tensor of type X and the same rank and shape as the “values”
   *      tensor, except for the first dimension which has size “lookup_size”.
   */
  ANEURALNETWORKS_EMBEDDING_LOOKUP = 7,

  /** Computes element-wise floor() on the input tensor.
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A tensor.
   *
   * Outputs:
   * * 0: The output, a tensor of the same type and dimensions as input0.
   */
  ANEURALNETWORKS_FLOOR = 8,
  /** Denotes a fully (densely) connected layer, which connects all elements in
   * the input tensor with each element in the output tensor.
   *
   * This layer implements the operation:
   *
   *     outputs = activation(inputs * weights’ + bias)
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4.
   *
   * Inputs:
   * * 0: A tensor, specifying the input. If rank is greater than 2, then it
   * gets flattened to a 2-D Tensor. The 2-D Tensor is handled as if dimensions
   * corresponded to shape [batch_size, input_size], where “batch_size”
   * corresponds to the batching dimension, and “input_size” is the size of the
   * input.
   * * 1: A 2-D tensor, specifying the weights, of shape [num_units,
   * input_size], where "num_units" corresponds to the number of output nodes.
   * * 2: A 1-D tensor, of shape [num_units], specifying the bias.
   *      For input tensor of {@link ANEURALNETWORKS_TENSOR_FLOAT32} type, the
   * bias should also be of {@link ANEURALNETWORKS_TENSOR_FLOAT32}. For input
   * tensor of {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM} type, the bias should
   * be of {@link ANEURALNETWORKS_TENSOR_INT32}.
   * * 3: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The output tensor, of shape [batch_size, num_units].
   */
  ANEURALNETWORKS_FULLY_CONNECTED = 9,

  /**
   * Looks up values of a hash table with given keys.
   *
   * Inputs:
   * * 0: Lookups. A 1-D int32 tensor with shape [ k ].
   * * 1: Keys. A 1-D int32 tensor with shape [ n ], *MUST* be sorted in
   *      ascending order.
   * * 2: Values. A tensor with shape [ n … ].
   *
   * Outputs:
   * * 0: Output. A tensor with shape [ k …].
   * * 1: Hits. A uint8 tensor with shape [ k ] indicates whether the lookup
   *      hits or not.
   */
  ANEURALNETWORKS_HASHTABLE_LOOKUP = 10,

  /** Applies L2 normalization along the depth dimension.
   *
   * The values in the output tensor are computed as:
   *
   *     output[batch, row, col, channel] =
   *         input[batch, row, col, channel] /
   *         sqrt(sum_{c} pow(input[batch, row, col, c], 2))
   *
   * For x with more dimensions, independently normalizes each 1-D slice along
   * dimension dim.
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Inputs:
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the
   * input.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, out_height, out_width,
   * depth].
   */
  ANEURALNETWORKS_L2_NORMALIZATION = 11,

  /** Performs an 2-D L2 pooling operation.
   *
   * The output dimensions are functions of the filter dimensions, stride, and
   * padding.
   *
   * The values in the output tensor are computed as:
   *
   *     output[batch, row, col, channel] =
   *         sqrt(sum_{i, j} pow(input[batch, row + i, col + j, channel], 2) /
   * sum(1))
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Inputs:
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the
   * input.
   * * 1: An INT32 value, specifying the padding on the left, in the ‘width’
   * dimension.
   * * 2: An INT32 value, specifying the padding on the right,in the ‘width’
   * dimension.
   * * 3: An INT32 value, specifying the padding on the top, in the ‘height’
   * dimension.
   * * 4: An INT32 value, specifying the padding on the bottom, in the ‘height’
   * dimension.
   * * 5: An INT32 value, specifying the output stride in the ‘width’ dimension.
   * * 6: An INT32 value, specifying the output stride in the ‘height’
   * dimension.
   * * 7: An INT32 value, specifying the filter width.
   * * 8: An INT32 value, specifying the filter height.
   * * 9: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, out_height, out_width,
   * depth].
   */
  ANEURALNETWORKS_L2_POOL_2D = 12,
  /** Applies Local Response Normalization along the depth dimension.
   *
   * The 4-D input tensor is treated as a 3-D array of 1-D vectors (along the
   * last dimension), and each vector is normalized independently. Within a
   * given vector, each component is divided by the weighted, squared sum of
   * inputs within depth_radius.
   *
   * The output is calculated using this formula:
   *
   *     sqr_sum[a, b, c, d] =
   *         sum(pow(input[a, b, c, d - depth_radius : d + depth_radius + 1], 2)
   *     output = input / pow((bias + alpha * sqr_sum), beta)
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Inputs:
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the
   * input.
   * * 1: An INT32 value, specifying the radius of the normalization window.
   * * 2: A FLOAT32 value, specifying the bias, must not be zero.
   * * 3: A FLOAT32 value, specifying the scale factor, alpha.
   * * 4: A FLOAT32 value, specifying the exponent, beta.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0.
   */
  ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION = 13,
  /** Computes sigmoid activation on the input tensor element-wise.
   *
   * The output is calculated using this formula:
   *
   *     output = 1 / (1 + exp(-input))
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4.
   *
   * Inputs:
   * * 0: A tensor, specifying the input.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0.
   */
  ANEURALNETWORKS_LOGISTIC = 14,

  /**
   * Projects an input to a bit vector via locality senstive hashing.
   *
   * Inputs:
   * * 0: Hash functions. Dim.size == 2, DataType: Float.
   *            Tensor[0].Dim[0]: Number of hash functions.
   *            Tensor[0].Dim[1]: Number of seeds per hash functions.
   *            Tensor[0].Dim[1] <= 32 in sparse case.
   *
   * * 1: Input. Dim.size >= 1, no restriction on DataType.
   * * 2: Weight. Optional. Dim.size == 1, DataType: Float.
   *     If not set, each input element is considered to have the same weight of
   *     1.0.
   *     Tensor[1].Dim[0] == Tensor[2].Dim[0]
   * * 3: Type:
   *        Sparse: Value LSHProjectionType_SPARSE(=1).
   *          Computed bit vector is considered to be sparse.
   *          Each output element is an int32 made up of multiple bits computed
   * from hash functions.
   *
   *        Dense: Value LSHProjectionType_DENSE(=2).
   *          Computed bit vector is considered to be dense. Each output element
   *          represents a bit and can take the value of either 0 or 1.
   *
   * Outputs:
   * * 0: If the projection type is sparse:
   *        Output.Dim == { Tensor[0].Dim[0] }
   *        A tensor of int32 that represents hash signatures.
   *      If the projection type is Dense:
   *        Output.Dim == { Tensor[0].Dim[0] * Tensor[0].Dim[1] }
   *        A flattened tensor that represents projected bit vectors.
   */
  ANEURALNETWORKS_LSH_PROJECTION = 15,

  /**
   * Long short-term memory unit (LSTM) recurrent network layer.
   *
   * The default non-peephole implementation is based on:
   * http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
   * S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural
   * Computation, 9(8):1735-1780, 1997.
   *
   * The peephole implementation is based on:
   * https://research.google.com/pubs/archive/43905.pdf
   * Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory
   * recurrent neural network architectures for large scale acoustic modeling."
   * INTERSPEECH, 2014.
   *
   * The coupling of input and forget gate (CIFG) is based on:
   * http://arxiv.org/pdf/1503.04069.pdf
   * Greff et al. "LSTM: A Search Space Odyssey"
   *
   * The class has the following independently optional inputs:
   * * If input gate (if CIFG): “input_to_forget_weights”,
   *   “recurrent_to_input_weights”, “cell_to_input_weights”, “input_gate_bias”.
   * * If no peephole connections: “cell_to_input_weights”,
   *   “cell_to_forget_weights”, “cell_to_output_weights”.
   * * If no projection layer: “projection_weights” and “projection_bias”.
   * * If no projection bias: “projection_bias”.
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Inputs:
   * * 0: Input.
   *      A 2-D tensor of type T, of shape [batch_size, input_size], where
   *      “batch_size” corresponds to the batching dimension, and “input_size”
   *      is the size of the input.
   * * 1: input_to_input_weights.
   *      A 2-D tensor of type T, of shape [num_units, input_size], where
   *      “num_units” corresponds to the number of cell units.
   * * 2: input_to_forget_weights.
   *      A 2-D tensor of type T, of shape [num_units, input_size].
   * * 3: input_to_cell_weights.
   *      A 2-D tensor of type T, of shape [num_units, input_size].
   * * 4: input_to_output_weights.
   *      A 2-D tensor of type T, of shape [num_units, input_size].
   * * 5: recurrent_to_input_weights.
   *      A 2-D tensor of type T, of shape [num_units, output_size], where
   *      “output_size” corresponds to either the number of cell units (i.e.,
   *      “num_units”), or the second dimension of the “projection_weights”, if
   *      defined.
   * * 6: recurrent_to_forget_weights.
   *      A 2-D tensor of type T, of shape [num_units, output_size].
   * * 7: recurrent_to_cell_weights.
   *      A 2-D tensor of type T, of shape [num_units, output_size].
   * * 8: recurrent_to_output_weights.
   *      A 2-D tensor of type T, of shape [num_units, output_size].
   * * 9: cell_to_input_weights.
   *      A 1-D tensor of type T, of shape [num_units].
   * * 10:cell_to_forget_weights.
   *      A 1-D tensor of type T, of shape [num_units].
   * * 11:cell_to_output_weights.
   *      A 1-D tensor of type T, of shape [num_units].
   * * 12:input_gate_bias.
   *      A 1-D tensor of type T, of shape [num_units].
   * * 13:forget_gate_bias.
   *      A 1-D tensor of type T, of shape [num_units].
   * * 14:cell_bias.
   *      A 1-D tensor of type T, of shape [num_units].
   * * 15:output_gate_bias.
   *      A 1-D tensor of type T, of shape [num_units].
   * * 16:projection_weights.
   *      A 2-D tensor of type T, of shape [output_size, num_units].
   * * 17:projection_bias.
   *      A 1-D tensor of type T, of shape [output_size].
   *
   * Parameters:
   * * 18:fused_activation_function.
   *      An (optional) ActivationFunctionType indicating the activation
   *      function.
   *      If “NONE” is specified then it results in a linear activation.
   * * 19:cell_clip.
   *      A clipping threshold for the cell state, such that values are bound
   *      within [-cell_clip, cell_clip]. If set to 0.0 then clipping is
   *      disabled.
   * * 20:proj_clip.
   *      A clipping threshold for the output from the projection layer, such
   *      that values are bound within [-proj_clip, proj_clip]. If set to 0.0
   *      then clipping is disabled.
   *
   * Outputs:
   * * 0: scratch_buffer.
   *      A 3-D tensor of type T, of shape [batch_size, num_cell, 4].
   * * 1: output_state.
   *      A 2-D tensor of type T, of shape [batch_size, output_size].
   * * 2: cell_state.
   *      A 2-D tensor of type T, of shape [batch_size, num_units].
   * * 3: output.
   *      A 2-D tensor of type T, of shape [batch_size, output_size]. This is
   *      effectively the same as the current “output_state” value.
   */
  ANEURALNETWORKS_LSTM = 16,

  /** Performs an 2-D max pooling operation.
   *
   * The output dimensions are functions of the filter dimensions, stride, and
   * padding.
   *
   * The values in the output tensor are computed as:
   *
   *     output[batch, row, col, channel] =
   *         max_{i, j} (input[batch, row + i, col + j, channel])
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Inputs:
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the
   * input.
   * * 1: An INT32 value, specifying the padding on the left, in the ‘width’
   * dimension.
   * * 2: An INT32 value, specifying the padding on the right,in the ‘width’
   * dimension.
   * * 3: An INT32 value, specifying the padding on the top, in the ‘height’
   * dimension.
   * * 4: An INT32 value, specifying the padding on the bottom, in the ‘height’
   * dimension.
   * * 5: An INT32 value, specifying the output stride in the ‘width’ dimension.
   * * 6: An INT32 value, specifying the output stride in the ‘height’
   * dimension.
   * * 7: An INT32 value, specifying the filter width.
   * * 8: An INT32 value, specifying the filter height.
   * * 9: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, out_height, out_width,
   * depth].
   */
  ANEURALNETWORKS_MAX_POOL_2D = 17,

  /** Multiplies two tensors, element-wise.
   *
   * Takes two input tensors of identical type and compatible dimensions. The
   * output is the product of both input tensors, optionally modified by an
   * activation function.
   *
   * Two dimensions are compatible when:
   *     1. they are equal, or
   *     2. one of them is 1
   *
   * The size of the resulting output is the maximum size along each dimension
   * of the input operands. It starts with the trailing dimensions, and works
   * its way forward.
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A tensor.
   * * 1: A tensor of the same type, and compatible dimensions as input0.
   * * 2: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The product, a tensor of the same type as input0.
   */
  ANEURALNETWORKS_MUL = 18,
  /** Computes rectified linear activation on the input tensor element-wise.
   *
   * The output is calculated using this formula:
   *
   *     output = max(0, input)
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4.
   *
   * Inputs:
   * * 0: A tensor, specifying the input.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0.
   */
  ANEURALNETWORKS_RELU = 19,
  /** Computes rectified linear 1 activation on the input tensor element-wise.
   *
   * The output is calculated using this formula:
   *
   *     output = min(1.f, max(-1.f, input))
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4.
   *
   * Inputs:
   * * 0: A tensor, specifying the input.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0.
   */
  ANEURALNETWORKS_RELU1 = 20,
  /** Computes rectified linear 6 activation on the input tensor element-wise.
   *
   * The output is calculated using this formula:
   *
   *     output = min(6, max(0, input))
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4.
   *
   * Inputs:
   * * 0: A tensor, specifying the input.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0.
   */
  ANEURALNETWORKS_RELU6 = 21,
  /** Reshapes a tensor.
   *
   * Given tensor, this operation returns a tensor that has the same values as
   * tensor, but with a newly specified shape.
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4.
   *
   * Inputs:
   * * 0: A tensor, specifying the tensor to be reshaped.
   * * 1: A 1-D tensor of type {@link ANEURALNETWORKS_TENSOR_INT32}, defining
   * the shape of the output tensor. The number of elements implied by shape
   * must be the same as the number of elements in the input tensor.
   *
   * Outputs:
   * * 0: The output tensor, of shape specified by the input shape.
   */
  ANEURALNETWORKS_RESHAPE = 22,
  /** Resizes images to given size using the bilinear interpretation.
   *
   * Resized images will be distorted if their original aspect ratio is not the
   * same as input.
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Inputs:
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the
   * input.
   * * 1: An INT32 value, specifying the output width of the output tensor.
   * * 2: An INT32 value, specifying the output height of the output tensor.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, new_height, new_width,
   * depth].
   */
  ANEURALNETWORKS_RESIZE_BILINEAR = 23,

  /**
   * A basic recurrent neural network layer.
   *
   * This layer implements the operation:
   * outputs = state = activation(inputs * input_weights + state *
   * recurrent_weights + bias)
   *
   * Where:
   * * “input_weights” is a weight matrix that multiplies the inputs;
   * * “recurrent_weights” is a weight matrix that multiplies the current
   *    “state” which itself is the output from the previous time step
   *    computation;
   * * “bias” is a bias vector (added to each output vector in the batch);
   * * “activation” is the function passed as the “fused_activation_function”
   *   argument (if not “NONE”).
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Inputs:
   * * 0: input.
   *      A 2-D tensor of type T, of shape [batch_size, input_size], where
   *      “batch_size” corresponds to the batching dimension, and “input_size”
   * is the size of the input.
   * * 1: weights.
   *      A 2-D tensor of type T, of shape [num_units, input_size], where
   *      “num_units” corresponds to the number of units.
   * * 2: recurrent_weights.
   *      A 2-D tensor of type T, of shape [num_units, num_units], with columns
   *      corresponding to the weights from each unit.
   * * 3: bias.
   *      A 1-D tensor of type T, of shape [num_units].
   *
   *    For FLOAT32 input tensor, bias must also be FLOAT32.
   *    For UINT8 input tensor, bias must be INT32.
   *
   * Parameters
   * * 4: fused_activation_function.
   *      An (optional) ActivationFunctionType indicating the activation
   *      function. If “NONE” is specified then it results in a linear
   *      activation.
   *
   * * 5: Hidden state.
   *      A 2-D tensor of type T, of shape [batch_size, num_units].
   *
   * Outputs:
   * * 0: output.
   *      A 2-D tensor of type T, of shape [batch_size, num_units]. This is
   *      effectively the same as the current state value.
   */
  ANEURALNETWORKS_RNN = 24,

  /** Computes the softmax activation on the input tensor element-wise, per
   * batch, by normalizing the input vector so the maximum coefficient is zero.
   *
   * The output is calculated using this formula:
   *
   *     output[batch, i] =
   *         exp((input[batch, i] - max(input[batch, :])) * beta) /
   *         sum_{k}{exp((input[batch, k] - max(input[batch, :])) * beta)}
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 2 or 4.
   *
   * Inputs:
   * * 0: A 2-D or 4-D tensor, specifying the tensor to be reshaped.
   * * 1: A FLOAT32 value, specifying the scaling factor for the exponent, beta.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0.
   */
  ANEURALNETWORKS_SOFTMAX = 25,

  /** Rearranges blocks of spatial data, into depth.
   *
   * More specifically, this op outputs a copy of the input tensor where values
   * from the height and width dimensions are moved to the depth dimension. The
   * value block_size indicates the input block size and how the data is moved.
   *
   * Chunks of data of size block_size * block_size from depth are rearranged
   * into non-overlapping blocks of size block_size x block_size.
   *
   * The depth of the output tensor is input_depth * block_size * block_size.
   * The input tensor's height and width must be divisible by block_size.
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Inputs:
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying
   * the input.
   * * 1: An INT32 value, specifying the block_size. block_size must be >=1 and
   *      block_size must be a divisor of both the input height and width.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batch, height/block_size,
   * width/block_size, depth*block_size*block_size].
   */
  ANEURALNETWORKS_SPACE_TO_DEPTH = 26,

  /**
   * SVDF op is a kind of stateful layer derived from the notion that a
   * densely connected layer that's processing a sequence of input frames can
   * be approximated by using a singular value decomposition of each of its
   * nodes. The implementation is based on:
   *
   * https://research.google.com/pubs/archive/43813.pdf
   *
   * P. Nakkiran, R. Alvarez, R. Prabhavalkar, C. Parada.
   * “Compressing Deep Neural Networks using a Rank-Constrained Topology”.
   * INTERSPEECH, 2015.
   *
   * It processes the incoming input using a 2-stage filtering mechanism:
   * * stage 1 performs filtering on the "features" dimension, whose outputs get
   *   pushed into a memory of fixed-size memory_size.
   * * stage 2 performs filtering on the "time" dimension of the memory_size
   *   memoized outputs of stage 1.
   *
   * Specifically, for rank 1, this layer implements the operation:
   *
   *    memory = push(conv1d(inputs, weights_feature, feature_dim, "VALID"));
   *    outputs = activation(memory * weights_time + bias);
   *
   * Where:
   * * “weights_feature” is a weights matrix that processes the inputs (by
   *   convolving the input with every “feature filter”), and whose outputs get
   *   pushed, stacked in order, into the fixed-size “memory” (the oldest entry
   *   gets dropped);
   * * “weights_time” is a weights matrix that processes the “memory” (by a
   *   batched matrix multiplication on the num_units);
   * * “bias” is an optional bias vector (added to each output vector in the
   *   batch); and
   * * “activation” is the function passed as the “fused_activation_function”
   *   argument (if not “NONE”).
   *
   * Each rank adds a dimension to the weights matrices by means of stacking
   * the filters.
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Inputs:
   * * 0: input.
   *      A 2-D tensor of type T, of shape [batch_size, input_size], where
   *      “batch_size” corresponds to the batching dimension, and “input_size”
   * is the size of the input.
   * * 1: weights_feature.
   *      A 2-D tensor of type T, of shape [num_units, input_size], where
   *      “num_units” corresponds to the number of units.
   * * 2: weights_time.
   *      A 2-D tensor of type T, of shape [num_units, memory_size], where
   *      “memory_size” corresponds to the fixed-size of the memory.
   * * 3: bias.
   *      A optional 1-D tensor of type T, of shape [num_units].
   *
   *    For FLOAT32 input tensor, bias must also be FLOAT32.
   *    For UINT8 input tensor, bias must be INT32.
   *
   * Parameters:
   * * 4: rank.
   *      The rank of the SVD approximation.
   * * 5: fused_activation_function.
   *      An (optional) ActivationFunctionType indicating the activation
   * function. If “NONE” is specified then it results in a linear activation.
   *
   * Outputs:
   * * 0: state.
   *      A 2-D tensor of type T, of shape [batch_size, (memory_size - 1) *
   * num_units * rank].
   * * 1: output.
   *      A 2-D tensor of type T, of shape [batch_size, num_units].
   */
  ANEURALNETWORKS_SVDF = 27,

  /** Computes hyperbolic tangent of input tensor element-wise.
   *
   * The output is calculated using this formula:
   *
   *     output = tanh(input)
   *
   * Supported tensor types:
   * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
   *
   * Supported tensor rank: up to 4.
   *
   * Inputs:
   * * 0: A tensor, specifying the input.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0.
   */
  ANEURALNETWORKS_TANH = 28,
};

/**
 * Fused activation function types.
 *
 */
enum {
  /** NO fused activation function. */
  ANEURALNETWORKS_FUSED_NONE = 0,
  /** Fused ReLU activation function. */
  ANEURALNETWORKS_FUSED_RELU = 1,
  /** Fused ReLU1 activation function. */
  ANEURALNETWORKS_FUSED_RELU1 = 2,
  /** Fused ReLU6 activation function. */
  ANEURALNETWORKS_FUSED_RELU6 = 3,
};

/**
 * Execution preferences.
 */
enum {
  /**
   * Prefer executing in a way that minimizes battery drain.
   * This is desirable for compilations that will be executed often.
   */
  ANEURALNETWORKS_PREFER_LOW_POWER = 0,
  /**
   * Prefer returning a single answer as fast as possible, even if this causes
   * more power consumption.
   */
  ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1,
  /**
   * Prefer maximizing the throughput of successive frames, for example when
   * processing successive frames coming from the camera.
   */
  ANEURALNETWORKS_PREFER_SUSTAINED_SPEED = 2,
};

/**
 * Result codes.
 */
enum {
  ANEURALNETWORKS_NO_ERROR = 0,
  ANEURALNETWORKS_OUT_OF_MEMORY = 1,
  ANEURALNETWORKS_INCOMPLETE = 2,
  ANEURALNETWORKS_UNEXPECTED_NULL = 3,
  ANEURALNETWORKS_BAD_DATA = 4,
  ANEURALNETWORKS_OP_FAILED = 5,
  ANEURALNETWORKS_UNMAPPABLE = 5,
  ANEURALNETWORKS_BAD_STATE = 6,
};

/**
 * ANeuralNetworksMemory is an opaque type that represents memory.
 *
 * This type is used to represent shared memory, memory mapped files,
 * and similar memories.
 *
 * By using shared memory, a program can efficiently communicate to the
 * runtime and drivers the tensors that define a model. See
 * {@link ANeuralNetworksModel_setOperandValueFromMemory}. An application
 * should typically create one shared memory object that contains every tensor
 * needed to define a model. {@link ANeuralNetworksMemory_createFromFd} can be
 * used to create shared memory from a file handle. {@link
 * ANeuralNetworksMemory_createShared} can be used to directly created shared
 * memory.
 *
 * Memory objects can also be used to specify the input and output arguments of
 * an execution. See {@link ANeuralNetworksExecution_setInputFromMemory}
 * and {@link ANeuralNetworksExecution_setOutputFromMemory}.
 */
typedef struct ANeuralNetworksMemory ANeuralNetworksMemory;

/**
 * ANeuralNetworksModel is an opaque type that contains a description of the
 * mathematical operations that constitute the model.
 *
 * <p>The model will be built by calling<ul>
 * <li>{@link ANeuralNetworksModel_create},</li>
 * <li>{@link ANeuralNetworksModel_addOperation},</li>
 * <li>{@link ANeuralNetworksModel_addOperand},</li>
 * </ul>
 *
 * A model is completed by calling {@link ANeuralNetworksModel_finish}.
 * A model is destroyed by calling {@link ANeuralNetworksModel_free}.
 *
 * <p>It is the application's responsibility to make sure that only one thread
 * modifies a model at a given time. It is however safe for more than one
 * thread to use the model once {@link ANeuralNetworksModel_finish} has
 * returned.</p>
 *
 * <p>It is also the application's responsibility to ensure that there are no
 * other uses of the model after calling {@link ANeuralNetworksModel_free}. This
 * includes any compilation or execution object created using the model.</p>
 */
typedef struct ANeuralNetworksModel ANeuralNetworksModel;

/**
 * ANeuralNetworksCompilation is an opaque type that can be used to compile
 * a machine learning model.
 *
 * <p>To use:<ul>
 *    <li>Create a new compilation instance by calling the
 *        {@link ANeuralNetworksCompilation_create} function.</li>
 *    <li>Perform the compilation with {@link
 * ANeuralNetworksCompilation_start}.</li> <li>Wait for the compilation to
 * complete with {@link ANeuralNetworksCompilation_wait}.</li> <li>Use the
 * compilation as many times as needed with {@link
 * ANeuralNetworksExecution_create}.</li> <li>Destroy the compilation with
 * {@link ANeuralNetworksCompilation_free} once all executions using the
 * compilation have completed.</li></ul></p>
 *
 * <p>A compilation cannot be modified once {@link
 * ANeuralNetworksCompilation_start} has been called on it.</p>
 *
 * <p>It is the application's responsibility to make sure that only one thread
 * modifies a compilation at a given time. It is however safe for more than one
 * thread to use {@link ANeuralNetworksCompilation_wait} at the same time.
 * It is also safe for multiple threads to use a compilation object once
 * {@link ANeuralNetworksCompilation_wait} has completed.</p>
 *
 * <p>It is also the application's responsibility to ensure that there are no
 * other uses of the compilation after calling {@link
 * ANeuralNetworksCompilation_free}. This includes any execution object created
 * using the compilation.</p>
 */
typedef struct ANeuralNetworksCompilation ANeuralNetworksCompilation;

/**
 * ANeuralNetworksExecution is an opaque type that can be used to apply a
 * machine learning model to a set of inputs.
 *
 * <p>To use:<ul>
 *    <li>Create a new execution instance by calling the
 *        {@link ANeuralNetworksExecution_create} function.</li>
 *    <li>Associate data to the model inputs with
 *        {@link ANeuralNetworksExecution_setInput} or
 *        {@link ANeuralNetworksExecution_setInputFromMemory}.</li>
 *    <li>Associate output buffers to the model outputs with
 *        {@link ANeuralNetworksExecution_setOutput} or
 *        {@link ANeuralNetworksExecution_setOutputFromMemory}.</li>
 *    <li>Apply the model with {@link
 * ANeuralNetworksExecution_startCompute}.</li> <li>Wait for the execution to
 * complete with {@link ANeuralNetworksExecution_wait}.</li> <li>Destroy the
 * execution with
 *        {@link ANeuralNetworksExecution_free}.</li></ul></p>
 *
 * <p>An execution cannot be modified once {@link
 * ANeuralNetworksExecution_start} has been called on it.</p>
 *
 * <p>An execution can be applied to a model with
 * {@link ANeuralNetworksExecution_startCompute} only once. Create new
 * executions to do new evaluations of the model.</p>
 *
 * <p>It is the application's responsibility to make sure that only one thread
 * modifies an execution at a given time. It is however safe for more than one
 * thread to use {@link ANeuralNetworksExecution_wait} at the same time.</p>
 *
 * <p>It is also the application's responsibility to ensure that there are no
 * other uses of the request after calling {@link
 * ANeuralNetworksRequest_free}.</p>
 */
typedef struct ANeuralNetworksExecution ANeuralNetworksExecution;

/**
 * ANeuralNetworksOperandType describes the type of an operand.
 * This structure is used to describe both scalars and tensors.
 */
typedef struct ANeuralNetworksOperandType {
  /** The data type, e.g ANEURALNETWORKS_INT8. */
  int32_t type;
  /** The number of dimensions. It should be 0 for scalars. */
  uint32_t dimensionCount;
  /** The dimensions of the tensor. It should be nullptr for scalars. */
  const uint32_t* dimensions;
  /** These two fields are only used for quantized tensors.
   * They should be zero for scalars and non-fixed point tensors.
   * The dequantized value of each entry is (value - offset) * scale.
   */
  float scale;
  int32_t zeroPoint;
} ANeuralNetworksOperandType;

/**
 * ANeuralNetworksEvent is an opaque type that represents an event
 * that will be signaled once an execution completes.
 */
typedef struct ANeuralNetworksEvent ANeuralNetworksEvent;

typedef int32_t ANeuralNetworksOperationType;

// nn api function types

typedef int (*ANeuralNetworksMemory_createFromFd_fn)(
    size_t size, int protect, int fd, size_t offset,
    ANeuralNetworksMemory** memory);

typedef void (*ANeuralNetworksMemory_free_fn)(ANeuralNetworksMemory* memory);

typedef int (*ANeuralNetworksModel_create_fn)(ANeuralNetworksModel** model);

typedef int (*ANeuralNetworksModel_finish_fn)(ANeuralNetworksModel* model);

typedef void (*ANeuralNetworksModel_free_fn)(ANeuralNetworksModel* model);

typedef int (*ANeuralNetworksCompilation_create_fn)(
    ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation);

typedef void (*ANeuralNetworksCompilation_free_fn)(
    ANeuralNetworksCompilation* compilation);

typedef int (*ANeuralNetworksCompilation_setPreference_fn)(
    ANeuralNetworksCompilation* compilation, int32_t preference);

typedef int (*ANeuralNetworksCompilation_finish_fn)(
    ANeuralNetworksCompilation* compilation);

typedef int (*ANeuralNetworksModel_addOperand_fn)(
    ANeuralNetworksModel* model, const ANeuralNetworksOperandType* type);

typedef int (*ANeuralNetworksModel_setOperandValue_fn)(
    ANeuralNetworksModel* model, int32_t index, const void* buffer,
    size_t length);

typedef int (*ANeuralNetworksModel_setOperandValueFromMemory_fn)(
    ANeuralNetworksModel* model, int32_t index,
    const ANeuralNetworksMemory* memory, size_t offset, size_t length);

typedef int (*ANeuralNetworksModel_addOperation_fn)(
    ANeuralNetworksModel* model, ANeuralNetworksOperationType type,
    uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
    const uint32_t* outputs);

typedef int (*ANeuralNetworksModel_identifyInputsAndOutputs_fn)(
    ANeuralNetworksModel* model, uint32_t inputCount, const uint32_t* inputs,
    uint32_t outputCount, const uint32_t* outputs);

typedef int (*ANeuralNetworksExecution_create_fn)(
    ANeuralNetworksCompilation* compilation,
    ANeuralNetworksExecution** execution);

typedef void (*ANeuralNetworksExecution_free_fn)(
    ANeuralNetworksExecution* execution);

typedef int (*ANeuralNetworksExecution_setInput_fn)(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const void* buffer, size_t length);

typedef int (*ANeuralNetworksExecution_setInputFromMemory_fn)(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    size_t offset, size_t length);

typedef int (*ANeuralNetworksExecution_setOutput_fn)(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, void* buffer, size_t length);

typedef int (*ANeuralNetworksExecution_setOutputFromMemory_fn)(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    size_t offset, size_t length);

typedef int (*ANeuralNetworksExecution_startCompute_fn)(
    ANeuralNetworksExecution* execution, ANeuralNetworksEvent** event);

typedef int (*ANeuralNetworksEvent_wait_fn)(ANeuralNetworksEvent* event);

typedef void (*ANeuralNetworksEvent_free_fn)(ANeuralNetworksEvent* event);

/**
 * Creates a shared memory object from a file descriptor.
 *
 * The shared memory is backed by a file descriptor via mmap.
 * See {@link ANeuralNetworksMemory} for a description on how to use
 * this shared memory.
 *
 * @param size The requested size in bytes.
 *             Must not be larger than the file size.
 * @param prot The desired memory protection for the mapping.
 *             It is either PROT_NONE or the bitwise OR of one or
 *             more of the following flags: PROT_READ, PROT_WRITE.
 * @param fd The requested file descriptor.
 *           The file descriptor has to be mmap-able. The file
 *           descriptor will be duplicated.
 * @param offset The offset to the beginning of the file of the area to map.
 *               The offset has to be aligned to a page size.
 * @param memory The memory object to be created.
 *               Set to NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the request completed normally.
 */
inline int ANeuralNetworksMemory_createFromFd(size_t size, int protect, int fd,
                                              size_t offset,
                                              ANeuralNetworksMemory** memory) {
  LOAD_FUNCTION(ANeuralNetworksMemory_createFromFd);
  EXECUTE_FUNCTION_RETURN(size, protect, fd, offset, memory);
}

/**
 * Delete a memory object.
 *
 * Destroys the object used by the run time to keep track of the memory.
 * This will free the underlying actual memory if no other code has open
 * handles to this memory.
 *
 * @param memory The memory object to be freed.
 */
inline void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory) {
  LOAD_FUNCTION(ANeuralNetworksMemory_free);
  EXECUTE_FUNCTION(memory);
}

/**
 * Create an empty {@link ANeuralNetworksModel}.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuralNetworksExecution_startCompute} is invoked.
 *
 * The model should be constructed with calls to
 * {@link ANeuralNetworksModel_addOperation} and
 * {@link ANeuralNetworksModel_addOperand}
 *
 * <p>{@link ANeuralNetworksModel_finish} should be called once the model
 * has been fully constructed.</p>
 *
 * <p>{@link ANeuralNetworksModel_free} should be called once the model
 * is no longer needed.</p>
 *
 * @param model The {@link ANeuralNetworksModel} to be created.
 *              Set to NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_create(ANeuralNetworksModel** model) {
  LOAD_FUNCTION(ANeuralNetworksModel_create);
  EXECUTE_FUNCTION_RETURN(model);
}

/**
 * Destroy a model.
 *
 * The model need not have been finished by a call to
 * {@link ANeuralNetworksModel_finish}.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be destroyed. Passing NULL is acceptable and
 *              results in no operation.
 */
inline void ANeuralNetworksModel_free(ANeuralNetworksModel* model) {
  LOAD_FUNCTION(ANeuralNetworksModel_free);
  EXECUTE_FUNCTION(model);
}

/**
 * Indicate that we have finished modifying a model. Required before
 * calling {@link ANeuralNetworksCompilation_compile}.
 *
 * An application is responsible to make sure that no other thread uses
 * the model at the same time.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be finished.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_finish(ANeuralNetworksModel* model) {
  LOAD_FUNCTION(ANeuralNetworksModel_finish);
  EXECUTE_FUNCTION_RETURN(model);
}

/**
 * Add an operand to a model.
 *
 * The order in which the operands are added is important. The first one added
 * to a model will have the index value 0, the second 1, etc. These indexes are
 * used as operand identifiers in {@link ANeuralNetworksModel_addOperation},
 * {@link ANeuralNetworksExecution_setInput},
 * {@link ANeuralNetworksExecution_setInputFromMemory},
 * {@link ANeuralNetworksExecution_setOutput},
 * {@link ANeuralNetworksExecution_setOutputFromMemory} and
 * {@link ANeuralNetworksExecution_setOperandValue}.
 *
 * To build a model that can accommodate inputs of various sizes, as you may
 * want to do for a CNN, set the size of the dimensions that will vary at run
 * time to 0. If you do so, provide the full dimensions when calling
 * {@link ANeuralNetworksExecution_setInput} or {@link
 * ANeuralNetworksExecution_setInputFromMemory}.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be modified.
 * @param type The {@link ANeuralNetworksOperandType} that describes the shape
 * of the operand.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_addOperand(
    ANeuralNetworksModel* model, const ANeuralNetworksOperandType* type) {
  LOAD_FUNCTION(ANeuralNetworksModel_addOperand);
  EXECUTE_FUNCTION_RETURN(model, type);
}

/**
 * Sets an operand to a constant value.
 *
 * For scalar values, the content of buffer is copied into the model.
 *
 * For tensor values, a pointer to the buffer is stored within the model.
 * The application is responsible for not changing the content of this region
 * until all executions using this model have completed. As the data may
 * be copied during processing, modifying the data after this call yields
 * undefined results.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param buffer A pointer to the data to use.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model,
                                                int32_t index,
                                                const void* buffer,
                                                size_t length) {
  LOAD_FUNCTION(ANeuralNetworksModel_setOperandValue);
  EXECUTE_FUNCTION_RETURN(model, index, buffer, length);
}

/**
 * Sets an operand to a value stored in a memory object.
 *
 * The content of the memory is not copied. A reference to that memory is stored
 * inside the model. The application is responsible for not changing the content
 * of the memory region until all executions using this model have completed.
 * As the data may be copied during processing, modifying the data after this
 * call yields undefined results.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param buffer A pointer to the data to use.
 * @param memory The memory containing the data.
 * @param offset This specifies the location of the data within the memory.
 *               The offset is in bytes from the start of memory.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_setOperandValueFromMemory(
    ANeuralNetworksModel* model, int32_t index,
    const ANeuralNetworksMemory* memory, size_t offset, size_t length) {
  LOAD_FUNCTION(ANeuralNetworksModel_setOperandValueFromMemory);
  EXECUTE_FUNCTION_RETURN(model, index, memory, offset, length);
}

/**
 * Add an operation to a model.
 *
 * @param model The model to be modified.
 * @param type The type of the operation.
 * @param inputCount The number of entries in the inputs array.
 * @param inputs An array of indexes identifying each operand.
 * @param outputCount The number of entries in the outputs array.
 * @param outputs An array of indexes identifying each operand.
 *
 * The operands specified by inputs and outputs must have been
 * previously added by calls to {@link ANeuralNetworksModel_addOperand}.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                             ANeuralNetworksOperationType type,
                                             uint32_t inputCount,
                                             const uint32_t* inputs,
                                             uint32_t outputCount,
                                             const uint32_t* outputs) {
  LOAD_FUNCTION(ANeuralNetworksModel_addOperation);
  EXECUTE_FUNCTION_RETURN(model, type, inputCount, inputs, outputCount,
                          outputs);
}

/**
 * Specfifies which operands will be the model's inputs and outputs.
 *
 * An operand cannot be used for both input and output. Doing so will
 * return an error.
 *
 * @param model The model to be modified.
 * @param inputCount The number of entries in the inputs array.
 * @param inputs An array of indexes identifying the input operands.
 * @param outputCount The number of entries in the outputs array.
 * @param outputs An array of indexes identifying the output operands.
 *
 * The operands specified by inputs and outputs must have been
 * previously added by calls to {@link ANeuralNetworksModel_addOperand}.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 */
inline int ANeuralNetworksModel_identifyInputsAndOutputs(
    ANeuralNetworksModel* model, uint32_t inputCount, const uint32_t* inputs,
    uint32_t outputCount, const uint32_t* outputs) {
  LOAD_FUNCTION(ANeuralNetworksModel_identifyInputsAndOutputs);
  EXECUTE_FUNCTION_RETURN(model, inputCount, inputs, outputCount, outputs);
}

/**
 * Create a {@link ANeuralNetworksCompilation} to compile the given model.
 * This only creates the object. Compilation is only performed once
 * {@link ANeuralNetworksCompilation_start} is invoked.
 *
 * <p>The provided model must outlive the compilation.</p>
 *
 * The model must already have been finished by a call to
 * {@link ANeuralNetworksModel_finish}.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * @param model The {@link ANeuralNetworksModel} to be compiled.
 * @param compilation The newly created object or NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
 *         if the model is invalid.
 */
inline int ANeuralNetworksCompilation_create(
    ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation) {
  LOAD_FUNCTION(ANeuralNetworksCompilation_create);
  EXECUTE_FUNCTION_RETURN(model, compilation);
}

/**
 * Destroy a compilation.
 *
 * <p>If called on a compilation for which
 * {@link ANeuralNetworksCompilation_start} has been called, the
 * function will return immediately but will mark the compilation to be deleted
 * once the compilation completes. The {@link ANeuralNetworksCompilation_wait}
 * will return ERROR_DELETED.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * @param compilation The compilation to be destroyed. Passing NULL is
 * acceptable and results in no operation.
 */
inline void ANeuralNetworksCompilation_free(
    ANeuralNetworksCompilation* compilation) {
  LOAD_FUNCTION(ANeuralNetworksCompilation_free);
  EXECUTE_FUNCTION(compilation);
}

/**
 * Sets the execution preference.
 *
 * <p>Provides guidance to the runtime when trade-offs are possible.</p>
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * @param compilation The compilation to be modified.
 * @param preference Either {@link PREFER_LOW_POWER},
 *                  {@link PREFER_SINGLE_FAST_ANSWER}, or
 *                  {@link PREFER_SUSTAINED_SPEED}.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksCompilation_setPreference(
    ANeuralNetworksCompilation* compilation, int32_t preference) {
  LOAD_FUNCTION(ANeuralNetworksCompilation_setPreference);
  EXECUTE_FUNCTION_RETURN(compilation, preference);
}

/**
 * Waits until the compilation completes.
 *
 * More than one thread can wait on a compilation. When the compilation
 * completes, all threads will be released.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the compilation completed normally.
 */
inline int ANeuralNetworksCompilation_finish(
    ANeuralNetworksCompilation* compilation) {
  LOAD_FUNCTION(ANeuralNetworksCompilation_finish);
  EXECUTE_FUNCTION_RETURN(compilation);
}
/**
 * Create a {@link ANeuralNetworksExecution} to apply the given compilation.
 * This only creates the object. Computation is only performed once
 * {@link ANeuralNetworksExecution_startCompute} is invoked.
 *
 * <p>The provided compilation must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param compilation The {@link ANeuralNetworksCompilation} to be evaluated.
 * @param execution The newly created object or NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
 *         if the compilation is invalid.
 */
inline int ANeuralNetworksExecution_create(
    ANeuralNetworksCompilation* compilation,
    ANeuralNetworksExecution** execution) {
  LOAD_FUNCTION(ANeuralNetworksExecution_create);
  EXECUTE_FUNCTION_RETURN(compilation, execution);
}

/**
 * Destroy an execution.
 *
 * <p>If called on an execution for which
 * {@link ANeuralNetworksExecution_startCompute} has been called, the
 * function will return immediately but will mark the execution to be deleted
 * once the computation completes.   The {link ANeuralNetworksExecution_wait}
 * will return ANEURALNETWORKS_ERROR_DELETED.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be destroyed. Passing NULL is acceptable
 * and results in no operation.
 */
inline void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution) {
  LOAD_FUNCTION(ANeuralNetworksExecution_free);
  EXECUTE_FUNCTION(execution);
}

/**
 * Associate a user buffer with an input of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * <p>The provided buffer must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be modified.
 * @param index The index of the input argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param type The type of the operand. This should be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other properties of the type must be the same as
 *             specified in the model. If the type is the same as specified
 *             when the model was built, NULL can be passed.
 * @param buffer The buffer containing the data.
 * @param length The length in bytes of the buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the input.
 */
inline int ANeuralNetworksExecution_setInput(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const void* buffer, size_t length) {
  LOAD_FUNCTION(ANeuralNetworksExecution_setInput);
  EXECUTE_FUNCTION_RETURN(execution, index, type, buffer, length);
}

/**
 * Associate part of a memory object with an input of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * <p>The provided memory must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be modified.
 * @param index The index of the input argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param type The type of the operand. This can be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other values must be the same as specified in the
 *             model. If the type is the same as specified when the model
 *             was built, NULL can be passed.
 * @param memory The memory containing the data.
 * @param offset This specifies the location of the data within the memory.
 *               The offset is in bytes from the start of memory.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the input.
 */
inline int ANeuralNetworksExecution_setInputFromMemory(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    size_t offset, size_t length) {
  LOAD_FUNCTION(ANeuralNetworksExecution_setInputFromMemory);
  EXECUTE_FUNCTION_RETURN(execution, index, type, memory, offset, length);
}

/**
 * Associate a user buffer with an output of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * <p>The provided buffer must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be modified.
 * @param index The index of the output argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param type The type of the operand. This can be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other values must be the same as specified in the
 *             model. If the type is the same as specified when the model
 *             was built, NULL can be passed.
 * @param buffer The buffer where the data is to be written.
 * @param length The length in bytes of the buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the output.
 */
inline int ANeuralNetworksExecution_setOutput(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, void* buffer, size_t length) {
  LOAD_FUNCTION(ANeuralNetworksExecution_setOutput);
  EXECUTE_FUNCTION_RETURN(execution, index, type, buffer, length);
}

/**
 * Associate part of a memory object with an output of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * <p>The provided memory must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be modified.
 * @param index The index of the output argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param type The type of the operand. This can be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other values must be the same as specified in the
 *             model. If the type is the same as specified when the model
 *             was built, NULL can be passed.
 * @param memory The memory where the data is to be stored.
 * @param offset This specifies the location of the data within the memory.
 *               The offset is in bytes from the start of memory.
 * @param length The length in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the output.
 */
inline int ANeuralNetworksExecution_setOutputFromMemory(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    size_t offset, size_t length) {
  LOAD_FUNCTION(ANeuralNetworksExecution_setOutputFromMemory);
  EXECUTE_FUNCTION_RETURN(execution, index, type, memory, offset, length);
}

/**
 * Schedule evaluation of the execution.
 *
 * <p>Schedules evaluation of the execution. Once the model has been
 * applied and the outputs are ready to be consumed, the execution will be
 * signaled. Use {@link ANeuralNetworksExecution_wait} to wait for that signal.
 * </p>
 *
 * Multiple executions can be scheduled and evaluated concurrently, and
 * compilations can be performed concurrently with executions. The runtime makes
 * no guarantee on the ordering of the completion of compilations and
 * executions. If it's important to the application, the application should
 * enforce the ordering by using {@link ANeuralNetworksCompilation_wait} and
 * {@link ANeuralNetworksExecution_wait}.
 *
 * ANeuralNetworksExecution_wait must be called to recuperate the resources used
 * by the execution.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be scheduled and executed.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksExecution_startCompute(
    ANeuralNetworksExecution* execution, ANeuralNetworksEvent** event) {
  LOAD_FUNCTION(ANeuralNetworksExecution_startCompute);
  EXECUTE_FUNCTION_RETURN(execution, event);
}

/**
 * Waits until the execution completes.
 *
 * More than one thread can wait on an event. When the execution completes,
 * all threads will be released.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the execution completed normally.
 */
inline int ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event) {
  LOAD_FUNCTION(ANeuralNetworksEvent_wait);
  EXECUTE_FUNCTION_RETURN(event);
}

/**
 * Destroys the event.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 */
inline void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event) {
  LOAD_FUNCTION(ANeuralNetworksEvent_free);
  EXECUTE_FUNCTION(event);
}

/**/

#endif  // NN_API_SHIM_H0

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_

#include <algorithm>
#include <cmath>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/lite/c/builtin_op_data.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {

// Not all backends support CpuBackendContext usage, so forward declare to avoid
// pulling in its implementation. Use of CpuBackendContext in method
// implementations is purely optional.
class CpuBackendContext;

namespace tensor_utils {

// Checks if all entries of vector are zero for float.
bool IsZeroVector(const float* vector, int v_size);

// Checks if all entries of vector are zero for int8.
bool IsZeroVector(const int8_t* vector, int v_size);

// Quantizes a buffer of floating point values using a symmetric quantization
// (i.e. linear quantization without an offset) to 8-bit signed integers.
// It also outputs the range (min, max) of the floating point buffer, and the
// scaling factor used to quantize the values.
void SymmetricQuantizeFloats(const float* values, const int size,
                             int8_t* quantized_values, float* min_value,
                             float* max_value, float* scaling_factor);

// Quantizes a buffer of floating point values using a symmetric quantization
// (i.e. linear quantization without an offset) to 8-bit signed integers.
// It uses the range (min, max) provided to the function to calculate the
// appropriate scaling factor to quantize the values.
void SymmetricQuantizeFloats(const float* values, const int size,
                             int8_t* quantized_values, float min_value,
                             float max_value, float* scaling_factor);

void AsymmetricQuantizeFloats(const float* values, const int size,
                              int8_t* quantized_values, float* scaling_factor,
                              int32_t* offset);

// Helper function to quantize floats.
// float_data_ptr     input float vectors
// n_batch            number of input vectors
// n_data             size of a single input vector
// quantized_data_ptr (out) vector with quantized data
// scaling_factors    (out) scaling factors (one per vector)
// zero_points        (out) zero points (one per vector)
// do_asymmetric      controls if the quantization should be asymmetric.
inline void BatchQuantizeFloats(const float* float_data_ptr, int n_batch,
                                int n_data, int8_t* quantized_data_ptr,
                                float* scaling_factors, int32_t* zero_points,
                                bool do_asymmetric) {
  for (int b = 0; b < n_batch; ++b) {
    const int offset = b * n_data;
    if (do_asymmetric) {
      tensor_utils::AsymmetricQuantizeFloats(
          float_data_ptr + offset, n_data, quantized_data_ptr + offset,
          &scaling_factors[b], &zero_points[b]);
    } else {
      float unused_min, unused_max;
      tensor_utils::SymmetricQuantizeFloats(
          float_data_ptr + offset, n_data, quantized_data_ptr + offset,
          &unused_min, &unused_max, &scaling_factors[b]);
    }
  }
}

// Multiplies a matrix by a "batched" vector (i.e. a matrix with a batch
// dimension composed by input vectors independent from each other). The result
// of the multiplication is accumulated to the passed result buffer.
// More specifically, for a matrix M of shape [n, i] and a batched-vector
// of shape [i, batch] it will first compute the product of shape [n, batch].
// This product will be accumulated to the result buffer.
void MatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                         int m_cols, const float* vector,
                                         int n_batch, float* result);

// Same as the function above, but the matrix is a sparse tensor with block
// pattern 1x4.
// This function assumes that m_cols is a multiple of the block size (4 in this
// case) so that there's no incomplete block.
void SparseMatrixBatchVectorMultiplyAccumulate1x4(
    const float* __restrict__ matrix, const int32_t* __restrict__ segments,
    const int32_t* __restrict__ indices, int m_rows, int m_cols,
    const float* __restrict__ vector, int n_batch, float* __restrict__ result);

// Same as the function above, but the matrix is stored in block compressed
// sparse row format with block pattern 1x16 which consists of two arrays:
//   1. A matrix array stores non-zero blocks of the matrix in row major.
//   2. A ledger array stores nrows groups, one group per row. Each group starts
//      with an integer representing the number of non-zero blocks for the
//      corresponding row and follows with column indexes of the first element
//      of each non-zero block.
// This function assumes that
//   1. m_cols is a multiple of 16 so that all blocks are full blocks.
//   2. m_cols < 254 * 16 so that block index can be represented by uint8.
void SparseMatrixBatchVectorMultiplyAccumulate(
    const float* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    int m_rows, int m_cols, const float* __restrict__ vector, int n_batch,
    float* __restrict__ result);

// Same as the function above, but for values quantized using symmetric
// quantization (e.g. by calling SymmetricQuantizeFloats).
// The passed scaling factors is a buffer of the quantization scaling factors
// that will be used to dequentize the products into the final result buffer.
// These scaling factors are the multiplication of the matrix scaling factor
// by the vector's scaling factor, one per batch (i.e. this allows quantizing
// each batch in the batch-vector matrix independently).
void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    float* __restrict__ result);

// Same as the function above, but provide a scratch buffer for the
// int8 x int8 -> int32 and a CpuBackendContext for the accumulator
// computation.
void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    int32_t* __restrict__ scratch, float* __restrict__ result,
    CpuBackendContext* __restrict__ context);

// Same as the function above except that vector values
// are quantized with asymmetric quantization per-batch and the matrix
// is quantized per row.
void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    float* __restrict__ result, const float* __restrict__ per_channel_scale,
    const int32_t* __restrict__ input_offset);

// Same as the function above except that can make use of cached row sums.
void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset, int32_t* scratch, int32_t* row_sums,
    bool* compute_row_sums, CpuBackendContext* context);

// Same as the function above, but the matrix is stored in block compressed
// sparse row format with block pattern 1x16 which consists of two arrays:
//   1. A matrix array stores non-zero blocks of the matrix in row major.
//   2. A ledger array stores nrows groups, one group per row. Each group starts
//      with an integer representing the number of non-zero blocks for the
//      corresponding row followed by column index of the first element of
//      each non-zero block.
// This function assumes that
//   1. m_cols is a multiple of 16 so that all blocks are full blocks.
//   2. m_cols < 254 * 16 so that block index can be represented by uint8.
void SparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    const int m_rows, const int m_cols, const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    float* __restrict__ result);

// Multiplies a matrix by a "batched" vector (i.e. a matrix with a batch
// dimension composed by input vectors independent from each other). The result
// of the multiplication is accumulated to the passed result buffer.
// More specifically, for a matrix M of shape [n, i] and a batched-vector
// of shape [i, batch] it will first compute the product of shape [n, batch].
// This product will be accumulated to the result buffer,
// Parameters:
//     - input: batch vector of size n_batch * n_input
//     - bias:  vector of size b_input
//     - input_to_gate_weights: matrix of size n_input * n_output
//     - multiplier: scalar
//     - shift: scalar
//     - n_batch: the batch size
//     - n_input: the input size
//     - n_output: the output size
//     - output_zp: the zero point of the output.
//     - scratch: batch vector of size n_batch * n_output
//     - output: the 16 bit output
// Notes:
//     - this is used for gate matmul: for non-cifg it is for input, forget,
//       cell, output gates; for cifg, it is for forget, cell, output gates.
//     - multiplier and shift combined gives the scale.
//     - assumes input zero point is 0.
//     - scratch is created for optimization purpose only.
//       TODO(b/152066492): this can be removed if some future optimization
//       work makes it unnecessary.
void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int16_t* output, CpuBackendContext* context);

// Multiplies a matrix by a "batched" vector (i.e. a matrix with a batch
// dimension composed by input vectors independent from each other). The result
// of the multiplication is accumulated to the passed result buffer.
// More specifically, for a matrix M of shape [n, i] and a batched-vector
// of shape [i, batch] it will first compute the product of shape [n, batch].
// This product will be accumulated to the result buffer,
// Parameters:
//     - input: batch vector of size n_batch * n_input
//     - bias:  vector of size b_input
//     - input_to_gate_weights: matrix of size n_input * n_output
//     - multiplier: scalar
//     - shift: scalar
//     - n_batch: the batch size
//     - n_input: the input size
//     - n_output: the output size
//     - output_zp: the zero point of the output.
//     - scratch: batch vector of size n_batch * n_output
//     - output: the 8 bit output
// Notes:
//     - this is used for projection matmul.
//     - multiplier and shift combined gives the scale.
//     - assumes input zero point is 0.
//     - scratch is created for optimization purpose only.
//       TODO(b/152066492): this can be removed if some future optimization
//       work makes it unnecessary.
void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int8_t* output, CpuBackendContext* context);

// Same as the above 8, 8, 8 integer matmul except for the presence of zero
// point and non-accumulative.
// TODO(b/148688698): remove this function by folding zero point calculation in
// prepare() function.
void MatrixBatchVectorMultiply(const int8_t* input, int32_t input_zeropoint,
                               const int8_t* input_to_gate_weights,
                               int32_t input_to_gate_effective_scale_a,
                               int32_t input_to_gate_effective_scale_b,
                               int32_t n_batch, int32_t n_input, int32_t n_cell,
                               int8_t* gate_output, int8_t gate_output_zp);

// Same as above but has 16 bit and 8 bit input and 8 bit output.
// Used in projection when hidden is 16bit.
void MatrixBatchVectorMultiply(const int16_t* hidden,
                               const int8_t* hidden_to_output_weights,
                               int32_t proj_effective_scale_a,
                               int32_t proj_effective_scale_b,
                               const int32_t* gate_bias, int32_t n_batch,
                               int32_t n_hidden, int32_t n_output,
                               int32_t output_zp, int8_t* proj_output);

// Multiplies a matrix with a scalar and reduce the result on each row to a
// scalar.
// Parameters:
//     - matrix: matrix of size n_row * n_col
//     - scalar: the scalar that is multiplied to each element in the matrix
//     - n_row:  the row count of the matrix
//     - n_col:  the column count of the matrix
//     - output: the 32bit output
// Note: We do not need saturation because the int8 * int8 is safe from overflow
// in (2^31-1) / (2^14) = 131072, which is bigger than the n_row. Non-zero
// initial output value is not exceptionally large.
void MatrixScalarMultiplyAccumulate(const int8_t* matrix, int32_t scalar,
                                    int32_t n_row, int32_t n_col,
                                    int32_t* output);

// Apply Layer Normalization (https://arxiv.org/abs/1607.06450) to a Quantized
// vector.
// Parameters:
//     - input: batch vector of size n_batch * n_input; 16 bit.
//     - layer_norm_weights:  the quantized layer normalization weights.
//     - bias: the bias for the layer normalization.
//     - layer_norm_scale_a: multiplier for scale factor.
//     - layer_norm_scale_b: shift for scale factor.
//     - variance_limit: the guard to make sure the inverse does not overflow.
//     - n_batch: the number of batches.
//     - n_input: the size for input and output.
//     - output:  the 16 bit output
void ApplyLayerNorm(const int16_t* input, const int16_t* layer_norm_weights,
                    const int32_t* bias, int32_t layer_norm_scale_a,
                    int32_t layer_norm_scale_b, int32_t variance_limit,
                    int n_batch, int n_input, int16_t* output);

// Same as above but the internal calculation is done in float.
void ApplyLayerNormFloat(const int16_t* input,
                         const int16_t* layer_norm_weights,
                         int32_t layer_norm_scale_a, int32_t layer_norm_scale_b,
                         const int32_t* bias, int n_batch, int n_input,
                         int16_t* output);

// Apply Sigmoid to a quantized vector.
// Parameters:
//     - input: batch vector of size n_batch * n_input; 16 bit.
//     - n_batch: the number of batches.
//     - n_input: the size for input and output.
//     - output:  the 16 bit output
// The input is in Q3.12 format and the output is in Q0.15 format.
void ApplySigmoid(const int16_t* input, int32_t n_batch, int32_t n_input,
                  int16_t* output);

// Same as above but the internal calcualtion is float.
void ApplySigmoidFloat(const int16_t* input, int32_t n_batch, int32_t n_input,
                       int16_t* output);

// Apply Tanh to a quantized vector.
// Parameters:
//     - integer_bits: the integer bits of the input.
//                     Currently supports 0, 1, 2, 3, 4, 5, 6.
//     - input: batch vector of size n_batch * n_input; 16 bit.
//     - n_batch: the number of batches.
//     - n_input: the size for input and output.
//     - output:  the 16 bit output
// The input is in Qm.15-m format and the output is in Q0.15 format.
void ApplyTanh(int32_t integer_bits, const int16_t* input, int32_t n_batch,
               int32_t n_input, int16_t* output);

// Apply Tanh to a quantized vector. Tbe internal calculation is in float.
//    - Input has 2^(integer_bits) as scale.
//    - Output has Q0.15 as scale.
void ApplyTanhFloat(const int16_t* input, int32_t n_batch, int32_t n_input,
                    int32_t integer_bits, int16_t* output);

// Element-wise multiplication of two quantized vectors.
// Parameters:
//     - input_1: batch vector of size n_batch * n_input; 16 bit.
//     - input_2: batch vector of size n_batch * n_input; 16 bit.
//     - n_batch: the number of batches.
//     - n_input: the size for input and output.
//     - shift:   the shift needed to produce the output.
//     - output:  the 16 bit output of size n_batch * n_input.
// Output does not need to be initialized.
void CwiseMul(const int16_t* input_1, const int16_t* input_2, int n_batch,
              int n_input, int shift, int16_t* output);

// Element-wise multiplication of two quantized vectors.
// Parameters:
//     - input_1: batch vector of size n_batch * n_input; 16 bit.
//     - input_2: batch vector of size n_batch * n_input; 16 bit.
//     - n_batch: the number of batches.
//     - n_input: the size for input and output.
//     - shift:   the shift needed to produce the output.
//     - output:  the 8 bit output of size n_batch * n_input.
// Output does not need to be initialized.
void CwiseMul(const int16_t* input_1, const int16_t* input_2, int n_batch,
              int n_input, int shift, int8_t* output);

// Element-wise multiplication of two quantized vectors with rescaling.
// Parameters:
//     - input_1:    batch vector of size n_batch * n_input; 16 bit.
//     - input_2:    batch vector of size n_batch * n_input; 16 bit.
//     - multiplier: the multiplier part of scale.
//     - shift:      the shift part of scale.
//     - n_batch:    the number of batches.
//     - n_input:    the size for input and output.
//     - output:     the 8 bit output of size n_batch * n_input.
//     - output_zp:  the zero point of output.
// Output does not need to be initialized.
// Multiplier ("m") and shift ("s") are connected to scale ("s") with s = m *
// 2^(s - 31).
void CwiseMul(const int16_t* input_1, const int16_t* input_2,
              int32_t multiplier, int32_t shift, int32_t n_batch,
              int32_t n_input, int32_t output_zp, int8_t* output);

// Element-wise saturating addition of two quantized vectors without rescaling.
// Parameters:
//     - input_1:    batch vector of size n_batch * n_input; 16 bit.
//     - input_2:    batch vector of size n_batch * n_input; 16 bit.
//     - n_batch:    the number of batches.
//     - n_input:    the size for input and output.
//     - output:     the 8 bit output of size n_batch * n_input.
// Output does not need to be initialized.
void CwiseAdd(const int16_t* input_1, const int16_t* input_2, int n_batch,
              int n_input, int16_t* output);

// Element-wise in-place clipping of a quantized vector.
// Parameters:
//     - input:          batch vector of size n_batch * n_input; 16 bit.
//     - clipping_value: the value used for clipping.
//     - n_batch:        the number of batches.
//     - n_input:        the size for input and output.
void CwiseClipping(int16_t* input, const int16_t clipping_value,
                   int32_t n_batch, int32_t n_input);

// Element-wise in-place clipping of a quantized vector.
// Parameters:
//     - input:          batch vector of size n_batch * n_input; 8 bit.
//     - clipping_value: the value used for clipping.
//     - n_batch:        the number of batches.
//     - n_input:        the size for input and output.
void CwiseClipping(int8_t* input, const int8_t clipping_value, int32_t n_batch,
                   int32_t n_input);

// Cwise product of two vectors.
template <typename T>
inline void VectorVectorCwiseProduct(const T* __restrict__ vector1,
                                     const T* __restrict__ vector2, int v_size,
                                     T* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    *result++ = *vector1++ * *vector2++;
  }
}

// Cwise product and accumulate of two vectors. Since it's a MAC operation, the
// assumption here is that result array is initialized to valid values.
template <typename T>
inline void VectorVectorCwiseProductAccumulate(const T* __restrict__ vector1,
                                               const T* __restrict__ vector2,
                                               int v_size,
                                               T* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    *result++ += *vector1++ * *vector2++;
  }
}

// Dot product of two vectors.
float VectorVectorDotProduct(const float* vector1, const float* vector2,
                             int v_size);

// Dot product of two batch vectors of size n_batch * v_size:
// vector1 = [x_1_1, x_1_2, ..., x_1_vsize,
//            x_2_1, x_2_2, ..., x_2_vsize,
//            ...
//            x_nbatch_1,..., x_nbatch_vsize]
// vector2 = [y_1_1, y_1_2, ..., y_1_vsize,
//            y_2_1, y_2_2, ..., y_2_vsize,
//            ...
//            y_nbatch_1,..., y_nbatch_vsize]
// Then result will be a vector of n_batch size starting from 'result':
// [x_1_1 * y_1_1 + x_1_2 * y_1_2 + ... + x_1_vsize * y_1_vsize,
//  x_2_1 * y_2_1 + x_2_2 * y_2_2 + ... + x_2_vsize * y_2_vsize,
//  ...
//  x_nbatch_1 * y_nbatch_1 + ... + x_nbatch_vsize * y_nbatch_vsize]
template <typename T>
inline void BatchVectorBatchVectorDotProduct(const T* vector1, const T* vector2,
                                             int v_size, int n_batch,
                                             T* result) {
  for (int b = 0; b < n_batch; b++) {
    result[b] = VectorVectorDotProduct(vector1, vector2, v_size);
    vector1 += v_size;
    vector2 += v_size;
  }
}

// Same as above but input is 16bit and output is 32bit.
void BatchVectorBatchVectorDotProduct(const int16_t* vector1,
                                      const int16_t* vector2, int v_size,
                                      int n_batch, int32_t* result);

// Cwise product of a vector and a batch-vector.
template <typename T>
inline void VectorBatchVectorCwiseProduct(const T* vector, int v_size,
                                          const T* batch_vector, int n_batch,
                                          T* result) {
  for (int b = 0; b < n_batch; b++) {
    VectorVectorCwiseProduct(vector, batch_vector, v_size, result);
    // Update the pointers.
    result += v_size;
    batch_vector += v_size;
  }
}

// Cwise product and accumulate of a vector and a batch-vector. Since it's a MAC
// operation, the assumption here is that result array is initialized to valid
// values.
template <typename T>
inline void VectorBatchVectorCwiseProductAccumulate(const T* vector, int v_size,
                                                    const T* batch_vector,
                                                    int n_batch, T* result) {
  for (int b = 0; b < n_batch; b++) {
    VectorVectorCwiseProductAccumulate(vector, batch_vector, v_size, result);
    // Update the pointers.
    result += v_size;
    batch_vector += v_size;
  }
}

// Same as above, but inputs are 16bit integer and output is 16bit integer.
void VectorBatchVectorCwiseProductAccumulate(const int16_t* vector, int v_size,
                                             const int16_t* batch_vector,
                                             int n_batch, int32_t multiplier,
                                             int shift, int16_t* result);

// Add another vector for each batch in the batch vector.
void VectorBatchVectorAdd(const float* vector, int v_size, int n_batch,
                          float* batch_vector);

// Batch vector initialization with another vector.
template <typename T>
void VectorBatchVectorAssign(const T* vector, int v_size, int n_batch,
                             T* batch_vector) {
  for (int b = 0; b < n_batch; b++) {
    std::copy_n(vector, v_size, batch_vector + b * v_size);
  }
}

// Apply Rectified Linear to elements of a vector.
inline void ApplyReluToVector(const float* __restrict__ vector, int v_size,
                              float* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    result[v] = std::max(0.0f, vector[v]);
  }
}

// Apply Rectified Linear 1 (cap to [-1;1]) to elements of a vector
inline void ApplyRelu1ToVector(const float* __restrict__ vector, int v_size,
                               float* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    result[v] = std::max(-1.0f, std::min(vector[v], 1.0f));
  }
}

// Apply Rectified Linear 6 (cap to [0;6]) to elements of a vector
inline void ApplyRelu6ToVector(const float* __restrict__ vector, int v_size,
                               float* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    result[v] = std::max(0.0f, std::min(vector[v], 6.0f));
  }
}

// Apply tanh to elements of a vector
inline void ApplyTanhToVector(const float* __restrict__ vector, int v_size,
                              float* __restrict__ result) {
  using VectorMap = Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>;
  VectorMap input_map(const_cast<float* __restrict__>(vector), v_size);
  VectorMap output_map(result, v_size);
  output_map.array() = input_map.array().tanh();
}

// Apply signbit to elements of a vector
inline void ApplySignbitToVector(const float* __restrict__ vector, int v_size,
                                 float* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    result[v] = std::signbit(vector[v]);
  }
}

// Apply sigmoid to elements of a vector.
inline void ApplySigmoidToVector(const float* __restrict__ vector, int v_size,
                                 float* __restrict__ result) {
  using VectorMap = Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>;
  VectorMap input_map(const_cast<float* __restrict__>(vector), v_size);
  VectorMap output_map(result, v_size);
  output_map.array() = input_map.array().logistic();
}

// Apply appropriate activation function to elements of a vector.
inline void ApplyActivationToVector(const float* __restrict__ vector,
                                    int v_size,
                                    TfLiteFusedActivation activation,
                                    float* __restrict__ result) {
  switch (activation) {
    case kTfLiteActNone:
      return;
    case kTfLiteActRelu:
      return ApplyReluToVector(vector, v_size, result);
    case kTfLiteActRelu1:
      return ApplyRelu1ToVector(vector, v_size, result);
    case kTfLiteActRelu6:
      return ApplyRelu6ToVector(vector, v_size, result);
    case kTfLiteActTanh:
      return ApplyTanhToVector(vector, v_size, result);
    case kTfLiteActSignBit:
      return ApplySignbitToVector(vector, v_size, result);
    case kTfLiteActSigmoid:
      return ApplySigmoidToVector(vector, v_size, result);
  }
}

// Compute "1.0f - elements of vector" (used in CIFG).
void Sub1Vector(const float* vector, int v_size, float* result);

// Compute "1.0f - elements of vector" (used in CIFG) for int16 input.
// "vector" has range [0, 32767] because it is the output of sigmoid function.
void Sub1Vector(const int16_t* vector, int v_size, int16_t* result);

// Multiply all elements of vector with a scalar.
void VectorScalarMultiply(const int8_t* vector, int v_size, float scale,
                          float* result);

// Clip elements of a vector using a abs_limit value.
void ClipVector(const float* vector, int v_size, float abs_limit,
                float* result);

// Reduce-sum on a float input vector:
// input_vector: float pointer to input vector.
// output_vector: float pointer to vector.
// output_size: output vector size.
// reduction_size: number of consecutive elements from input vector which are
// added to get one element of output.
void ReductionSumVector(const float* input_vector, float* output_vector,
                        int output_size, int reduction_size);

// Same as above but input/output is 32 bit integer.
void ReductionSumVector(const int32_t* input_vector, int32_t* output_vector,
                        int output_size, int reduction_size);

// Same as above but input is 8 bit integer.
void ReductionSumVector(const int8_t* input_vector, int32_t* output_vector,
                        int output_size, int reduction_size);

// Layer norm for each batch.
void MeanStddevNormalization(const float* input_vector, float* output_vector,
                             int v_size, int n_batch);

// Saturate Add with rescale on both inputs.
void TwoGateSaturationgAdd(const int8_t* input, int8_t input_zp,
                           const int8_t* recurrent, int8_t recurrent_zp,
                           int32_t input_effective_scale_a,
                           int32_t input_effective_scale_b,
                           int32_t recurrent_effective_scale_a,
                           int32_t recurrent_effective_scale_b, int32_t n_batch,
                           int32_t n_cell, int16_t* output);

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_

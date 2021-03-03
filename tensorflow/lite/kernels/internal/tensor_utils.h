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
#include <cstdint>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/tensor_utils_common.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {

// Not all backends support CpuBackendContext usage, so forward declare to avoid
// pulling in its implementation. Use of CpuBackendContext in method
// implementations is purely optional.
class CpuBackendContext;

namespace tensor_utils {

// Same as the function above, but provide a scratch buffer for the
// int8 x int8 -> int32 and a CpuBackendContext for the accumulator
// computation.
void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    int32_t* __restrict__ scratch, float* __restrict__ result,
    CpuBackendContext* __restrict__ context);

// Same as the function above except that can make use of cached row sums.
void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset, int32_t* scratch, int32_t* row_sums,
    bool* compute_row_sums, CpuBackendContext* context);

// Same as the function above, but provides separate scaling factor for the
// matrix and the vectors. The scaling factors are multiplied in the
// scaling_factor_scratch buffer.
inline void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float matrix_scaling_factor,
    const float* vector_scaling_factors, int n_batch,
    float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset, int32_t* scratch, int32_t* row_sums,
    bool* compute_row_sums, float* scaling_factor_scratch,
    CpuBackendContext* context) {
  for (int b = 0; b < n_batch; ++b) {
    scaling_factor_scratch[b] =
        vector_scaling_factors[b] * matrix_scaling_factor;
  }
  MatrixBatchVectorMultiplyAccumulate(matrix, m_rows, m_cols, vectors,
                                      scaling_factor_scratch, n_batch, result,
                                      per_channel_scale, input_offset, scratch,
                                      row_sums, compute_row_sums, context);
}

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
// TODO(b/152066492): this can be removed if some future optimization
// work makes it unnecessary.
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
// TODO(b/152066492): this can be removed if some future optimization
// work makes it unnecessary.
void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int8_t* output, CpuBackendContext* context);

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
    case kTfLiteActReluN1To1:
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

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_

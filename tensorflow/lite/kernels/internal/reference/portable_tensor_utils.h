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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PORTABLE_TENSOR_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PORTABLE_TENSOR_UTILS_H_

#include "tensorflow/lite/kernels/internal/reference/portable_tensor_utils_impl.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {
namespace tensor_utils {

// Check if all entries of a vector are zero for float.
bool IsZeroVector(const float* vector, int v_size) {
  return PortableIsZeroVector(vector, v_size);
}

// Check if all entries of a vector are zero for int8_t.
bool IsZeroVector(const int8_t* vector, int v_size) {
  return PortableIsZeroVector(vector, v_size);
}

void SymmetricQuantizeFloats(const float* values, const int size,
                             int8_t* quantized_values, float* min, float* max,
                             float* scaling_factor) {
  PortableSymmetricQuantizeFloats(values, size, quantized_values, min, max,
                                  scaling_factor);
}

void SymmetricQuantizeFloats(const float* values, const int size,
                             int8_t* quantized_values, float min_value,
                             float max_value, float* scaling_factor) {
  PortableSymmetricQuantizeFloats(values, size, quantized_values, min_value,
                                  max_value, scaling_factor);
}

void AsymmetricQuantizeFloats(const float* values, const int size,
                              int8_t* quantized_values, float* scaling_factor,
                              int32_t* offset) {
  PortableAsymmetricQuantizeFloats(values, size, quantized_values,
                                   scaling_factor, offset);
}

void MatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                         int m_cols, const float* vector,
                                         int n_batch, float* result) {
  PortableMatrixBatchVectorMultiplyAccumulate(matrix, m_rows, m_cols, vector,
                                              n_batch, result);
}

void MatrixBatchVectorMultiplyAccumulate(const int8_t* __restrict__ matrix,
                                         const int m_rows, const int m_cols,
                                         const int8_t* __restrict__ vector,
                                         const float* scaling_factors,
                                         int n_batch,
                                         float* __restrict__ result) {
  PortableMatrixBatchVectorMultiplyAccumulate(matrix, m_rows, m_cols, vector,
                                              scaling_factors, n_batch, result);
}

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset, int32_t* scratch, int32_t* row_sums,
    bool* compute_row_sums, CpuBackendContext* context) {
  PortableMatrixBatchVectorMultiplyAccumulate(
      matrix, m_rows, m_cols, vectors, scaling_factors, n_batch, result,
      per_channel_scale, input_offset, scratch, row_sums, compute_row_sums,
      context);
}

void MatrixBatchVectorMultiplyAccumulate(const int8_t* __restrict__ matrix,
                                         const int m_rows, const int m_cols,
                                         const int8_t* __restrict__ vector,
                                         const float* scaling_factors,
                                         int n_batch, int32_t* scratch,
                                         float* __restrict__ result,
                                         CpuBackendContext* context) {
  PortableMatrixBatchVectorMultiplyAccumulate(matrix, m_rows, m_cols, vector,
                                              scaling_factors, n_batch, result);
}

void SparseMatrixBatchVectorMultiplyAccumulate1x4(
    const float* __restrict__ matrix, const int32_t* __restrict__ segments,
    const int32_t* __restrict__ indices, int m_rows, int m_cols,
    const float* __restrict__ vector, int n_batch, float* __restrict__ result) {
  PortableSparseMatrixBatchVectorMultiplyAccumulate1x4(
      matrix, segments, indices, m_rows, m_cols, vector, n_batch, result);
}

void SparseMatrixBatchVectorMultiplyAccumulate(
    const float* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    int m_rows, int m_cols, const float* __restrict__ vector, int n_batch,
    float* __restrict__ result) {
  PortableSparseMatrixBatchVectorMultiplyAccumulate(
      matrix, ledger, m_rows, m_cols, vector, n_batch, result);
}

void SparseMatrixBatchVectorMultiplyAccumulate1x16(
    const int8_t* __restrict__ matrix, const int32_t* __restrict__ segments,
    const int32_t* __restrict__ indices, int m_rows, int m_cols,
    const int8_t* __restrict__ vector, const int32_t* __restrict__ bias_vector,
    int n_batch, const int32_t input_offset, const int32_t output_multiplier,
    const int32_t output_shift, const int32_t* per_channel_scale,
    const int32_t* per_channel_shift, const int32_t output_offset,
    const int32_t output_activation_min, const int32_t output_activation_max,

    int8_t* __restrict__ result) {
  PortableSparseMatrixBatchVectorMultiplyAccumulate1x16(
      matrix, segments, indices, m_rows, m_cols, vector, bias_vector, n_batch,
      input_offset, output_multiplier, output_shift, per_channel_scale,
      per_channel_shift, output_offset, output_activation_min,
      output_activation_max, result);
}

void SparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result,
    const float* per_channel_scale) {
  PortableSparseMatrixBatchVectorMultiplyAccumulate(
      matrix, ledger, m_rows, m_cols, vectors, scaling_factors, n_batch, result,
      per_channel_scale);
}

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int16_t* output, CpuBackendContext* context) {
  PortableMatrixBatchVectorMultiplyAccumulate(
      input, bias, input_to_gate_weights, multiplier, shift, n_batch, n_input,
      n_output, output_zp, scratch, output, context);
}

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int8_t* output, CpuBackendContext* context) {
  PortableMatrixBatchVectorMultiplyAccumulate(
      input, bias, input_to_gate_weights, multiplier, shift, n_batch, n_input,
      n_output, output_zp, scratch, output, context);
}

void MatrixScalarMultiplyAccumulate(const int8_t* matrix, int32_t scalar,
                                    int32_t n_row, int32_t n_col,
                                    int32_t* output) {
  PortableMatrixScalarMultiplyAccumulate(matrix, scalar, n_row, n_col, output);
}

void MatrixBatchVectorMultiply(const int8_t* input, int32_t input_zeropoint,
                               const int8_t* input_to_gate_weights,
                               int32_t input_to_gate_effective_scale_a,
                               int32_t input_to_gate_effective_scale_b,
                               int32_t n_batch, int32_t n_input, int32_t n_cell,
                               int8_t* gate_output, int8_t gate_output_zp) {
  PortableMatrixBatchVectorMultiply(
      input, input_zeropoint, input_to_gate_weights,
      input_to_gate_effective_scale_a, input_to_gate_effective_scale_b, n_batch,
      n_input, n_cell, gate_output, gate_output_zp);
}

void MatrixBatchVectorMultiply(const int16_t* hidden,
                               const int8_t* hidden_to_output_weights,
                               int32_t proj_effective_scale_a,
                               int32_t proj_effective_scale_b,
                               const int32_t* gate_bias, int32_t n_batch,
                               int32_t n_hidden, int32_t n_output,
                               int32_t output_zp, int8_t* proj_output) {
  PortableMatrixBatchVectorMultiply(hidden, hidden_to_output_weights,
                                    proj_effective_scale_a,
                                    proj_effective_scale_b, gate_bias, n_batch,
                                    n_hidden, n_output, output_zp, proj_output);
}

void ApplyLayerNorm(const int16_t* input, const int16_t* layer_norm_weights,
                    const int32_t* bias, int32_t layer_norm_scale_a,
                    int32_t layer_norm_scale_b, int32_t variance_limit,
                    int n_batch, int n_input, int16_t* output) {
  PortableApplyLayerNorm(input, layer_norm_weights, bias, layer_norm_scale_a,
                         layer_norm_scale_b, variance_limit, n_batch, n_input,
                         output);
}

void ApplyLayerNormFloat(const int16_t* input,
                         const int16_t* layer_norm_weights,
                         int32_t layer_norm_scale_a, int32_t layer_norm_scale_b,
                         const int32_t* bias, int n_batch, int n_input,
                         int16_t* output) {
  PortableApplyLayerNormFloat(input, layer_norm_weights, layer_norm_scale_a,
                              layer_norm_scale_b, bias, n_batch, n_input,
                              output);
}

void ApplySigmoid(const int16_t* input, int32_t n_batch, int32_t n_input,
                  int16_t* output) {
  PortableApplySigmoid(input, n_batch, n_input, output);
}

void ApplySigmoidFloat(const int16_t* input, int32_t n_batch, int32_t n_input,
                       int16_t* output) {
  PortableApplySigmoidFloat(input, n_batch, n_input, output);
}

void ApplyTanh(int32_t integer_bits, const int16_t* input, int32_t n_batch,
               int32_t n_input, int16_t* output) {
  PortableApplyTanh(integer_bits, input, n_batch, n_input, output);
}

void ApplyTanhFloat(const int16_t* input, int32_t n_batch, int32_t n_input,
                    int32_t integer_bits, int16_t* output) {
  PortableApplyTanhFloat(input, n_batch, n_input, integer_bits, output);
}

void CwiseMul(const int16_t* input_1, const int16_t* input_2, int n_batch,
              int n_input, int shift, int16_t* output) {
  PortableCwiseMul(input_1, input_2, n_batch, n_input, shift, output);
}

void CwiseMul(const int16_t* input_1, const int16_t* input_2,
              int32_t multiplier, int32_t shift, int32_t n_batch,
              int32_t n_input, int32_t output_zp, int8_t* output) {
  PortableCwiseMul(input_1, input_2, multiplier, shift, n_batch, n_input,
                   output_zp, output);
}

void CwiseAdd(const int16_t* input_1, const int16_t* input_2, int n_batch,
              int n_input, int16_t* output) {
  PortableCwiseAdd(input_1, input_2, n_batch, n_input, output);
}

void CwiseClipping(float* vector, const int v_size,
                   const float clipping_value) {
  PortableCwiseClipping(vector, v_size, clipping_value);
}

void CwiseClipping(int16_t* vector, const int v_size,
                   const int16_t clipping_value) {
  PortableCwiseClipping(vector, v_size, clipping_value);
}

void CwiseClipping(int8_t* vector, const int v_size,
                   const int8_t clipping_value) {
  PortableCwiseClipping(vector, v_size, clipping_value);
}

void VectorBatchVectorCwiseProductAccumulate(const int16_t* vector, int v_size,
                                             const int16_t* batch_vector,
                                             int n_batch, int32_t multiplier,
                                             int shift, int16_t* result) {
  PortableVectorBatchVectorCwiseProductAccumulate(
      vector, v_size, batch_vector, n_batch, multiplier, shift, result);
}

float VectorVectorDotProduct(const float* vector1, const float* vector2,
                             int v_size) {
  return PortableVectorVectorDotProduct(vector1, vector2, v_size);
}

void BatchVectorBatchVectorDotProduct(const int16_t* vector1,
                                      const int16_t* vector2, int v_size,
                                      int n_batch, int32_t* result) {
  PortableBatchVectorBatchVectorDotProduct(vector1, vector2, v_size, n_batch,
                                           result);
}

void Sub1Vector(const float* vector, int v_size, float* result) {
  PortableSub1Vector(vector, v_size, result);
}

void Sub1Vector(const int16_t* vector, int v_size, int16_t* result) {
  PortableSub1Vector(vector, v_size, result);
}

// Multiply all elements of vector with a scalar.
void VectorScalarMultiply(const int8_t* vector, int v_size, float scale,
                          float* result) {
  PortableVectorScalarMultiply(vector, v_size, scale, result);
}

void ReductionSumVector(const float* input_vector, float* output_vector,
                        int output_size, int reduction_size) {
  PortableReductionSumVector(input_vector, output_vector, output_size,
                             reduction_size);
}

void ReductionSumVector(const int32_t* input_vector, int32_t* output_vector,
                        int output_size, int reduction_size) {
  PortableReductionSumVector(input_vector, output_vector, output_size,
                             reduction_size);
}

void ReductionSumVector(const int8_t* input_vector, int32_t* output_vector,
                        int output_size, int reduction_size) {
  PortableReductionSumVector(input_vector, output_vector, output_size,
                             reduction_size);
}

void MeanStddevNormalization(const float* input_vector, float* output_vector,
                             int v_size, int n_batch) {
  PortableMeanStddevNormalization(input_vector, output_vector, v_size, n_batch);
}

void TwoGateSaturatingAdd(const int8_t* input, int8_t input_zp,
                          const int8_t* recurrent, int8_t recurrent_zp,
                          int32_t input_effective_scale_a,
                          int32_t input_effective_scale_b,
                          int32_t recurrent_effective_scale_a,
                          int32_t recurrent_effective_scale_b, int32_t n_batch,
                          int32_t n_cell, int16_t* output) {
  PortableTwoGateSaturatingAdd(
      input, input_zp, recurrent, recurrent_zp, input_effective_scale_a,
      input_effective_scale_b, recurrent_effective_scale_a,
      recurrent_effective_scale_b, n_batch, n_cell, output);
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PORTABLE_TENSOR_UTILS_H_

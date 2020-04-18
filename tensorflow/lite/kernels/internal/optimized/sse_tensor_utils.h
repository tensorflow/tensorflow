/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SSE_TENSOR_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SSE_TENSOR_UTILS_H_

// Note: This file is a copy-paste version of neon_tensor_utils.h, only
// difference is in MatrixBatchVectorMultiplyAccumulate and
// SparseMatrixBatchVectorMultiplyAccumulate (other functions do not have SSE
// implementation yet).

// Note: Most of the functions below use NEON_OR_PORTABLE, through the Intel
// NEON_2_SSE translator library. If a native SSE version of a function is
// implemented, replace the appropriate one to SSE_OR_PORTABLE.

// TODO(ghodrat): Remove this header file and the dependency to internal data
// structure.
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_tensor_utils_impl.h"
#include "tensorflow/lite/kernels/internal/optimized/sse_check.h"
#include "tensorflow/lite/kernels/internal/optimized/sse_tensor_utils_impl.h"
#include "tensorflow/lite/kernels/internal/reference/portable_tensor_utils_impl.h"

namespace tflite {
namespace tensor_utils {

void MatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                         int m_cols, const float* vector,
                                         int n_batch, float* result) {
  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols,
                   vector, n_batch, result);
}

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    float* __restrict__ result) {
  SSE_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols,
                  vectors, scaling_factors, n_batch, result);
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

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    int32_t* __restrict__ scratch, float* __restrict__ result,
    CpuBackendContext* __restrict__ context) {
  SSE_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols,
                  vectors, scaling_factors, n_batch, result);
}

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    float* __restrict__ result, const float* __restrict__ per_channel_scale,
    const int32_t* __restrict__ input_offset) {
  SSE_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols,
                  vectors, scaling_factors, n_batch, result, per_channel_scale,
                  input_offset);
}

void SparseMatrixBatchVectorMultiplyAccumulate(
    const float* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    int m_rows, int m_cols, const float* __restrict__ vector, int n_batch,
    float* __restrict__ result) {
  NEON_OR_PORTABLE(SparseMatrixBatchVectorMultiplyAccumulate, matrix, ledger,
                   m_rows, m_cols, vector, n_batch, result);
}

void SparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    const int m_rows, const int m_cols, const int8_t* __restrict__ vectors,
    const float* __restrict__ scaling_factors, int n_batch,
    float* __restrict__ result) {
  SSE_OR_PORTABLE(SparseMatrixBatchVectorMultiplyAccumulate, matrix, ledger,
                  m_rows, m_cols, vectors, scaling_factors, n_batch, result);
}

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* input_zeropoint_times_weights,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int16_t* output, CpuBackendContext* context) {
  PortableMatrixBatchVectorMultiplyAccumulate(
      input, input_zeropoint_times_weights, input_to_gate_weights, multiplier,
      shift, n_batch, n_input, n_output, output_zp, scratch, output, context);
}

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* input_zeropoint_times_weights,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int8_t* output, CpuBackendContext* context) {
  PortableMatrixBatchVectorMultiplyAccumulate(
      input, input_zeropoint_times_weights, input_to_gate_weights, multiplier,
      shift, n_batch, n_input, n_output, output_zp, scratch, output, context);
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

void MatrixScalarMultiplyAccumulate(const int8_t* matrix, int32_t scalar,
                                    int32_t n_row, int32_t n_col,
                                    int32_t* output) {
  PortableMatrixScalarMultiplyAccumulate(matrix, scalar, n_row, n_col, output);
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

void ApplyTanh(int32_t intger_bits, const int16_t* input, int32_t n_batch,
               int32_t n_input, int16_t* output) {
  PortableApplyTanh(intger_bits, input, n_batch, n_input, output);
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

void CwiseClipping(int16_t* input, const int16_t clipping_value,
                   int32_t n_batch, int32_t n_input) {
  PortableCwiseClipping(input, clipping_value, n_batch, n_input);
}

void CwiseClipping(int8_t* input, const int8_t clipping_value, int32_t n_batch,
                   int32_t n_input) {
  PortableCwiseClipping(input, clipping_value, n_batch, n_input);
}

void BatchVectorBatchVectorDotProduct(const int16_t* vector1,
                                      const int16_t* vector2, int v_size,
                                      int n_batch, int32_t* result) {
  PortableBatchVectorBatchVectorDotProduct(vector1, vector2, v_size, n_batch,
                                           result);
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
  return NEON_OR_PORTABLE(VectorVectorDotProduct, vector1, vector2, v_size);
}

void VectorBatchVectorAdd(const float* vector, int v_size, int n_batch,
                          float* batch_vector) {
  PortableVectorBatchVectorAdd(vector, v_size, n_batch, batch_vector);
}

void Sub1Vector(const float* vector, int v_size, float* result) {
  NEON_OR_PORTABLE(Sub1Vector, vector, v_size, result);
}

void Sub1Vector(const int16_t* vector, int v_size, int16_t* result) {
  PortableSub1Vector(vector, v_size, result);
}

// Check if all entries of a vector are zero for float.
bool IsZeroVector(const float* vector, int v_size) {
  return NEON_OR_PORTABLE(IsZeroVector, vector, v_size);
}

// Check if all entries of a vector are zero for int8.
bool IsZeroVector(const int8_t* vector, int v_size) {
  return PortableIsZeroVector(vector, v_size);
}

void VectorScalarMultiply(const int8_t* vector, int v_size, float scale,
                          float* result) {
  NEON_OR_PORTABLE(VectorScalarMultiply, vector, v_size, scale, result);
}
void ClipVector(const float* vector, int v_size, float abs_limit,
                float* result) {
  NEON_OR_PORTABLE(ClipVector, vector, v_size, abs_limit, result);
}

void SymmetricQuantizeFloats(const float* values, const int size,
                             int8_t* quantized_values, float* min_value,
                             float* max_value, float* scaling_factor) {
  NEON_OR_PORTABLE(SymmetricQuantizeFloats, values, size, quantized_values,
                   min_value, max_value, scaling_factor);
}

void SymmetricQuantizeFloats(const float* values, const int size,
                             int8_t* quantized_values, float min_value,
                             float max_value, float* scaling_factor) {
  NEON_OR_PORTABLE(SymmetricQuantizeFloats, values, size, quantized_values,
                   min_value, max_value, scaling_factor);
}

void AsymmetricQuantizeFloats(const float* values, const int size,
                              int8_t* quantized_values, float* scaling_factor,
                              int32_t* offset) {
  NEON_OR_PORTABLE(AsymmetricQuantizeFloats, values, size, quantized_values,
                   scaling_factor, offset);
}

void ReductionSumVector(const float* input_vector, float* output_vector,
                        int output_size, int reduction_size) {
  NEON_OR_PORTABLE(ReductionSumVector, input_vector, output_vector, output_size,
                   reduction_size);
}

void ReductionSumVector(const int32_t* input_vector, int32_t* output_vector,
                        int output_size, int reduction_size) {
  PortableReductionSumVector(input_vector, output_vector, output_size,
                             reduction_size);
}

void ReductionSumVector(const int8_t* input_vector, int32_t* output_vector,
                        int output_size, int reduction_size) {
  NEON_OR_PORTABLE(ReductionSumVector, input_vector, output_vector, output_size,
                   reduction_size);
}

void MeanStddevNormalization(const float* input_vector, float* output_vector,
                             int v_size, int n_batch) {
  PortableMeanStddevNormalization(input_vector, output_vector, v_size, n_batch);
}

void TwoGateSaturationgAdd(const int8_t* input, int8_t input_zp,
                           const int8_t* recurrent, int8_t recurrent_zp,
                           int32_t input_effective_scale_a,
                           int32_t input_effective_scale_b,
                           int32_t recurrent_effective_scale_a,
                           int32_t recurrent_effective_scale_b, int32_t n_batch,
                           int32_t n_cell, int16_t* output) {
  PortableTwoGateSaturationgAdd(
      input, input_zp, recurrent, recurrent_zp, input_effective_scale_a,
      input_effective_scale_b, recurrent_effective_scale_a,
      recurrent_effective_scale_b, n_batch, n_cell, output);
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SSE_TENSOR_UTILS_H_

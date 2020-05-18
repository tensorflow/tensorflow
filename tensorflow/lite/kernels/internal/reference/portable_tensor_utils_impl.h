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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PORTABLE_TENSOR_UTILS_IMPL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PORTABLE_TENSOR_UTILS_IMPL_H_

#include <cstdint>

// TODO(ghodrat): Remove this header file and the dependency to internal data
// structure.
#include "tensorflow/lite/c/builtin_op_data.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {

// Not all backends support CpuBackendContext usage, so forward declare to avoid
// pulling in its implementation.
class CpuBackendContext;

namespace tensor_utils {

// Limit a float input f between +abs_limit and -abs_limit.
float PortableClip(float f, float abs_limit);

template <typename T>
bool PortableIsZeroVector(const T* vector, int v_size) {
  for (int i = 0; i < v_size; ++i) {
    if (vector[i] != 0) {
      return false;
    }
  }
  return true;
}

void PortableSymmetricQuantizeFloats(const float* values, const int size,
                                     int8_t* quantized_values, float* min_value,
                                     float* max_value, float* scaling_factor);

void PortableSymmetricQuantizeFloats(const float* values, const int size,
                                     int8_t* quantized_values, float min_value,
                                     float max_value, float* scaling_factor);

void PortableAsymmetricQuantizeFloats(const float* values, const int size,
                                      int8_t* quantized_values,
                                      float* scaling_factor, int32_t* offset);

// Multiply a matrix by a batch vector, and store results in a batch-size
// vector.
void PortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result);

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result);

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset, int32_t* scratch, int32_t* row_sums,
    bool* compute_row_sums, CpuBackendContext* context);

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vector, const float* scaling_factors,
    int n_batch, int32_t* scratch, float* __restrict__ result,
    CpuBackendContext* context);

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset);

void PortableSparseMatrixBatchVectorMultiplyAccumulate1x4(
    const float* __restrict__ matrix, const int32_t* __restrict__ segments,
    const int32_t* __restrict__ indices, int m_rows, int m_cols,
    const float* __restrict__ vector, int n_batch, float* __restrict__ result);

void PortableSparseMatrixBatchVectorMultiplyAccumulate(
    const float* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    int m_rows, int m_cols, const float* __restrict__ vector, int n_batch,
    float* __restrict__ result);

void PortableSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result);

// Dot product of two vectors.
float PortableVectorVectorDotProduct(const float* vector1, const float* vector2,
                                     int v_size);

void PortableBatchVectorBatchVectorDotProduct(const int16_t* vector1,
                                              const int16_t* vector2,
                                              int v_size, int n_batch,
                                              int32_t* result);

void PortableVectorBatchVectorCwiseProductAccumulate(
    const int16_t* vector, int v_size, const int16_t* batch_vector, int n_batch,
    int32_t multiplier, int shift, int16_t* result);

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int16_t* output, CpuBackendContext* context);

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int8_t* output, CpuBackendContext* context);

void PortableMatrixBatchVectorMultiply(const int8_t* input,
                                       int32_t input_zeropoint,
                                       const int8_t* input_to_gate_weights,
                                       int32_t input_to_gate_effective_scale_a,
                                       int32_t input_to_gate_effective_scale_b,
                                       int32_t n_batch, int32_t n_input,
                                       int32_t n_cell, int8_t* gate_output,
                                       int8_t gate_output_zp);

void PortableMatrixBatchVectorMultiply(
    const int16_t* hidden, const int8_t* hidden_to_output_weights,
    int32_t proj_effective_scale_a, int32_t proj_effective_scale_b,
    const int32_t* gate_bias, int32_t n_batch, int32_t n_hidden,
    int32_t n_output, int32_t output_zp, int8_t* proj_output);

void PortableMatrixScalarMultiplyAccumulate(const int8_t* matrix,
                                            int32_t scalar, int32_t n_row,
                                            int32_t n_col, int32_t* output);

void PortableApplyLayerNorm(const int16_t* input,
                            const int16_t* layer_norm_weights,
                            const int32_t* bias, int32_t layer_norm_scale_a,
                            int32_t layer_norm_scale_b, int32_t variance_limit,
                            int n_batch, int n_input, int16_t* output);

void PortableApplyLayerNormFloat(const int16_t* input,
                                 const int16_t* layer_norm_weights,
                                 int32_t layer_norm_scale_a,
                                 int32_t layer_norm_scale_b,
                                 const int32_t* bias, int n_batch, int n_input,
                                 int16_t* output);

void PortableApplySigmoid(const int16_t* input, int32_t n_batch,
                          int32_t n_input, int16_t* output);

void PortableApplySigmoidFloat(const int16_t* input, int32_t n_batch,
                               int32_t n_input, int16_t* output);

void PortableApplyTanh(int32_t integer_bits, const int16_t* input,
                       int32_t n_batch, int32_t n_input, int16_t* output);

void PortableApplyTanhFloat(const int16_t* input, int32_t n_batch,
                            int32_t n_input, int32_t integer_bits,
                            int16_t* output);

void PortableCwiseMul(const int16_t* input_1, const int16_t* input_2,
                      int n_batch, int n_input, int shift, int16_t* output);

void PortableCwiseMul(const int16_t* input_1, const int16_t* input_2,
                      int32_t multiplier, int32_t shift, int32_t n_batch,
                      int32_t n_input, int32_t output_zp, int8_t* output);

void PortableCwiseAdd(const int16_t* input_1, const int16_t* input_2,
                      int n_batch, int n_input, int16_t* output);

void PortableCwiseClipping(int16_t* input, const int16_t clipping_value,
                           int32_t n_batch, int32_t n_input);

void PortableCwiseClipping(int8_t* input, const int8_t clipping_value,
                           int32_t n_batch, int32_t n_input);

// Batch vector initialization with another vector.
void PortableVectorBatchVectorAssign(const float* vector, int v_size,
                                     int n_batch, float* batch_vector);

// Add another vector for each batch in the batch vector.
void PortableVectorBatchVectorAdd(const float* vector, int v_size, int n_batch,
                                  float* batch_vector);

// Compute "1.0f - elements of vector" (used in CIFG).
void PortableSub1Vector(const float* vector, int v_size, float* result);

void PortableSub1Vector(const int16_t* vector, int v_size, int16_t* result);

// Multiply all elements of vector with a scalar.
void PortableVectorScalarMultiply(const int8_t* vector, int v_size, float scale,
                                  float* result);

// Clip elements of a vector using a abs_limit value.
void PortableClipVector(const float* vector, int v_size, float abs_limit,
                        float* result);

// Reduce-sum on a float input vector:
// input_vector: float pointer to input vector.
// output_vector: float pointer to vector.
// output_size: output vector size.
// reduction_size: number of consecutive elements from input vector which are
// added to get one element of output.
void PortableReductionSumVector(const float* input_vector, float* output_vector,
                                int output_size, int reduction_size);

void PortableReductionSumVector(const int32_t* input_vector,
                                int32_t* output_vector, int output_size,
                                int reduction_size);

void PortableReductionSumVector(const int8_t* input_vector,
                                int32_t* output_vector, int output_size,
                                int reduction_size);

// Layer norm for each batch.
void PortableMeanStddevNormalization(const float* input_vector,
                                     float* output_vector, int v_size,
                                     int n_batch);

// Saturate Add.
void PortableTwoGateSaturationgAdd(const int8_t* input, int8_t input_zp,
                                   const int8_t* recurrent, int8_t recurrent_zp,
                                   int32_t input_effective_scale_a,
                                   int32_t input_effective_scale_b,
                                   int32_t recurrent_effective_scale_a,
                                   int32_t recurrent_effective_scale_b,
                                   int32_t n_batch, int32_t n_cell,
                                   int16_t* output);

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PORTABLE_TENSOR_UTILS_IMPL_H_

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_TENSOR_UTILS_IMPL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_TENSOR_UTILS_IMPL_H_

// TODO(ghodrat): Remove this header file and the dependency to internal data
// structure.
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {
namespace tensor_utils {

#ifdef USE_NEON

// Multiply a matrix by a batch vector, and store results in a batch-size
// vector.
void NeonMatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                             int m_cols, const float* vector,
                                             int n_batch, float* result);

// Matrix multiplication for quantized values using symmetric quantization.
void NeonMatrixBatchVectorMultiplyAccumulate(const int8_t* __restrict__ matrix,
                                             const int m_rows, const int m_cols,
                                             const int8_t* __restrict__ vectors,
                                             const float* scaling_factors,
                                             int n_batch,
                                             float* __restrict__ result);

// Same as above but with a scratch buffer and CpuBackendContext for the
// int8 x int8 -> int32 accumulation computation
void NeonMatrixBatchVectorMultiplyAccumulate(const int8_t* __restrict__ matrix,
                                             const int m_rows, const int m_cols,
                                             const int8_t* __restrict__ vectors,
                                             const float* scaling_factors,
                                             int n_batch, int32_t* scratch,
                                             float* __restrict__ result,
                                             CpuBackendContext* context);

// Matrix multiplication for quantized values using asymmetric quantization.
void NeonMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset, int32_t* scratch, int32_t* row_sums,
    bool* compute_row_sums, CpuBackendContext* context);

void NeonMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, const float* per_channel_scale,
    const int32_t* input_offset);

void NeonApplyLayerNorm(const int16_t* input, const int16_t* layer_norm_weights,
                        const int32_t* bias, int32_t layer_norm_scale_a,
                        int32_t layer_norm_scale_b, int32_t variance_limit,
                        int n_batch, int n_input, int16_t* output);

void NeonApplySigmoid(const int16_t* input, int32_t n_batch, int32_t n_input,
                      int16_t* output);

void NeonApplyTanh(int32_t integer_bits, const int16_t* input, int32_t n_batch,
                   int32_t n_input, int16_t* output);

void NeonCwiseMul(const int16_t* input_1, const int16_t* input_2, int n_batch,
                  int n_input, int shift, int16_t* output);

void NeonCwiseMul(const int16_t* input_1, const int16_t* input_2,
                  int32_t multiplier, int shift, int n_batch, int n_input,
                  int32_t output_zp, int8_t* output);

void NeonCwiseAdd(const int16_t* input_1, const int16_t* input_2, int n_batch,
                  int n_input, int16_t* output);

void NeonCwiseClipping(int16_t* input, const int16_t clipping_value,
                       int32_t n_batch, int32_t n_input);

void NeonCwiseClipping(int8_t* input, const int8_t clipping_value,
                       int32_t n_batch, int32_t n_input);

void NeonMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int8_t* output, CpuBackendContext* context);

void NeonMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, const int32_t* bias,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t* scratch, int16_t* output, CpuBackendContext* context);

void NeonMatrixScalarMultiplyAccumulate(const int8_t* matrix, int32_t scalar,
                                        int32_t n_row, int32_t n_col,
                                        int32_t* output);

void NeonSparseMatrixBatchVectorMultiplyAccumulate1x4(
    const float* __restrict__ matrix, const int32_t* __restrict__ segments,
    const int32_t* __restrict__ indices, int m_rows, int m_cols,
    const float* __restrict__ vector, int n_batch, float* __restrict__ result);

// Multiply a matrix by a batch vector, and store results in a batch-size
// vector. Sparse version.
void NeonSparseMatrixBatchVectorMultiplyAccumulate(
    const float* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    int m_rows, int m_cols, const float* __restrict__ vector, int n_batch,
    float* __restrict__ result);

// Matrix multiplication for quantized values using symmetric quantization.
// Sparse version.
void NeonSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result);

// Dot product of two vectors.
float NeonVectorVectorDotProduct(const float* vector1, const float* vector2,
                                 int v_size);

// Compute "1.0f - elements of vector" (used in CIFG).
void NeonSub1Vector(const float* vector, int v_size, float* result);

void NeonSub1Vector(const int16_t* vector, int v_size, int16_t* result);

// Clip elements of a vector using a abs_limit value.
void NeonClipVector(const float* vector, int v_size, float abs_limit,
                    float* result);

// Multiply all elements of vector with a scalar.
void NeonVectorScalarMultiply(const int8_t* vector, int v_size, float scale,
                              float* result);

// Check if all entries of a vector are zero.
bool NeonIsZeroVector(const float* vector, int v_size);

// Check if all entries of a vector are zero.
bool NeonIsZeroVector(const int8_t* vector, int v_size);

// Symmetric quantizer.
void NeonSymmetricQuantizeFloats(const float* values, const int size,
                                 int8_t* quantized_values, float* min,
                                 float* max, float* scaling_factor);

// Symmetric quantizer.
void NeonSymmetricQuantizeFloats(const float* values, const int size,
                                 int8_t* quantized_values, float min, float max,
                                 float* scaling_factor);

// Asymmetric quantizer.
void NeonAsymmetricQuantizeFloats(const float* values, const int size,
                                  int8_t* quantized_values,
                                  float* scaling_factor, int32_t* offset);

// Reduce-sum on a float input vector:
// input_vector: float pointer to input vector.
// output_vector: float pointer to vector.
// output_size: output vector size.
// reduction_size: number of consecutive elements from input vector which are
// added to get one element of output.
void NeonReductionSumVector(const float* input_vector, float* output_vector,
                            int output_size, int reduction_size);

void NeonReductionSumVector(const int8_t* input_vector, int32_t* output_vector,
                            int output_size, int reduction_size);

#endif  // USE_NEON

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_TENSOR_UTILS_IMPL_H_

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_TENSOR_UTILS_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_TENSOR_UTILS_H_

// TODO(ghodrat): Remove this header file and the dependency to internal data
// structure.
#include "tensorflow/contrib/lite/c/builtin_op_data.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/tensor_utils_impl.h"

namespace tflite {
namespace tensor_utils {

void MatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                         int m_cols, const float* vector,
                                         int n_batch, float* result,
                                         int result_stride) {
  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols,
                   vector, n_batch, result, result_stride);
}

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, int result_stride) {
  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols,
                   vectors, scaling_factors, n_batch, result, result_stride);
}

void VectorVectorCwiseProduct(const float* vector1, const float* vector2,
                              int v_size, float* result) {
  NEON_OR_PORTABLE(VectorVectorCwiseProduct, vector1, vector2, v_size, result);
}

void VectorVectorCwiseProductAccumulate(const float* vector1,
                                        const float* vector2, int v_size,
                                        float* result) {
  NEON_OR_PORTABLE(VectorVectorCwiseProductAccumulate, vector1, vector2, v_size,
                   result);
}

void VectorBatchVectorCwiseProduct(const float* vector, int v_size,
                                   const float* batch_vector, int n_batch,
                                   float* result) {
  NEON_OR_PORTABLE(VectorBatchVectorCwiseProduct, vector, v_size, batch_vector,
                   n_batch, result);
}

void VectorBatchVectorCwiseProductAccumulate(const float* vector, int v_size,
                                             const float* batch_vector,
                                             int n_batch, float* result) {
  NEON_OR_PORTABLE(VectorBatchVectorCwiseProductAccumulate, vector, v_size,
                   batch_vector, n_batch, result);
}

float VectorVectorDotProduct(const float* vector1, const float* vector2,
                             int v_size) {
  return NEON_OR_PORTABLE(VectorVectorDotProduct, vector1, vector2, v_size);
}

void BatchVectorBatchVectorDotProduct(const float* vector1,
                                      const float* vector2, int v_size,
                                      int n_batch, float* result,
                                      int result_stride) {
  NEON_OR_PORTABLE(BatchVectorBatchVectorDotProduct, vector1, vector2, v_size,
                   n_batch, result, result_stride);
}

void VectorBatchVectorAdd(const float* vector, int v_size, int n_batch,
                          float* batch_vector) {
  PortableVectorBatchVectorAdd(vector, v_size, n_batch, batch_vector);
}

void VectorBatchVectorAssign(const float* vector, int v_size, int n_batch,
                             float* batch_vector) {
  PortableVectorBatchVectorAssign(vector, v_size, n_batch, batch_vector);
}

void ApplySigmoidToVector(const float* vector, int v_size, float* result) {
  PortableApplySigmoidToVector(vector, v_size, result);
}

void ApplyActivationToVector(const float* vector, int v_size,
                             TfLiteFusedActivation activation, float* result) {
  PortableApplyActivationToVector(vector, v_size, activation, result);
}

void CopyVector(const float* vector, int v_size, float* result) {
  PortableCopyVector(vector, v_size, result);
}

void Sub1Vector(const float* vector, int v_size, float* result) {
  NEON_OR_PORTABLE(Sub1Vector, vector, v_size, result);
}

void ZeroVector(float* vector, int v_size) {
  PortableZeroVector(vector, v_size);
}

float Clip(float f, float abs_limit) { return PortableClip(f, abs_limit); }

// Check if all entries of a vector are zero.
bool IsZeroVector(const float* vector, int v_size) {
  return NEON_OR_PORTABLE(IsZeroVector, vector, v_size);
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

void VectorShiftLeft(float* vector, int v_size, float shift_value) {
  NEON_OR_PORTABLE(VectorShiftLeft, vector, v_size, shift_value);
}

void ReductionSumVector(const float* input_vector, float* output_vector,
                        int output_size, int reduction_size) {
  NEON_OR_PORTABLE(ReductionSumVector, input_vector, output_vector, output_size,
                   reduction_size);
}

void MeanStddevNormalization(const float* input_vector, float* output_vector,
                             int v_size, int n_batch,
                             float normalization_epsilon) {
  PortableMeanStddevNormalization(input_vector, output_vector, v_size, n_batch,
                                  normalization_epsilon);
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_TENSOR_UTILS_H_

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_PORTABLE_TENSOR_UTILS_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_PORTABLE_TENSOR_UTILS_H_

// TODO(ghodrat): Remove this header file and the dependency to internal data
// structure.
#include "tensorflow/contrib/lite/builtin_op_data.h"

namespace tflite {
namespace tensor_utils {

// Limit a float input f between +abs_limit and -abs_limit.
float PortableClip(float f, float abs_limit);

void PortableSymmetricQuantizeFloats(const float* values, const int size,
                                     int8_t* quantized_values, float* min,
                                     float* max, float* scaling_factor);

// Multiply a matrix by a batch vector, and store results in a batch-size
// vector.
void PortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride);

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, int result_stride);

// Cwise product of two vectors.
void PortableVectorVectorCwiseProduct(const float* vector1,
                                      const float* vector2, int v_size,
                                      float* result);

// Cwise product and accumulate of two vectors. Since it's a MAC opertation, the
// assumption here is that result array is initialized to valid values.
void PortableVectorVectorCwiseProductAccumulate(const float* vector1,
                                                const float* vector2,
                                                int v_size, float* result);

// Dot product of two vectors.
float PortableVectorVectorDotProduct(const float* vector1, const float* vector2,
                                     int v_size);

// Dot product of two batch vectors.
void PortableBatchVectorBatchVectorDotProduct(const float* vector1,
                                              const float* vector2, int v_size,
                                              int n_batch, float* result,
                                              int result_stride);

// Cwise product and accumulate of a vector and a batch-vector. Since it's a MAC
// operation, the assumption here is that result array is initialized to valid
// values.
void PortableVectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result);

// Batch vector initialization with another vector.
void PortableVectorBatchVectorAssign(const float* vector, int v_size,
                                     int n_batch, float* batch_vector);

// Apply sigmoid to elements of a vector.
void PortableApplySigmoidToVector(const float* vector, int v_size,
                                  float* result);

// Apply activation function to elements of a vector.
void PortableApplyActivationToVector(const float* vector, int v_size,
                                     TfLiteFusedActivation activation,
                                     float* result);

// Copy vector to another vector.
void PortableCopyVector(const float* vector, int v_size, float* result);

// Compute "1.0f - elements of vector" (used in CIFG).
void PortableSub1Vector(const float* vector, int v_size, float* result);

// Fill vector with 0.f.
void PortableZeroVector(float* vector, int v_size);

// Clip elements of a vector using a abs_limit value.
void PortableClipVector(const float* vector, int v_size, float abs_limit,
                        float* result);

// Shift left a vector in place with v_size size.
void PortableVectorShiftLeft(float* vector, int v_size, float shift_value);

// Reduce-sum on a float input vector:
// input_vector: float pointer to input vector.
// output_vector: float pointer to vector.
// output_size: output vector size.
// reduction_size: number of consecutive elements from input vector which are
// added to get one element of output.
void PortableReductionSumVector(const float* input_vector, float* output_vector,
                                int output_size, int reduction_size);

float Clip(float f, float abs_limit) { return PortableClip(f, abs_limit); }

void SymmetricQuantizeFloats(const float* values, const int size,
                             int8_t* quantized_values, float* min, float* max,
                             float* scaling_factor) {
  return PortableSymmetricQuantizeFloats(values, size, quantized_values, min,
                                         max, scaling_factor);
}

void MatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                         int m_cols, const float* vector,
                                         int n_batch, float* result,
                                         int result_stride) {
  PortableMatrixBatchVectorMultiplyAccumulate(matrix, m_rows, m_cols, vector,
                                              n_batch, result, result_stride);
}

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vector, const float* scaling_factors,
    int n_batch, float* __restrict__ result, int result_stride) {
  PortableMatrixBatchVectorMultiplyAccumulate(matrix, m_rows, m_cols, vector,
                                              scaling_factors, n_batch, result,
                                              result_stride);
}

void VectorVectorCwiseProduct(const float* vector1, const float* vector2,
                              int v_size, float* result) {
  PortableVectorVectorCwiseProduct(vector1, vector2, v_size, result);
}

void VectorVectorCwiseProductAccumulate(const float* vector1,
                                        const float* vector2, int v_size,
                                        float* result) {
  PortableVectorVectorCwiseProductAccumulate(vector1, vector2, v_size, result);
}

void VectorBatchVectorCwiseProductAccumulate(const float* vector, int v_size,
                                             const float* batch_vector,
                                             int n_batch, float* result) {
  PortableVectorBatchVectorCwiseProductAccumulate(vector, v_size, batch_vector,
                                                  n_batch, result);
}

float VectorVectorDotProduct(const float* vector1, const float* vector2,
                             int v_size) {
  return PortableVectorVectorDotProduct(vector1, vector2, v_size);
}

void BatchVectorBatchVectorDotProduct(const float* vector1,
                                      const float* vector2, int v_size,
                                      int n_batch, float* result,
                                      int result_stride) {
  PortableBatchVectorBatchVectorDotProduct(vector1, vector2, v_size, n_batch,
                                           result, result_stride);
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
  PortableSub1Vector(vector, v_size, result);
}

void ZeroVector(float* vector, int v_size) {
  PortableZeroVector(vector, v_size);
}

void ClipVector(const float* vector, int v_size, float abs_limit,
                float* result) {
  PortableClipVector(vector, v_size, abs_limit, result);
}

void VectorShiftLeft(float* vector, int v_size, float shift_value) {
  PortableVectorShiftLeft(vector, v_size, shift_value);
}

void ReductionSumVector(const float* input_vector, float* output_vector,
                        int output_size, int reduction_size) {
  PortableReductionSumVector(input_vector, output_vector, output_size,
                             reduction_size);
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_PORTABLE_TENSOR_UTILS_H_

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_

#include "tensorflow/contrib/lite/builtin_op_data.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {
namespace tensor_utils {

// Limit a float input f between +abs_limit and -abs_limit.
float Clip(float f, float abs_limit);

// Checks if all entries of vector are zero.
bool IsZeroVector(const float* vector, int v_size);

// Quantizes a buffer of floating point values using a symmetric quantization
// (i.e. linear quantization without an offset) to 8-bit signed integers.
// It also outputs the range (min, max) of the floating point buffer, and the
// scaling factor used to quantize the values.
void SymmetricQuantizeFloats(const float* values, const int size,
                             int8_t* quantized_values, float* min_value,
                             float* max_value, float* scaling_factor);

// Multiplies a matrix by a "batched" vector (i.e. a matrix with a batch
// dimension composed by input vectors independent from each other). The result
// of the multiplication is accumulated to the passed result buffer.
// More specifically, for a matrix M of shape [n, i] and a batched-vector
// of shape [i, batch] it will first compute the product of shape [n, batch].
// This product will be accumulated to the result buffer, using a stride value
// provided in result_stride (the number of elements between consecutive result
// values). For example result_stride = 1, will cause the output to look like
// this:
// [O_1, 0_2, ... O_rows]
// but result_stride = 3, will cause it to be arranged like this in memory:
// [O_1, x, x, 0_2, x, x, ..., O_rows]
void MatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                         int m_cols, const float* vector,
                                         int n_batch, float* result,
                                         int result_stride);

// Same as the function above, but for values quantized using symmetric
// quantization (e.g. by calling SymmetricQuantizeFloats).
// The passed scaling factors is a buffer of the quantization scaling factors
// that will be used to dequentize the products into the final result buffer.
// These scaling factors are the multiplication of the matrix scaling factor
// by the vector's scaling factor, one per batch (i.e. this allows quantizing
// each batch in the batch-vector matrix independently).
void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, int result_stride);

// Cwise product of two vectors.
void VectorVectorCwiseProduct(const float* vector1, const float* vector2,
                              int v_size, float* result);

// Cwise product and accumulate of two vectors. Since it's a MAC opertation, the
// assumption here is that result array is initialized to valid values.
void VectorVectorCwiseProductAccumulate(const float* vector1,
                                        const float* vector2, int v_size,
                                        float* result);

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
// Then result will be a vector of n_batch size which will be saved with a
// stride of result_stride in memory starting from 'result':
// [x_1_1 * y_1_1 + x_1_2 * y_1_2 + ... + x_1_vsize * y_1_vsize,
//  x_2_1 * y_2_1 + x_2_2 * y_2_2 + ... + x_2_vsize * y_2_vsize,
//  ...
//  x_nbatch_1 * y_nbatch_1 + ... + x_nbatch_vsize * y_nbatch_vsize]
void BatchVectorBatchVectorDotProduct(const float* vector1,
                                      const float* vector2, int v_size,
                                      int n_batch, float* result,
                                      int result_stride);

// Cwise product of a vector and a batch-vector.
void VectorBatchVectorCwiseProduct(const float* vector, int v_size,
                                   const float* batch_vector, int n_batch,
                                   float* result);

// Cwise product and accumulate of a vector and a batch-vector. Since it's a MAC
// operation, the assumption here is that result array is initialized to valid
// values.
void VectorBatchVectorCwiseProductAccumulate(const float* vector, int v_size,
                                             const float* batch_vector,
                                             int n_batch, float* result);

// Add another vector for each batch in the batch vector.
void VectorBatchVectorAdd(const float* vector, int v_size, int n_batch,
                          float* batch_vector);

// Batch vector initialization with another vector.
void VectorBatchVectorAssign(const float* vector, int v_size, int n_batch,
                             float* batch_vector);

// Apply sigmoid to elements of a vector.
void ApplySigmoidToVector(const float* vector, int v_size, float* result);

// Apply activation function to elements of a vector.
void ApplyActivationToVector(const float* vector, int v_size,
                             TfLiteFusedActivation activation, float* result);

// Copy vector to another vector.
void CopyVector(const float* vector, int v_size, float* result);

// Compute "1.0f - elements of vector" (used in CIFG).
void Sub1Vector(const float* vector, int v_size, float* result);

// Fill vector with 0.f.
void ZeroVector(float* vector, int v_size);

// Multiply all elements of vector with a scalar.
void VectorScalarMultiply(const int8_t* vector, int v_size, float scale,
                          float* result);

// Clip elements of a vector using a abs_limit value.
void ClipVector(const float* vector, int v_size, float abs_limit,
                float* result);

// Shift left a vector in place with v_size size.
void VectorShiftLeft(float* vector, int v_size, float shift_value);

// Reduce-sum on a float input vector:
// input_vector: float pointer to input vector.
// output_vector: float pointer to vector.
// output_size: output vector size.
// reduction_size: number of consecutive elements from input vector which are
// added to get one element of output.
void ReductionSumVector(const float* input_vector, float* output_vector,
                        int output_size, int reduction_size);

// Layer norm for each batch.
// normalization_epsilon is added to avoid divergence.
void MeanStddevNormalization(const float* input_vector, float* output_vector,
                             int v_size, int n_batch,
                             float normalization_epsilon);
}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_

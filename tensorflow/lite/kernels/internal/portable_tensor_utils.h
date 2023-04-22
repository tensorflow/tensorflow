/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_PORTABLE_TENSOR_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_PORTABLE_TENSOR_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {

namespace tensor_utils {

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

// Add another vector for each batch in the batch vector.
template <typename T>
void VectorBatchVectorAdd(const T* vector, int v_size, int n_batch,
                          T* batch_vector) {
  for (int b = 0; b < n_batch; b++) {
    for (int i = 0; i < v_size; ++i) {
      batch_vector[i] += vector[i];
    }
    batch_vector += v_size;
  }
}

// Cwise product of two vectors.
template <typename T>
inline void VectorVectorCwiseProduct(const T* __restrict__ vector1,
                                     const T* __restrict__ vector2, int v_size,
                                     T* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    *result++ = *vector1++ * *vector2++;
  }
}

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

// Batch vector initialization with another vector.
template <typename T>
void VectorBatchVectorAssign(const T* vector, int v_size, int n_batch,
                             T* batch_vector) {
  for (int b = 0; b < n_batch; b++) {
    std::copy_n(vector, v_size, batch_vector + b * v_size);
  }
}

}  // namespace tensor_utils

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_PORTABLE_TENSOR_UTILS_H_

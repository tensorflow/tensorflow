/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BATCH_MATMUL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BATCH_MATMUL_H_

#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_utils_common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {
namespace batch_matmul {

// Determine which dimension is the broadcast dimension.
inline int broadcast_dim(int lhs_dim, int rhs_dim) {
  if (lhs_dim == rhs_dim) return lhs_dim;
  if (lhs_dim == 1) return rhs_dim;
  TFLITE_DCHECK_EQ(rhs_dim, 1);
  return lhs_dim;
}

// Compute the "extent" for iterating on this dimension.
// If we are broadcasting, then don't advance (i.e return 0).
inline int extent(const RuntimeShape& shape, int x) {
  if (shape.Dims(x) == 1) {
    return 0;
  }
  int prod = 1;
  for (int i = x + 1; i < shape.DimensionsCount(); ++i) {
    prod *= shape.Dims(i);
  }
  return prod;
}

}  // namespace batch_matmul

inline void BatchMatMul(const RuntimeShape& lhs_shape, const float* lhs_data,
                        const RuntimeShape& rhs_shape, const float* rhs_data,
                        const RuntimeShape& output_shape, float* output_data) {
  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(5, rhs_shape);

  const int batch_dim0 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(0), extended_rhs_shape.Dims(0));
  const int batch_dim1 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(1), extended_rhs_shape.Dims(1));
  const int batch_dim2 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(2), extended_rhs_shape.Dims(2));

  const int lhs_ext0 = batch_matmul::extent(extended_lhs_shape, 0);
  const int lhs_ext1 = batch_matmul::extent(extended_lhs_shape, 1);
  const int lhs_ext2 = batch_matmul::extent(extended_lhs_shape, 2);
  const int rhs_ext0 = batch_matmul::extent(extended_rhs_shape, 0);
  const int rhs_ext1 = batch_matmul::extent(extended_rhs_shape, 1);
  const int rhs_ext2 = batch_matmul::extent(extended_rhs_shape, 2);

  // Set params for each matrix multiply.
  const int lhs_rows = extended_lhs_shape.Dims(3);
  const int rhs_cols = extended_rhs_shape.Dims(4);
  const int accum_depth = extended_lhs_shape.Dims(4);

  for (int b0 = 0; b0 < batch_dim0; ++b0) {
    const float* lhs_ptr0 = lhs_data + (b0 * lhs_ext0);
    const float* rhs_ptr0 = rhs_data + (b0 * rhs_ext0);
    for (int b1 = 0; b1 < batch_dim1; ++b1) {
      const float* lhs_ptr1 = lhs_ptr0 + b1 * lhs_ext1;
      const float* rhs_ptr1 = rhs_ptr0 + b1 * rhs_ext1;
      for (int b2 = 0; b2 < batch_dim2; ++b2) {
        const float* lhs_ptr2 = lhs_ptr1 + b2 * lhs_ext2;
        const float* rhs_ptr2 = rhs_ptr1 + b2 * rhs_ext2;
        float* out_ptr = output_data + ((b0 * batch_dim1 * batch_dim2) +
                                        b1 * batch_dim2 + b2) *
                                           lhs_rows * rhs_cols;
        for (int j = 0; j < rhs_cols; ++j) {
          for (int i = 0; i < lhs_rows; ++i) {
            float total = 0.f;
            for (int k = 0; k < accum_depth; ++k) {
              total +=
                  lhs_ptr2[accum_depth * i + k] * rhs_ptr2[j * accum_depth + k];
            }
            int idx = lhs_rows * j + i;
            out_ptr[idx] = total;
          }
        }
      }
    }
  }
}

inline void BatchMatMul(const RuntimeShape& lhs_shape, const int8_t* lhs_data,
                        const RuntimeShape& rhs_shape, const int8_t* rhs_data,
                        const float* scaling_factors,
                        const int32_t* input_offset, int32_t* row_sums,
                        const RuntimeShape& output_shape, float* output_data,
                        bool* compute_row_sums) {
  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(5, rhs_shape);

  const int batch_dim0 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(0), extended_rhs_shape.Dims(0));
  const int batch_dim1 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(1), extended_rhs_shape.Dims(1));
  const int batch_dim2 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(2), extended_rhs_shape.Dims(2));

  const int lhs_ext0 = batch_matmul::extent(extended_lhs_shape, 0);
  const int lhs_ext1 = batch_matmul::extent(extended_lhs_shape, 1);
  const int lhs_ext2 = batch_matmul::extent(extended_lhs_shape, 2);
  const int rhs_ext0 = batch_matmul::extent(extended_rhs_shape, 0);
  const int rhs_ext1 = batch_matmul::extent(extended_rhs_shape, 1);
  const int rhs_ext2 = batch_matmul::extent(extended_rhs_shape, 2);

  // Set params for each matrix multiply.
  const int lhs_rows = extended_lhs_shape.Dims(3);
  const int rhs_cols = extended_rhs_shape.Dims(4);
  const int accum_depth = extended_lhs_shape.Dims(4);

  const int ioff_ext0 = rhs_ext0 == 0 ? 0 : rhs_cols;
  const int ioff_ext1 = rhs_ext1 == 0 ? 0 : rhs_cols;
  const int ioff_ext2 = rhs_ext2 == 0 ? 0 : rhs_cols;
  const int woff_ext0 = lhs_ext0 == 0 ? 0 : lhs_rows;
  const int woff_ext1 = lhs_ext1 == 0 ? 0 : lhs_rows;
  const int woff_ext2 = lhs_ext2 == 0 ? 0 : lhs_rows;

  if (!compute_row_sums || *compute_row_sums) {
    int num_weights_matrices = 1;
    for (int i = 1; i < extended_lhs_shape.DimensionsCount() - 2; ++i) {
      num_weights_matrices *= extended_lhs_shape.Dims(i);
    }
    tensor_utils::ReductionSumVector(
        lhs_data, row_sums, num_weights_matrices * lhs_rows, accum_depth);
    if (compute_row_sums) {
      *compute_row_sums = false;
    }
  }

  for (int b0 = 0; b0 < batch_dim0; ++b0) {
    const int8_t* lhs_ptr0 = lhs_data + (b0 * lhs_ext0);
    const int8_t* rhs_ptr0 = rhs_data + (b0 * rhs_ext0);
    const int32_t* ioff_ptr0 = input_offset + (b0 * ioff_ext0);
    const float* scale_ptr0 = scaling_factors + (b0 * ioff_ext0);
    const int32_t* woff_ptr0 = row_sums + (b0 * woff_ext0);
    for (int b1 = 0; b1 < batch_dim1; ++b1) {
      const int8_t* lhs_ptr1 = lhs_ptr0 + b1 * lhs_ext1;
      const int8_t* rhs_ptr1 = rhs_ptr0 + b1 * rhs_ext1;
      const int32_t* ioff_ptr1 = ioff_ptr0 + (b1 * ioff_ext1);
      const float* scale_ptr1 = scale_ptr0 + (b1 * ioff_ext1);
      const int32_t* woff_ptr1 = woff_ptr0 + (b1 * woff_ext1);
      for (int b2 = 0; b2 < batch_dim2; ++b2) {
        const int8_t* lhs_ptr2 = lhs_ptr1 + b2 * lhs_ext2;
        const int8_t* rhs_ptr2 = rhs_ptr1 + b2 * rhs_ext2;
        const int32_t* ioff_ptr2 = ioff_ptr1 + (b2 * ioff_ext2);
        const float* scale_ptr2 = scale_ptr1 + (b2 * ioff_ext2);
        const int32_t* woff_ptr2 = woff_ptr1 + (b2 * woff_ext2);
        float* out_ptr = output_data + ((b0 * batch_dim1 * batch_dim2) +
                                        b1 * batch_dim2 + b2) *
                                           lhs_rows * rhs_cols;
        for (int j = 0; j < rhs_cols; ++j) {
          const float batch_scaling_factor = scale_ptr2[j];
          const float batch_offset = static_cast<float>(ioff_ptr2[j]);
          for (int i = 0; i < lhs_rows; ++i) {
            int32_t total = 0;
            for (int k = 0; k < accum_depth; ++k) {
              total +=
                  lhs_ptr2[accum_depth * i + k] * rhs_ptr2[j * accum_depth + k];
            }
            int32_t row_sum = woff_ptr2[i];
            total -= row_sum * batch_offset;
            int idx = lhs_rows * j + i;
            out_ptr[idx] += batch_scaling_factor * total;
          }
        }
      }
    }
  }
}

template <typename T, typename AccumT>
inline void BatchMatMul(const FullyConnectedParams& params,
                        const RuntimeShape& lhs_shape, const T* lhs_data,
                        const RuntimeShape& rhs_shape, const T* rhs_data,
                        const RuntimeShape& output_shape, T* output_data) {
  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(5, rhs_shape);

  const int batch_dim0 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(0), extended_rhs_shape.Dims(0));
  const int batch_dim1 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(1), extended_rhs_shape.Dims(1));
  const int batch_dim2 = batch_matmul::broadcast_dim(
      extended_lhs_shape.Dims(2), extended_rhs_shape.Dims(2));

  const int lhs_ext0 = batch_matmul::extent(extended_lhs_shape, 0);
  const int lhs_ext1 = batch_matmul::extent(extended_lhs_shape, 1);
  const int lhs_ext2 = batch_matmul::extent(extended_lhs_shape, 2);
  const int rhs_ext0 = batch_matmul::extent(extended_rhs_shape, 0);
  const int rhs_ext1 = batch_matmul::extent(extended_rhs_shape, 1);
  const int rhs_ext2 = batch_matmul::extent(extended_rhs_shape, 2);

  // Set params for each matrix multiply.
  const int lhs_rows = extended_lhs_shape.Dims(3);
  const int rhs_cols = extended_rhs_shape.Dims(4);
  const int accum_depth = extended_lhs_shape.Dims(4);

  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  for (int b0 = 0; b0 < batch_dim0; ++b0) {
    const T* lhs_ptr0 = lhs_data + (b0 * lhs_ext0);
    const T* rhs_ptr0 = rhs_data + (b0 * rhs_ext0);
    for (int b1 = 0; b1 < batch_dim1; ++b1) {
      const T* lhs_ptr1 = lhs_ptr0 + b1 * lhs_ext1;
      const T* rhs_ptr1 = rhs_ptr0 + b1 * rhs_ext1;
      for (int b2 = 0; b2 < batch_dim2; ++b2) {
        const T* lhs_ptr2 = lhs_ptr1 + b2 * lhs_ext2;
        const T* rhs_ptr2 = rhs_ptr1 + b2 * rhs_ext2;
        T* out_ptr = output_data +
                     ((b0 * batch_dim1 * batch_dim2) + b1 * batch_dim2 + b2) *
                         lhs_rows * rhs_cols;

        for (int j = 0; j < rhs_cols; ++j) {
          for (int i = 0; i < lhs_rows; ++i) {
            AccumT total = 0;
            for (int k = 0; k < accum_depth; ++k) {
              AccumT lhs_val = lhs_ptr2[accum_depth * i + k];
              AccumT rhs_val = rhs_ptr2[accum_depth * j + k];
              total += (lhs_val + filter_offset) * (rhs_val + input_offset);
            }
            int32_t total_scaled = MultiplyByQuantizedMultiplier(
                total, output_multiplier, output_shift);
            total_scaled += output_offset;
            total_scaled = std::max(total_scaled, output_activation_min);
            total_scaled = std::min(total_scaled, output_activation_max);
            const int idx = lhs_rows * j + i;
            out_ptr[idx] = static_cast<T>(total_scaled);
          }
        }
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BATCH_MATMUL_H_

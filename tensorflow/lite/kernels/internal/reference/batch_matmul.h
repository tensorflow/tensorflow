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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

inline void BatchMatMul(const RuntimeShape& lhs_shape, const float* lhs_data,
                        const RuntimeShape& rhs_shape, const float* rhs_data,
                        const RuntimeShape& output_shape, float* output_data) {
  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(5, rhs_shape);

  // Determine which dimension is the broadcast dimension.
  auto broadcast_dim = [](int lhs_dim, int rhs_dim) {
    if (lhs_dim == rhs_dim) return lhs_dim;
    if (lhs_dim == 1) return rhs_dim;
    TFLITE_DCHECK_EQ(rhs_dim, 1);
    return lhs_dim;
  };

  // Compute the "extent" for iterating on this dimension.
  // If we are broadcasting, then don't advance (i.e return 0).
  auto extent = [](const RuntimeShape& shape, int x) {
    if (shape.Dims(x) == 1) {
      return 0;
    }
    int prod = 1;
    for (int i = x + 1; i < shape.DimensionsCount(); ++i) {
      prod *= shape.Dims(i);
    }
    return prod;
  };

  const int batch_dim0 =
      broadcast_dim(extended_lhs_shape.Dims(0), extended_rhs_shape.Dims(0));
  const int batch_dim1 =
      broadcast_dim(extended_lhs_shape.Dims(1), extended_rhs_shape.Dims(1));
  const int batch_dim2 =
      broadcast_dim(extended_lhs_shape.Dims(2), extended_rhs_shape.Dims(2));

  const int lhs_ext0 = extent(extended_lhs_shape, 0);
  const int lhs_ext1 = extent(extended_lhs_shape, 1);
  const int lhs_ext2 = extent(extended_lhs_shape, 2);
  const int rhs_ext0 = extent(extended_rhs_shape, 0);
  const int rhs_ext1 = extent(extended_rhs_shape, 1);
  const int rhs_ext2 = extent(extended_rhs_shape, 2);

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

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BATCH_MATMUL_H_

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CUMSUM_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CUMSUM_H_

#include <cstdint>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

template <typename T>
inline void CumSum(const T* input_data, const RuntimeShape& shape, int32_t axis,
                   bool exclusive, bool reverse, T* output_data) {
  const int32_t rank = shape.DimensionsCount();
  TFLITE_DCHECK_GE(rank, 1);
  TFLITE_DCHECK_GE(axis, 0);
  TFLITE_DCHECK_LT(axis, rank);

  size_t inner = 1;
  size_t outer = 1;
  size_t depth = 1;
  for (int32_t i = 0; i < rank; i++) {
    if (i < axis)
      inner *= shape.Dims(i);
    else if (i > axis)
      outer *= shape.Dims(i);
    else
      depth = shape.Dims(i);
  }

  for (size_t outer_index = 0; outer_index < outer; outer_index++) {
    size_t outer_index_adj;
    if (reverse)
      outer_index_adj = (outer - 1) - outer_index;
    else
      outer_index_adj = outer_index;
    for (size_t inner_index = 0; inner_index < inner; inner_index++) {
      T accumulator = 0;
      size_t inner_index_adj;
      if (reverse)
        inner_index_adj = (inner - 1) - inner_index;
      else
        inner_index_adj = inner_index;
      for (size_t depth_index = 0; depth_index < depth; depth_index++) {
        size_t depth_index_adj;
        if (reverse)
          depth_index_adj = (depth - 1) - depth_index;
        else
          depth_index_adj = depth_index;

        size_t index = outer_index_adj;
        index += inner_index_adj * depth * outer;
        index += depth_index_adj * outer;

        if (exclusive) {
          output_data[index] = accumulator;
          accumulator += input_data[index];
        } else {
          accumulator += input_data[index];
          output_data[index] = accumulator;
        }
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CUMSUM_H_

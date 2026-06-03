/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BROADCAST_LOOP_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BROADCAST_LOOP_H_

#include <algorithm>
#include <vector>

#include "tensorflow/lite/kernels/internal/runtime_shape.h"

namespace tflite {
namespace reference_ops {

inline std::vector<int> BroadcastStridesForShape(
    const RuntimeShape& unextended_shape,
    const RuntimeShape& extended_output_shape) {
  const int dims_count = extended_output_shape.DimensionsCount();
  const RuntimeShape extended_shape =
      RuntimeShape::ExtendedShape(dims_count, unextended_shape);
  std::vector<int> strides(dims_count);
  int stride = 1;
  for (int i = dims_count - 1; i >= 0; --i) {
    const int dim = extended_shape.Dims(i);
    const int output_dim = extended_output_shape.Dims(i);
    strides[i] = (dim == 1 && output_dim != 1) ? 0 : stride;
    stride *= dim;
  }
  return strides;
}

inline std::vector<int> StridesForShape(const RuntimeShape& shape) {
  const int dims_count = shape.DimensionsCount();
  std::vector<int> strides(dims_count);
  int stride = 1;
  for (int i = dims_count - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape.Dims(i);
  }
  return strides;
}

template <typename Fn>
inline void ForEachBroadcastedElement(const RuntimeShape& input1_shape,
                                      const RuntimeShape& input2_shape,
                                      const RuntimeShape& output_shape, Fn fn) {
  const int dims_count = std::max(
      output_shape.DimensionsCount(),
      std::max(input1_shape.DimensionsCount(), input2_shape.DimensionsCount()));
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(dims_count, output_shape);
  const std::vector<int> output_strides =
      StridesForShape(extended_output_shape);
  const std::vector<int> input1_strides =
      BroadcastStridesForShape(input1_shape, extended_output_shape);
  const std::vector<int> input2_strides =
      BroadcastStridesForShape(input2_shape, extended_output_shape);

  const int flat_size = output_shape.FlatSize();
  for (int output_index = 0; output_index < flat_size; ++output_index) {
    int remaining_index = output_index;
    int input1_index = 0;
    int input2_index = 0;
    for (int dim = 0; dim < dims_count; ++dim) {
      const int output_stride = output_strides[dim];
      const int coordinate =
          output_stride == 0 ? 0 : remaining_index / output_stride;
      if (output_stride != 0) {
        remaining_index %= output_stride;
      }
      input1_index += coordinate * input1_strides[dim];
      input2_index += coordinate * input2_strides[dim];
    }
    fn(output_index, input1_index, input2_index);
  }
}

template <typename Fn>
inline void ForEachBroadcastedElement(const RuntimeShape& input1_shape,
                                      const RuntimeShape& input2_shape,
                                      const RuntimeShape& input3_shape,
                                      const RuntimeShape& output_shape, Fn fn) {
  const int dims_count = std::max(
      std::max(output_shape.DimensionsCount(), input1_shape.DimensionsCount()),
      std::max(input2_shape.DimensionsCount(), input3_shape.DimensionsCount()));
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(dims_count, output_shape);
  const std::vector<int> output_strides =
      StridesForShape(extended_output_shape);
  const std::vector<int> input1_strides =
      BroadcastStridesForShape(input1_shape, extended_output_shape);
  const std::vector<int> input2_strides =
      BroadcastStridesForShape(input2_shape, extended_output_shape);
  const std::vector<int> input3_strides =
      BroadcastStridesForShape(input3_shape, extended_output_shape);

  const int flat_size = output_shape.FlatSize();
  for (int output_index = 0; output_index < flat_size; ++output_index) {
    int remaining_index = output_index;
    int input1_index = 0;
    int input2_index = 0;
    int input3_index = 0;
    for (int dim = 0; dim < dims_count; ++dim) {
      const int output_stride = output_strides[dim];
      const int coordinate =
          output_stride == 0 ? 0 : remaining_index / output_stride;
      if (output_stride != 0) {
        remaining_index %= output_stride;
      }
      input1_index += coordinate * input1_strides[dim];
      input2_index += coordinate * input2_strides[dim];
      input3_index += coordinate * input3_strides[dim];
    }
    fn(output_index, input1_index, input2_index, input3_index);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BROADCAST_LOOP_H_

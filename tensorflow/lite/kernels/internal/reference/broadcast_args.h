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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BROADCAST_ARGS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BROADCAST_ARGS_H_

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

template <typename T>
void BroadcastArgs(const RuntimeShape& input1_shape, const T* input1_data,
                   const RuntimeShape& input2_shape, const T* input2_data,
                   const RuntimeShape& output_shape, T* output_data) {
  // Gets data at the backward index i of the shape tensor. Returns 1 if the
  // index is out of range.
  auto get_shape_data = [](const RuntimeShape& shape, const T* data,
                           int backward_idx) -> T {
    int forward_idx = shape.FlatSize() - 1 - backward_idx;
    if (forward_idx < 0) return 1;
    return data[forward_idx];
  };

  int output_num_elements = output_shape.FlatSize();
  for (int i = 0; i < output_num_elements; ++i) {
    int backward_i = output_num_elements - 1 - i;
    int shape1_i = get_shape_data(input1_shape, input1_data, i);
    int shape2_i = get_shape_data(input2_shape, input2_data, i);
    if (shape1_i == 1) {
      output_data[backward_i] = shape2_i;
    } else if (shape2_i == 1) {
      output_data[backward_i] = shape1_i;
    } else {
      TFLITE_CHECK_EQ(shape1_i, shape2_i);
      output_data[backward_i] = shape1_i;
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BROADCAST_ARGS_H_

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BINARY_FUNCTION_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BINARY_FUNCTION_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/reference/broadcast_loop.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

// Also appears to duplicate MinimumMaximum.
//
// R: Result type. T1: Input 1 type. T2: Input 2 type.
template <typename R, typename T1, typename T2>
inline void BroadcastBinaryFunction4DSlow(
    const RuntimeShape& unextended_input1_shape, const T1* input1_data,
    const RuntimeShape& unextended_input2_shape, const T2* input2_data,
    const RuntimeShape& unextended_output_shape, R* output_data,
    R (*func)(T1, T2)) {
  ForEachBroadcastedElement(
      unextended_input1_shape, unextended_input2_shape, unextended_output_shape,
      [&](int output_index, int input1_index, int input2_index) {
        output_data[output_index] =
            func(input1_data[input1_index], input2_data[input2_index]);
      });
}

// R: Result type. T1: Input 1 type. T2: Input 2 type.
template <typename R, typename T1, typename T2>
inline void BinaryFunction(const RuntimeShape& input1_shape,
                           const T1* input1_data,
                           const RuntimeShape& input2_shape,
                           const T2* input2_data,
                           const RuntimeShape& output_shape, R* output_data,
                           R (*func)(T1, T2)) {
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = func(input1_data[i], input2_data[i]);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_BINARY_FUNCTION_H_

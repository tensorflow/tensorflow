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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_FULLY_CONNECTED_FULLY_CONNECTED_CORE_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_FULLY_CONNECTED_FULLY_CONNECTED_CORE_H_

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

template <typename T>
class KernelCore;

template <>
class KernelCore<uint8_t> {
 public:
  static void run(uint8_t* output, const uint8_t* input, const uint8_t* weights,
                  const int32_t* sum_of_weights_factor,
                  int32_t sum_of_inputs_factor, int accum_depth,
                  int output_depth, int32_t output_offset,
                  int32_t output_multiplier, int output_shift,
                  int32_t activation_min, int32_t activation_max) {
    int32_t acc;
    for (int out_c = 0; out_c < output_depth; out_c++) {
      // Multiply and accumulate inputs and weights
      acc = *sum_of_weights_factor + sum_of_inputs_factor;
      for (int d = 0; d < accum_depth; ++d) {
        acc += weights[d] * input[d];
      }
      // Re-quantize and clamp
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = ActivationFunctionWithMinMax(acc, activation_min, activation_max);
      *output = static_cast<uint8_t>(acc);
      // Increment pointers
      output++;
      sum_of_weights_factor++;
      weights += accum_depth;
    }
  }
};

// No sign extension feature in assumed streamer for now...
template <>
class KernelCore<int8_t> {
 public:
  static void run(int8_t* output, const int8_t* input, const int8_t* weights,
                  const int32_t* sum_of_weights_factor,
                  int32_t sum_of_inputs_factor, int accum_depth,
                  int output_depth, int32_t output_offset,
                  int32_t output_multiplier, int output_shift,
                  int32_t activation_min, int32_t activation_max) {
    int32_t acc;
    for (int out_c = 0; out_c < output_depth; out_c++) {
      // Multiply and accumulate inputs and weights
      acc = *sum_of_weights_factor + sum_of_inputs_factor;

      for (int d = 0; d < accum_depth; ++d) {
        acc += weights[d] * input[d];
      }
      // Re-quantize and clamp
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = ActivationFunctionWithMinMax(acc, activation_min, activation_max);
      *output = static_cast<int8_t>(acc);
      // Increment pointers
      output++;
      sum_of_weights_factor++;
      weights += accum_depth;
    }
  }
};

}  // namespace
}  // namespace tflite

#endif // TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_FULLY_CONNECTED_FULLY_CONNECTED_CORE_H_

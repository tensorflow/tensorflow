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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ADD_N_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ADD_N_H_

#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

// T is expected to be either float or int.
template <typename T>
inline void AddN(const RuntimeShape& input_shape, const size_t num_inputs,
                 const T* const* input_data, T* output_data) {
  // All inputs and output should have the same shape, this is checked during
  // Prepare stage.
  const size_t size = input_shape.FlatSize();
  for (size_t i = 0; i < size; ++i) {
    T x = 0;
    for (size_t j = 0; j < num_inputs; ++j) {
      x += input_data[j][i];
    }
    output_data[i] = x;
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ADD_N_H_

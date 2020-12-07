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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_FILL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_FILL_H_

#include <cmath>

#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

template <typename T>
void Fill(const RuntimeShape& value_shape, const T* value_data,
          const RuntimeShape& output_shape, T* output_data) {
  TFLITE_DCHECK_EQ(value_shape.DimensionsCount(), 0);
  const int flat_size = output_shape.FlatSize();
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = *value_data;
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_FILL_H_

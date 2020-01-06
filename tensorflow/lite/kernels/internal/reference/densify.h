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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DENSIFY_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DENSIFY_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

template <typename T>
inline void Densify(const TfLiteSparsity* sparsity,
                    const RuntimeShape& input_shape, const T* input_data,
                    const RuntimeShape& output_shape, T* output_data) {
  const int flat_size = output_shape.FlatSize();
  memset(output_data, 0, sizeof(T) * flat_size);
}

}  // namespace reference_ops

}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DENSIFY_H_

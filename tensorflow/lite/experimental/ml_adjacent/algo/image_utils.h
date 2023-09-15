/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_IMAGE_UTILS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_IMAGE_UTILS_H_

#include "tensorflow/lite/experimental/ml_adjacent/lib.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace ml_adj {

inline void ConvertColorSpace(dim_t batches, dim_t height, dim_t width,
                              const float* input_data, float* output_data,
                              const float* kernel_data, dim_t kernel_size) {
  TFLITE_DCHECK(kernel_size == 9);

  const dim_t kNumChannels = 3;
  const dim_t output_num_pixels = batches * width * height;
  const float* src_data_prt = input_data;
  float* dst_data_prt = output_data;

  for (int i = 0; i < output_num_pixels; ++i) {
    dst_data_prt[0] = kernel_data[0] * src_data_prt[0] +
                      kernel_data[1] * src_data_prt[1] +
                      kernel_data[2] * src_data_prt[2];

    dst_data_prt[1] = kernel_data[3] * src_data_prt[0] +
                      kernel_data[4] * src_data_prt[1] +
                      kernel_data[5] * src_data_prt[2];

    dst_data_prt[2] = kernel_data[6] * src_data_prt[0] +
                      kernel_data[7] * src_data_prt[1] +
                      kernel_data[8] * src_data_prt[2];

    src_data_prt += kNumChannels;
    dst_data_prt += kNumChannels;
  }
}

}  // namespace ml_adj

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_IMAGE_UTILS_H_

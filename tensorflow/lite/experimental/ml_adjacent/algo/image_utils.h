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
  TFLITE_CHECK_EQ(kernel_size, 9);

  const dim_t num_channels = 3;
  const ind_t output_num_pixels = static_cast<ind_t>(batches) * width * height;
  const float* src_data_ptr = input_data;
  float* dst_data_ptr = output_data;

  for (ind_t i = 0; i < output_num_pixels; ++i) {
    dst_data_ptr[0] = kernel_data[0] * src_data_ptr[0] +
                      kernel_data[1] * src_data_ptr[1] +
                      kernel_data[2] * src_data_ptr[2];

    dst_data_ptr[1] = kernel_data[3] * src_data_ptr[0] +
                      kernel_data[4] * src_data_ptr[1] +
                      kernel_data[5] * src_data_ptr[2];

    dst_data_ptr[2] = kernel_data[6] * src_data_ptr[0] +
                      kernel_data[7] * src_data_ptr[1] +
                      kernel_data[8] * src_data_ptr[2];

    src_data_ptr += num_channels;
    dst_data_ptr += num_channels;
  }
}

}  // namespace ml_adj

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_ALGO_IMAGE_UTILS_H_

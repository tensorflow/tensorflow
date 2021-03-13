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

#include "tensorflow/lite/micro/micro_utils.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {

int ElementCount(const TfLiteIntArray& dims) {
  int result = 1;
  for (int i = 0; i < dims.size; ++i) {
    result *= dims.data[i];
  }
  return result;
}

void SignedSymmetricPerChannelQuantize(const float* values,
                                       TfLiteIntArray* dims,
                                       int quantized_dimension,
                                       int8_t* quantized_values,
                                       float* scaling_factors) {
  int input_size = ElementCount(*dims);
  int channel_count = dims->data[quantized_dimension];
  int per_channel_size = input_size / channel_count;

  int stride;
  int channel_stride;
  if (quantized_dimension == 0) {
    stride = 1;
    channel_stride = per_channel_size;
  } else if (quantized_dimension == 3) {
    stride = channel_count;
    channel_stride = 1;
  } else {
    TF_LITE_FATAL("quantized dimension must be 0 or 3");
  }

  // Calculate scales for each channel.
  for (int channel = 0; channel < channel_count; channel++) {
    float min = 0;
    float max = 0;

    for (int i = 0; i < per_channel_size; i++) {
      int idx = channel * channel_stride + i * stride;
      min = fminf(min, values[idx]);
      max = fmaxf(max, values[idx]);
    }
    scaling_factors[channel] =
        fmaxf(fabs(min), fabs(max)) / std::numeric_limits<int8_t>::max();
    for (int i = 0; i < per_channel_size; i++) {
      int idx = channel * channel_stride + i * stride;
      const int32_t quantized_value =
          static_cast<int32_t>(roundf(values[idx] / scaling_factors[channel]));
      // Clamp: just in case some odd numeric offset.
      quantized_values[idx] =
          fminf(std::numeric_limits<int8_t>::max(),
                fmaxf(std::numeric_limits<int8_t>::min() + 1, quantized_value));
    }
  }
}

}  // namespace tflite

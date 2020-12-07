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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_
#define TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "tensorflow/lite/c/common.h"

namespace tflite {

// Returns number of elements in the shape array.

int ElementCount(const TfLiteIntArray& dims);

// Converts a float value into a quantized value.  Note that large values (close
// to max int and min int) may see significant error due to a lack of floating
// point granularity for large values.
template <typename T>
T FloatToQuantizedType(const float value, const float scale, int zero_point) {
  int32_t result = round(value / scale) + zero_point;
  result =
      std::max(static_cast<int32_t>(std::numeric_limits<T>::min()), result);
  result =
      std::min(static_cast<int32_t>(std::numeric_limits<T>::max()), result);
  return result;
}

template <typename T>
T FloatToSymmetricQuantizedType(const float value, const float scale) {
  int32_t result = round(value / scale);
  result =
      std::max(static_cast<int32_t>(std::numeric_limits<T>::min() + 1), result);
  result =
      std::min(static_cast<int32_t>(std::numeric_limits<T>::max()), result);
  return result;
}

// Helper methods to quantize arrays of floats to the desired format.
//
// There are several key flavors of quantization in TfLite:
//        asymmetric symmetric  per channel
// int8_t  |     X    |    X    |     X      |
// uint8_t |     X    |    X    |            |
// int16_t |     X    |         |            |
// int32_t |          |    X    |     X      |
//
// The per-op quantization spec can be found here:
// https://www.tensorflow.org/lite/performance/quantization_spec
template <typename T>
void Quantize(const float* input, T* output, int num_elements, float scale,
              int zero_point) {
  for (int i = 0; i < num_elements; i++) {
    output[i] = FloatToQuantizedType<T>(input[i], scale, zero_point);
  }
}

template <typename T>
void SymmetricQuantize(const float* input, T* output, int num_elements,
                       float scale) {
  for (int i = 0; i < num_elements; i++) {
    output[i] = FloatToSymmetricQuantizedType<T>(input[i], scale);
  }
}

template <typename T>
void SymmetricPerChannelQuantize(const float* input, T* output,
                                 int num_elements, int num_channels,
                                 float* scales) {
  int elements_per_channel = num_elements / num_channels;
  for (int i = 0; i < num_channels; i++) {
    for (int j = 0; j < elements_per_channel; j++) {
      output[i * elements_per_channel + j] = FloatToSymmetricQuantizedType<T>(
          input[i * elements_per_channel + j], scales[i]);
    }
  }
}

void SignedSymmetricPerChannelQuantize(const float* values,
                                       TfLiteIntArray* dims,
                                       int quantized_dimension,
                                       int8_t* quantized_values,
                                       float* scaling_factor);

// Quantizes inputs based on the values provided, choosing the smallest range
// which includes all input values.
template <typename T>
void SymmetricQuantizeCalculateScales(const float* values, TfLiteIntArray* dims,
                                      T* output, float* scale) {
  int input_size = ElementCount(*dims);

  float min = 0;
  float max = 0;
  for (int i = 0; i < input_size; i++) {
    min = fminf(min, values[i]);
    max = fmaxf(max, values[i]);
  }
  *scale = fmaxf(std::abs(min), std::abs(max)) / std::numeric_limits<T>::max();
  for (int i = 0; i < input_size; i++) {
    const int32_t quantized_value =
        static_cast<int32_t>(roundf(values[i] / *scale));
    // Clamp: just in case some odd numeric offset.
    quantized_value = fminf(std::numeric_limits<T>::max(), quantized_value);
    quantized_value = fmaxf(std::numeric_limits<T>::min() + 1, quantized_value);
    output[i] = quantized_value;
  }
}

template <typename T>
void Dequantize(const T* values, const int size, const float scale,
                int zero_point, float* dequantized_values) {
  for (int i = 0; i < size; ++i) {
    dequantized_values[i] = (values[i] - zero_point) * scale;
  }
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_UTILS_H_

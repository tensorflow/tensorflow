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

#include "tensorflow/lite/experimental/micro/micro_utils.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>

#include "tensorflow/lite/c/c_api_internal.h"

namespace tflite {

namespace {

static const uint8_t kAsymmetricUInt8Min = 0;
static const uint8_t kAsymmetricUInt8Max = 255;
static const uint8_t kSymmetricUInt8Min = 1;
static const uint8_t kSymmetricUInt8Max = 255;
static const int8_t kAsymmetricInt8Min = -128;
static const int8_t kAsymmetricInt8Max = 127;
static const int kSymmetricInt8Scale = kAsymmetricInt8Max;

}  // namespace

int ElementCount(const TfLiteIntArray& dims) {
  int result = 1;
  for (int i = 0; i < dims.size; ++i) {
    result *= dims.data[i];
  }
  return result;
}

// Converts a float value into an unsigned eight-bit quantized value.
uint8_t FloatToAsymmetricQuantizedUInt8(const float value, const float scale,
                                        const int zero_point) {
  int32_t result = round(value / scale) + zero_point;
  if (result < kAsymmetricUInt8Min) {
    result = kAsymmetricUInt8Min;
  }
  if (result > kAsymmetricUInt8Max) {
    result = kAsymmetricUInt8Max;
  }
  return result;
}

uint8_t FloatToSymmetricQuantizedUInt8(const float value, const float scale) {
  int32_t result = round(value / scale);
  if (result < kSymmetricUInt8Min) {
    result = kSymmetricUInt8Min;
  }
  if (result > kSymmetricUInt8Max) {
    result = kSymmetricUInt8Max;
  }
  return result;
}

int8_t FloatToAsymmetricQuantizedInt8(const float value, const float scale,
                                      const int zero_point) {
  return FloatToAsymmetricQuantizedUInt8(value, scale,
                                         zero_point - kAsymmetricInt8Min) +
         kAsymmetricInt8Min;
}

int8_t FloatToSymmetricQuantizedInt8(const float value, const float scale) {
  return FloatToAsymmetricQuantizedInt8(value, scale, 0.0f);
}

int32_t FloatToSymmetricQuantizedInt32(const float value, const float scale) {
  float quantized = round(value / scale);
  if (quantized > INT_MAX) {
    quantized = INT_MAX;
  } else if (quantized < INT_MIN) {
    quantized = INT_MIN;
  }

  return static_cast<int>(quantized);
}

void AsymmetricQuantize(const float* input, int8_t* output, int num_elements,
                        float scale, int zero_point) {
  for (int i = 0; i < num_elements; i++) {
    output[i] = FloatToAsymmetricQuantizedInt8(input[i], scale, zero_point);
  }
}

void AsymmetricQuantize(const float* input, uint8_t* output, int num_elements,
                        float scale, int zero_point) {
  for (int i = 0; i < num_elements; i++) {
    output[i] = FloatToAsymmetricQuantizedUInt8(input[i], scale, zero_point);
  }
}

void SymmetricQuantize(const float* input, int32_t* output, int num_elements,
                       float scale) {
  for (int i = 0; i < num_elements; i++) {
    output[i] = FloatToSymmetricQuantizedInt32(input[i], scale);
  }
}

void SymmetricPerChannelQuantize(const float* input, int32_t* output,
                                 int num_elements, int num_channels,
                                 float* scales) {
  int elements_per_channel = num_elements / num_channels;
  for (int i = 0; i < num_channels; i++) {
    for (int j = 0; j < elements_per_channel; j++) {
      output[i * elements_per_channel + j] = FloatToSymmetricQuantizedInt32(
          input[i * elements_per_channel + j], scales[i]);
    }
  }
}

void SignedSymmetricPerChannelQuantize(const float* values,
                                       TfLiteIntArray* dims,
                                       int quantized_dimension,
                                       int8_t* quantized_values,
                                       float* scaling_factors) {
  int input_size = ElementCount(*dims);
  int channel_count = dims->data[quantized_dimension];
  int per_channel_size = input_size / channel_count;
  for (int channel = 0; channel < channel_count; channel++) {
    float min = 0;
    float max = 0;
    int stride = 1;
    for (int i = 0; i < quantized_dimension; i++) {
      stride *= dims->data[i];
    }
    int channel_stride = per_channel_size / stride;
    // Calculate scales for each channel.
    for (int i = 0; i < per_channel_size; i++) {
      int idx = channel * channel_stride + i * stride;
      min = fminf(min, values[idx]);
      max = fmaxf(max, values[idx]);
    }
    scaling_factors[channel] =
        fmaxf(fabs(min), fabs(max)) / kSymmetricInt8Scale;
    for (int i = 0; i < per_channel_size; i++) {
      int idx = channel * channel_stride + i * stride;
      const int32_t quantized_value =
          static_cast<int32_t>(roundf(values[idx] / scaling_factors[channel]));
      // Clamp: just in case some odd numeric offset.
      quantized_values[idx] = fminf(
          kSymmetricInt8Scale, fmaxf(-kSymmetricInt8Scale, quantized_value));
    }
  }
}

void SignedSymmetricQuantize(const float* values, TfLiteIntArray* dims,
                             int8_t* quantized_values, float* scaling_factor) {
  int input_size = ElementCount(*dims);

  float min = 0;
  float max = 0;
  for (int i = 0; i < input_size; i++) {
    min = fminf(min, values[i]);
    max = fmaxf(max, values[i]);
  }
  *scaling_factor = fmaxf(fabs(min), fabs(max)) / kSymmetricInt8Scale;
  for (int i = 0; i < input_size; i++) {
    const int32_t quantized_value =
        static_cast<int32_t>(roundf(values[i] / *scaling_factor));
    // Clamp: just in case some odd numeric offset.
    quantized_values[i] = fminf(kSymmetricInt8Scale,
                                fmaxf(-kSymmetricInt8Scale, quantized_value));
  }
}

void SymmetricQuantize(const float* values, TfLiteIntArray* dims,
                       uint8_t* quantized_values, float* scaling_factor) {
  SignedSymmetricQuantize(values, dims,
                          reinterpret_cast<int8_t*>(quantized_values),
                          scaling_factor);
}

void SymmetricDequantize(const int8_t* values, const int size,
                         const float dequantization_scale,
                         float* dequantized_values) {
  for (int i = 0; i < size; ++i) {
    dequantized_values[i] = values[i] * dequantization_scale;
  }
}

}  // namespace tflite

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_TESTING_TEST_UTILS_H_
#define TENSORFLOW_LITE_MICRO_TESTING_TEST_UTILS_H_

#include <cmath>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

// Note: These methods are deprecated, do not use.  See b/141332970.

// Derives the quantization range max from scaling factor and zero point.
template <typename T>
inline float MaxFromZeroPointScale(const int zero_point, const float scale) {
  return (std::numeric_limits<T>::max() - zero_point) * scale;
}

// Derives the quantization range min from scaling factor and zero point.
template <typename T>
inline float MinFromZeroPointScale(const int zero_point, const float scale) {
  return (std::numeric_limits<T>::min() - zero_point) * scale;
}

// Derives the quantization scaling factor from a min and max range.
template <typename T>
inline float ScaleFromMinMax(const float min, const float max) {
  return (max - min) /
         static_cast<float>((std::numeric_limits<T>::max() * 1.0) -
                            std::numeric_limits<T>::min());
}

// Derives the quantization zero point from a min and max range.
template <typename T>
inline int ZeroPointFromMinMax(const float min, const float max) {
  return static_cast<int>(std::numeric_limits<T>::min()) +
         static_cast<int>(-min / ScaleFromMinMax<T>(min, max) + 0.5f);
}

// Converts a float value into an unsigned eight-bit quantized value.
uint8_t F2Q(float value, float min, float max);

// Converts a float value into a signed eight-bit quantized value.
int8_t F2QS(const float value, const float min, const float max);

// Converts a float value into a signed thirty-two-bit quantized value.  Note
// that values close to max int and min int may see significant error due to
// a lack of floating point granularity for large values.
int32_t F2Q32(const float value, const float scale);

// TODO(b/141330728): Move this method elsewhere as part clean up.
void PopulateContext(TfLiteTensor* tensors, int tensors_size,
                     ErrorReporter* error_reporter, TfLiteContext* context);

TfLiteTensor CreateQuantizedTensor(const uint8_t* data, TfLiteIntArray* dims,
                                   float min, float max,
                                   bool is_variable = false);

TfLiteTensor CreateQuantizedTensor(const int8_t* data, TfLiteIntArray* dims,
                                   float min, float max,
                                   bool is_variable = false);

TfLiteTensor CreateQuantizedTensor(const float* data, uint8_t* quantized_data,
                                   TfLiteIntArray* dims,
                                   bool is_variable = false);

TfLiteTensor CreateQuantizedTensor(const float* data, int8_t* quantized_data,
                                   TfLiteIntArray* dims,
                                   bool is_variable = false);

TfLiteTensor CreateQuantizedTensor(const float* data, int16_t* quantized_data,
                                   TfLiteIntArray* dims,
                                   bool is_variable = false);

TfLiteTensor CreateQuantized32Tensor(const int32_t* data, TfLiteIntArray* dims,
                                     float scale, bool is_variable = false);

template <typename input_type = int32_t,
          TfLiteType tensor_input_type = kTfLiteInt32>
inline TfLiteTensor CreateTensor(const input_type* data, TfLiteIntArray* dims,
                                 bool is_variable = false) {
  TfLiteTensor result;
  result.type = tensor_input_type;
  result.data.raw = reinterpret_cast<char*>(const_cast<input_type*>(data));
  result.dims = dims;
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(input_type);
  result.is_variable = is_variable;
  return result;
}

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_TESTING_TEST_UTILS_H_

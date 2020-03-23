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
#include <initializer_list>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

// Note: These methods are deprecated, do not use.  See b/141332970.

// TODO(kreeger): Don't use this anymore in our tests. Optimized compiler
// settings can play with pointer placement on the stack (b/140130236).
inline TfLiteIntArray* IntArrayFromInitializer(
    std::initializer_list<int> int_initializer) {
  return IntArrayFromInts(int_initializer.begin());
}

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
  return (max - min) / ((std::numeric_limits<T>::max() * 1.0) -
                        std::numeric_limits<T>::min());
}

// Derives the quantization zero point from a min and max range.
template <typename T>
inline int ZeroPointFromMinMax(const float min, const float max) {
  return static_cast<int>(std::numeric_limits<T>::min()) +
         static_cast<int>(-min / ScaleFromMinMax<T>(min, max) + 0.5f);
}

// Converts a float value into an unsigned eight-bit quantized value.
inline uint8_t F2Q(const float value, const float min, const float max) {
  int32_t result = ZeroPointFromMinMax<uint8_t>(min, max) +
                   (value / ScaleFromMinMax<uint8_t>(min, max)) + 0.5f;
  if (result < std::numeric_limits<uint8_t>::min()) {
    result = std::numeric_limits<uint8_t>::min();
  }
  if (result > std::numeric_limits<uint8_t>::max()) {
    result = std::numeric_limits<uint8_t>::max();
  }
  return result;
}

// Converts a float value into a signed eight-bit quantized value.
inline int8_t F2QS(const float value, const float min, const float max) {
  return F2Q(value, min, max) + std::numeric_limits<int8_t>::min();
}

// Converts a float value into a signed thirty-two-bit quantized value.  Note
// that values close to max int and min int may see significant error due to
// a lack of floating point granularity for large values.
inline int32_t F2Q32(const float value, const float scale) {
  double quantized = value / scale;
  if (quantized > std::numeric_limits<int32_t>::max()) {
    quantized = std::numeric_limits<int32_t>::max();
  } else if (quantized < std::numeric_limits<int32_t>::min()) {
    quantized = std::numeric_limits<int32_t>::min();
  }
  return static_cast<int>(quantized);
}

// TODO(b/141330728): Move this method elsewhere as part clean up.
void PopulateContext(TfLiteTensor* tensors, int tensors_size,
                     ErrorReporter* error_reporter, TfLiteContext* context);

inline TfLiteTensor CreateFloatTensor(std::initializer_list<float> data,
                                      TfLiteIntArray* dims, const char* name,
                                      bool is_variable = false) {
  return CreateFloatTensor(data.begin(), dims, name, is_variable);
}

inline TfLiteTensor CreateBoolTensor(std::initializer_list<bool> data,
                                     TfLiteIntArray* dims, const char* name,
                                     bool is_variable = false) {
  return CreateBoolTensor(data.begin(), dims, name, is_variable);
}

inline TfLiteTensor CreateQuantizedTensor(const uint8_t* data,
                                          TfLiteIntArray* dims,
                                          const char* name, float min,
                                          float max, bool is_variable = false) {
  TfLiteTensor result;
  result.type = kTfLiteUInt8;
  result.data.uint8 = const_cast<uint8_t*>(data);
  result.dims = dims;
  result.params = {ScaleFromMinMax<uint8_t>(min, max),
                   ZeroPointFromMinMax<uint8_t>(min, max)};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(uint8_t);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = false;
  return result;
}

inline TfLiteTensor CreateQuantizedTensor(std::initializer_list<uint8_t> data,
                                          TfLiteIntArray* dims,
                                          const char* name, float min,
                                          float max, bool is_variable = false) {
  return CreateQuantizedTensor(data.begin(), dims, name, min, max, is_variable);
}

inline TfLiteTensor CreateQuantizedTensor(const int8_t* data,
                                          TfLiteIntArray* dims,
                                          const char* name, float min,
                                          float max, bool is_variable = false) {
  TfLiteTensor result;
  result.type = kTfLiteInt8;
  result.data.int8 = const_cast<int8_t*>(data);
  result.dims = dims;
  result.params = {ScaleFromMinMax<int8_t>(min, max),
                   ZeroPointFromMinMax<int8_t>(min, max)};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(int8_t);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = is_variable;
  return result;
}

inline TfLiteTensor CreateQuantizedTensor(std::initializer_list<int8_t> data,
                                          TfLiteIntArray* dims,
                                          const char* name, float min,
                                          float max, bool is_variable = false) {
  return CreateQuantizedTensor(data.begin(), dims, name, min, max, is_variable);
}

inline TfLiteTensor CreateQuantizedTensor(float* data, uint8_t* quantized_data,
                                          TfLiteIntArray* dims,
                                          const char* name,
                                          bool is_variable = false) {
  TfLiteTensor result;
  SymmetricQuantize(data, dims, quantized_data, &result.params.scale);
  result.data.uint8 = quantized_data;
  result.type = kTfLiteUInt8;
  result.dims = dims;
  result.params.zero_point = 128;
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(uint8_t);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = is_variable;
  return result;
}

inline TfLiteTensor CreateQuantizedTensor(float* data, int8_t* quantized_data,
                                          TfLiteIntArray* dims,
                                          const char* name,
                                          bool is_variable = false) {
  TfLiteTensor result;
  SignedSymmetricQuantize(data, dims, quantized_data, &result.params.scale);
  result.data.int8 = quantized_data;
  result.type = kTfLiteInt8;
  result.dims = dims;
  result.params.zero_point = 0;
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(int8_t);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = is_variable;
  return result;
}

inline TfLiteTensor CreateQuantizedTensor(float* data, int16_t* quantized_data,
                                          TfLiteIntArray* dims,
                                          const char* name,
                                          bool is_variable = false) {
  TfLiteTensor result;
  SignedSymmetricQuantize(data, dims, quantized_data, &result.params.scale);
  result.data.i16 = quantized_data;
  result.type = kTfLiteInt16;
  result.dims = dims;
  result.params.zero_point = 0;
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(int16_t);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = is_variable;
  return result;
}

inline TfLiteTensor CreateQuantized32Tensor(const int32_t* data,
                                            TfLiteIntArray* dims,
                                            const char* name, float scale,
                                            bool is_variable = false) {
  TfLiteTensor result;
  result.type = kTfLiteInt32;
  result.data.i32 = const_cast<int32_t*>(data);
  result.dims = dims;
  // Quantized int32 tensors always have a zero point of 0, since the range of
  // int32 values is large, and because zero point costs extra cycles during
  // processing.
  result.params = {scale, 0};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(int32_t);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = is_variable;
  return result;
}

inline TfLiteTensor CreateQuantized32Tensor(std::initializer_list<int32_t> data,
                                            TfLiteIntArray* dims,
                                            const char* name, float scale,
                                            bool is_variable = false) {
  return CreateQuantized32Tensor(data.begin(), dims, name, scale, is_variable);
}

template <typename input_type = int32_t,
          TfLiteType tensor_input_type = kTfLiteInt32>
inline TfLiteTensor CreateTensor(const input_type* data, TfLiteIntArray* dims,
                                 const char* name, bool is_variable = false) {
  TfLiteTensor result;
  result.type = tensor_input_type;
  result.data.raw = reinterpret_cast<char*>(const_cast<input_type*>(data));
  result.dims = dims;
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(input_type);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = is_variable;
  return result;
}

template <typename input_type = int32_t,
          TfLiteType tensor_input_type = kTfLiteInt32>
inline TfLiteTensor CreateTensor(std::initializer_list<input_type> data,
                                 TfLiteIntArray* dims, const char* name,
                                 bool is_variable = false) {
  return CreateTensor<input_type, tensor_input_type>(data.begin(), dims, name,
                                                     is_variable);
}

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_TESTING_TEST_UTILS_H_

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_TESTING_TEST_UTILS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_TESTING_TEST_UTILS_H_

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

// TODO(kreeger): Move to common code!
inline void SignedSymmetricQuantizeFloats(const float* values, const int size,
                                          float* min_value, float* max_value,
                                          int8_t* quantized_values,
                                          float* scaling_factor) {
  // First, find min/max in values
  *min_value = values[0];
  *max_value = values[0];
  for (int i = 1; i < size; ++i) {
    if (values[i] < *min_value) {
      *min_value = values[i];
    }
    if (values[i] > *max_value) {
      *max_value = values[i];
    }
  }
  const float range = fmaxf(fabsf(*min_value), fabsf(*max_value));
  if (range == 0.0f) {
    for (int i = 0; i < size; ++i) {
      quantized_values[i] = 0;
    }
    *scaling_factor = 1;
    return;
  }

  const int kScale = 127;
  *scaling_factor = range / kScale;
  const float scaling_factor_inv = kScale / range;
  for (int i = 0; i < size; ++i) {
    const int32_t quantized_value =
        static_cast<int32_t>(roundf(values[i] * scaling_factor_inv));
    // Clamp: just in case some odd numeric offset.
    quantized_values[i] = fminf(kScale, fmaxf(-kScale, quantized_value));
  }
}

inline void SymmetricQuantizeFloats(const float* values, const int size,
                                    float* min_value, float* max_value,
                                    uint8_t* quantized_values,
                                    float* scaling_factor) {
  SignedSymmetricQuantizeFloats(values, size, min_value, max_value,
                                reinterpret_cast<int8_t*>(quantized_values),
                                scaling_factor);
}

// How many elements are in the array with this shape.
inline int ElementCount(const TfLiteIntArray& dims) {
  int result = 1;
  for (int i = 0; i < dims.size; ++i) {
    result *= dims.data[i];
  }
  return result;
}

// Wrapper to forward kernel errors to the interpreter's error reporter.
inline void ReportOpError(struct TfLiteContext* context, const char* format,
                          ...) {
  ErrorReporter* error_reporter = static_cast<ErrorReporter*>(context->impl_);
  va_list args;
  va_start(args, format);
  error_reporter->Report(format, args);
  va_end(args);
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

inline void PopulateContext(TfLiteTensor* tensors, int tensors_size,
                            TfLiteContext* context) {
  context->tensors_size = tensors_size;
  context->tensors = tensors;
  context->impl_ = static_cast<void*>(micro_test::reporter);
  context->GetExecutionPlan = nullptr;
  context->ResizeTensor = nullptr;
  context->ReportError = ReportOpError;
  context->AddTensors = nullptr;
  context->GetNodeAndRegistration = nullptr;
  context->ReplaceNodeSubsetsWithDelegateKernels = nullptr;
  context->recommended_num_threads = 1;
  context->GetExternalContext = nullptr;
  context->SetExternalContext = nullptr;

  for (int i = 0; i < tensors_size; ++i) {
    if (context->tensors[i].is_variable) {
      tflite::ResetVariableTensor(&context->tensors[i]);
    }
  }
}

inline TfLiteIntArray* IntArrayFromInts(const int* int_array) {
  return const_cast<TfLiteIntArray*>(
      reinterpret_cast<const TfLiteIntArray*>(int_array));
}

// TODO(kreeger): Don't use this anymore in our tests. Optimized compiler
// settings can play with pointer placement on the stack (b/140130236).
inline TfLiteIntArray* IntArrayFromInitializer(
    std::initializer_list<int> int_initializer) {
  return IntArrayFromInts(int_initializer.begin());
}

inline TfLiteTensor CreateFloatTensor(const float* data, TfLiteIntArray* dims,
                                      const char* name,
                                      bool is_variable = false) {
  TfLiteTensor result;
  result.type = kTfLiteFloat32;
  result.data.f = const_cast<float*>(data);
  result.dims = dims;
  result.params = {};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(float);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = is_variable;
  return result;
}

inline TfLiteTensor CreateFloatTensor(std::initializer_list<float> data,
                                      TfLiteIntArray* dims, const char* name,
                                      bool is_variable = false) {
  return CreateFloatTensor(data.begin(), dims, name, is_variable);
}

inline void PopulateFloatTensor(TfLiteTensor* tensor, float* begin,
                                float* end) {
  float* p = begin;
  float* v = tensor->data.f;
  while (p != end) {
    *v++ = *p++;
  }
}

inline TfLiteTensor CreateBoolTensor(const bool* data, TfLiteIntArray* dims,
                                     const char* name,
                                     bool is_variable = false) {
  TfLiteTensor result;
  result.type = kTfLiteBool;
  result.data.b = const_cast<bool*>(data);
  result.dims = dims;
  result.params = {};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(bool);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = is_variable;
  return result;
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
  float min, max;
  SymmetricQuantizeFloats(data, ElementCount(*dims), &min, &max, quantized_data,
                          &result.params.scale);
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
  float min, max;
  SignedSymmetricQuantizeFloats(data, ElementCount(*dims), &min, &max,
                                quantized_data, &result.params.scale);
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

// Do a simple string comparison for testing purposes, without requiring the
// standard C library.
inline int TestStrcmp(const char* a, const char* b) {
  if ((a == nullptr) || (b == nullptr)) {
    return -1;
  }
  while ((*a != 0) && (*a == *b)) {
    a++;
    b++;
  }
  return *(const unsigned char*)a - *(const unsigned char*)b;
}

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_TESTING_TEST_UTILS_H_

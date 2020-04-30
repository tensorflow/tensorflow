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

#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {

uint8_t F2Q(float value, float min, float max) {
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
int8_t F2QS(float value, float min, float max) {
  return F2Q(value, min, max) + std::numeric_limits<int8_t>::min();
}

int32_t F2Q32(float value, float scale) {
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
                     ErrorReporter* error_reporter, TfLiteContext* context) {
  context->tensors_size = tensors_size;
  context->tensors = tensors;
  context->impl_ = static_cast<void*>(error_reporter);
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
      ResetVariableTensor(&context->tensors[i]);
    }
  }
}

TfLiteTensor CreateFloatTensor(std::initializer_list<float> data,
                               TfLiteIntArray* dims, const char* name,
                               bool is_variable) {
  return CreateFloatTensor(data.begin(), dims, name, is_variable);
}

TfLiteTensor CreateBoolTensor(std::initializer_list<bool> data,
                              TfLiteIntArray* dims, const char* name,
                              bool is_variable) {
  return CreateBoolTensor(data.begin(), dims, name, is_variable);
}

TfLiteTensor CreateQuantizedTensor(const uint8_t* data, TfLiteIntArray* dims,
                                   const char* name, float min, float max,
                                   bool is_variable) {
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

TfLiteTensor CreateQuantizedTensor(std::initializer_list<uint8_t> data,
                                   TfLiteIntArray* dims, const char* name,
                                   float min, float max, bool is_variable) {
  return CreateQuantizedTensor(data.begin(), dims, name, min, max, is_variable);
}

TfLiteTensor CreateQuantizedTensor(const int8_t* data, TfLiteIntArray* dims,
                                   const char* name, float min, float max,
                                   bool is_variable) {
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

TfLiteTensor CreateQuantizedTensor(std::initializer_list<int8_t> data,
                                   TfLiteIntArray* dims, const char* name,
                                   float min, float max, bool is_variable) {
  return CreateQuantizedTensor(data.begin(), dims, name, min, max, is_variable);
}

TfLiteTensor CreateQuantizedTensor(float* data, uint8_t* quantized_data,
                                   TfLiteIntArray* dims, const char* name,
                                   bool is_variable) {
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

TfLiteTensor CreateQuantizedTensor(float* data, int8_t* quantized_data,
                                   TfLiteIntArray* dims, const char* name,
                                   bool is_variable) {
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

TfLiteTensor CreateQuantizedTensor(float* data, int16_t* quantized_data,
                                   TfLiteIntArray* dims, const char* name,
                                   bool is_variable) {
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

TfLiteTensor CreateQuantized32Tensor(const int32_t* data, TfLiteIntArray* dims,
                                     const char* name, float scale,
                                     bool is_variable) {
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

TfLiteTensor CreateQuantized32Tensor(std::initializer_list<int32_t> data,
                                     TfLiteIntArray* dims, const char* name,
                                     float scale, bool is_variable) {
  return CreateQuantized32Tensor(data.begin(), dims, name, scale, is_variable);
}

}  // namespace testing
}  // namespace tflite

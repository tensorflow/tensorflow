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

#include "tensorflow/lite/experimental/micro/memory_helpers.h"

#include <cstdint>

#include "tensorflow/lite/core/api/flatbuffer_conversions.h"

namespace tflite {

uint8_t* AlignPointerUp(uint8_t* data, size_t alignment) {
  std::uintptr_t data_as_uintptr_t = reinterpret_cast<std::uintptr_t>(data);
  uint8_t* aligned_result = reinterpret_cast<uint8_t*>(
      ((data_as_uintptr_t + (alignment - 1)) / alignment) * alignment);
  return aligned_result;
}

uint8_t* AlignPointerDown(uint8_t* data, size_t alignment) {
  std::uintptr_t data_as_uintptr_t = reinterpret_cast<std::uintptr_t>(data);
  uint8_t* aligned_result =
      reinterpret_cast<uint8_t*>((data_as_uintptr_t / alignment) * alignment);
  return aligned_result;
}

size_t AlignSizeUp(size_t size, size_t alignment) {
  size_t aligned_size = (((size + (alignment - 1)) / alignment) * alignment);
  return aligned_size;
}

TfLiteStatus TfLiteTypeSizeOf(TfLiteType type, size_t* size,
                              ErrorReporter* reporter) {
  switch (type) {
    case kTfLiteFloat32:
      *size = sizeof(float);
      break;
    case kTfLiteInt16:
      *size = sizeof(int16_t);
      break;
    case kTfLiteInt32:
      *size = sizeof(int32_t);
      break;
    case kTfLiteUInt8:
      *size = sizeof(uint8_t);
      break;
    case kTfLiteInt8:
      *size = sizeof(int8_t);
      break;
    case kTfLiteInt64:
      *size = sizeof(int64_t);
      break;
    case kTfLiteBool:
      *size = sizeof(bool);
      break;
    case kTfLiteComplex64:
      *size = sizeof(float) * 2;
      break;
    default:
      reporter->Report("Type %s (%d) not is not supported",
                       TfLiteTypeGetName(type), type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus BytesRequiredForTensor(const tflite::Tensor& flatbuffer_tensor,
                                    size_t* bytes, size_t* type_size,
                                    ErrorReporter* error_reporter) {
  int element_count = 1;
  for (size_t n = 0; n < flatbuffer_tensor.shape()->Length(); ++n) {
    element_count *= flatbuffer_tensor.shape()->Get(n);
  }

  TfLiteType tf_lite_type;
  TF_LITE_ENSURE_STATUS(ConvertTensorType(flatbuffer_tensor.type(),
                                          &tf_lite_type, error_reporter));
  TF_LITE_ENSURE_STATUS(
      TfLiteTypeSizeOf(tf_lite_type, type_size, error_reporter));
  *bytes = element_count * (*type_size);
  return kTfLiteOk;
}

}  // namespace tflite

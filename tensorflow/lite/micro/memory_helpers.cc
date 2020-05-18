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

#include "tensorflow/lite/micro/memory_helpers.h"

#include <cstddef>
#include <cstdint>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

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
  // If flatbuffer_tensor.shape == nullptr, then flatbuffer_tensor is a scalar
  // so has 1 element.
  if (flatbuffer_tensor.shape() != nullptr) {
    for (size_t n = 0; n < flatbuffer_tensor.shape()->Length(); ++n) {
      element_count *= flatbuffer_tensor.shape()->Get(n);
    }
  }

  TfLiteType tf_lite_type;
  TF_LITE_ENSURE_STATUS(ConvertTensorType(flatbuffer_tensor.type(),
                                          &tf_lite_type, error_reporter));
  TF_LITE_ENSURE_STATUS(
      TfLiteTypeSizeOf(tf_lite_type, type_size, error_reporter));
  *bytes = element_count * (*type_size);
  return kTfLiteOk;
}

TfLiteStatus AllocateOutputDimensionsFromInput(TfLiteContext* context, const TfLiteTensor* input1,
                                               const TfLiteTensor* input2, TfLiteTensor* output) {
    int size = 1, i = 0;
    const TfLiteTensor* input = nullptr;

    TF_LITE_ENSURE(context, input1->dims != nullptr);
    TF_LITE_ENSURE(context, input2->dims != nullptr);
    TF_LITE_ENSURE(context, output->dims->size == 0);

    input = input1->dims->size > input2->dims->size ? input1 : input2;
    TF_LITE_ENSURE(context, output->type == input->type);

    const int dimensions_count = tflite::GetTensorShape(input).DimensionsCount();
    for (i = 0; i < dimensions_count; i++) {
      size *= input->dims->data[i];
    }
    output->bytes += size;

    TF_LITE_ENSURE_STATUS(context->AllocatePersistentBuffer(
        context, TfLiteIntArrayGetSizeInBytes(size),
      reinterpret_cast<void**>(&output->dims)));

    output->dims->size = input->dims->size;
    for (i = 0; i < dimensions_count; i++) {
      output->dims->data[i] = input->dims->data[i];
    }

    return kTfLiteOk;
}

}  // namespace tflite

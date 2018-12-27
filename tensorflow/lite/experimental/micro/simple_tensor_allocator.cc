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

#include "tensorflow/lite/experimental/micro/simple_tensor_allocator.h"

#include "tensorflow/lite/core/api/flatbuffer_conversions.h"

namespace tflite {
namespace {

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
      reporter->Report(
          "Only float32, int16, int32, int64, uint8, bool, complex64 "
          "supported currently.");
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus BytesRequired(const tflite::Tensor& flatbuffer_tensor,
                           size_t dims_size, size_t* bytes, size_t* type_size,
                           ErrorReporter* error_reporter) {
  TfLiteType tf_lite_type;
  TF_LITE_ENSURE_STATUS(ConvertTensorType(flatbuffer_tensor.type(),
                                          &tf_lite_type, error_reporter));
  TF_LITE_ENSURE_STATUS(
      TfLiteTypeSizeOf(tf_lite_type, type_size, error_reporter));
  *bytes = dims_size * (*type_size);
  return kTfLiteOk;
}

uint8_t* AlignPointerRoundUp(uint8_t* data, size_t alignment) {
  size_t data_as_size_t = reinterpret_cast<size_t>(data);
  uint8_t* aligned_result = reinterpret_cast<uint8_t*>(
      ((data_as_size_t + (alignment - 1)) / alignment) * alignment);
  return aligned_result;
}

}  // namespace

TfLiteStatus SimpleTensorAllocator::AllocateTensor(
    const tflite::Tensor& flatbuffer_tensor, int create_before,
    int destroy_after,
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    ErrorReporter* error_reporter, TfLiteTensor* result) {
  TF_LITE_ENSURE_STATUS(ConvertTensorType(flatbuffer_tensor.type(),
                                          &result->type, error_reporter));
  result->is_variable = flatbuffer_tensor.is_variable();

  result->data.raw = nullptr;
  result->bytes = 0;
  if (auto* buffer = (*buffers)[flatbuffer_tensor.buffer()]) {
    if (auto* array = buffer->data()) {
      if (size_t array_size = array->size()) {
        result->data.raw =
            const_cast<char*>(reinterpret_cast<const char*>(array->data()));
        size_t type_size;
        TF_LITE_ENSURE_STATUS(BytesRequired(flatbuffer_tensor, array_size,
                                            &result->bytes, &type_size,
                                            error_reporter));
      }
    }
  }
  if (result->data.raw) {
    result->allocation_type = kTfLiteMmapRo;
  } else {
    int data_size = 1;
    for (int n = 0; n < flatbuffer_tensor.shape()->Length(); ++n) {
      data_size *= flatbuffer_tensor.shape()->Get(n);
    }
    size_t type_size;
    TF_LITE_ENSURE_STATUS(BytesRequired(flatbuffer_tensor, data_size,
                                        &result->bytes, &type_size,
                                        error_reporter));
    result->data.raw =
        reinterpret_cast<char*>(AllocateMemory(result->bytes, type_size));
    if (result->data.raw == nullptr) {
      const char* tensor_name = flatbuffer_tensor.name()->c_str();
      if (tensor_name == nullptr) {
        tensor_name = "<None>";
      }
      error_reporter->Report(
          "Couldn't allocate memory for tensor '%s', wanted %d bytes but only "
          "%d were available",
          tensor_name, result->bytes, (data_size_max_ - data_size_));
      return kTfLiteError;
    }
    result->allocation_type = kTfLiteArenaRw;
  }
  result->dims = reinterpret_cast<TfLiteIntArray*>(AllocateMemory(
      sizeof(int) * (flatbuffer_tensor.shape()->Length() + 1), sizeof(int)));
  result->dims->size = flatbuffer_tensor.shape()->Length();
  for (int n = 0; n < flatbuffer_tensor.shape()->Length(); ++n) {
    result->dims->data[n] = flatbuffer_tensor.shape()->Get(n);
  }
  if (flatbuffer_tensor.quantization()) {
    result->params.scale = flatbuffer_tensor.quantization()->scale()->Get(0);
    result->params.zero_point =
        flatbuffer_tensor.quantization()->zero_point()->Get(0);
  }
  result->allocation = nullptr;
  if (flatbuffer_tensor.name()) {
    result->name = flatbuffer_tensor.name()->c_str();
  } else {
    result->name = "<No name>";
  }
  result->delegate = nullptr;
  result->buffer_handle = 0;
  result->data_is_stale = false;
  return kTfLiteOk;
}

uint8_t* SimpleTensorAllocator::AllocateMemory(size_t size, size_t alignment) {
  uint8_t* current_data = data_ + data_size_;
  uint8_t* aligned_result = AlignPointerRoundUp(current_data, alignment);
  uint8_t* next_free = aligned_result + size;
  size_t aligned_size = (next_free - current_data);
  if ((data_size_ + aligned_size) > data_size_max_) {
    // TODO(petewarden): Add error reporting beyond returning null!
    return nullptr;
  }
  data_size_ += aligned_size;
  return aligned_result;
}

}  // namespace tflite

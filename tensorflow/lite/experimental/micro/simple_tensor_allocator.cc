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

#include "tensorflow/lite/experimental/micro/allocator_utils.h"
#include "tensorflow/lite/experimental/micro/simple_tensor_allocator.h"

namespace tflite {

TfLiteStatus SimpleTensorAllocator::AllocateBuffer(TfLiteTensor *tensor,
                                                   ErrorReporter *error_reporter) {

  size_t type_size;
  TF_LITE_ENSURE_STATUS(TfLiteTypeSizeOf(tensor->type, &type_size, error_reporter));

  uint8_t *allocated_memory;
  if (this->AllocateStaticMemory(tensor->bytes, type_size,
                                 error_reporter, &allocated_memory) != kTfLiteOk) {
    error_reporter->Report("Failed to allocate memory for tensor '%s'", tensor->name);
    return kTfLiteError;
  }

  tensor->data.uint8 = allocated_memory;
  return kTfLiteOk;
}

TfLiteStatus SimpleTensorAllocator::DeallocateBuffer(TfLiteTensor *tensor,
                                                     ErrorReporter *error_reporter) {
  return kTfLiteOk;
}

TfLiteStatus SimpleTensorAllocator::AllocateStaticMemory(size_t size, size_t alignment,
        ErrorReporter* error_reporter, uint8_t **output) {
  uint8_t* current_data = data_ + data_size_;
  uint8_t* aligned_result = AlignPointerRoundUp(current_data, alignment);
  uint8_t* next_free = aligned_result + size;
  size_t aligned_size = (next_free - current_data);
  if ((data_size_ + aligned_size) > data_size_max_) {
    error_reporter->Report("Failed to allocate memory: wanted %d bytes,"
                           " but only %d were available", size, data_size_max_ - data_size_);
    *output = nullptr;
    return kTfLiteError;
  }
  data_size_ += aligned_size;
  *output = aligned_result;
  return kTfLiteOk;
}

}  // namespace tflite

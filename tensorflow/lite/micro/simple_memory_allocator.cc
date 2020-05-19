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

#include "tensorflow/lite/micro/simple_memory_allocator.h"

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/memory_helpers.h"

namespace tflite {

SimpleMemoryAllocator* CreateInPlaceSimpleMemoryAllocator(
    ErrorReporter* error_reporter, uint8_t* buffer, size_t buffer_size) {
  SimpleMemoryAllocator tmp =
      SimpleMemoryAllocator(error_reporter, buffer, buffer_size);
  SimpleMemoryAllocator* in_place_allocator =
      reinterpret_cast<SimpleMemoryAllocator*>(tmp.AllocateFromTail(
          sizeof(SimpleMemoryAllocator), alignof(SimpleMemoryAllocator)));
  *in_place_allocator = tmp;
  return in_place_allocator;
}

uint8_t* SimpleMemoryAllocator::AllocateFromHead(size_t size,
                                                 size_t alignment) {
  uint8_t* const aligned_result = AlignPointerUp(head_, alignment);
  const size_t available_memory = tail_ - aligned_result;
  if (available_memory < size) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Failed to allocate memory. Requested: %u, available %u, missing: %u",
        size, available_memory, size - available_memory);
    return nullptr;
  }
  head_ = aligned_result + size;
  return aligned_result;
}

uint8_t* SimpleMemoryAllocator::AllocateFromTail(size_t size,
                                                 size_t alignment) {
  uint8_t* const aligned_result = AlignPointerDown(tail_ - size, alignment);
  if (aligned_result < head_) {
    const size_t missing_memory = head_ - aligned_result;
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Failed to allocate memory. Requested: %u, available %u, missing: %u",
        size, size - missing_memory, missing_memory);
    return nullptr;
  }
  tail_ = aligned_result;
  return aligned_result;
}

}  // namespace tflite

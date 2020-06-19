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

#include "tensorflow/lite/micro/recording_simple_memory_allocator.h"

#include <new>

#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {

RecordingSimpleMemoryAllocator::RecordingSimpleMemoryAllocator(
    ErrorReporter* error_reporter, uint8_t* buffer_head, size_t buffer_size)
    : SimpleMemoryAllocator(error_reporter, buffer_head, buffer_size),
      requested_bytes_(0),
      used_bytes_(0),
      alloc_count_(0) {}

RecordingSimpleMemoryAllocator::~RecordingSimpleMemoryAllocator() {}

RecordingSimpleMemoryAllocator* RecordingSimpleMemoryAllocator::Create(
    ErrorReporter* error_reporter, uint8_t* buffer_head, size_t buffer_size) {
  TFLITE_DCHECK(error_reporter != nullptr);
  TFLITE_DCHECK(buffer_head != nullptr);
  RecordingSimpleMemoryAllocator tmp =
      RecordingSimpleMemoryAllocator(error_reporter, buffer_head, buffer_size);

  uint8_t* allocator_buffer =
      tmp.AllocateFromTail(sizeof(RecordingSimpleMemoryAllocator),
                           alignof(RecordingSimpleMemoryAllocator));
  // Use the default copy constructor to populate internal states.
  return new (allocator_buffer) RecordingSimpleMemoryAllocator(tmp);
}

size_t RecordingSimpleMemoryAllocator::GetRequestedBytes() const {
  return requested_bytes_;
}

size_t RecordingSimpleMemoryAllocator::GetUsedBytes() const {
  return used_bytes_;
}

size_t RecordingSimpleMemoryAllocator::GetAllocatedCount() const {
  return alloc_count_;
}

uint8_t* RecordingSimpleMemoryAllocator::AllocateFromHead(size_t size,
                                                          size_t alignment) {
  const uint8_t* previous_head = GetHead();
  uint8_t* result = SimpleMemoryAllocator::AllocateFromHead(size, alignment);
  if (result != nullptr) {
    used_bytes_ += GetHead() - previous_head;
    requested_bytes_ += size;
    alloc_count_++;
  }
  return result;
}

uint8_t* RecordingSimpleMemoryAllocator::AllocateFromTail(size_t size,
                                                          size_t alignment) {
  const uint8_t* previous_tail = GetTail();
  uint8_t* result = SimpleMemoryAllocator::AllocateFromTail(size, alignment);
  if (result != nullptr) {
    used_bytes_ += previous_tail - GetTail();
    requested_bytes_ += size;
    alloc_count_++;
  }
  return result;
}

}  // namespace tflite

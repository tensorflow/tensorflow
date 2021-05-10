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

#ifndef TENSORFLOW_LITE_MICRO_RECORDING_SIMPLE_MEMORY_ALLOCATOR_H_
#define TENSORFLOW_LITE_MICRO_RECORDING_SIMPLE_MEMORY_ALLOCATOR_H_

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"

namespace tflite {

// Utility class used to log allocations of a SimpleMemoryAllocator. Should only
// be used in debug/evaluation settings or unit tests to evaluate allocation
// usage.
class RecordingSimpleMemoryAllocator : public SimpleMemoryAllocator {
 public:
  RecordingSimpleMemoryAllocator(ErrorReporter* error_reporter,
                                 uint8_t* buffer_head, size_t buffer_size);
  // TODO(b/157615197): Cleanup constructors/destructor and use factory
  // functions.
  ~RecordingSimpleMemoryAllocator() override;

  static RecordingSimpleMemoryAllocator* Create(ErrorReporter* error_reporter,
                                                uint8_t* buffer_head,
                                                size_t buffer_size);

  // Returns the number of bytes requested from the head or tail.
  size_t GetRequestedBytes() const;

  // Returns the number of bytes actually allocated from the head or tail. This
  // value will be >= to the number of requested bytes due to padding and
  // alignment.
  size_t GetUsedBytes() const;

  // Returns the number of alloc calls from the head or tail.
  size_t GetAllocatedCount() const;

  TfLiteStatus SetHeadBufferSize(size_t size, size_t alignment) override;
  uint8_t* AllocateFromTail(size_t size, size_t alignment) override;

 private:
  size_t requested_head_bytes_;
  size_t requested_tail_bytes_;
  size_t used_bytes_;
  size_t alloc_count_;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_RECORDING_SIMPLE_MEMORY_ALLOCATOR_H_

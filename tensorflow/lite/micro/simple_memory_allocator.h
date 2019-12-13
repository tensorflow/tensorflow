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

#ifndef TENSORFLOW_LITE_MICRO_SIMPLE_MEMORY_ALLOCATOR_H_
#define TENSORFLOW_LITE_MICRO_SIMPLE_MEMORY_ALLOCATOR_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// TODO(petewarden): This allocator never frees up or reuses  any memory, even
// though we have enough information about lifetimes of the tensors to do so.
// This makes it pretty wasteful, so we should use a more intelligent method.
class SimpleMemoryAllocator {
 public:
  SimpleMemoryAllocator(uint8_t* buffer, size_t buffer_size)
      : data_size_max_(buffer_size), data_(buffer) {}

  // Allocates memory starting at the end of the arena (highest address and
  // moving downwards, so that tensor buffers can be allocated from the start
  // in ascending order.
  uint8_t* AllocateFromTail(size_t size, size_t alignment);

  int GetDataSize() const { return data_size_; }

  // Child allocator is something like a temporary allocator. Memory allocated
  // by the child allocator will be freed once the child allocator is
  // deallocated. Child allocator could be cascaded to have for example
  // grandchild allocator. But at any given time, only the latest child
  // allocator can be used. All its ancestors will be locked to avoid memory
  // corruption. Locked means that the allocator can't allocate memory.
  // WARNING: Parent allocator needs to live longer than the child allocator.
  SimpleMemoryAllocator CreateChildAllocator();

  // Unlocks parent allocator when the child allocator is deconstructed.
  ~SimpleMemoryAllocator();

 private:
  int data_size_ = 0;
  size_t data_size_max_;
  uint8_t* data_;
  SimpleMemoryAllocator* parent_allocator_ = nullptr;
  // The allocator is locaked if it has a child.
  bool has_child_allocator_ = false;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_SIMPLE_MEMORY_ALLOCATOR_H_

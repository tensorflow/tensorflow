/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_SIMPLE_MEMORY_ARENA_H_
#define TENSORFLOW_LITE_SIMPLE_MEMORY_ARENA_H_

#include <cstdint>
#include <list>
#include <memory>

#include "tensorflow/lite/c/common.h"

namespace tflite {

// This little structure holds the offset and the size for a dynamic memory
// allocation in the memory arena as well as first_node and last_node that use
// corresponding tensor. It means that continuous part of memory with this size
// needs to be allocated before execution of operation in the first node and can
// be deallocated after execution of the operation in the last_node. When the
// arena is committed and the underlying buffer is set, the alloc can be
// resolved into an actual memory pointer.
struct ArenaAllocWithUsageInterval {
  ArenaAllocWithUsageInterval() { reset(); }

  size_t offset;
  size_t size;
  int32_t tensor;
  int32_t first_node;
  int32_t last_node;

  inline void reset() {
    offset = 0;
    size = 0;
    tensor = -1;
    first_node = -1;
    last_node = -1;
  }

  inline bool operator<(const ArenaAllocWithUsageInterval& other) const {
    return offset < other.offset;
  }
};

// This small class is responsible for allocating, deallocating and reusing
// dynamic memory from a common underlying buffer. The arena can be used in
// scenarios when the pattern of memory allocations and deallocations is
// repetitive, e.g. running NN inference in multiple iterations. Note that
// zero-sized allocations are explicitly allowed, and will resolve to null.
class SimpleMemoryArena {
 public:
  explicit SimpleMemoryArena(size_t arena_alignment)
      : committed_(false),
        arena_alignment_(arena_alignment),
        high_water_mark_(0),
        underlying_buffer_size_(0),
        ordered_allocs_() {}

  // Schedule memory allocation for a tensor with a given size, assuming that it
  // needs to be allocated before the execution of first_node, and deallocated
  // after the execution of last_node.
  TfLiteStatus Allocate(TfLiteContext* context, size_t alignment, size_t size,
                        int32_t tensor, int32_t first_node, int32_t last_node,
                        ArenaAllocWithUsageInterval* new_alloc);

  TfLiteStatus Deallocate(TfLiteContext* context,
                          const ArenaAllocWithUsageInterval& alloc);

  inline size_t RequiredBufferSize() {
    // Add in a small amount of padding to reduce the chance of resize events
    // for small allocations.
    size_t padding = arena_alignment_;
    return arena_alignment_ + high_water_mark_ + padding;
  }

  TfLiteStatus Commit(TfLiteContext* context);

  TfLiteStatus ResolveAlloc(TfLiteContext* context,
                            const ArenaAllocWithUsageInterval& alloc,
                            char** output_ptr);

  // This clears allocation details but does not release the underlying buffer.
  // New allocations should be committed & resolved before using this arena
  // again.
  TfLiteStatus ClearPlan();

  // This releases the underlying buffer but does not clear the allocation plan.
  // Since all associated pointers are invalidated, the arena cannot be used
  // again until Commit() is called & tensor allocations are resolved.
  TfLiteStatus ReleaseBuffer();

  size_t GetBufferSize() { return underlying_buffer_size_; }

  std::intptr_t BasePointer() const {
    return reinterpret_cast<std::intptr_t>(underlying_buffer_aligned_ptr_);
  }

 private:
  bool committed_;
  size_t arena_alignment_;
  size_t high_water_mark_;
  std::unique_ptr<char[]> underlying_buffer_;
  size_t underlying_buffer_size_;
  char* underlying_buffer_aligned_ptr_;
  std::list<ArenaAllocWithUsageInterval> ordered_allocs_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_SIMPLE_MEMORY_ARENA_H_

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
#ifndef TENSORFLOW_CONTRIB_LITE_SIMPLE_MEMORY_ARENA_H_
#define TENSORFLOW_CONTRIB_LITE_SIMPLE_MEMORY_ARENA_H_

#include <list>
#include <memory>
#include "tensorflow/contrib/lite/context.h"

namespace tflite {

// This little structure holds the offset and the size for a dynamic memory
// allocation in the memory arena. When the arena is committed and the
// underlying buffer is set, the alloc can be resolved into an actual memory
// pointer.
struct ArenaAlloc {
  ArenaAlloc() : offset(0), size(0) {}

  size_t offset;
  size_t size;

  inline bool operator<(const ArenaAlloc& other) const {
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
        allocs_() {}

  TfLiteStatus Allocate(TfLiteContext* context, size_t alignment, size_t size,
                        ArenaAlloc* new_alloc);

  TfLiteStatus Deallocate(TfLiteContext* context, const ArenaAlloc& alloc);

  inline size_t RequiredBufferSize() {
    // Add in a small amount of padding to reduce the chance of resize events
    // for small allocations.
    size_t padding = arena_alignment_;
    return arena_alignment_ + high_water_mark_ + padding;
  }

  TfLiteStatus Commit(TfLiteContext* context);

  TfLiteStatus ResolveAlloc(TfLiteContext* context, const ArenaAlloc& alloc,
                            char** output_ptr);

  TfLiteStatus Clear();

  int64_t BasePointer() const {
    return reinterpret_cast<int64_t>(underlying_buffer_aligned_ptr_);
  }

 private:
  bool committed_;
  size_t arena_alignment_;
  size_t high_water_mark_;
  std::unique_ptr<char[]> underlying_buffer_;
  size_t underlying_buffer_size_;
  char* underlying_buffer_aligned_ptr_;
  // TODO(maciekc): add list iterator to the ArenaAlloc to lookup quickly.
  std::list<ArenaAlloc> allocs_;
};

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_SIMPLE_MEMORY_ARENA_H_

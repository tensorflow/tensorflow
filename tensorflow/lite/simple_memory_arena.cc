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

#include "tensorflow/lite/simple_memory_arena.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/macros.h"

#ifdef TF_LITE_TENSORFLOW_PROFILER
#include "tensorflow/lite/tensorflow_profiler_logger.h"
#endif  // TF_LITE_TENSORFLOW_PROFILER

#if defined(__ANDROID__)
// Android has C11 aligned_alloc only with API 28 or newer, even with C++17 or
// C11 compilation (this is a non-standard behavior).
#define TF_LITE_HAS_ALIGNED_ALLOC (__ANDROID_API__ >= 28)
#elif defined(__APPLE__)
// Apple does not provide aligned_alloc, even with C++17 or C11 compilation
// (this is a non-standard behavior).
#define TF_LITE_HAS_ALIGNED_ALLOC 0
#elif defined(_WIN32)
// Windows does not provide aligned_alloc, even with C++17 or C11 compilation
// (this is a non-standard behavior). However, it provides _aligned_malloc,
// _aligned_realloc, and _aligned_free, with a slightly different behavior than
// the C11/C++17 standard functions (size requirement, and free function name.)
#define TF_LITE_HAS_ALIGNED_ALLOC 0
#elif __cplusplus >= 201703L || __STDC_VERSION__ >= 201112L
// C++17 or C11 has (std::)aligned_alloc
#define TF_LITE_HAS_ALIGNED_ALLOC 1
#endif

namespace {

template <typename T>
T AlignTo(size_t alignment, T offset) {
  return offset % alignment == 0 ? offset
                                 : offset + (alignment - offset % alignment);
}

// Allocates memory and aligns it to the specified size. Returns a pair of the
// allocation pointer and the aligned pointer.
tflite::PointerAlignedPointerPair AlignedAlloc(size_t size, size_t alignment);

// Frees up aligned memory.
void AlignedFree(const tflite::PointerAlignedPointerPair& buffer);

// Reallocates aligned memory
//
// The function either extends the memory allocation in-place, or if that is not
// possible a new allocation is created, the data is copied, and the old buffer
// is deallocated. It is an error to change the alignment during reallocation.
// If the previous allocation is null, this is equivalent to AlignedAlloc.
// Returns pointers to the new allocation.
tflite::PointerAlignedPointerPair AlignedRealloc(
    const tflite::PointerAlignedPointerPair& old_buffer, size_t old_size,
    size_t new_size, size_t alignment);

#if defined(_WIN32)
// On Windows <cstdlib> provides _aligned_malloc, _aligned_free, and
// _aligned_realloc, use them to implement the Aligned functions.

tflite::PointerAlignedPointerPair AlignedAlloc(size_t size, size_t alignment) {
  char* pointer = reinterpret_cast<char*>(_aligned_malloc(size, alignment));
  char* aligned_ptr = pointer;
  return {pointer, aligned_ptr};
}

void AlignedFree(const tflite::PointerAlignedPointerPair& buffer) {
  _aligned_free(buffer.pointer);
}

tflite::PointerAlignedPointerPair AlignedRealloc(
    const tflite::PointerAlignedPointerPair& old_buffer, size_t old_size,
    size_t new_size, size_t alignment) {
  char* pointer = reinterpret_cast<char*>(
      _aligned_realloc(old_buffer.pointer, new_size, alignment));
  char* aligned_ptr = pointer;
  return {pointer, aligned_ptr};
}
#else
// Default implementation: Use malloc, allocating extra memory, and align the
// pointer in the allocated buffer.

tflite::PointerAlignedPointerPair AlignedAlloc(size_t size, size_t alignment) {
#if TF_LITE_HAS_ALIGNED_ALLOC
  // (std::)aligned_alloc requires size to be multiple of alignment.
  // TODO(b/311495100): when bug is fixed, remove `size + alignment - 1` part.
  const size_t allocation_size = AlignTo(alignment, size + alignment - 1);
  char* pointer =
      reinterpret_cast<char*>(::aligned_alloc(alignment, allocation_size));
  char* aligned_ptr = pointer;
#else
  // TODO(b/311495100): when bug is fixed, change this to
  // `size + std::max(size_t{0}, alignment - alignof(std::max_align_t))`
  const size_t allocation_size = size + alignment - 1;
  char* pointer = reinterpret_cast<char*>(std::malloc(allocation_size));
  char* aligned_ptr = reinterpret_cast<char*>(
      AlignTo(alignment, reinterpret_cast<std::uintptr_t>(pointer)));
#endif
#if defined(__clang__)
#if __has_feature(memory_sanitizer)
  std::memset(pointer, 0, allocation_size);
#endif
#endif
  return {pointer, aligned_ptr};
}

void AlignedFree(const tflite::PointerAlignedPointerPair& buffer) {
  std::free(buffer.pointer);
}

tflite::PointerAlignedPointerPair AlignedRealloc(
    const tflite::PointerAlignedPointerPair& old_buffer, size_t old_size,
    size_t new_size, size_t alignment) {
  tflite::PointerAlignedPointerPair new_buffer =
      AlignedAlloc(new_size, alignment);
  if (new_size > 0 && old_size > 0) {
    // Copy data when both old and new buffers are bigger than 0 bytes.
    const size_t copy_amount = std::min(new_size, old_size);
    std::memcpy(new_buffer.aligned_pointer, old_buffer.aligned_pointer,
                copy_amount);
  }
  AlignedFree(old_buffer);
  return new_buffer;
}
#endif
}  // namespace

namespace tflite {

bool ResizableAlignedBuffer::Resize(size_t new_size) {
  if (new_size <= data_size_) {
    // Skip reallocation when resizing down.
    return false;
  }
#ifdef TF_LITE_TENSORFLOW_PROFILER
  PauseHeapMonitoring(/*pause=*/true);
  OnTfLiteArenaAlloc(subgraph_index_, reinterpret_cast<std::uintptr_t>(this),
                     new_size);
  if (data_size_ > 0) {
    OnTfLiteArenaDealloc(subgraph_index_,
                         reinterpret_cast<std::uintptr_t>(this), data_size_);
  }
#endif
  auto new_buffer = AlignedRealloc(buffer_, data_size_, new_size, alignment_);
  bool reallocated = (new_buffer.aligned_pointer != buffer_.aligned_pointer);
  buffer_ = new_buffer;
  data_size_ = new_size;
#ifdef TF_LITE_TENSORFLOW_PROFILER
  PauseHeapMonitoring(/*pause=*/false);
#endif
  return reallocated;
}

void ResizableAlignedBuffer::Release() {
  if (buffer_.pointer == nullptr) {
    return;
  }
#ifdef TF_LITE_TENSORFLOW_PROFILER
  OnTfLiteArenaDealloc(subgraph_index_, reinterpret_cast<std::uintptr_t>(this),
                       data_size_);
#endif
  AlignedFree(buffer_);
  buffer_.pointer = nullptr;
  buffer_.aligned_pointer = nullptr;
  data_size_ = 0;
}

void SimpleMemoryArena::PurgeAfter(int32_t node) {
  for (int i = 0; i < active_allocs_.size(); ++i) {
    if (active_allocs_[i].first_node > node) {
      // alloc is allocated after node, so mark it for deletion.
      active_allocs_[i].tensor = -1;
    }
  }
  active_allocs_.erase(
      std::remove_if(active_allocs_.begin(), active_allocs_.end(),
                     [](ArenaAllocWithUsageInterval& alloc) {
                       return alloc.tensor == -1;
                     }),
      active_allocs_.end());
}

void SimpleMemoryArena::PurgeActiveAllocs(int32_t node) {
  for (int i = 0; i < active_allocs_.size(); ++i) {
    if (active_allocs_[i].last_node < node) {
      // alloc is deallocated before node, so mark it for deletion..
      active_allocs_[i].tensor = -1;
    }
  }
  active_allocs_.erase(
      std::remove_if(active_allocs_.begin(), active_allocs_.end(),
                     [](ArenaAllocWithUsageInterval& alloc) {
                       return alloc.tensor == -1;
                     }),
      active_allocs_.end());
}

void SimpleMemoryArena::CalculateActiveAllocs(
    const std::vector<ArenaAllocWithUsageInterval>& allocs, int32_t node) {
  active_allocs_.clear();
  for (int i = 0; i < allocs.size(); ++i) {
    if (allocs[i].first_node <= node && allocs[i].last_node >= node) {
      active_allocs_.push_back(allocs[i]);
    }
  }
  std::sort(active_allocs_.begin(), active_allocs_.end());
}

void SimpleMemoryArena::ResetAllocs() { active_allocs_.clear(); }

TfLiteStatus SimpleMemoryArena::Allocate(
    TfLiteContext* context, size_t alignment, size_t size, int32_t tensor,
    int32_t first_node, int32_t last_node,
    ArenaAllocWithUsageInterval* new_alloc) {
  TF_LITE_ENSURE(context, alignment <= underlying_buffer_.GetAlignment());
  new_alloc->tensor = tensor;
  new_alloc->first_node = first_node;
  new_alloc->last_node = last_node;
  new_alloc->size = size;
  if (size == 0) {
    new_alloc->offset = 0;
    return kTfLiteOk;
  }
  // If we don't find a better gap just allocate at the end of the buffer.
  const size_t kOffsetNotAssigned = std::numeric_limits<size_t>::max();
  size_t best_offset = kOffsetNotAssigned;
  size_t best_offset_fit = kOffsetNotAssigned;

  // Go through the sorted allocs and look at the gaps between them.
  size_t current_offset = 0;
  for (const auto& alloc : active_allocs_) {
    if (alloc.last_node < first_node || alloc.first_node > last_node) {
      // Usage interval of alloc doesn't intersect with current tensor's usage
      // interval, so we skip it.
      continue;
    }
    size_t aligned_current_offset = AlignTo(alignment, current_offset);
    // If we found a gap larger than required size, and smaller than previous
    // best fit, take it.
    if (aligned_current_offset + size <= alloc.offset &&
        alloc.offset - aligned_current_offset < best_offset_fit) {
      best_offset = aligned_current_offset;
      best_offset_fit = alloc.offset - current_offset;
    }
    current_offset = std::max(current_offset, alloc.offset + alloc.size);
    // A gap of zero is as good as it gets, no point continuing.
    if (best_offset_fit == 0) {
      break;
    }
  }
  if (best_offset == kOffsetNotAssigned) {
    best_offset = AlignTo(alignment, current_offset);
  }

  // Update the required buffer size.
  high_water_mark_ = std::max(high_water_mark_, best_offset + size);
  new_alloc->offset = best_offset;

  auto insertion_it = std::upper_bound(active_allocs_.begin(),
                                       active_allocs_.end(), *new_alloc);
  active_allocs_.insert(insertion_it, *new_alloc);
  return kTfLiteOk;
}

TfLiteStatus SimpleMemoryArena::Commit(bool* arena_reallocated) {
  // Resize the arena to the high water mark (calculated by Allocate), retaining
  // old contents and alignment in the process. Since Alloc pointers are offset
  // based, they will remain valid in the new memory block.
  *arena_reallocated = underlying_buffer_.Resize(high_water_mark_);
  committed_ = true;
  return kTfLiteOk;
}

TfLiteStatus SimpleMemoryArena::ResolveAlloc(
    TfLiteContext* context, const ArenaAllocWithUsageInterval& alloc,
    char** output_ptr) {
  TF_LITE_ENSURE(context, committed_);
  TF_LITE_ENSURE(context, output_ptr != nullptr);
  TF_LITE_ENSURE(context,
                 underlying_buffer_.GetSize() >= (alloc.offset + alloc.size));
  if (alloc.size == 0) {
    *output_ptr = nullptr;
  } else {
    *output_ptr = underlying_buffer_.GetPtr() + alloc.offset;
  }
  return kTfLiteOk;
}

TfLiteStatus SimpleMemoryArena::ClearPlan() {
  committed_ = false;
  high_water_mark_ = 0;
  active_allocs_.clear();
  return kTfLiteOk;
}

TfLiteStatus SimpleMemoryArena::ReleaseBuffer() {
  committed_ = false;
  underlying_buffer_.Release();
  return kTfLiteOk;
}

// Using weak symbols to create a pluggable debugging module.
TFLITE_ATTRIBUTE_WEAK void DumpArenaInfo(
    const std::string& name, const std::vector<int>& execution_plan,
    size_t arena_size, const std::vector<ArenaAllocWithUsageInterval>& allocs) {
}

void SimpleMemoryArena::DumpDebugInfo(
    const std::string& name, const std::vector<int>& execution_plan) const {
  tflite::DumpArenaInfo(name, execution_plan, underlying_buffer_.GetSize(),
                        active_allocs_);
}

}  // namespace tflite

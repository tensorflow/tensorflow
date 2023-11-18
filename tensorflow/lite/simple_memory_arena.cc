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

namespace {

template <typename T>
T AlignTo(size_t alignment, T offset) {
  return offset % alignment == 0 ? offset
                                 : offset + (alignment - offset % alignment);
}

}  // namespace

namespace tflite {

bool ResizableAlignedBuffer::Resize(size_t new_size) {
  const size_t new_allocation_size = RequiredAllocationSize(new_size);
  if (new_allocation_size <= allocation_size_) {
    // Skip reallocation when resizing down.
    return false;
  }
#ifdef TF_LITE_TENSORFLOW_PROFILER
  PauseHeapMonitoring(/*pause=*/true);
  OnTfLiteArenaAlloc(subgraph_index_, reinterpret_cast<std::uintptr_t>(this),
                     new_allocation_size);
#endif
  char* new_buffer = reinterpret_cast<char*>(std::malloc(new_allocation_size));
  char* new_aligned_ptr = reinterpret_cast<char*>(
      AlignTo(alignment_, reinterpret_cast<std::uintptr_t>(new_buffer)));
  if (new_size > 0 && allocation_size_ > 0) {
    // Copy data when both old and new buffers are bigger than 0 bytes.
    const size_t new_alloc_alignment_adjustment = new_aligned_ptr - new_buffer;
    const size_t old_alloc_alignment_adjustment = aligned_ptr_ - buffer_;
    const size_t copy_amount =
        std::min(allocation_size_ - old_alloc_alignment_adjustment,
                 new_allocation_size - new_alloc_alignment_adjustment);
    std::memcpy(new_aligned_ptr, aligned_ptr_, copy_amount);
  }
  std::free(buffer_);
  buffer_ = new_buffer;
  aligned_ptr_ = new_aligned_ptr;
#ifdef TF_LITE_TENSORFLOW_PROFILER
  if (allocation_size_ > 0) {
    OnTfLiteArenaDealloc(subgraph_index_,
                         reinterpret_cast<std::uintptr_t>(this),
                         allocation_size_);
  }
#endif
  allocation_size_ = new_allocation_size;
#ifdef TF_LITE_TENSORFLOW_PROFILER
  PauseHeapMonitoring(/*pause=*/false);
#endif
  return true;
}

void ResizableAlignedBuffer::Release() {
  if (buffer_ == nullptr) {
    return;
  }
#ifdef TF_LITE_TENSORFLOW_PROFILER
  OnTfLiteArenaDealloc(subgraph_index_, reinterpret_cast<std::uintptr_t>(this),
                       allocation_size_);
#endif
  std::free(buffer_);
  buffer_ = nullptr;
  allocation_size_ = 0;
  aligned_ptr_ = nullptr;
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
  TF_LITE_ENSURE(context, underlying_buffer_.GetAllocationSize() >=
                              (alloc.offset + alloc.size));
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
  tflite::DumpArenaInfo(name, execution_plan,
                        underlying_buffer_.GetAllocationSize(), active_allocs_);
}

}  // namespace tflite

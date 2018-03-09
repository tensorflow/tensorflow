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

#include "tensorflow/contrib/lite/simple_memory_arena.h"

#include <cstring>
#include <limits>
#include <vector>

namespace {

template <typename T>
T AlignTo(size_t alignment, T offset) {
  return offset % alignment == 0 ? offset
                                 : offset + (alignment - offset % alignment);
}

}  // namespace

namespace tflite {

TfLiteStatus SimpleMemoryArena::Allocate(TfLiteContext* context,
                                         size_t alignment, size_t size,
                                         ArenaAlloc* new_alloc) {
  TF_LITE_ENSURE(context, alignment < arena_alignment_);

  size_t current_top = 0;

  if (!allocs_.empty()) {
    auto last = allocs_.rbegin();
    current_top = last->offset + last->size;
  }

  // If we don't find a better gap just allocate at the end of the buffer.
  size_t best_offset = AlignTo(alignment, current_top);
  size_t best_offset_fit = std::numeric_limits<size_t>::max();
  auto best_insertion_it = allocs_.end();

  // Go through the sorted allocs and look at the gaps between them.
  size_t current_offset = 0;
  for (auto it = allocs_.begin(); it != allocs_.end(); ++it) {
    size_t aligned_current_offset = AlignTo(alignment, current_offset);
    // If we found a gap larger than required size, and smaller than previous
    // best fit, take it.
    if (aligned_current_offset + size <= it->offset &&
        it->offset - current_offset < best_offset_fit) {
      best_offset = aligned_current_offset;
      best_offset_fit = it->offset - current_offset;
      best_insertion_it = it;
    }
    current_offset = it->offset + it->size;
  }

  // Update the required buffer size.
  high_water_mark_ = std::max(high_water_mark_, best_offset + size);

  new_alloc->offset = best_offset;
  new_alloc->size = size;
  allocs_.insert(best_insertion_it, *new_alloc);

  return kTfLiteOk;
}

TfLiteStatus SimpleMemoryArena::Deallocate(TfLiteContext* context,
                                           const ArenaAlloc& alloc) {
  int erased_allocs_count = 0;
  auto it = allocs_.begin();
  while (it != allocs_.end()) {
    if (it->offset == alloc.offset) {
      TF_LITE_ENSURE_EQ(context, it->size, alloc.size);
      erased_allocs_count++;
      it = allocs_.erase(it);
    } else {
      ++it;
    }
  }
  TF_LITE_ENSURE_EQ(context, erased_allocs_count, 1);
  return kTfLiteOk;
}

TfLiteStatus SimpleMemoryArena::Commit(TfLiteContext* context) {
  size_t required_size = RequiredBufferSize();
  if (required_size > underlying_buffer_size_) {
    char* new_alloc = new char[required_size];
    char* new_underlying_buffer_aligned_ptr = reinterpret_cast<char*>(
        AlignTo(arena_alignment_, reinterpret_cast<intptr_t>(new_alloc)));

    // If the arena had been previously allocated, copy over the old memory.
    // Since Alloc pointers are offset based, they will remain valid in the new
    // memory block.
    if (high_water_mark_ > 0 && underlying_buffer_size_ > 0) {
      size_t copy_amount = std::min(
          underlying_buffer_.get() + underlying_buffer_size_ -
              underlying_buffer_aligned_ptr_,
          new_alloc + required_size - new_underlying_buffer_aligned_ptr);
      memcpy(new_underlying_buffer_aligned_ptr, underlying_buffer_aligned_ptr_,
             copy_amount);
    }

    underlying_buffer_.reset(new_alloc);
    underlying_buffer_size_ = required_size;
    underlying_buffer_aligned_ptr_ = new_underlying_buffer_aligned_ptr;
  }
  commited_ = true;
  return underlying_buffer_ != nullptr ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus SimpleMemoryArena::ResolveAlloc(TfLiteContext* context,
                                             const ArenaAlloc& alloc,
                                             char** output_ptr) {
  TF_LITE_ENSURE(context, commited_);
  TF_LITE_ENSURE(context, output_ptr != nullptr);
  *output_ptr = underlying_buffer_aligned_ptr_ + alloc.offset;
  return kTfLiteOk;
}

TfLiteStatus SimpleMemoryArena::Clear() {
  commited_ = false;
  high_water_mark_ = 0;
  allocs_.clear();
  return kTfLiteOk;
}

}  // namespace tflite

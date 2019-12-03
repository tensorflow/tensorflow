/* Copyright 2019 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_ALLOCATOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/size_util.h"

namespace ruy {

namespace detail {

inline void* VoidPtrAdd(void* p, std::ptrdiff_t offset) {
  RUY_DCHECK(p);
  std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(p) + offset;
  return reinterpret_cast<void*>(addr);
}

// Minimum alignment for blocks.
//
// Considerations:
//  - This needs to be at least the alignment of any usual data type.
//  - It's useful that this is at least the size of a cache line to limit
//    possible cache side effects (if only on performance behavior).
//  - It's useful that this is at least the size of SIMD registers, as
//    some SIMD instruction sets have at least performance behavior
//    differences (e.g. NEON) or even different requirements (e.g. SSE)
//    based on that.
//  - It's useful that this is at least the size of an "exclusive reservation
//    granule" on ARM, meaning that if we use this Allocator to allocate
//    an atomic variable, there will be no side effects from other things
//    contending for exclusive/atomic memory accesses to it. While the
//    ARM reference manual mentions that this granule size may be as large
//    as 2048 bytes, in practice we observe it to be 64 bytes. It can
//    be queried cheaply, at runtime, from userspace, if needed.
static constexpr std::ptrdiff_t kMinimumBlockAlignment = 64;

// Primitive allocation functions obtaining aligned memory from the
// operating system.
void* SystemAlignedAlloc(std::ptrdiff_t num_bytes);
void SystemAlignedFree(void* ptr);

// Specialized allocator designed to converge to a steady-state where all
// allocations are bump-ptr allocations from an already-allocated buffer.
//
// To support these constraints, this allocator only supports two
// operations.
// - AllocateAlignedBytes: allocates a pointer to storage of a specified
// size, which must be aligned to kMinimumBlockAlignment.
// - FreeAll: frees all previous allocations (but retains the internal
// buffer to minimize future calls into the system allocator).
//
// This class is specialized for supporting just those two operations
// under this specific steady-state usage pattern. Extending this class
// with new allocation interfaces that don't fit that pattern is probably not
// the right choice. Instead, build a new class on top of
// SystemAlignedAlloc/SystemAlignedFree.
//
// All operations happen on aligned blocks for simplicity.
class AlignedAllocator {
 public:
  void operator=(const AlignedAllocator&) = delete;
  ~AlignedAllocator() {
    FreeAll();
    SystemAlignedFree(ptr_);
  }

  void* AllocateAlignedBytes(std::ptrdiff_t num_bytes) {
    RUY_DCHECK_GT(num_bytes, 0);
    RUY_DCHECK((num_bytes & (kMinimumBlockAlignment - 1)) == 0);
    if (void* p = AllocateFast(num_bytes)) {
      return p;
    }
    return AllocateSlow(num_bytes);
  }

  void FreeAll() {
    current_ = 0;
    if (fallback_blocks_.empty()) {
      return;
    }

    // No rounding-up of the size means linear instead of logarithmic
    // bound on the number of allocation in some worst-case calling patterns.
    // This is considered worth it because minimizing memory usage is important
    // and actual calling patterns in applications that we care about still
    // reach the no-further-allocations steady state in a small finite number
    // of iterations.
    std::ptrdiff_t new_size = size_ + fallback_blocks_total_size_;
    SystemAlignedFree(ptr_);
    ptr_ = SystemAlignedAlloc(new_size);
    size_ = new_size;

    for (void* p : fallback_blocks_) {
      SystemAlignedFree(p);
    }
    fallback_blocks_.clear();
    fallback_blocks_total_size_ = 0;
  }

 private:
  void* AllocateFast(std::ptrdiff_t num_bytes) {
    if (current_ + num_bytes > size_) {
      return nullptr;
    }
    void* ret = VoidPtrAdd(ptr_, current_);
    current_ += num_bytes;
    return ret;
  }

  void* AllocateSlow(std::ptrdiff_t num_bytes) {
    void* p = SystemAlignedAlloc(num_bytes);
    fallback_blocks_total_size_ += num_bytes;
    fallback_blocks_.push_back(p);
    return p;
  }

  // Theory of operation:
  //
  // - ptr_, current_, and size_ implement a basic bump-ptr allocator.
  //
  // - in AllocateAlignedBytes, the fast path is just a bump-ptr
  // allocation. If our bump-ptr allocator doesn't have enough space for an
  // allocation, then we allocate a block from the system allocator to
  // service the allocation request. We save that block in fallback_blocks_
  // and track the total size of the fallback blocks in
  // fallback_blocks_total_size_.
  //
  // - in FreeAll, the fast path just resets the bump-ptr allocator. If
  // there are any fallback blocks, we free them and reallocate the
  // bump-ptr allocator's buffer so that the next sequence of allocations
  // will hopefully not need any fallback blocks.
  void* ptr_ = nullptr;
  std::ptrdiff_t current_ = 0;
  std::ptrdiff_t size_ = 0;
  std::vector<void*> fallback_blocks_;
  std::ptrdiff_t fallback_blocks_total_size_ = 0;
};

}  // namespace detail

// The main Allocator class, with a convenient interface for allocating a
// typed buffer.
class Allocator {
 public:
  void* AllocateBytes(std::ptrdiff_t num_bytes) {
    if (num_bytes == 0) {
      return nullptr;
    }
    return aligned.AllocateAlignedBytes(
        round_up_pot(num_bytes, detail::kMinimumBlockAlignment));
  }
  template <typename Pointer>
  void Allocate(std::ptrdiff_t count, Pointer* out) {
    using T = typename std::pointer_traits<Pointer>::element_type;
    *out = static_cast<T*>(AllocateBytes(count * sizeof(T)));
  }

  void FreeAll() { aligned.FreeAll(); }

 private:
  detail::AlignedAllocator aligned;
};

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_ALLOCATOR_H_

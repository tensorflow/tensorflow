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

// A simple CPU allocator that intercepts malloc/free calls from MKL library
// and redirects them to Tensorflow allocator

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_MKL_CPU_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_MKL_CPU_ALLOCATOR_H_

#ifdef INTEL_MKL

#include <cstdlib>

#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/numa.h"

#ifdef _WIN32
typedef unsigned int uint;
#endif

namespace tensorflow {

static bool mkl_small_allocator_collect_stats = false;

class MklSubAllocator : public BasicCPUAllocator {
 public:
  MklSubAllocator() : BasicCPUAllocator(port::kNUMANoAffinity, {}, {}) {}
  ~MklSubAllocator() override {}
};

// CPU allocator that handles small-size allocations by calling
// suballocator directly. Mostly, it is just a wrapper around a suballocator
// (that calls malloc and free directly) with support for bookkeeping.
class MklSmallSizeAllocator : public Allocator {
 public:
  MklSmallSizeAllocator(SubAllocator* sub_allocator, size_t total_memory,
                        const string& name)
      : sub_allocator_(sub_allocator), name_(name) {
    stats_.bytes_limit = total_memory;
  }
  ~MklSmallSizeAllocator() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(MklSmallSizeAllocator);

  inline string Name() override { return name_; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    void* ptr = port::AlignedMalloc(num_bytes, alignment);
    if (mkl_small_allocator_collect_stats) IncrementStats(num_bytes);
    return ptr;
  }

  void DeallocateRaw(void* ptr) override {
    if (ptr == nullptr) {
      LOG(ERROR) << "tried to deallocate nullptr";
      return;
    }

    if (mkl_small_allocator_collect_stats) {
      const size_t alloc_size = port::MallocExtension_GetAllocatedSize(ptr);
      DecrementStats(alloc_size);
    }
    port::AlignedFree(ptr);
  }

  absl::optional<AllocatorStats> GetStats() override {
    mutex_lock l(mutex_);
    return stats_;
  }

  void ClearStats() override {
    mutex_lock l(mutex_);
    stats_.num_allocs = 0;
    stats_.peak_bytes_in_use = 0;
    stats_.largest_alloc_size = 0;
    stats_.bytes_in_use = 0;
    stats_.bytes_limit = 0;
  }

 private:
  // Increment statistics for the allocator handling small allocations.
  inline void IncrementStats(size_t alloc_size) TF_LOCKS_EXCLUDED(mutex_) {
    mutex_lock l(mutex_);
    ++stats_.num_allocs;
    stats_.bytes_in_use += alloc_size;
    stats_.peak_bytes_in_use =
        std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
    stats_.largest_alloc_size =
        std::max(alloc_size, static_cast<size_t>(stats_.largest_alloc_size));
  }

  // Decrement statistics for the allocator handling small allocations.
  inline void DecrementStats(size_t dealloc_size) TF_LOCKS_EXCLUDED(mutex_) {
    mutex_lock l(mutex_);
    stats_.bytes_in_use -= dealloc_size;
  }

  SubAllocator* sub_allocator_;  // Not owned by this class.

  // Mutex for protecting updates to map of allocations.
  mutable mutex mutex_;

  // Allocator name
  string name_;

  // Allocator stats for small allocs
  AllocatorStats stats_ TF_GUARDED_BY(mutex_);
};

/// CPU allocator for MKL that wraps BFC allocator and intercepts
/// and redirects memory allocation calls from MKL.
class MklCPUAllocator : public Allocator {
 public:
  // Constructor and other standard functions

  /// Environment variable that user can set to upper bound on memory allocation
  static constexpr const char* kMaxLimitStr = "TF_MKL_ALLOC_MAX_BYTES";

  /// Default upper limit on allocator size - 64GB
  static constexpr size_t kDefaultMaxLimit = 64LL << 30;

  MklCPUAllocator() { TF_CHECK_OK(Initialize()); }

  ~MklCPUAllocator() override {
    delete small_size_allocator_;
    delete large_size_allocator_;
  }

  Status Initialize() {
    VLOG(2) << "MklCPUAllocator: In MklCPUAllocator";

    // Set upper bound on memory allocation to physical RAM available on the
    // CPU unless explicitly specified by user
    uint64 max_mem_bytes = kDefaultMaxLimit;
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    max_mem_bytes =
        (uint64)sysconf(_SC_PHYS_PAGES) * (uint64)sysconf(_SC_PAGESIZE);
#endif
    char* user_mem_bytes = getenv(kMaxLimitStr);

    if (user_mem_bytes != NULL) {
      uint64 user_val = 0;
      if (!strings::safe_strtou64(user_mem_bytes, &user_val)) {
        return errors::InvalidArgument("Invalid memory limit (", user_mem_bytes,
                                       ") specified for MKL allocator through ",
                                       kMaxLimitStr);
      }
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
      if (user_val > max_mem_bytes) {
        LOG(WARNING) << "The user specified a memory limit " << kMaxLimitStr
                     << "=" << user_val
                     << " greater than available physical memory: "
                     << max_mem_bytes
                     << ". This could significantly reduce performance!";
      }
#endif
      max_mem_bytes = user_val;
    }

    VLOG(1) << "MklCPUAllocator: Setting max_mem_bytes: " << max_mem_bytes;

    sub_allocator_ = new MklSubAllocator();

    // SubAllocator is owned by BFCAllocator, so we do not need to deallocate
    // it in MklSmallSizeAllocator.
    small_size_allocator_ =
        new MklSmallSizeAllocator(sub_allocator_, max_mem_bytes, kName);
    large_size_allocator_ =
        new BFCAllocator(sub_allocator_, max_mem_bytes, kAllowGrowth, kName);
    return Status::OK();
  }

  inline string Name() override { return kName; }
  inline bool IsSmallSizeAllocation(const void* ptr) const
      TF_LOCKS_EXCLUDED(mutex_) {
    mutex_lock l(mutex_);
    return large_allocations_map_.find(ptr) == large_allocations_map_.end();
  }
  // AddLargeAllocMap and RemoveLargeAllocMap are always called with a lock held
  inline void AddLargeAllocMap(void* ptr, size_t num_bytes)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    if (ptr != nullptr) {
      std::pair<void*, size_t> map_val(ptr, num_bytes);
      large_allocations_map_.insert(map_val);
    }
  }
  inline void RemoveLargeAllocMap(void* ptr)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    auto map_iter = large_allocations_map_.find(ptr);
    if (map_iter != large_allocations_map_.end()) {
      large_allocations_map_.erase(map_iter);
    } else {
      LOG(ERROR) << "tried to deallocate invalid pointer";
    }
    return;
  }

  inline void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    // If the allocation size is less than threshold, call small allocator,
    // otherwise call large-size allocator (BFC). We found that BFC allocator
    // does not deliver good performance for small allocations when
    // inter_op_parallelism_threads is high.
    if (num_bytes < kSmallAllocationsThreshold) {
      return small_size_allocator_->AllocateRaw(alignment, num_bytes);
    } else {
      mutex_lock l(mutex_);
      void* ptr = large_size_allocator_->AllocateRaw(alignment, num_bytes);
      AddLargeAllocMap(ptr, num_bytes);
      return ptr;
    }
  }

  inline void DeallocateRaw(void* ptr) override {
    // Check if ptr is for "small" allocation. If it is, then call Free
    // directly. Otherwise, call BFC to handle free.
    if (IsSmallSizeAllocation(ptr)) {
      small_size_allocator_->DeallocateRaw(ptr);
    } else {
      mutex_lock l(mutex_);
      RemoveLargeAllocMap(ptr);
      large_size_allocator_->DeallocateRaw(ptr);
    }
  }

  absl::optional<AllocatorStats> GetStats() override {
    auto s_stats = small_size_allocator_->GetStats();
    auto l_stats = large_size_allocator_->GetStats();

    // Combine statistics from small-size and large-size allocator.
    mutex_lock l(mutex_);
    stats_.num_allocs = l_stats->num_allocs + s_stats->num_allocs;
    stats_.bytes_in_use = l_stats->bytes_in_use + s_stats->bytes_in_use;
    stats_.peak_bytes_in_use =
        l_stats->peak_bytes_in_use + s_stats->peak_bytes_in_use;

    // Since small-size allocations go to MklSmallSizeAllocator,
    // max_alloc_size from large_size_allocator would be the maximum
    // size allocated by MklCPUAllocator.
    stats_.largest_alloc_size = l_stats->largest_alloc_size;
    stats_.bytes_limit = std::max(s_stats->bytes_limit, l_stats->bytes_limit);
    return stats_;
  }

  void ClearStats() override {
    small_size_allocator_->ClearStats();
    large_size_allocator_->ClearStats();
  }

 private:
  // Hooks provided by this allocator for memory allocation routines from MKL

  static inline void* MallocHook(size_t size) {
    VLOG(3) << "MklCPUAllocator: In MallocHook";
    return cpu_allocator()->AllocateRaw(kAlignment, size);
  }

  static inline void FreeHook(void* ptr) {
    VLOG(3) << "MklCPUAllocator: In FreeHook";
    cpu_allocator()->DeallocateRaw(ptr);
  }

  static inline void* CallocHook(size_t num, size_t size) {
    Status s = Status(error::Code::UNIMPLEMENTED,
                      "Unimplemented case for hooking MKL function.");
    TF_CHECK_OK(s);  // way to assert with an error message
    return nullptr;  // return a value and make static code analyzers happy
  }

  static inline void* ReallocHook(void* ptr, size_t size) {
    Status s = Status(error::Code::UNIMPLEMENTED,
                      "Unimplemented case for hooking MKL function.");
    TF_CHECK_OK(s);  // way to assert with an error message
    return nullptr;  // return a value and make static code analyzers happy
  }

  // Do we allow growth in BFC Allocator
  static const bool kAllowGrowth = true;

  // Name
  static constexpr const char* kName = "mklcpu";

  // The alignment that we need for the allocations
  static constexpr const size_t kAlignment = 64;

  Allocator* large_size_allocator_;              // owned by this class
  MklSmallSizeAllocator* small_size_allocator_;  // owned by this class.

  SubAllocator* sub_allocator_;  // not owned by this class
  mutable mutex mutex_;
  AllocatorStats stats_ TF_GUARDED_BY(mutex_);

  // Hash map to keep track of "BFC" allocations
  // We do not use BFC allocator for small allocations.
  std::unordered_map<const void*, size_t> large_allocations_map_
      TF_GUARDED_BY(mutex_);

  // Size in bytes that defines the upper-bound for "small" allocations.
  // Any allocation below this threshold is "small" allocation.
  static constexpr const size_t kSmallAllocationsThreshold = 4096;

  // Prevent copying and assignment
  TF_DISALLOW_COPY_AND_ASSIGN(MklCPUAllocator);
};

}  // namespace tensorflow

#endif  // INTEL_MKL

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_MKL_CPU_ALLOCATOR_H_

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_POOL_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_POOL_ALLOCATOR_H_

// Simple LRU pool allocators for various flavors of CPU RAM.

#include <atomic>
#include <map>
#include <memory>
#include <vector>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Interface of an object that rounds up integers.
class RoundUpInterface {
 public:
  virtual ~RoundUpInterface() {}
  virtual size_t RoundUp(size_t num_bytes) = 0;
};

// Size-limited pool of memory buffers obtained from a SubAllocator
// instance.  Pool eviction policy is LRU.
class PoolAllocator : public Allocator {
 public:
  // "pool_size_limit" is the maximum number of returned, re-usable
  // memory buffers to keep in the pool.  If pool_size_limit == 0, the
  // pool is effectively a thin wrapper around the allocator.
  // If "auto_resize" is true, then the pool_size_limit will gradually
  // be raised so that deallocations happen very rarely, if at all.
  // Transitory start-up objects may deallocate, but the long-term
  // working-set should not. Auto-resizing can raise pool_size_limit
  // but will never lower it.
  // "allocator" is the object that performs the underlying memory
  // malloc/free operations.  This object takes ownership of allocator.
  PoolAllocator(size_t pool_size_limit, bool auto_resize,
                SubAllocator* allocator, RoundUpInterface* size_rounder,
                string name);
  ~PoolAllocator() override;

  string Name() override { return name_; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override;

  void DeallocateRaw(void* ptr) override;

  // Allocate an unused memory region of size "num_bytes".  Fetch from
  // the pool if available, otherwise call allocator_.
  void* Get(size_t num_bytes);

  // Return a no-longer needed memory region to the pool.  It is an error
  // to deference "ptr" after this call.  If the pool is full, the least
  // recently used region will be deallocated.
  void Put(void* ptr, size_t num_bytes);

  // Reset the pool to empty.
  void Clear();

  // The following accessors permit monitoring the effectiveness of
  // the pool at avoiding repeated malloc/frees on the underlying
  // allocator.  Read locks are not taken on the theory that value
  // consistency with other threads is not important.

  // Number of Get() requests satisfied from pool.
  int64 get_from_pool_count() const TF_NO_THREAD_SAFETY_ANALYSIS {
    return get_from_pool_count_;
  }
  // Number of Put() requests.
  int64 put_count() const TF_NO_THREAD_SAFETY_ANALYSIS { return put_count_; }
  // Number of Get() requests requiring a fresh allocation.
  int64 allocated_count() const TF_NO_THREAD_SAFETY_ANALYSIS {
    return allocated_count_;
  }
  // Number of pool evictions.
  int64 evicted_count() const TF_NO_THREAD_SAFETY_ANALYSIS {
    return evicted_count_;
  }
  // Current size limit.
  size_t size_limit() const TF_NO_THREAD_SAFETY_ANALYSIS {
    return pool_size_limit_;
  }

 private:
  struct PtrRecord {
    void* ptr;
    size_t num_bytes;
    PtrRecord* prev;
    PtrRecord* next;
  };

  // Remove "pr" from the double-linked LRU list.
  void RemoveFromList(PtrRecord* pr) TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Add "pr" to the head of the double-linked LRU list.
  void AddToList(PtrRecord* pr) TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Delete the least recently used record.
  void EvictOne() TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  const string name_;
  const bool has_size_limit_;
  const bool auto_resize_;
  size_t pool_size_limit_;
  std::unique_ptr<SubAllocator> allocator_;
  std::unique_ptr<RoundUpInterface> size_rounder_;
  mutex mutex_;
  std::multimap<const size_t, PtrRecord*> pool_ TF_GUARDED_BY(mutex_);
  PtrRecord* lru_head_ TF_GUARDED_BY(mutex_) = nullptr;
  PtrRecord* lru_tail_ TF_GUARDED_BY(mutex_) = nullptr;
  int64 get_from_pool_count_ TF_GUARDED_BY(mutex_) = 0;
  int64 put_count_ TF_GUARDED_BY(mutex_) = 0;
  int64 allocated_count_ TF_GUARDED_BY(mutex_) = 0;
  int64 evicted_count_ TF_GUARDED_BY(mutex_) = 0;
};

// Do-nothing rounder. Passes through sizes unchanged.
class NoopRounder : public RoundUpInterface {
 public:
  size_t RoundUp(size_t num_bytes) override { return num_bytes; }
};

// Power of 2 rounder: rounds up to nearest power of 2 size.
class Pow2Rounder : public RoundUpInterface {
 public:
  size_t RoundUp(size_t num_bytes) override {
    return 1uLL << Log2Ceiling64(num_bytes);
  }
};

class BasicCPUAllocator : public SubAllocator {
 public:
  BasicCPUAllocator(int numa_node, const std::vector<Visitor>& alloc_visitors,
                    const std::vector<Visitor>& free_visitors)
      : SubAllocator(alloc_visitors, free_visitors), numa_node_(numa_node) {}

  ~BasicCPUAllocator() override {}

  void* Alloc(size_t alignment, size_t num_bytes) override;

  void Free(void* ptr, size_t num_bytes) override;

 private:
  int numa_node_;

  TF_DISALLOW_COPY_AND_ASSIGN(BasicCPUAllocator);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_POOL_ALLOCATOR_H_

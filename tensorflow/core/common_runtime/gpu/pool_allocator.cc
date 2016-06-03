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

#include "tensorflow/core/common_runtime/gpu/pool_allocator.h"

#include <errno.h>
#include <strings.h>
#include <sys/mman.h>  // for munmap

#include <map>

#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

PoolAllocator::PoolAllocator(size_t pool_size_limit, bool auto_resize,
                             SubAllocator* allocator,
                             RoundUpInterface* size_rounder, string name)
    : name_(name),
      has_size_limit_(pool_size_limit > 0),
      auto_resize_(auto_resize),
      pool_size_limit_(pool_size_limit),
      allocator_(allocator),
      size_rounder_(size_rounder),
      allocation_begun_(false) {
  if (auto_resize) {
    CHECK_LT(size_t{0}, pool_size_limit)
        << "size limit must be > 0 if auto_resize is true.";
  }
}

PoolAllocator::~PoolAllocator() { Clear(); }

namespace {
// Pools contain Chunks allocated from the underlying Allocator.
// Chunk alignment is always on kPoolAlignment boundaries.  Each Chunk
// begins with a descriptor (ChunkPrefix) that gives its size and a
// pointer to itself.  The pointer returned to the user is just past
// the ChunkPrefix.  If the user asks for a larger alignment, we will
// increase the size of the chunk, then adjust the returned user
// pointer and also re-write the ChunkPrefix.chunk_ptr value
// immediately before it.  This way the Chunk address and size can be
// recovered from the returned user pointer, regardless of alignment.
// Note that this dereferencing of the pointers means that we cannot
// handle GPU memory, only CPU memory.
struct ChunkPrefix {
  size_t num_bytes;
  void* chunk_ptr;
};
// kPoolAlignment cannot be less than the size of ChunkPrefix.
static const int kPoolAlignment = sizeof(ChunkPrefix);

void* PrepareChunk(void* chunk, size_t alignment, size_t num_bytes) {
  ChunkPrefix* cp = reinterpret_cast<ChunkPrefix*>(chunk);
  cp->num_bytes = num_bytes;
  cp->chunk_ptr = chunk;
  void* user_ptr = reinterpret_cast<void*>(cp + 1);
  if (alignment > kPoolAlignment) {
    // Move user_ptr forward to the first satisfying offset, and write
    // chunk_ptr just before it.
    size_t aligned_ptr = reinterpret_cast<size_t>(user_ptr) + alignment;
    user_ptr = reinterpret_cast<void*>(aligned_ptr & ~(alignment - 1));
    (reinterpret_cast<ChunkPrefix*>(user_ptr) - 1)->chunk_ptr = chunk;
  }
  // Safety check that user_ptr is always past the ChunkPrefix.
  CHECK_GE(user_ptr, reinterpret_cast<ChunkPrefix*>(chunk) + 1);
  return user_ptr;
}

ChunkPrefix* FindPrefix(void* user_ptr) {
  ChunkPrefix* cp = reinterpret_cast<ChunkPrefix*>(user_ptr) - 1;
  return reinterpret_cast<ChunkPrefix*>(cp->chunk_ptr);
}
}  // namespace

void* PoolAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  if (!allocation_begun_) allocation_begun_ = true;
  if (num_bytes == 0) return nullptr;

  // If alignment is larger than kPoolAlignment, increase num_bytes so that we
  // are guaranteed to be able to return an aligned ptr by advancing user_ptr
  // without overrunning the end of the chunk.
  if (alignment > kPoolAlignment) {
    num_bytes += alignment;
  }
  num_bytes += sizeof(ChunkPrefix);
  num_bytes = size_rounder_->RoundUp(num_bytes);
  PtrRecord* pr = nullptr;
  if (has_size_limit_) {
    {
      mutex_lock lock(mutex_);
      auto iter = pool_.find(num_bytes);
      if (iter == pool_.end()) {
        allocated_count_++;
        // Deliberately fall out of lock scope before
        // calling the allocator.  No further modification
        // to the pool will be performed.
      } else {
        get_from_pool_count_++;
        pr = iter->second;
        RemoveFromList(pr);
        pool_.erase(iter);
        // Fall out of lock scope and do the result without the lock held.
      }
    }
  }
  if (pr != nullptr) {
    void* r = pr->ptr;
    delete pr;
    return PrepareChunk(r, alignment, num_bytes);
  } else {
    void* ptr = allocator_->Alloc(kPoolAlignment, num_bytes);
    for (auto v : alloc_visitors_) {
      v(ptr, num_bytes);
    }
    return PrepareChunk(ptr, alignment, num_bytes);
  }
}

void PoolAllocator::DeallocateRaw(void* ptr) {
  if (ptr == nullptr) return;
  ChunkPrefix* cp = FindPrefix(ptr);
  CHECK_LE((void*)cp, (void*)ptr);
  if (!has_size_limit_ && !auto_resize_) {
    for (auto v : free_visitors_) {
      v(cp, cp->num_bytes);
    }
    allocator_->Free(cp, cp->num_bytes);
  } else {
    mutex_lock lock(mutex_);
    ++put_count_;
    while (pool_.size() >= pool_size_limit_) {
      EvictOne();
    }
    PtrRecord* pr = new PtrRecord;
    pr->num_bytes = cp->num_bytes;
    pr->ptr = cp;
    AddToList(pr);
    pool_.insert(std::make_pair(cp->num_bytes, pr));
  }
}

void PoolAllocator::Clear() {
  if (has_size_limit_) {
    mutex_lock lock(mutex_);
    for (auto iter : pool_) {
      PtrRecord* pr = iter.second;
      for (auto v : free_visitors_) {
        v(pr->ptr, pr->num_bytes);
      }
      allocator_->Free(pr->ptr, pr->num_bytes);
      delete pr;
    }
    pool_.clear();
    get_from_pool_count_ = 0;
    put_count_ = 0;
    allocated_count_ = 0;
    evicted_count_ = 0;
    lru_head_ = nullptr;
    lru_tail_ = nullptr;
  }
}

void PoolAllocator::RemoveFromList(PtrRecord* pr) {
  if (pr->prev == nullptr) {
    DCHECK_EQ(lru_head_, pr);
    lru_head_ = nullptr;
  } else {
    pr->prev->next = pr->next;
  }
  if (pr->next == nullptr) {
    DCHECK_EQ(lru_tail_, pr);
    lru_tail_ = pr->prev;
  } else {
    pr->next->prev = pr->prev;
    if (lru_head_ == nullptr) {
      lru_head_ = pr->next;
    }
  }
}

void PoolAllocator::AddToList(PtrRecord* pr) {
  pr->prev = nullptr;
  if (lru_head_ == nullptr) {
    CHECK(lru_tail_ == nullptr);
    lru_tail_ = pr;
    pr->next = nullptr;
  } else {
    pr->next = lru_head_;
    pr->next->prev = pr;
  }
  lru_head_ = pr;
}

void PoolAllocator::EvictOne() {
  DCHECK(lru_tail_ != nullptr);
  PtrRecord* prec = lru_tail_;
  RemoveFromList(prec);
  auto iter = pool_.find(prec->num_bytes);
  while (iter->second != prec) {
    ++iter;
    DCHECK(iter != pool_.end());
  }
  pool_.erase(iter);
  for (auto v : free_visitors_) {
    v(prec->ptr, prec->num_bytes);
  }
  allocator_->Free(prec->ptr, prec->num_bytes);
  delete prec;
  ++evicted_count_;
  // Auto-resizing, and warning messages.
  static const double kTolerable = 2e-3;
  static const int kCheckInterval = 1000;
  static const double kIncreaseFactor = 1.1;
  static const int kMinPoolSize = 100;
  if (0 == evicted_count_ % kCheckInterval) {
    const double eviction_rate =
        evicted_count_ / static_cast<double>(put_count_);
    const int64 alloc_request_count = allocated_count_ + get_from_pool_count_;
    const double alloc_rate =
        (alloc_request_count == 0)
            ? 0.0
            : allocated_count_ / static_cast<double>(alloc_request_count);
    static int log_counter = 0;
    // (counter increment not thread safe but it's just for logging, so we
    // don't care).
    bool should_log = ((log_counter++ % 10) == 0);
    if (should_log) {
      LOG(INFO) << "PoolAllocator: After " << alloc_request_count
                << " get requests, put_count=" << put_count_
                << " evicted_count=" << evicted_count_
                << " eviction_rate=" << eviction_rate
                << " and unsatisfied allocation rate=" << alloc_rate;
    }
    if (auto_resize_ && (eviction_rate > kTolerable) &&
        (alloc_rate > kTolerable)) {
      size_t new_size_limit = (pool_size_limit_ < kMinPoolSize)
                                  ? kMinPoolSize
                                  : (kIncreaseFactor * pool_size_limit_);
      if (should_log) {
        LOG(INFO) << "Raising pool_size_limit_ from " << pool_size_limit_
                  << " to " << new_size_limit;
      }
      pool_size_limit_ = new_size_limit;
      // Reset all the counters so that ratios are relative to new sizes
      // at next test interval.
      put_count_ = 0;
      allocated_count_ = 0;
      evicted_count_ = 0;
      get_from_pool_count_ = 0;
    }
  }
}

void PoolAllocator::AddAllocVisitor(Visitor visitor) {
  mutex_lock lock(mutex_);
  CHECK(!allocation_begun_)
      << "AddAllocVisitor may not be called after pool allocation "
      << "has begun.";
  alloc_visitors_.push_back(visitor);
}

void PoolAllocator::AddFreeVisitor(Visitor visitor) {
  mutex_lock lock(mutex_);
  CHECK(!allocation_begun_)
      << "AddFreeVisitor may not be called after pool allocation "
      << "has begun.";
  free_visitors_.push_back(visitor);
}

}  // namespace tensorflow

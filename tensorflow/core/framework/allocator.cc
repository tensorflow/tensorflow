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

#include "tensorflow/core/framework/visitable_allocator.h"

#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

void AllocatorStats::Clear() {
  this->num_allocs = 0;
  this->bytes_in_use = 0;
  this->max_bytes_in_use = 0;
  this->max_alloc_size = 0;
  this->bytes_limit = 0;
}

string AllocatorStats::DebugString() const {
  return strings::Printf(
      "Limit:        %20lld\n"
      "InUse:        %20lld\n"
      "MaxInUse:     %20lld\n"
      "NumAllocs:    %20lld\n"
      "MaxAllocSize: %20lld\n",
      this->bytes_limit, this->bytes_in_use, this->max_bytes_in_use,
      this->num_allocs, this->max_alloc_size);
}

constexpr size_t Allocator::kAllocatorAlignment;

Allocator::~Allocator() {}

void RunResourceCtor(ResourceHandle* p, size_t n) {
  for (size_t i = 0; i < n; ++p, ++i) new (p) ResourceHandle();
}

void RunResourceDtor(ResourceHandle* p, size_t n) {
  for (size_t i = 0; i < n; ++p, ++i) p->~ResourceHandle();
}

// If true, cpu allocator collects more stats.
static bool cpu_allocator_collect_stats = false;
// If true, cpu allocator collects full stats.
static bool cpu_allocator_collect_full_stats = false;

void EnableCPUAllocatorStats(bool enable) {
  cpu_allocator_collect_stats = enable;
}
void EnableCPUAllocatorFullStats(bool enable) {
  cpu_allocator_collect_full_stats = enable;
}

class CPUAllocator : public VisitableAllocator {
 public:
  CPUAllocator() : allocation_begun_(false) {}

  ~CPUAllocator() override {}

  string Name() override { return "cpu"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    if (!allocation_begun_) {
      allocation_begun_ = true;
    }

    void* p = port::AlignedMalloc(num_bytes, alignment);
    if (cpu_allocator_collect_stats) {
      const std::size_t alloc_size = port::MallocExtension_GetAllocatedSize(p);
      mutex_lock l(mu_);
      ++stats_.num_allocs;
      stats_.bytes_in_use += alloc_size;
      stats_.max_bytes_in_use =
          std::max<int64>(stats_.max_bytes_in_use, stats_.bytes_in_use);
      stats_.max_alloc_size =
          std::max<int64>(stats_.max_alloc_size, alloc_size);
    }

    // visit each Visitor in alloc_visitors_
    if (p != nullptr) {
      for (const Visitor& v : alloc_visitors_) {
        v(p, num_bytes);
      }
    }

    return p;
  }

  void DeallocateRaw(void* ptr) override {
    std::size_t alloc_size;
    bool init_alloc_size = false;
    if (cpu_allocator_collect_stats) {
      alloc_size = port::MallocExtension_GetAllocatedSize(ptr);
      init_alloc_size = true;
      mutex_lock l(mu_);
      stats_.bytes_in_use -= alloc_size;
    }

    // visit each Visitor in free_visitors_
    if (ptr != nullptr) {
      if (!init_alloc_size) {
        alloc_size = port::MallocExtension_GetAllocatedSize(ptr);
        init_alloc_size = true;
      }
      for (const Visitor& v : free_visitors_) {
        v(ptr, alloc_size);
      }
    }

    port::AlignedFree(ptr);
  }

  void GetStats(AllocatorStats* stats) override {
    mutex_lock l(mu_);
    *stats = stats_;
  }

  void ClearStats() override {
    mutex_lock l(mu_);
    stats_.num_allocs = 0;
    stats_.max_bytes_in_use = stats_.bytes_in_use;
    stats_.max_alloc_size = 0;
  }

  size_t AllocatedSizeSlow(const void* ptr) override {
    return port::MallocExtension_GetAllocatedSize(ptr);
  }

  // REQUIRES: can only add visitors before the first Allocate call

  void AddAllocVisitor(Visitor visitor) override {
    mutex_lock lock(visitor_mutex_);
    CHECK(!allocation_begun_)
        << "AddAllocVisitor may not be called after allocation has begun.";
    alloc_visitors_.push_back(visitor);
  }

  void AddFreeVisitor(Visitor visitor) override {
    mutex_lock lock(visitor_mutex_);
    CHECK(!allocation_begun_)
        << "AddFreeVisitor may not be called after allocation has begun.";
    free_visitors_.push_back(visitor);
  }

 private:
  mutex mu_;
  AllocatorStats stats_ GUARDED_BY(mu_);

  // visitor_mutex_ protects write access to alloc_visitors_ and free_visitors_.
  // While write access is mutually exclusive, reads may happen concurrently.
  // This is okay because we may only append to alloc_visitors_ and
  // free_visitors_ before first allocation, and subsequently we only read these
  // vectors.
  mutex visitor_mutex_;
  std::vector<Visitor> alloc_visitors_;
  std::vector<Visitor> free_visitors_;
  std::atomic<bool> allocation_begun_;

  TF_DISALLOW_COPY_AND_ASSIGN(CPUAllocator);
};

Allocator* cpu_allocator() {
  static Allocator* cpu_alloc = AllocatorRegistry::Global()->GetAllocator();
  if (cpu_allocator_collect_full_stats && !cpu_alloc->TracksAllocationSizes()) {
    cpu_alloc = new TrackingAllocator(cpu_alloc, true);
  }
  return cpu_alloc;
}

REGISTER_MEM_ALLOCATOR("DefaultCPUAllocator", 100, CPUAllocator);

}  // namespace tensorflow

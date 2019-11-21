/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <atomic>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// If true, cpu allocator collects more stats.
static bool cpu_allocator_collect_stats = false;

void EnableCPUAllocatorStats(bool enable) {
  cpu_allocator_collect_stats = enable;
}
bool CPUAllocatorStatsEnabled() { return cpu_allocator_collect_stats; }

static const int kMaxTotalAllocationWarnings = 1;

static const int kMaxSingleAllocationWarnings = 5;

// If cpu_allocator_collect_stats is true, warn when the total allocated memory
// exceeds this threshold.
static const double kTotalAllocationWarningThreshold = 0.5;

// Individual allocations large than this amount will trigger a warning.
static const double kLargeAllocationWarningThreshold = 0.1;

// Cache first invocation to port::AvailableRam, as it can be expensive.
static int64_t LargeAllocationWarningBytes() {
  static int64_t value = static_cast<int64>(port::AvailableRam() *
                                            kLargeAllocationWarningThreshold);
  return value;
}

static int64_t TotalAllocationWarningBytes() {
  static int64_t value = static_cast<int64>(port::AvailableRam() *
                                            kTotalAllocationWarningThreshold);
  return value;
}

namespace {

// A default Allocator for CPU devices.  ProcessState::GetCPUAllocator() will
// return a different version that may perform better, but may also lack the
// optional stats triggered by the functions above.  TODO(tucker): migrate all
// uses of cpu_allocator() except tests to use ProcessState instead.
class CPUAllocator : public Allocator {
 public:
  CPUAllocator()
      : single_allocation_warning_count_(0),
        total_allocation_warning_count_(0) {}

  ~CPUAllocator() override {}

  string Name() override { return "cpu"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    if (num_bytes > LargeAllocationWarningBytes() &&
        single_allocation_warning_count_ < kMaxSingleAllocationWarnings) {
      ++single_allocation_warning_count_;
      LOG(WARNING) << "Allocation of " << num_bytes << " exceeds "
                   << 100 * kLargeAllocationWarningThreshold
                   << "% of system memory.";
    }

    void* p = port::AlignedMalloc(num_bytes, alignment);
    if (cpu_allocator_collect_stats) {
      const std::size_t alloc_size = port::MallocExtension_GetAllocatedSize(p);
      mutex_lock l(mu_);
      ++stats_.num_allocs;
      stats_.bytes_in_use += alloc_size;
      stats_.peak_bytes_in_use =
          std::max<int64>(stats_.peak_bytes_in_use, stats_.bytes_in_use);
      stats_.largest_alloc_size =
          std::max<int64>(stats_.largest_alloc_size, alloc_size);

      if (stats_.bytes_in_use > TotalAllocationWarningBytes() &&
          total_allocation_warning_count_ < kMaxTotalAllocationWarnings) {
        ++total_allocation_warning_count_;
        LOG(WARNING) << "Total allocated memory " << stats_.bytes_in_use
                     << "exceeds " << 100 * kTotalAllocationWarningThreshold
                     << "% of system memory";
      }
    }
    return p;
  }

  void DeallocateRaw(void* ptr) override {
    if (cpu_allocator_collect_stats) {
      const std::size_t alloc_size =
          port::MallocExtension_GetAllocatedSize(ptr);
      mutex_lock l(mu_);
      stats_.bytes_in_use -= alloc_size;
    }
    port::AlignedFree(ptr);
  }

  absl::optional<AllocatorStats> GetStats() override {
    mutex_lock l(mu_);
    return stats_;
  }

  void ClearStats() override {
    mutex_lock l(mu_);
    stats_.num_allocs = 0;
    stats_.peak_bytes_in_use = stats_.bytes_in_use;
    stats_.largest_alloc_size = 0;
  }

  size_t AllocatedSizeSlow(const void* ptr) const override {
    return port::MallocExtension_GetAllocatedSize(ptr);
  }

 private:
  mutex mu_;
  AllocatorStats stats_ GUARDED_BY(mu_);

  // Use <atomic> for single allocations to avoid mutex contention when
  // statistics are disabled.
  std::atomic<int> single_allocation_warning_count_;
  int total_allocation_warning_count_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(CPUAllocator);
};

class CPUAllocatorFactory : public AllocatorFactory {
 public:
  Allocator* CreateAllocator() override { return new CPUAllocator; }

  SubAllocator* CreateSubAllocator(int numa_node) override {
    return new CPUSubAllocator(new CPUAllocator);
  }

 private:
  class CPUSubAllocator : public SubAllocator {
   public:
    explicit CPUSubAllocator(CPUAllocator* cpu_allocator)
        : SubAllocator({}, {}), cpu_allocator_(cpu_allocator) {}

    void* Alloc(size_t alignment, size_t num_bytes) override {
      return cpu_allocator_->AllocateRaw(alignment, num_bytes);
    }

    void Free(void* ptr, size_t num_bytes) override {
      cpu_allocator_->DeallocateRaw(ptr);
    }

   private:
    CPUAllocator* cpu_allocator_;
  };
};

REGISTER_MEM_ALLOCATOR("DefaultCPUAllocator", 100, CPUAllocatorFactory);
}  // namespace

}  // namespace tensorflow

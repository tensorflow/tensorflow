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

#include <algorithm>
#include <atomic>

#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/allocator_registry.h"
#include "xla/tsl/framework/tracking_allocator.h"
#include "tsl/platform/mem.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/strcat.h"
#include "tsl/platform/stringprintf.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/lib/scoped_memory_debug_annotation.h"
#include "tsl/profiler/lib/traceme.h"

namespace tsl {

// If true, cpu allocator collects more stats.
static bool cpu_allocator_collect_stats = false;

void EnableCPUAllocatorStats() { cpu_allocator_collect_stats = true; }
void DisableCPUAllocatorStats() { cpu_allocator_collect_stats = false; }
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
  static int64_t value = static_cast<int64_t>(port::AvailableRam() *
                                              kLargeAllocationWarningThreshold);
  return value;
}

static int64_t TotalAllocationWarningBytes() {
  static int64_t value = static_cast<int64_t>(port::AvailableRam() *
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

  ~CPUAllocator() override = default;

  string Name() override { return "cpu"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    if (num_bytes > static_cast<size_t>(LargeAllocationWarningBytes()) &&
        single_allocation_warning_count_ < kMaxSingleAllocationWarnings) {
      ++single_allocation_warning_count_;
      LOG(WARNING) << "Allocation of " << num_bytes << " exceeds "
                   << 100 * kLargeAllocationWarningThreshold
                   << "% of free system memory.";
    }

    void* p = port::AlignedMalloc(num_bytes, alignment);
    if (cpu_allocator_collect_stats) {
      const std::size_t alloc_size = port::MallocExtension_GetAllocatedSize(p);
      mutex_lock l(mu_);
      ++stats_.num_allocs;
      stats_.bytes_in_use += alloc_size;
      stats_.peak_bytes_in_use =
          std::max<int64_t>(stats_.peak_bytes_in_use, stats_.bytes_in_use);
      stats_.largest_alloc_size =
          std::max<int64_t>(stats_.largest_alloc_size, alloc_size);

      if (stats_.bytes_in_use > TotalAllocationWarningBytes() &&
          total_allocation_warning_count_ < kMaxTotalAllocationWarnings) {
        ++total_allocation_warning_count_;
        LOG(WARNING) << "Total allocated memory " << stats_.bytes_in_use
                     << "exceeds " << 100 * kTotalAllocationWarningThreshold
                     << "% of free system memory";
      }
      if (p != nullptr) {
        AddTraceMe("MemoryAllocation", p, num_bytes, alloc_size);
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
      AddTraceMe("MemoryDeallocation", ptr, 0, alloc_size);
    }
    port::AlignedFree(ptr);
  }

  void AddTraceMe(absl::string_view traceme_name, const void* chunk_ptr,
                  std::size_t req_bytes, std::size_t alloc_bytes) {
    tsl::profiler::TraceMe::InstantActivity(
        [this, traceme_name, chunk_ptr, req_bytes,
         alloc_bytes]() TF_NO_THREAD_SAFETY_ANALYSIS {
          const auto& annotation =
              tsl::profiler::ScopedMemoryDebugAnnotation::CurrentAnnotation();
          return tsl::profiler::TraceMeEncode(
              traceme_name, {{"allocator_name", Name()},
                             {"bytes_reserved", stats_.bytes_reserved},
                             {"bytes_allocated", stats_.bytes_in_use},
                             {"peak_bytes_in_use", stats_.peak_bytes_in_use},
                             {"requested_bytes", req_bytes},
                             {"allocation_bytes", alloc_bytes},
                             {"addr", reinterpret_cast<uint64>(chunk_ptr)},
                             {"tf_op", annotation.pending_op_name},
                             {"id", annotation.pending_step_id},
                             {"region_type", annotation.pending_region_type},
                             {"data_type", annotation.pending_data_type},
                             {"shape", annotation.pending_shape_func()}});
        },
        /*level=*/tsl::profiler::TraceMeLevel::kInfo);
  }

  absl::optional<AllocatorStats> GetStats() override {
    if (!cpu_allocator_collect_stats) return absl::nullopt;
    mutex_lock l(mu_);
    return stats_;
  }

  bool ClearStats() override {
    if (!cpu_allocator_collect_stats) return false;
    mutex_lock l(mu_);
    stats_.num_allocs = 0;
    stats_.peak_bytes_in_use = stats_.bytes_in_use;
    stats_.largest_alloc_size = 0;
    return true;
  }

  size_t AllocatedSizeSlow(const void* ptr) const override {
    return port::MallocExtension_GetAllocatedSize(ptr);
  }

  AllocatorMemoryType GetMemoryType() const override {
    return AllocatorMemoryType::kHostPageable;
  }

 private:
  mutex mu_;
  AllocatorStats stats_ TF_GUARDED_BY(mu_);

  // Use <atomic> for single allocations to avoid mutex contention when
  // statistics are disabled.
  std::atomic<int> single_allocation_warning_count_;
  int total_allocation_warning_count_ TF_GUARDED_BY(mu_);

  CPUAllocator(const CPUAllocator&) = delete;
  void operator=(const CPUAllocator&) = delete;
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

    void* Alloc(size_t alignment, size_t num_bytes,
                size_t* bytes_received) override {
      tsl::profiler::TraceMe traceme("CPUSubAllocator::Alloc");
      *bytes_received = num_bytes;
      return cpu_allocator_->AllocateRaw(alignment, num_bytes);
    }

    void Free(void* ptr, size_t num_bytes) override {
      tsl::profiler::TraceMe traceme("CPUSubAllocator::Free");
      cpu_allocator_->DeallocateRaw(ptr);
    }

    bool SupportsCoalescing() const override { return false; }

    AllocatorMemoryType GetMemoryType() const override {
      return cpu_allocator_->GetMemoryType();
    }

   private:
    CPUAllocator* cpu_allocator_;
  };
};

REGISTER_MEM_ALLOCATOR("DefaultCPUAllocator", 100, CPUAllocatorFactory);
}  // namespace

}  // namespace tsl

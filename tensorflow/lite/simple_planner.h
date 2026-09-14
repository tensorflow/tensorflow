/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_SIMPLE_PLANNER_H_
#define TENSORFLOW_LITE_SIMPLE_PLANNER_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/util.h"

namespace tflite {

// A structure to keep heap allocation records. This structure is used by
// SimplePlanner::allocs_.
struct SimpleAlloc {
  SimpleAlloc() { reset(); }

  // Size of allocation.
  size_t size;
  // The index of the node that first needs to use this tensor.
  int32_t node;
  // Allocated heap memory address of allocation.
  char* ptr;

  // Reset member variables.
  inline void reset() {
    size = 0;
    node = 0;
    ptr = nullptr;
  }

  inline bool is_allocated() const { return ptr != nullptr; }

  // Allocate heap memory for a tensor with the given size and first_node
  // information. Returns true on success, false on failure.
  inline bool alloc(size_t new_size, int32_t new_first_node,
                    TfLiteAllocator* allocator = nullptr) {
    if (new_size == 0) {
      return false;
    }
    assert(ptr == nullptr);
    if (allocator != nullptr && allocator->allocate != nullptr) {
      ptr = static_cast<char*>(allocator->allocate(allocator->data, new_size,
                                                   kDefaultTensorAlignment));
    } else {
      ptr = static_cast<char*>(malloc(new_size));
    }
    if (ptr == nullptr) {
      return false;
    }
    size = new_size;
    node = new_first_node;
    return true;
  }

  // Free allocated heap memory and reset member variables.
  inline void free(TfLiteAllocator* allocator = nullptr) {
    if (ptr) {
      if (allocator != nullptr && allocator->deallocate != nullptr) {
        allocator->deallocate(allocator->data, ptr, size,
                              kDefaultTensorAlignment);
      } else {
        ::free(ptr);
      }
    }
    reset();
  }

  // Free allocated heap memory but preserve size and node information.
  inline void release(TfLiteAllocator* allocator = nullptr) {
    if (ptr) {
      if (allocator != nullptr && allocator->deallocate != nullptr) {
        allocator->deallocate(allocator->data, ptr, size,
                              kDefaultTensorAlignment);
      } else {
        ::free(ptr);
      }
      ptr = nullptr;
    }
  }
};

// A memory planner that makes allocations using malloc()/free() or a custom
// allocator, designed for AddressSanitizer and per-tensor lifecycle management.
class SimplePlanner : public MemoryPlanner {
 public:
  // Ownership of 'context' is not taken and it must remain until the
  // planner is destroyed.
  explicit SimplePlanner(TfLiteContext* context,
                         std::unique_ptr<GraphInfo> graph_info,
                         bool preserve_all_tensors = false,
                         bool enable_reclamation = true,
                         TfLiteAllocator* allocator = nullptr);
  ~SimplePlanner() override;
  SimplePlanner(const SimplePlanner&) = delete;
  SimplePlanner& operator=(const SimplePlanner&) = delete;

  TfLiteStatus ResetAllocations() override;
  TfLiteStatus ResetAllocationsAfter(int node) override;
  TfLiteStatus PlanAllocations() override;
  TfLiteStatus ExecuteAllocations(int first_node, int last_node) override;
  TfLiteStatus ReleaseNonPersistentMemory() override;
  TfLiteStatus AcquireNonPersistentMemory() override;
  bool HasNonPersistentMemory() override;
  void DumpDebugInfo(const std::vector<int>& execution_plan) const override {}
  void GetAllocInfo(size_t* arena_size,
                    size_t* arena_persist_size) const override;

  // Execution lifecycle hooks for reclamation mode.
  TfLiteStatus BeginInvocation() override;
  TfLiteStatus BeforeNode(int execution_plan_index) override;
  TfLiteStatus AfterNode(int execution_plan_index) override;
  void EndInvocation(bool completed_successfully) override;

  // Configuration and options.
  void SetReclamationMode(bool enable) override {
    if (reclamation_mode_ != enable) {
      reclamation_mode_ = enable;
      alloc_node_.clear();
      allocate_before_.clear();
      release_after_.clear();
    }
  }
  bool is_reclamation_enabled() const { return reclamation_mode_; }

  void SetPreserveAllTensors(bool preserve) override {
    if (preserve_all_tensors_ != preserve) {
      preserve_all_tensors_ = preserve;
      alloc_node_.clear();
      allocate_before_.clear();
      release_after_.clear();
    }
  }
  bool preserve_all_tensors() const { return preserve_all_tensors_; }

  void SetCustomAllocator(TfLiteAllocator* allocator) {
    allocator_ = allocator;
  }

  // Allocation telemetry and metrics.
  size_t current_outstanding_bytes() const {
    return current_outstanding_bytes_;
  }
  size_t peak_outstanding_bytes() const { return peak_outstanding_bytes_; }
  int64_t total_allocations() const { return total_allocations_; }
  int64_t total_deallocations() const { return total_deallocations_; }
  void ResetStats() {
    peak_outstanding_bytes_ = current_outstanding_bytes_;
    total_allocations_ = 0;
    total_deallocations_ = 0;
  }

  // Query tensor pinning status.
  bool IsPinned(int tensor_index) const {
    if (tensor_index >= 0 &&
        tensor_index < static_cast<int>(is_pinned_.size())) {
      return is_pinned_[tensor_index];
    }
    return false;
  }

  const std::vector<int>& GetAllocationsBefore(int execution_plan_index) const {
    static const std::vector<int> kEmpty;
    if (execution_plan_index >= 0 &&
        execution_plan_index < static_cast<int>(allocate_before_.size())) {
      return allocate_before_[execution_plan_index];
    }
    return kEmpty;
  }

  const std::vector<int>& GetReleasesAfter(int execution_plan_index) const {
    static const std::vector<int> kEmpty;
    if (execution_plan_index >= 0 &&
        execution_plan_index < static_cast<int>(release_after_.size())) {
      return release_after_[execution_plan_index];
    }
    return kEmpty;
  }

 private:
  // Free all allocations.
  void FreeAllAllocations();

  // Assign absolute memory location to a tensor.
  TfLiteStatus ResolveTensorAllocation(int tensor_index);

  // Initialize pinning status for tensors.
  void UpdatePinnedStatus();

  TfLiteContext* context_;
  std::unique_ptr<GraphInfo> graph_info_;
  bool preserve_all_tensors_ = false;
  bool reclamation_mode_ = false;
  TfLiteAllocator* allocator_ = nullptr;
  bool non_persistent_memory_available_ = true;

  // Stores allocation data for all tensors.
  std::vector<SimpleAlloc> allocs_;

  // First node that uses the tensor.
  std::vector<int32_t> alloc_node_;

  // Last node that uses the tensor.
  std::vector<int32_t> dealloc_node_;

  // Whether a tensor is pinned (not reclaimable during execution).
  std::vector<bool> is_pinned_;

  // Execution-time schedules (indexed by execution_plan_index):
  std::vector<std::vector<int>> allocate_before_;
  std::vector<std::vector<int>> release_after_;

  // Telemetry counters.
  size_t current_outstanding_bytes_ = 0;
  size_t peak_outstanding_bytes_ = 0;
  int64_t total_allocations_ = 0;
  int64_t total_deallocations_ = 0;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_SIMPLE_PLANNER_H_

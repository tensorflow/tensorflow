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
#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow/lite/c/common.h"
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

  // Allocate heap memory for a tensor with the given size and first_node
  // information.
  inline bool alloc(size_t new_size, int32_t new_first_node) {
    if (new_size == 0) {
      return false;
    }
    size = new_size;
    node = new_first_node;
    assert(ptr == nullptr);
    ptr = static_cast<char*>(malloc(new_size));
    return true;
  }

  // Free allocated heap memory and reset member variables.
  inline void free() {
    if (ptr) {
      ::free(ptr);
    }
    reset();
  }
};

// A memory planner that makes all the allocations using malloc()/free().
//
// This is simple implementation of MemoryPlanner which uses malloc()/free()
// instead of memory areana. This planner is designed for AddressSanitizer.
class SimplePlanner : public MemoryPlanner {
 public:
  // Ownership of 'context' is not taken and it must remain util the
  // ArenaPlanner is destroyed. The inputs to the graph will not share
  // memory with any other tensor, effectively preserving them until the end
  // of inference.
  SimplePlanner(TfLiteContext* context, std::unique_ptr<GraphInfo> graph_info);
  ~SimplePlanner() override;
  SimplePlanner(const SimplePlanner&) = delete;
  SimplePlanner& operator=(const SimplePlanner&) = delete;

  TfLiteStatus ResetAllocations() override;
  TfLiteStatus ResetAllocationsAfter(int node) override;
  TfLiteStatus PlanAllocations() override;
  TfLiteStatus ExecuteAllocations(int first_node, int last_node) override;
  TfLiteStatus ReleaseNonPersistentMemory() override;
  TfLiteStatus AcquireNonPersistentMemory() override;
  bool HasNonPersistentMemory() override { return true; };
  void DumpDebugInfo(const std::vector<int>& execution_plan) const override{};

 private:
  // Free all the all allocations.
  void FreeAllAllocations();

  // Assign absolute memory location to a tensor.
  TfLiteStatus ResolveTensorAllocation(int tensor_index);

  TfLiteContext* context_;
  std::unique_ptr<GraphInfo> graph_info_;

  // Stores allocation data for all tensors.
  std::vector<SimpleAlloc> allocs_;

  // First node, that uses the tensor. It needs to be allocated before
  // execution of the node's operation.
  std::vector<int32_t> alloc_node_;

  // Last node, that uses the tensor. It can be deallocated after execution of
  // the node's operation.
  std::vector<int32_t> dealloc_node_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_SIMPLE_PLANNER_H_

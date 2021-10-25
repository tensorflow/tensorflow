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
#ifndef TENSORFLOW_LITE_ARENA_PLANNER_H_
#define TENSORFLOW_LITE_ARENA_PLANNER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/simple_memory_arena.h"
#include "tensorflow/lite/util.h"

namespace tflite {

constexpr const int kDefaultArenaAlignment = 64;
struct AllocationInfo;

// A memory planner that makes all the allocations using arenas.
//
// Before a model is executed by the interpreter, this class determines when
// each tensor needs to be allocated and deallocated, and preallocates all the
// necessary memory (the PlanAllocations phase). It then assigns portions of
// this memory buffer to each tensor (the ExecuteAllocations phase). Tensors may
// share some of the buffer if a tensor B is to be allocated after another
// tensor A has been deallocated.
//
// If dynamic tensors are used the planning steps can be repeated during model
// execution. Since dynamic tensors don't have sizes until after the
// corresponding operation is executed, this class supports incremental
// planning.
class ArenaPlanner : public MemoryPlanner {
 public:
  // Ownership of 'context' is not taken and it must remain util the
  // ArenaPlanner is destroyed. The inputs to the graph will not share
  // memory with any other tensor, effectively preserving them until the end
  // of inference.
  ArenaPlanner(TfLiteContext* context, std::unique_ptr<GraphInfo> graph_info,
               bool preserve_all_tensors, int tensor_alignment);
  ~ArenaPlanner() override;
  ArenaPlanner(const ArenaPlanner&) = delete;
  ArenaPlanner& operator=(const ArenaPlanner&) = delete;

  TfLiteStatus ResetAllocations() override;
  TfLiteStatus ResetAllocationsAfter(int node) override;
  TfLiteStatus PlanAllocations() override;
  TfLiteStatus ExecuteAllocations(int first_node, int last_node) override;
  TfLiteStatus ReleaseNonPersistentMemory() override;
  TfLiteStatus AcquireNonPersistentMemory() override;
  bool HasNonPersistentMemory() override;
  void DumpDebugInfo(const std::vector<int>& execution_plan) const override;

  // Returns the base arena location for a given allocation type.
  std::intptr_t BasePointer(TfLiteAllocationType type);

 private:
  // Make sure all the arenas have reserved enough memory to store all their
  // tensors.
  TfLiteStatus Commit();

  // Returns vector of tensor number ordered by the following algorithm.
  // Comparator to sort tensors for the allocation algorithm:
  // - Tensors that have lifespan through the whole model inference time go
  // first;
  // - Other tensors (e.g. intermediate and temporary ones) are sorted in
  // non-increasing order of their size. If sizes of two tensors are equal, the
  // one that needs to be allocated earlier goes first.
  std::vector<int32_t> CreateTensorAllocationVector(int first_node,
                                                    int last_node);

  // Traverse the allocation queue and reserve space in the appropriate arena
  // for all tensors affected by ops in the interval [first_node, last_node].
  TfLiteStatus CalculateAllocations(int first_node, int last_node);

  // Assign absolute memory location to a tensor, based on its relative
  // position inside the corresponding arena buffer.
  TfLiteStatus ResolveTensorAllocation(int tensor_index);

  // Register an allocation for all internal (temporary) tensors of
  // 'node_index'.
  TfLiteStatus CalculateAllocationOfInternalTensors(int node_index);

  // Register a deallocation for all internal (temporary) tensors of
  // 'node_index'.
  TfLiteStatus CalculateDeallocationOfInternalTensors(int node_index);

  TfLiteContext* context_;
  std::unique_ptr<GraphInfo> graph_info_;

  // Stores allocation data for all tensors.
  std::vector<ArenaAllocWithUsageInterval> allocs_;

  // First node, that uses the tensor. It needs to be allocated before
  // execution of the node's operation.
  std::vector<int32_t> alloc_node_;

  // Last node, that uses the tensor. It can be deallocated after execution of
  // the node's operation.
  std::vector<int32_t> dealloc_node_;

  // Raw memory buffer that is allocated for all temporary and graph outputs
  // that are declared kTfLiteArenaRw.
  SimpleMemoryArena arena_;

  // Raw memory buffer that is allocated for persistent tensors that are
  // declared as kTfLiteArenaRwPersistent.
  SimpleMemoryArena persistent_arena_;

  // If true, then no overlapping of memory areas is done, meaning intermediate
  // tensors and temporary tensors can be queried after running.
  // (modulo running delegates)
  bool preserve_all_tensors_;

  // Number of bytes that tensor buffers should be aligned to.
  int tensor_alignment_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_ARENA_PLANNER_H_

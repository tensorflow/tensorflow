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

namespace tflite {

// Memory allocation tuning
constexpr const int kDefaultArenaAlignment = 64;
constexpr const int kDefaultTensorAlignment = 64;

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
//
// TODO(b/127354079): Remove the constrain below when the issue is fixed.
// WARNING: MemoryPlanner's behavior must be deterministic. If the first N
// nodes are unchanged, it must produce exactly the same allocation plan for
// the first N nodes.
class ArenaPlanner : public MemoryPlanner {
 public:
  // Ownership of 'context' is not taken and it must remain util the
  // ArenaPlanner is destroyed. If 'preserve_inputs' is true the inputs to the
  // graph will not share memory with any other tensor, effectively preserving
  // them until the end of inference.
  ArenaPlanner(TfLiteContext* context, std::unique_ptr<GraphInfo> graph_info,
               bool preserve_inputs, bool preserve_intermediates,
               int tensor_alignment = kDefaultTensorAlignment);
  ~ArenaPlanner() override;
  ArenaPlanner(const ArenaPlanner&) = delete;
  ArenaPlanner& operator=(const ArenaPlanner&) = delete;

  TfLiteStatus ResetAllocations() override;
  TfLiteStatus PlanAllocations() override;
  TfLiteStatus ExecuteAllocations(int first_node, int last_node) override;
  TfLiteStatus ReleaseNonPersistentMemory() override;
  TfLiteStatus AcquireNonPersistentMemory() override;
  bool HasNonPersistentMemory() override;

  // Returns the base arena location for a given allocation type.
  std::intptr_t BasePointer(TfLiteAllocationType type);

 private:
  // Make sure all the arenas have reserved enough memory to store all their
  // tensors.
  TfLiteStatus Commit();

  // Traverse the allocation queue and reserve space in the appropriate arena
  // for all tensors affected by ops in the interval [first_node, last_node].
  TfLiteStatus CalculateAllocations(int first_node, int last_node);

  // Assign absolute memory location to a tensor, based on its relative
  // position inside the corresponding arena buffer.
  TfLiteStatus ResolveTensorAllocation(int tensor_index);

  // Register an allocation for the given tensor.
  TfLiteStatus CalculateTensorAllocation(int tensor_index);

  // Register a deallocation for the given tensor.
  TfLiteStatus CalculateTensorDeallocation(int tensor_index);

  // Register an allocation for all internal (temporary) tensors of
  // 'node_index'.
  TfLiteStatus CalculateAllocationOfInternalTensors(int node_index);

  // Register a deallocation for all internal (temporary) tensors of
  // 'node_index'.
  TfLiteStatus CalculateDeallocationOfInternalTensors(int node_index);

  TfLiteContext* context_;
  std::unique_ptr<GraphInfo> graph_info_;

  // Stores allocation data for all tensors.
  std::vector<ArenaAlloc> allocs_;

  // A chronological list of instructions to allocate and deallocate tensors,
  // reflecting the way they are used in the graph.
  std::vector<AllocationInfo> alloc_queue_;

  // Raw memory buffer that is allocated for all temporary and graph outputs
  // that are declared kTfLiteArenaRw.
  SimpleMemoryArena arena_;

  // Raw memory buffer that is allocated for persistent tensors that are
  // declared as kTfLiteArenaRwPersistent.
  SimpleMemoryArena persistent_arena_;

  // Ensure that the memory self-allocated for inputs is never reused by the
  // allocator. This allows for example, multiple runs without getting
  // unpredictable results.
  bool preserve_inputs_;

  // If true, then no overlapping of memory areas is done, meaning intermediate
  // results can be queried after running (modulo running delegates).
  bool preserve_intermediates_;

  // Number of bytes that tensor buffers should be aligned to.
  int tensor_alignment_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_ARENA_PLANNER_H_

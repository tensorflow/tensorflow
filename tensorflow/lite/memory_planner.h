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
#ifndef TENSORFLOW_LITE_MEMORY_PLANNER_H_
#define TENSORFLOW_LITE_MEMORY_PLANNER_H_

#include <vector>

#include "tensorflow/lite/core/c/common.h"

namespace tflite {

// A MemoryPlanner is responsible for planning and executing a number of
// memory-related operations that are necessary in TF Lite.
class MemoryPlanner {
 public:
  virtual ~MemoryPlanner() {}

  // Plans the necessary memory allocations. This is the MemoryPlanner's
  // pre-processing step and is called when the graph structure is known but
  // actual size of the tensors is not.
  virtual TfLiteStatus PlanAllocations() = 0;

  // Allocates the necessary memory to execute all nodes in the interval
  // [first_node, last_node].
  virtual TfLiteStatus ExecuteAllocations(int first_node, int last_node) = 0;

  // Invalidates allocations made earlier. This is called when tensors sizes
  // have changed. All planned allocations remain, but can't be used until
  // ExecuteAllocations() is called.
  virtual TfLiteStatus ResetAllocations() = 0;

  // Invalidates allocations after the given node execution.
  virtual TfLiteStatus ResetAllocationsAfter(int node) = 0;

  // NOTE: The following two methods modify the data pointers for all tensors on
  // the non-persistent arena (inputs, outputs, intermediates). If the user has
  // manually set the pointers for any of these, they would need to be set
  // again.

  // This releases memory allocated for non-persistent tensors.
  // It does NOT clear the allocation plan, but the memory can't be used
  // until AcquireNonPersistentMemory() is called.
  // It is safe to call Reset/PlanAllocations after this method, without calling
  // ReleaseTemporaryAllocations in case tensor sizes change.
  virtual TfLiteStatus ReleaseNonPersistentMemory() = 0;

  // Allocates the necessary memory to contain non-persistent tensors.
  virtual TfLiteStatus AcquireNonPersistentMemory() = 0;

  // Returns true if the non-persistent memory is available.
  virtual bool HasNonPersistentMemory() = 0;

  // Dumps the memory planning information against the specified op node
  // execution plan (i.e. `execution_plan`) for the purpose of debugging.
  virtual void DumpDebugInfo(const std::vector<int>& execution_plan) const = 0;

  // Returns a map of allocation information. It's only used for debugging.
  virtual void GetAllocInfo(size_t *arena_size,
                            size_t *arena_persist_size) const = 0;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MEMORY_PLANNER_H_

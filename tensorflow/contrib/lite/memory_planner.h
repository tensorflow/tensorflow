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
#ifndef TENSORFLOW_CONTRIB_LITE_MEMORY_PLANNER_H_
#define TENSORFLOW_CONTRIB_LITE_MEMORY_PLANNER_H_

#include "tensorflow/contrib/lite/context.h"

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

  // Invalidates allocations made earliers. This is called when tensors sizes
  // have change. All planned allocations remain, but can't be used until
  // ExecuteAllocations() is called.
  virtual TfLiteStatus ResetAllocations() = 0;
};

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_MEMORY_PLANNER_H_

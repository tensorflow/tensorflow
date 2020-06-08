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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_TYPES_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_TYPES_H_

#include <cstdint>
#include <memory>
#include <vector>

namespace tflite {
namespace gpu {

using TaskId = size_t;
using UsageGraph = std::vector<std::vector<size_t>>;

// Record, containing tensor size/shape and IDs of the first and the last task,
// that use this tensor as input or output. For example: tensor #3 with size
// tensor_size=65536 is first introduced in program #2 (first_task=2) and used
// for the last time in program #7 (last_task=7).
template <typename TensorSizeT>
struct TensorUsageRecord {
  TensorSizeT tensor_size;
  TaskId first_task;
  TaskId last_task;

  TensorUsageRecord(TensorSizeT size, TaskId first, TaskId last)
      : tensor_size(size), first_task(first), last_task(last) {}

  // Default order of tensor usage records is increasing order of first_task.
  bool operator<(const TensorUsageRecord<TensorSizeT>& other) const {
    return first_task < other.first_task;
  }
};

// Information about assignment of tensors to shared objects
template <typename TensorSizeT>
struct ObjectsAssignment {
  // shared_object_ids_[i] is ID of shared object, that tensor i will be using.
  std::vector<size_t> object_ids;
  // shared_object_sizes_[i] is a size of shared object with ID equal to i.
  std::vector<TensorSizeT> object_sizes;
};

// Information about assignment of tensors to offsets for the case, when all of
// them are going to be allocated in one continuous memory block.
struct OffsetsAssignment {
  std::vector<size_t> offsets;
  size_t total_size;
};

// This function takes the graph of tensor dependencies as an input and returns
// reallocation graph as an output. Tensor dependencies graph is a directed
// graph, with edge x->y existing if and only if tensor x is used for
// calculating of tensor y. This graph can be generated with following
// pseudocode: for op in operations do
//   for input_tensor in op.input_tensors do
//       for output_tensor in op.output_tensors do
//         if both input_tensor and output_tensor are intermediate tensors then
//           deps_graph[input_tensor].push_back(output_tensor)
// Reallocation graph is an undirected graph, that has edge x<->y if and only if
// tensors x and y can share memory in ANY order of operations parallel
// execution.
UsageGraph ReallocationGraph(const UsageGraph& deps_graph);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_TYPES_H_

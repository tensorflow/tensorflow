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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

using TaskId = size_t;

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

// Converts given assignment of tensors to shared objects to the assignment of
// the same tensors to offsets in continuous memory block.
OffsetsAssignment ObjectsToOffsets(
    const ObjectsAssignment<size_t>& obj_assignment);

enum class MemoryStrategy {
  // Naive strategy is to allocate each object separately.
  // Can be useful for debugging to see all intermediate outputs.
  NAIVE,

  // Equality strategy allows to reuse the same part of memory for several
  // tensors with the same size, but non-intersecting usage intervals.
  EQUALITY,

  // Greedy strategy uses greedy algorithm, iterating through all the tensors in
  // order of their first_task, to reuse memory from tensors, that
  // won't be used anymore, for new ones.
  GREEDY_IN_ORDER,

  // Greedy by size strategy uses greedy algorithm, iterating through all the
  // tasks in non-increasing of their breadth, and calculating allocations for
  // tensors used in these tasks. By breadth of the task we understand sum of
  // sizes of all tensors in its TaskProfile.
  GREEDY_BY_BREADTH,

  // Greedy by size strategy uses greedy algorithm, iterating through all the
  // tensors in non-increasing of their size, to reuse memory from tensors, that
  // won't be used anymore, for new ones.
  GREEDY_BY_SIZE,

  // Choose greedy strategy from several fast algorithms, that provides best
  // memory allocation for the given usage records.
  GREEDY_BEST,

  // Mincostflow strategy consists of building auxiliary flow graph and solving
  // the minimum-cost flow problem in it. In the end edges with zero residual
  // capacity determine assignment of shared objects to tensors.
  MINCOSTFLOW,
};

// Calculates the assignement of shared objects to given tensors, including
// objects' sizes. Initial tensor sizes are given as size_t. This function is
// intended to use with GPU buffers and one-dimensional textures.
Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<size_t>* assignment);

// Calculates the assignement of shared objects to given tensors, including
// objects' sizes. Initial tensor sizes are given as BHWC. This function is
// intended to use with OpenCL textures.
Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<BHWC>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<BHWC>* assignment);

// Calculates the assignement of shared objects to given tensors, including
// objects' sizes. Initial tensor sizes are given as uint2. This function is
// intended to use with OpenGL textures.
Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<uint2>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<uint2>* assignment);

// Calculates the assignement of shared objects to given tensors, including
// objects' sizes. Initial tensor sizes are given as uint3. This function is
// intended to use with OpenGL textures.
Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<uint3>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<uint3>* assignment);

// Calculates the assignement of tensors to offsets, considering those tensors
// are going to be allocated in one continuous memory block.
Status AssignOffsetsToTensors(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    const MemoryStrategy& strategy, OffsetsAssignment* assignment);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_H_

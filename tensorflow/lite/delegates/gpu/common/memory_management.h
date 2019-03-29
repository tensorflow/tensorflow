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
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

using TaskId = size_t;

// Record, containing tensor size and IDs of the first and the last task, that
// use this tensor as input or output.
// For example: tensor #3 with size tensor_size=65536 is first introduced in
// program #2 (first_task=2) and used for the last time in program #7
// (last_task=7).
struct TensorUsageRecord {
  uint32_t tensor_size;
  TaskId first_task;
  TaskId last_task;

  TensorUsageRecord(uint32_t size, TaskId first, TaskId last)
      : tensor_size(size), first_task(first), last_task(last) {}
};

// Information about assignment of tensors to shared objects
struct ObjectsAssignment {
  // shared_object_ids_[i] is ID of shared object, that tensor i will be using.
  std::vector<size_t> object_ids;
  // shared_object_sizes_[i] is a size of shared object with ID equal to i.
  std::vector<uint32_t> object_sizes;
};

enum class MemoryStrategy {
  // Naive strategy is to allocate each object separately.
  // Can be useful for debugging to see all intermediate outputs.
  NAIVE,

  // Greedy strategy uses greedy algorithm to reuse memory from tensors, that
  // won't be used anymore, for new ones.
  GREEDY,
};

// Calculates the assignement of shared objects to given tensors, including
// objects' sizes.
Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord>& usage_records,
    const MemoryStrategy& strategy, ObjectsAssignment* assignment);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_H_

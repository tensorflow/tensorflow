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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_INTERNAL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_INTERNAL_H_

#include <stddef.h>

#include <limits>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

const size_t kNotAssigned = std::numeric_limits<size_t>::max();

// This structure is used to save the initial indices of usage records after
// they are sorted.
template <typename TensorSizeT>
struct TensorUsageWithIndex {
  const TensorUsageRecord<TensorSizeT>* usage_record;
  size_t idx;

  TensorUsageWithIndex(const TensorUsageRecord<TensorSizeT>* usage_record,
                       size_t idx)
      : usage_record(usage_record), idx(idx) {}
};

bool CompareBySize(const TensorUsageWithIndex<size_t>& first,
                   const TensorUsageWithIndex<size_t>& second);

// TaskProfile is a vector with information about all intermediate tensors, that
// should exist in memory during the execution of the task. Elements of the
// vector must be sorted in non-increasing order of corresponding tensors sizes.
using TaskProfile = std::vector<TensorUsageWithIndex<size_t>>;

// Size of object, that covers both input objects (2-dimensional case).
bool IsCoveringObject(const uint2& first_object, const uint2& second_object);

// Size of object, that covers both input objects (3-dimensional case).
bool IsCoveringObject(const uint3& first_object, const uint3& second_object);

// Difference between two objects in elements count (1-dimensional case).
size_t AbsDiffInElements(const size_t first_size, const size_t second_size);

// Difference between two objects in elements count (2-dimensional case).
size_t AbsDiffInElements(const uint2& first_size, const uint2& second_size);

// Difference between two objects in elements count (3-dimensional case).
size_t AbsDiffInElements(const uint3& first_size, const uint3& second_size);

template <typename ObjectSizeT>
struct PoolRecord {
  PoolRecord(ObjectSizeT size, size_t obj_id)
      : object_size(size), object_id(obj_id) {}

  // Objects in pool are ordered by size.
  bool operator<(const PoolRecord& other) const {
    return (object_size < other.object_size) ||
           (object_size == other.object_size && object_id < other.object_id);
  }

  ObjectSizeT object_size;
  size_t object_id;
};

struct QueueRecord {
  QueueRecord(TaskId task_id, size_t obj_id)
      : last_task(task_id), object_id(obj_id) {}

  // Objects in queue are ordered by last_task.
  bool operator<(const QueueRecord& other) const {
    return (last_task > other.last_task) ||
           (last_task == other.last_task && object_id > other.object_id);
  }

  // Last task, where shared object is used.
  TaskId last_task;
  size_t object_id;
};

// Returns a vector that contains TaskProfile for each task.
std::vector<TaskProfile> CalculateTaskProfiles(
    const std::vector<TensorUsageRecord<size_t>>& usage_records);

// Iterates over all task profiles to calculate maximum at each position.
std::vector<size_t> CalculatePositionalMaximums(
    const std::vector<TensorUsageRecord<size_t>>& usage_records);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_INTERNAL_H_

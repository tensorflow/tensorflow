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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_EQUALITY_ASSIGNMENT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_EQUALITY_ASSIGNMENT_H_

#include <queue>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/internal.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

// Fast version of Equality Assignments for hashable types.
template <typename TensorSizeT>
Status EqualityAssignmentWithHash(
    const std::vector<TensorUsageRecord<TensorSizeT>>& usage_records,
    ObjectsAssignment<TensorSizeT>* assignment) {
  size_t num_records = usage_records.size();
  assignment->object_sizes.clear();
  assignment->object_ids.assign(num_records, kNotAssigned);

  // Pool is a map with size as a key and vector with ids of free shared objects
  // of this size as a value.
  absl::flat_hash_map<TensorSizeT, std::vector<size_t>> pool;
  std::priority_queue<QueueRecord> objects_in_use;
  for (size_t i = 0; i < num_records; ++i) {
    // Pop from the queue and add to the pool all objects that are no longer
    // in use at the time of execution of the first_task of i-th intermediate
    // tensor.
    while (!objects_in_use.empty() &&
           objects_in_use.top().last_task < usage_records[i].first_task) {
      auto object_id = objects_in_use.top().object_id;
      pool[assignment->object_sizes[object_id]].push_back(object_id);
      objects_in_use.pop();
    }

    const TensorSizeT tensor_size = usage_records[i].tensor_size;
    auto pool_it = pool.find(tensor_size);
    if (pool_it == pool.end() || pool_it->second.empty()) {
      // No free shared object with size equal to tensor_size. Create a new one,
      // assign i-th tensor to it and add to the queue of objects in use.
      assignment->object_ids[i] = assignment->object_sizes.size();
      assignment->object_sizes.push_back(tensor_size);
      objects_in_use.push(
          {usage_records[i].last_task, assignment->object_ids[i]});
    } else {
      // Shared object with id it->second has size equal to tensor_size. Reuse
      // this object: erase it from pool and add to the queue of objects in use.
      assignment->object_ids[i] = pool_it->second.back();
      pool_it->second.pop_back();
      objects_in_use.push(
          {usage_records[i].last_task, assignment->object_ids[i]});
    }
  }
  return OkStatus();
}

// Slower version of Equality Assignments for unhashable types.
template <typename TensorSizeT>
Status EqualityAssignment(
    const std::vector<TensorUsageRecord<TensorSizeT>>& usage_records,
    ObjectsAssignment<TensorSizeT>* assignment) {
  size_t num_records = usage_records.size();
  assignment->object_sizes.clear();
  assignment->object_ids.assign(num_records, kNotAssigned);

  // Index of operation, after execution of which the shared object can be
  // deallocated.
  std::vector<size_t> dealloc_task;
  for (size_t i = 0; i < num_records; ++i) {
    const TensorSizeT tensor_size = usage_records[i].tensor_size;
    size_t best_obj = kNotAssigned;
    for (size_t obj = 0; obj < assignment->object_sizes.size(); ++obj) {
      // Find a shared object, that has equal size with current tensor and has
      // been deallocated before the execution of its first_task.
      if (dealloc_task[obj] < usage_records[i].first_task &&
          assignment->object_sizes[obj] == tensor_size) {
        best_obj = obj;
        break;
      }
    }
    if (best_obj == kNotAssigned) {
      // No free shared object with size equal to tensor_size. Create a new one,
      // assign i-th tensor to it and save its last task as deallocation task.
      assignment->object_ids[i] = assignment->object_sizes.size();
      assignment->object_sizes.push_back(tensor_size);
      dealloc_task.push_back(usage_records[i].last_task);
    } else {
      // Shared object with id it->second has size equal to tensor_size. Reuse
      // this object and update its deallocation task.
      assignment->object_ids[i] = best_obj;
      dealloc_task[best_obj] = usage_records[i].last_task;
    }
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_EQUALITY_ASSIGNMENT_H_

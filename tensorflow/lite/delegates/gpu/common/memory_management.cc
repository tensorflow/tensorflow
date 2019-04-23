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

#include "tensorflow/lite/delegates/gpu/common/memory_management.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace {

struct PoolRecord {
  PoolRecord(uint32_t size, size_t obj_id)
      : object_size(size), object_id(obj_id) {}

  // objects in pool are ordered by size
  bool operator<(const PoolRecord& other) const {
    return (object_size < other.object_size) ||
           (object_size == other.object_size && object_id < other.object_id);
  }

  uint32_t object_size;
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

// Implements memory management with a naive algorithm.
//
// The problem of memory management is NP-complete. This implements a
// naive algorithm that assigns each tensor to a separate object ine memory.
Status NaiveAssignment(const std::vector<TensorUsageRecord>& usage_records,
                       ObjectsAssignment* assignment) {
  assignment->object_sizes.resize(usage_records.size());
  assignment->object_ids.resize(usage_records.size());
  for (size_t i = 0; i < usage_records.size(); i++) {
    auto& record = usage_records[i];
    assignment->object_ids[i] = i;
    assignment->object_sizes[i] = record.tensor_size;
  }
  return OkStatus();
}

// Implements memory management with a greedy algorithm.
//
// The problem of memory management is NP-complete. This implements a
// greedy algorithm that approximates an optimal solution with following
// heuristic:
//
//   1. Iterates through all tensor usage records and for every object reference
//      assigns shared object from the pool. When object reference is used
//      for the last time, corresponding shared object is returned back to
//      the pool.
//
//   2. Shared object pool grows when there are no free shared object
//      available.
//
//   3. Shared object size may increase when tensor requests larger size.
Status GreedyAssignment(const std::vector<TensorUsageRecord>& usage_records,
                        ObjectsAssignment* assignment) {
  assignment->object_sizes.clear();
  assignment->object_ids.resize(usage_records.size());

  // Pool of free shared objects is ordered by object size, because we perform
  // lower_bound search in it.
  std::set<PoolRecord> pool;
  // Queue of shared objects in use, ordered by their last_task.
  std::priority_queue<QueueRecord> objects_in_use;
  for (size_t i = 0; i < usage_records.size(); i++) {
    // Pop from the queue and add to the pool all objects that are no longer
    // in use at the time of execution of the first_task of i-th intermediate
    // tensor.
    while (!objects_in_use.empty() &&
           objects_in_use.top().last_task < usage_records[i].first_task) {
      auto object_id = objects_in_use.top().object_id;
      pool.insert({assignment->object_sizes[object_id], object_id});
      objects_in_use.pop();
    }
    uint32_t tensor_size = usage_records[i].tensor_size;
    if (pool.empty()) {
      // No free shared object, creating a new one, assign i-th tensor to
      // it and add to the queue of objects in use.
      assignment->object_ids[i] = assignment->object_sizes.size();
      assignment->object_sizes.push_back(tensor_size);
      objects_in_use.push(
          {usage_records[i].last_task, assignment->object_ids[i]});
    } else {
      auto best_it = pool.end();
      // Find shared object from pool, that will waste the least possible
      // amount of memory when reused for current tensor.
      auto pool_it = pool.lower_bound({tensor_size, 0});
      uint32_t size_diff = 0;
      if (pool_it != pool.end()) {
        // Try smallest shared object from pool with size >= tensor_size.
        size_diff = pool_it->object_size - tensor_size;
        best_it = pool_it;
      }
      if (pool_it != pool.begin()) {
        // Try largest shared object from pool with size < tensor_size.
        pool_it--;
        if (best_it == pool.end() ||
            tensor_size - pool_it->object_size < size_diff) {
          size_diff = tensor_size - pool_it->object_size;
          best_it = pool_it;
        }
      }
      // best_it can't be equal to pool.end(), because pool is not empty
      if (best_it == pool.end()) {
        return InternalError(
            "No shared object is found in non-empty pool in GreedyAssignment.");
      }
      size_t shared_id = best_it->object_id;
      pool.erase(best_it);
      assignment->object_ids[i] = shared_id;
      assignment->object_sizes[shared_id] =
          std::max(assignment->object_sizes[shared_id], tensor_size);
      objects_in_use.push(
          {usage_records[i].last_task, assignment->object_ids[i]});
    }
  }
  return OkStatus();
}

}  // namespace

Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord>& usage_records,
    const MemoryStrategy& strategy, ObjectsAssignment* assignment) {
  switch (strategy) {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    case MemoryStrategy::GREEDY:
      return GreedyAssignment(usage_records, assignment);
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace tflite

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
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace {

const size_t kNotAssigned = std::numeric_limits<size_t>::max();

struct PoolRecord {
  PoolRecord(size_t size, size_t obj_id)
      : object_size(size), object_id(obj_id) {}

  // Objects in pool are ordered by size.
  bool operator<(const PoolRecord& other) const {
    return (object_size < other.object_size) ||
           (object_size == other.object_size && object_id < other.object_id);
  }

  size_t object_size;
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

bool CompareBySize(const TensorUsageWithIndex<size_t>& first,
                   const TensorUsageWithIndex<size_t>& second) {
  return first.usage_record->tensor_size > second.usage_record->tensor_size;
}

// Implements memory management with a naive algorithm.
//
// The problem of memory management is NP-complete. This implements a
// naive algorithm that assigns each tensor to a separate object in memory.
template <typename TensorSizeT>
Status NaiveAssignment(
    const std::vector<TensorUsageRecord<TensorSizeT>>& usage_records,
    ObjectsAssignment<TensorSizeT>* assignment) {
  assignment->object_sizes.resize(usage_records.size());
  assignment->object_ids.assign(usage_records.size(), kNotAssigned);
  for (size_t i = 0; i < usage_records.size(); i++) {
    auto& record = usage_records[i];
    assignment->object_ids[i] = i;
    assignment->object_sizes[i] = record.tensor_size;
  }
  return OkStatus();
}

template <typename TensorSizeT>
Status EqualityAssignment(
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

    TensorSizeT tensor_size = usage_records[i].tensor_size;
    auto pool_it = pool.find(tensor_size);
    if (pool_it == pool.end() || pool_it->second.empty()) {
      // No free shared object with size equal to tensor_size. Create a new one,
      // assign i-th tensor to it and add to the queue of objects in use.
      assignment->object_ids[i] = assignment->object_sizes.size();
      assignment->object_sizes.push_back(tensor_size);
      objects_in_use.push(
          {usage_records[i].last_task, assignment->object_ids[i]});
    } else {
      // Share object with id it->second has size equal to tensor_size. Reuse
      // this object: erase it from pool and add to the queue of objects in use.
      assignment->object_ids[i] = pool_it->second.back();
      pool_it->second.pop_back();
      objects_in_use.push(
          {usage_records[i].last_task, assignment->object_ids[i]});
    }
  }
  return OkStatus();
}

// Implements memory management with a greedy algorithm.
//
// The problem of memory management is NP-complete. This implements a
// greedy algorithm that approximates an optimal solution with following
// heuristic:
//
//   1. Iterates through all tensor usage records and for every object
//   reference
//      assigns shared object from the pool. When object reference is used
//      for the last time, corresponding shared object is returned back to
//      the pool.
//
//   2. Shared object pool grows when there are no free shared object
//      available.
//
//   3. Shared object size may increase when tensor requests larger size.
Status GreedyAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    ObjectsAssignment<size_t>* assignment) {
  size_t num_records = usage_records.size();
  assignment->object_sizes.clear();
  assignment->object_ids.assign(num_records, kNotAssigned);

  // Pool of free shared objects is ordered by object size, because we perform
  // lower_bound search in it.
  std::set<PoolRecord> pool;
  // Queue of shared objects in use, ordered by their last_task.
  std::priority_queue<QueueRecord> objects_in_use;
  for (size_t i = 0; i < num_records; i++) {
    // Pop from the queue and add to the pool all objects that are no longer
    // in use at the time of execution of the first_task of i-th intermediate
    // tensor.
    while (!objects_in_use.empty() &&
           objects_in_use.top().last_task < usage_records[i].first_task) {
      auto object_id = objects_in_use.top().object_id;
      pool.insert({assignment->object_sizes[object_id], object_id});
      objects_in_use.pop();
    }
    size_t tensor_size = usage_records[i].tensor_size;
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
      size_t size_diff = 0;
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
            "No shared object is found in non-empty pool in "
            "GreedyAssignment.");
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

// Assigns given tensors to offsets, using the following greedy algorithm:
// - We have tensor usage records of all intermideate tensors as an input. Each
// record consists of tensor size, first and last tasks, that use it. Let's call
// [first_task..last_task] a tensor usage interval;
// - Iterate through tensor usage records in non-increasing order of
// corresponding tensor sizes;
// - For each of these records consider already assigned tensors, which usage
// intervals intersect with usage interval of current tensor, and find the
// smallest gap in memory between them such, that current tensor fits into that
// gap;
// - If such a gap has been found, current tensor should be allocated into this
// gap. Otherwise we can allocate it after the rightmost tensor, which usage
// interval intersects with usage inteval of current tensor. So we assign
// corresponding offset to current tensor and the tensor becomes assigned.
Status GreedyBySizeAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    OffsetsAssignment* assignment) {
  const size_t num_tensors = usage_records.size();
  assignment->offsets.resize(num_tensors);
  assignment->total_size = 0;

  // Ordered records are to be sorted by size of corrseponding tensor.
  std::vector<TensorUsageWithIndex<size_t>> ordered_records;
  for (size_t i = 0; i < num_tensors; ++i) {
    ordered_records.emplace_back(&usage_records[i], i);
  }
  std::sort(ordered_records.begin(), ordered_records.end(), CompareBySize);

  // Vector of ids of already allocated tensors, ordered by offset.
  std::vector<size_t> ordered_allocs;

  for (const auto& rec_with_idx : ordered_records) {
    const TensorUsageRecord<size_t>* rec = rec_with_idx.usage_record;
    size_t best_diff = kNotAssigned;
    size_t best_offset = kNotAssigned;
    size_t prev_offset = 0;
    for (const auto& allocated_id : ordered_allocs) {
      if (usage_records[allocated_id].last_task < rec->first_task ||
          usage_records[allocated_id].first_task > rec->last_task) {
        // Tensor allocated_id has usage interval, that doesn't intersect with
        // current tensor's usage interval, so we skip it.
        continue;
      }
      size_t cur_offset = assignment->offsets[allocated_id];
      if (cur_offset >= prev_offset) {
        size_t diff = cur_offset - prev_offset;
        // Check, if current_tensor fits into the gap, located directly to the
        // left of tensor allocated_id offset, and that this gap is the smallest
        // of previously considered suitable gaps.
        if (diff >= rec->tensor_size && diff < best_diff) {
          best_diff = diff;
          best_offset = prev_offset;
        }
      }
      prev_offset = std::max(
          prev_offset, cur_offset + usage_records[allocated_id].tensor_size);
    }
    if (assignment->total_size < prev_offset) {
      return InternalError("Total size is wrong.");
    }

    // If no suitable gap found, we should allocate current tensor after the
    // rightmost tensor, which usage interval intersects with the current one.
    if (best_offset == kNotAssigned) {
      best_offset = prev_offset;
    }

    // Assign best_offset to the current tensor and find the correct place to
    // insert information about it into ordered_allocs to save the order.
    auto it = ordered_allocs.begin();
    while (it != ordered_allocs.end() &&
           assignment->offsets[*it] <= best_offset) {
      ++it;
    }
    ordered_allocs.insert(it, rec_with_idx.idx);
    assignment->offsets[rec_with_idx.idx] = best_offset;
    assignment->total_size =
        std::max(assignment->total_size, best_offset + rec->tensor_size);
  }
  return OkStatus();
}

// This class build flow graph and solves Minimum-cost flow problem in it.
class MinCostFlowSolver {
 public:
  // Build auxiliary flow graph, based on information about intermediate
  // tensors.
  void Build(const std::vector<TensorUsageRecord<size_t>>& usage_records) {
    usage_records_ = &usage_records;
    num_tensors_ = usage_records.size();
    source_ = 2 * num_tensors_;
    sink_ = source_ + 1;
    edges_from_.resize(sink_ + 1);
    std::vector<size_t> old_record_ids;
    std::priority_queue<QueueRecord> objects_in_use;
    for (size_t i = 0; i < usage_records.size(); i++) {
      // Pop from the queue all objects that are no longer in use at the time
      // of execution of the first_task of i-th intermediate tensor.
      while (!objects_in_use.empty() &&
             objects_in_use.top().last_task < usage_records[i].first_task) {
        old_record_ids.push_back(objects_in_use.top().object_id);
        objects_in_use.pop();
      }
      objects_in_use.push({usage_records[i].last_task, i});
      AddEdge(source_, i, 1, 0);
      AddEdge(RightPartTwin(i), sink_, 1, 0);

      // Edge from source_ to i-th vertex in the right part of flow graph
      // are added for the case of allocation of new shared object for i-th
      // tensor. Cost of these edges is equal to the size of i-th tensor.
      AddEdge(source_, RightPartTwin(i), 1, usage_records[i].tensor_size);

      // Edges from vertices of the left part of flow graph, corresponding to
      // old_record_ids, to i-th vertex in the right part of flow graph are
      // added for the case of reusing previously created shared objects for
      // i-th tensor. Cost of these edges is an approximation of the size of
      // new allocated memory.
      for (auto record_id : old_record_ids) {
        int cost = 0;
        if (usage_records[i].tensor_size >
            usage_records[record_id].tensor_size) {
          cost = usage_records[i].tensor_size -
                 usage_records[record_id].tensor_size;
        }
        AddEdge(record_id, RightPartTwin(i), 1, cost);
      }
    }
  }

  // Solve Minimum-cost flow problem with Shortest Path Faster Algorithm.
  void Solve() {
    const int kInf = std::numeric_limits<int>::max();
    std::vector<size_t> prev_edge(sink_ + 1);
    while (true) {
      std::queue<size_t> cur_queue, next_queue;
      std::vector<size_t> last_it_in_queue(sink_ + 1);
      std::vector<size_t> dist(sink_ + 1, kInf);
      size_t it = 1;
      cur_queue.push(source_);
      last_it_in_queue[source_] = it;
      dist[source_] = 0;
      // Find shortest path from source_ to sink_, using only edges with
      // positive capacity.
      while (!cur_queue.empty()) {
        ++it;
        while (!cur_queue.empty()) {
          auto v = cur_queue.front();
          cur_queue.pop();
          for (const auto& edge_id : edges_from_[v]) {
            const Edge& edge = edges_[edge_id];
            if (edge.cap > 0) {
              auto u = edge.dst;
              int new_dist = dist[v] + edge.cost;
              if (new_dist < dist[u]) {
                dist[u] = new_dist;
                prev_edge[u] = edge_id;
                if (last_it_in_queue[u] != it) {
                  next_queue.push(u);
                  last_it_in_queue[u] = it;
                }
              }
            }
          }
        }
        std::swap(cur_queue, next_queue);
      }
      // If path is not found, final result is ready.
      if (dist[sink_] == kInf) break;

      // If path is found, we need to decrease the capacity of its edges, and
      // increase the capacity of its reversed edges.
      for (size_t v = sink_; v != source_;) {
        --edges_[prev_edge[v]].cap;
        Edge& rev_edge = edges_[prev_edge[v] ^ 1];
        ++rev_edge.cap;
        v = rev_edge.dst;
      }
    }
  }

  void CalculateAssignment(ObjectsAssignment<size_t>* assignment) {
    assignment->object_sizes.clear();
    assignment->object_ids.assign(num_tensors_, kNotAssigned);
    is_tensor_assigned_.resize(num_tensors_);
    for (const auto& edge_id : edges_from_[source_]) {
      const Edge& edge = edges_[edge_id];
      if (edge.cap == 0 && IsRightPartVertex(edge.dst)) {
        assignment->object_sizes.push_back(
            AssignTensorsToNewSharedObject(LeftPartTwin(edge.dst), assignment));
      }
    }
  }

 private:
  struct Edge {
    Edge(size_t dst, int cap, int cost) : dst(dst), cap(cap), cost(cost) {}

    size_t dst;
    int cap;
    int cost;
  };

  // Add edge from vertex src to vertex dst with given capacity and cost and
  // its reversed edge to the flow graph. If some edge has index idx, its
  // reversed edge has index idx^1.
  void AddEdge(size_t src, size_t dst, int cap, int cost) {
    edges_from_[src].push_back(edges_.size());
    edges_.emplace_back(dst, cap, cost);
    edges_from_[dst].push_back(edges_.size());
    edges_.push_back({src, 0, -cost});
  }

  // Check, if vertex_id belongs to right part of the flow graph.
  bool IsRightPartVertex(size_t vertex_id) const {
    return vertex_id >= num_tensors_ && vertex_id < 2 * num_tensors_;
  }

  // Return vertex from another part of the graph, that corresponds to the
  // same intermediate tensor.
  size_t LeftPartTwin(size_t vertex_id) const {
    return vertex_id - num_tensors_;
  }
  size_t RightPartTwin(size_t vertex_id) const {
    return vertex_id + num_tensors_;
  }

  // This function uses recursive implementation of depth-first search and
  // returns maximum size from tensor tensor_id and all tensors, that will be
  // allocated at the same place with it after all operations that use
  // tensor_id are executed. Next tensor to be allocated at the same place
  // with tensor_id is a left part twin of such vertex v, that the edge
  // tensor_id->v is saturated (has zero residual capacity).
  size_t AssignTensorsToNewSharedObject(size_t tensor_id,
                                        ObjectsAssignment<size_t>* assignment) {
    size_t cost = (*usage_records_)[tensor_id].tensor_size;
    is_tensor_assigned_[tensor_id] = true;
    assignment->object_ids[tensor_id] = assignment->object_sizes.size();
    for (const auto& edge_id : edges_from_[tensor_id]) {
      const Edge& edge = edges_[edge_id];
      size_t v = edge.dst;
      size_t left_twin = LeftPartTwin(v);
      if (edge.cap == 0 && IsRightPartVertex(v) &&
          !is_tensor_assigned_[left_twin]) {
        cost = std::max(cost,
                        AssignTensorsToNewSharedObject(left_twin, assignment));
      }
    }
    return cost;
  }

  size_t source_;
  size_t sink_;
  size_t num_tensors_;
  const std::vector<TensorUsageRecord<size_t>>* usage_records_;
  std::vector<Edge> edges_;
  std::vector<std::vector<size_t>> edges_from_;
  std::vector<bool> is_tensor_assigned_;
};

// Implements memory management with a Minimum-cost flow matching algorithm.
//
// The problem of memory management is NP-complete. This function creates
// auxiliary flow graph, find minimum-cost flow in it and calculates the
// assignment of shared objects to tensors, using the result of the flow
// algorithm.
Status MinCostFlowAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    ObjectsAssignment<size_t>* assignment) {
  MinCostFlowSolver solver;
  solver.Build(usage_records);
  solver.Solve();
  solver.CalculateAssignment(assignment);
  return OkStatus();
}

}  // namespace

bool CompareBySize(const TensorUsageWithIndex<size_t>& first,
                   const TensorUsageWithIndex<size_t>& second) {
  return first.usage_record->tensor_size > second.usage_record->tensor_size;
}

OffsetsAssignment ObjectsToOffsets(
    const ObjectsAssignment<size_t>& obj_assignment) {
  size_t num_tensors = obj_assignment.object_ids.size();
  size_t num_objects = obj_assignment.object_sizes.size();
  OffsetsAssignment result = {/*offsets=*/std::vector<size_t>(num_tensors),
                              /*total_size=*/0};
  std::vector<size_t> ids_to_offset(num_objects);
  for (size_t i = 0; i < num_objects; ++i) {
    ids_to_offset[i] = result.total_size;
    result.total_size += obj_assignment.object_sizes[i];
  }
  for (size_t i = 0; i < num_tensors; ++i) {
    result.offsets[i] = ids_to_offset[obj_assignment.object_ids[i]];
  }
  return result;
}

Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    const MemoryStrategy& strategy, ObjectsAssignment<size_t>* assignment) {
  switch (strategy) {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment<size_t>(usage_records, assignment);
    case MemoryStrategy::EQUALITY:
      return EqualityAssignment<size_t>(usage_records, assignment);
    case MemoryStrategy::GREEDY:
      return GreedyAssignment(usage_records, assignment);
    case MemoryStrategy::MINCOSTFLOW:
      return MinCostFlowAssignment(usage_records, assignment);
    default:
      return InternalError(
          "MemoryStrategy is not supported with current tensor size type.");
  }
  return OkStatus();
}

Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<BHWC>>& usage_records,
    const MemoryStrategy& strategy, ObjectsAssignment<BHWC>* assignment) {
  switch (strategy) {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment<BHWC>(usage_records, assignment);
    case MemoryStrategy::EQUALITY:
      return EqualityAssignment<BHWC>(usage_records, assignment);
    default:
      return InternalError(
          "MemoryStrategy is not supported with current tensor size type.");
  }
  return OkStatus();
}

Status AssignOffsetsToTensors(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    const MemoryStrategy& strategy, OffsetsAssignment* assignment) {
  if (strategy == MemoryStrategy::GREEDY_BY_SIZE) {
    return GreedyBySizeAssignment(usage_records, assignment);
  }
  ObjectsAssignment<size_t> objects_assignment;
  RETURN_IF_ERROR(
      AssignObjectsToTensors(usage_records, strategy, &objects_assignment));
  *assignment = ObjectsToOffsets(objects_assignment);
  return OkStatus();
}

}  // namespace gpu
}  // namespace tflite

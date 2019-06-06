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

#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace {

struct PoolRecord {
  PoolRecord(uint32_t size, size_t obj_id)
      : object_size(size), object_id(obj_id) {}

  // Objects in pool are ordered by size.
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
// naive algorithm that assigns each tensor to a separate object in memory.
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

// This class build flow graph and solves Minimum-cost flow problem in it.
class MinCostFlowSolver {
 public:
  // Build auxiliary flow graph, based on information about intermediate
  // tensors.
  void Build(const std::vector<TensorUsageRecord>& usage_records) {
    usage_records_ = &usage_records;
    num_tensors_ = usage_records.size();
    source_ = 2 * num_tensors_;
    sink_ = source_ + 1;
    edges_from_.resize(sink_ + 1);
    std::vector<size_t> old_record_ids;
    std::priority_queue<QueueRecord> objects_in_use;
    for (size_t i = 0; i < usage_records.size(); i++) {
      // Pop from the queue all objects that are no longer in use at the time of
      // execution of the first_task of i-th intermediate tensor.
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
      // i-th tensor. Cost of these edges is an approximation of the size of new
      // allocated memory.
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

  void CalculateAssignment(ObjectsAssignment* assignment) {
    assignment->object_sizes.clear();
    assignment->object_ids.resize(num_tensors_);
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

  // Add edge from vertex src to vertex dst with given capacity and cost and its
  // reversed edge to the flow graph. If some edge has index idx, its reversed
  // edge has index idx^1.
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

  // Return vertex from another part of the graph, that corresponds to the same
  // intermediate tensor.
  size_t LeftPartTwin(size_t vertex_id) const {
    return vertex_id - num_tensors_;
  }
  size_t RightPartTwin(size_t vertex_id) const {
    return vertex_id + num_tensors_;
  }

  // This function uses recursive implementation of depth-first search and
  // returns maximum size from tensor tensor_id and all tensors, that will be
  // allocated at the same place with it after all operations that use tensor_id
  // are executed. Next tensor to be allocated at the same place with tensor_id
  // is a left part twin of such vertex v, that the edge tensor_id->v is
  // saturated (has zero residual capacity).
  uint32_t AssignTensorsToNewSharedObject(size_t tensor_id,
                                          ObjectsAssignment* assignment) {
    uint32_t cost = (*usage_records_)[tensor_id].tensor_size;
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
  const std::vector<TensorUsageRecord>* usage_records_;
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
    const std::vector<TensorUsageRecord>& usage_records,
    ObjectsAssignment* assignment) {
  MinCostFlowSolver solver;
  solver.Build(usage_records);
  solver.Solve();
  solver.CalculateAssignment(assignment);
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
    case MemoryStrategy::MINCOSTFLOW:
      return MinCostFlowAssignment(usage_records, assignment);
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace tflite

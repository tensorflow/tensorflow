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

#include "tensorflow/lite/delegates/gpu/common/memory_management/min_cost_flow_assignment.h"

#include <algorithm>
#include <queue>
#include <set>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/memory_management/internal.h"

namespace tflite {
namespace gpu {
namespace {

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

}  // namespace

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

}  // namespace gpu
}  // namespace tflite

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

#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"

#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>

namespace tflite {
namespace gpu {
namespace {

void BFS(size_t start, const UsageGraph& deps_graph,
         std::vector<size_t>* num_visits) {
  std::queue<size_t> queue;
  std::vector<char> is_visited(deps_graph.size(),
                               0);  // use char instead of bool, because
                                    // std::vector<bool> is based on bitset
  queue.push(start);
  is_visited[start] = true;
  while (!queue.empty()) {
    size_t from_vertex = queue.front();
    queue.pop();
    for (const auto& to_vertex : deps_graph[from_vertex]) {
      if (!is_visited[to_vertex]) {
        queue.push(to_vertex);
        is_visited[to_vertex] = true;
        (*num_visits)[to_vertex] += 1;
      }
    }
  }
}

}  // namespace

UsageGraph ReallocationGraph(const UsageGraph& deps_graph) {
  size_t num_vertices = deps_graph.size();
  UsageGraph reallocation_graph(num_vertices);
  for (size_t root = 0; root < num_vertices; ++root) {
    std::vector<size_t> num_visits(num_vertices, 0);
    const std::vector<size_t>& children = deps_graph[root];
    if (children.empty()) {
      // Check
      continue;
    }
    for (const auto& child : children) {
      BFS(child, deps_graph, &num_visits);
    }
    for (size_t vertex = 0; vertex < num_vertices; ++vertex) {
      if (num_visits[vertex] == children.size()) {
        reallocation_graph[root].push_back(vertex);
        reallocation_graph[vertex].push_back(root);
      }
    }
  }
  for (size_t vertex = 0; vertex < num_vertices; ++vertex) {
    std::sort(reallocation_graph[vertex].begin(),
              reallocation_graph[vertex].end());
  }
  return reallocation_graph;
}

}  // namespace gpu
}  // namespace tflite

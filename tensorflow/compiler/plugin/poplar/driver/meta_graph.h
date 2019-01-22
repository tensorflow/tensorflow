/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_META_GRAPH_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_META_GRAPH_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

#include <queue>
#include <stack>

namespace xla {
namespace poplarplugin {

template <typename T>
class MetaGraph {
  using Graph = absl::flat_hash_map<T, absl::flat_hash_set<T>>;

  MetaGraph(){};

  template <typename Predicate>
  absl::flat_hash_set<T> FindConsumers(T node, Predicate pred, bool inclusive,
                                       absl::flat_hash_set<T>& visited) const {
    absl::flat_hash_set<T> consumers;

    const auto itr = graph_.find(node);
    if (itr != graph_.end()) {
      for (const auto& neighbour : itr->second) {
        if (inclusive) {
          consumers.insert(neighbour);
        }
        const bool already_visited = visited.count(neighbour);
        if (pred(neighbour) && !already_visited) {
          consumers.insert(neighbour);
          visited.insert(neighbour);
          consumers.merge(FindConsumers(neighbour, pred, inclusive, visited));
        }
      }
    }

    return consumers;
  }

  Graph graph_;

 public:
  template <typename NodeIt>
  MetaGraph(std::vector<T> root_nodes, NodeIt node_iterator_getter) {
    // DF traversal to create the initial graph.
    std::stack<T> to_visit;
    absl::flat_hash_set<T> visited;
    for (T root_node : root_nodes) {
      to_visit.push(root_node);
    }

    while (!to_visit.empty()) {
      // Get the current node
      T current = to_visit.top();
      to_visit.pop();

      if (visited.count(current) != 0) {
        continue;
      }
      visited.insert(current);

      for (T operand : node_iterator_getter(current)) {
        graph_[operand].insert(current);
        to_visit.push(operand);
      }
    }
  };

  template <typename NodeIt>
  MetaGraph(T root_node, NodeIt node_iterator_getter)
      : MetaGraph(std::vector<T>({root_node}), node_iterator_getter) {}

  template <typename InputIt, typename NodeValueGetter>
  MetaGraph(InputIt input_it, NodeValueGetter node_value_getter) {
    for (T input : input_it) {
      graph_[input] = node_value_getter(input);
    }
  };

  MetaGraph Transpose() const {
    MetaGraph<T> result;

    for (auto& edge : graph_) {
      for (auto v2 : edge.second) {
        result[v2].insert(edge.first);
      }
    }

    return result;
  }

  absl::flat_hash_set<T> GetVertices() const {
    absl::flat_hash_set<T> result;

    for (auto pair : graph_) {
      result.insert(pair.first);
      result.merge(pair.second);
    }

    return result;
  }

  template <typename Predicate>
  absl::flat_hash_set<T> FindConsumers(T node, Predicate pred,
                                       bool inclusive = false) const {
    // FindConsumers is a depth first traversal - this is a wrapper for it where
    // we create a set of visited nodes to prevent getting stuck in cycles.
    absl::flat_hash_set<T> visited;
    return FindConsumers(node, pred, inclusive, visited);
  }

  template <typename Predicate>
  absl::flat_hash_set<T> FindVertices(Predicate pred) const {
    absl::flat_hash_set<T> result;

    for (const auto& v : GetVertices()) {
      if (pred(v)) {
        result.insert(v);
      }
    }

    return result;
  }

  std::vector<T> ShortestPath(T src, T dst) const {
    absl::flat_hash_map<T, int> dist;
    absl::flat_hash_map<T, T> prev;
    absl::flat_hash_set<T> visited;

    const auto comp = [&](T a, T b) { return dist[a] < dist[b]; };

    std::priority_queue<T, std::vector<T>, decltype(comp)> queue(comp);

    const auto vs = GetVertices();
    for (const auto& v : vs) {
      dist[v] = std::numeric_limits<int>::max();
    }

    dist[src] = 0;
    queue.push(src);

    while (!queue.empty() && dist[dst] == std::numeric_limits<int>::max()) {
      const auto top = queue.top();
      queue.pop();
      visited.insert(top);

      const auto itr = graph_.find(top);
      std::for_each(itr->second.begin(), itr->second.end(), [&](T v) {
        if (!visited.contains(v)) {
          dist[v] = dist[top] + 1;
          prev[v] = top;
          queue.push(v);
        }
      });
    }

    std::vector<T> path = {dst};
    while (path.back() != src) {
      path.push_back(prev[path.back()]);
    }
    std::reverse(path.begin(), path.end());

    return path;
  }

  template <typename Predicate>
  static bool IsPathOk(const std::vector<T>& path, Predicate pred) {
    for (unsigned i = 0; i < path.size(); i++) {
      T node = path[i];
      if (!pred(node, i, path.size())) {
        return false;
      }
    }
    return true;
  };

  absl::flat_hash_set<T>& operator[](T& key) { return graph_[key]; }

  const absl::flat_hash_set<T>& operator[](const T key) const {
    return graph_.at(key);
  }

  bool contains(T key) const { return graph_.find(key) != graph_.end(); }

  typename Graph::const_iterator begin() const { return graph_.begin(); }

  typename Graph::const_iterator end() const { return graph_.end(); }

  typename Graph::const_iterator find(T key) const { return graph_.find(key); }
};

}  // namespace poplarplugin
}  // namespace xla
#endif

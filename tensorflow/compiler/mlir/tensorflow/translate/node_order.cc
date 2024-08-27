/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/translate/node_order.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

void TopologicalOrdering(
    const Graph& g, const std::function<void(Node*)>& emit,
    const std::function<std::string(Node*)>& get_grouping_key) {
  std::unordered_map<std::string, int> group_key_string_to_integer;
  absl::flat_hash_map<Node*, int> node_to_group;
  absl::flat_hash_map<Node*, int> remaining_incoming_nodes;
  absl::flat_hash_map<Node*, int> node_to_position;
  using Ready = std::vector<Node*>;
  std::vector<Ready> group_members_that_are_ready;
  std::set<int> groups_that_are_ready;

  // Visit all nodes once, for initialization. It doesn't matter whether we use
  // BFS or DFS.
  int i = 0;
  DFS(
      g, [](Node*) {},
      [&](Node* n) {
        // Find which group this node belongs to.
        std::string group_key_string = get_grouping_key(n);
        auto entry = group_key_string_to_integer.try_emplace(
            group_key_string, group_key_string_to_integer.size());
        int group_key = entry.first->second;
        node_to_position[n] = i++;
        node_to_group[n] = group_key;
        if (entry.second) {
          group_members_that_are_ready.push_back({});
        }

        // Count the incoming nodes and store. Also remember nodes ("sources")
        // that don't have any inputs.
        auto in_nodes = n->in_nodes();
        int num_incoming = std::distance(in_nodes.begin(), in_nodes.end());
        remaining_incoming_nodes[n] = num_incoming;
        if (num_incoming == 0) {
          // NO_CDC: This array is max(group_key) + 1.
          group_members_that_are_ready[group_key].push_back(n);
          groups_that_are_ready.emplace(group_key);
        }
      },
      [](const Node* n1, const Node* n2) { return n1->name() < n2->name(); });

  assert(group_key_string_to_integer.size() ==
         group_members_that_are_ready.size());

  int num_nodes = remaining_incoming_nodes.size();

  // We emit one node per step, thus we just run this as often as we have nodes.
  int current_group = 0;
  for (int i = 0; i < num_nodes; i++) {
    if (groups_that_are_ready.find(current_group) ==
        groups_that_are_ready.end()) {
      current_group = *groups_that_are_ready.begin();
    }

    // NO_CDC: This array is max(group_key) + 1.
    int size = group_members_that_are_ready[current_group].size();
    assert(size);
    // NO_CDC: This array is max(group_key) + 1.
    Node* node = group_members_that_are_ready[current_group][--size];
    // NO_CDC: This array is max(group_key) + 1.
    group_members_that_are_ready[current_group].pop_back();
    if (size == 0) {
      groups_that_are_ready.erase(current_group);
    }

    // Emit the operation and make its results available.
    emit(node);

    auto out_nodes = node->out_nodes();
    std::vector<Node*> nodes_sorted(out_nodes.begin(), out_nodes.end());
    std::sort(nodes_sorted.begin(), nodes_sorted.end(), [&](Node* a, Node* b) {
      return node_to_position[a] < node_to_position[b];
    });

    for (Node* out : nodes_sorted) {
      remaining_incoming_nodes[out]--;
      if (remaining_incoming_nodes[out] == 0) {
        int group_key = node_to_group[out];
        // NO_CDC: This array is max(group_key) + 1.
        if (group_members_that_are_ready[group_key].empty()) {
          groups_that_are_ready.emplace(group_key);
        }
        // NO_CDC: This array is max(group_key) + 1.
        group_members_that_are_ready[group_key].push_back(out);
      }
    }
  }
}

}  // namespace tensorflow

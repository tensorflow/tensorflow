/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/graph/collective_order.h"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {
namespace {

// Find all CollectiveReduce nodes and the existing data dependencies between
// them.
absl::Status DiscoverDataDependencies(
    const Graph* graph, std::vector<Node*>* collective_nodes,
    std::vector<int32_t>* instance_keys,
    absl::flat_hash_map<Node*, absl::flat_hash_set<int32_t>>*
        data_dependencies) {
  absl::Status s;
  // Algorithm: do Reverse DFS starting at sink.  `node_leave` is called when
  // all parents of `node` have been visited.  At that point,
  // `data_dependencies[node]` is a list containing `instance_key` of every
  // `CollectiveReduce` on which `node` has a data dependency.
  // For this node's children, add all these instance keys.  Also, if this node
  // is collective, add as a dependency for the children.
  auto node_leave = [collective_nodes, instance_keys, data_dependencies,
                     &s](Node* node) {
    int32_t instance_key;
    bool enter_node =
        node->IsCollective() && node->type_string() == "CollectiveReduce";
    if (enter_node) {
      absl::Status get_attr_status =
          GetNodeAttr(node->attrs(), "instance_key", &instance_key);
      s.Update(get_attr_status);
      collective_nodes->push_back(node);
      instance_keys->push_back(instance_key);
      VLOG(2) << "collective node " << node->DebugString();
    }
    // Avoid reference invalidation of `node_deps`.
    data_dependencies->reserve(data_dependencies->size() + 1 +
                               node->out_edges().size());
    const auto& node_deps = (*data_dependencies)[node];
    for (const Edge* out_edge : node->out_edges()) {
      auto& child_deps = (*data_dependencies)[out_edge->dst()];
      child_deps.insert(node_deps.begin(), node_deps.end());
      if (enter_node && s.ok()) {
        child_deps.insert(instance_key);
      }
    }
  };
  ReverseDFS(*graph, nullptr, node_leave);
  return s;
}

// Given a list of `collective_nodes` and `data_dependencies` between the
// collective nodes, create control dependencies between concurrent collectives
// and store in `dependency_edges`.
// If there exists an edge a -> b then `dependency_edges[a]` contains `b`
absl::Status CreateControlDependencies(
    const std::vector<Node*>& collective_nodes,
    const std::vector<int32_t>& instance_keys,
    absl::flat_hash_map<Node*, absl::flat_hash_set<int32_t>>* data_dependencies,
    absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>* dependency_edges) {
  const int num_collectives = collective_nodes.size();
  // Reachability over `collective_nodes`, indexed by position, as a dense bit
  // matrix: bit (a, b) is set iff there is a path a -> ... -> b. One bit per
  // ordered pair replaces a `flat_hash_set<Node*>` per node.
  const int words_per_row = (num_collectives + 63) / 64;
  std::vector<uint64_t> reaches(
      static_cast<size_t>(num_collectives) * words_per_row, 0);
  const auto set_reaches = [&](int a, int b) {
    reaches[static_cast<size_t>(a) * words_per_row + b / 64] |=
        uint64_t{1} << (b % 64);
  };
  const auto test_reaches = [&](int a, int b) {
    return (reaches[static_cast<size_t>(a) * words_per_row + b / 64] >>
            (b % 64)) &
           1;
  };
  absl::flat_hash_map<Node*, int> collective_index;
  collective_index.reserve(num_collectives);
  for (int i = 0; i < num_collectives; ++i) {
    collective_index[collective_nodes[i]] = i;
  }
  // Successors by index, mirroring `dependency_edges`.
  std::vector<std::vector<int>> successors(num_collectives);
  for (int i = 0; i < collective_nodes.size() - 1; i++) {
    if (!collective_nodes[i]->IsCollective() ||
        collective_nodes[i]->type_string() != "CollectiveReduce") {
      return absl::InternalError(
          absl::StrCat("Unexpected node ", collective_nodes[i]->DebugString()));
    }
    const auto& deps_i = (*data_dependencies)[collective_nodes[i]];
    for (int j = i + 1; j < collective_nodes.size(); j++) {
      if (collective_nodes[i]->requested_device() !=
          collective_nodes[j]->requested_device()) {
        continue;
      }
      if (instance_keys[i] == instance_keys[j]) {
        return absl::InternalError(
            absl::StrCat("Unexpected same instance_key ", instance_keys[i],
                         " on 2 nodes with the same device ",
                         collective_nodes[i]->requested_device()));
      }
      const auto& deps_j = (*data_dependencies)[collective_nodes[j]];
      if (deps_i.find(instance_keys[j]) == deps_i.end() &&
          deps_j.find(instance_keys[i]) == deps_j.end()) {
        int src_idx = instance_keys[i] > instance_keys[j] ? i : j;
        int dst_idx = instance_keys[i] > instance_keys[j] ? j : i;
        Node* src_node = collective_nodes[src_idx];
        Node* dst_node = collective_nodes[dst_idx];
        VLOG(1) << "Adding control dependency from node " << src_node->name()
                << " instance " << instance_keys[src_idx] << " to node "
                << dst_node->name() << " instance " << instance_keys[dst_idx];
        (*dependency_edges)[src_node].insert(dst_node);
        successors[src_idx].push_back(dst_idx);
      }
    }
  }

  // Close `reaches` transitively. Every edge runs from a higher instance key to
  // a lower one, so instance keys strictly decrease along any path and
  // ascending instance key is a reverse topological order: when a node is
  // processed, all of its successors are already closed, so one pass suffices.
  std::vector<int> by_ascending_key(num_collectives);
  std::iota(by_ascending_key.begin(), by_ascending_key.end(), 0);
  absl::c_stable_sort(by_ascending_key, [&](int a, int b) {
    return instance_keys[a] < instance_keys[b];
  });
  for (int idx : by_ascending_key) {
    for (int succ : successors[idx]) {
      set_reaches(idx, succ);
      const size_t src_row = static_cast<size_t>(idx) * words_per_row;
      const size_t succ_row = static_cast<size_t>(succ) * words_per_row;
      for (int w = 0; w < words_per_row; ++w) {
        reaches[src_row + w] |= reaches[succ_row + w];
      }
    }
  }

  // Prune dependency edges so that if there are edges a -> b, b -> c, and a ->
  // c, then remove a -> c.  This dependency would be handled naturally during
  // op scheduling.
  //
  // This is the transitive reduction of `dependency_edges`, which is unique for
  // a DAG: edge u -> v survives iff no other successor of u reaches v. Deciding
  // each edge independently against the closed `reaches` makes the surviving
  // set a function of the graph alone, so it no longer depends on the iteration
  // order of `neighbor_set`, which is keyed by `Node*`.
  std::vector<Node*> redundant;
  for (int i = 0; i < num_collectives; ++i) {
    auto edges_it = dependency_edges->find(collective_nodes[i]);
    if (edges_it == dependency_edges->end()) continue;
    absl::flat_hash_set<Node*>& neighbor_set = edges_it->second;
    redundant.clear();
    for (Node* v : neighbor_set) {
      const int v_idx = collective_index[v];
      for (Node* w : neighbor_set) {
        if (w == v) continue;
        if (test_reaches(collective_index[w], v_idx)) {
          redundant.push_back(v);
          break;
        }
      }
    }
    for (Node* v : redundant) neighbor_set.erase(v);
  }

  return absl::OkStatus();
}

// Insert control dependencies defined by `dependency_edges` in `graph`.  If
// `order_type` is `kEdges`, insert explicit control edges, else if `order_type`
// is `kAttrs`, encode dependencies as an attribute on collective node.
absl::Status InsertControlDependencies(
    Graph* graph, GraphCollectiveOrder order_type,
    const absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>&
        dependency_edges) {
  if (order_type == GraphCollectiveOrder::kEdges) {
    for (const auto& pair : dependency_edges) {
      Node* src_node = pair.first;
      for (Node* dst_node : pair.second) {
        graph->AddControlEdge(src_node, dst_node);
      }
    }
  } else if (order_type == GraphCollectiveOrder::kAttrs) {
    // `wait_for` is the inverse of `dependency_edges`, i.e. `wait_for[node]`
    // contains the list of instance keys for which `node` must wait.
    absl::flat_hash_map<Node*, absl::flat_hash_set<int32_t>> wait_for;
    for (const auto& pair : dependency_edges) {
      int32_t src_instance;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(pair.first->attrs(), "instance_key", &src_instance));
      for (Node* dst_node : pair.second) {
        wait_for[dst_node].insert(src_instance);
      }
    }
    for (const auto& pair : wait_for) {
      std::vector<int32_t> wait_for_list(pair.second.begin(),
                                         pair.second.end());
      pair.first->ClearAttr("wait_for");
      pair.first->AddAttr("wait_for", wait_for_list);
    }
  } else {
    return absl::InternalError(absl::StrCat(
        "Unexpected GraphCollectiveOrder type ", static_cast<int>(order_type)));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status OrderCollectives(Graph* graph, GraphCollectiveOrder order_type) {
  // `instance_keys[i]` corresponds to `collective_nodes[i]`
  std::vector<Node*> collective_nodes;
  std::vector<int32_t> instance_keys;
  // node -> set of collectives on which node depends.
  absl::flat_hash_map<Node*, absl::flat_hash_set<int32_t>> data_dependencies;
  TF_RETURN_IF_ERROR(DiscoverDataDependencies(
      graph, &collective_nodes, &instance_keys, &data_dependencies));

  if (collective_nodes.empty()) return absl::OkStatus();

  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> dependency_edges;
  // For all pairs of collective nodes n1 and n2 on the same device, if n1 does
  // not depend on n2 and n2 does not depend on n1, then they are potentially
  // concurrent.  Create an arbitrary, deterministic ordering between them.
  TF_RETURN_IF_ERROR(CreateControlDependencies(
      collective_nodes, instance_keys, &data_dependencies, &dependency_edges));

  return InsertControlDependencies(graph, order_type, dependency_edges);
}

}  // namespace tensorflow

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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {
namespace {

// Find all CollectiveReduce nodes and the existing data dependencies between
// them.
Status DiscoverDataDependencies(
    const Graph* graph, std::vector<Node*>* collective_nodes,
    std::vector<int32>* instance_keys,
    absl::flat_hash_map<Node*, absl::flat_hash_set<int32>>* data_dependencies) {
  Status s;
  // Algorithm: do Reverse DFS starting at sink.  `node_leave` is called when
  // all parents of `node` have been visited.  At that point,
  // `data_dependencies[node]` is a list containing `instance_key` of every
  // `CollectiveReduce` on which `node` has a data dependency.
  // For this node's children, add all these instance keys.  Also, if this node
  // is collective, add as a dependency for the children.
  auto node_leave = [collective_nodes, instance_keys, data_dependencies,
                     &s](Node* node) {
    int32 instance_key;
    bool enter_node =
        node->IsCollective() && node->type_string() == "CollectiveReduce";
    if (enter_node) {
      Status get_attr_status =
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
Status CreateControlDependencies(
    const std::vector<Node*>& collective_nodes,
    const std::vector<int32>& instance_keys,
    absl::flat_hash_map<Node*, absl::flat_hash_set<int32>>* data_dependencies,
    absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>* dependency_edges) {
  // If there exists some path a -> ... -> b then `all_paths[a]` contains `b`
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> all_paths;
  for (int i = 0; i < collective_nodes.size() - 1; i++) {
    if (!collective_nodes[i]->IsCollective() ||
        collective_nodes[i]->type_string() != "CollectiveReduce") {
      return errors::Internal("Unexpected node ",
                              collective_nodes[i]->DebugString());
    }
    const auto& deps_i = (*data_dependencies)[collective_nodes[i]];
    for (int j = i + 1; j < collective_nodes.size(); j++) {
      if (collective_nodes[i]->requested_device() !=
          collective_nodes[j]->requested_device()) {
        continue;
      }
      if (instance_keys[i] == instance_keys[j]) {
        return errors::Internal("Unexpected same instance_key ",
                                instance_keys[i],
                                " on 2 nodes with the same device ",
                                collective_nodes[i]->requested_device());
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
        auto& src_paths = all_paths[src_node];
        src_paths.insert(dst_node);
        for (Node* downstream_node : all_paths[dst_node]) {
          src_paths.insert(downstream_node);
        }
      }
    }
  }

  // Prune dependency edges so that if there are edges a -> b, b -> c, and a ->
  // c, then remove a -> c.  This dependency would be handled naturally during
  // op scheduling.
  for (int i = 0; i < collective_nodes.size(); ++i) {
    Node* node = collective_nodes[i];
    auto& neighbor_set = (*dependency_edges)[node];
    std::vector<Node*> neighbor_list(neighbor_set.begin(), neighbor_set.end());
    // For all n1, n2 in `neighbor_list` if there is a path from n1 -> n2 then
    // eliminate n2 from `neighbor_set` and `neighbor_list`.  We remove from
    // `neighbor_list` by replacing with a `nullptr`, hence the `nullptr` checks
    // below.
    for (int j = 0; j < neighbor_list.size(); ++j) {
      Node* n1 = neighbor_list[j];
      if (n1 == nullptr) continue;
      auto& n1_paths = all_paths[n1];
      for (int k = 0; k < neighbor_list.size(); ++k) {
        Node* n2 = neighbor_list[k];
        if (j == k || n2 == nullptr) continue;
        if (n1_paths.find(n2) != n1_paths.end()) {
          neighbor_set.erase(n2);
          neighbor_list[k] = nullptr;
        }
      }
    }
  }

  return Status::OK();
}

// Insert control dependencies defined by `dependency_edges` in `graph`.  If
// `order_type` is `kEdges`, insert explicit control edges, else if `order_type`
// is `kAttrs`, encode dependencies as an attribute on collective node.
Status InsertControlDependencies(
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
    absl::flat_hash_map<Node*, absl::flat_hash_set<int32>> wait_for;
    for (const auto& pair : dependency_edges) {
      int32 src_instance;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(pair.first->attrs(), "instance_key", &src_instance));
      for (Node* dst_node : pair.second) {
        wait_for[dst_node].insert(src_instance);
      }
    }
    for (const auto& pair : wait_for) {
      std::vector<int32> wait_for_list(pair.second.begin(), pair.second.end());
      pair.first->ClearAttr("wait_for");
      pair.first->AddAttr("wait_for", wait_for_list);
    }
  } else {
    return errors::Internal("Unexpected GraphCollectiveOrder type ",
                            static_cast<int>(order_type));
  }
  return Status::OK();
}

}  // namespace

Status OrderCollectives(Graph* graph, GraphCollectiveOrder order_type) {
  // `instance_keys[i]` corresponds to `collective_nodes[i]`
  std::vector<Node*> collective_nodes;
  std::vector<int32> instance_keys;
  // node -> set of collectives on which node depends.
  absl::flat_hash_map<Node*, absl::flat_hash_set<int32>> data_dependencies;
  TF_RETURN_IF_ERROR(DiscoverDataDependencies(
      graph, &collective_nodes, &instance_keys, &data_dependencies));

  if (collective_nodes.empty()) return Status::OK();

  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> dependency_edges;
  // For all pairs of collective nodes n1 and n2 on the same device, if n1 does
  // not depend on n2 and n2 does not depend on n1, then they are potentially
  // concurrent.  Create an arbitrary, deterministic ordering between them.
  TF_RETURN_IF_ERROR(CreateControlDependencies(
      collective_nodes, instance_keys, &data_dependencies, &dependency_edges));

  return InsertControlDependencies(graph, order_type, dependency_edges);
}

}  // namespace tensorflow

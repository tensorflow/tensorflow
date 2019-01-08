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

#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {

Status OrderCollectives(Graph* graph) {
  // `instance_keys[i]` corresponds to `collective_nodes[i]`
  std::vector<Node*> collective_nodes;
  std::vector<int32> instance_keys;
  // node -> set of collectives on which node depends.
  std::unordered_map<Node*, std::unordered_set<int32>> node_dependencies;
  Status s;

  // Algorithm: do Reverse DFS starting at sink.  `node_leave` is called when
  // all parents of `node` have been visited.  At that point, the collectives
  // on which this node depends on are up to date.  For this node's children,
  // add all these collectives.  Also, if this node is collective, add as a
  // dependency for the children.
  auto node_leave = [&collective_nodes, &instance_keys, &node_dependencies,
                     &s](Node* node) {
    int32 instance_key;
    if (node->IsCollective()) {
      Status get_attr_status =
          GetNodeAttr(node->attrs(), "instance_key", &instance_key);
      s.Update(get_attr_status);
      collective_nodes.push_back(node);
      instance_keys.push_back(instance_key);
      VLOG(2) << "collective node " << node->DebugString();
    }
    const auto& node_deps = node_dependencies[node];
    for (const Edge* out_edge : node->out_edges()) {
      auto& child_deps = node_dependencies[out_edge->dst()];
      child_deps.insert(node_deps.begin(), node_deps.end());
      if (node->IsCollective() && s.ok()) {
        child_deps.insert(instance_key);
      }
    }
  };
  ReverseDFS(*graph, nullptr, node_leave);
  if (!s.ok()) return s;

  // For all pairs of collective nodes n1 and n2 on the same device, if n1 does
  // not depend on n2 and n2 does not depend on n1, then they are potentially
  // concurrent.  Add an arbitrary, deterministic control edge between them.
  for (int i = 0; i < collective_nodes.size() - 1; i++) {
    if (!collective_nodes[i]->IsCollective()) {
      return errors::Internal("Unexpected node ",
                              collective_nodes[i]->DebugString());
    }
    const auto& deps_i = node_dependencies[collective_nodes[i]];
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
      const auto& deps_j = node_dependencies[collective_nodes[j]];
      if (deps_i.find(instance_keys[j]) == deps_i.end() &&
          deps_j.find(instance_keys[i]) == deps_j.end()) {
        int src_idx = instance_keys[i] < instance_keys[j] ? i : j;
        int dst_idx = instance_keys[i] < instance_keys[j] ? j : i;
        Node* src_node = collective_nodes[src_idx];
        Node* dst_node = collective_nodes[dst_idx];
        VLOG(1) << "Adding control edge from node " << src_node->name()
                << " instance " << instance_keys[src_idx] << " to node "
                << dst_node->name() << " instance " << instance_keys[dst_idx];
        graph->AddControlEdge(src_node, dst_node);
      }
    }
  }

  return Status::OK();
}

}  // namespace tensorflow

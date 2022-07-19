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

#include "tensorflow/core/grappler/utils/transitive_fanin.h"

#include <queue>
#include <vector>

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace grappler {

Status ComputeTransitiveFanin(
    const GraphDef& graph, const std::vector<string>& terminal_nodes,
    std::unordered_map<string, const NodeDef*>* name_to_fanin_node,
    std::vector<const NodeDef*>* fanin_nodes) {
  std::unordered_map<string, const NodeDef*> name_to_node;
  std::unordered_map<string, const NodeDef*> name_to_send;
  for (const auto& node : graph.node()) {
    name_to_node[node.name()] = &node;
    if (node.op() == "_Send") {
      const auto& attr = node.attr();
      name_to_send[attr.at("tensor_name").s()] = &node;
    }
  }

  std::vector<const NodeDef*> queue;
  for (const string& root : terminal_nodes) {
    const NodeDef* node = name_to_node[NodeName(root)];
    if (!node) {
      return errors::InvalidArgument("Graph does not contain terminal node ",
                                     root, ".");
    }
    queue.push_back(node);
  }

  std::unordered_set<const NodeDef*> visited;

  while (!queue.empty()) {
    const NodeDef* node = queue.back();
    queue.pop_back();
    if (!visited.insert(node).second) {
      // The node has already been visited.
      continue;
    }
    fanin_nodes->push_back(node);
    if (name_to_fanin_node) {
      name_to_fanin_node->insert(
          std::pair<string, const NodeDef*>(node->name(), node));
    }
    for (const string& input : node->input()) {
      const NodeDef* in = name_to_node[NodeName(input)];
      if (!in) {
        return errors::InvalidArgument("Graph does not contain input ",
                                       NodeName(input), " of node ",
                                       node->name(), ".");
      }
      queue.push_back(in);
    }
    if (node->op() == "_Recv") {
      const auto& attr = node->attr();
      const NodeDef* send = name_to_send[attr.at("tensor_name").s()];
      if (send) {
        queue.push_back(send);
      }
      // Subgraph after partitioning may have either _Send or _Recv, not both.
      // So, we do not set ill_formed for missing _Send.
    }
  }
  return OkStatus();
}

Status ComputeTransitiveFanin(const GraphDef& graph,
                              const std::vector<string>& terminal_nodes,
                              std::vector<const NodeDef*>* fanin_nodes) {
  return ComputeTransitiveFanin(graph, terminal_nodes, nullptr, fanin_nodes);
}

Status SetTransitiveFaninGraph(const GraphDef& input_graph,
                               GraphDef* output_graph,
                               const std::vector<string>& terminal_nodes) {
  // Determines transitive fanin nodes from terminal nodes and add them to the
  // output graph.
  std::vector<const NodeDef*> keep;
  TF_RETURN_IF_ERROR(
      ComputeTransitiveFanin(input_graph, terminal_nodes, &keep));
  // Try to keep the nodes ordered somewhat topologically since this helps
  // further optimizations perform better.
  output_graph->mutable_node()->Reserve(keep.size());
  for (int i = keep.size() - 1; i >= 0; --i) {
    *output_graph->add_node() = *keep[i];
  }

  return OkStatus();
}

}  // namespace grappler
}  // namespace tensorflow

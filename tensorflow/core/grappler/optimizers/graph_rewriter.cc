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

#include "tensorflow/core/grappler/optimizers/graph_rewriter.h"
#include <unordered_map>
#include <unordered_set>
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

GraphRewriter::GraphRewriter(const GrapplerItem& item) {
  for (auto& node : item.graph.node()) {
    nodes_[node.name()] = &node;
  }

  for (auto& node : item.graph.node()) {
    for (const auto& input : node.input()) {
      int position = 0;
      string input_node_name = ParseNodeName(input, &position);
      if (position < 0) {
        // This is a control edge
        auto itr = nodes_.find(input_node_name);
        CHECK(itr != nodes_.end());
        control_dependency_drivers_.insert(itr->second);
      }
    }
  }
}

void GraphRewriter::ForwardInputs(
    const NodeDef& original_node,
    const std::unordered_set<const NodeDef*>& nodes_to_delete,
    NodeDef* new_node) {
  for (const auto& input : original_node.input()) {
    string input_node_name = NodeName(input);
    auto itr = nodes_.find(input_node_name);
    CHECK(itr != nodes_.end());
    const NodeDef* input_node = itr->second;
    if (nodes_to_delete.find(input_node) != nodes_to_delete.end()) {
      ForwardInputs(*input_node, nodes_to_delete, new_node);
    } else {
      *new_node->add_input() = input;
    }
  }
}

bool GraphRewriter::DrivesControlDependency(const NodeDef& node) const {
  return control_dependency_drivers_.find(&node) !=
         control_dependency_drivers_.end();
}

}  // end namespace grappler
}  // end namespace tensorflow

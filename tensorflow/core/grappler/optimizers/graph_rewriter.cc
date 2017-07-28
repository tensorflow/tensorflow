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
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

GraphRewriter::GraphRewriter(const GrapplerItem& item) {
  OpRegistryInterface* op_registry = OpRegistry::Global();
  for (auto& node : item.graph.node()) {
    NodeInfo* info = new NodeInfo();
    info->def = &node;

    const OpRegistrationData* op_reg_data = nullptr;
    Status s = op_registry->LookUp(node.op(), &op_reg_data);
    // TODO(bsteiner): make this not a best-effort lookup and evaluation?
    if (s.ok()) {
      DataTypeVector inputs;
      s = InOutTypesForNode(node, op_reg_data->op_def, &inputs, &info->outputs);
      if (!s.ok()) {
        info->outputs.clear();
      }
    }

    nodes_[node.name()].reset(info);
  }

  std::unordered_set<string> function_names;
  for (const auto& function : item.graph.library().function()) {
    function_names.insert(function.signature().name());
  }

  for (auto& node : item.graph.node()) {
    RecordConnectivity(node, function_names);
  }
}

void GraphRewriter::ForwardInputs(
    const NodeDef& original_node,
    const std::unordered_set<const NodeDef*>& nodes_to_delete,
    NodeDef* new_node) {
  ForwardInputsInternal(original_node, nodes_to_delete, new_node);
  if (!new_node->name().empty()) {
    optimized_nodes_[new_node->name()] = new_node;
  }
}

bool GraphRewriter::DrivesControlDependency(const NodeDef& node) const {
  return control_dependency_drivers_.find(&node) !=
         control_dependency_drivers_.end();
}

bool GraphRewriter::IsDrivenByControlDependency(const NodeDef& node) const {
  for (const auto& input : node.input()) {
    CHECK(!input.empty());
    if (input[0] == '^') {
      return true;
    }
  }
  return false;
}

bool GraphRewriter::IsConnectedToFunction(const NodeDef& node) const {
  return function_neighbors_.find(&node) != function_neighbors_.end();
}

bool GraphRewriter::IsDrivenByAnotherDevice(const NodeDef& node) const {
  return cross_device_receivers_.find(&node) != cross_device_receivers_.end();
}

bool GraphRewriter::ReceivesRefValue(const NodeDef& node) const {
  return ref_receivers_.find(&node) != ref_receivers_.end();
}

void GraphRewriter::RecordConnectivity(
    const NodeDef& node, const std::unordered_set<string>& function_names) {
  const bool is_function =
      function_names.find(node.op()) != function_names.end();

  bool ref_receiver = false;
  for (const auto& input : node.input()) {
    int position = 0;
    string input_node_name = ParseNodeName(input, &position);
    auto itr = nodes_.find(input_node_name);
    if (itr == nodes_.end()) {
      continue;
    }
    const NodeInfo* fanin_info = itr->second.get();
    const NodeDef* fanin = fanin_info->def;
    if (position < 0) {
      // This is a control edge
      control_dependency_drivers_.insert(fanin);
    } else {
      // This is a regular edge
      if (function_names.find(fanin->op()) != function_names.end()) {
        function_neighbors_.insert(&node);
      }
      if (is_function) {
        function_neighbors_.insert(fanin);
      }

      if (position < fanin_info->outputs.size() &&
          IsRefType(fanin_info->outputs[position])) {
        ref_receiver = true;
      }
    }
    if (fanin->device() != node.device()) {
      cross_device_receivers_.insert(&node);
    }
  }

  if (ref_receiver) {
    ref_receivers_.insert(&node);
  }
}

void GraphRewriter::ForwardInputsInternal(
    const NodeDef& node,
    const std::unordered_set<const NodeDef*>& nodes_to_delete,
    NodeDef* new_node) {
  // To speed things up, use the optimized version of the node if
  // available.
  auto itr = optimized_nodes_.find(node.name());
  if (itr != optimized_nodes_.end()) {
    for (const string& input : itr->second->input()) {
      *new_node->add_input() = input;
    }
    return;
  }
  for (const auto& input : node.input()) {
    string input_node_name = NodeName(input);
    auto itr = nodes_.find(input_node_name);
    if (itr == nodes_.end()) {
      // Invalid input, preserve it as is.
      *new_node->add_input() = input;
      continue;
    }
    const NodeDef* input_node = itr->second->def;
    if (nodes_to_delete.find(input_node) != nodes_to_delete.end()) {
      ForwardInputsInternal(*input_node, nodes_to_delete, new_node);
    } else {
      *new_node->add_input() = input;
    }
  }
}

}  // end namespace grappler
}  // end namespace tensorflow

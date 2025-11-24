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

#include "tensorflow/core/grappler/optimizers/graph_optimizer_stage.h"
#include "tensorflow/core/graph/tensor_id.h"

namespace tensorflow {
namespace grappler {

const NodeScopeAndName ParseNodeScopeAndName(const string& node_name) {
  auto pos = node_name.find_last_of('/');
  if (pos == string::npos) {
    return {"", node_name};
  } else {
    return {node_name.substr(0, pos), node_name.substr(pos + 1)};
  }
};

absl::Status GetInputNode(const GraphOptimizerContext& ctx, const string& input,
                          NodeDef** node) {
  string node_name = NodeName(input);
  NodeDef* node_by_name = ctx.node_map->GetNode(node_name);
  if (node_by_name == nullptr) {
    return errors::FailedPrecondition("Node ", node_name,
                                      " doesn't exists in a node map");
  }
  *node = node_by_name;
  return absl::OkStatus();
}

absl::Status GetTensorProperties(const GraphOptimizerContext& ctx,
                                 const string& tensor,
                                 const OpInfo::TensorProperties** properties) {
  if (ctx.graph_properties == nullptr) {
    return errors::InvalidArgument("Graph properties are unknown.");
  }

  // TODO(ezhulenev): Make it TensorId when graph properties will support
  // absl::string_view lookup.
  SafeTensorId tensor_id = ParseTensorName(tensor);

  if (tensor_id.index() < 0) {
    return errors::InvalidArgument(
        "Can't get tensor properties of control dependency ", tensor);
  }

  const auto& output_properties =
      ctx.graph_properties->GetOutputProperties(tensor_id.node());
  int num_outputs = output_properties.size();

  if (num_outputs == 0 || tensor_id.index() > num_outputs - 1) {
    return errors::InvalidArgument(
        "Node ", tensor_id.node(),
        " is missing output properties at position :", tensor_id.index(),
        " (num_outputs=", num_outputs, ")");
  }

  *properties = &output_properties[tensor_id.index()];
  return absl::OkStatus();
}

NodeDef* AddCopyNode(const GraphOptimizerContext& ctx, const string& name,
                     const NodeDef* node_to_copy) {
  CHECK(node_to_copy != nullptr);
  CHECK(!ctx.node_map->NodeExists(name))
      << "Node " << name << " already exists in a graph";
  NodeDef* new_node = ctx.optimized_graph->add_node();
  *new_node = *node_to_copy;
  new_node->set_name(name);
  ctx.node_map->AddNode(name, new_node);
  return new_node;
}

NodeDef* AddEmptyNode(const GraphOptimizerContext& ctx, const string& name) {
  std::string new_name = name;
  for (int count = 0; ctx.node_map->NodeExists(new_name); ++count) {
    LOG(WARNING) << name << " already exists in the graph.";
    new_name = absl::StrCat(name, "_", count);
  }
  NodeDef* new_node = ctx.optimized_graph->add_node();
  new_node->set_name(new_name);
  ctx.node_map->AddNode(new_name, new_node);
  return new_node;
}

const string MakeOptimizedNodeName(const NodeScopeAndName& node,
                                   const string& sub_scope,
                                   const string& prefix) {
  CHECK(!sub_scope.empty() || !prefix.empty())
      << "Either optimized node name prefix or sub-scope must be non-empty";
  string optimized_node_name;
  if (!node.scope.empty()) {
    absl::StrAppend(&optimized_node_name, node.scope, "/");
  }
  if (!sub_scope.empty()) {
    absl::StrAppend(&optimized_node_name, sub_scope, "/");
  }
  if (!prefix.empty()) {
    absl::StrAppend(&optimized_node_name, prefix, "_");
  }
  absl::StrAppend(&optimized_node_name, node.name);
  return optimized_node_name;
}

const string MakeOptimizedNodeName(const NodeScopeAndName& root,
                                   const std::vector<string> node_names,
                                   const string& sub_scope,
                                   const string& prefix) {
  string optimized_node_name = MakeOptimizedNodeName(root, sub_scope, prefix);
  for (const string& node_name : node_names) {
    auto name_and_scope = ParseNodeScopeAndName(node_name);
    absl::StrAppend(&optimized_node_name, "_", name_and_scope.name);
  }
  return optimized_node_name;
}

}  // end namespace grappler
}  // end namespace tensorflow

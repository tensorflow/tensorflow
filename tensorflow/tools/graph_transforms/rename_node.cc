/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

Status RenameNode(const GraphDef& input_graph_def,
                  const TransformFuncContext& context,
                  GraphDef* output_graph_def) {
  if (!context.params.count("old_node_name") ||
      (context.params.at("old_node_name").size() != 1) ||
      !context.params.count("new_node_name") ||
      (context.params.at("new_node_name").size() != 1)) {
    return errors::InvalidArgument(
        "rename_node expects exactly one 'old_node_name' and one "
        "'new_node_name' argument, e.g. "
        "rename_node(old_attribute_name=super/deep/output, "
        "new_attribute_name=output)");
  }

  const std::string old_node_name = context.params.at("old_node_name")[0];
  const std::string new_node_name = context.params.at("new_node_name")[0];

  output_graph_def->Clear();
  for (const NodeDef& input_node : input_graph_def.node()) {
    NodeDef* node = output_graph_def->mutable_node()->Add();
    *node = input_node;
    if (node->name() == new_node_name) {
      return Status(error::Code::INVALID_ARGUMENT,
                    "A node is alreading using " + new_node_name + "as name.");
    }
    if (node->name() == old_node_name) {
      node->set_name(new_node_name);
    }
    for (std::string& input_name : *node->mutable_input()) {
      std::string prefix;
      std::string input_node_name;
      std::string suffix;
      NodeNamePartsFromInput(input_name, &prefix, &input_node_name, &suffix);
      if (input_node_name == old_node_name) {
        std::string new_input_name = prefix + new_node_name + suffix;
        input_name = new_input_name;
      }
    }
  }

  return OkStatus();
}

REGISTER_GRAPH_TRANSFORM("rename_node", RenameNode);
}  // namespace graph_transforms
}  // namespace tensorflow

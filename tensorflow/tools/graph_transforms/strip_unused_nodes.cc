/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

namespace {

Status TypeForPlaceholder(const TransformFuncContext& context,
                          const string& node_name, DataType* result) {
  // If we don't find anything else, return float.
  *result = DT_FLOAT;

  // Check to see if we have been given a default for all placeholders.
  if (context.params.count("type")) {
    if (context.params.at("type").size() != 1) {
      return errors::InvalidArgument(
          "You must pass no more than one default 'type' to "
          "strip_unused_nodes");
    }
    const string& type_string = context.params.at("type")[0];
    if (!DataTypeFromString(type_string, result)) {
      return errors::InvalidArgument("Couldn't understand type argument '",
                                     type_string, "'");
    }
  }

  // See if there's a particular type specified for this placeholder.
  if (context.params.count("name") || context.params.count("type_for_name")) {
    if (!context.params.count("name") ||
        !context.params.count("type_for_name") ||
        (context.params.at("type_for_name").size() !=
         context.params.at("name").size())) {
      return errors::InvalidArgument(
          "You must pass a 'type_for_name' arg for every 'name', e.g. "
          "strip_unused_nodes(name=foo, type_for_name=float, name=bar, "
          "type_for_name=quint8");
    }
    const int name_count = context.params.at("name").size();
    for (int i = 0; i < name_count; ++i) {
      if (context.params.at("name")[i] == node_name) {
        const string& type_string = context.params.at("type_for_name")[i];
        if (!DataTypeFromString(type_string, result)) {
          return errors::InvalidArgument("Couldn't understand type argument '",
                                         type_string, "'");
        }
      }
    }
  }

  return OkStatus();
}

Status ShapeForPlaceholder(const TransformFuncContext& context,
                           const string& node_name, TensorShape* result) {
  // If we don't find anything else, return scalar.
  *result = {};

  // Check to see if we have been given a default for all placeholders.
  if (context.params.count("shape")) {
    if (context.params.at("shape").size() != 1) {
      return errors::InvalidArgument(
          "You must pass no more than one default 'shape' to "
          "strip_unused_nodes");
    }
    const string& shape_string = context.params.at("shape")[0];
    TF_RETURN_IF_ERROR(TensorShapeFromString(shape_string, result));
  }

  // See if there's a particular type specified for this placeholder.
  if (context.params.count("name") || context.params.count("shape_for_name")) {
    if (!context.params.count("name") ||
        !context.params.count("shape_for_name") ||
        (context.params.at("shape_for_name").size() !=
         context.params.at("name").size())) {
      return errors::InvalidArgument(
          "You must pass a 'shape_for_name' arg for every 'name', e.g. "
          "strip_unused_nodes(name=foo, shape_for_name=\"2,2,1\", name=bar, "
          "shape_for_name=\"1\"");
    }
    const int name_count = context.params.at("name").size();
    for (int i = 0; i < name_count; ++i) {
      if (context.params.at("name")[i] == node_name) {
        const string& shape_string = context.params.at("shape_for_name")[i];
        TF_RETURN_IF_ERROR(TensorShapeFromString(shape_string, result));
      }
    }
  }

  return OkStatus();
}
}  // namespace

// Delete any nodes that don't contribute to the inference result.
Status StripUnusedNodes(const GraphDef& input_graph_def,
                        const TransformFuncContext& context,
                        GraphDef* output_graph_def) {
  std::set<string> required_nodes;
  std::set<string> input_nodes;
  for (const string& input : context.input_names) {
    required_nodes.insert(NodeNameFromInput(input));
    input_nodes.insert(NodeNameFromInput(input));
  }
  for (const string& output : context.output_names) {
    required_nodes.insert(output);
  }

  std::map<string, const NodeDef*> node_lookup;
  MapNamesToNodes(input_graph_def, &node_lookup);

  std::vector<string> current_inputs;
  for (const string& output_name : context.output_names) {
    current_inputs.push_back(NodeNameFromInput(output_name));
  }

  while (!current_inputs.empty()) {
    std::set<string> next_inputs;
    for (const string& current_input : current_inputs) {
      required_nodes.insert(current_input);
      if (input_nodes.count(current_input)) {
        continue;
      }
      if (!node_lookup.count(current_input)) {
        return errors::InvalidArgument("Input node ", current_input,
                                       " not found in graph");
      }
      const NodeDef* current_node = node_lookup[current_input];
      for (const string& input_name : current_node->input()) {
        string input_node_name = NodeNameFromInput(input_name);
        if (!required_nodes.count(input_node_name)) {
          next_inputs.insert(input_node_name);
        }
      }
    }
    current_inputs =
        std::vector<string>(next_inputs.begin(), next_inputs.end());
  }

  GraphDef filtered_graph_def;
  FilterGraphDef(input_graph_def,
                 [&](const NodeDef& node) {
                   return required_nodes.count(node.name()) > 0;
                 },
                 &filtered_graph_def);

  output_graph_def->Clear();
  for (const NodeDef& node : filtered_graph_def.node()) {
    if (input_nodes.count(node.name())) {
      NodeDef placeholder_node;
      if (node.op() == "Placeholder") {
        placeholder_node = node;
      } else {
        placeholder_node.set_op("Placeholder");
        placeholder_node.set_name(node.name());
        DataType type;
        TF_RETURN_IF_ERROR(TypeForPlaceholder(context, node.name(), &type));
        TensorShape shape;
        TF_RETURN_IF_ERROR(ShapeForPlaceholder(context, node.name(), &shape));
        SetNodeAttr("dtype", type, &placeholder_node);
        SetNodeAttr("shape", shape, &placeholder_node);
      }
      *(output_graph_def->mutable_node()->Add()) = placeholder_node;
    } else {
      *(output_graph_def->mutable_node()->Add()) = node;
    }
  }
  return OkStatus();
}

REGISTER_GRAPH_TRANSFORM("strip_unused_nodes", StripUnusedNodes);

}  // namespace graph_transforms
}  // namespace tensorflow

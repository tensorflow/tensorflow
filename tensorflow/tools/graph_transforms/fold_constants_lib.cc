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

#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

Status ReplaceSendRecvs(const GraphDef& original_graph_def,
                        const GraphDef& rewritten_graph_def,
                        const std::vector<string>& inputs,
                        const std::vector<string>& outputs,
                        GraphDef* output_graph_def) {
  std::map<string, const NodeDef*> original_map;
  MapNamesToNodes(original_graph_def, &original_map);
  std::map<string, string> new_node_names;
  for (const NodeDef& node : rewritten_graph_def.node()) {
    // If the op isn't a Recv, or it was in the original, nothing to do.
    if ((node.op() != "_Recv") || (original_map.count(node.name()) == 1)) {
      continue;
    }
    // See if it matches an input from the original.
    for (const string& input : inputs) {
      // Here we rely on the naming convention for the Recv nodes that
      // RewriteGraphForExecution adds in the place of the feed inputs.
      string input_prefix = "_recv_" + input + "_";
      if (StringPiece(node.name()).starts_with(input_prefix)) {
        // If it does, prepare to rename any inputs that refer to it.
        new_node_names[node.name()] = input;
      }
    }
  }

  std::vector<NodeDef> nodes_to_add;
  for (const NodeDef& node : rewritten_graph_def.node()) {
    if ((node.op() == "_Send") || (node.op() == "_Recv")) {
      // If the op is a Send or Recv that wasn't in the original, skip it.
      if (original_map.count(node.name()) == 0) {
        continue;
      }
    }
    NodeDef new_node;
    new_node = node;
    new_node.mutable_input()->Clear();
    for (const string& old_input : node.input()) {
      string input_prefix;
      string input_node_name;
      string input_suffix;
      NodeNamePartsFromInput(old_input, &input_prefix, &input_node_name,
                             &input_suffix);
      string new_input;
      if (new_node_names.count(input_node_name) > 0) {
        new_input =
            input_prefix + new_node_names[input_node_name] + input_suffix;
      } else {
        new_input = old_input;
      }
      *(new_node.mutable_input()->Add()) = new_input;
    }
    nodes_to_add.push_back(new_node);
  }
  for (std::pair<string, string> entry : new_node_names) {
    string removed_node_name = entry.second;
    const NodeDef* removed_node = original_map[removed_node_name];
    NodeDef new_node;
    new_node = *removed_node;
    nodes_to_add.push_back(new_node);
  }

  for (const NodeDef& node : nodes_to_add) {
    *output_graph_def->mutable_node()->Add() = node;
  }
  return Status::OK();
}

Status RemoveUnusedNodes(const GraphDef& input_graph_def,
                         const TransformFuncContext& context,
                         GraphDef* output_graph_def) {
  std::map<string, const NodeDef*> node_map;
  MapNamesToNodes(input_graph_def, &node_map);

  std::set<string> used_nodes;
  for (const string& input : context.input_names) {
    used_nodes.insert(input);
  }
  std::vector<string> current_nodes = context.output_names;
  while (!current_nodes.empty()) {
    std::set<string> next_nodes;
    for (const string& node_name : current_nodes) {
      used_nodes.insert(node_name);
      if (node_map.count(node_name) == 0) {
        LOG(ERROR) << "Bad graph structure, no node named '" << node_name
                   << "' found for input lookup";
        return errors::InvalidArgument("Bad graph structure, no node named '",
                                       node_name, "' found for input lookup");
      }
      const NodeDef& node = *(node_map[node_name]);
      for (const string& input_name : node.input()) {
        const string& input_node_name = NodeNameFromInput(input_name);
        if (used_nodes.count(input_node_name) == 0) {
          next_nodes.insert(input_node_name);
        }
      }
    }
    current_nodes = std::vector<string>(next_nodes.begin(), next_nodes.end());
  }
  FilterGraphDef(
      input_graph_def,
      [&](const NodeDef& node) { return used_nodes.count(node.name()) > 0; },
      output_graph_def);

  return Status::OK();
}

// Converts any sub-graphs that can be resolved into constant expressions into
// single Const ops.
Status FoldConstants(const GraphDef& input_graph_def,
                     const TransformFuncContext& context,
                     GraphDef* output_graph_def) {
  // Some older GraphDefs have saved _output_shapes attributes which are out of
  // date and cause import errors, so clean them up first.
  GraphDef cleaned_graph_def;
  RemoveAttributes(input_graph_def, {"_output_shapes"}, &cleaned_graph_def);
  Graph input_graph(OpRegistry::Global());
  ImportGraphDefOptions import_opts;
  TF_RETURN_IF_ERROR(
      ImportGraphDef(import_opts, cleaned_graph_def, &input_graph, nullptr));
  DeviceAttributes device_attributes;
  subgraph::RewriteGraphMetadata metadata;
  TF_RETURN_IF_ERROR(subgraph::RewriteGraphForExecution(
      &input_graph, context.input_names, context.output_names, {},
      device_attributes, false /* use_function_convention */, &metadata));
  bool was_mutated;
  // Exclude specified nodes from constant folding.
  ConstantFoldingOptions cf_opts;
  if (context.params.count("exclude_op") > 0) {
    const auto& excluded_nodes = context.params.at("exclude_op");
    const std::set<string> excluded_nodes_set(excluded_nodes.begin(),
                                              excluded_nodes.end());
    cf_opts.consider = [excluded_nodes_set](const Node* n) {
      return excluded_nodes_set.find(n->op_def().name()) ==
             excluded_nodes_set.end();
    };
  }
  TF_RETURN_IF_ERROR(ConstantFold(cf_opts, nullptr, Env::Default(), nullptr,
                                  &input_graph, &was_mutated));
  GraphDef folded_graph_def;
  input_graph.ToGraphDef(&folded_graph_def);
  GraphDef send_recvs_replaced;
  TF_RETURN_IF_ERROR(ReplaceSendRecvs(input_graph_def, folded_graph_def,
                                      context.input_names, context.output_names,
                                      &send_recvs_replaced));
  TF_RETURN_IF_ERROR(
      RemoveUnusedNodes(send_recvs_replaced, context, output_graph_def));
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("fold_constants", FoldConstants);

}  // namespace graph_transforms
}  // namespace tensorflow

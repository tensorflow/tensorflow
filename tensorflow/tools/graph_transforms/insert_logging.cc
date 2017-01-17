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

// Clears the device field of all ops in the graph.
Status InsertLogging(const GraphDef& input_graph_def,
                     const TransformFuncContext& context,
                     GraphDef* output_graph_def) {
  std::unordered_set<string> ops;
  bool has_ops;
  if (context.params.count("op")) {
    has_ops = true;
    for (const string& op : context.params.at("op")) {
      ops.insert(op);
    }
  } else {
    has_ops = false;
  }

  std::unordered_set<string> prefixes;
  bool has_prefixes;
  if (context.params.count("prefix")) {
    has_prefixes = true;
    for (const string& prefix : context.params.at("prefix")) {
      prefixes.insert(prefix);
    }
  } else {
    has_prefixes = false;
  }

  string message;
  TF_RETURN_IF_ERROR(context.GetOneStringParameter("message", "", &message));

  bool show_name;
  TF_RETURN_IF_ERROR(
      context.GetOneBoolParameter("show_name", false, &show_name));

  bool show_op;
  TF_RETURN_IF_ERROR(context.GetOneBoolParameter("show_op", false, &show_op));

  int32 first_n;
  TF_RETURN_IF_ERROR(context.GetOneInt32Parameter("first_n", -1, &first_n));

  int32 summarize;
  TF_RETURN_IF_ERROR(
      context.GetOneInt32Parameter("summarize", 1024, &summarize));

  std::unordered_map<string, std::set<int>> node_outputs;
  for (const NodeDef& node : input_graph_def.node()) {
    for (const string& input : node.input()) {
      const string canonical_input = CanonicalInputName(input);
      string prefix;
      string name;
      string suffix;
      NodeNamePartsFromInput(canonical_input, &prefix, &name, &suffix);
      const string output_index_string = suffix.substr(1, suffix.size() - 1);
      int32 output_index;
      if (!strings::safe_strto32(output_index_string, &output_index)) {
        return errors::InvalidArgument("Couldn't understand output number in ",
                                       input);
      }
      node_outputs[name].insert(output_index);
    }
  }

  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> ignore_when_renaming;
  GraphDef logged_graph_def;
  for (const NodeDef& node : input_graph_def.node()) {
    NodeDef* new_node = logged_graph_def.mutable_node()->Add();
    new_node->CopyFrom(node);
    if (node_outputs[node.name()].empty()) {
      // There were no outputs found to this node, so skip it.
      continue;
    }
    const bool op_matches = (ops.count(node.op()) > 0);
    bool prefix_matches = false;
    for (const string& prefix : prefixes) {
      if (StringPiece(node.name()).starts_with(prefix)) {
        prefix_matches = true;
      }
    }
    // If we're not looking for ops, or we found the right op, and if we're not
    // looking for prefixes or we found the right prefix, then add logging here.
    if ((!has_ops || op_matches) && (!has_prefixes || prefix_matches)) {
      const string name_suffix = "__print__";
      DataTypeVector input_types;
      DataTypeVector output_types;
      TF_RETURN_IF_ERROR(GetInOutTypes(node, &input_types, &output_types));
      NodeDef* print_node = logged_graph_def.mutable_node()->Add();
      print_node->set_op("Print");
      print_node->set_name(strings::StrCat(node.name(), name_suffix));
      string node_message;
      if (show_op) {
        node_message += ";" + node.op() + ";";
      }
      if (show_name) {
        node_message += ";" + print_node->name() + ";";
      }
      node_message += message;
      SetNodeAttr("message", node_message, print_node);
      SetNodeAttr("first_n", first_n, print_node);
      SetNodeAttr("summarize", summarize, print_node);
      print_node->add_input(node.name() + ":0");
      SetNodeAttr("T", output_types[0], print_node);
      for (int output_index : node_outputs[node.name()]) {
        print_node->add_input(strings::StrCat(node.name(), ":", output_index));
      }
      SetNodeAttr("U", output_types, print_node);
      ignore_when_renaming.insert(print_node->name());
      // Rewrite the graph so all references to the first input of the original
      // op now pull from the print op instead, so it's executed.
      inputs_to_rename[node.name() + ":0"] =
          strings::StrCat(node.name(), name_suffix, ":0");
    }
  }

  output_graph_def->Clear();
  RenameNodeInputs(logged_graph_def, inputs_to_rename, ignore_when_renaming,
                   output_graph_def);

  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("insert_logging", InsertLogging);

}  // namespace graph_transforms
}  // namespace tensorflow

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

// Deletes any specified types of nodes, unless they're necessary for the
// graph's inputs or outputs.
Status RemoveNodes(const GraphDef& input_graph_def,
                   const TransformFuncContext& context,
                   GraphDef* output_graph_def) {
  if (!context.params.count("op")) {
    return errors::InvalidArgument(
        "remove_nodes expects at least one 'op'"
        "argument, e.g. remove_nodes(op=Identity)");
  }
  int32_t max_inputs;
  TF_RETURN_IF_ERROR(
      context.GetOneInt32Parameter("max_inputs", 1, &max_inputs));

  // Make sure we don't get rid of any nodes used as graph inputs or outputs.
  std::set<string> required_nodes;
  for (const string& input : context.input_names) {
    required_nodes.insert(NodeNameFromInput(input));
  }
  for (const string& output : context.output_names) {
    required_nodes.insert(NodeNameFromInput(output));
  }

  std::vector<string> ops_to_remove = context.params.at("op");
  GraphDef current_graph_def = input_graph_def;
  for (const string& op : ops_to_remove) {
    for (int num_inputs = 1; num_inputs <= max_inputs; ++num_inputs) {
      // Look for a variable number of inputs.
      OpTypePattern pattern = {op};
      pattern.inputs.resize(num_inputs);
      for (int i = 0; i < num_inputs; ++i) {
        pattern.inputs[i] = {"*"};
      }
      // Keep looking for nodes to remove until there are no more changes.
      bool any_nodes_removed;
      do {
        any_nodes_removed = false;
        std::map<string, string> inputs_to_rename;
        GraphDef replaced_graph_def;
        TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
            current_graph_def, pattern,
            [&inputs_to_rename, &required_nodes, &any_nodes_removed](
                const NodeMatch& match, const std::set<string>& input_nodes,
                const std::set<string>& output_nodes,
                std::vector<NodeDef>* new_nodes) {
              const NodeDef& replace_node = match.node;
              // If this node is needed in the inputs or outputs don't replace
              // it.
              if (required_nodes.count(replace_node.name())) {
                LOG(INFO) << "Skipping replacement for " << replace_node.name();
                CopyOriginalMatch(match, new_nodes);
                return OkStatus();
              }
              const NodeDef& input_node = match.inputs[0].node;
              string target_name = input_node.name();
              for (const string& input : replace_node.input()) {
                if (!input.compare(0, target_name.size(), target_name)) {
                  if (input.size() == target_name.size() ||
                      input[target_name.size()] == ':') {
                    target_name = input;
                    break;
                  }
                }
              }
              inputs_to_rename[replace_node.name()] = target_name;
              inputs_to_rename["^" + replace_node.name()] =
                  "^" + input_node.name();
              new_nodes->push_back(input_node);
              any_nodes_removed = true;
              return OkStatus();
            },
            {true}, &replaced_graph_def));
        // Make sure all references to removed nodes now point to their inputs.
        TF_RETURN_IF_ERROR(
            RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                             std::unordered_set<string>(), &current_graph_def));
      } while (any_nodes_removed);
    }
  }

  *output_graph_def = current_graph_def;
  return OkStatus();
}

REGISTER_GRAPH_TRANSFORM("remove_nodes", RemoveNodes);

}  // namespace graph_transforms
}  // namespace tensorflow

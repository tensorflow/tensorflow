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

// Renames all nodes not uses as graph inputs or outputs to short numerical
// forms.
absl::Status ObfuscateNames(const GraphDef& input_graph_def,
                            const TransformFuncContext& context,
                            GraphDef* output_graph_def) {
  std::unordered_set<string> required_nodes;
  for (const string& input : context.input_names) {
    required_nodes.insert(input);
  }
  for (const string& output : context.output_names) {
    required_nodes.insert(output);
  }

  const string valid_chars =
      "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const int64_t chars_size = valid_chars.size();

  std::map<string, string> new_names;
  int64_t name_index = 0;
  for (const NodeDef& input_node : input_graph_def.node()) {
    const string& old_name = input_node.name();
    string new_name;
    if (required_nodes.count(old_name)) {
      new_name = old_name;
    } else {
      do {
        int64_t remaining = name_index;
        new_name = "";
        while (true) {
          const int64_t remainder = (remaining % chars_size);
          const char current_char = valid_chars[remainder];
          new_name = current_char + new_name;
          remaining /= chars_size;
          if (remaining <= 0) {
            break;
          }
        }
        ++name_index;
      } while (required_nodes.count(new_name));
    }
    new_names[old_name] = new_name;
  }

  output_graph_def->Clear();
  for (const NodeDef& input_node : input_graph_def.node()) {
    NodeDef* node = output_graph_def->mutable_node()->Add();
    *node = input_node;
    const string& old_name = input_node.name();
    node->set_name(new_names[old_name]);
    node->mutable_input()->Clear();
    for (const string& input_name : input_node.input()) {
      string prefix;
      string input_node_name;
      string suffix;
      NodeNamePartsFromInput(input_name, &prefix, &input_node_name, &suffix);
      if (new_names.count(input_node_name) == 0) {
        return errors::InvalidArgument("No node named ", input_node_name,
                                       " for input to ", old_name);
      }
      string new_input_name = prefix + new_names[input_node_name] + suffix;
      *(node->mutable_input()->Add()) = new_input_name;
    }
  }

  return absl::OkStatus();
}

REGISTER_GRAPH_TRANSFORM("obfuscate_names", ObfuscateNames);

}  // namespace graph_transforms
}  // namespace tensorflow

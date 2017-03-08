/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tools/graph_transforms/transform_utils.h"

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace graph_transforms {

void MapNamesToNodes(const GraphDef& graph_def,
                     std::map<string, const NodeDef*>* result) {
  for (const NodeDef& node : graph_def.node()) {
    (*result)[node.name()] = &node;
  }
}

void NodeNamePartsFromInput(string input_name, string* prefix,
                            string* node_name, string* suffix) {
  std::vector<string> input_parts = str_util::Split(input_name, ':');
  if (input_parts.size() < 2) {
    *suffix = "";
  } else {
    *suffix = ":" + input_parts[1];
  }
  StringPiece node_name_piece(input_parts[0]);
  if (node_name_piece.Consume("^")) {
    *prefix = "^";
  } else {
    *prefix = "";
  }
  *node_name = node_name_piece.ToString();
}

string NodeNameFromInput(string input_name) {
  string prefix;
  string node_name;
  string suffix;
  NodeNamePartsFromInput(input_name, &prefix, &node_name, &suffix);
  return node_name;
}

void FilterGraphDef(const GraphDef& input_graph_def,
                    std::function<bool(const NodeDef&)> selector,
                    GraphDef* output_graph_def) {
  output_graph_def->mutable_node()->Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    if (selector(node)) {
      output_graph_def->mutable_node()->Add()->CopyFrom(node);
    }
  }
}

void RemoveAttributes(const GraphDef& input_graph_def,
                      const std::vector<string>& attributes,
                      GraphDef* output_graph_def) {
  output_graph_def->mutable_node()->Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    NodeDef* new_node = output_graph_def->mutable_node()->Add();
    new_node->CopyFrom(node);
    for (const string& attribute : attributes) {
      new_node->mutable_attr()->erase(attribute);
    }
  }
}

}  // namespace graph_transforms
}  // namespace tensorflow

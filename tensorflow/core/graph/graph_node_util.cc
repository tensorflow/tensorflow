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
#include "tensorflow/core/graph/graph_node_util.h"

#include <vector>

#include "absl/container/btree_set.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

string SummarizeNode(const Node& node) { return SummarizeNodeDef(node.def()); }

string FormatNodeForError(const Node& node) {
  return FormatNodeDefForError(node.def());
}

absl::Status NameRangesForNode(const Node& node, const OpDef& op_def,
                               NameRangeMap* inputs, NameRangeMap* outputs) {
  return NameRangesForNode(node.def(), op_def, inputs, outputs);
}

absl::Status AttachDef(const absl::Status& status, const Node& node,
                       bool allow_multiple_formatted_node) {
  return AttachDef(status, node.def(), allow_multiple_formatted_node);
}

absl::btree_set<string> GetMergedNames(const std::vector<string>& from_names,
                                       const std::vector<string>& to_names) {
  absl::btree_set<string> merged_names;
  merged_names.insert(from_names.begin(), from_names.end());
  merged_names.insert(to_names.begin(), to_names.end());
  return merged_names;
}

void MergeDebugInfo(const NodeDebugInfo& from, Node* to_node) {
  NodeDebugInfo to = NodeDebugInfo(*to_node);
  if (!from.original_node_names.empty()) {
    auto node_names =
        GetMergedNames(from.original_node_names, to.original_node_names);
    to_node->set_original_node_names({node_names.begin(), node_names.end()});
  }
  if (!from.original_func_names.empty()) {
    auto func_names =
        GetMergedNames(from.original_func_names, to.original_func_names);
    to_node->set_original_func_names({func_names.begin(), func_names.end()});
  }
}

void MergeDebugInfo(const NodeDebugInfo& from, NodeDef* to_node_def) {
  NodeDebugInfo to = NodeDebugInfo(*to_node_def);
  if (!from.original_node_names.empty()) {
    auto node_names =
        GetMergedNames(from.original_node_names, to.original_node_names);
    to_node_def->mutable_experimental_debug_info()->clear_original_node_names();
    *to_node_def->mutable_experimental_debug_info()
         ->mutable_original_node_names() = {node_names.begin(),
                                            node_names.end()};
  }
  if (!from.original_func_names.empty()) {
    auto func_names =
        GetMergedNames(from.original_func_names, to.original_func_names);
    to_node_def->mutable_experimental_debug_info()->clear_original_func_names();
    *to_node_def->mutable_experimental_debug_info()
         ->mutable_original_func_names() = {func_names.begin(),
                                            func_names.end()};
  }
}

void MergeDebugInfo(const NodeDef& from_node_def, NodeDef* to_node_def) {
  MergeDebugInfo(NodeDebugInfo(from_node_def), to_node_def);
}
}  // namespace tensorflow

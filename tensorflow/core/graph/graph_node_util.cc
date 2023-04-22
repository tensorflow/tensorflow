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

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

string SummarizeNode(const Node& node) { return SummarizeNodeDef(node.def()); }

string FormatNodeForError(const NodeDebugInfo& debug_info) {
  return debug_info.original_node_names.empty()
             ? errors::FormatNodeNameForError(debug_info.name)
             : errors::FormatNodeNamesForError(debug_info.original_node_names);
}

string FormatNodeForError(const Node& node) {
  return FormatNodeForError(NodeDebugInfo(node));
}

Status NameRangesForNode(const Node& node, const OpDef& op_def,
                         NameRangeMap* inputs, NameRangeMap* outputs) {
  return NameRangesForNode(node.def(), op_def, inputs, outputs);
}

Status AttachDef(const Status& status, const Node& node,
                 bool allow_multiple_formatted_node) {
  return AttachDef(status, node.def(), allow_multiple_formatted_node);
}

void GetMergedOriginalNodeNames(const NodeDebugInfo& from,
                                const NodeDebugInfo& to,
                                std::set<string>* names) {
  if (!from.original_node_names.empty()) {
    names->insert(from.original_node_names.begin(),
                  from.original_node_names.end());
  } else {
    names->insert(from.name);
  }
  names->insert(to.original_node_names.begin(), to.original_node_names.end());
}

void MergeDebugInfo(const NodeDebugInfo& from, Node* to) {
  std::set<string> names;
  GetMergedOriginalNodeNames(from, NodeDebugInfo(*to), &names);
  to->set_original_node_names({names.begin(), names.end()});
}

void MergeDebugInfo(const NodeDebugInfo& from, NodeDef* to) {
  std::set<string> names;
  GetMergedOriginalNodeNames(from, NodeDebugInfo(*to), &names);
  to->mutable_experimental_debug_info()->clear_original_node_names();
  if (!names.empty()) {
    *to->mutable_experimental_debug_info()->mutable_original_node_names() = {
        names.begin(), names.end()};
  }
}

void MergeDebugInfo(const NodeDef& from, NodeDef* to) {
  MergeDebugInfo(NodeDebugInfo(from), to);
}
}  // namespace tensorflow

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
#ifndef TENSORFLOW_CORE_GRAPH_GRAPH_NODE_UTIL_H_
#define TENSORFLOW_CORE_GRAPH_GRAPH_NODE_UTIL_H_

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
class Node;
struct NodeDebugInfo;

// We forward declare protos so that kernels don't need to depend on them
class NodeDef;
class OpDef;

// Produce a human-readable version of a Node or NodeDef that is more concise
// than a text-format proto.
string SummarizeNode(const Node& node);

// Produces a formatted string pattern from the node which can uniquely identify
// this node upstream to produce an informative error message. The pattern
// followed is: {{node <node_name>}}
string FormatNodeForError(const Node& node);

// Merges the original node names from the debug information of 'from' to the
// debug information of 'to'.
void MergeDebugInfo(const NodeDebugInfo& from, Node* to);
void MergeDebugInfo(const NodeDebugInfo& from, NodeDef* to);
void MergeDebugInfo(const NodeDef& from, NodeDef* to);

// Computes the mapping from input/output argument name to the
// corresponding input/output index range.  For example,
// input "foo" corresponds to input indices
//   [ (*inputs)["foo"].first, (*inputs)["foo"].second ).
// NOTE(mrry): To reduce allocations when the map is used and save
// space, the returned `NameRangeMap` objects borrow the input/output
// argument names from `op_def`. The `op_def` must outlive the
// returned `NameRangeMap` objects.
Status NameRangesForNode(const Node& node, const OpDef& op_def,
                         NameRangeMap* inputs, NameRangeMap* outputs);

// Returns "status" with formatted Node attached as additional text
// in the error message. If 'allow_multiple_formatted_node' is false and there
// is already a formatted Node present in 'status', we simply attach the name
// of the Node instead of the formatted string.
Status AttachDef(const Status& status, const Node& node,
                 bool allow_multiple_formatted_node = false);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_GRAPH_NODE_UTIL_H_

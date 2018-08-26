/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_FUNCTIONALIZE_CONTROL_FLOW_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_FUNCTIONALIZE_CONTROL_FLOW_UTIL_H_

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/graph/graph.h"

// Utility functions shared between functionalize cond and while.

namespace tensorflow {

// Check that the graph has no cycle containing the given node.
Status CheckNodeNotInCycle(const Node* node, const int num_nodes);

// Comparison function used for sorting nodes consistently.
// a) resource variables are last, and
// b) sort lexicographically by name (for deterministic output).
struct NodeCmpByNameResourcesLast {
  bool operator()(const Node* lhs, const Node* rhs) const;
};

// Returns the Node* created from the NodeDef in the Graph.
xla::StatusOr<Node*> AddNodeDefToGraph(const NodeDef& node_def, Graph* graph);

// Build a retval node of given type and index.
xla::StatusOr<Node*> BuildRetvalNode(Graph* graph, DataType type, int index);

// Returns a textual representation of the names of the nodes in the input.
template <typename T>
string NodesToString(const T& nodes) {
  return strings::StrCat("{",
                         absl::StrJoin(nodes, ",",
                                       [](string* output, const Node* node) {
                                         strings::StrAppend(output,
                                                            node->name());
                                       }),
                         "}");
}

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_FUNCTIONALIZE_CONTROL_FLOW_UTIL_H_

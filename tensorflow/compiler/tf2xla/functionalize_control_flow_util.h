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
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph.h"

// Utility functions shared between functionalize cond and while
// or used by other graph optimization passes.

namespace tensorflow {

using NodeFilter = std::function<bool(const Node*)>;

// Information about a loop argument.
struct WhileLoopArg {
  // Every loop argument has an Enter node.
  Node* enter;

  // Is the loop argument a loop-invariant value? Taken from the `is_constant`
  // attribute on the Enter node.
  bool is_loop_invariant;

  // If 'is_loop_invariant' is true, the following are all nullptr. Non-constant
  // arguments must have all of the following nodes:
  Node* merge = nullptr;
  Node* switch_node = nullptr;
  Node* next_iteration = nullptr;
  Node* exit = nullptr;
};

// Information about a loop frame.
struct WhileLoopFrame {
  string name;

  // Pointer to the parent frame. The root frame has a pointer to itself.
  WhileLoopFrame* parent = nullptr;
  int num_children = 0;

  // Arguments to this loop.
  std::vector<WhileLoopArg> args;

  // The loop condition of the loop. There should be exactly one loop condition
  // in every loop.
  Node* loop_cond = nullptr;

  // Set of nodes that belong to the loop frame.
  std::unordered_set<Node*> nodes;

  // After `ExtractWhileLoopFrames` this is true if for all control flow nodes
  // of this frame `node_filter` returns true, i.e., the frame should be
  // functionalized, and false otherwise.
  bool should_be_functionalized = true;
};

// Extracts v1 while loops within a graph and creates a map of
// <ControlFLowInfo.name, WhileLoopFrame>.
// If `node_filter` is defined, then we keep track of frames that should be
// functionalized according to the filter (see comment for
// `FunctionalizeControlFlow` for more details about node filters).
Status ExtractWhileLoopFrames(
    const std::vector<ControlFlowInfo>& cf_info, const Graph* graph,
    std::unordered_map<string, WhileLoopFrame>* frames,
    const NodeFilter& node_filter = {});

// Check that the graph has no cycle containing the given node.
Status CheckNodeNotInCycle(const Node* node, const int num_nodes);

// Comparison function used for sorting nodes consistently.
// a) resource variables are last, and
// b) sort lexicographically by name (for deterministic output).
struct NodeCmpByNameResourcesLast {
  bool operator()(const Node* lhs, const Node* rhs) const;
};

// Returns the Node* created from the NodeDef in the Graph.
StatusOr<Node*> AddNodeDefToGraph(const NodeDef& node_def, Graph* graph);

// Build a retval node of given type and index.
StatusOr<Node*> BuildRetvalNode(Graph* graph, DataType type, int index);

// Returns a textual representation of the names of the nodes in the input.
template <typename T>
string NodesToString(const T& nodes) {
  return absl::StrCat("{",
                      absl::StrJoin(nodes, ",",
                                    [](string* output, const Node* node) {
                                      absl::StrAppend(output, node->name());
                                    }),
                      "}");
}

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_FUNCTIONALIZE_CONTROL_FLOW_UTIL_H_

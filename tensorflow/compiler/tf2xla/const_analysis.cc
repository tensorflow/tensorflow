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

#include "tensorflow/compiler/tf2xla/const_analysis.h"

#include <unordered_map>
#include <unordered_set>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {
// Backwards dataflow analysis that finds arguments to a graph that must be
// compile-time constants.
Status BackwardsConstAnalysis(const Graph& g,
                              std::vector<bool>* compile_time_const_arg_indices,
                              std::vector<bool>* compile_time_const_nodes,
                              std::function<bool(const Edge&)> edge_filter) {
  std::vector<bool> compile_time_const_nodes_impl;
  if (compile_time_const_nodes) {
    CHECK_EQ(compile_time_const_nodes->size(), g.num_node_ids());
  } else {
    compile_time_const_nodes_impl.resize(g.num_node_ids());
    compile_time_const_nodes = &compile_time_const_nodes_impl;
  }

  Status status;
  auto visit = [&](Node* node) {
    if (!status.ok()) return;

    // If this is a metadata-only op, don't propagate the const requirement.
    if (XlaOpRegistry::IsMetadataOp(node->type_string())) {
      return;
    }

    // If this node must be const, and it isn't a metadata op, then all of its
    // parents must be const.
    if ((*compile_time_const_nodes)[node->id()]) {
      if (node->type_string() == "_Arg") {
        int index;
        status = GetNodeAttr(node->attrs(), "index", &index);
        if (!status.ok()) return;
        if (compile_time_const_arg_indices) {
          (*compile_time_const_arg_indices)[index] = true;
        }
        return;
      }
      for (const Edge* pred : node->in_edges()) {
        if (!pred->IsControlEdge() && edge_filter(*pred)) {
          (*compile_time_const_nodes)[pred->src()->id()] = true;
        }
      }
      return;
    }

    // Mark any compile-time constant operator arguments as const.
    std::vector<int> const_input_idxs;
    status = XlaOpRegistry::CompileTimeConstantInputs(
        node->def(), node->op_def(), &const_input_idxs);

    if (!status.ok()) {
      return;
    }

    for (Edge const* edge : node->in_edges()) {
      if (absl::c_binary_search(const_input_idxs, edge->dst_input()) &&
          edge_filter(*edge)) {
        (*compile_time_const_nodes)[edge->src()->id()] = true;
      }
    }
  };

  // Post-order traversal visits nodes in reverse topological order for an
  // acyclic graph.
  DFS(g, /*enter=*/{}, /*leave=*/visit, NodeComparatorName{},
      [](const Edge& edge) { return !edge.src()->IsNextIteration(); });
  return status;
}

}  // namespace tensorflow

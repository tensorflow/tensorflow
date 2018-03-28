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

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {

// Backwards dataflow analysis that finds arguments to a graph that must be
// compile-time constants.
Status BackwardsConstAnalysis(const Graph& g,
                              std::vector<bool>* compile_time_const_args) {
  // Operators that don't look at the data of their inputs, just the shapes.
  const std::unordered_set<string> metadata_ops = {
      "Rank",
      "Shape",
      "ShapeN",
      "Size",
  };

  Status status;
  std::unordered_set<const Node*> must_be_const;
  auto visit = [&status, &metadata_ops, &must_be_const,
                compile_time_const_args](Node* node) {
    if (!status.ok()) return;

    // If this is a metadata-only op, don't propagate the const requirement.
    if (metadata_ops.find(node->type_string()) != metadata_ops.end()) return;

    // If this node must be const, and it isn't a metadata op, then all of its
    // parents must be const.
    if (must_be_const.find(node) != must_be_const.end()) {
      if (node->type_string() == "_Arg") {
        int index;
        status = GetNodeAttr(node->attrs(), "index", &index);
        if (!status.ok()) return;
        compile_time_const_args->at(index) = true;
        return;
      }
      for (const Edge* pred : node->in_edges()) {
        if (!pred->IsControlEdge()) {
          must_be_const.insert(pred->src());
        }
      }
      return;
    }

    // Mark any compile-time constant operator arguments as const.
    const std::unordered_set<string>* const_inputs =
        XlaOpRegistry::CompileTimeConstantInputs(node->type_string());
    if (!const_inputs || const_inputs->empty()) return;

    NameRangeMap input_name_ranges;
    status =
        NameRangesForNode(*node, node->op_def(), &input_name_ranges, nullptr);
    if (!status.ok()) return;

    for (const string& input : *const_inputs) {
      auto name_range = input_name_ranges.find(input);
      if (name_range == input_name_ranges.end()) continue;

      for (Edge const* edge : node->in_edges()) {
        if (edge->dst_input() >= name_range->second.first &&
            edge->dst_input() < name_range->second.second) {
          must_be_const.insert(edge->src());
        }
      }
    }
  };

  // Post-order traversal visits nodes in reverse topological order for an
  // acyclic graph.
  DFS(g, {}, visit);
  return status;
}

}  // namespace tensorflow

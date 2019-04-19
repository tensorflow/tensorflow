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

#include "tensorflow/core/common_runtime/graph_optimizer.h"

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/optimizer_cse.h"

namespace tensorflow {
namespace {

// Replaces occurrences of parallel_concat with the implementation based on
// unsafe ops. Sets removed_any to true if any parallel_concats were removed;
// leaves it untouched otherwise.
class ParallelConcatRemovePass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (options.graph == nullptr) {
      // TODO(apassos) returning OK feels weird here as we can't do anything
      // without a graph, but some tests require this.
      return Status::OK();
    }
    Graph* g = options.graph->get();
    if (g == nullptr) {
      return errors::Internal(
          "Parallel concat removal should happen before partitioning and a "
          "graph should be available.");
    }
    gtl::InlinedVector<Node*, 2> matches;
    for (Node* n : g->op_nodes()) {
      if (n->type_string() == "ParallelConcat") {
        matches.push_back(n);
      }
    }
    for (Node* n : matches) {
      AttrSlice n_attrs = n->attrs();
      auto base_make_node = [n, &n_attrs](const string& op,
                                          const string& name) {
        NodeDebugInfo debug_info(*n);
        NodeBuilder node_builder(name, op, OpRegistry::Global(), &debug_info);
        node_builder.Device(n->requested_device());
        string colo;
        if (GetNodeAttr(n_attrs, "_class", &colo).ok()) {
          node_builder.Attr("_class", colo);
        }
        return node_builder;
      };
      auto make_node = [n, g, &base_make_node](string op) {
        return base_make_node(
            op, g->NewName(strings::StrCat(n->name(), "/Internal")));
      };
      DataType dtype;
      TF_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "T", &dtype));
      TensorShapeProto shape;
      TF_RETURN_IF_ERROR(GetNodeAttr(n_attrs, "shape", &shape));

      // Add the start node
      Node* start;
      TF_RETURN_IF_ERROR(make_node("_ParallelConcatStart")
                             .Attr("shape", shape)
                             .Attr("dtype", dtype)
                             .Finalize(g, &start));

      // Add all the inplace_updates.
      std::vector<Node*> control_nodes;
      for (const Edge* input_edge : n->in_edges()) {
        if (input_edge->IsControlEdge()) {
          g->AddControlEdge(input_edge->src(), start);
          continue;
        }

        Node* update;
        TF_RETURN_IF_ERROR(
            make_node("_ParallelConcatUpdate")
                .Attr("loc", input_edge->dst_input())
                .Input(start)
                .Input(input_edge->src(), input_edge->src_output())
                .Finalize(g, &update));
        control_nodes.push_back(update);
      }

      // Add the final identity.
      NodeBuilder identity_def = base_make_node("Identity", n->name());
      identity_def.Input(start, 0);
      for (Node* s : control_nodes) {
        identity_def.ControlInput(s);
      }
      Node* identity_node;
      TF_RETURN_IF_ERROR(identity_def.Finalize(g, &identity_node));

      // Remove the node and redirect edges.
      for (auto* e : n->out_edges()) {
        if (e->IsControlEdge()) {
          g->AddControlEdge(identity_node, e->dst());
        } else {
          g->AddEdge(identity_node, 0, e->dst(), e->dst_input());
        }
      }
      g->RemoveNode(n);
    }
    return Status::OK();
  }
};
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 0,
                      ParallelConcatRemovePass);

}  // namespace
}  // namespace tensorflow

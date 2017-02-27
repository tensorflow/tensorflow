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
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace {

// Replaces ReadVariableOp nodes which are only used by Sends and sinks with
// _UnsafeReadVariable nodes, as this transforamtion is safe and will improve
// performance.
class ResourceVariableReadPass : public GraphOptimizationPass {
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
          "Read to unsafe read conversion should happen before partitioning "
          "and a graph should be available.");
    }
    gtl::InlinedVector<Node*, 2> matches;
    for (Node* n : g->nodes()) {
      if (n->type_string() == "ReadVariableOp") {
        bool skip = false;
        for (const Edge* e : n->out_edges()) {
          if (!e->dst()->IsSend() && e->dst()->name() != "_SINK") {
            skip = true;
          }
        }
        if (!skip) {
          matches.push_back(n);
        }
      }
    }
    for (Node* read : matches) {
      DataType dtype;
      TF_RETURN_IF_ERROR(GetNodeAttr(AttrSlice(read->def()), "dtype", &dtype));
      std::vector<Node*> in_control_edges;
      std::vector<std::pair<Node*, int>> in_edges;
      for (const Edge* edge : read->in_edges()) {
        if (edge->IsControlEdge()) {
          in_control_edges.push_back(edge->src());
        } else {
          in_edges.push_back({edge->src(), edge->src_output()});
        }
      }
      std::vector<Node*> out_control_edges;
      std::vector<std::pair<Node*, int>> out_edges;
      for (const Edge* edge : read->out_edges()) {
        if (edge->IsControlEdge()) {
          out_control_edges.push_back(edge->dst());
        } else {
          out_edges.push_back({edge->dst(), edge->dst_input()});
        }
      }
      string name = read->name();
      string device_name = read->assigned_device_name();
      g->RemoveNode(read);
      Node* unsafe_read;
      NodeBuilder unsafe_read_builder(g->NewName(name), "_UnsafeReadVariable");
      for (Node* node : in_control_edges) {
        unsafe_read_builder.ControlInput(node);
      }
      for (const std::pair<Node*, int>& p : in_edges) {
        unsafe_read_builder.Input(p.first, p.second);
      }
      TF_RETURN_IF_ERROR(
          unsafe_read_builder.Attr("dtype", dtype).Finalize(g, &unsafe_read));
      unsafe_read->set_assigned_device_name(device_name);
      for (Node* node : out_control_edges) {
        g->AddControlEdge(unsafe_read, node);
      }
      for (std::pair<Node*, int> p : out_edges) {
        g->AddEdge(unsafe_read, 0, p.first, p.second);
      }
    }
    return Status::OK();
  }
};
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 0,
                      ResourceVariableReadPass);

}  // namespace
}  // namespace tensorflow

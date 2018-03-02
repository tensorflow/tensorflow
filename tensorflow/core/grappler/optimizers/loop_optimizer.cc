/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/loop_optimizer.h"

#include <unordered_map>
#include <unordered_set>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace grappler {
namespace {

Status RemoveStackOps(const GraphDef& graph, GraphDef* optimized_graph) {
  SimpleGraphView graph_view;
  TF_RETURN_IF_ERROR(graph_view.Initialize(graph));
  const std::unordered_set<string> op_types_to_traverse(
      {"Stack", "StackV2", "Enter", "Switch", "RefSwitch", "Identity"});
  std::set<int> nodes_to_delete;
  for (int node_idx = 0; node_idx < graph.node_size(); ++node_idx) {
    const NodeDef& node = graph.node(node_idx);
    if (IsStackOp(node)) {
      std::set<int> nodes_found;
      graph_view.DepthFirstSearch(op_types_to_traverse, node_idx, &nodes_found);
      bool found_pop = false;
      bool found_unexpected = false;
      for (int found_idx : nodes_found) {
        const NodeDef& node = graph.node(found_idx);
        if (IsStackPushOp(node) || IsStackOp(node) || IsStackCloseOp(node)) {
          continue;
        } else if (IsStackPopOp(node)) {
          found_pop = true;
        } else {
          // Don't modify the graph if we found an unexpected op. There may be
          // a pop hiding behind it.
          found_unexpected = true;
        }
      }
      if (!found_unexpected && !found_pop) {
        VLOG(1) << "Found stack node with no pop: " << node.DebugString();
        // Remove all pushes.
        for (int found_idx : nodes_found) {
          const NodeDef& node = graph.node(found_idx);
          if (IsStackPushOp(node)) {
            nodes_to_delete.insert(found_idx);
          }
        }
      }
    }
  }

  *optimized_graph = graph;
  if (!nodes_to_delete.empty()) {
    int last = optimized_graph->node_size() - 1;
    for (auto it = nodes_to_delete.rbegin(); it != nodes_to_delete.rend();
         ++it) {
      const int node_to_delete = *it;
      optimized_graph->mutable_node()->SwapElements(node_to_delete, last);
      --last;
    }
    optimized_graph->mutable_node()->DeleteSubrange(last + 1,
                                                    nodes_to_delete.size());
  }
  return Status::OK();
}

}  // namespace

Status LoopOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* optimized_graph) {
  Status status = RemoveStackOps(item.graph, optimized_graph);
  return status;
}

void LoopOptimizer::Feedback(Cluster* /*cluster*/, const GrapplerItem& /*item*/,
                             const GraphDef& /*optimized_graph*/,
                             double /*result*/) {
  // Nothing to do for LoopOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow

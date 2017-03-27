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

#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include <unordered_set>
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/graph_rewriter.h"

namespace tensorflow {
namespace grappler {

Status ModelPruner::Optimize(Cluster* cluster, const GrapplerItem& item,
                             GraphDef* pruned_graph) {
  GraphRewriter rewriter(item);

  std::unordered_set<const NodeDef*> nodes_to_delete;
  for (auto& node : item.graph.node()) {
    // Remove the stop gradient nodes since they serve no purpose once the graph
    // is built.
    if (node.op() != "StopGradient") {
      continue;
    }
    nodes_to_delete.insert(&node);
  }

  for (auto& node : item.graph.node()) {
    if (nodes_to_delete.find(&node) != nodes_to_delete.end()) {
      continue;
    }
    NodeDef* new_node = pruned_graph->add_node();
    *new_node = node;
    new_node->clear_input();
    rewriter.ForwardInputs(node, nodes_to_delete, new_node);
  }

  return Status::OK();
}

void ModelPruner::Feedback(Cluster* cluster, const GrapplerItem& item,
                           const GraphDef& optimize_output, double result) {
  // Nothing to do for ModelPruner.
}

}  // end namespace grappler
}  // end namespace tensorflow

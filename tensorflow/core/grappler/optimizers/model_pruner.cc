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
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/graph_rewriter.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

Status ModelPruner::Optimize(Cluster* cluster, const GrapplerItem& item,
                             GraphDef* pruned_graph) {
  GraphRewriter rewriter(item);

  std::unordered_set<string> nodes_to_preserve;
  for (const auto& node : item.fetch) {
    nodes_to_preserve.insert(NodeName(node));
  }
  for (const auto& feed : item.feed) {
    nodes_to_preserve.insert(NodeName(feed.first));
  }
  for (const auto& node : item.init_ops) {
    nodes_to_preserve.insert(NodeName(node));
  }

  std::unordered_set<const NodeDef*> nodes_to_delete;
  for (auto& node : item.graph.node()) {
    // Remove the stop gradient nodes since they serve no purpose once the graph
    // is built. Also remove Identity ops.
    if (node.op() != "StopGradient" && node.op() != "Identity") {
      continue;
    }
    // Don't remove nodes that must be preserved.
    if (nodes_to_preserve.find(node.name()) != nodes_to_preserve.end()) {
      continue;
    }
    // Don't remove nodes that drive control dependencies.
    // Don't remove nodes that are driven by control dependencies either since
    // we can't ensure (yet) that we won't increase the number of control
    // dependency edges by deleting them (for example, removing a node driven by
    // 10 control edges and driving 10 control edges would result in the
    // creation of 100 edges).
    if (!rewriter.DrivesControlDependency(node) &&
        !rewriter.IsDrivenByControlDependency(node)) {
      nodes_to_delete.insert(&node);
    }
  }

  for (auto& node : item.graph.node()) {
    NodeDef* new_node = pruned_graph->add_node();
    *new_node = node;
    new_node->clear_input();
    rewriter.ForwardInputs(node, nodes_to_delete, new_node);
  }

  VLOG(1) << "Pruned " << nodes_to_delete.size()
          << " nodes from the graph. The graph now contains "
          << pruned_graph->node_size() << " nodes.";

  *pruned_graph->mutable_library() = item.graph.library();
  *pruned_graph->mutable_versions() = item.graph.versions();

  return Status::OK();
}

void ModelPruner::Feedback(Cluster* cluster, const GrapplerItem& item,
                           const GraphDef& pruned_graph, double result) {
  // Nothing to do for ModelPruner.
}

}  // end namespace grappler
}  // end namespace tensorflow

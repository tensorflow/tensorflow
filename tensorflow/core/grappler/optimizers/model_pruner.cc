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
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/graph_rewriter.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

int NumNonControlInputs(const NodeDef& node) {
  int num_inputs = node.input_size();
  for (int i = 0; i < node.input_size(); ++i) {
    if (!node.input(i).empty() && node.input(i)[0] == '^') {
      num_inputs--;
    }
  }
  return num_inputs;
}

bool IsTrivialOp(const NodeDef& node) {
  // Remove the stop gradient nodes since they serve no purpose once the graph
  // is built. Also remove Identity ops.
  if (IsStopGradient(node) || IsIdentity(node)) {
    return true;
  }
  if (IsAddN(node) && NumNonControlInputs(node) <= 1) {
    return true;
  }

  return false;
}

Status ModelPruner::Optimize(Cluster* cluster, const GrapplerItem& item,
                             GraphDef* pruned_graph) {
  std::unordered_set<string> nodes_to_preserve = item.NodesToPreserve();

  // Prune all the nodes that won't be executed, ie all the nodes that aren't in
  // the fanin of a fetch node. If fetch nodes aren't specified, we'll assume
  // the whole graph might be executed.
  GrapplerItem runnable_item;
  if (!nodes_to_preserve.empty()) {
    std::vector<string> terminal_nodes(nodes_to_preserve.begin(),
                                       nodes_to_preserve.end());
    bool ill_formed = false;
    std::vector<const NodeDef*> keep =
        ComputeTransitiveFanin(item.graph, terminal_nodes, &ill_formed);
    if (ill_formed) {
      // Some graph edges are invalid, or some of the feeds/fetch don't exist:
      // let's be conservative and preserve the graph as is.
      return errors::InvalidArgument("Invalid input graph.");
    }
    // Try to keep the nodes ordored somewhat topologically since this helps
    // further optimizations perform better.
    for (int i = keep.size() - 1; i >= 0; --i) {
      *runnable_item.graph.add_node() = *keep[i];
    }
  } else {
    runnable_item = item;
  }

  GraphRewriter rewriter(runnable_item);

  // Check if we can further prune the graph, by removing the trivial ops.
  std::unordered_set<const NodeDef*> nodes_to_delete;
  for (auto& node : runnable_item.graph.node()) {
    if (!IsTrivialOp(node)) {
      continue;
    }

    // Don't remove nodes that must be preserved.
    if (nodes_to_preserve.find(node.name()) != nodes_to_preserve.end()) {
      continue;
    }

    // - Don't remove nodes that drive control dependencies.
    // - Don't remove nodes that are driven by control dependencies either since
    //   we can't ensure (yet) that we won't increase the number of control
    //   dependency edges by deleting them (for example, removing a node driven
    //   by 10 control edges and driving 10 control edges would result in the
    //   creation of 100 edges).
    // - Don't modify nodes that are connected to functions since that can
    //   result in inlining failures later on.
    // - Don't prune nodes that are driven by another device since these could
    //   be used to reduce cross device communication.
    // - Don't remove nodes that receive reference values, as those can be
    //   converting references to non-references. It is important to preserve
    //   these non-references since the partitioner will avoid sending
    //   non-references across partitions more than once.
    if (!rewriter.DrivesControlDependency(node) &&
        !rewriter.IsDrivenByControlDependency(node) &&
        !rewriter.IsConnectedToFunction(node) &&
        !rewriter.IsDrivenByAnotherDevice(node) &&
        !rewriter.ReceivesRefValue(node)) {
      nodes_to_delete.insert(&node);
    }
  }

  *pruned_graph->mutable_library() = item.graph.library();
  *pruned_graph->mutable_versions() = item.graph.versions();

  if (nodes_to_delete.empty()) {
    pruned_graph->mutable_node()->Swap(runnable_item.graph.mutable_node());
    return Status::OK();
  }

  for (auto& node : runnable_item.graph.node()) {
    NodeDef* new_node = pruned_graph->add_node();
    *new_node = node;
    new_node->clear_input();
    rewriter.ForwardInputs(node, nodes_to_delete, new_node);
  }

  VLOG(1) << "Pruned " << nodes_to_delete.size()
          << " nodes from the graph. The graph now contains "
          << pruned_graph->node_size() << " nodes.";

  return Status::OK();
}

void ModelPruner::Feedback(Cluster* cluster, const GrapplerItem& item,
                           const GraphDef& pruned_graph, double result) {
  // Nothing to do for ModelPruner.
}

}  // end namespace grappler
}  // end namespace tensorflow

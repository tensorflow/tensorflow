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

#include "tensorflow/core/grappler/optimizers/data/make_numa_aware.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"

namespace tensorflow {
namespace grappler {
namespace {

NodeDef MakeNumaAwareNode(const NodeDef& node, MutableGraphView* graph) {
  NodeDef numa_aware_node = node;
  graph_utils::SetUniqueGraphNodeName("make_numa_aware", graph->graph(),
                                      &numa_aware_node);
  numa_aware_node.set_op("ExperimentalNumaMapAndBatchDataset");
  return numa_aware_node;
}

}  // namespace

Status MakeNumaAware::OptimizeAndCollectStats(Cluster* cluster,
                                              const GrapplerItem& item,
                                              GraphDef* output,
                                              OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;

  for (const NodeDef& node : item.graph.node()) {
    if (node.op() != "ExperimentalMapAndBatchDataset") continue;

    auto* numa_node = graph.AddNode(MakeNumaAwareNode(node, &graph));
    TF_RETURN_IF_ERROR(graph.UpdateFanouts(node.name(), numa_node->name()));
    nodes_to_delete.insert(node.name());
    stats->num_changes++;
  }
  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(MakeNumaAware, "make_numa_aware");

}  // namespace grappler
}  // namespace tensorflow

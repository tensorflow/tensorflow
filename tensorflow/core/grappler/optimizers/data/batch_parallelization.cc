/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/batch_parallelization.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kBatchDataset[] = "BatchDatasetV2";
constexpr char kParallelBatchDataset[] = "ParallelBatchDataset";

NodeDef MakeParallelBatch(const string& name, MutableGraphView* graph) {
  // The inputs of the node to be parallelized could be changed by the
  // optimization pass, so we need to look it up in the modified graph.
  int index = graph_utils::FindGraphNodeWithName(name, *graph->graph());
  DCHECK_NE(index, -1) << "Failed to find node " << name
                       << " in the optimized graph.";
  NodeDef parallel_batch = graph->graph()->node(index);
  graph_utils::SetUniqueGraphNodeName(kParallelBatchDataset, graph->graph(),
                                      &parallel_batch);
  parallel_batch.set_op(kParallelBatchDataset);
  auto* num_parallel_calls =
      graph_utils::AddScalarConstNode(data::model::kAutotune, graph);
  string drop_remainder_name = parallel_batch.input(2);
  parallel_batch.set_input(2, num_parallel_calls->name());
  parallel_batch.add_input(drop_remainder_name);

  return parallel_batch;
}

}  // namespace

Status BatchParallelization::OptimizeAndCollectStats(Cluster* cluster,
                                                     const GrapplerItem& item,
                                                     GraphDef* output,
                                                     OptimizationStats* stats) {
  *output = item.graph;
  if (!autotune_) {
    VLOG(1) << "The optimization batch_parallelization is not applied if "
               "autotune is off.";
    return Status::OK();
  }
  MutableGraphView graph(output);

  // If the GrapplerItem is derived from a FunctionDef, we don't optimize it,
  // because we only want to enable extra batch parallelism on the main dataset
  // pipeline.
  if (graph_utils::IsItemDerivedFromFunctionDef(item, graph))
    return Status::OK();

  absl::flat_hash_set<string> nodes_to_delete;
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item.graph.library());
  auto get_batch_node = [](const NodeDef& node) -> const NodeDef* {
    if (node.op() == kBatchDataset) return &node;
    return nullptr;
  };

  for (const NodeDef& node : item.graph.node()) {
    const NodeDef* batch_node = get_batch_node(node);
    if (!batch_node) continue;

    auto* parallel_batch =
        graph.AddNode(MakeParallelBatch(batch_node->name(), &graph));
    TF_RETURN_IF_ERROR(
        graph.UpdateFanouts(batch_node->name(), parallel_batch->name()));
    nodes_to_delete.insert(batch_node->name());
    stats->num_changes++;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(BatchParallelization, "batch_parallelization");

}  // namespace grappler
}  // namespace tensorflow

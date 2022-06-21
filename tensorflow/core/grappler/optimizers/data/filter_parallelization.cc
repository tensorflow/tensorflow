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

#include "tensorflow/core/grappler/optimizers/data/filter_parallelization.h"

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

constexpr char kFilterDataset[] = "FilterDataset";
constexpr char kParallelFilterDataset[] = "ParallelFilterDataset";

NodeDef MakeParallelFilter(const string& name, MutableGraphView* graph) {
  // The inputs of the node to be parallelized could be changed by the
  // optimization pass, so we need to look it up in the modified graph.
  int index = graph_utils::FindGraphNodeWithName(name, *graph->graph());
  DCHECK_NE(index, -1) << "Failed to find node " << name
                       << " in the optimized graph.";
  NodeDef parallel_filter = graph->graph()->node(index);
  graph_utils::SetUniqueGraphNodeName(kParallelFilterDataset, graph->graph(),
                                      &parallel_filter);
  parallel_filter.set_op(kParallelFilterDataset);
  auto* num_parallel_calls = graph_utils::AddScalarConstNode(
      static_cast<int64_t>(data::model::kAutotune), graph);
  parallel_filter.add_input(num_parallel_calls->name());
  AddNodeAttr("deterministic", "true", &parallel_filter);

  return parallel_filter;
}

}  // namespace

Status FilterParallelization::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
  *output = item.graph;
  if (!autotune_) {
    VLOG(1) << "The optimization filter_parallelization is not applied if "
               "autotune is off.";
    return OkStatus();
  }
  MutableGraphView graph(output);

  // If the GrapplerItem is derived from a FunctionDef, we don't optimize it,
  // because we only want to enable extra filter parallelism on the main dataset
  // pipeline.
  if (graph_utils::IsItemDerivedFromFunctionDef(item, graph)) return OkStatus();

  absl::flat_hash_set<string> nodes_to_delete;
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item.graph.library());
  auto get_filter_node = [](const NodeDef& node) -> const NodeDef* {
    if (node.op() == kFilterDataset) return &node;
    return nullptr;
  };

  for (const NodeDef& node : item.graph.node()) {
    const NodeDef* filter_node = get_filter_node(node);
    if (!filter_node) continue;

    auto* function = function_library.Find(
        filter_node->attr().at("predicate").func().name());
    if (function_utils::IsFunctionStateful(function_library, *function, true)) {
      continue;
    }
    auto* parallel_filter =
        graph.AddNode(MakeParallelFilter(filter_node->name(), &graph));
    TF_RETURN_IF_ERROR(
        graph.UpdateFanouts(filter_node->name(), parallel_filter->name()));
    nodes_to_delete.insert(filter_node->name());
    stats->num_changes++;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return OkStatus();
}

REGISTER_GRAPH_OPTIMIZER_AS(FilterParallelization, "filter_parallelization");

}  // namespace grappler
}  // namespace tensorflow

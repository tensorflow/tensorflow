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

#include "tensorflow/core/grappler/optimizers/data/map_parallelization.h"

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kMapDataset[] = "MapDataset";
constexpr char kParallelMapDataset[] = "ParallelMapDataset";
constexpr int kAutotune = -1;

bool CanParallelize(const FunctionDef& function,
                    const FunctionLibraryDefinition& library) {
  if (!function.signature().is_stateful()) return true;

  for (const auto& node : function.node_def()) {
    const OpDef* op_def;
    TF_CHECK_OK(library.LookUpOpDef(node.op(), &op_def));
    // Assert is marked as stateful, but it does not have any state (except
    // changing io).  Similarly to CUDA, we do not give guarantee that the
    // assert operation that would fail would be the first one, so that we can
    // parallelize it.
    if (op_def->is_stateful() && op_def->name() != "Assert") return false;
  }

  return true;
}

NodeDef MakeParallelMap(const string& name, MutableGraphView* graph) {
  // The inputs of the node to be parallelized could be changed by the
  // optimization pass, so we need to look it up in the modified graph.
  int index = graph_utils::FindGraphNodeWithName(name, *graph->graph());
  DCHECK_NE(index, -1) << "Failed to find node " << name
                       << " in the optimized graph.";
  NodeDef parallel_map = graph->graph()->node(index);
  graph_utils::SetUniqueGraphNodeName(kParallelMapDataset, graph->graph(),
                                      &parallel_map);
  parallel_map.set_op(kParallelMapDataset);
  auto* num_parallel_calls = graph_utils::AddScalarConstNode(kAutotune, graph);
  parallel_map.add_input(num_parallel_calls->name());

  return parallel_map;
}

}  // namespace

Status MapParallelization::OptimizeAndCollectStats(Cluster* cluster,
                                                   const GrapplerItem& item,
                                                   GraphDef* output,
                                                   OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  std::set<string> nodes_to_delete;
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item.graph.library());
  auto get_map_node = [](const NodeDef& node) -> const NodeDef* {
    if (node.op() == kMapDataset) return &node;
    return nullptr;
  };

  for (const NodeDef& node : item.graph.node()) {
    const NodeDef* map_node = get_map_node(node);
    if (!map_node) continue;

    auto* function =
        function_library.Find(map_node->attr().at("f").func().name());
    if (!CanParallelize(*function, function_library)) continue;

    auto* parallel_map =
        graph.AddNode(MakeParallelMap(map_node->name(), &graph));
    TF_RETURN_IF_ERROR(
        graph.UpdateFanouts(map_node->name(), parallel_map->name()));
    nodes_to_delete.insert(map_node->name());
    stats->num_changes++;
  }

  graph.DeleteNodes(nodes_to_delete);
  return Status::OK();
}

void MapParallelization::Feedback(Cluster* cluster, const GrapplerItem& item,
                                  const GraphDef& optimize_output,
                                  double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(MapParallelization, "map_parallelization");

}  // namespace grappler
}  // namespace tensorflow

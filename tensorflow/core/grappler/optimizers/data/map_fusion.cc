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

#include "tensorflow/core/grappler/optimizers/data/map_fusion.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/fusion_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {

// Sets basic function parameters and copies attributes from parent and map
// node.
NodeDef MakeFusedNode(const NodeDef& parent_map_node, const NodeDef& map_node,
                      const FunctionDef& fused_function,
                      MutableGraphView* graph) {
  NodeDef fused_node;
  graph_utils::SetUniqueGraphNodeName("fused_map", graph->graph(), &fused_node);
  fused_node.set_op("MapDataset");
  fused_node.add_input(parent_map_node.input(0));

  auto attr = parent_map_node.attr().at("f");
  *attr.mutable_func()->mutable_name() = fused_function.signature().name();
  (*fused_node.mutable_attr())["f"] = std::move(attr);

  graph_utils::CopyAttribute("Targuments", parent_map_node, &fused_node);
  for (auto key : {"output_shapes", "output_types"})
    graph_utils::CopyAttribute(key, map_node, &fused_node);

  auto value_or_false = [](const AttrValue* attr) {
    if (!attr) return false;
    return attr->b();
  };

  const auto* first_parallelism =
      gtl::FindOrNull(parent_map_node.attr(), "use_inter_op_parallelism");
  const auto* second_parallelism =
      gtl::FindOrNull(map_node.attr(), "use_inter_op_parallelism");
  // Some graphs cannot execute with use_inter_op_parallelism=False, so we need
  // to set it to true if one of the ops have it set to true.
  (*fused_node.mutable_attr())["use_inter_op_parallelism"].set_b(
      value_or_false(first_parallelism) || value_or_false(second_parallelism));

  const auto* first_cardinality =
      gtl::FindOrNull(parent_map_node.attr(), "preserve_cardinality");
  const auto* second_cardinality =
      gtl::FindOrNull(map_node.attr(), "preserve_cardinality");
  (*fused_node.mutable_attr())["preserve_cardinality"].set_b(
      value_or_false(first_cardinality) && value_or_false(second_cardinality));

  return fused_node;
}

}  // namespace

Status MapFusion::Optimize(Cluster* cluster, const GrapplerItem& item,
                           GraphDef* output) {
  GraphDef sorted_old_graph = item.graph;
  TF_RETURN_IF_ERROR(TopologicalSort(&sorted_old_graph));
  *output = sorted_old_graph;

  MutableGraphView graph(output);
  std::set<string> nodes_to_delete;
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item.graph.library());

  auto get_map_node = [](const NodeDef& node) -> const NodeDef* {
    // TODO(prazek): we could also handle ParallelMapDataset and
    // MapAndBatchDataset.
    if (node.op() == "MapDataset") return &node;
    return nullptr;
  };

  auto get_fused_function = [&function_library, &output](
                                const NodeDef* parent_map_node,
                                const NodeDef* map_node) -> FunctionDef* {
    const auto& parent_fun = parent_map_node->attr().at("f");
    const FunctionDef* parent_func =
        function_library.Find(parent_fun.func().name());
    const auto& fun = map_node->attr().at("f");
    const FunctionDef* func = function_library.Find(fun.func().name());

    if (!fusion_utils::CanCompose(parent_func->signature(),
                                  func->signature())) {
      VLOG(1) << "Can't fuse two maps because the output signature of the "
                 "first map function does not match the input signature of the "
                 "second function\n";
      return nullptr;
    }
    return fusion_utils::FuseFunctions(
        *parent_func, *func, "fused_map", fusion_utils::ComposeSignature,
        fusion_utils::ComposeInput, fusion_utils::ComposeOutput,
        fusion_utils::MergeNodes, output->mutable_library());
  };

  for (const NodeDef& node : sorted_old_graph.node()) {
    const NodeDef* map_node = get_map_node(node);
    if (!map_node) continue;

    const NodeDef* parent_map_node =
        get_map_node(*graph_utils::GetInputNode(*map_node, graph));
    if (!parent_map_node) continue;

    const auto* fused_function = get_fused_function(parent_map_node, map_node);
    if (fused_function == nullptr) continue;
    const auto* fused_maps_node = graph.AddNode(
        MakeFusedNode(*parent_map_node, *map_node, *fused_function, &graph));

    graph.UpdateFanouts(map_node->name(), fused_maps_node->name());

    // TODO(prazek): we should run some optimizations on the fused map
    // functions, or make sure that optimization passes run after map
    // fusion.
    TF_RETURN_IF_ERROR(function_library.AddFunctionDef(*fused_function));

    // TODO(b/116285210): we could also remove map functions from library if
    // they are not used anymore.
    nodes_to_delete.insert(parent_map_node->name());
    nodes_to_delete.insert(map_node->name());
  }

  graph.DeleteNodes(nodes_to_delete);
  return Status::OK();
}

void MapFusion::Feedback(Cluster* cluster, const GrapplerItem& item,
                         const GraphDef& optimize_output, double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(MapFusion, "map_fusion");

}  // end namespace grappler
}  // end namespace tensorflow

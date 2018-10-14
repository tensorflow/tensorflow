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

#include "tensorflow/core/grappler/optimizers/data/map_and_filter_fusion.h"

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

NodeDef MakeFusedNode(const NodeDef& map_node,
                      const FunctionDef& fused_function,
                      MutableGraphView* graph) {
  NodeDef fused_node;
  graph_utils::SetUniqueGraphNodeName("fused_map", graph->GetGraph(),
                                      &fused_node);
  fused_node.set_op("MapDataset");
  fused_node.add_input(map_node.input(0));

  auto attr = map_node.attr().at("f");
  attr.mutable_func()->set_name(fused_function.signature().name());
  (*fused_node.mutable_attr())["f"] = std::move(attr);

  graph_utils::CopyAttribute("Targuments", map_node, &fused_node);

  for (auto key : {"output_shapes", "output_types"})
    graph_utils::CopyAttribute(key, map_node, &fused_node);

  if (const auto* attr =
          gtl::FindOrNull(map_node.attr(), "use_inter_op_parallelism"))
    (*fused_node.mutable_attr())["use_inter_op_parallelism"] = *attr;

  // Add the predicate output attributes.
  (*fused_node.mutable_attr())["output_types"]
      .mutable_list()
      ->mutable_type()
      ->Add(DT_BOOL);
  (*fused_node.mutable_attr())["output_shapes"]
      .mutable_list()
      ->mutable_shape()
      ->Add();

  return fused_node;
}

NodeDef MakeFilterByLastComponentNode(const NodeDef& fused_map_node,
                                      const NodeDef& filter_node,
                                      MutableGraphView* graph) {
  NodeDef filter_by_component;
  graph_utils::SetUniqueGraphNodeName("FilterByLastComponent",
                                      graph->GetGraph(), &filter_by_component);
  filter_by_component.set_op("FilterByLastComponentDataset");
  filter_by_component.add_input(fused_map_node.name());

  for (auto key : {"output_shapes", "output_types"}) {
    (*filter_by_component.mutable_attr())[key] = filter_node.attr().at(key);
  }
  return filter_by_component;
}

}  // namespace

Status MapAndFilterFusion::Optimize(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* output) {
  GraphDef sorted_old_graph = item.graph;
  TF_RETURN_IF_ERROR(TopologicalSort(&sorted_old_graph));
  // TODO(prazek): We might have some problems with performance if we copy
  // the whole graph too much.
  *output = sorted_old_graph;

  MutableGraphView graph(output);
  std::set<string> nodes_to_delete;
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item.graph.library());
  auto get_map_node = [](const NodeDef& node) -> const NodeDef* {
    if (node.op() == "MapDataset") return &node;
    return nullptr;
  };

  auto get_filter_node = [](const NodeDef& node) -> const NodeDef* {
    if (node.op() == "FilterDataset") return &node;
    return nullptr;
  };

  auto make_fused_function = [&function_library, &output](
                                 const NodeDef* map_node,
                                 const NodeDef* filter_node) -> FunctionDef* {
    const auto& parent_fun = map_node->attr().at("f");
    const FunctionDef* map_func =
        function_library.Find(parent_fun.func().name());
    const auto& fun = filter_node->attr().at("predicate");
    const FunctionDef* filter_func = function_library.Find(fun.func().name());
    if (!fusion_utils::CanCompose(map_func->signature(),
                                  filter_func->signature())) {
      VLOG(1) << "Can't fuse map and filter because the output signature of "
                 "the map function does not match the input signature of the "
                 "filter function\n";
      return nullptr;
    }
    return fusion_utils::FuseFunctions(
        *map_func, *filter_func, "fused_map_and_filter_function",
        fusion_utils::CombineSignature, fusion_utils::ComposeInput,
        fusion_utils::CombineOutput, fusion_utils::MergeNodes,
        output->mutable_library());
  };

  for (const NodeDef& node : sorted_old_graph.node()) {
    const NodeDef* filter_node = get_filter_node(node);
    if (!filter_node) continue;

    const NodeDef* map_node =
        get_map_node(*graph_utils::GetInputNode(*filter_node, graph));
    if (!map_node) continue;

    const auto* fused_function = make_fused_function(map_node, filter_node);
    if (fused_function == nullptr) continue;

    const auto* fused_maps =
        graph.AddNode(MakeFusedNode(*map_node, *fused_function, &graph));

    const auto* filter_by_component = graph.AddNode(
        MakeFilterByLastComponentNode(*fused_maps, *filter_node, &graph));

    graph.ReplaceInput(*filter_node, *filter_by_component);
    TF_RETURN_IF_ERROR(function_library.AddFunctionDef(*fused_function));

    // TODO(prazek): we could also remove functions from library if they are not
    // used anymore.
    nodes_to_delete.insert(map_node->name());
    nodes_to_delete.insert(filter_node->name());
  }

  graph.DeleteNodes(nodes_to_delete);
  return Status::OK();
}

void MapAndFilterFusion::Feedback(Cluster* cluster, const GrapplerItem& item,
                                  const GraphDef& optimize_output,
                                  double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(MapAndFilterFusion, "map_and_filter_fusion");

}  // end namespace grappler
}  // end namespace tensorflow

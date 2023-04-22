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

#include "absl/container/flat_hash_set.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/fusion_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/kernels/function_ops.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {

NodeDef MakeFusedNode(const NodeDef& map_node,
                      const FunctionDef& fused_function,
                      MutableGraphView* graph) {
  NodeDef fused_node;
  graph_utils::SetUniqueGraphNodeName("fused_map", graph->graph(), &fused_node);
  fused_node.set_op(map_node.op());

  // Copy over inputs.
  for (int i = 0; i < map_node.input_size(); ++i) {
    fused_node.add_input(map_node.input(i));
  }

  auto attr = map_node.attr().at("f");
  attr.mutable_func()->set_name(fused_function.signature().name());
  (*fused_node.mutable_attr())["f"] = std::move(attr);

  // Required attrs.
  for (auto key : {"Targuments", "output_shapes", "output_types"}) {
    graph_utils::CopyAttribute(key, map_node, &fused_node);
  }

  // Optional attrs.
  for (auto key :
       {"use_inter_op_parallelism", "sloppy", "preserve_cardinality"}) {
    if (gtl::FindOrNull(map_node.attr(), key)) {
      graph_utils::CopyAttribute(key, map_node, &fused_node);
    }
  }

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

NodeDef MakeFilterNode(const NodeDef& fused_map,
                       const FunctionDef& fused_map_func,
                       MutableGraphView* graph, FunctionDefLibrary* library) {
  NodeDef filter_node;
  graph_utils::SetUniqueGraphNodeName("FilterByLast", graph->graph(),
                                      &filter_node);
  filter_node.set_op("FilterDataset");
  filter_node.add_input(fused_map.name());

  for (auto key : {"output_shapes", "output_types"}) {
    graph_utils::CopyAttribute(key, fused_map, &filter_node);
  }

  AddNodeAttr("Targuments", std::vector<DataType>({}), &filter_node);

  OpDef fused_sig = fused_map_func.signature();
  FunctionDef* func = library->add_function();
  OpDef* sig = func->mutable_signature();
  sig->set_name("GetLast");
  for (const auto& arg : fused_sig.output_arg()) {
    *(sig->add_input_arg()) = arg;
  }
  OpDef::ArgDef* arg = sig->add_output_arg();
  arg->set_name("predicate_result");
  arg->set_description("predicate result computed in the fused map");
  arg->set_type(DT_BOOL);
  sig->set_description("returns the last argument");
  (*func->mutable_ret())["predicate_result"] = strings::StrCat(
      fused_sig.output_arg(fused_sig.output_arg_size() - 1).name(), ":0");

  (*filter_node.mutable_attr())["predicate"] =
      FunctionDefHelper::FunctionRef(func->signature().name()).proto;
  return filter_node;
}

NodeDef MakeMapNode(const NodeDef& updated_filter, const NodeDef& original_map,
                    const FunctionDef& fused_map_func, MutableGraphView* graph,
                    FunctionDefLibrary* library) {
  NodeDef map_node;
  graph_utils::SetUniqueGraphNodeName("DropLast", graph->graph(), &map_node);
  // We use MapDataset even if the original map was ParallelMap. Non-parallel
  // map is more performant for simple short-circuit functions like (x, y) -> x.
  map_node.set_op("MapDataset");
  map_node.add_input(updated_filter.name());

  for (auto key : {"output_shapes", "output_types"}) {
    graph_utils::CopyAttribute(key, original_map, &map_node);
  }

  AddNodeAttr("Targuments", std::vector<DataType>({}), &map_node);

  for (auto key : {"use_inter_op_parallelism", "preserve_cardinality"}) {
    if (gtl::FindOrNull(original_map.attr(), key)) {
      graph_utils::CopyAttribute(key, original_map, &map_node);
    }
  }

  OpDef fused_sig = fused_map_func.signature();
  FunctionDef* func = library->add_function();
  OpDef* sig = func->mutable_signature();
  sig->set_name("DropLast");
  for (const auto& o : fused_sig.output_arg()) {
    *(sig->add_input_arg()) = o;
  }
  for (int i = 0; i < fused_sig.output_arg_size() - 1; ++i) {
    auto arg_i = fused_sig.output_arg(i);
    *(sig->add_output_arg()) = arg_i;
    (*func->mutable_ret())[arg_i.name()] = strings::StrCat(arg_i.name(), ":0");
  }
  sig->set_description("drops the last argument");

  (*map_node.mutable_attr())["f"] =
      FunctionDefHelper::FunctionRef(func->signature().name()).proto;
  return map_node;
}

}  // namespace

Status MapAndFilterFusion::OptimizeAndCollectStats(Cluster* cluster,
                                                   const GrapplerItem& item,
                                                   GraphDef* output,
                                                   OptimizationStats* stats) {
  GraphDef sorted_old_graph = item.graph;
  TF_RETURN_IF_ERROR(TopologicalSort(&sorted_old_graph));
  // TODO(prazek): We might have some problems with performance if we copy
  // the whole graph too much.
  *output = sorted_old_graph;

  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item.graph.library());
  auto get_map_node = [](const NodeDef& node) -> const NodeDef* {
    // TODO(b/148614315): Support captured inputs.
    if ((node.op() == "MapDataset" && node.input_size() == 1) ||
        (node.op() == "ParallelMapDataset" && node.input_size() == 2)) {
      return &node;
    }
    return nullptr;
  };

  auto get_filter_node = [](const NodeDef& node) -> const NodeDef* {
    // TODO(b/148614315): Support captured inputs.
    if (node.op() == "FilterDataset" && node.input_size() == 1) return &node;
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

    const auto* new_filter_node = graph.AddNode(MakeFilterNode(
        *fused_maps, *fused_function, &graph, output->mutable_library()));

    const auto* new_map_node =
        graph.AddNode(MakeMapNode(*new_filter_node, *map_node, *fused_function,
                                  &graph, output->mutable_library()));

    TF_RETURN_IF_ERROR(
        graph.UpdateFanouts(filter_node->name(), new_map_node->name()));
    TF_RETURN_IF_ERROR(function_library.AddFunctionDef(*fused_function));

    nodes_to_delete.insert(map_node->name());
    nodes_to_delete.insert(filter_node->name());
    stats->num_changes++;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(MapAndFilterFusion, "map_and_filter_fusion");

}  // namespace grappler
}  // namespace tensorflow

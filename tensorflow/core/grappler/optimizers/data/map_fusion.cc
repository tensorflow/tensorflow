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

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
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
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kMapDatasetOp[] = "MapDataset";
constexpr char kParallelMapDatasetOp[] = "ParallelMapDatasetV2";
constexpr char kDeterministicAttr[] = "deterministic";
constexpr char kConstOp[] = "Const";
constexpr char kValueAttr[] = "value";
constexpr int kAutotuneValue = -1;

// Returns true if it is a `tf.data.AUTOTUNE` node.
bool IsAutotuneNode(const string& node_name, const MutableGraphView& graph) {
  const NodeDef* node = graph.GetNode(node_name);
  if (!node) return false;
  if (node->op() != kConstOp) return false;

  const auto* value = gtl::FindOrNull(node->attr(), kValueAttr);
  if (!value) return false;

  if (value->has_tensor()) {
    if (value->tensor().int64_val_size()) {
      return value->tensor().int64_val(0) == kAutotuneValue;
    }
  }

  return false;
}

// Returns true if both parent and child parallel map nodes have same
// `determistic` attr value.
bool SameDeterministicAttr(const NodeDef& parallel_map_node,
                           const NodeDef& parent_parallel_map_node) {
  const auto* first_deterministic_attr =
      gtl::FindOrNull(parallel_map_node.attr(), kDeterministicAttr);
  const auto* second_deterministic_attr =
      gtl::FindOrNull(parent_parallel_map_node.attr(), kDeterministicAttr);
  const bool first_deterministic_val =
      (first_deterministic_attr == nullptr) ||
      (first_deterministic_attr->s() == "true" ||
       first_deterministic_attr->s() == "default");
  const bool second_deterministic_val =
      (second_deterministic_attr == nullptr) ||
      (second_deterministic_attr->s() == "true" ||
       second_deterministic_attr->s() == "default");
  return first_deterministic_val == second_deterministic_val;
}

// Returns a name for a new node or function that fuses the inputs.
// - For nodes, this is only for debugging.
// - For functions, this additionally prevents collisions (upstream of this
// optimizer, the act of optimizing a single graph entails individually
// optimizing each function in that graph and later aggregating any new
// functions introduced during these individual optimizations into that single
// graph's collective function library).
// TODO(mpcallanan): Look at deduping names in a more generic fashion upstream.
string GetFusedName(const NodeDef& parent, const NodeDef& child) {
  return absl::StrCat("map_fusion_nodes/", parent.name(), "/", child.name());
}
string GetFusedName(const FunctionDef& parent, const FunctionDef& child) {
  return absl::StrCat("map_fusion_funcs/", parent.signature().name(), "/",
                      child.signature().name());
}

// Sets basic function parameters and copies attributes from parent and map
// node.
NodeDef MakeFusedNode(const NodeDef& parent_map_node, const NodeDef& map_node,
                      const FunctionDef& fused_function,
                      MutableGraphView* graph) {
  NodeDef fused_node;
  graph_utils::SetUniqueGraphNodeName(GetFusedName(parent_map_node, map_node),
                                      graph->graph(), &fused_node);

  if (map_node.op() == kMapDatasetOp) {
    fused_node.set_op(kMapDatasetOp);
    fused_node.add_input(parent_map_node.input(0));  // `input_dataset`
  } else if (map_node.op() == kParallelMapDatasetOp) {
    fused_node.set_op(kParallelMapDatasetOp);
    fused_node.add_input(parent_map_node.input(0));  // `input_dataset`
    fused_node.add_input(parent_map_node.input(1));  // `num_parallel_calls`
  }

  auto attr = parent_map_node.attr().at("f");
  *attr.mutable_func()->mutable_name() = fused_function.signature().name();
  (*fused_node.mutable_attr())["f"] = std::move(attr);

  graph_utils::CopyAttribute("Targuments", parent_map_node, &fused_node);
  graph_utils::CopyShapesAndTypesAttrs(map_node, &fused_node);

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

  graph_utils::MaybeSetFusedMetadata(parent_map_node, map_node, &fused_node);

  if (map_node.op() == kParallelMapDatasetOp) {
    graph_utils::CopyAttribute(kDeterministicAttr, map_node, &fused_node);
  }

  return fused_node;
}

}  // namespace

absl::Status MapFusion::OptimizeAndCollectStats(Cluster* cluster,
                                                const GrapplerItem& item,
                                                GraphDef* output,
                                                OptimizationStats* stats) {
  GraphDef sorted_old_graph = item.graph;
  TF_RETURN_IF_ERROR(TopologicalSort(&sorted_old_graph));
  *output = sorted_old_graph;

  if (!autotune_) {
    VLOG(1) << "The optimization map_fusion is not applied if "
               "autotune is off.";
    return absl::OkStatus();
  }

  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item.graph.library());

  auto get_map_node = [&graph](const NodeDef& node) -> const NodeDef* {
    // TODO(b/148614504): Support ParallelMapDataset and MapAndBatchDataset.
    // TODO(b/148614315): Support captured inputs and additionally look into a
    // Python test for control outputs per b/171265131.
    if (node.op() == kMapDatasetOp && node.input_size() == 1) return &node;
    // Only parallel map with no captured inputs (empty `other_arguments`) and
    // parallelism set to "AUTOTUNE" would be eligible for rewrite.
    if (node.op() == kParallelMapDatasetOp) {
      if (node.input_size() != 2) return nullptr;
      if (!IsAutotuneNode(node.input(1), graph)) return nullptr;
      return &node;
    }
    return nullptr;
  };

  auto make_fused_function = [&function_library, &output](
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
        *parent_func, *func, GetFusedName(*parent_func, *func),
        fusion_utils::ComposeSignature, fusion_utils::ComposeInput,
        fusion_utils::ComposeOutput, fusion_utils::MergeNodes,
        output->mutable_library());
  };

  for (const NodeDef& node : sorted_old_graph.node()) {
    const NodeDef* map_node = get_map_node(node);
    if (!map_node) continue;
    // Do not fuse ParallelMap node that uses the unbounded thread pool.
    if (map_node->attr().find("use_unbounded_threadpool") !=
            map_node->attr().end() &&
        map_node->attr().at("use_unbounded_threadpool").b()) {
      continue;
    }

    const NodeDef* parent_map_node =
        get_map_node(*graph_utils::GetInputNode(*map_node, graph));
    if (!parent_map_node) continue;
    // Do not fuse ParallelMap node that uses the unbounded thread pool.
    if (parent_map_node->attr().find("use_unbounded_threadpool") !=
            parent_map_node->attr().end() &&
        parent_map_node->attr().at("use_unbounded_threadpool").b()) {
      continue;
    }

    // TODO(b/148614504): Support fusing different types of map operations.
    if (parent_map_node->op() != map_node->op()) continue;

    // TODO(b/148614504): Support fusing parallel map operations with different
    // `deterministic` attr values.
    if (map_node->op() == kParallelMapDatasetOp) {
      if (!SameDeterministicAttr(*parent_map_node, *map_node)) continue;
    }

    const auto* fused_function = make_fused_function(parent_map_node, map_node);
    if (fused_function == nullptr) continue;

    const auto* fused_maps_node = graph.AddNode(
        MakeFusedNode(*parent_map_node, *map_node, *fused_function, &graph));

    TF_RETURN_IF_ERROR(
        graph.UpdateFanouts(map_node->name(), fused_maps_node->name()));

    TF_RETURN_IF_ERROR(function_library.AddFunctionDef(*fused_function));

    nodes_to_delete.insert(parent_map_node->name());
    nodes_to_delete.insert(map_node->name());
    stats->num_changes++;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return absl::OkStatus();
}

REGISTER_GRAPH_OPTIMIZER_AS(MapFusion, "map_fusion");

}  // namespace grappler
}  // namespace tensorflow

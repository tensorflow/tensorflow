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

#include "tensorflow/core/grappler/optimizers/data/noop_elimination.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kIdentity[] = "Identity";

bool IsTakeAll(const NodeDef& take_node, const MutableGraphView& graph) {
  if (take_node.op() != "TakeDataset") return false;

  const auto& count_node = *graph.GetNode(take_node.input(1));
  if (count_node.op() != "Const") return false;
  // We are looking only for 'take' with negative count.
  const auto& tensor = count_node.attr().at("value").tensor();
  if (tensor.int64_val_size()) return tensor.int64_val(0) < 0;
  return false;
}

bool IsConstNodeWithValue(const NodeDef& node, int value) {
  if (node.op() != "Const") return false;
  const auto& tensor = node.attr().at("value").tensor();
  if (tensor.int64_val_size()) return tensor.int64_val(0) == value;
  return value == 0;
}

bool IsSkipNone(const NodeDef& skip_node, const MutableGraphView& graph) {
  if (skip_node.op() != "SkipDataset") return false;
  // We are looking only for skip(0) nodes.
  return IsConstNodeWithValue(*graph.GetNode(skip_node.input(1)), 0);
}

bool IsRepeatOne(const NodeDef& repeat_node, const MutableGraphView& graph) {
  if (repeat_node.op() != "RepeatDataset") return false;
  // We are looking only for repeat(1) nodes.
  return IsConstNodeWithValue(*graph.GetNode(repeat_node.input(1)), 1);
}

bool IsShardOne(const NodeDef& shard_node, const MutableGraphView& graph) {
  if (shard_node.op() != "ShardDataset") return false;
  // We are looking only for shard(0) nodes.
  return IsConstNodeWithValue(*graph.GetNode(shard_node.input(1)), 1);
}

bool IsOutputIdentityOfInput(const FunctionDef& fdef, const string& output_arg,
                             const string& input_arg) {
  if (!fdef.ret().contains(output_arg)) {
    LOG(WARNING)
        << "Malformed FunctionDef: ret dict does not contain output arg key.";
    return false;
  }

  const auto& ret_val = fdef.ret().at(output_arg);
  auto input = function_utils::FunctionDefTensorDesc(ret_val);

  // Walk from output to input. If any node along the path is not an
  // Identity node, return false.
  while (function_utils::ContainsFunctionNodeWithName(input.node_name, fdef)) {
    int idx = function_utils::FindFunctionNodeWithName(input.node_name, fdef);

    const NodeDef& node = fdef.node_def(idx);
    if (node.op() != kIdentity) {
      return false;
    }

    input = function_utils::FunctionDefTensorDesc(node.input(0));
  }

  // If we get here, input is not a node. Check that it matches the correct
  // input arg name.
  return input.node_name == input_arg;
}

bool IsMapIdentity(const NodeDef& map_node, const MutableGraphView& graph,
                   const FunctionLibraryDefinition& function_library) {
  if (map_node.op() != "MapDataset" && map_node.op() != "ParallelMapDataset" &&
      map_node.op() != "ParallelMapDatasetV2") {
    return false;
  }

  // We are looking only for map(lambda *x: x) nodes.

  // Don't eliminate map nodes with captured arguments.
  if (map_node.attr().at("Targuments").list().type_size() != 0) return false;

  const FunctionDef* fdef =
      function_library.Find(map_node.attr().at("f").func().name());

  // Don't eliminate map nodes with stateful functions.
  if (function_utils::IsFunctionStateful(function_library, *fdef)) {
    return false;
  }

  const auto& sig = fdef->signature();
  if (sig.input_arg_size() != sig.output_arg_size()) return false;

  // For each output, check that it maps to input i
  for (int i = 0; i < sig.input_arg_size(); ++i) {
    if (!IsOutputIdentityOfInput(*fdef, sig.output_arg(i).name(),
                                 sig.input_arg(i).name())) {
      return false;
    }
  }
  return true;
}

bool IsNoOp(const NodeDef& node, const MutableGraphView& graph,
            const FunctionLibraryDefinition& function_library) {
  return IsTakeAll(node, graph) || IsSkipNone(node, graph) ||
         IsRepeatOne(node, graph) || IsShardOne(node, graph) ||
         IsMapIdentity(node, graph, function_library);
}

}  // namespace

absl::Status NoOpElimination::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             graph.graph()->library());
  for (const NodeDef& node : item.graph.node()) {
    if (!IsNoOp(node, graph, function_library)) continue;

    NodeDef* const parent = graph_utils::GetInputNode(node, graph);
    TF_RETURN_IF_ERROR(graph.UpdateFanouts(node.name(), parent->name()));

    nodes_to_delete.insert(node.name());
    stats->num_changes++;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return absl::OkStatus();
}

REGISTER_GRAPH_OPTIMIZER_AS(NoOpElimination, "noop_elimination");

}  // namespace grappler
}  // namespace tensorflow

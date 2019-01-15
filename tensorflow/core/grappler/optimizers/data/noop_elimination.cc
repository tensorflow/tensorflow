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

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {

bool IsTakeAll(const NodeDef& take_node, const MutableGraphView& graph) {
  if (take_node.op() != "TakeDataset") return false;

  const auto& count_node = *graph.GetNode(take_node.input(1));
  if (count_node.op() != "Const") return false;
  // We are looking only for 'take' with negative count.
  return count_node.attr().at("value").tensor().int64_val(0) < 0;
}

bool IsConstNodeWithValue(const NodeDef& node, int value) {
  if (node.op() != "Const") return false;
  return node.attr().at("value").tensor().int64_val(0) == value;
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

bool IsPrefetchZero(const NodeDef& prefetch_node,
                    const MutableGraphView& graph) {
  if (prefetch_node.op() != "PrefetchDataset") return false;
  // We are looking only for prefetch(0) nodes.
  return IsConstNodeWithValue(*graph.GetNode(prefetch_node.input(1)), 0);
}

bool IsNoOp(const NodeDef& node, const MutableGraphView& graph) {
  return IsTakeAll(node, graph) || IsSkipNone(node, graph) ||
         IsRepeatOne(node, graph) || IsPrefetchZero(node, graph);
}

}  // namespace

Status NoOpElimination::OptimizeAndCollectStats(Cluster* cluster,
                                                const GrapplerItem& item,
                                                GraphDef* output,
                                                OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  std::set<string> nodes_to_delete;
  for (const NodeDef& node : item.graph.node()) {
    if (!IsNoOp(node, graph)) continue;

    NodeDef* const parent = graph_utils::GetInputNode(node, graph);
    TF_RETURN_IF_ERROR(graph.UpdateFanouts(node.name(), parent->name()));

    nodes_to_delete.insert(node.name());
    stats->num_changes++;
  }

  graph.DeleteNodes(nodes_to_delete);
  return Status::OK();
}

void NoOpElimination::Feedback(Cluster* cluster, const GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(NoOpElimination, "noop_elimination");

}  // namespace grappler
}  // namespace tensorflow

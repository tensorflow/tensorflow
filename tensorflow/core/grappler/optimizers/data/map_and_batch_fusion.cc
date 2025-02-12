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

#include "tensorflow/core/grappler/optimizers/data/map_and_batch_fusion.h"

#include "absl/container/flat_hash_set.h"
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

constexpr char kFusedOpName[] = "MapAndBatchDataset";
constexpr char kParallelMap[] = "ParallelMapDataset";
constexpr char kParallelMapV2[] = "ParallelMapDatasetV2";

bool IsParallelMap(const NodeDef& node) {
  return node.op() == kParallelMap || node.op() == kParallelMapV2;
}

NodeDef MakeMapAndBatchNode(const NodeDef& map_node, const NodeDef& batch_node,
                            MutableGraphView* graph) {
  NodeDef new_node;
  new_node.set_op(kFusedOpName);
  graph_utils::SetUniqueGraphNodeName(kFusedOpName, graph->graph(), &new_node);

  // Set the `input` input argument.
  new_node.add_input(map_node.input(0));

  // Set the `other_arguments` input arguments.
  int num_other_args;
  if (IsParallelMap(map_node)) {
    num_other_args = map_node.input_size() - 2;
  } else {
    num_other_args = map_node.input_size() - 1;
  }
  for (int i = 0; i < num_other_args; i++) {
    new_node.add_input(map_node.input(i + 1));
  }

  // Set the `batch_size` input argument.
  new_node.add_input(batch_node.input(1));

  // Set the `num_parallel_calls` input argument.
  if (map_node.op() == kParallelMap) {
    // The type of the `num_parallel_calls` argument in ParallelMapDataset
    // and MapAndBatchDataset is different (int32 and int64 respectively)
    // so we cannot reuse the same Const node and thus create a new one.
    NodeDef* v = graph->GetNode(map_node.input(map_node.input_size() - 1));
    NodeDef* tmp = graph_utils::AddScalarConstNode<int64_t>(
        v->attr().at("value").tensor().int_val(0), graph);
    new_node.add_input(tmp->name());
  } else if (map_node.op() == kParallelMapV2) {
    new_node.add_input(map_node.input(map_node.input_size() - 1));
  } else {
    NodeDef* tmp = graph_utils::AddScalarConstNode<int64_t>(1, graph);
    new_node.add_input(tmp->name());
  }

  // Set the `drop_remainder` input argument.
  if (batch_node.op() == "BatchDatasetV2") {
    new_node.add_input(batch_node.input(2));
  } else {
    NodeDef* tmp = graph_utils::AddScalarConstNode<bool>(false, graph);
    new_node.add_input(tmp->name());
  }

  // Required attributes.
  for (auto key : {"f", "Targuments"}) {
    graph_utils::CopyAttribute(key, map_node, &new_node);
  }
  graph_utils::CopyShapesAndTypesAttrs(batch_node, &new_node);

  // Optional attributes.
  // TODO(jsimsa): Support `use_inter_op_parallelism` and `sloppy`.
  for (auto key : {"preserve_cardinality"}) {
    if (gtl::FindOrNull(map_node.attr(), key)) {
      graph_utils::CopyAttribute(key, map_node, &new_node);
    }
  }
  graph_utils::MaybeSetFusedMetadata(map_node, batch_node, &new_node);
  return new_node;
}

}  // namespace

absl::Status MapAndBatchFusion::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;
  for (const NodeDef& node : item.graph.node()) {
    if (node.op() != "BatchDataset" && node.op() != "BatchDatasetV2") {
      continue;
    }

    // Use a more descriptive variable name now that we know the node type.
    const NodeDef& batch_node = node;
    NodeDef* node2 = graph_utils::GetInputNode(batch_node, graph);

    if (node2->op() != "MapDataset" && !IsParallelMap(*node2)) {
      continue;
    }
    // Do not fuse ParallelMap node that uses the unbounded thread pool.
    if (node2->attr().find("use_unbounded_threadpool") != node2->attr().end() &&
        node2->attr().at("use_unbounded_threadpool").b()) {
      continue;
    }
    // Use a more descriptive variable name now that we know the node type.
    NodeDef* map_node = node2;

    auto* new_node =
        graph.AddNode(MakeMapAndBatchNode(*map_node, batch_node, &graph));
    TF_RETURN_IF_ERROR(
        graph.UpdateFanouts(batch_node.name(), new_node->name()));

    // Mark the `Map` and `Batch` nodes for removal.
    nodes_to_delete.insert(map_node->name());
    nodes_to_delete.insert(batch_node.name());
    stats->num_changes++;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return absl::OkStatus();
}

REGISTER_GRAPH_OPTIMIZER_AS(MapAndBatchFusion, "map_and_batch_fusion");

}  // namespace grappler
}  // namespace tensorflow

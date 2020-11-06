/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/disable_intra_op_parallelism.h"

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

constexpr char kMaxIntraOpParallelismDataset[] = "MaxIntraOpParallelismDataset";
constexpr char kModelDataset[] = "ModelDataset";

constexpr std::array<const char*, 2> kMaxIntraOpParallelismDatasetOps = {
    "MaxIntraOpParallelismDataset",
    "ExperimentalMaxIntraOpParallelismDataset",
};

}  // namespace

Status DisableIntraOpParallelism::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);

  // If the GrapplerItem is derived from a FunctionDef, we don't optimize it,
  // because we only want to disable intra op parallelism on the main dataset
  // pipeline.
  if (graph_utils::IsItemDerivedFromFunctionDef(item, graph))
    return Status::OK();

  if (item.fetch.size() != 1) {
    return errors::InvalidArgument(
        "Expected only one fetch node but there were ", item.fetch.size(), ": ",
        absl::StrJoin(item.fetch, ", "));
  }

  for (const NodeDef& node : item.graph.node()) {
    for (const auto& target_dataset_op : kMaxIntraOpParallelismDatasetOps) {
      if (node.op() == target_dataset_op) {
        // If parallelism is set by the user, we keep the user setting instead
        // of disabling it.
        return Status::OK();
      }
    }
  }

  NodeDef* sink_node = graph.GetNode(item.fetch.at(0));
  NodeDef* last_node = graph_utils::GetInputNode(*sink_node, graph);
  // If the pipeline is autotuned (ModelDataset exists as the last dataset in
  // the pipeline), we insert MaxIntraOpParallelismDataset before ModelDataset.
  // If the pipeline is not autotuned (ModelDataset doesn't exist), we insert
  // MaxIntraOpParallelismDataset as the last dataset in the pipeline.
  //
  // In general, if exists, ModelDataset should be the last dataset in the
  // pipeline.
  if (last_node->op() == kModelDataset) {
    last_node = graph_utils::GetInputNode(*last_node, graph);
  }

  // Add a const node with value 1
  NodeDef* max_parallelism_value =
      graph_utils::AddScalarConstNode(int64{1}, &graph);

  NodeDef insert_node;
  graph_utils::SetUniqueGraphNodeName("intra_op_parallelism", graph.graph(),
                                      &insert_node);
  insert_node.set_op(kMaxIntraOpParallelismDataset);

  // `input_dataset` input
  *insert_node.mutable_input()->Add() = last_node->name();
  // `max_intra_op_parallelism` input
  *insert_node.mutable_input()->Add() = max_parallelism_value->name();

  // Set `output_types` and `output_shapes` attributes by copying the relevant
  // attrs from the input node. If we fail to set the attributes, we abort the
  // rewrite.
  for (auto attr : {"output_shapes", "output_types"}) {
    if (last_node->attr().find(attr) != last_node->attr().end()) {
      graph_utils::CopyAttribute(attr, *last_node, &insert_node);
    } else {
      return Status::OK();
    }
  }

  auto* added_node = graph.AddNode(std::move(insert_node));
  TF_RETURN_IF_ERROR(
      graph.UpdateFanouts(last_node->name(), added_node->name()));

  stats->num_changes++;
  return Status::OK();
}

void DisableIntraOpParallelism::Feedback(Cluster* cluster,
                                         const GrapplerItem& item,
                                         const GraphDef& optimize_output,
                                         double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(DisableIntraOpParallelism,
                            "disable_intra_op_parallelism");

}  // namespace grappler
}  // namespace tensorflow

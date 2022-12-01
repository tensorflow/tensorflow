/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/use_private_thread_pool.h"

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

constexpr char kPrivateThreadPoolDataset[] = "PrivateThreadPoolDataset";
constexpr char kModelDataset[] = "ModelDataset";

}  // namespace

Status UsePrivateThreadPool::OptimizeAndCollectStats(Cluster* cluster,
                                                     const GrapplerItem& item,
                                                     GraphDef* output,
                                                     OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);

  // If the GrapplerItem is derived from a FunctionDef, we don't optimize it.
  if (graph_utils::IsItemDerivedFromFunctionDef(item, graph)) return OkStatus();

  if (item.fetch.size() != 1) {
    return errors::InvalidArgument(
        "Expected only one fetch node but there were ", item.fetch.size(), ": ",
        absl::StrJoin(item.fetch, ", "));
  }

  for (const NodeDef& node : item.graph.node()) {
    if (node.op() == kPrivateThreadPoolDataset) {
      // If private thread pool is set by the user, we keep the user setting
      // instead of rewriting it.
      return OkStatus();
    }
  }

  NodeDef* sink_node = graph.GetNode(item.fetch.at(0));
  NodeDef* last_node = graph_utils::GetInputNode(*sink_node, graph);
  // If the pipeline is autotuned (ModelDataset exists as the last dataset in
  // the pipeline), we insert PrivateThreadPoolDataset before ModelDataset.
  // If the pipeline is not autotuned (ModelDataset doesn't exist), we insert
  // PrivateThreadPoolDataset as the last dataset in the pipeline.
  //
  // In general, if exists, ModelDataset should be the last dataset in the
  // pipeline.
  if (last_node->op() == kModelDataset) {
    last_node = graph_utils::GetInputNode(*last_node, graph);
  }

  // Add a const node with value 0 to indicate it is not set by users.
  NodeDef* num_threads_value =
      graph_utils::AddScalarConstNode(int64_t{0}, &graph);

  NodeDef insert_node;
  graph_utils::SetUniqueGraphNodeName("private_thread_pool", graph.graph(),
                                      &insert_node);
  insert_node.set_op(kPrivateThreadPoolDataset);

  // `input_dataset` input
  *insert_node.mutable_input()->Add() = last_node->name();
  // `num_threads` input
  *insert_node.mutable_input()->Add() = num_threads_value->name();

  // Set `output_types` and `output_shapes` attributes by copying the relevant
  // attrs from the input node. If we fail to set the attributes, we abort the
  // rewrite.
  if (!graph_utils::CopyShapesAndTypesAttrs(*last_node, &insert_node))
    return OkStatus();

  auto* added_node = graph.AddNode(std::move(insert_node));
  TF_RETURN_IF_ERROR(
      graph.UpdateFanouts(last_node->name(), added_node->name()));

  stats->num_changes++;
  return OkStatus();
}

REGISTER_GRAPH_OPTIMIZER_AS(UsePrivateThreadPool, "use_private_thread_pool");

}  // namespace grappler
}  // namespace tensorflow

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

#include "tensorflow/core/grappler/optimizers/data/inject_prefetch.h"

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/model.h"
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

constexpr char kPrefetchDataset[] = "PrefetchDataset";
constexpr std::array<const char*, 5> kAsyncTransforms = {
    "MapAndBatchDataset", "ParallelBatchDataset", "ParallelInterleaveDataset",
    "ParallelMapDataset", "PrefetchDataset"};
constexpr std::array<const char*, 8> kDatasetsToSkip = {
    "AssertNextDataset",
    "ExperimentalAssertNextDataset",
    "IgnoreErrorsDataset",
    "OptionsDataset",
    "ModelDataset",
    "OptimizeDataset",
    "MaxIntraOpParallelismDataset",
    "PrivateThreadPoolDataset",
};

// This function returns false if the last dataset after skipping all the
// non-user defined datasets, such as OptionsDataset, is a PrefetchDataset; true
// otherwise.
bool ShouldInjectPrefetch(const NodeDef* last_node,
                          const MutableGraphView& graph) {
  // Skip all datasets that could be chained by tf.data to the user defined
  // pipeline because of optimization, etc.
  while (last_node != nullptr &&
         absl::c_any_of(kDatasetsToSkip, [last_node](const char* dataset) {
           return data::MatchesAnyVersion(dataset, last_node->op());
         })) {
    last_node = graph_utils::GetInputNode(*last_node, graph);
  }
  if (last_node == nullptr) {
    VLOG(1) << "The optimization inject_prefetch is not applied because graph "
               "rewrite failed to find a dataset node.";
    return false;
  }
  if (absl::c_any_of(kAsyncTransforms, [last_node](const char* dataset) {
        return data::MatchesAnyVersion(dataset, last_node->op());
      })) {
    VLOG(1) << "The optimization inject_prefetch is not applied because the "
               "last transformation of the input pipeline is an asynchronous "
               "transformation: "
            << last_node->op();
    return false;
  }
  return true;
}

}  // namespace

Status InjectPrefetch::OptimizeAndCollectStats(Cluster* cluster,
                                               const GrapplerItem& item,
                                               GraphDef* output,
                                               OptimizationStats* stats) {
  *output = item.graph;
  if (!autotune_) {
    VLOG(1) << "The optimization inject_prefetch is not applied if autotune is "
               "off.";
    return OkStatus();
  }
  MutableGraphView graph(output);

  // If the GrapplerItem is derived from a FunctionDef, we don't optimize it.
  if (graph_utils::IsItemDerivedFromFunctionDef(item, graph)) {
    return OkStatus();
  }

  if (item.fetch.size() != 1) {
    return errors::InvalidArgument(
        "Expected only one fetch node but there were ", item.fetch.size(), ": ",
        absl::StrJoin(item.fetch, ", "));
  }

  NodeDef* sink_node = graph.GetNode(item.fetch.at(0));
  NodeDef* last_node = graph_utils::GetInputNode(*sink_node, graph);
  if (!ShouldInjectPrefetch(last_node, graph)) {
    return OkStatus();
  }

  // Insert `prefetch(AUTOTUNE)` after the last node.
  NodeDef prefetch_node;
  graph_utils::SetUniqueGraphNodeName(
      strings::StrCat("inject/prefetch_", last_node->name()), graph.graph(),
      &prefetch_node);
  prefetch_node.set_op(kPrefetchDataset);
  // `input_dataset` input
  *prefetch_node.mutable_input()->Add() = last_node->name();
  // `buffer_size` input
  NodeDef* autotune_value =
      graph_utils::AddScalarConstNode(data::model::kAutotune, &graph);
  *prefetch_node.mutable_input()->Add() = autotune_value->name();

  // Set `output_types` and `output_shapes` attributes by copying the relevant
  // attrs from the input node. If we fail to set the attributes, we abort the
  // rewrite.
  if (!graph_utils::CopyShapesAndTypesAttrs(*last_node, &prefetch_node))
    return OkStatus();

  TF_RETURN_IF_ERROR(
      graph_utils::SetMetadataName(prefetch_node.name(), &prefetch_node));

  auto* added_node = graph.AddNode(std::move(prefetch_node));
  TF_RETURN_IF_ERROR(
      graph.UpdateFanouts(last_node->name(), added_node->name()));

  stats->num_changes++;
  return OkStatus();
}

REGISTER_GRAPH_OPTIMIZER_AS(InjectPrefetch, "inject_prefetch");

}  // namespace grappler
}  // namespace tensorflow

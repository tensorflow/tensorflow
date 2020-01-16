/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

constexpr char kLegacyAutotune[] = "legacy_autotune";
constexpr char kPrefetchDataset[] = "PrefetchDataset";

constexpr std::array<const char*, 4> kAsyncDatasetOps = {
    "ExperimentalMapAndBatchDataset",
    "ParallelMapDataset",
    "ParallelInterleaveDatasetV2",
    "MapAndBatchDataset",
};

}  // namespace

Status InjectPrefetch::OptimizeAndCollectStats(Cluster* cluster,
                                               const GrapplerItem& item,
                                               GraphDef* output,
                                               OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);

  std::vector<const NodeDef*> async_datasets;
  for (const NodeDef& node : item.graph.node()) {
    for (const auto& async_dataset_op : kAsyncDatasetOps) {
      if (node.op() == async_dataset_op) {
        async_datasets.push_back(&node);
        break;
      }
    }
  }

  if (async_datasets.empty()) return Status::OK();

  // Add a const node with value kAutotune
  NodeDef* autotune_value =
      graph_utils::AddScalarConstNode(data::model::kAutotune, &graph);

  for (const NodeDef* async_dataset_node : async_datasets) {
    NodeDef prefetch_node;
    graph_utils::SetUniqueGraphNodeName(
        strings::StrCat("inject/prefetch_", async_dataset_node->name()),
        graph.graph(), &prefetch_node);
    prefetch_node.set_op("PrefetchDataset");
    // `input_dataset` input
    *prefetch_node.mutable_input()->Add() = async_dataset_node->name();
    // `buffer_size` input
    *prefetch_node.mutable_input()->Add() = autotune_value->name();

    for (const auto& attr_name : {"output_types", "output_shapes"}) {
      graph_utils::CopyAttribute(attr_name, *async_dataset_node,
                                 &prefetch_node);
    }

    auto* added_node = graph.AddNode(std::move(prefetch_node));
    TF_RETURN_IF_ERROR(
        graph.UpdateFanouts(async_dataset_node->name(), added_node->name()));
  }

  for (NodeDef& node : *output->mutable_node()) {
    if (node.op() == kPrefetchDataset) {
      (*node.mutable_attr())[kLegacyAutotune].set_b(false);
      stats->num_changes++;
    }
  }

  return Status::OK();
}

void InjectPrefetch::Feedback(Cluster* cluster, const GrapplerItem& item,
                              const GraphDef& optimize_output, double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(InjectPrefetch, "inject_prefetch");

}  // namespace grappler
}  // namespace tensorflow

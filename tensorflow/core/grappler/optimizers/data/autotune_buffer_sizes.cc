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

#include "tensorflow/core/grappler/optimizers/data/autotune_buffer_sizes.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
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

constexpr char kBufferSizeMin[] = "buffer_size_min";
constexpr char kPrefetchDataset[] = "PrefetchDataset";

constexpr std::array<const char*, 8> kAsyncDatasetOps = {
    "ExperimentalMapAndBatchDataset",
    "MapAndBatchDataset",
    "ParallelBatchDataset",
    "ParallelInterleaveDatasetV2",
    "ParallelInterleaveDatasetV3",
    "ParallelInterleaveDatasetV4",
    "ParallelMapDataset",
    "ParallelMapDatasetV2",
};

}  // namespace

absl::Status AutotuneBufferSizes::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
  *output = item.graph;
  if (!autotune_) {
    VLOG(1) << "The optimization autotune_buffer_sizes is not applied if "
               "autotune is off.";
    return absl::OkStatus();
  }
  MutableGraphView graph(output);

  // Add a const node with value kAutotune.
  NodeDef* autotune_value =
      graph_utils::AddScalarConstNode(data::model::kAutotune, &graph);

  absl::flat_hash_set<string> already_prefetched;

  // 1) Collect about all existing `PrefetchDataset` nodes, replacing
  // `prefetch(N)` with `prefetch(AUTOTUNE, buffer_size_min=N)` for all N !=-1.
  for (NodeDef& node : *output->mutable_node()) {
    if (node.op() == kPrefetchDataset) {
      NodeDef* buffer_size_node = graph.GetNode(node.input(1));
      // We only consider to rewrite if `buffer_size` is constant.
      if (buffer_size_node->op() == "Const") {
        int64_t initial_buffer_size =
            buffer_size_node->attr().at("value").tensor().int64_val(0);
        if (initial_buffer_size != data::model::kAutotune) {
          TF_RETURN_IF_ERROR(graph.UpdateFanin(node.name(),
                                               {buffer_size_node->name(), 0},
                                               {autotune_value->name(), 0}));
          node.mutable_attr()->at(kBufferSizeMin).set_i(initial_buffer_size);
          stats->num_changes++;
        }
      } else {
        return absl::FailedPreconditionError(
            "The autotune_buffer_sizes rewrite does not currently support "
            "non-constant buffer_size input.");
      }
      NodeDef* prefetched_node = graph_utils::GetInputNode(node, graph);
      if (prefetched_node) {
        already_prefetched.insert(prefetched_node->name());
      }
    }
  }

  std::vector<const NodeDef*> async_datasets;
  // 2) Insert `prefetch(AUTOTUNE)` after all asynchronous transformations that
  // are not followed by a `prefetch` yet.
  for (const NodeDef& node : item.graph.node()) {
    if (already_prefetched.find(node.name()) != already_prefetched.end()) {
      continue;
    }
    for (const auto& async_dataset_op : kAsyncDatasetOps) {
      if (node.op() == async_dataset_op) {
        async_datasets.push_back(&node);
        stats->num_changes++;
        break;
      }
    }
  }

  if (async_datasets.empty()) return absl::OkStatus();

  for (const NodeDef* async_dataset_node : async_datasets) {
    NodeDef prefetch_node;
    graph_utils::SetUniqueGraphNodeName(
        strings::StrCat("inject/prefetch_", async_dataset_node->name()),
        graph.graph(), &prefetch_node);
    prefetch_node.set_op(kPrefetchDataset);
    // `input_dataset` input
    *prefetch_node.mutable_input()->Add() = async_dataset_node->name();
    // `buffer_size` input
    *prefetch_node.mutable_input()->Add() = autotune_value->name();

    graph_utils::CopyShapesAndTypesAttrs(*async_dataset_node, &prefetch_node);

    auto* added_node = graph.AddNode(std::move(prefetch_node));
    TF_RETURN_IF_ERROR(
        graph.UpdateFanouts(async_dataset_node->name(), added_node->name()));
  }

  return absl::OkStatus();
}

REGISTER_GRAPH_OPTIMIZER_AS(AutotuneBufferSizes, "autotune_buffer_sizes");

}  // namespace grappler
}  // namespace tensorflow

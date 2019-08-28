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

#include "tensorflow/core/grappler/optimizers/data/slack.h"

#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
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

constexpr char kRetValOp[] = "_Retval";
constexpr char kPrefetchDatasetOp[] = "PrefetchDataset";

template <std::size_t SIZE>
bool IsDatasetNodeOfType(const NodeDef& node,
                         const std::array<const char*, SIZE>& arr) {
  for (const auto& dataset_op_name : arr) {
    if (node.op() == dataset_op_name) return true;
  }
  return false;
}

// We don't pass through "Batch*" ops and nested dataset ops (FlatMap, etc)
// because the correct slack_period cannot be determined directly in those
// cases.
constexpr std::array<const char*, 2> kMultipleInputsDatasetOps = {
    "ZipDataset", "ConcatenateDataset"};

constexpr std::array<const char*, 21> kPassThroughOps = {
    "CacheDataset",
    "CacheDatasetV2",
    "ExperimentalMaxIntraOpParallelismDataset",
    "ExperimentalPrivateThreadPoolDataset",
    "FilterDataset",
    "Identity",
    "MapDataset",
    "MaxIntraOpParallelismDataset",
    "ModelDataset",
    "OptimizeDataset",
    "ParallelMapDataset",
    "PrivateThreadPoolDataset",
    "ReduceDataset",
    "RepeatDataset",
    "ShardDataset",
    "ShuffleAndRepeatDataset",
    "ShuffleDataset",
    "ShuffleDatasetV2",
    "SkipDataset",
    "TakeDataset",
    "WindowDataset",
};

}  // namespace

Status Slack::RecursivelyHandleOp(const MutableGraphView& graph,
                                  NodeDef* dataset_node) {
  if (dataset_node->op() == kPrefetchDatasetOp) {
    if (HasNodeAttr(*dataset_node, "slack_period")) {
      (*dataset_node->mutable_attr())["slack_period"].set_i(slack_period_);
    } else {
      AddNodeAttr("slack_period", slack_period_, dataset_node);
    }
    return Status::OK();
  }
  if (IsDatasetNodeOfType(*dataset_node, kPassThroughOps)) {
    NodeDef* input_node = graph_utils::GetInputNode(*dataset_node, graph, 0);
    return RecursivelyHandleOp(graph, input_node);
  }
  if (IsDatasetNodeOfType(*dataset_node, kMultipleInputsDatasetOps)) {
    // For all multiple input datasets, all inputs are datasets themselves
    for (int i = 0; i < dataset_node->input_size(); ++i) {
      NodeDef* input_node = graph_utils::GetInputNode(*dataset_node, graph, i);
      TF_RETURN_IF_ERROR(RecursivelyHandleOp(graph, input_node));
    }
    return Status::OK();
  }

  return errors::InvalidArgument(
      "Encountered unsupported op \"", dataset_node->op(),
      "\" when rewriting the input pipeline graph to use slack in its "
      "final prefetch transformation.");
}

Status Slack::OptimizeAndCollectStats(Cluster* cluster,
                                      const GrapplerItem& item,
                                      GraphDef* output,
                                      OptimizationStats* stats) {
  if (slack_period_ < 1)
    return errors::InvalidArgument("Invalid `slack_period` parameter: ",
                                   slack_period_);

  *output = item.graph;
  MutableGraphView graph(output);
  for (const auto& fetch_name : item.fetch) {
    // If the GrapplerItem is derived from a FunctionDef, we don't optimize it,
    // because we only want to add slack to the prefetch on the main dataset
    // pipeline.
    auto fetch = graph.GetNode(fetch_name);
    if (fetch == nullptr || fetch->op() == kRetValOp) {
      // Heuristic: If the fetch nodes are Retval ops, this item is from a
      // function.
      return Status::OK();
    }
  }
  if (item.fetch.size() != 1) {
    return errors::InvalidArgument(
        "Expected only one fetch node but there were ", item.fetch.size(), ": ",
        absl::StrJoin(item.fetch, ", "));
  }
  // Walks the input pipeline backwards from the fetch node to find the last
  // PrefetchDataset node in the pipeline.
  NodeDef* dataset_node = graph.GetNode(item.fetch.at(0));
  return RecursivelyHandleOp(graph, dataset_node);
}

void Slack::Feedback(Cluster* cluster, const GrapplerItem& item,
                     const GraphDef& optimize_output, double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(Slack, "slack");

}  // namespace grappler
}  // namespace tensorflow

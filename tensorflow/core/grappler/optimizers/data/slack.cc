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

}  // namespace

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
  // Walk the input pipeline backwards from the fetch node to find the last
  // PrefetchDataset node in the pipeline.
  // TODO(rachelim): This doesn't do the right thing when the "final" prefetch
  // is nested under an interleave or flat_map. Make this work, similar to
  // `auto_shard.cc` and `rebatch.cc`.
  NodeDef* dataset_node = graph.GetNode(item.fetch.at(0));
  while (true) {
    if (dataset_node->op() == "PrefetchDataset") {
      if (HasNodeAttr(*dataset_node, "slack_period")) {
        (*dataset_node->mutable_attr())["slack_period"].set_i(slack_period_);
      } else {
        AddNodeAttr("slack_period", slack_period_, dataset_node);
      }
      return Status::OK();
    }
    if (dataset_node->op() == "Identity" ||
        (absl::EndsWith(dataset_node->op(), "Dataset") &&
         dataset_node->input_size() > 0)) {
      dataset_node = graph_utils::GetInputNode(*dataset_node, graph);
    } else {
      break;
    }
  }
  return Status::OK();
}

void Slack::Feedback(Cluster* cluster, const GrapplerItem& item,
                     const GraphDef& optimize_output, double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(Slack, "slack");

}  // namespace grappler
}  // namespace tensorflow

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

#include "tensorflow/core/grappler/optimizers/data/hoist_data_discarding_ops.h"

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

constexpr std::array<const char*, 3> kDataDiscarding = {
    "ShardDataset", "SkipDataset", "TakeDataset",
};

constexpr std::array<const char*, 6> kCardinalityPreserving = {
    "CacheDataset", "CacheDatasetV2", "PrefetchDataset",
    "MapDataset", "ParallelMapDataset", "ParallelMapDatasetV2",
};

bool IsDataDiscarding(const NodeDef& node) {
  for (const auto& data_discarding_op : kDataDiscarding) {
    if (node.op() == data_discarding_op) {
      return true;
    }
  }
  return false;
}

bool IsCardinalityPreserving(const NodeDef& node) {
  for (const auto& cardinality_preserving_op : kCardinalityPreserving) {
    if (node.op() == cardinality_preserving_op) {
      return true;
    }
  }
  return false;
}

}  // namepsace

Status HoistDataDiscardingOps::OptimizeAndCollectStats(Cluster* cluster,
                                                       const GrapplerItem& item,
                                                       GraphDef* output,
                                                       OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  bool updated;
  do {
    updated = false;
    for (NodeDef node : graph.graph()->node()) {
      if (IsDataDiscarding(node)) {
        NodeDef* start = &node;
        NodeDef* start_parent = graph_utils::GetInputNode(*start, graph);
        while (IsCardinalityPreserving(*start_parent) &&
               NumOutputs(*start_parent, graph.graph()) == 1) {
          start = start_parent;
          start_parent = graph_utils::GetInputNode(*start, graph);
        }
        // no cardinality preserving op with indegree 1.
        if (start->name() == node.name()) {
          continue;
        }
        NodeDef hoisted_node = node;
        if (!absl::StartsWith(node.name(), "hoist_data_dsicarding_op/")) {
          graph_utils::SetUniqueGraphNodeName(
            strings::StrCat("hoist_data_discarding_ops/", node.name()),
            graph.graph(), &hoisted_node
          );
        }
        for (const auto& attr_name : {"output_types", "output_shapes"}) {
          graph_utils::CopyAttribute(attr_name, *start_parent,
                                     &hoisted_node);
        }
        *hoisted_node.mutable_input(0) = start_parent->name();
        *start->mutable_input(0) = hoisted_node.name();

        auto parent = graph_utils::GetInputNode(node, graph);
        TF_RETURN_IF_ERROR(graph.UpdateFanouts(node.name(), parent->name()));
        graph.DeleteNodes({node.name()});
        graph.AddNode(std::move(hoisted_node));
        updated = true;
      }
    }
  } while (updated);
  return Status::OK();
}

void HoistDataDiscardingOps::Feedback(Cluster* cluster, const GrapplerItem& item,
                                      const GraphDef& optimize_output,
                                      double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(HoistDataDiscardingOps, "hoist_data_discarding_ops");

}  // namespace grappler
}  // namespace tensorflow

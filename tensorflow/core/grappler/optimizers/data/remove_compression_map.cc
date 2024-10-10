/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/remove_compression_map.h"

#include <string>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace grappler {

namespace {

absl::StatusOr<std::string> GetCompressionFunctionName(const GraphDef& graph) {
  for (const auto& function : graph.library().function()) {
    for (const auto& node : function.node_def()) {
      if (node.op() == "CompressElement") {
        return function.signature().name();
      }
    }
  }
  return errors::Internal("Compression function not found.");
}

absl::StatusOr<NodeDef> GetCompressionMapNode(const GraphDef& graph) {
  TF_ASSIGN_OR_RETURN(std::string compression_function_name,
                      GetCompressionFunctionName(graph));
  for (const auto& node : graph.node()) {
    if (node.op() != "ParallelMapDatasetV2") {
      continue;
    }
    if (auto it = node.attr().find("f");
        it != node.attr().end() && it->second.has_func() &&
        it->second.func().name() == compression_function_name) {
      return node;
    }
  }
  return errors::Internal("Compression map node not found.");
}

}  // namespace

absl::Status RemoveCompressionMap::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
  *output = item.graph;
  TF_ASSIGN_OR_RETURN(NodeDef compression_map_node,
                      GetCompressionMapNode(*output));
  MutableGraphView graph(output);
  for (const auto& compression_map_output :
       graph.GetFanout(graph.GetOutputPort(compression_map_node.name(), 0))) {
    compression_map_output.node->clear_input();
    compression_map_output.node->add_input(compression_map_node.input().Get(0));
    ++stats->num_changes;
  }
  return absl::OkStatus();
}

REGISTER_GRAPH_OPTIMIZER_AS(RemoveCompressionMap, "remove_compression_map");

}  // namespace grappler
}  // namespace tensorflow

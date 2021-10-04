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

#include "tensorflow/core/grappler/optimizers/data/shuffle_and_repeat_fusion.h"

#include "absl/container/flat_hash_set.h"
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
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kShuffleDataset[] = "ShuffleDataset";
constexpr char kShuffleDatasetV2[] = "ShuffleDatasetV2";
constexpr char kShuffleDatasetV3[] = "ShuffleDatasetV3";
constexpr char kRepeatDataset[] = "RepeatDataset";
constexpr char kShuffleAndRepeatDataset[] = "ShuffleAndRepeatDataset";
constexpr char kShuffleAndRepeatDatasetV2[] = "ShuffleAndRepeatDatasetV2";

constexpr char kReshuffleEachIteration[] = "reshuffle_each_iteration";

Status FuseShuffleV1AndRepeat(const NodeDef& shuffle_node,
                              const NodeDef& repeat_node,
                              MutableGraphView* graph, GraphDef* output,
                              NodeDef* fused_node) {
  fused_node->set_op(kShuffleAndRepeatDataset);
  graph_utils::SetUniqueGraphNodeName(kShuffleAndRepeatDataset, output,
                                      fused_node);

  // Set the `input` input argument.
  fused_node->add_input(shuffle_node.input(0));

  // Set the `buffer_size` input argument.
  fused_node->add_input(shuffle_node.input(1));

  // Set the `seed` input argument.
  fused_node->add_input(shuffle_node.input(2));

  // Set the `seed2` input argument.
  fused_node->add_input(shuffle_node.input(3));

  // Set the `count` input argument.
  fused_node->add_input(repeat_node.input(1));

  // Set `output_types`, `output_shapes`, and `reshuffle_each_iteration`
  // attributes.
  graph_utils::CopyShapesAndTypesAttrs(shuffle_node, fused_node);
  graph_utils::CopyAttribute(kReshuffleEachIteration, shuffle_node, fused_node);

  // Optionally set the `metadata` attribute.
  graph_utils::MaybeSetFusedMetadata(shuffle_node, repeat_node, fused_node);

  return Status::OK();
}

Status FuseShuffleV2AndRepeat(const NodeDef& shuffle_node,
                              const NodeDef& repeat_node,
                              MutableGraphView* graph, GraphDef* output,
                              NodeDef* fused_node) {
  fused_node->set_op(kShuffleAndRepeatDatasetV2);
  graph_utils::SetUniqueGraphNodeName(kShuffleAndRepeatDatasetV2, output,
                                      fused_node);

  NodeDef zero_node = *graph_utils::AddScalarConstNode<int64_t>(0, graph);

  // Set the `input` input argument.
  fused_node->add_input(shuffle_node.input(0));

  // Set the `buffer_size` input argument.
  fused_node->add_input(shuffle_node.input(1));

  // Default the `seed` input argument to 0.
  fused_node->add_input(zero_node.name());

  // Default the `seed2` input argument to 0.
  fused_node->add_input(zero_node.name());

  // Set the `count` input argument.
  fused_node->add_input(repeat_node.input(1));

  // Set the `seed_generator` input argument.
  fused_node->add_input(shuffle_node.input(2));

  // Set `output_types` and `output_shapes` attributes.
  graph_utils::CopyShapesAndTypesAttrs(shuffle_node, fused_node);

  // Default the `reshuffle_each_iteration` attribute to true.
  (*fused_node->mutable_attr())[kReshuffleEachIteration].set_b(true);

  // Optionally set the `metadata` attribute.
  graph_utils::MaybeSetFusedMetadata(shuffle_node, repeat_node, fused_node);

  return Status::OK();
}

Status FuseShuffleV3AndRepeat(const NodeDef& shuffle_node,
                              const NodeDef& repeat_node,
                              MutableGraphView* graph, GraphDef* output,
                              NodeDef* fused_node) {
  fused_node->set_op(kShuffleAndRepeatDatasetV2);
  graph_utils::SetUniqueGraphNodeName(kShuffleAndRepeatDataset, output,
                                      fused_node);

  // Set the `input` input argument.
  fused_node->add_input(shuffle_node.input(0));

  // Set the `buffer_size` input argument.
  fused_node->add_input(shuffle_node.input(1));

  // Set the `seed` input argument.
  fused_node->add_input(shuffle_node.input(2));

  // Set the `seed2` input argument.
  fused_node->add_input(shuffle_node.input(3));

  // Set the `count` input argument.
  fused_node->add_input(repeat_node.input(1));

  // Set the `seed_generator` input argument.
  fused_node->add_input(shuffle_node.input(4));

  // Set `output_types`, `output_shapes`, and `reshuffle_each_iteration`
  // attributes.
  graph_utils::CopyShapesAndTypesAttrs(shuffle_node, fused_node);
  graph_utils::CopyAttribute(kReshuffleEachIteration, shuffle_node, fused_node);

  // Optionally set the `metadata` attribute.
  graph_utils::MaybeSetFusedMetadata(shuffle_node, repeat_node, fused_node);

  return Status::OK();
}

}  // namespace

Status ShuffleAndRepeatFusion::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;

  for (const NodeDef& repeat_node : item.graph.node()) {
    if (repeat_node.op() != kRepeatDataset) {
      continue;
    }

    const NodeDef& shuffle_node =
        *graph_utils::GetInputNode(repeat_node, graph);

    NodeDef fused_node;
    if (shuffle_node.op() == kShuffleDataset) {
      TF_RETURN_IF_ERROR(FuseShuffleV1AndRepeat(shuffle_node, repeat_node,
                                                &graph, output, &fused_node));
    } else if (shuffle_node.op() == kShuffleDatasetV2) {
      TF_RETURN_IF_ERROR(FuseShuffleV2AndRepeat(shuffle_node, repeat_node,
                                                &graph, output, &fused_node));

    } else if (shuffle_node.op() == kShuffleDatasetV3) {
      TF_RETURN_IF_ERROR(FuseShuffleV3AndRepeat(shuffle_node, repeat_node,
                                                &graph, output, &fused_node));
    } else {
      continue;
    }

    NodeDef& shuffle_and_repeat_node = *graph.AddNode(std::move(fused_node));
    TF_RETURN_IF_ERROR(graph.UpdateFanouts(repeat_node.name(),
                                           shuffle_and_repeat_node.name()));
    // Update shuffle node fanouts to shuffle_and_repeat fanouts to take care of
    // control dependencies.
    TF_RETURN_IF_ERROR(graph.UpdateFanouts(shuffle_node.name(),
                                           shuffle_and_repeat_node.name()));

    // Mark the `Shuffle` and `Repeat` nodes for removal (as long as neither of
    // them needs to be preserved).
    const auto nodes_to_preserve = item.NodesToPreserve();
    if (nodes_to_preserve.find(shuffle_node.name()) ==
            nodes_to_preserve.end() &&
        nodes_to_preserve.find(repeat_node.name()) == nodes_to_preserve.end()) {
      nodes_to_delete.insert(shuffle_node.name());
      nodes_to_delete.insert(repeat_node.name());
    }
    stats->num_changes++;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(ShuffleAndRepeatFusion,
                            "shuffle_and_repeat_fusion");

}  // namespace grappler
}  // namespace tensorflow

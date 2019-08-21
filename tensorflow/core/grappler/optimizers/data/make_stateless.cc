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

#include "tensorflow/core/grappler/optimizers/data/make_stateless.h"

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kCacheDataset[] = "CacheDataset";
constexpr char kCacheDatasetV2[] = "CacheDatasetV2";
constexpr char kReshuffleEachIteration[] = "reshuffle_each_iteration";
constexpr char kShuffleDataset[] = "ShuffleDataset";
constexpr char kShuffleDatasetV2[] = "ShuffleDatasetV2";

}  // namespace

Status MakeStateless::OptimizeAndCollectStats(Cluster* cluster,
                                              const GrapplerItem& item,
                                              GraphDef* output,
                                              OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);

  NodeDef* zero_node = graph_utils::AddScalarConstNode<int64>(0, &graph);

  for (NodeDef& node : *output->mutable_node()) {
    if (node.op() == kShuffleDatasetV2) {
      *node.mutable_op() = kShuffleDataset;
      // remove `seed_generator` input
      node.mutable_input()->RemoveLast();
      // add `seed` input
      node.add_input(zero_node->name());
      // add `seed2` input
      node.add_input(zero_node->name());
      // set `reshuffle_each_iteration` attr
      (*node.mutable_attr())[kReshuffleEachIteration].set_b(true);
    } else if (node.op() == kCacheDatasetV2) {
      *node.mutable_op() = kCacheDataset;
      // remove `cache` input
      node.mutable_input()->RemoveLast();
    }
  }

  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(MakeStateless, "make_stateless");

}  // namespace grappler
}  // namespace tensorflow

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

#include "tensorflow/core/framework/attr_value.pb.h"
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

constexpr char kFusedOpName[] = "ShuffleAndRepeatDataset";

}  // namespace

Status ShuffleAndRepeatFusion::Optimize(Cluster* cluster,
                                        const GrapplerItem& item,
                                        GraphDef* output) {
  *output = item.graph;
  MutableGraphView graph(output);
  std::set<string> nodes_to_delete;

  auto make_shuffle_and_repeat_node = [&output](const NodeDef& shuffle_node,
                                                const NodeDef& repeat_node) {
    NodeDef new_node;
    new_node.set_op(kFusedOpName);
    graph_utils::SetUniqueGraphNodeName(kFusedOpName, output, &new_node);

    // Set the `input` input argument.
    new_node.add_input(shuffle_node.input(0));

    // Set the `buffer_size` input argument.
    new_node.add_input(shuffle_node.input(1));

    // Set the `seed` input argument.
    new_node.add_input(shuffle_node.input(2));

    // Set the `seed2` input argument.
    new_node.add_input(shuffle_node.input(3));

    // Set the `count` input argument.
    new_node.add_input(repeat_node.input(1));

    // Set `output_types` and `output_shapes` attributes.
    for (auto key : {"output_shapes", "output_types"}) {
      (*new_node.mutable_attr())[key] = repeat_node.attr().at(key);
    }
    return new_node;
  };

  for (const NodeDef& node : item.graph.node()) {
    if (node.op() != "RepeatDataset") {
      continue;
    }

    // Use a more descriptive variable name now that we know the node type.
    const NodeDef& repeat_node = node;
    GraphView::InputPort input_port = graph.GetInputPort(repeat_node.name(), 0);
    NodeDef* node2 = graph.GetRegularFanin(input_port).node;
    if (node2->op() != "ShuffleDataset") {
      continue;
    }
    // Use a more descriptive variable name now that we know the node type.
    const NodeDef& shuffle_node = *node2;

    NodeDef* shuffle_and_repeat_node =
        graph.AddNode(make_shuffle_and_repeat_node(shuffle_node, repeat_node));
    graph.ReplaceInput(repeat_node, *shuffle_and_repeat_node);

    // Mark the `Shuffle` and `Repeat` nodes for removal.
    nodes_to_delete.insert(shuffle_node.name());
    nodes_to_delete.insert(repeat_node.name());
  }

  graph.DeleteNodes(nodes_to_delete);
  return Status::OK();
}

void ShuffleAndRepeatFusion::Feedback(Cluster* cluster,
                                      const GrapplerItem& item,
                                      const GraphDef& optimize_output,
                                      double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(ShuffleAndRepeatFusion,
                            "shuffle_and_repeat_fusion");

}  // end namespace grappler
}  // end namespace tensorflow

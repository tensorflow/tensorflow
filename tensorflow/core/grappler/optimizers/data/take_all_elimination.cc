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

#include "tensorflow/core/grappler/optimizers/data/take_all_elimination.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {

Status TakeAllElimination::Optimize(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* output) {
  *output = item.graph;
  GraphView graph(output);
  std::set<string> nodes_to_delete;
  for (const NodeDef& node : item.graph.node()) {
    if (node.op() != "TakeDataset") continue;

    // Use a more descriptive variable name now that we know the node type.
    const auto take_node(node);

    const NodeDef& count_node = *graph.GetNode(take_node.input(1));

    // We are looking only for take(-1) nodes.
    if (count_node.attr().at("value").tensor().int64_val(0) >= 0) continue;

    GraphView::InputPort input_port = graph.GetInputPort(take_node.name(), 0);
    NodeDef* const parent = graph.GetRegularFanin(input_port).node;
    graph_utils::ReplaceInput(take_node, *parent, &graph);

    nodes_to_delete.insert(take_node.name());
  }
  TF_RETURN_IF_ERROR(graph_utils::DeleteNodes(nodes_to_delete, output));
  return Status::OK();
}

void TakeAllElimination::Feedback(Cluster* cluster, const GrapplerItem& item,
                                  const GraphDef& optimize_output,
                                  double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(TakeAllElimination, "take_all_elimination");

}  // end namespace grappler
}  // end namespace tensorflow

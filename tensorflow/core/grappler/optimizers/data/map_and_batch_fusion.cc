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

#include "tensorflow/core/grappler/optimizers/data/map_and_batch_fusion.h"

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

Status MapAndBatchFusion::Optimize(Cluster* cluster, const GrapplerItem& item,
                                   GraphDef* output) {
  *output = item.graph;
  GraphView graph(output);
  std::set<string> nodes_to_delete;
  for (const NodeDef& node : item.graph.node()) {
    if (node.op() != "BatchDataset") {
      continue;
    }

    // Use a more descriptive variable name now that we now the node type.
    NodeDef batch_node(node);
    GraphView::InputPort input_port = graph.GetInputPort(batch_node.name(), 0);
    NodeDef* node2 = graph.GetRegularFanin(input_port).node;
    if (node2->op() != "MapDataset" && node2->op() != "ParallelMapDataset") {
      continue;
    }

    // Use a more descriptive variable name now that we now the node type.
    NodeDef* map_node = node2;
    NodeDef* new_node = output->mutable_node()->Add();
    new_node->set_op("MapAndBatchDatasetV2");
    new_node->set_name(
        strings::StrCat("MapAndBatchDatasetV2/_", output->node_size()));

    // Set the `input` input argument.
    new_node->add_input(map_node->input(0));

    // Set the `other_arguments` input arguments.
    int num_other_args;
    if (map_node->op() == "ParallelMapDataset") {
      num_other_args = map_node->input_size() - 2;
    } else {
      num_other_args = map_node->input_size() - 1;
    }
    for (int i = 0; i < num_other_args; i++) {
      new_node->add_input(map_node->input(i + 1));
    }

    // Set the `batch_size` input argument.
    new_node->add_input(batch_node.input(1));

    // Set the `num_parallel_calls` input argument.
    if (map_node->op() == "ParallelMapDataset") {
      // The type of the `num_parallel_calls` argument in ParallelMapDataset
      // and MapAndBatchDataset is different (int32 and int64 respectively)
      // so we cannot reuse the same Const node and thus create a new one.
      NodeDef* v = graph.GetNode(map_node->input(map_node->input_size() - 1));
      NodeDef* tmp;
      TF_RETURN_IF_ERROR(graph_utils::AddScalarConstNode<int64>(
          v->attr().at("value").tensor().int_val(0), output, &tmp));
      new_node->add_input(tmp->name());
    } else {
      NodeDef* tmp;
      TF_RETURN_IF_ERROR(
          graph_utils::AddScalarConstNode<int64>(1, output, &tmp));
      new_node->add_input(tmp->name());
    }

    // Set the `drop_remainder` input argument.
    {
      NodeDef* tmp;
      TF_RETURN_IF_ERROR(
          graph_utils::AddScalarConstNode<bool>(false, output, &tmp));
      new_node->add_input(tmp->name());
    }

    // Set `f` and `Targuments` attributes.
    for (auto key : {"f", "Targuments"}) {
      (*new_node->mutable_attr())[key] = map_node->attr().at(key);
    }
    // Set `output_types` and `output_shapes` attributes.
    for (auto key : {"output_shapes", "output_types"}) {
      (*new_node->mutable_attr())[key] = batch_node.attr().at(key);
    }

    // Mark the `Map` and `Batch` nodes for removal.
    nodes_to_delete.insert(map_node->name());
    nodes_to_delete.insert(batch_node.name());

    // Update the input of the outputs of the `Batch` node to use
    // `MapAndBatch`.
    GraphView::OutputPort output_port =
        graph.GetOutputPort(batch_node.name(), 0);
    auto fanout = graph.GetFanout(output_port);
    for (auto it = fanout.begin(); it != fanout.end(); ++it) {
      NodeDef* node = it->node;
      node->set_input(0, new_node->name());
    }
  }
  TF_RETURN_IF_ERROR(graph_utils::DeleteNodes(nodes_to_delete, output));
  return Status::OK();
}

void MapAndBatchFusion::Feedback(Cluster* cluster, const GrapplerItem& item,
                                 const GraphDef& optimize_output,
                                 double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(MapAndBatchFusion, "map_and_batch_fusion");

}  // end namespace grappler
}  // end namespace tensorflow

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

#include "tensorflow/core/grappler/optimizers/data/rebatch.h"

#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"

namespace tensorflow {
namespace grappler {

Status RebatchOptimizer::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  if (!config) return Status::OK();

  num_workers_ = config->parameter_map().at("num_workers").i();
  return Status::OK();
}

namespace {

constexpr char kCastOp[] = "Cast";
constexpr char kRealDivOp[] = "RealDiv";
constexpr char kBatchDatasetOp[] = "BatchDatasetV2";

NodeDef* AddCastNode(const string& input, DataType src_t, DataType dst_t,
                     MutableGraphView* graph) {
  NodeDef cast_node;
  cast_node.set_op(kCastOp);
  cast_node.add_input(input);
  graph_utils::SetUniqueGraphNodeName(cast_node.op(), graph->graph(),
                                      &cast_node);
  AddNodeAttr("SrcT", src_t, &cast_node);
  AddNodeAttr("DstT", dst_t, &cast_node);

  return graph->AddNode(std::move(cast_node));
}

NodeDef* AddBinaryNode(const string& input_x, const string& input_y,
                       const string& op, DataType type,
                       MutableGraphView* graph) {
  NodeDef node;
  node.set_op(op);
  node.add_input(input_x);
  node.add_input(input_y);
  graph_utils::SetUniqueGraphNodeName(op, graph->graph(), &node);
  AddNodeAttr("T", type, &node);

  return graph->AddNode(std::move(node));
}

NodeDef* AddFloatDivNode(const string& input_x, const string& input_y,
                         MutableGraphView* graph) {
  return AddBinaryNode(input_x, input_y, kRealDivOp, DT_FLOAT, graph);
}

}  // anonymous namespace

Status RebatchOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                  GraphDef* output) {
  *output = item.graph;
  MutableGraphView graph(output);

  absl::flat_hash_set<string> nodes_to_delete;
  for (const NodeDef& node : item.graph.node()) {
    if (node.op() == kBatchDatasetOp) {
      NodeDef* batch_size_node = graph_utils::GetInputNode(node, graph, 1);
      NodeDef tmp_node;
      tmp_node = *batch_size_node;
      graph_utils::SetUniqueGraphNodeName(tmp_node.op(), graph.graph(),
                                          &tmp_node);
      NodeDef* copy_batch_size_node = graph.AddNode(std::move(tmp_node));
      NodeDef* float_copy_batch_size_node =
          AddCastNode(copy_batch_size_node->name(), DT_INT64, DT_FLOAT, &graph);
      NodeDef* num_worker_node =
          graph_utils::AddScalarConstNode<int64>(num_workers_, &graph);
      NodeDef* float_num_worker_node =
          AddCastNode(num_worker_node->name(), DT_INT64, DT_FLOAT, &graph);
      NodeDef* divided_batch_size_node =
          AddFloatDivNode(float_copy_batch_size_node->name(),
                          float_num_worker_node->name(), &graph);
      NodeDef* cast_new_batch_size_node = AddCastNode(
          divided_batch_size_node->name(), DT_FLOAT, DT_INT64, &graph);
      TF_RETURN_IF_ERROR(graph.UpdateFanouts(batch_size_node->name(),
                                             cast_new_batch_size_node->name()));
      nodes_to_delete.insert(batch_size_node->name());
      break;
    }
  }
  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return Status::OK();
}

void RebatchOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                                const GraphDef& optimize_output,
                                double result) {}

REGISTER_GRAPH_OPTIMIZER_AS(RebatchOptimizer, "tf_data_rebatcher");

}  // namespace grappler
}  // namespace tensorflow

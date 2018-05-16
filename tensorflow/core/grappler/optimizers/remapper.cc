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

#include "tensorflow/core/grappler/optimizers/remapper.h"

#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {

void AddBatchNormNodes(GraphDef* optimized_graph, const NodeDef& fused_node) {
  const string& x = fused_node.input(0);
  const string& scale = fused_node.input(1);
  const string& offset = fused_node.input(2);
  const string& mean = fused_node.input(3);
  const string& variance = fused_node.input(4);

  float epsilon = 0.0f;
  if (fused_node.attr().count("epsilon")) {
    epsilon = fused_node.attr().at("epsilon").f();
  }
  DataType dtype = fused_node.attr().at("T").type();
  Tensor value(dtype, TensorShape());
  value.scalar<float>()() = epsilon;
  NodeDef* variance_epsilon = optimized_graph->add_node();
  TF_CHECK_OK(ConstantFolding::CreateNodeDef(
      AddPrefixToNodeName("Const", fused_node.name()), &value,
      variance_epsilon));
  variance_epsilon->set_device(fused_node.device());

  NodeDef* variance_plus_epsilon = optimized_graph->add_node();
  variance_plus_epsilon->set_name(
      AddPrefixToNodeName("VarPlusEpsilon", fused_node.name()));
  variance_plus_epsilon->set_op("Add");
  (*variance_plus_epsilon->mutable_attr())["T"].set_type(dtype);
  variance_plus_epsilon->set_device(fused_node.device());
  *variance_plus_epsilon->add_input() = variance;
  *variance_plus_epsilon->add_input() = variance_epsilon->name();

  NodeDef* inv = optimized_graph->add_node();
  inv->set_name(AddPrefixToNodeName("Inv", fused_node.name()));
  inv->set_op("Rsqrt");
  inv->set_device(fused_node.device());
  (*inv->mutable_attr())["T"].set_type(dtype);
  *inv->add_input() = variance_plus_epsilon->name();

  NodeDef* scaled = optimized_graph->add_node();
  scaled->set_name(AddPrefixToNodeName("Scaled", fused_node.name()));
  scaled->set_op("Mul");
  scaled->set_device(fused_node.device());
  (*scaled->mutable_attr())["T"].set_type(dtype);
  *scaled->add_input() = inv->name();
  *scaled->add_input() = scale;

  NodeDef* a = optimized_graph->add_node();
  a->set_name(AddPrefixToNodeName("Mul", fused_node.name()));
  a->set_op("Mul");
  a->set_device(fused_node.device());
  (*a->mutable_attr())["T"].set_type(dtype);
  *a->add_input() = x;
  *a->add_input() = scaled->name();

  NodeDef* b = optimized_graph->add_node();
  b->set_name(AddPrefixToNodeName("Mul2", fused_node.name()));
  b->set_op("Mul");
  b->set_device(fused_node.device());
  (*b->mutable_attr())["T"].set_type(dtype);
  *b->add_input() = mean;
  *b->add_input() = scaled->name();

  NodeDef* c = optimized_graph->add_node();
  c->set_name(AddPrefixToNodeName("Offset", fused_node.name()));
  c->set_op("Sub");
  c->set_device(fused_node.device());
  (*c->mutable_attr())["T"].set_type(dtype);
  *c->add_input() = offset;
  *c->add_input() = b->name();

  NodeDef* r = optimized_graph->add_node();
  r->set_name(fused_node.name());
  r->set_op("Add");
  r->set_device(fused_node.device());
  (*r->mutable_attr())["T"].set_type(dtype);
  *r->add_input() = a->name();
  *r->add_input() = c->name();
}

Status Remapper::Optimize(Cluster* /*cluster*/, const GrapplerItem& item,
                          GraphDef* optimized_graph) {
  GraphProperties properties(item);
  TF_RETURN_IF_ERROR(properties.InferStatically(false));
  GraphView graph(const_cast<GraphDef*>(&item.graph));

  // During inference, most of the inputs to FusedBatchNorm are constant, and we
  // can therefore replace the op with a much cheaper set of primitives.
  for (const NodeDef& node : item.graph.node()) {
    if (node.op() == "FusedBatchNorm" || node.op() == "FusedBatchNormV2") {
      bool optimizable = (node.attr().count("T") == 0 ||
                          node.attr().at("T").type() == DT_FLOAT);
      optimizable &= (node.attr().count("is_training") == 0 ||
                      !node.attr().at("is_training").b());
      if (optimizable) {
        std::unordered_set<int> const_inputs;
        for (const string& input : node.input()) {
          int pos;
          const string input_node = ParseNodeName(input, &pos);
          if (properties.HasInputProperties(input_node)) {
            const auto& props = properties.GetInputProperties(input_node);
            if (props.size() > pos && props[pos].has_value()) {
              const_inputs.insert(pos);
            }
          }
        }
        // TODO(bsteiner): use the cost model to compare the cost of fused batch
        // norm against that of the optimized form.
        optimizable = (const_inputs.size() >= 4);
      }
      if (optimizable) {
        for (GraphView::Edge edge : graph.GetFanoutEdges(node, false)) {
          if (edge.src.port_id != 0) {
            // The optimized version only generates the first output.
            optimizable = false;
            break;
          }
        }
      }
      if (optimizable) {
        AddBatchNormNodes(optimized_graph, node);
        continue;
      }
    }
    *optimized_graph->add_node() = node;
  }

  *optimized_graph->mutable_library() = item.graph.library();
  *optimized_graph->mutable_versions() = item.graph.versions();

  return Status::OK();
}

void Remapper::Feedback(Cluster* /*cluster*/, const GrapplerItem& /*item*/,
                        const GraphDef& /*optimized_graph*/,
                        double /*result*/) {
  // Nothing to do for ArithmeticOptimizer.
}

}  // namespace grappler
}  // namespace tensorflow

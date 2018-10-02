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
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {

void AddBatchNormNodes(GraphDef* optimized_graph, const NodeDef& fused_node) {
  const string& x = fused_node.input(0);
  string scale = fused_node.input(1);
  string offset = fused_node.input(2);
  string mean = fused_node.input(3);
  string variance = fused_node.input(4);

  if (fused_node.attr().at("data_format").s() == "NCHW") {
    // Need to reshape the last 4 inputs
    NodeDef* new_shape = optimized_graph->add_node();
    new_shape->set_name(AddPrefixToNodeName("NCHWShape", fused_node.name()));
    new_shape->set_op("Const");
    new_shape->set_device(fused_node.device());
    *new_shape->add_input() = AsControlDependency(scale);
    (*new_shape->mutable_attr())["dtype"].set_type(DT_INT32);
    Tensor t(DT_INT32, {4});
    t.flat<int32>()(0) = 1;
    t.flat<int32>()(1) = -1;
    t.flat<int32>()(2) = 1;
    t.flat<int32>()(3) = 1;
    t.AsProtoTensorContent(
        (*new_shape->mutable_attr())["value"].mutable_tensor());

    NodeDef* reshaped_scale = optimized_graph->add_node();
    reshaped_scale->set_name(
        AddPrefixToNodeName("NCHWShapedScale", fused_node.name()));
    reshaped_scale->set_op("Reshape");
    reshaped_scale->set_device(fused_node.device());
    *reshaped_scale->add_input() = scale;
    *reshaped_scale->add_input() = new_shape->name();
    (*reshaped_scale->mutable_attr())["T"] = fused_node.attr().at("T");
    (*reshaped_scale->mutable_attr())["Tshape"].set_type(DT_INT32);
    scale = reshaped_scale->name();

    NodeDef* reshaped_offset = optimized_graph->add_node();
    reshaped_offset->set_name(
        AddPrefixToNodeName("NCHWShapedOffset", fused_node.name()));
    reshaped_offset->set_op("Reshape");
    reshaped_offset->set_device(fused_node.device());
    *reshaped_offset->add_input() = offset;
    *reshaped_offset->add_input() = new_shape->name();
    (*reshaped_offset->mutable_attr())["T"] = fused_node.attr().at("T");
    (*reshaped_offset->mutable_attr())["Tshape"].set_type(DT_INT32);
    offset = reshaped_offset->name();

    NodeDef* reshaped_mean = optimized_graph->add_node();
    reshaped_mean->set_name(
        AddPrefixToNodeName("NCHWShapedMean", fused_node.name()));
    reshaped_mean->set_op("Reshape");
    reshaped_mean->set_device(fused_node.device());
    *reshaped_mean->add_input() = mean;
    *reshaped_mean->add_input() = new_shape->name();
    (*reshaped_mean->mutable_attr())["T"] = fused_node.attr().at("T");
    (*reshaped_mean->mutable_attr())["Tshape"].set_type(DT_INT32);
    mean = reshaped_mean->name();

    NodeDef* reshaped_variance = optimized_graph->add_node();
    reshaped_variance->set_name(
        AddPrefixToNodeName("NCHWShapedVariance", fused_node.name()));
    reshaped_variance->set_op("Reshape");
    reshaped_variance->set_device(fused_node.device());
    *reshaped_variance->add_input() = variance;
    *reshaped_variance->add_input() = new_shape->name();
    (*reshaped_variance->mutable_attr())["T"] = fused_node.attr().at("T");
    (*reshaped_variance->mutable_attr())["Tshape"].set_type(DT_INT32);
    variance = reshaped_variance->name();
  }

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
  bool inferred_properties = false;
  GraphView graph(const_cast<GraphDef*>(&item.graph));

  // During inference, most of the inputs to FusedBatchNorm are constant, and we
  // can therefore replace the op with a much cheaper set of primitives.
  optimized_graph->mutable_node()->Reserve(item.graph.node_size());
  for (const NodeDef& node : item.graph.node()) {
    if (node.op() == "FusedBatchNorm" || node.op() == "FusedBatchNormV2") {
      bool optimizable = (node.attr().count("T") == 0 ||
                          node.attr().at("T").type() == DT_FLOAT);
      optimizable &= (node.attr().count("is_training") == 0 ||
                      !node.attr().at("is_training").b());
      if (optimizable) {
        int const_inputs = 0;
        if (!inferred_properties) {
          // Infer properties lazily in case they are not needed.
          TF_RETURN_IF_ERROR(properties.InferStatically(false));
          inferred_properties = true;
        }
        const auto& props = properties.GetInputProperties(node.name());
        for (const auto& prop : props) {
          if (prop.has_value()) {
            const_inputs += 1;
          }
        }
        // TODO(bsteiner): use the cost model to compare the cost of fused batch
        // norm against that of the optimized form.
        optimizable = (const_inputs >= 4);
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
        VLOG(1) << "Optimizing fused batch norm node " << node.DebugString();
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
  // Nothing to do for RemapperOptimizer.
}

}  // namespace grappler
}  // namespace tensorflow

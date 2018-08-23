/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/auto_parallel.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class AutoParallelTest : public ::testing::Test {};

TEST_F(AutoParallelTest, SimpleParallel) {
  tensorflow::Scope s = tensorflow::Scope::DisabledShapeInferenceScope();
  Output constant_a = ops::Const(s.WithOpName("constant_a"), 1.0f, {1});
  Output constant_b = ops::Const(s.WithOpName("constant_b"), 1, {1});
  Output constant_c =
      ops::Const(s.WithOpName("constant_c"), {1.0f, 2.0f, 3.0f}, {3});
  Output constant_d = ops::Const(s.WithOpName("constant_d"), 2.0f, {1});
  Output var = ops::Variable(s.WithOpName("var"), {1}, DT_FLOAT);
  Output assign = ops::Assign(s.WithOpName("assign"), {var}, {constant_a});
  Output identity = ops::Identity(s.WithOpName("identity"), {var});
  Output embedding = ops::Variable(s.WithOpName("embedding"), {3}, DT_FLOAT);
  Output emb_assign =
      ops::Assign(s.WithOpName("emb_assign"), {embedding}, {constant_c});
  Output fifo_queue = ops::FIFOQueue(s.WithOpName("fifo_queue"), {DT_FLOAT});
  auto dequeue = ops::QueueDequeueMany(s.WithOpName("dequeue"), {fifo_queue},
                                       {constant_b}, {DT_FLOAT});
  Output add = ops::AddN(s.WithOpName("add"), {constant_a, dequeue[0]});
  Output clip =
      ops::ClipByValue(s.WithOpName("clip"), {add}, {constant_a}, {constant_d});
  Output indices = ops::Const(s.WithOpName("indices"), 1, {1});
  Output values = ops::Const(s.WithOpName("values"), 2.0f, {1});
  Output learning_rate = ops::Const(s.WithOpName("learning_rate"), 0.01f, {1});
  Output apply_gradient = ops::ApplyGradientDescent(
      s.WithOpName("apply_gradient"), {var}, {learning_rate}, {clip});
  Output update_sparse = ops::ScatterAdd(s.WithOpName("scatter_add"),
                                         {embedding}, {indices}, {values});

  GrapplerItem item;
  item.init_ops.push_back("assign");
  item.init_ops.push_back("emb_assign");
  item.fetch.push_back("apply_gradient");
  item.fetch.push_back("scatter_add");
  // add trainable variables.
  item.trainable_variables.push_back("var");
  item.trainable_variables.push_back("embedding");
  // add gradients information.
  GradientsInfoDef_TensorInfoDef var_tensor;
  var_tensor.set_tensor_type(GradientsInfoDef_TensorInfoDef::TENSOR);
  var_tensor.set_values_tensor_name("var");
  GradientsInfoDef_TensorInfoDef var_grad_tensor;
  var_grad_tensor.set_tensor_type(GradientsInfoDef_TensorInfoDef::TENSOR);
  var_grad_tensor.set_values_tensor_name("add");
  item.gradients_info.push_back(std::make_pair(var_tensor, var_grad_tensor));
  GradientsInfoDef_TensorInfoDef embedding_tensor;
  embedding_tensor.set_tensor_type(GradientsInfoDef_TensorInfoDef::TENSOR);
  embedding_tensor.set_values_tensor_name("embedding");
  GradientsInfoDef_TensorInfoDef embedding_grad_tensor;
  embedding_grad_tensor.set_tensor_type(
      GradientsInfoDef_TensorInfoDef::INDEXED_SLICES);
  embedding_grad_tensor.set_indices_tensor_name("indices");
  embedding_grad_tensor.set_values_tensor_name("values");
  item.gradients_info.push_back(
      std::make_pair(embedding_tensor, embedding_grad_tensor));
  // add op_def information.
  OpDef const_op_def;
  const_op_def.set_name("Const");
  const_op_def.add_output_arg();
  const_op_def.mutable_output_arg(0)->set_name("output");
  const_op_def.mutable_output_arg(0)->set_type_attr("dtype");
  const_op_def.add_attr();
  const_op_def.mutable_attr(0)->set_name("dtype");
  const_op_def.mutable_attr(0)->set_type("type");
  item.op_def.insert(std::make_pair("Const", const_op_def));
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  std::vector<std::string> shared_nodes = {
      "constant_b",    "constant_c",     "constant_d", "var",        "assign",
      "identity",      "embedding",      "emb_assign", "fifo_queue", "clip",
      "learning_rate", "apply_gradient", "scatter_add"};
  std::vector<std::string> replica_nodes = {"constant_a", "dequeue", "add",
                                            "indices", "values"};
  std::vector<std::string> aggregation_shared_nodes = {
      "AutoParallel-NumReplicas-Int-Const",
      "AutoParallel-NumReplicas-Float-Const",
      "AutoParallel-Add-add",
      "AutoParallel-Div-add",
      "AutoParallel-Accum-values",
      "AutoParallel-TakeGrad-values",
      "AutoParallel-AccumApplyControl-values",
      "AutoParallel-TakeGrad-values-Cast"};
  std::vector<std::string> aggregation_replica_nodes = {
      "MaxLong-Const", "AccumApply-values", "Cast-indices"};

  AutoParallel parallel(2);
  GraphDef output;
  Status status = parallel.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  int expected_node_size =
      shared_nodes.size() + aggregation_shared_nodes.size() +
      (replica_nodes.size() + aggregation_replica_nodes.size()) * 2;
  EXPECT_EQ(expected_node_size, output.node_size());

  std::map<std::string, NodeDef> all_nodes;
  for (int i = 0; i < output.node_size(); i++) {
    all_nodes.insert(std::make_pair(output.node(i).name(), output.node(i)));
  }

  shared_nodes.insert(shared_nodes.end(), aggregation_shared_nodes.begin(),
                      aggregation_shared_nodes.end());
  for (const auto& shared_node : shared_nodes) {
    EXPECT_TRUE(all_nodes.find(shared_node) != all_nodes.end());
  }

  replica_nodes.insert(replica_nodes.end(), aggregation_replica_nodes.begin(),
                       aggregation_replica_nodes.end());
  auto prefix_0 = "AutoParallel-Replica-0";
  auto prefix_1 = "AutoParallel-Replica-1";
  for (const auto& replica_node : replica_nodes) {
    EXPECT_TRUE(all_nodes.find(replica_node) == all_nodes.end());
    EXPECT_TRUE(all_nodes.find(AddPrefixToNodeName(replica_node, prefix_0)) !=
                all_nodes.end());
    EXPECT_TRUE(all_nodes.find(AddPrefixToNodeName(replica_node, prefix_1)) !=
                all_nodes.end());
  }

  const NodeDef& node_apply = all_nodes.find("apply_gradient")->second;
  EXPECT_EQ("clip", node_apply.input(2));

  const NodeDef& node_clip = all_nodes.find("clip")->second;
  EXPECT_EQ("AutoParallel-Div-add", node_clip.input(0));

  const NodeDef& node_div = all_nodes.find("AutoParallel-Div-add")->second;
  EXPECT_EQ("AutoParallel-Add-add", node_div.input(0));
  EXPECT_EQ("AutoParallel-NumReplicas-Float-Const", node_div.input(1));

  const NodeDef& node_add_n = all_nodes.find("AutoParallel-Add-add")->second;
  EXPECT_EQ(AddPrefixToNodeName("add", prefix_0), node_add_n.input(0));
  EXPECT_EQ(AddPrefixToNodeName("add", prefix_1), node_add_n.input(1));

  const NodeDef& node_scatter_add = all_nodes.find("scatter_add")->second;
  EXPECT_EQ("AutoParallel-TakeGrad-values-Cast", node_scatter_add.input(1));
  EXPECT_EQ("AutoParallel-TakeGrad-values:1", node_scatter_add.input(2));

  const NodeDef& node_take_grad =
      all_nodes.find("AutoParallel-TakeGrad-values")->second;
  EXPECT_EQ("AutoParallel-Accum-values", node_take_grad.input(0));
  EXPECT_EQ("AutoParallel-NumReplicas-Int-Const", node_take_grad.input(1));
  EXPECT_EQ("^AutoParallel-AccumApplyControl-values", node_take_grad.input(2));

  const NodeDef& node_apply_control =
      all_nodes.find("AutoParallel-AccumApplyControl-values")->second;
  EXPECT_EQ(
      strings::StrCat("^", AddPrefixToNodeName("AccumApply-values", prefix_0)),
      node_apply_control.input(0));
  EXPECT_EQ(
      strings::StrCat("^", AddPrefixToNodeName("AccumApply-values", prefix_1)),
      node_apply_control.input(1));

  for (int i = 0; i < 2; i++) {
    auto prefix = strings::StrCat("AutoParallel-Replica-", i);
    const NodeDef& node_accum_apply =
        all_nodes.find(AddPrefixToNodeName("AccumApply-values", prefix))
            ->second;
    EXPECT_EQ("AutoParallel-Accum-values", node_accum_apply.input(0));
    EXPECT_EQ(AddPrefixToNodeName("MaxLong-Const", prefix),
              node_accum_apply.input(1));
    EXPECT_EQ(AddPrefixToNodeName("Cast-indices", prefix),
              node_accum_apply.input(2));
    EXPECT_EQ(AddPrefixToNodeName("values", prefix), node_accum_apply.input(3));
    EXPECT_EQ(false, node_accum_apply.attr().at("has_known_shape").b());

    const NodeDef& node_cast_indices =
        all_nodes.find(AddPrefixToNodeName("Cast-indices", prefix))->second;
    EXPECT_EQ(AddPrefixToNodeName("indices", prefix),
              node_cast_indices.input(0));
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

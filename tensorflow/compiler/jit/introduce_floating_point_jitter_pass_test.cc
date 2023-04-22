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

#include "tensorflow/compiler/jit/introduce_floating_point_jitter_pass_internal.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/linalg_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/compiler/jit/node_matchers.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using testing::matchers::Const;
using testing::matchers::Inputs;
using testing::matchers::Name;
using testing::matchers::NodeWith;
using testing::matchers::Op;
using testing::matchers::Out;

TEST(IntroduceFloatingPointJitterTest, SingleOutputFP32) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output input_a = ops::Placeholder(root.WithOpName("input_a"), DT_FLOAT);
  Output input_b = ops::Placeholder(root.WithOpName("input_b"), DT_FLOAT);

  Output sigmoid_a = ops::Sigmoid(root.WithOpName("sigmoid_a"), input_a);
  Output sigmoid_b = ops::Sigmoid(root.WithOpName("sigmoid_b"), input_b);

  Output tanh_a = ops::Tanh(root.WithOpName("tanh_a"), sigmoid_a);
  Output tanh_b = ops::Tanh(root.WithOpName("tanh_b"), sigmoid_b);

  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  std::vector<string> tensor_names;
  tensor_names.push_back("sigmoid_a");
  tensor_names.push_back("sigmoid_b");

  TF_ASSERT_OK(IntroduceFloatingPointJitter(graph.get(), tensor_names, 0.01f));
  VLOG(1) << graph->ToGraphDefDebug().DebugString();

  auto m_sigmoid_a = Out(NodeWith(Name("sigmoid_a")));
  auto m_sigmoid_a_with_jitter =
      NodeWith(Op("Add"), Inputs(Const(0.01f), m_sigmoid_a));
  auto m_tanh_a = NodeWith(Op("Tanh"), Inputs(Out(m_sigmoid_a_with_jitter)));

  auto m_sigmoid_b = Out(NodeWith(Name("sigmoid_b")));
  auto m_sigmoid_b_with_jitter =
      NodeWith(Op("Add"), Inputs(Const(0.01f), m_sigmoid_b));
  auto m_tanh_b = NodeWith(Op("Tanh"), Inputs(Out(m_sigmoid_b_with_jitter)));

  Node* tanh_a_transformed = testing::FindNodeByName(graph.get(), "tanh_a");
  Node* tanh_b_transformed = testing::FindNodeByName(graph.get(), "tanh_b");

  ASSERT_NE(tanh_a_transformed, nullptr);
  ASSERT_NE(tanh_b_transformed, nullptr);

  EXPECT_THAT(tanh_a_transformed, m_tanh_a);
  EXPECT_THAT(tanh_b_transformed, m_tanh_b);
}

TEST(IntroduceFloatingPointJitterTest, TwoNodesOneUser) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output input_a = ops::Placeholder(root.WithOpName("input_a"), DT_FLOAT);
  Output input_b = ops::Placeholder(root.WithOpName("input_b"), DT_FLOAT);

  Output sigmoid_a = ops::Sigmoid(root.WithOpName("sigmoid_a"), input_a);
  Output sigmoid_b = ops::Sigmoid(root.WithOpName("sigmoid_b"), input_b);

  Output add = ops::Add(root.WithOpName("add"), sigmoid_a, sigmoid_b);

  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  std::vector<string> tensor_names;
  tensor_names.push_back("sigmoid_a");
  tensor_names.push_back("sigmoid_b");

  TF_ASSERT_OK(IntroduceFloatingPointJitter(graph.get(), tensor_names, 0.01f));
  VLOG(1) << graph->ToGraphDefDebug().DebugString();

  auto m_sigmoid_a = Out(NodeWith(Name("sigmoid_a")));
  auto m_sigmoid_a_with_jitter =
      NodeWith(Op("Add"), Inputs(Const(0.01f), m_sigmoid_a));

  auto m_sigmoid_b = Out(NodeWith(Name("sigmoid_b")));
  auto m_sigmoid_b_with_jitter =
      NodeWith(Op("Add"), Inputs(Const(0.01f), m_sigmoid_b));

  auto m_add = NodeWith(Op("Add"), Inputs(Out(m_sigmoid_a_with_jitter),
                                          Out(m_sigmoid_b_with_jitter)));

  Node* add_transformed = testing::FindNodeByName(graph.get(), "add");

  ASSERT_NE(add_transformed, nullptr);

  EXPECT_THAT(add_transformed, m_add);
}

TEST(IntroduceFloatingPointJitterTest, NotFP32) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output input = ops::Placeholder(root.WithOpName("input"), DT_HALF);

  Output sigmoid = ops::Sigmoid(root.WithOpName("sigmoid"), input);

  Output tanh = ops::Tanh(root.WithOpName("tanh"), sigmoid);

  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  std::vector<string> tensor_names;
  tensor_names.push_back("sigmoid");

  TF_ASSERT_OK(IntroduceFloatingPointJitter(graph.get(), tensor_names, 0.01f));
  VLOG(1) << graph->ToGraphDefDebug().DebugString();

  auto m_sigmoid = Out(NodeWith(Name("sigmoid")));
  auto m_sigmoid_with_jitter =
      NodeWith(Op("Add"), Inputs(Const(Tensor(Eigen::half(0.01f))), m_sigmoid));
  auto m_tanh = NodeWith(Op("Tanh"), Inputs(Out(m_sigmoid_with_jitter)));

  Node* tanh_transformed = testing::FindNodeByName(graph.get(), "tanh");

  ASSERT_NE(tanh_transformed, nullptr);

  EXPECT_THAT(tanh_transformed, m_tanh);
}

TEST(IntroduceFloatingPointJitterTest, MultiOutput) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output input = ops::Placeholder(root.WithOpName("input"), DT_HALF);

  ops::Svd svd(root.WithOpName("svd"), input);

  Output tanh_s = ops::Tanh(root.WithOpName("tanh_s"), svd.s);
  Output tanh_u = ops::Tanh(root.WithOpName("tanh_u"), svd.u);
  Output tanh_v = ops::Tanh(root.WithOpName("tanh_v"), svd.v);

  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  std::vector<string> tensor_names;
  tensor_names.push_back("svd:0");
  tensor_names.push_back("svd:2");

  TF_ASSERT_OK(IntroduceFloatingPointJitter(graph.get(), tensor_names, 0.01f));
  VLOG(1) << graph->ToGraphDefDebug().DebugString();

  auto m_svd_s = Out(0, NodeWith(Name("svd")));
  auto m_svd_s_with_jitter = Out(
      NodeWith(Op("Add"), Inputs(Const(Tensor(Eigen::half(0.01f))), m_svd_s)));

  auto m_svd_u = Out(1, NodeWith(Name("svd")));

  auto m_svd_v = Out(2, NodeWith(Name("svd")));
  auto m_svd_v_with_jitter = Out(
      NodeWith(Op("Add"), Inputs(Const(Tensor(Eigen::half(0.01f))), m_svd_v)));

  auto m_tanh_s = NodeWith(Op("Tanh"), Inputs(m_svd_s_with_jitter));
  auto m_tanh_u = NodeWith(Op("Tanh"), Inputs(m_svd_u));
  auto m_tanh_v = NodeWith(Op("Tanh"), Inputs(m_svd_v_with_jitter));

  Node* tanh_s_transformed = testing::FindNodeByName(graph.get(), "tanh_s");
  ASSERT_NE(tanh_s_transformed, nullptr);

  Node* tanh_u_transformed = testing::FindNodeByName(graph.get(), "tanh_u");
  ASSERT_NE(tanh_u_transformed, nullptr);

  Node* tanh_v_transformed = testing::FindNodeByName(graph.get(), "tanh_v");
  ASSERT_NE(tanh_v_transformed, nullptr);

  EXPECT_THAT(tanh_s_transformed, m_tanh_s);
  EXPECT_THAT(tanh_u_transformed, m_tanh_u);
  EXPECT_THAT(tanh_v_transformed, m_tanh_v);
}
}  // namespace
}  // namespace tensorflow

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

#include "tensorflow/core/grappler/optimizers/model_pruner.h"

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/no_op.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kDeviceCPU0[] = "/device:CPU:0";
constexpr char kDeviceGPU0[] = "/device:GPU:0";

class ModelPrunerTest : public GrapplerTest {};

TEST_F(ModelPrunerTest, NoPruning) {
  // This trivial graph is so basic there's nothing to prune.
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  ModelPruner pruner;
  GraphDef output;
  TF_ASSERT_OK(pruner.Optimize(nullptr, item, &output));

  CompareGraphs(item.graph, output);
}

TEST_F(ModelPrunerTest, StopGradientPruning) {
  // Build a simple graph with a few trivially prunable ops.
  GrapplerItem item;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
    Output b = ops::Sqrt(s.WithOpName("b"), {a});
    Output c = ops::StopGradient(s.WithOpName("c"), b);
    Output d = ops::StopGradient(s.WithOpName("d"), c);
    Output e = ops::Sqrt(s.WithOpName("e"), {d});

    TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  }

  ModelPruner pruner;
  GraphDef output;
  TF_ASSERT_OK(pruner.Optimize(nullptr, item, &output));

  GraphDef expected;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
    Output b = ops::Sqrt(s.WithOpName("b"), {a});
    Output c = ops::StopGradient(s.WithOpName("c"), b);
    Output d = ops::StopGradient(s.WithOpName("d"), b);
    Output e = ops::Sqrt(s.WithOpName("e"), {b});

    TF_ASSERT_OK(s.ToGraphDef(&expected));
  }

  CompareGraphs(expected, output);

  std::vector<string> fetch = {"e"};
  auto expected_tensors = EvaluateNodes(item.graph, fetch);
  auto actual_tensors = EvaluateNodes(output, fetch);
  ASSERT_EQ(expected_tensors.size(), 1);
  ASSERT_EQ(actual_tensors.size(), 1);
  test::ExpectTensorEqual<float>(actual_tensors[0], expected_tensors[0]);
}

TEST_F(ModelPrunerTest, IdentityPruning) {
  // Build a simple graph with a few trivially prunable ops.
  GrapplerItem item;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
    Output b = ops::Sqrt(s.WithOpName("b"), {a});
    Output c = ops::Identity(s.WithOpName("c").WithControlDependencies(b), b);
    Output d = ops::Identity(s.WithOpName("d"), c);
    Output e = ops::Sqrt(s.WithOpName("e"), {d});

    TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  }
  item.fetch.push_back("e");

  ModelPruner pruner;
  GraphDef output;
  TF_ASSERT_OK(pruner.Optimize(nullptr, item, &output));

  GraphDef expected;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
    Output b = ops::Sqrt(s.WithOpName("b"), {a});
    Output e = ops::Sqrt(s.WithOpName("e"), {b});

    TF_ASSERT_OK(s.ToGraphDef(&expected));
  }

  CompareGraphs(expected, output);

  auto actual_tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(actual_tensors.size(), 1);
  auto expected_tensors = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(expected_tensors.size(), 1);
  test::ExpectTensorEqual<float>(actual_tensors[0], expected_tensors[0]);
}

TEST_F(ModelPrunerTest, IdentityNInputPruning) {
  GrapplerItem item;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    Output a = ops::Const(s.WithOpName("a"), 2.0f, {10, 10});
    Output b = ops::Sqrt(s.WithOpName("b"), {a});
    // Node "c" is pruned along with fanins of node "c".
    Output c = ops::Const(s.WithOpName("c"), 3.0f, {10, 10});
    // Node "d" will be pruned because it only has control outputs.
    Output d = ops::Const(s.WithOpName("d"), 4.0f, {10, 10});
    auto e =
        ops::IdentityN(s.WithOpName("e").WithControlDependencies(d), {a, b, c});
    auto f = ops::IdentityN(s.WithOpName("f"), {e[2], e[1], e[0]});
    Output g = ops::Sqrt(s.WithOpName("g"), {f[1]});
    Output h = ops::Sqrt(s.WithOpName("h"), {f[2]});

    TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  }

  item.fetch = {"g", "h"};
  ModelPruner pruner;
  GraphDef output;
  TF_ASSERT_OK(pruner.Optimize(nullptr, item, &output));

  GraphDef expected;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    Output a = ops::Const(s.WithOpName("a"), 2.0f, {10, 10});
    Output b = ops::Sqrt(s.WithOpName("b"), {a});
    auto e = ops::IdentityN(s.WithOpName("e"), {a, b});
    auto f = ops::IdentityN(s.WithOpName("f"), {e[1], e[0]});
    Output g = ops::Sqrt(s.WithOpName("g"), {f[0]});
    Output h = ops::Sqrt(s.WithOpName("h"), {f[1]});

    TF_ASSERT_OK(s.ToGraphDef(&expected));
  }

  CompareGraphs(expected, output);

  auto actual_tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(actual_tensors.size(), 2);
  auto expected_tensors = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(expected_tensors.size(), 2);
  for (int i = 0; i < actual_tensors.size(); i++) {
    test::ExpectTensorEqual<float>(actual_tensors[i], expected_tensors[i]);
  }
}

TEST_F(ModelPrunerTest, IdentityNInputPruningWithIdentityNInFetch) {
  GrapplerItem item;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    Output a = ops::Const(s.WithOpName("a"), 2.0f, {10, 10});
    Output b = ops::Sqrt(s.WithOpName("b"), {a});
    Output c = ops::Const(s.WithOpName("c"), 3.0f, {10, 10});
    // d will be pruned because it only has control outputs.
    Output d = ops::Const(s.WithOpName("d"), 4.0f, {10, 10});
    auto e =
        ops::IdentityN(s.WithOpName("e").WithControlDependencies(d), {a, b, c});
    auto f = ops::IdentityN(s.WithOpName("f"), {e[0], e[1], e[2]});
    auto g = ops::IdentityN(s.WithOpName("g"), {f[1]});

    TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  }

  item.fetch = {"g"};
  ModelPruner pruner;
  GraphDef output;
  TF_ASSERT_OK(pruner.Optimize(nullptr, item, &output));

  GraphDef expected;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    Output a = ops::Const(s.WithOpName("a"), 2.0f, {10, 10});
    Output b = ops::Sqrt(s.WithOpName("b"), {a});
    auto e = ops::IdentityN(s.WithOpName("e"), {b});
    // Single output IdentityN (node "f") was pruned.
    auto g = ops::IdentityN(s.WithOpName("g"), {e[0]});

    TF_ASSERT_OK(s.ToGraphDef(&expected));
  }

  CompareGraphs(expected, output);

  auto actual_tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(actual_tensors.size(), 1);
  auto expected_tensors = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(expected_tensors.size(), 1);
  test::ExpectTensorEqual<float>(actual_tensors[0], expected_tensors[0]);
}

TEST_F(ModelPrunerTest, NoOpPruning) {
  // Build a simple graph with a few trivially prunable ops.
  GrapplerItem item;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
    Output b = ops::AddN(s.WithOpName("b"), {a});
    Output c = ops::AddN(s.WithOpName("c"), {b});
    Output d = ops::AddN(s.WithOpName("d").WithControlDependencies(b), {c});
    Output e = ops::AddN(s.WithOpName("e"), {d});

    TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  }

  ModelPruner pruner;
  GraphDef output;
  TF_ASSERT_OK(pruner.Optimize(nullptr, item, &output));

  GraphDef expected;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
    Output b = ops::AddN(s.WithOpName("b"), {a});
    Output c = ops::AddN(s.WithOpName("c"), {a});
    Output d = ops::AddN(s.WithOpName("d"), {a});
    Output e = ops::AddN(s.WithOpName("e"), {a});

    TF_ASSERT_OK(s.ToGraphDef(&expected));
  }

  CompareGraphs(expected, output);

  std::vector<string> fetch = {"e"};
  auto actual_tensors = EvaluateNodes(output, fetch);
  ASSERT_EQ(actual_tensors.size(), 1);
  auto expected_tensors = EvaluateNodes(item.graph, fetch);
  ASSERT_EQ(expected_tensors.size(), 1);
  test::ExpectTensorEqual<float>(actual_tensors[0], expected_tensors[0]);
}

TEST_F(ModelPrunerTest, PreserveIdentities) {
  GrapplerItem item;
  {
    tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

    ops::Variable v_in(scope.WithOpName("v_in"), {3}, DT_FLOAT);
    ops::Variable v_ctrl(scope.WithOpName("v_ctrl"), {}, DT_BOOL);
    ops::Switch s(scope.WithOpName("switch"), v_in, v_ctrl);
    // id0 is preserved because it is fed by a Switch and drives a control
    // dependency.
    Output id0 = ops::Identity(scope.WithOpName("id0"), s.output_true);
    // id1 is preserved because it feeds a Merge.
    Output id1 =
        ops::Identity(scope.WithOpName("id1").WithControlDependencies(v_ctrl),
                      s.output_false);
    Output id2 = ops::Identity(scope.WithOpName("id2"), id0);
    Output id3 = ops::Identity(
        scope.WithOpName("id3").WithControlDependencies(id0), id1);
    auto merge = ops::Merge(scope.WithOpName("merge"), {id0, id1});

    TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  }

  item.fetch = {"id2", "id3", "merge"};
  ModelPruner pruner;
  GraphDef output;
  TF_ASSERT_OK(pruner.Optimize(nullptr, item, &output));

  CompareGraphs(item.graph, output);

  auto v_in_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3}));
  Tensor v_ctrl_t(DT_BOOL, TensorShape({}));
  v_ctrl_t.flat<bool>()(0) = true;
  auto actual_tensors = EvaluateNodes(output, {"merge", "id2"},
                                      {{"v_in", v_in_t}, {"v_ctrl", v_ctrl_t}});
  ASSERT_EQ(actual_tensors.size(), 2);
  auto expected_tensors = EvaluateNodes(
      item.graph, {"merge", "id2"}, {{"v_in", v_in_t}, {"v_ctrl", v_ctrl_t}});
  ASSERT_EQ(expected_tensors.size(), 2);
  for (int i = 0; i < actual_tensors.size(); i++) {
    test::ExpectTensorEqual<float>(actual_tensors[i], expected_tensors[i]);
  }
}

TEST_F(ModelPrunerTest, PruningSkipsRefOutputs) {
  GrapplerItem item;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    // Make graph of Identity(Identity(Identity(Identity(Variable)))).
    Output a = ops::Variable(s.WithOpName("a"), {}, DT_INT64);
    Output b = ops::Identity(s.WithOpName("b"), a);
    Output c = ops::Identity(s.WithOpName("c"), b);
    Output d = ops::Identity(s.WithOpName("d"), c);
    Output e = ops::Identity(s.WithOpName("e"), d);

    TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  }

  ModelPruner pruner;
  GraphDef output;
  TF_ASSERT_OK(pruner.Optimize(nullptr, item, &output));

  GraphDef expected;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    Output a = ops::Variable(s.WithOpName("a"), {}, DT_INT64);
    Output b = ops::Identity(s.WithOpName("b"), a);
    Output c = ops::Identity(s.WithOpName("c"), b);
    Output d = ops::Identity(s.WithOpName("d"), b);
    Output e = ops::Identity(s.WithOpName("e"), b);

    TF_ASSERT_OK(s.ToGraphDef(&expected));
  }

  CompareGraphs(expected, output);

  std::vector<string> fetch = {"e"};
  auto a_t = GenerateRandomTensor<DT_INT64>(TensorShape({}));
  auto actual_tensors = EvaluateNodes(output, fetch, {{"a", a_t}});
  ASSERT_EQ(actual_tensors.size(), 1);
  auto expected_tensors = EvaluateNodes(item.graph, fetch, {{"a", a_t}});
  ASSERT_EQ(expected_tensors.size(), 1);
  test::ExpectTensorEqual<int64_t>(actual_tensors[0], expected_tensors[0]);
}

// TODO(rmlarsen): Reenable this test when the issues with
// //robotics/learning/sensor_predict:utils_multi_sensor_rnn_test
// have been resolved.
/*
TEST_F(ModelPrunerTest, PruningForwardsCtrlDependencies) {
  // Build a simple graph with a few trivially prunable ops.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
  Output b = ops::Sqrt(s.WithOpName("b"), {a});
  Output c = ops::Sqrt(s.WithOpName("c"), {a});
  Output d = ops::Identity(s.WithOpName("d").WithControlDependencies(b), c);
  Output e = ops::Identity(s.WithOpName("e").WithControlDependencies(c), d);
  Output f = ops::Sqrt(s.WithOpName("f"), {d});
  Output g = ops::Sqrt(s.WithOpName("g"), {e});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch.push_back("f");
  item.fetch.push_back("g");

  ModelPruner pruner;
  GraphDef output;
  Status status = pruner.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  LOG(INFO) << "After: " << output.DebugString();

  EXPECT_EQ(5, output.node_size());
  for (const auto& new_node : output.node()) {
    // "d" and "e" should be removed.
    EXPECT_NE("d", new_node.name());
    EXPECT_NE("e", new_node.name());
    if (new_node.name() == "g") {
      EXPECT_EQ(2, new_node.input_size());
      // The input from switch should be forwarded to id3.
      EXPECT_EQ("c", new_node.input(0));
      EXPECT_EQ("^b", new_node.input(1));
    }
    if (new_node.name() == "f") {
      EXPECT_EQ(2, new_node.input_size());
      // The input from switch should be forwarded to id3.
      EXPECT_EQ("c", new_node.input(0));
      EXPECT_EQ("^b", new_node.input(1));
    }
  }
}
*/

TEST_F(ModelPrunerTest, PruningPreservesFetch) {
  // Build a simple graph with a few trivially prunable ops.
  GrapplerItem item;
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
    Output b = ops::Sqrt(s.WithOpName("b"), {a});
    Output c = ops::Identity(s.WithOpName("c"), b);
    Output d = ops::Identity(s.WithOpName("d"), c);

    TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  }

  item.fetch = {"c"};
  ModelPruner pruner;
  GraphDef output;
  TF_ASSERT_OK(pruner.Optimize(nullptr, item, &output));

  GraphDef expected;
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
    Output b = ops::Sqrt(s.WithOpName("b"), {a});
    Output c = ops::Identity(s.WithOpName("c"), b);

    TF_ASSERT_OK(s.ToGraphDef(&expected));
  }

  CompareGraphs(expected, output);

  auto actual_tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(actual_tensors.size(), 1);
  auto expected_tensors = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(expected_tensors.size(), 1);
  test::ExpectTensorEqual<float>(actual_tensors[0], expected_tensors[0]);
}

TEST_F(ModelPrunerTest, PruningPreservesCrossDeviceIdentity) {
  GrapplerItem item;
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    Output c =
        ops::Const(s.WithOpName("c").WithDevice(kDeviceCPU0), 0.0f, {10, 10});

    // Node i1 should be preserved.
    Output i1 = ops::Identity(s.WithOpName("i1").WithDevice(kDeviceGPU0), c);
    Output a1 = ops::Identity(s.WithOpName("a1").WithDevice(kDeviceGPU0), i1);
    Output a2 = ops::Identity(s.WithOpName("a2").WithDevice(kDeviceGPU0), i1);

    // Node i2 should be pruned since it resides on the sender's device.
    Output i2 = ops::Identity(s.WithOpName("i2").WithDevice(kDeviceCPU0), c);
    Output a3 = ops::Identity(s.WithOpName("a3").WithDevice(kDeviceGPU0), i2);
    Output a4 = ops::Identity(s.WithOpName("a4").WithDevice(kDeviceGPU0), i2);

    TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  }

  item.fetch = {"a1", "a2", "a3", "a4"};
  ModelPruner pruner;
  GraphDef output;
  TF_ASSERT_OK(pruner.Optimize(nullptr, item, &output));

  GraphDef expected;
  {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    Output c =
        ops::Const(s.WithOpName("c").WithDevice(kDeviceCPU0), 0.0f, {10, 10});

    // Node i1 should be preserved.
    Output i1 = ops::Identity(s.WithOpName("i1").WithDevice(kDeviceGPU0), c);
    Output a1 = ops::Identity(s.WithOpName("a1").WithDevice(kDeviceGPU0), i1);
    Output a2 = ops::Identity(s.WithOpName("a2").WithDevice(kDeviceGPU0), i1);

    // Node i2 should be pruned since it resides on the sender's device.
    Output a3 = ops::Identity(s.WithOpName("a3").WithDevice(kDeviceGPU0), c);
    Output a4 = ops::Identity(s.WithOpName("a4").WithDevice(kDeviceGPU0), c);

    TF_ASSERT_OK(s.ToGraphDef(&expected));
  }

  CompareGraphs(expected, output);

  if (GetNumAvailableGPUs() > 0) {
    auto actual_tensors = EvaluateNodes(output, item.fetch);
    ASSERT_EQ(actual_tensors.size(), 4);
    auto expected_tensors = EvaluateNodes(item.graph, item.fetch);
    ASSERT_EQ(expected_tensors.size(), 4);
    for (int i = 0; i < actual_tensors.size(); i++) {
      test::ExpectTensorNear<float>(actual_tensors[i], expected_tensors[i],
                                    1e-6);
    }
  }
}

TEST_F(ModelPrunerTest, PruneNoOpsWithoutInputs) {
  GrapplerItem item;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    // Add an explicit no-op node without inputs. It should be pruned even
    // though it has a path to the fetch node.
    auto n1 = ops::NoOp(s.WithOpName("no_op1"));
    Output c1 = ops::Const(s.WithOpName("c1"), 0.0f, {1, 1});
    // Add an explicit no-op node with a control input. It should not be pruned.
    auto n2 = ops::NoOp(s.WithOpName("no_op2").WithControlDependencies(c1));
    // Add NoOps as control inputs to fetch node.
    Output id1 = ops::Identity(
        s.WithOpName("id1").WithControlDependencies({n1, n2}), c1);

    TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  }

  item.fetch = {"id1"};
  ModelPruner pruner;
  GraphDef output;
  TF_ASSERT_OK(pruner.Optimize(nullptr, item, &output));

  GraphDef expected;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    Output c1 = ops::Const(s.WithOpName("c1"), 0.0f, {1, 1});
    auto n2 = ops::NoOp(s.WithOpName("no_op2").WithControlDependencies(c1));
    Output id1 =
        ops::Identity(s.WithOpName("id1").WithControlDependencies({n2}), c1);

    TF_ASSERT_OK(s.ToGraphDef(&expected));
  }

  CompareGraphs(expected, output);
}

TEST_F(ModelPrunerTest, PruneConstantsWithoutInputsAndOutputs) {
  GrapplerItem item;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    // c0 has an non-control output => will NOT be pruned.
    Output c0 = ops::Const(s.WithOpName("c0"), 0.0f, {1, 1});
    // c1 has neither inputs nor outputs => will be pruned.
    Output c1 = ops::Const(s.WithOpName("c1"), 1.0f, {1, 1});
    // c2 has a control input and a control output  => will NOT be pruned.
    Output c2 = ops::Const(s.WithOpName("c2").WithControlDependencies({c0}),
                           2.0f, {1, 1});
    // c3 has no inputs and one control output  => will be pruned.
    Output c3 = ops::Const(s.WithOpName("c3"), 3.0f, {1, 1});
    Output id1 = ops::Identity(s.WithOpName("id1")
                                   .WithControlDependencies({c2})
                                   .WithControlDependencies({c3}),
                               c0);

    TF_ASSERT_OK(s.ToGraphDef(&item.graph));
  }

  item.fetch = {"id1"};
  ModelPruner pruner;
  GraphDef output;
  absl::Status status = pruner.Optimize(nullptr, item, &output);
  TF_ASSERT_OK(status);

  GraphDef expected;
  {
    tensorflow::Scope s = CreateScopeWithDevice(kDeviceCPU0);

    // c0 has an non-control output => will NOT be pruned.
    Output c0 = ops::Const(s.WithOpName("c0"), 0.0f, {1, 1});
    // c2 has a control input and a control output  => will NOT be pruned.
    Output c2 = ops::Const(s.WithOpName("c2").WithControlDependencies({c0}),
                           2.0f, {1, 1});
    Output id1 =
        ops::Identity(s.WithOpName("id1").WithControlDependencies({c2}), c0);

    TF_ASSERT_OK(s.ToGraphDef(&expected));
  }

  CompareGraphs(expected, output);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

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

#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {
namespace {

class ConstantFoldingTest : public ::testing::Test {
 protected:
  std::vector<Tensor> EvaluateNodes(const GraphDef& graph,
                                    const std::vector<string>& fetch) {
    SessionOptions options;
    std::unique_ptr<tensorflow::Session> session(NewSession(options));
    TF_CHECK_OK(session->Create(graph));
    RunOptions run_options;
    std::vector<Tensor> output_tensors;
    TF_CHECK_OK(
        session->Run(run_options, {}, fetch, fetch, &output_tensors, nullptr));
    TF_CHECK_OK(session->Close());
    return output_tensors;
  }
};

TEST_F(ConstantFoldingTest, SimpleFolding) {
  // Build a simple graph with a few trivially prunable ops.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Const(s.WithOpName("a"), 1.0f, {1});
  Output b = ops::Const(s.WithOpName("b"), 2.0f, {1});
  Output c = ops::AddN(s.WithOpName("c"), {a, b});
  Output d = ops::AddN(s.WithOpName("d"), {b, c});

  GrapplerItem item;
  item.fetch.push_back("d");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding fold;
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(5, output.node_size());

  const NodeDef& new_c = output.node(0);
  EXPECT_EQ("ConstantFolding/c", new_c.name());
  EXPECT_EQ("Const", new_c.op());

  const NodeDef& new_a = output.node(1);
  EXPECT_EQ("a", new_a.name());

  const NodeDef& new_b = output.node(2);
  EXPECT_EQ("b", new_b.name());

  const NodeDef& old_c = output.node(3);
  EXPECT_EQ("c", old_c.name());

  const NodeDef& new_d = output.node(4);
  EXPECT_EQ("d", new_d.name());
  EXPECT_EQ("ConstantFolding/c", new_d.input(1));

  std::vector<string> fetch = {"a", "b", "c", "d"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(4, tensors_expected.size());
  EXPECT_EQ(4, tensors.size());
  for (int i = 0; i < 4; i++) {
    test::ExpectTensorEqual<float>(tensors_expected[i], tensors[i]);
  }
}

TEST_F(ConstantFoldingTest, FoldingNodeWithTwoOutputs) {
  // Build a simple graph with a few trivially prunable ops.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Const(s.WithOpName("a"), 10, {3});
  auto b = ops::Unique(s.WithOpName("b"), {a});
  Output c = ops::Identity(s.WithOpName("c"), {b.y});
  Output d = ops::Identity(s.WithOpName("d"), {b.idx});

  GrapplerItem item;
  item.fetch.push_back("c");
  item.fetch.push_back("d");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding fold;
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(6, output.node_size());

  const NodeDef& new_b_0 = output.node(0);
  EXPECT_EQ("ConstantFolding/b-0", new_b_0.name());
  EXPECT_EQ("Const", new_b_0.op());

  const NodeDef& new_b_1 = output.node(1);
  EXPECT_EQ("ConstantFolding/b-1", new_b_1.name());
  EXPECT_EQ("Const", new_b_1.op());

  const NodeDef& new_a = output.node(2);
  EXPECT_EQ("a", new_a.name());

  const NodeDef& new_b = output.node(3);
  EXPECT_EQ("b", new_b.name());

  const NodeDef& new_c = output.node(4);
  EXPECT_EQ("c", new_c.name());
  EXPECT_EQ("ConstantFolding/b-0", new_c.input(0));

  const NodeDef& new_d = output.node(5);
  EXPECT_EQ("d", new_d.name());
  EXPECT_EQ("ConstantFolding/b-1", new_d.input(0));

  std::vector<string> fetch = {"a", "b", "c", "d"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(4, tensors_expected.size());
  EXPECT_EQ(4, tensors.size());
  for (int i = 0; i < 4; i++) {
    test::ExpectTensorEqual<int>(tensors_expected[i], tensors[i]);
  }
}

TEST_F(ConstantFoldingTest, ControlDependencies) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output dflt = ops::Const(scope.WithOpName("dflt"), 3.14f, {1});
  Output p1 = ops::PlaceholderWithDefault(scope.WithOpName("p1"), dflt, {1});
  Output p2 = ops::PlaceholderWithDefault(scope.WithOpName("p2"), dflt, {1});
  Output c = ops::Const(scope.WithOpName("c"), 10, {3});
  Output i1 = ops::Identity(scope.WithOpName("i1"), {c});
  Output i2 = ops::Identity(scope.WithOpName("i2"), {i1});
  Output i3 = ops::Identity(scope.WithOpName("e"), {i2});

  GrapplerItem item;
  item.fetch.push_back("i3");
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  ASSERT_EQ("c", item.graph.node(3).name());
  (*item.graph.mutable_node(3)->add_input()) = "^p1";
  ASSERT_EQ("i2", item.graph.node(5).name());
  (*item.graph.mutable_node(5)->add_input()) = "^p2";

  ConstantFolding fold;
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "ConstantFolding/i1") {
      ++found;
      auto folded = EvaluateNodes(output, {"ConstantFolding/i1"});
      auto expected = EvaluateNodes(item.graph, {"i1"});
      EXPECT_EQ(1, expected.size());
      EXPECT_EQ(1, folded.size());
      test::ExpectTensorEqual<int>(folded[0], expected[0]);
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^p1", node.input(0));
    }
    if (node.name() == "ConstantFolding/i2") {
      ++found;
      auto folded = EvaluateNodes(output, {"ConstantFolding/i2"});
      auto expected = EvaluateNodes(item.graph, {"i2"});
      EXPECT_EQ(1, expected.size());
      EXPECT_EQ(1, folded.size());
      test::ExpectTensorEqual<int>(folded[0], expected[0]);
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("^p1", node.input(0));
      EXPECT_EQ("^p2", node.input(1));
    }
  }
  EXPECT_EQ(2, found);
}

TEST_F(ConstantFoldingTest, VariableNumberOfOutputs) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  // Add a DynamicPartition node to the graph
  Output input = ops::Const(scope.WithOpName("in0"), 314, {3, 4, 5});
  Output indices = ops::Const(scope.WithOpName("indices"), 1, {3, 4});
  int num_partitions = 4;
  ops::DynamicPartition part(scope.WithOpName("partition"), input, indices,
                             num_partitions);

  std::vector<string> outputs;
  for (int i = 0; i < num_partitions; ++i) {
    string part_out_name = strings::StrCat("part_out", i);
    ops::Identity partition_out(scope.WithOpName(part_out_name),
                                {part.outputs[i]});
    outputs.push_back(part_out_name);
  }

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  // Add a ConcatOffset node to the graph
  Tensor initial_val(DT_INT32, TensorShape({3}));
  test::FillIota<int>(&initial_val, 7);
  for (int i = 1; i < 5; ++i) {
    TF_CHECK_OK(NodeDefBuilder(strings::StrCat("in", i), "Const")
                    .Attr("dtype", DT_INT32)
                    .Attr("value", initial_val)
                    .Finalize(item.graph.add_node()));
  }
  Tensor concat_dim(DT_INT32, TensorShape({}));
  test::FillIota<int>(&concat_dim, 0);
  TF_CHECK_OK(NodeDefBuilder("concat_dim", "Const")
                  .Attr("dtype", DT_INT32)
                  .Attr("value", concat_dim)
                  .Finalize(item.graph.add_node()));

  TF_CHECK_OK(NodeDefBuilder("concat_offsets", "ConcatOffset")
                  .Input("concat_dim", 0, DT_INT32)
                  .Input({NodeDefBuilder::NodeOut("in1", 0, DT_INT32),
                          NodeDefBuilder::NodeOut("in2", 0, DT_INT32),
                          NodeDefBuilder::NodeOut("in3", 0, DT_INT32),
                          NodeDefBuilder::NodeOut("in4", 0, DT_INT32)})
                  .Finalize(item.graph.add_node()));

  for (int i = 0; i < 4; ++i) {
    string concat_offset_out_name = strings::StrCat("concat_offset_out", i);
    TF_CHECK_OK(NodeDefBuilder(concat_offset_out_name, "Identity")
                    .Attr("T", DT_INT32)
                    .Input("concat_offsets", i, DT_INT32)
                    .Finalize(item.graph.add_node()));
    outputs.push_back(concat_offset_out_name);
  }

  item.fetch = outputs;
  ConstantFolding fold;
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int constant_folded = 0;
  for (const auto& node : output.node()) {
    if (node.name().find("ConstantFolding/partition") != string::npos ||
        node.name().find("ConstantFolding/concat_offsets") != string::npos) {
      ++constant_folded;
      EXPECT_EQ("Const", node.op());
    }
  }
  EXPECT_EQ(8, constant_folded);

  auto expected = EvaluateNodes(item.graph, outputs);
  auto optimized = EvaluateNodes(output, outputs);
  ASSERT_EQ(expected.size(), optimized.size());
  for (int i = 0; i < expected.size(); ++i) {
    test::ExpectTensorEqual<int>(expected[i], optimized[i]);
  }
}

TEST_F(ConstantFoldingTest, ShapeMaterialization) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output v1 = ops::Variable(scope.WithOpName("v1"), {3}, DT_FLOAT);
  Output v2 = ops::Variable(scope.WithOpName("v2"), {5, 7}, DT_FLOAT);
  Output v3 = ops::Variable(scope.WithOpName("v3"), {11, 13}, DT_FLOAT);
  Output rank = ops::Rank(scope.WithOpName("rank"), v1);
  Output shape = ops::Shape(scope.WithOpName("shape"), v2);
  Output size = ops::Size(scope.WithOpName("size"), v3);
  Output p1 = ops::Multiply(scope.WithOpName("p1"), size, rank);
  Output p2 = ops::Multiply(scope.WithOpName("p2"), p1, shape);

  GrapplerItem item;
  item.fetch.push_back("p2");
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold;
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "size") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^v3", node.input(0));
      Tensor value;
      CHECK(value.FromProto(node.attr().at("value").tensor()));
      EXPECT_EQ(11 * 13, value.flat<int>()(0));
    } else if (node.name() == "rank") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^v1", node.input(0));
      Tensor value;
      CHECK(value.FromProto(node.attr().at("value").tensor()));
      EXPECT_EQ(1, value.flat<int>()(0));
    } else if (node.name() == "shape") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^v2", node.input(0));
      Tensor value;
      CHECK(value.FromProto(node.attr().at("value").tensor()));
      EXPECT_EQ(5, value.flat<int>()(0));
      EXPECT_EQ(7, value.flat<int>()(1));
    }
  }
  EXPECT_EQ(3, found);
}

TEST_F(ConstantFoldingTest, SwitchNodes) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  ops::Variable v_in(scope.WithOpName("v_in"), {3}, DT_FLOAT);
  ops::Variable v_ctrl(scope.WithOpName("v_ctrl"), {}, DT_BOOL);
  ops::Switch s1(scope.WithOpName("switch"), v_in, v_ctrl);
  ops::Rank rank(scope.WithOpName("rank"), s1.output_false);
  ops::Identity i(scope.WithOpName("i"), s1.output_true);
  ops::Size size(scope.WithOpName("size"), i);
  ops::Square p1(scope.WithOpName("p1"), rank);
  ops::Square p2(scope.WithOpName("p2"), size);
  ops::Merge m(scope.WithOpName("m"), {p1.y, p2.y});

  Output predicate =
      ops::Const(scope.WithOpName("false"), false, TensorShape({}));
  Output constant =
      ops::Const(scope.WithOpName("constant"), 1.0f, TensorShape({1}));
  ops::Switch s2(scope.WithOpName("switch2"), constant, predicate);
  ops::Identity statically_known(scope.WithOpName("i2"), s2.output_false);
  ops::Identity never_generated(scope.WithOpName("i3"), s2.output_true);
  ops::Merge m2(scope.WithOpName("m2"),
                {statically_known.output, never_generated.output});

  GrapplerItem item;
  item.fetch.push_back("m");
  item.fetch.push_back("m2");

  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold;
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  for (const auto& node : output.node()) {
    if (node.name() == "rank") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^ConstantFoldingCtrl/switch_0", node.input(0));
    }
    if (node.name() == "size") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^i", node.input(0));
    }
    if (node.name() == "ConstantFolding/switch2-0") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(0, node.input_size());
    }
    if (node.name() == "ConstantFolding/i2") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(0, node.input_size());
    }
    if (node.name() == "i3") {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("switch2:1", node.input(0));
    }
  }
}

TEST_F(ConstantFoldingTest, NoOpReduction) {
  // Build a simple graph with a reduction that can be reduced to the identity.
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output d = ops::Const(scope.WithOpName("d"), 3.14f, {3, 5, 7});
  Output v = ops::PlaceholderWithDefault(scope.WithOpName("v"), d, {3, 5, 7});
  Output c = ops::Const(scope.WithOpName("c"), 0, {0});
  Output i = ops::Identity(scope.WithOpName("i"), c);
  Output p = ops::Prod(scope.WithOpName("p"), v, i);
  Output s = ops::Square(scope.WithOpName("s"), p);

  GrapplerItem item;
  item.fetch.push_back("s");
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  ASSERT_EQ("c", item.graph.node(2).name());
  (*item.graph.mutable_node(2)->add_input()) = "^v";

  ConstantFolding fold;
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  auto expected = EvaluateNodes(item.graph, {"s"});
  auto optimized = EvaluateNodes(output, {"s"});
  EXPECT_EQ(1, expected.size());
  EXPECT_EQ(1, optimized.size());
  test::ExpectTensorEqual<float>(expected[0], optimized[0]);

  bool found = false;
  for (const auto& node : output.node()) {
    if (node.name() == "p") {
      found = true;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("v", node.input(0));
      EXPECT_EQ("^v", node.input(1));
    }
  }
  EXPECT_TRUE(found);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

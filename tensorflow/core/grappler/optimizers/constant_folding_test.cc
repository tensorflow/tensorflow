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
#include "tensorflow/cc/ops/array_ops_internal.h"
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
  Output c = ops::AddN(s.WithOpName("c").WithDevice("/CPU:0"), {a, b});
  Output d = ops::AddN(s.WithOpName("d"), {b, c});

  GrapplerItem item;
  item.fetch.push_back("d");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(1, output.node_size());

  const NodeDef& node_d = output.node(0);
  EXPECT_EQ("d", node_d.name());
  EXPECT_EQ("Const", node_d.op());

  std::vector<string> fetch = {"d"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors_expected.size());
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(ConstantFoldingTest, FoldingNodeWithTwoOutputs) {
  // Build a simple graph with a few trivially prunable ops.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Const(s.WithOpName("a"), 10, {3});
  auto b = ops::Unique(s.WithOpName("b"), {a});
  Output c = ops::Identity(s.WithOpName("c"), {b.y});
  Output d = ops::Identity(s.WithOpName("d"), {b.idx});
  Output e = ops::Identity(s.WithOpName("e"), {c});
  Output f = ops::Identity(s.WithOpName("f"), {d});

  GrapplerItem item;
  item.fetch.push_back("e");
  item.fetch.push_back("f");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(2, output.node_size());

  const NodeDef& new_c = output.node(0);
  EXPECT_EQ("e", new_c.name());
  EXPECT_EQ("Const", new_c.op());

  const NodeDef& new_d = output.node(1);
  EXPECT_EQ("f", new_d.name());
  EXPECT_EQ("Const", new_d.op());

  std::vector<string> fetch = {"e", "f"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(fetch.size(), tensors_expected.size());
  EXPECT_EQ(fetch.size(), tensors.size());
  for (int i = 0; i < fetch.size(); i++) {
    test::ExpectTensorEqual<int>(tensors_expected[i], tensors[i]);
  }
}

TEST_F(ConstantFoldingTest, ControlDependencies) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output dflt = ops::Const(scope.WithOpName("dflt"), 3.14f, {1});
  Output p1 = ops::PlaceholderWithDefault(scope.WithOpName("p1"), dflt, {1});
  Output p2 = ops::PlaceholderWithDefault(scope.WithOpName("p2"), dflt, {1});
  Output c =
      ops::Const(scope.WithOpName("c").WithControlDependencies(p1), 10, {3});
  Output i1 = ops::Identity(scope.WithOpName("i1"), {c});
  Output i2 =
      ops::Identity(scope.WithOpName("i2").WithControlDependencies(p2), {i1});
  Output i3 = ops::Identity(scope.WithOpName("e"), {i2});

  GrapplerItem item;
  item.fetch.push_back("e");
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::vector<string> expected_nodes = {"dflt", "p1", "p2", "e"};
  EXPECT_EQ(output.node_size(), expected_nodes.size());
  int i = 0;
  int found = 0;
  for (const auto& node : output.node()) {
    EXPECT_EQ(expected_nodes[i], output.node(i).name());
    i++;
    if (node.name() == "e") {
      EXPECT_EQ("Const", node.op());
      ++found;
      auto folded = EvaluateNodes(output, {"e"});
      auto expected = EvaluateNodes(item.graph, {"e"});
      EXPECT_EQ(1, expected.size());
      EXPECT_EQ(1, folded.size());
      test::ExpectTensorEqual<int>(folded[0], expected[0]);
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("^p1", node.input(0));
      EXPECT_EQ("^p2", node.input(1));
    }
  }
  EXPECT_EQ(1, found);
}

TEST_F(ConstantFoldingTest, ControlDependenciesEmptyFetch) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output dflt = ops::Const(scope.WithOpName("dflt"), 3.14f, {1});
  Output p1 = ops::PlaceholderWithDefault(scope.WithOpName("p1"), dflt, {1});
  Output p2 = ops::PlaceholderWithDefault(scope.WithOpName("p2"), dflt, {1});
  Output c =
      ops::Const(scope.WithOpName("c").WithControlDependencies(p1), 10, {3});
  Output i1 = ops::Identity(scope.WithOpName("i1"), {c});
  Output i2 =
      ops::Identity(scope.WithOpName("i2").WithControlDependencies(p2), {i1});
  Output i3 = ops::Identity(scope.WithOpName("e"), {i2});

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::vector<string> expected_nodes = {"dflt", "p1", "p2", "c",
                                        "i1",   "i2", "e"};
  EXPECT_EQ(output.node_size(), expected_nodes.size());
  int i = 0;
  int found = 0;
  for (const auto& node : output.node()) {
    EXPECT_EQ(expected_nodes[i], output.node(i).name());
    i++;
    if (node.name() == "i1") {
      EXPECT_EQ("Const", node.op());
      ++found;
      auto folded = EvaluateNodes(output, {"i1"});
      auto expected = EvaluateNodes(item.graph, {"i1"});
      EXPECT_EQ(1, expected.size());
      EXPECT_EQ(1, folded.size());
      test::ExpectTensorEqual<int>(folded[0], expected[0]);
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^p1", node.input(0));
    }
    if (node.name() == "i2") {
      EXPECT_EQ("Const", node.op());
      ++found;
      auto folded = EvaluateNodes(output, {"i2"});
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

TEST_F(ConstantFoldingTest, ControlDependenciesDeduplicate) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output dflt = ops::Const(scope.WithOpName("dflt"), 3.14f, {1});
  Output p1 = ops::PlaceholderWithDefault(scope.WithOpName("p1"), dflt, {1});
  Output p2 = ops::PlaceholderWithDefault(scope.WithOpName("p2"), dflt, {1});
  Output c =
      ops::Const(scope.WithOpName("c").WithControlDependencies(p1), 10, {3});
  Output i1 = ops::Identity(scope.WithOpName("i1")
                                .WithControlDependencies(p2)
                                .WithControlDependencies(p1),
                            {c});
  Output i2 = ops::Identity(scope.WithOpName("i2"), {i1});

  GrapplerItem item;
  item.fetch.push_back("i2");
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::vector<string> expected_nodes = {"dflt", "p1", "p2", "i2"};
  EXPECT_EQ(output.node_size(), expected_nodes.size());
  int i = 0;
  for (const auto& node : output.node()) {
    EXPECT_EQ(expected_nodes[i], output.node(i).name());
    i++;
    if (node.name() == "i2") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("^p1", node.input(0));
      EXPECT_EQ("^p2", node.input(1));
    }
  }
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
  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int constant_folded = 0;
  for (const auto& node : output.node()) {
    if (node.name().find("part_out") != string::npos ||
        node.name().find("concat_offset_out") != string::npos) {
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

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "p2") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("^v3", node.input(0));
      EXPECT_EQ("^v1", node.input(1));
      EXPECT_EQ("^v2", node.input(2));
      Tensor value;
      CHECK(value.FromProto(node.attr().at("value").tensor()));
      // rank = 1, shape = (5, 7), size = 143 = 11*13
      // p2 = (715, 1001) = (5*143, 7*143)
      EXPECT_EQ(715, value.flat<int>()(0));
      EXPECT_EQ(1001, value.flat<int>()(1));
    }
  }
  EXPECT_EQ(1, found);
}

TEST_F(ConstantFoldingTest, ShapeMaterializationEmptyFetch) {
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
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
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

TEST_F(ConstantFoldingTest, ShapeMaterializationShapeN) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output v1 = ops::Variable(scope.WithOpName("v1"), {3, -1}, DT_FLOAT);
  Output v2 = ops::Variable(scope.WithOpName("v2"), {}, DT_FLOAT);
  Output v3 = ops::Variable(scope.WithOpName("v3"), {4, 6}, DT_FLOAT);
  auto s = ops::ShapeN(scope.WithOpName("s"), {v1, v2, v3});
  Output i1a = ops::Identity(scope.WithOpName("i1a"), s[0]);
  Output i1b = ops::Identity(scope.WithOpName("i1b"), s[0]);
  Output i2a = ops::Identity(scope.WithOpName("i2a"), s[1]);
  Output i2b = ops::Identity(scope.WithOpName("i2b"), s[1]);
  Output i2c = ops::Identity(scope.WithOpName("i2c"), s[1]);
  Output i3a = ops::Identity(scope.WithOpName("i3a"), s[2]);
  Output i3b = ops::Identity(scope.WithOpName("i3b"), s[2]);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  int found = 0;
  for (const auto& node : output.node()) {
    EXPECT_NE(AddPrefixToNodeName("s-0", kConstantFoldingConst), node.name());
    EXPECT_NE(AddPrefixToNodeName("s-1", kConstantFoldingConst), node.name());
    if (node.name() == "i1a" || node.name() == "i1b") {
      ++found;
      EXPECT_EQ("s", node.input(0));
    }
    if (node.name() == "i2a" || node.name() == "i2b" || node.name() == "i2c") {
      ++found;
      EXPECT_EQ("s:1", node.input(0));
    }
    if (node.name() == "i3a" || node.name() == "i3b") {
      ++found;
      EXPECT_EQ(AddPrefixToNodeName("s-2", kConstantFoldingConst),
                node.input(0));
    }
    if (node.name() == "s") {
      ++found;
      EXPECT_EQ("ShapeN", node.op());
      EXPECT_EQ("v1", node.input(0));
      EXPECT_EQ("v2", node.input(1));
      EXPECT_EQ("v3", node.input(2));
    }
    if (node.name() == AddPrefixToNodeName("s-2", kConstantFoldingConst)) {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ("^s", node.input(0));
      Tensor value;
      CHECK(value.FromProto(node.attr().at("value").tensor()));
      EXPECT_EQ(4, value.flat<int>()(0));
      EXPECT_EQ(6, value.flat<int>()(1));
    }
  }
  EXPECT_EQ(9, found);
}

TEST_F(ConstantFoldingTest, SwitchNodesEmptyFetch) {
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
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::set<string> present_nodes = {"v_in",     "v_ctrl",
                                    "switch",   "i",
                                    "p1",       "p2",
                                    "m",        "false",
                                    "constant", "switch2",
                                    "i2",       "i3",
                                    "m2",       "ConstantFoldingCtrl/switch_0",
                                    "rank",     "size"};
  std::set<string> not_present_nodes = {"ConstantFolding/switch2-0"};
  EXPECT_EQ(present_nodes.size(), output.node_size());
  int found = 0;
  for (const auto& node : output.node()) {
    EXPECT_TRUE(present_nodes.find(node.name()) != present_nodes.end());
    EXPECT_TRUE(not_present_nodes.find(node.name()) == not_present_nodes.end());
    present_nodes.erase(node.name());
    not_present_nodes.erase(node.name());
    if (node.name() == "rank") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^ConstantFoldingCtrl/switch_0", node.input(0));
    }
    if (node.name() == "size") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^i", node.input(0));
    }
    if (node.name() == "i2") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(0, node.input_size());
    }
    if (node.name() == "i3") {
      ++found;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("switch2:1", node.input(0));
    }
  }
  EXPECT_EQ(4, found);
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

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  std::set<string> present_nodes = {"v_in",     "v_ctrl",
                                    "switch",   "i",
                                    "p1",       "p2",
                                    "m",        "false",
                                    "constant", "switch2",
                                    "i2",       "i3",
                                    "m2",       "ConstantFoldingCtrl/switch_0"};
  std::set<string> not_present_nodes = {"rank", "size",
                                        "ConstantFolding/switch2-0"};
  EXPECT_EQ(present_nodes.size(), output.node_size());

  int found = 0;
  for (const auto& node : output.node()) {
    EXPECT_TRUE(present_nodes.find(node.name()) != present_nodes.end());
    EXPECT_TRUE(not_present_nodes.find(node.name()) == not_present_nodes.end());
    present_nodes.erase(node.name());
    not_present_nodes.erase(node.name());
    if (node.name() == "i2") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(0, node.input_size());
    }
    if (node.name() == "i3") {
      ++found;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("switch2:1", node.input(0));
    }
  }
  EXPECT_EQ(2, found);
}

TEST_F(ConstantFoldingTest, MergeNodes) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output x =
      ops::RandomNormal(scope.WithOpName("x"), {3, 5}, DataType::DT_FLOAT);
  Output y =
      ops::RandomNormal(scope.WithOpName("y"), {3, 5}, DataType::DT_FLOAT);
  Output const1 =
      ops::Const(scope.WithOpName("const1").WithControlDependencies(x), 2.7f,
                 TensorShape({3, 5}));
  Output const2 =
      ops::Const(scope.WithOpName("const2"), 3.14f, TensorShape({3, 5}));
  Output const3 =
      ops::Const(scope.WithOpName("const3").WithControlDependencies(x), 3.14f,
                 TensorShape({3, 5}));

  // Create 3 merge nodes: m1 is foldable, m2 and m3 aren't.
  ops::Merge m1(scope.WithOpName("m1"), {x, const1, const2});
  ops::Merge m2(scope.WithOpName("m2"), {const1, const3});
  ops::Merge m3(scope.WithOpName("m3"), {x, y});

  ops::Identity out1(scope.WithOpName("out1"), m1.output);
  ops::Identity idx1(scope.WithOpName("idx1"), m1.value_index);
  ops::Identity out2(scope.WithOpName("out2"), m2.output);
  ops::Identity idx2(scope.WithOpName("idx2"), m2.value_index);
  ops::Identity out3(scope.WithOpName("out3"), m3.output);
  ops::Identity idx3(scope.WithOpName("idx3"), m3.value_index);

  GrapplerItem item;
  item.fetch = {"out1", "idx1", "out2", "idx2", "out3", "idx3"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found_nodes = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "out1") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^m1", node.input(0));
      ++found_nodes;
    } else if (node.name() == "idx1") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^m1", node.input(0));
      ++found_nodes;
    } else if (node.name() == "ConstantFolding/m1") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^m1", node.input(0));
      ++found_nodes;
    } else if (node.name() == "ConstantFolding/m1_index") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^m1", node.input(0));
      ++found_nodes;
    } else if (node.name() == "out2") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("m2", node.input(0));
      ++found_nodes;
    } else if (node.name() == "idx2") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("m2:1", node.input(0));
      ++found_nodes;
    } else if (node.name() == "out3") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("m3", node.input(0));
      ++found_nodes;
    } else if (node.name() == "idx3") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("m3:1", node.input(0));
      ++found_nodes;
    }
  }
  // Make sure the graph contains all the nodes we're expecting.
  EXPECT_EQ(6, found_nodes);

  std::vector<string> fetch = {"out1", "idx1"};
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(2, tensors.size());
  const Tensor& out_value = tensors[0];
  EXPECT_EQ(3 * 5, out_value.NumElements());
  for (int i = 0; i < 3 * 5; ++i) {
    EXPECT_EQ(3.14f, out_value.flat<float>()(i));
  }
  const Tensor& out_idx = tensors[1];
  EXPECT_EQ(1, out_idx.NumElements());
  EXPECT_EQ(2, out_idx.flat<int32>()(0));
}

TEST_F(ConstantFoldingTest, NoOpReduction) {
  // Build a simple graph with a reduction that can be reduced to the identity.
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output v = ops::Variable(scope.WithOpName("v"), {3, 5, 7}, DT_FLOAT);
  Output c =
      ops::Const(scope.WithOpName("c").WithControlDependencies(v), 0, {0});
  Output i = ops::Identity(scope.WithOpName("i"), c);
  Output p = ops::Prod(scope.WithOpName("p"), v, i);
  Output s = ops::Square(scope.WithOpName("s"), p);

  GrapplerItem item;
  item.fetch.push_back("s");
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

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

TEST_F(ConstantFoldingTest, NoOpReshape) {
  // Build a simple graph with a reshape that can be reduced to the identity.
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  // A reshape than can be optimized
  Output d1 = ops::Const(scope.WithOpName("d1"), 3.14f, {17});
  Output v1 = ops::Variable(scope.WithOpName("v1"), {17}, DT_FLOAT);
  Output c1 =
      ops::Const(scope.WithOpName("c1").WithControlDependencies(v1), 17, {1});
  Output i1 = ops::Identity(scope.WithOpName("i1"), c1);
  Output r1 =
      ops::Reshape(scope.WithOpName("r1").WithControlDependencies(d1), v1, i1);
  Output s1 = ops::Square(scope.WithOpName("s1"), r1);

  // A multi dimensional reshape than can be optimized
  Output v3 = ops::Variable(scope.WithOpName("v3"), {5, 5, 5}, DT_FLOAT);
  Output c3 =
      ops::Const(scope.WithOpName("c3").WithControlDependencies(v3), 5, {3});
  Output i3 = ops::Identity(scope.WithOpName("i3"), c3);
  Output r3 = ops::Reshape(scope.WithOpName("r3"), v3, i3);
  Output s3 = ops::Square(scope.WithOpName("s3"), r3);

  // A multi dimensional partially defined reshape than can be optimized
  Output v4 = ops::Variable(scope.WithOpName("v4"), {5, 5, 5}, DT_FLOAT);
  Output c4 = ops::Const(scope.WithOpName("c4").WithControlDependencies(v4),
                         {5, -1, 5}, {3});
  Output i4 = ops::Identity(scope.WithOpName("i4"), c4);
  Output r4 = ops::Reshape(scope.WithOpName("r4"), v4, i4);
  Output s4 = ops::Square(scope.WithOpName("s4"), r4);

  // A reshape that can't be optimized
  Output v2 = ops::Variable(scope.WithOpName("v2"), {17, 1}, DT_FLOAT);
  Output c2 =
      ops::Const(scope.WithOpName("c2").WithControlDependencies(v2), 17, {1});
  Output r2 = ops::Reshape(scope.WithOpName("r2"), v2, c2);
  Output s2 = ops::Square(scope.WithOpName("s2"), r2);

  GrapplerItem item;
  item.fetch = {"s1", "s2", "s3", "s4"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "r1") {
      ++found;
      EXPECT_EQ("Identity", node.op());
      ASSERT_EQ(3, node.input_size());
      EXPECT_EQ("v1", node.input(0));
      EXPECT_EQ("^d1", node.input(1));
      EXPECT_EQ("^v1", node.input(2));
    } else if (node.name() == "r3") {
      ++found;
      EXPECT_EQ("Identity", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("v3", node.input(0));
      EXPECT_EQ("^v3", node.input(1));
    } else if (node.name() == "r4") {
      ++found;
      EXPECT_EQ("Identity", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("v4", node.input(0));
      EXPECT_EQ("^v4", node.input(1));
    } else if (node.name() == "r2") {
      ++found;
      EXPECT_EQ("Reshape", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("v2", node.input(0));
      EXPECT_EQ("c2", node.input(1));
    }
  }
  EXPECT_EQ(4, found);
}

TEST_F(ConstantFoldingTest, Packing) {
  // Build a simple graph with a large constant that can be folded.
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output c = ops::Const(scope.WithOpName("c"), 3.14f, {1000});
  Output i1 = ops::Identity(scope.WithOpName("i1"), c);
  Output i2 = ops::Identity(scope.WithOpName("i2"), c);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // Make sure that the representation of the folded constant is space
  // efficient: in particular, the whole message should be smaller than 8k (the
  // size needed to naively encode 1000 floats folded twice).
  EXPECT_GT(8000, output.ByteSizeLong());
}

TEST_F(ConstantFoldingTest, MaterializeBroadcastGradientArgs) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a =
      ops::Placeholder(s.WithOpName("a"), DT_FLOAT,
                       ops::Placeholder::Shape(PartialTensorShape({-1, -1})));
  Output b = ops::Square(s.WithOpName("b"), a);
  Output c = ops::Mul(s.WithOpName("c"), a, b);
  Output d = ops::Shape(s.WithOpName("d"), a);
  Output e = ops::Shape(s.WithOpName("e"), b);

  auto f = ops::internal::BroadcastGradientArgs(s.WithOpName("f"), d, e);
  Output o1 = ops::Identity(s.WithOpName("o1"), f.r0);
  Output o2 = ops::Identity(s.WithOpName("o2"), f.r1);

  Output g = ops::Placeholder(s.WithOpName("g"), DT_FLOAT,
                              ops::Placeholder::Shape(PartialTensorShape({1})));
  Output h = ops::Shape(s.WithOpName("h"), g);
  auto i = ops::internal::BroadcastGradientArgs(s.WithOpName("i"), d, h);
  Output p1 = ops::Identity(s.WithOpName("p1"), i.r0);
  Output p2 = ops::Identity(s.WithOpName("p2"), i.r1);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding fold(RewriterConfig::AGGRESSIVE, nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // Run a second time to make sure the optimization is idempotent.
  item.graph.Swap(&output);
  status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "o1") {
      ++found;
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("ConstantFolding/f-0", node.input(0));
    } else if (node.name() == "o2") {
      ++found;
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("ConstantFolding/f-1", node.input(0));
    } else if (node.name() == "ConstantFolding/f-0") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^f", node.input(0));
      EXPECT_EQ(0, TensorShape(node.attr().at("value").tensor().tensor_shape())
                       .num_elements());
    } else if (node.name() == "ConstantFolding/f-1") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^f", node.input(0));
      EXPECT_EQ(0, TensorShape(node.attr().at("value").tensor().tensor_shape())
                       .num_elements());
    } else if (node.name() == "p1") {
      ++found;
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("ConstantFolding/i-0", node.input(0));
    } else if (node.name() == "p2") {
      ++found;
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("i:1", node.input(0));
    } else if (node.name() == "ConstantFolding/i-0") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^i", node.input(0));
      EXPECT_EQ(0, TensorShape(node.attr().at("value").tensor().tensor_shape())
                       .num_elements());
    }
  }
  EXPECT_EQ(7, found);
}

TEST_F(ConstantFoldingTest, MaterializeReductionIndices) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input =
      ops::Placeholder(s.WithOpName("input"), DT_FLOAT,
                       ops::Placeholder::Shape(PartialTensorShape({-1, -1})));
  Output indices = ops::Placeholder(s.WithOpName("indices"), DT_INT32);
  Output sum = ops::Sum(s.WithOpName("sum"), input, indices);
  Output size = ops::Const(s.WithOpName("size"), 1, {1});
  Output reshape = ops::Reshape(s.WithOpName("reshape"), sum, size);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch.push_back("reshape");

  ConstantFolding fold(RewriterConfig::AGGRESSIVE, nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // Run a second time to make sure the optimization is idempotent.
  item.graph.Swap(&output);
  status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "ConstantFolding/sum-reduction_indices") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ("^indices", node.input(0));
      EXPECT_EQ(2, TensorShape(node.attr().at("value").tensor().tensor_shape())
                       .num_elements());
    } else if (node.name() == "sum") {
      ++found;
      EXPECT_EQ("ConstantFolding/sum-reduction_indices", node.input(1));
    } else if (node.name() == "indices") {
      ++found;
    }
  }
  EXPECT_EQ(3, found);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

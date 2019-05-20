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

#include "tensorflow/core/grappler/optimizers/debug_stripper.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class DebugStripperTest : public GrapplerTest {};

TEST_F(DebugStripperTest, OutputEqualToInput) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Placeholder(s, DT_FLOAT, ops::Placeholder::Shape({}));
  Output y = ops::Placeholder(s, DT_FLOAT, ops::Placeholder::Shape({}));
  Output add = ops::Add(s, x, y);
  Output result = ops::Identity(s, add);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  DebugStripper optimizer;
  GraphDef output;
  EXPECT_EQ(optimizer.Optimize(nullptr, item, &output),
            errors::Aborted("Nothing to do."));
}

TEST_F(DebugStripperTest, StripAssertOnTwoOutputs) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT,
                                  ops::Placeholder::Shape({6}));
  auto split =
      ops::Split(s.WithOpName("split"), /*axis=*/0, input, /*num_split=*/2);
  Output x = split[0];
  Output y = split[1];
  Output ge = ops::GreaterEqual(s.WithOpName("GreaterEqual"), x, y);
  auto assert = ops::Assert(s.WithOpName("Assert"), ge, {x, y});
  Output add = ops::Add(
      s.WithOpName("add").WithControlDependencies({assert.operation}), x, y);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  DebugStripper optimizer;
  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  for (const NodeDef& node : output.node()) {
    for (const string& input : node.input()) {
      if (IsControlInput(input)) {
        EXPECT_EQ(input.find(':'), -1);
      }
    }
  }
}

TEST_F(DebugStripperTest, StripAssertFromGraph) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Placeholder(s.WithOpName("x"), DT_FLOAT,
                              ops::Placeholder::Shape({}));
  Output y = ops::Placeholder(s.WithOpName("y"), DT_FLOAT,
                              ops::Placeholder::Shape({}));
  auto greaterequal = ops::GreaterEqual(s.WithOpName("GreaterEqual"), x, y);
  auto assert = ops::Assert(s.WithOpName("Assert"), greaterequal, {x, y});
  Output add = ops::Add(
      s.WithOpName("z").WithControlDependencies({assert.operation}), x, y);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  DebugStripper optimizer;
  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "x") {
      count++;
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "y") {
      count++;
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "GreaterEqual") {
      count++;
      EXPECT_EQ("GreaterEqual", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("y", node.input(1));
    } else if (node.name() == "Assert") {
      count++;
      EXPECT_EQ("NoOp", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("^GreaterEqual", node.input(0));
      EXPECT_EQ("^x", node.input(1));
      EXPECT_EQ("^y", node.input(2));
    } else if (node.name() == "z") {
      count++;
      EXPECT_EQ("Add", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("y", node.input(1));
      EXPECT_EQ("^Assert", node.input(2));
    }
  }
  EXPECT_EQ(5, count);

  Tensor x_t(DT_FLOAT, TensorShape({}));
  Tensor y_t(DT_FLOAT, TensorShape({}));
  x_t.flat<float>()(0) = 1.0f;
  y_t.flat<float>()(0) = 0.5f;
  std::vector<Tensor> expected =
      EvaluateNodes(item.graph, {"z"}, {{"x", x_t}, {"y", y_t}});
  std::vector<Tensor> optimized =
      EvaluateNodes(output, {"z"}, {{"x", x_t}, {"y", y_t}});
  test::ExpectTensorEqual<float>(expected[0], optimized[0]);
}

TEST_F(DebugStripperTest, StripCheckNumericsFromGraph) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Placeholder(s.WithOpName("x"), DT_FLOAT,
                              ops::Placeholder::Shape({}));
  Output y = ops::Placeholder(s.WithOpName("y"), DT_FLOAT,
                              ops::Placeholder::Shape({}));
  auto check1 = ops::CheckNumerics(s.WithOpName("CheckNumerics1"), x, "foo");
  auto check2 = ops::CheckNumerics(s.WithOpName("CheckNumerics2"), y, "foo");
  Output add = ops::Add(s.WithOpName("z"), check1, check2);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  DebugStripper optimizer;
  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "x") {
      count++;
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "y") {
      count++;
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "CheckNumerics1") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ(1, node.attr_size());
    } else if (node.name() == "CheckNumerics2") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y", node.input(0));
      EXPECT_EQ(1, node.attr_size());
    } else if (node.name() == "z") {
      count++;
      EXPECT_EQ("Add", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("CheckNumerics1", node.input(0));
      EXPECT_EQ("CheckNumerics2", node.input(1));
    }
  }
  EXPECT_EQ(5, count);

  Tensor x_t(DT_FLOAT, TensorShape({}));
  Tensor y_t(DT_FLOAT, TensorShape({}));
  x_t.flat<float>()(0) = 1.0f;
  y_t.flat<float>()(0) = 0.5f;
  std::vector<Tensor> expected =
      EvaluateNodes(item.graph, {"z"}, {{"x", x_t}, {"y", y_t}});
  std::vector<Tensor> optimized =
      EvaluateNodes(output, {"z"}, {{"x", x_t}, {"y", y_t}});
  test::ExpectTensorEqual<float>(expected[0], optimized[0]);
}

TEST_F(DebugStripperTest, StripPrintFromGraph) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Placeholder(s.WithOpName("x"), DT_FLOAT,
                              ops::Placeholder::Shape({}));
  Output print = ops::Print(s.WithOpName("Print"), x, {x});
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  DebugStripper optimizer;
  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  for (const NodeDef& node : output.node()) {
    if (node.name() == "x") {
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "Print") {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("^x", node.input(1));
      EXPECT_EQ(1, node.attr_size());
    }
  }

  EXPECT_EQ(2, output.node_size());

  Tensor x_t(DT_FLOAT, TensorShape({}));
  x_t.flat<float>()(0) = 1.0f;
  std::vector<Tensor> expected =
      EvaluateNodes(item.graph, {"Print"}, {{"x", x_t}});
  std::vector<Tensor> optimized =
      EvaluateNodes(output, {"Print"}, {{"x", x_t}});
  test::ExpectTensorEqual<float>(expected[0], optimized[0]);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

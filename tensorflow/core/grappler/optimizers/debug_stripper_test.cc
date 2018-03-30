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
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class DebugStripperTest : public GrapplerTest {};

TEST_F(DebugStripperTest, OutputEqualToInput) {
  constexpr char device[] = "/device:CPU:0";
  GrapplerItem item;
  item.graph = test::function::GDef(
      {test::function::NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}},
                            device),
       test::function::NDef("y", "XTimesTwo", {"x"}, {}, device),
       test::function::NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, device)},
      {});

  DebugStripper optimizer;
  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));
  CompareGraphs(item.graph, output);
}

TEST_F(DebugStripperTest, StripAssertFromGraph) {
  constexpr char device[] = "/device:CPU:0";
  GrapplerItem item;
  item.graph = test::function::GDef(
      {test::function::NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}},
                            device),
       test::function::NDef("y", "Placeholder", {}, {{"dtype", DT_FLOAT}},
                            device),
       test::function::NDef("GreaterEqual", "GreaterEqual", {"x", "y"},
                            {{"T", DT_FLOAT}}, device),
       test::function::NDef("Assert", "Assert", {"GreaterEqual"},
                            {{"T", DT_FLOAT}}, device),
       test::function::NDef("z", "Add", {"x", "y", "^Assert"}, {}, device)},
      {});

  DebugStripper optimizer;
  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "x") {
      count++;
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "y") {
      count++;
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "GreaterEqual") {
      count++;
      EXPECT_EQ("GreaterEqual", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("y", node.input(1));
    } else if (node.name() == "Assert") {
      count++;
      EXPECT_EQ("NoOp", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^GreaterEqual", node.input(0));
      EXPECT_EQ(0, node.attr_size());
    } else if (node.name() == "z") {
      count++;
      EXPECT_EQ("Add", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("y", node.input(1));
      EXPECT_EQ("^Assert", node.input(2));
    }
  }
  EXPECT_EQ(5, count);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

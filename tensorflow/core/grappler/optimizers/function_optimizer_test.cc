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

#include "tensorflow/core/grappler/optimizers/function_optimizer.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace grappler {
namespace {

class FunctionOptimizerTest : public GrapplerTest {};

TEST_F(FunctionOptimizerTest, SimpleFunction) {
  // Build a graph to compute y = XTimesTwo(x)
  GrapplerItem item;
  constexpr char device[] = "/device:CPU:0";
  item.graph = test::function::GDef(
      {test::function::NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}},
                            device),
       test::function::NDef("y", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}}, device),
       test::function::NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, device)},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  FunctionOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "y/inlined_inputs") {
      count++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("x", node.input(0));
    } else if (node.name() == "y/x") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/inlined_inputs:0", node.input(0));
    } else if (node.name() == "y/two") {
      count++;
      EXPECT_EQ("Const", node.op());
    } else if (node.name() == "y/scale") {
      count++;
      EXPECT_EQ("Cast", node.op());
    } else if (node.name() == "y/y") {
      count++;
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("y/x", node.input(0));
      EXPECT_EQ("y/scale:0", node.input(1));
    } else if (node.name() == "y") {
      count++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/y", node.input(0));
    } else if (node.name() == "z") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y", node.input(0));
    }
  }
  EXPECT_EQ(7, count);

  item.fetch = {"z"};
  Tensor pi(DT_FLOAT, {});
  pi.flat<float>()(0) = 3.14f;
  item.feed.emplace_back("x", pi);
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized(item, std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

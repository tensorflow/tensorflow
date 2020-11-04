/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/enable_gradient_descent.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

Status OptimizeWithEnableGradientDescent(const GrapplerItem &item,
                                         GraphDef *output, bool autotune) {
  EnableGradientDescent optimizer;
  RewriterConfig_CustomGraphOptimizer config;
  if (autotune) {
    (*config.mutable_parameter_map())["autotune"].set_s("true");
  } else {
    (*config.mutable_parameter_map())["autotune"].set_s("false");
  }
  TF_RETURN_IF_ERROR(optimizer.Init(&config));
  return optimizer.Optimize(nullptr, item, output);
}

class SimpleRewrite
    : public ::testing::TestWithParam<std::tuple<bool, int64, string>> {};

TEST_P(SimpleRewrite, EnableGradientDescentTest) {
  const bool autotune = std::get<0>(GetParam());
  const int64 algorithm_index = std::get<1>(GetParam());
  const string op = std::get<2>(GetParam());

  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("batch_size", "Const", {}, {{"value", 5}, {"dtype", DT_INT32}}),
       NDef("batch", "BatchDataset", {"range", "batch_size"}, {}),
       NDef("model", "ModelDataset", {"batch"},
            {{"algorithm", algorithm_index}}),
       NDef("Sink", op, {"model"}, {})});
  item.fetch.push_back("Sink");

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithEnableGradientDescent(item, &output, autotune));
  EXPECT_EQ(item.graph.node().size(), output.node().size());

  NodeDef model_node =
      output.node(graph_utils::FindGraphNodeWithName("model", output));
  EXPECT_EQ(model_node.attr().at("algorithm").i(),
            (autotune && op != "_Retval") ? 1 : algorithm_index);
}

INSTANTIATE_TEST_SUITE_P(
    Test, SimpleRewrite,
    ::testing::Combine(::testing::Values(false, true), ::testing::Values(0, 1),
                       ::testing::Values("Identity", "_Retval")));
}  // namespace
}  // namespace grappler
}  // namespace tensorflow

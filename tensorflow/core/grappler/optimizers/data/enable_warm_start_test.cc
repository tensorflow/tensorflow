/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/enable_warm_start.h"

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

TEST(Basic, EnableWarmStartTest) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("batch_size", "Const", {}, {{"value", 5}, {"dtype", DT_INT32}}),
       NDef("batch", "BatchDataset", {"range", "batch_size"}, {}),
       NDef("model", "ModelDataset", {"batch"}, {}),
       NDef("Sink", "Identity", {"model"}, {})});
  item.fetch.push_back("Sink");

  GraphDef output;
  EnableWarmStart optimizer;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_EQ(item.graph.node().size(), output.node().size());
  NodeDef model_node =
      output.node(graph_utils::FindGraphNodeWithName("model", output));
  EXPECT_TRUE(model_node.attr().contains("warm_start"));
  EXPECT_TRUE(model_node.attr().at("warm_start").b());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

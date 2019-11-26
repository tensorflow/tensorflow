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

#include "tensorflow/core/grappler/optimizers/data/make_stateless.h"

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

TEST(MakeStateless, Cache) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("filename", "Const", {}, {{"value", ""}, {"dtype", DT_INT64}}),
       NDef("handle", "Const", {}, {{"value", 1}, {"dtype", DT_RESOURCE}}),
       graph_tests_utils::MakeCacheV2Node("cache", "range", "filename",
                                          "handle")},
      {});

  MakeStateless optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("cache", output));
  int index = graph_utils::FindGraphNodeWithName("cache", output);
  EXPECT_EQ(output.node(index).op(), "CacheDataset");
  EXPECT_EQ(output.node(index).input_size(), 2);
}

TEST(MakeStateless, Shuffle) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("buffer_size", "Const", {}, {{"value", 1}, {"dtype", DT_INT64}}),
       NDef("handle", "Const", {}, {{"value", 1}, {"dtype", DT_RESOURCE}}),
       graph_tests_utils::MakeShuffleV2Node("shuffle", "range", "buffer_size",
                                            "handle")},
      {});

  MakeStateless optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("shuffle", output));
  int index = graph_utils::FindGraphNodeWithName("shuffle", output);
  EXPECT_EQ(output.node(index).op(), "ShuffleDataset");
  EXPECT_EQ(output.node(index).input_size(), 4);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

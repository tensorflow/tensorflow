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

#include "tensorflow/core/grappler/optimizers/data/make_sloppy.h"

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

using graph_tests_utils::MakeParallelInterleaveNode;
using graph_tests_utils::MakeParallelMapNode;
using graph_tests_utils::MakeParseExampleNode;

TEST(MakeSloppy, ParallelInterleave) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("cycle_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("block_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       MakeParallelInterleaveNode("interleave", "range", "cycle_length",
                                  "block_length", "num_parallel_calls",
                                  "XTimesTwo", /*sloppy=*/false)},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  MakeSloppy optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("interleave", output));
  int index = graph_utils::FindGraphNodeWithName("interleave", output);
  EXPECT_TRUE(output.node(index).attr().at("sloppy").b());
}

TEST(MakeSloppy, ParallelMap) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       MakeParallelMapNode("map", "range", "num_parallel_calls", "XTimesTwo",
                           /*sloppy=*/false)},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  MakeSloppy optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("map", output));
  int index = graph_utils::FindGraphNodeWithName("map", output);
  EXPECT_TRUE(output.node(index).attr().at("sloppy").b());
}

TEST(MakeSloppy, ParseExampleDataset) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       MakeParseExampleNode("parse_example", "range", "num_parallel_calls",
                            /*sloppy=*/false)},
      // FunctionLib
      {});

  MakeSloppy optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("parse_example", output));
  int index = graph_utils::FindGraphNodeWithName("parse_example", output);
  EXPECT_TRUE(output.node(index).attr().at("sloppy").b());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

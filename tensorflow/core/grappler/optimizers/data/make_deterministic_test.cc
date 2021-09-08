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

#include "tensorflow/core/grappler/optimizers/data/make_deterministic.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/determinism.h"

namespace tensorflow {
namespace grappler {
namespace {

class MakeDeterministicTest : public ::testing::TestWithParam<bool> {};

TEST_P(MakeDeterministicTest, NoRewriteInterleave) {
  using test::function::NDef;
  GrapplerItem item;
  bool nest = GetParam();
  std::string func_name = nest ? "OuterXTimesTwo" : "XTimesTwo";

  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("cycle_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("block_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeParallelInterleaveV2Node(
           "interleave", "range", "cycle_length", "block_length",
           "num_parallel_calls", func_name, /*sloppy=*/false)},
      // FunctionLib
      {test::function::XTimesTwo(),
       FunctionDefHelper::Define(
           // Name
           "OuterXTimesTwo",
           // Args
           {"x: float"},
           // Return values
           {"y: float"},
           // Attr def
           {},
           {{{"y"},
             "PartitionedCall",
             {"x"},
             {{"Tin", DataTypeSlice{DT_FLOAT}},
              {"Tout", DataTypeSlice{DT_FLOAT}},
              {"f", FunctionDefHelper::FunctionRef("XTimesTwo",
                                                   {{"T", DT_FLOAT}})}}}})});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithName("interleave", output);
  ASSERT_GE(index, 0);
  NodeDef node_def = output.node(index);
  ASSERT_EQ(node_def.op(), "ParallelInterleaveDatasetV2");
}

TEST_P(MakeDeterministicTest, NoRewriteMap) {
  using test::function::NDef;
  GrapplerItem item;
  bool nest = GetParam();
  std::string func_name = nest ? "OuterXTimesTwo" : "XTimesTwo";
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeParallelMapNode("map", "range",
                                              "num_parallel_calls", func_name,
                                              /*sloppy=*/false)},
      // FunctionLib
      {test::function::XTimesTwo(),
       FunctionDefHelper::Define(
           // Name
           "OuterXTimesTwo",
           // Args
           {"x: float"},
           // Return values
           {"y: float"},
           // Attr def
           {},
           {{{"y"},
             "PartitionedCall",
             {"x"},
             {{"Tin", DataTypeSlice{DT_FLOAT}},
              {"Tout", DataTypeSlice{DT_FLOAT}},
              {"f", FunctionDefHelper::FunctionRef("XTimesTwo",
                                                   {{"T", DT_FLOAT}})}}}})});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithName("map", output);
  ASSERT_GE(index, 0);
  NodeDef node_def = output.node(index);
  ASSERT_EQ(node_def.op(), "ParallelMapDataset");
}

TEST_P(MakeDeterministicTest, RewriteInterleave) {
  using test::function::NDef;
  typedef FunctionDefHelper FDH;
  GrapplerItem item;
  bool nest = GetParam();
  std::string func_name = nest ? "OuterRandomUniform" : "RandomUniform";

  NodeDef interleave_node_def = graph_tests_utils::MakeParallelInterleaveV2Node(
      "interleave", "range", "cycle_length", "block_length",
      "num_parallel_calls", func_name, /*sloppy=*/false);
  interleave_node_def.add_input("^start");

  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("cycle_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("block_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       interleave_node_def},
      // FunctionLib
      {test::function::RandomUniform(),
       FDH::Define(
           // Name
           "OuterRandomUniform",
           // Args
           {"x: float"},
           // Return values
           {"random_uniform: int64"},
           // Attr def
           {},
           {{{"random_uniform"},
             "StatefulPartitionedCall",
             {"x"},
             {{"Tin", DataTypeSlice{DT_FLOAT}},
              {"Tout", DataTypeSlice{DT_INT64}},
              {"f", FDH::FunctionRef("RandomUniform", {{"T", DT_FLOAT}})}}}})});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithOp("InterleaveDataset", output);
  ASSERT_GE(index, 0);
  NodeDef node_def = output.node(index);
  ASSERT_EQ(node_def.input_size(), 5);
  ASSERT_EQ(node_def.input(0), "range");
  ASSERT_EQ(node_def.input(1), "cycle_length");
  ASSERT_EQ(node_def.input(2), "block_length");
  ASSERT_EQ(node_def.input(3), "^num_parallel_calls");
  ASSERT_EQ(node_def.input(4), "^start");
}

TEST_P(MakeDeterministicTest, RewriteMap) {
  using test::function::NDef;
  typedef FunctionDefHelper FDH;
  GrapplerItem item;
  bool nest = GetParam();
  std::string func_name = nest ? "OuterRandomUniform" : "RandomUniform";

  NodeDef map_node_def = graph_tests_utils::MakeParallelMapNode(
      "map", "range", "num_parallel_calls", func_name,
      /*sloppy=*/false);
  map_node_def.add_input("^start");

  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       map_node_def},
      // FunctionLib
      {test::function::RandomUniform(),
       FDH::Define(
           // Name
           "OuterRandomUniform",
           // Args
           {"x: float"},
           // Return values
           {"random_uniform: int64"},
           // Attr def
           {},
           {{{"random_uniform"},
             "StatefulPartitionedCall",
             {"x"},
             {{"Tin", DataTypeSlice{DT_FLOAT}},
              {"Tout", DataTypeSlice{DT_INT64}},
              {"f", FDH::FunctionRef("RandomUniform", {{"T", DT_FLOAT}})}}}})});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithOp("MapDataset", output);
  ASSERT_GE(index, 0);
  NodeDef node_def = output.node(index);
  ASSERT_EQ(node_def.input_size(), 3);
  ASSERT_EQ(node_def.input(0), "range");
  ASSERT_EQ(node_def.input(1), "^num_parallel_calls");
  ASSERT_EQ(node_def.input(2), "^start");
}

INSTANTIATE_TEST_SUITE_P(Test, MakeDeterministicTest,
                         ::testing::Values(false, true));

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

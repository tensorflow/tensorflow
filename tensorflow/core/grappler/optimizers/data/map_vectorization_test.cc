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

#include "tensorflow/core/grappler/optimizers/data/map_vectorization.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

using test::function::GDef;
using test::function::NDef;

NodeDef MakeMapNodeHelper(StringPiece name, StringPiece input_node_name,
                          StringPiece function_name, StringPiece map_op_name,
                          gtl::ArraySlice<PartialTensorShape> output_shapes,
                          gtl::ArraySlice<DataType> output_types) {
  return test::function::NDef(
      name, map_op_name, {string(input_node_name)},
      {{"f", FunctionDefHelper::FunctionRef(string(function_name))},
       {"Targuments", {}},
       {"output_shapes", output_shapes},
       {"output_types", output_types}});
}

NodeDef MakeMapNode(StringPiece name, StringPiece input_node_name,
                    StringPiece function_name,
                    gtl::ArraySlice<PartialTensorShape> output_shapes,
                    gtl::ArraySlice<DataType> output_types) {
  return MakeMapNodeHelper(name, input_node_name, function_name, "MapDataset",
                           output_shapes, output_types);
}

NodeDef MakeBatchNode(StringPiece name, StringPiece input_node_name,
                      StringPiece input_batch_size_name,
                      gtl::ArraySlice<PartialTensorShape> output_shapes,
                      gtl::ArraySlice<DataType> output_types) {
  return NDef(
      name, "BatchDataset",
      {string(input_node_name), string(input_batch_size_name)},
      {{"output_types", output_types}, {"output_shapes", output_shapes}});
}

NodeDef MakeBatchV2Node(StringPiece name, StringPiece input_node_name,
                        StringPiece input_batch_size_name,
                        StringPiece input_drop_remainder_name,
                        gtl::ArraySlice<PartialTensorShape> output_shapes,
                        gtl::ArraySlice<DataType> output_types) {
  return NDef(
      name, "BatchDatasetV2",
      {string(input_node_name), string(input_batch_size_name),
       string(input_drop_remainder_name)},
      {{"output_types", output_types}, {"output_shapes", output_shapes}});
}

NodeDef MakeRangeNode(StringPiece name, gtl::ArraySlice<string> inputs) {
  return NDef(name, "RangeDataset", inputs,
              {{"output_shapes", gtl::ArraySlice<TensorShape>({{}})},
               {"output_types", gtl::ArraySlice<DataType>({DT_INT64})}});
}

TEST(MapVectorizationTest, VectorizeMapWithBatch) {
  GrapplerItem item;
  item.graph = GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("batch_size", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       MakeRangeNode("range", {"start", "stop", "step"}),
       MakeMapNode("map", "range", "XTimesTwo", {{}}, {DT_INT32}),
       MakeBatchNode("batch", "map", "batch_size", {{-1}}, {DT_INT32})},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });
  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(graph_utils::FindAllGraphNodesWithOp("MapDataset", output).size(),
            1);
  EXPECT_EQ(graph_utils::FindAllGraphNodesWithOp("BatchDataset", output).size(),
            1);
  const NodeDef& map_node =
      output.node(graph_utils::FindGraphNodeWithOp("MapDataset", output));
  const NodeDef& batch_node =
      output.node(graph_utils::FindGraphNodeWithOp("BatchDataset", output));
  EXPECT_EQ(map_node.input(0), batch_node.name());
  EXPECT_EQ(batch_node.input(0), "range");
}

TEST(MapVectorizationTest, VectorizeMapWithBatchV2) {
  GrapplerItem item;
  item.graph = GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("batch_size", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("drop_remainder", "Const", {},
            {{"value", false}, {"dtype", DT_BOOL}}),
       MakeRangeNode("range", {"start", "stop", "step"}),
       MakeMapNode("map", "range", "XTimesTwo", {{}}, {DT_INT32}),
       MakeBatchV2Node("batch", "map", "batch_size", "drop_remainder", {{-1}},
                       {DT_INT32})},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });
  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(graph_utils::FindAllGraphNodesWithOp("MapDataset", output).size(),
            1);
  EXPECT_EQ(
      graph_utils::FindAllGraphNodesWithOp("BatchDatasetV2", output).size(), 1);
  const NodeDef& map_node =
      output.node(graph_utils::FindGraphNodeWithOp("MapDataset", output));
  const NodeDef& batch_node =
      output.node(graph_utils::FindGraphNodeWithOp("BatchDatasetV2", output));
  EXPECT_EQ(map_node.input(0), batch_node.name());
  EXPECT_EQ(batch_node.input(0), "range");
}

TEST(MapVectorizationTest, VectorizeWithUndefinedOutputShape) {
  GrapplerItem item;
  item.graph = GDef(
      {NDef("batch_size", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("input", "InputDataset", {},
            {{"output_types", gtl::ArraySlice<DataType>({DT_INT32})}}),
       MakeMapNode("map", "input", "XTimesTwo", {{}}, {DT_INT32}),
       MakeBatchNode("batch", "map", "batch_size", {{-1}}, {DT_INT32})},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });
  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
}

TEST(MapVectorizationTest, VectorizeWithUndefinedOutputTypes) {
  GrapplerItem item;
  item.graph = GDef(
      {NDef("batch_size", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("input", "InputDataset", {},
            {{"output_shapes", gtl::ArraySlice<TensorShape>({{}})}}),
       MakeMapNode("map", "input", "XTimesTwo", {{}}, {DT_INT32}),
       MakeBatchNode("batch", "map", "batch_size", {{-1}}, {DT_INT32})},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });
  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
}

TEST(MapVectorizationTest, VectorizeWithFullyDefinedFunction) {
  GrapplerItem item;
  item.graph = GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("batch_size", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       MakeRangeNode("range", {"start", "stop", "step"}),
       MakeMapNode("map", "range", "Func", {{}}, {DT_INT32}),
       MakeBatchNode("batch", "map", "batch_size", {{-1}}, {DT_INT32})},
      // FunctionLib
      {FunctionDefHelper::Create(
          "Func", {"x: int64", "y: int64"}, {"res: int64", "res2: int64"}, {},
          {{{"o"}, "Mul", {"x", "x"}, {{"T", DT_INT64}}}},
          {{"res", "o:z"}, {"res2", "o:z"}})});
  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(graph_utils::FindAllGraphNodesWithOp("MapDataset", output).size(),
            1);
  EXPECT_EQ(graph_utils::FindAllGraphNodesWithOp("BatchDataset", output).size(),
            1);
  const NodeDef& map_node =
      output.node(graph_utils::FindGraphNodeWithOp("MapDataset", output));
  const NodeDef& batch_node =
      output.node(graph_utils::FindGraphNodeWithOp("BatchDataset", output));
  EXPECT_EQ(map_node.input(0), batch_node.name());
  EXPECT_EQ(batch_node.input(0), "range");
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

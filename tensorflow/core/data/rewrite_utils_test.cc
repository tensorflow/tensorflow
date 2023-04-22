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
#include "tensorflow/core/data/rewrite_utils.h"

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::test::AsScalar;
using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;
using ::testing::ElementsAre;

NodeDef GetMapNode(absl::string_view name, absl::string_view input_node_name,
                   absl::string_view function_name) {
  return NDef(
      name, /*op=*/"MapDataset", {std::string(input_node_name)},
      {{"f", FunctionDefHelper::FunctionRef(std::string(function_name))},
       {"Targuments", {}},
       {"output_shapes", gtl::ArraySlice<TensorShape>{TensorShape()}},
       {"output_types", gtl::ArraySlice<DataType>{DT_INT64}}});
}

FunctionDef XTimesX() {
  return FunctionDefHelper::Create(
      /*function_name=*/"XTimesX",
      /*in_def=*/{"x: int64"},
      /*out_def=*/{"y: int64"},
      /*attr_def=*/{},
      /*node_def=*/{{{"y"}, "Mul", {"x", "x"}, {{"T", DT_INT64}}}},
      /*ret_def=*/{{"y", "y:z:0"}});
}

GraphDef GetRangeSquareDatasetDef(const int64 range) {
  return GDef(
      {NDef("start", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64>(0)}, {"dtype", DT_INT64}}),
       NDef("stop", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64>(range)}, {"dtype", DT_INT64}}),
       NDef("step", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64>(1)}, {"dtype", DT_INT64}}),
       NDef("range", "RangeDataset", /*inputs=*/{"start", "stop", "step"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{TensorShape()}},
             {"output_types", gtl::ArraySlice<DataType>{DT_INT64}}}),
       GetMapNode("map", "range", "XTimesX"),
       NDef("dataset", "_Retval", /*inputs=*/{"map"},
            {{"T", DT_VARIANT}, {"index", 0}})},
      {XTimesX()});
}

TEST(GraphUtilTest, GetFetchNode) {
  GraphDef graph = GetRangeSquareDatasetDef(10);
  TF_ASSERT_OK_AND_ASSIGN(std::string dataset_node, GetDatasetNode(graph));
  std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
      GetGrapplerItem(&graph, &dataset_node, /*add_fake_sinks=*/false);
  EXPECT_THAT(grappler_item->fetch, ElementsAre("Sink"));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow

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

#include "tensorflow/core/grappler/optimizers/data/hoist_random_uniform.h"

#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

TEST(HoistRandomUniform, SimpleHoisting) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("filename", "Const", {}, {{"value", ""}, {"dtype", DT_STRING}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
             {"output_types", gtl::ArraySlice<DataType>{}}}),
       graph_tests_utils::MakeMapNode("map1", "range", "RandomUniform"),
       NDef("cache", "CacheDataset", {"map1", "filename"}, {})},
      // FunctionLib
      {
          test::function::RandomUniform(),
      });

  HoistRandomUniform optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map1", output));
  const int new_map_id = graph_utils::FindGraphNodeWithOp("MapDataset", output);
  const int zip_dataset_id =
      graph_utils::FindGraphNodeWithOp("ZipDataset", output);
  const int random_dataset_id =
      graph_utils::FindGraphNodeWithOp("RandomDataset", output);
  const int batch_random_id =
      graph_utils::FindGraphNodeWithOp("BatchDatasetV2", output);
  ASSERT_NE(random_dataset_id, -1);
  ASSERT_NE(zip_dataset_id, -1);
  ASSERT_NE(new_map_id, -1);
  ASSERT_NE(batch_random_id, -1);

  const auto& new_map = output.node(new_map_id);
  const auto& zip = output.node(zip_dataset_id);
  const auto& random = output.node(random_dataset_id);
  const auto& batch = output.node(batch_random_id);

  ASSERT_EQ(new_map.input_size(), 1);
  EXPECT_EQ(new_map.input(0), zip.name());

  ASSERT_EQ(zip.input_size(), 2);
  EXPECT_EQ(zip.input(0), "range");
  EXPECT_EQ(zip.input(1), batch.name());

  ASSERT_EQ(batch.input_size(), 3);
  EXPECT_EQ(batch.input(0), random.name());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

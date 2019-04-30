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
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

void RunHardcodeMinMaxUInt8(const std::vector<int>& indices_val,
                            const std::vector<int>& depth_val,
                            const std::vector<uint8_t>& on_value,
                            const std::vector<uint8_t>& off_value,
                            const std::vector<int>& output_shape,
                            const std::vector<double>& expected_result) {
  Model model;
  Array& indices = model.GetOrCreateArray("Indices");
  Array& depth = model.GetOrCreateArray("Depth");
  Array& on = model.GetOrCreateArray("Onvalue");
  Array& off = model.GetOrCreateArray("Offvalue");
  Array& output = model.GetOrCreateArray("output");

  indices.data_type = ArrayDataType::kInt32;
  indices.GetMutableBuffer<ArrayDataType::kInt32>().data = indices_val;

  depth.data_type = ArrayDataType::kInt32;
  depth.GetMutableBuffer<ArrayDataType::kInt32>().data = depth_val;

  on.data_type = ArrayDataType::kUint8;
  auto& onval = on.GetOrCreateMinMax();
  onval.min = on_value[0];
  onval.max = on_value[1];

  off.data_type = ArrayDataType::kUint8;
  auto& offval = off.GetOrCreateMinMax();
  offval.min = off_value[0];
  offval.max = off_value[1];

  *output.mutable_shape()->mutable_dims() = output_shape;

  auto onehot_op = new OneHotOperator;
  onehot_op->inputs = {"Indices", "Depth", "Onvalue", "Offvalue"};
  onehot_op->outputs = {"output"};

  /*Stack everything with the model*/
  model.operators.push_back(std::unique_ptr<Operator>(onehot_op));

  bool modified;
  ASSERT_TRUE(HardcodeMinMax().Run(&model, 0, &modified).ok());

  Array& res_output = model.GetOrCreateArray("output");
  const auto& output_minmax = res_output.GetMinMax();

  EXPECT_THAT(output_minmax.min, expected_result[0]);
  EXPECT_THAT(output_minmax.max, expected_result[1]);
}

void RunHardcodeMinMaxInt8(const std::vector<int>& indices_val,
                           const std::vector<int>& depth_val,
                           const std::vector<int8_t>& on_value,
                           const std::vector<int8_t>& off_value,
                           const std::vector<int>& output_shape,
                           const std::vector<double>& expected_result) {
  Model model;
  Array& indices = model.GetOrCreateArray("Indices");
  Array& depth = model.GetOrCreateArray("Depth");
  Array& on = model.GetOrCreateArray("Onvalue");
  Array& off = model.GetOrCreateArray("Offvalue");
  Array& output = model.GetOrCreateArray("output");

  indices.data_type = ArrayDataType::kInt32;
  indices.GetMutableBuffer<ArrayDataType::kInt32>().data = indices_val;

  depth.data_type = ArrayDataType::kInt32;
  depth.GetMutableBuffer<ArrayDataType::kInt32>().data = depth_val;

  on.data_type = ArrayDataType::kInt8;
  auto& onval = on.GetOrCreateMinMax();
  onval.min = on_value[0];
  onval.max = on_value[1];

  off.data_type = ArrayDataType::kInt8;
  auto& offval = off.GetOrCreateMinMax();
  offval.min = off_value[0];
  offval.max = off_value[1];

  *output.mutable_shape()->mutable_dims() = output_shape;

  auto onehot_op = new OneHotOperator;
  onehot_op->inputs = {"Indices", "Depth", "Onvalue", "Offvalue"};
  onehot_op->outputs = {"output"};

  /*Stack everything with the model*/
  model.operators.push_back(std::unique_ptr<Operator>(onehot_op));

  bool modified;
  ASSERT_TRUE(HardcodeMinMax().Run(&model, 0, &modified).ok());

  Array& res_output = model.GetOrCreateArray("output");
  const auto& output_minmax = res_output.GetMinMax();

  EXPECT_THAT(output_minmax.min, expected_result[0]);
  EXPECT_THAT(output_minmax.max, expected_result[1]);
}
}  // namespace

TEST(HardcodeMinMax, SimpleUInt8Test) {
  RunHardcodeMinMaxUInt8(
      // Indices
      {3},

      // depth
      {4},

      // on min max
      {0, 5},

      // off min max
      {0, 5},

      // output shape
      {3, 4},

      // expected result
      {0.0, 1.0});
}

TEST(HardcodeMinMax, SimpleInt8Test) {
  RunHardcodeMinMaxInt8(
      // Indices
      {3},

      // depth
      {4},

      // on min max
      {-5, 5},

      // off min max
      {-5, 5},

      // output shape
      {3, 4},

      // expected result
      {0.0, 1.0});
}

}  // namespace toco

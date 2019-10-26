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
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

void RunResolveSum(const std::vector<float>& input,
                   const std::vector<int>& input_shape,
                   const std::vector<int>& axis,
                   const std::vector<int>& output_shape,
                   const std::vector<float>& expected_output) {
  Model model;
  const std::string output_name("output");
  model.flags.add_output_arrays(output_name);
  Array& input0 = model.GetOrCreateArray("input0");
  Array& input1 = model.GetOrCreateArray("input1");
  Array& output = model.GetOrCreateArray(output_name);

  *input0.mutable_shape()->mutable_dims() = input_shape;
  input0.data_type = ArrayDataType::kFloat;
  input0.GetMutableBuffer<ArrayDataType::kFloat>().data = input;

  *input1.mutable_shape()->mutable_dims() = {static_cast<int>(axis.size())};
  input1.GetMutableBuffer<ArrayDataType::kInt32>().data = axis;
  input1.data_type = ArrayDataType::kInt32;

  *output.mutable_shape()->mutable_dims() = output_shape;

  auto sum_op = absl::make_unique<TensorFlowSumOperator>();
  sum_op->keep_dims = true;
  sum_op->inputs = {"input0", "input1"};
  sum_op->outputs = {output_name};
  model.operators.push_back(std::move(sum_op));
  bool modified;
  ASSERT_TRUE(ResolveConstantUnaryOperator().Run(&model, 0, &modified).ok());
  EXPECT_EQ(model.GetArray("output").GetBuffer<ArrayDataType::kFloat>().data,
            expected_output);
  EXPECT_EQ(model.GetArray("output").shape().dims(), output_shape);
}

// Reduce a 2d array across axis 0
TEST(ResolveConstantUnary, ResolveSumAxis0_2D) {
  // clang-format off
  RunResolveSum(
      // Input data
      {3, 1, 4, 1,
       5, 9, 2, 6,
       5, 3, 5, 8},

      // Input shape
      {3, 4},

      // Axes
      {0},

      // Expected output shape,
      {1, 4},

      // Expected output
      {13, 13, 11, 15});
  // clang-format on
}

// Reduce a 2d array across axis 1
TEST(ResolveConstantUnary, ResolveSumAxis1_2D) {
  // clang-format off
  RunResolveSum(
      // Input data
      {3, 1, 4, 1,
       5, 9, 2, 6,
       5, 3, 5, 8},

      // Input shape
      {3, 4},

      // Axes
      {1},

      // Expected output shape,
      {3, 1},

      // Expected output
      {9, 22, 21});
  // clang-format on
}

// Reduce a 3d tensor across axes 0 and 2.
TEST(ResolveConstantUnary, ResolveSumAxis0_2_3D) {
  // clang-format off
  RunResolveSum(
      // Input data
      {  0,   1,   2,
         3,  10,  11,
        12,  13,  20,
        21,  22,  23,

       100, 101, 102,
       103, 110, 111,
       112, 113, 120,
       121, 122, 123,

       200, 201, 202,
       203, 210, 211,
       212, 213, 220,
       221, 222, 223 },

      // Input shape
      {3, 4, 3},

      // Axes
      {0, 2},

      // Expected output shape,
      {1, 4, 1},

      // Expected output, generated using octave.
      { 909, 972, 1035, 1098});
  // clang-format on
}

}  // namespace
}  // namespace toco

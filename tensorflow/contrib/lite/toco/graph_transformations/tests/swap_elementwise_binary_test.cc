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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"

namespace toco {

namespace {

int ShapeCount(const std::vector<int>& size) {
  CHECK(size.size());
  int count = 1;
  for (int dim : size) {
    count *= dim;
  }
  return count;
}

// Adds a new parameter array to the model.
void AddConstArray(const string& name, const float* data,
                   const std::vector<int>& size, Model* model) {
  Array& array = model->GetOrCreateArray(name);
  array.data_type = ArrayDataType::kFloat;
  Shape* shape = array.mutable_shape();
  *(shape->mutable_dims()) = size;

  auto& buffer = array.GetMutableBuffer<toco::ArrayDataType::kFloat>();
  buffer.data.resize(ShapeCount(size));
  std::copy(data, data + ShapeCount(size), buffer.data.data());
}

}  // namespace

TEST(SwapElementwiseBinaryTest, SwapsReshape) {
  Model model;
  const float parameters[2][4] = {{0., 1., 2., 3.}, {10., 11., 12., 13.}};

  AddConstArray("before_reshape", parameters[0], {2, 2}, &model);
  AddConstArray("add_vector", parameters[1], {1, 4}, &model);

  auto reshape_op = absl::make_unique<TensorFlowReshapeOperator>();
  reshape_op->shape = {1, 4};
  reshape_op->inputs = {"before_reshape"};
  reshape_op->outputs = {"after_reshape"};
  Array& reshape_array = model.GetOrCreateArray("after_reshape");
  *(reshape_array.mutable_shape()) = {1, 4};

  auto add_op = absl::make_unique<AddOperator>();
  add_op->inputs = {"after_reshape", "add_vector"};
  add_op->outputs = {"add"};
  Array& add_array = model.GetOrCreateArray("add");
  *(add_array.mutable_shape()) = {1, 4};

  model.operators.push_back(std::move(reshape_op));
  model.operators.push_back(std::move(add_op));

  auto transformation = absl::make_unique<toco::SwapElementwiseBinary>();
  ASSERT_TRUE(transformation->Run(&model, 1));

  Operator* op = GetOpWithOutput(model, "add");
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(OperatorType::kAdd, op->type);
  ASSERT_EQ(2, op->inputs.size());
  for (const string& input : op->inputs) {
    EXPECT_TRUE(IsConstantParameterArray(model, input))
        << input << " is not const input";
  }
}

}  // namespace toco

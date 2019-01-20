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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {
// A gmock matcher that check that elements of a float vector match to a given
// tolerance.
std::vector<testing::Matcher<float>> ArrayFloatNear(
    const std::vector<float>& values, float max_abs_error = 1e-5) {
  std::vector<testing::Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float& v : values) {
    matchers.emplace_back(testing::FloatNear(v, max_abs_error));
  }
  return matchers;
}
}  // namespace

class FuseBinaryIntoFollowingAffineTest : public ::testing::Test {
 protected:
  FuseBinaryIntoFollowingAffineTest() {}

  void SetUp() override { model_.reset(new Model); }

  void CreateArray(const string& name, const std::vector<int>& shape) {
    Array& array = model_->GetOrCreateArray(name);
    array.data_type = ArrayDataType::kFloat;
    Shape* array_shape = array.mutable_shape();
    *(array_shape->mutable_dims()) = shape;
  }

  void CreateConstantArray(const string& name, const std::vector<int>& shape,
                           const std::vector<float>& data) {
    CreateArray(name, shape);
    Array& array = model_->GetOrCreateArray(name);
    auto& array_buffer = array.GetMutableBuffer<ArrayDataType::kFloat>();
    int bufsize = 1;
    for (int dim : shape) {
      bufsize *= dim;
    }
    array_buffer.data.resize(bufsize);
    float* buf_ptr = array_buffer.data.data();
    for (int i = 0; i < bufsize; ++i) {
      buf_ptr[i] = data[i];
    }
  }

  std::unique_ptr<Model> model_;
};

TEST_F(FuseBinaryIntoFollowingAffineTest, FuseMulIntoFullyConnected) {
  // Creating a model.
  {
    CreateArray("Input", {2, 2});
    CreateConstantArray("MulInput2", {1}, {2.0});
    CreateArray("MulOutput", {2, 2});
    CreateConstantArray("FCWeight", {2, 2}, {1.0, 2.0, 3.0, 4.0});
    CreateConstantArray("FCBias", {1}, {1.0});
    CreateArray("Output", {2, 2});

    auto* mul_op = new MulOperator;
    mul_op->inputs = {"Input", "MulInput2"};
    mul_op->outputs = {"MulOutput"};
    model_->operators.push_back(std::unique_ptr<Operator>(mul_op));

    auto* fc_op = new FullyConnectedOperator;
    fc_op->inputs = {"MulOutput", "FCWeight", "FCBias"};
    fc_op->outputs = {"Output"};
    model_->operators.push_back(std::unique_ptr<Operator>(fc_op));
  }
  toco::FuseBinaryIntoFollowingAffine transformation;
  bool modified;
  ASSERT_TRUE(transformation.Run(model_.get(), /*op_index=*/0, &modified).ok());
  EXPECT_TRUE(modified);

  // `Mul` should be fused into `FullyConnected`. Only 1 op is left.
  ASSERT_EQ(model_->operators.size(), 1);
  const auto& op = model_->operators[0];
  ASSERT_EQ(op->type, OperatorType::kFullyConnected);
  ASSERT_EQ(op->inputs.size(), 3);

  auto& weights_array = model_->GetArray(op->inputs[1]);
  EXPECT_THAT(weights_array.GetBuffer<toco::ArrayDataType::kFloat>().data,
              ElementsAreArray(ArrayFloatNear({2.0, 4.0, 6.0, 8.0})));

  auto& bias_array = model_->GetArray(op->inputs[2]);
  EXPECT_THAT(bias_array.GetBuffer<toco::ArrayDataType::kFloat>().data,
              ElementsAreArray(ArrayFloatNear({1.0})));
}

// This is a regression test of b/121287325. Toco crashes before the fix.
TEST_F(FuseBinaryIntoFollowingAffineTest, DoNotFuseWithMultipleConsumers) {
  // Creating a model.
  {
    CreateArray("Input", {2, 2});
    CreateConstantArray("MulInput2", {1}, {2.0});
    CreateArray("MulOutput", {2, 2});
    CreateConstantArray("FCWeight", {2, 2}, {1.0, 2.0, 3.0, 4.0});
    CreateConstantArray("FCBias", {1}, {1.0});
    CreateArray("Output", {2, 2});
    CreateArray("AnotherOutput", {2, 2});

    auto* mul_op = new MulOperator;
    mul_op->inputs = {"Input", "MulInput2"};
    mul_op->outputs = {"MulOutput"};
    model_->operators.push_back(std::unique_ptr<Operator>(mul_op));

    auto* fc_op = new FullyConnectedOperator;
    fc_op->inputs = {"MulOutput", "FCWeight", "FCBias"};
    fc_op->outputs = {"Output"};
    model_->operators.push_back(std::unique_ptr<Operator>(fc_op));

    auto identity_op = new TensorFlowIdentityOperator;
    identity_op->inputs = {"MulOutput"};
    identity_op->outputs = {"AnotherOutput"};
    model_->operators.push_back(std::unique_ptr<Operator>(identity_op));
  }

  toco::FuseBinaryIntoFollowingAffine transformation;
  bool modified;
  ASSERT_TRUE(transformation.Run(model_.get(), /*op_index=*/0, &modified).ok());
  // Do not modify the graph if the binary operator has another output.
  EXPECT_FALSE(modified);
  EXPECT_EQ(model_->operators.size(), 3);
}

}  // namespace toco

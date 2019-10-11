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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace {

using ::testing::Test;

class RemoveSuccessiveTransposeTest : public Test {
 protected:
  RemoveSuccessiveTransposeTest() {}

  void SetUp() override { model_.reset(new toco::Model); }

  void CreateArray(const std::string& name, const std::vector<int>& shape) {
    toco::Array& array = model_->GetOrCreateArray(name);
    array.data_type = toco::ArrayDataType::kFloat;
    toco::Shape* array_shape = array.mutable_shape();
    *(array_shape->mutable_dims()) = shape;
  }

  void CreateConstantArray(const std::string& name,
                           const std::vector<int>& shape,
                           const std::vector<float>& data) {
    CreateArray(name, shape);
    toco::Array& array = model_->GetOrCreateArray(name);
    auto& array_buffer = array.GetMutableBuffer<toco::ArrayDataType::kFloat>();
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

  void CreateGraph(const std::vector<int>& perm1,
                   const std::vector<int>& perm2) {
    CreateArray("InputA", {2, 2});
    CreateArray("InputB", {2, 2});
    CreateArray("Input", {2, 2});
    CreateArray("InputTranspose", {2, 2});
    CreateArray("InputTransposeTranspose", {2, 2});
    CreateArray("InputTransposeTransposePlusB", {2, 2});

    auto* add_op = new toco::AddOperator;
    add_op->inputs = {"InputA", "InputB"};
    add_op->outputs = {"Input"};
    model_->operators.push_back(std::unique_ptr<toco::Operator>(add_op));

    auto* transpose_op = new toco::TransposeOperator;
    transpose_op->inputs = {"Input"};
    transpose_op->perm = perm1;
    transpose_op->outputs = {"InputTranspose"};
    model_->operators.push_back(std::unique_ptr<toco::Operator>(transpose_op));

    auto* transpose2_op = new toco::TransposeOperator;
    transpose2_op->inputs = {"InputTranspose"};
    transpose2_op->perm = perm2;
    transpose2_op->outputs = {"InputTransposeTranspose"};
    model_->operators.push_back(std::unique_ptr<toco::Operator>(transpose2_op));

    auto* add2_op = new toco::AddOperator;
    add2_op->inputs = {"InputTransposeTranspose", "InputB"};
    add2_op->outputs = {"InputTransposeTransposePlusB"};
    model_->operators.push_back(std::unique_ptr<toco::Operator>(add2_op));
  }

  std::unique_ptr<toco::Model> model_;
};

TEST_F(RemoveSuccessiveTransposeTest, RemoveTranspose) {
  // Creating a model.
  CreateGraph({1, 0}, {1, 0});

  toco::RemoveSuccesiveTranspose transformation;
  bool modified;
  ASSERT_TRUE(transformation.Run(model_.get(), /*op_index=*/1, &modified).ok());
  EXPECT_TRUE(modified);

  ASSERT_EQ(model_->operators.size(), 2);
  ASSERT_EQ(model_->operators[0]->type, toco::OperatorType::kAdd);
  ASSERT_EQ(model_->operators[1]->type, toco::OperatorType::kAdd);
  ASSERT_EQ(model_->operators[1]->inputs[0], model_->operators[0]->outputs[0]);
}

TEST_F(RemoveSuccessiveTransposeTest, DontRemoveNotIdentityTranspose) {
  // Creating a model.
  CreateGraph({0, 2, 1}, {1, 0, 2});

  toco::RemoveSuccesiveTranspose transformation;
  bool modified;
  ASSERT_TRUE(transformation.Run(model_.get(), /*op_index=*/1, &modified).ok());
  EXPECT_FALSE(modified);
}

TEST_F(RemoveSuccessiveTransposeTest, DontRemoveTransposeOutputUnused) {
  CreateArray("InputA", {2, 2});
  CreateArray("InputB", {2, 2});
  CreateArray("Input", {2, 2});
  CreateArray("InputTranspose", {2, 2});
  CreateArray("InputTransposeTranspose", {2, 2});

  auto* add_op = new toco::AddOperator;
  add_op->inputs = {"InputA", "InputB"};
  add_op->outputs = {"Input"};
  model_->operators.push_back(std::unique_ptr<toco::Operator>(add_op));

  auto* transpose_op = new toco::TransposeOperator;
  transpose_op->inputs = {"Input"};
  transpose_op->perm = {0, 2, 1};
  transpose_op->outputs = {"InputTranspose"};
  model_->operators.push_back(std::unique_ptr<toco::Operator>(transpose_op));

  auto* transpose2_op = new toco::TransposeOperator;
  transpose2_op->inputs = {"InputTranspose"};
  transpose2_op->perm = {0, 2, 1};
  transpose2_op->outputs = {"InputTransposeTranspose"};
  model_->operators.push_back(std::unique_ptr<toco::Operator>(transpose2_op));

  toco::RemoveSuccesiveTranspose transformation;
  bool modified;
  ASSERT_TRUE(transformation.Run(model_.get(), /*op_index=*/1, &modified).ok());
  EXPECT_FALSE(modified);
}
}  // namespace

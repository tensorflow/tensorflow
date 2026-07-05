/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"

namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::Test;

class ReorderReshapeTransposeTest : public Test {
 protected:
  void CreateArray(const std::string& name, const std::vector<int>& shape) {
    toco::Array& array = model_.GetOrCreateArray(name);
    array.data_type = toco::ArrayDataType::kFloat;
    toco::Shape* array_shape = array.mutable_shape();
    *(array_shape->mutable_dims()) = shape;
  }

  void CreateConstantInt32Array(const std::string& name,
                                const std::vector<int>& shape,
                                const std::vector<int>& data) {
    toco::Array& array = model_.GetOrCreateArray(name);
    array.data_type = toco::ArrayDataType::kInt32;
    toco::Shape* array_shape = array.mutable_shape();
    *(array_shape->mutable_dims()) = shape;
    array.GetMutableBuffer<toco::ArrayDataType::kInt32>().data = data;
  }

  void CreateGraph(const std::vector<int>& input_shape,
                   const std::vector<int>& reshape_shape,
                   const std::vector<int>& perm) {
    CreateArray("Input", input_shape);
    CreateConstantInt32Array("ReshapeShape",
                             {static_cast<int>(reshape_shape.size())},
                             reshape_shape);
    CreateArray("ReshapeOutput", reshape_shape);

    CreateConstantInt32Array("TransposePerm", {static_cast<int>(perm.size())},
                             perm);
    CreateArray("Output", reshape_shape);

    auto reshape_op = std::make_unique<toco::TensorFlowReshapeOperator>();
    reshape_op->inputs = {"Input", "ReshapeShape"};
    reshape_op->outputs = {"ReshapeOutput"};
    reshape_op->shape = reshape_shape;
    model_.operators.push_back(std::move(reshape_op));

    auto transpose_op = std::make_unique<toco::TransposeOperator>();
    transpose_op->inputs = {"ReshapeOutput", "TransposePerm"};
    transpose_op->outputs = {"Output"};
    transpose_op->perm = perm;
    model_.operators.push_back(std::move(transpose_op));
  }

  toco::Model model_;
};

TEST_F(ReorderReshapeTransposeTest, InvalidPermutationReturnsError) {
  CreateGraph({1, 4}, {1, 2, 2}, {0, 5, 1});

  toco::ReorderReshapeTranspose transformation;
  bool modified = false;
  EXPECT_THAT(transformation.Run(&model_, /*op_index=*/1, &modified),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ReorderReshapeTransposeTest, ValidPermutationSucceeds) {
  CreateGraph({1, 4}, {4, 1}, {1, 0});

  toco::ReorderReshapeTranspose transformation;
  bool modified = false;
  EXPECT_THAT(transformation.Run(&model_, /*op_index=*/1, &modified), IsOk());
  EXPECT_TRUE(modified);
}

}  // namespace

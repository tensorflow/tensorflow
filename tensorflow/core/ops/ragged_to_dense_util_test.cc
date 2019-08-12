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

#include "tensorflow/core/ops/ragged_to_dense_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(CombineRaggedTensorToTensorShapes, UnknownShapeUnknownValue) {
  TensorShapeProto shape_proto;
  shape_proto.set_unknown_rank(true);
  TensorShapeProto value_shape_proto;
  value_shape_proto.set_unknown_rank(true);
  int ragged_rank = 1;

  TensorShapeProto actual_output_shape_proto;
  TF_ASSERT_OK(CombineRaggedTensorToTensorShapes(
      ragged_rank, shape_proto, value_shape_proto, &actual_output_shape_proto));

  EXPECT_EQ(true, actual_output_shape_proto.unknown_rank());
}

TEST(CombineRaggedTensorToTensorShapes, UnknownShape) {
  TensorShapeProto shape_proto;
  shape_proto.set_unknown_rank(true);
  TensorShapeProto value_shape_proto;
  value_shape_proto.add_dim()->set_size(6);
  int ragged_rank = 1;

  TensorShapeProto actual_output_shape_proto;
  TF_ASSERT_OK(CombineRaggedTensorToTensorShapes(
      ragged_rank, shape_proto, value_shape_proto, &actual_output_shape_proto));

  ASSERT_EQ(actual_output_shape_proto.dim_size(), 2);
  EXPECT_EQ(actual_output_shape_proto.dim(0).size(), -1);
  EXPECT_EQ(actual_output_shape_proto.dim(1).size(), -1);
}

TEST(CombineRaggedTensorToTensorShapes, UnknownShapeDenseValue) {
  TensorShapeProto shape_proto;
  shape_proto.set_unknown_rank(true);
  TensorShapeProto value_shape_proto;
  value_shape_proto.add_dim()->set_size(6);
  value_shape_proto.add_dim()->set_size(3);
  int ragged_rank = 1;

  TensorShapeProto actual_output_shape_proto;
  TF_ASSERT_OK(CombineRaggedTensorToTensorShapes(
      ragged_rank, shape_proto, value_shape_proto, &actual_output_shape_proto));

  ASSERT_EQ(actual_output_shape_proto.dim_size(), 3);
  EXPECT_EQ(actual_output_shape_proto.dim(0).size(), -1);
  EXPECT_EQ(actual_output_shape_proto.dim(1).size(), -1);
  EXPECT_EQ(actual_output_shape_proto.dim(2).size(), 3);
}

TEST(GetRowPartitionTypesHelper, BasicTest) {
  const std::vector<string> row_partition_type_strings = {
      "FIRST_DIM_SIZE", "VALUE_ROWIDS", "ROW_SPLITS"};
  std::vector<RowPartitionType> row_partition_types;
  TF_ASSERT_OK(GetRowPartitionTypesHelper(row_partition_type_strings,
                                          &row_partition_types));
  EXPECT_THAT(row_partition_types,
              ::testing::ElementsAre(RowPartitionType::FIRST_DIM_SIZE,
                                     RowPartitionType::VALUE_ROWIDS,
                                     RowPartitionType::ROW_SPLITS));
}

TEST(RowPartitionTypeToString, BasicTest) {
  EXPECT_EQ("FIRST_DIM_SIZE",
            RowPartitionTypeToString(RowPartitionType::FIRST_DIM_SIZE));
  EXPECT_EQ("VALUE_ROWIDS",
            RowPartitionTypeToString(RowPartitionType::VALUE_ROWIDS));
  EXPECT_EQ("ROW_SPLITS",
            RowPartitionTypeToString(RowPartitionType::ROW_SPLITS));
}

TEST(ValidateDefaultValueShape, UnknownDefaultValueShape) {
  TensorShapeProto default_value_shape_proto;
  default_value_shape_proto.set_unknown_rank(true);
  TensorShapeProto value_shape_proto;
  value_shape_proto.add_dim()->set_size(6);
  TF_EXPECT_OK(
      ValidateDefaultValueShape(default_value_shape_proto, value_shape_proto));
}

TEST(ValidateDefaultValueShape, UnknownValueShape) {
  TensorShapeProto default_value_shape_proto;
  default_value_shape_proto.add_dim()->set_size(5);
  TensorShapeProto value_shape_proto;
  value_shape_proto.set_unknown_rank(true);
  TF_EXPECT_OK(
      ValidateDefaultValueShape(default_value_shape_proto, value_shape_proto));
}

TEST(ValidateDefaultValueShape, ScalarShape) {
  TensorShapeProto default_value_shape_proto;
  TensorShapeProto value_shape_proto;
  value_shape_proto.add_dim()->set_size(5);
  TF_EXPECT_OK(
      ValidateDefaultValueShape(default_value_shape_proto, value_shape_proto));
}

TEST(ValidateDefaultValueShape, TensorShapeEqual) {
  TensorShapeProto default_value_shape_proto;
  default_value_shape_proto.add_dim()->set_size(2);
  default_value_shape_proto.add_dim()->set_size(3);
  TensorShapeProto value_shape_proto;
  value_shape_proto.add_dim()->set_size(5);
  value_shape_proto.add_dim()->set_size(2);
  value_shape_proto.add_dim()->set_size(3);
  TF_EXPECT_OK(
      ValidateDefaultValueShape(default_value_shape_proto, value_shape_proto));
}

TEST(ValidateDefaultValueShape, TensorDimensionUnknown) {
  TensorShapeProto default_value_shape_proto;
  default_value_shape_proto.add_dim()->set_size(-1);
  default_value_shape_proto.add_dim()->set_size(3);
  TensorShapeProto value_shape_proto;
  value_shape_proto.add_dim()->set_size(5);
  value_shape_proto.add_dim()->set_size(2);
  value_shape_proto.add_dim()->set_size(3);
  TF_EXPECT_OK(
      ValidateDefaultValueShape(default_value_shape_proto, value_shape_proto));
}

TEST(ValidateDefaultValueShape, TensorDimensionUnknownForValue) {
  TensorShapeProto default_value_shape_proto;
  default_value_shape_proto.add_dim()->set_size(2);
  default_value_shape_proto.add_dim()->set_size(3);
  TensorShapeProto value_shape_proto;
  value_shape_proto.add_dim()->set_size(5);
  value_shape_proto.add_dim()->set_size(-1);
  value_shape_proto.add_dim()->set_size(3);
  TF_EXPECT_OK(
      ValidateDefaultValueShape(default_value_shape_proto, value_shape_proto));
}

TEST(ValidateDefaultValueShape, TensorDimensionFewDims) {
  TensorShapeProto default_value_shape_proto;
  default_value_shape_proto.add_dim()->set_size(3);
  TensorShapeProto value_shape_proto;
  value_shape_proto.add_dim()->set_size(5);
  value_shape_proto.add_dim()->set_size(-1);
  value_shape_proto.add_dim()->set_size(3);
  TF_EXPECT_OK(
      ValidateDefaultValueShape(default_value_shape_proto, value_shape_proto));
}

TEST(ValidateDefaultValueShape, WrongNumberOfDimensions) {
  // I have modified this test to make the default value shape have more
  // dimensions, instead of the same number.
  TensorShapeProto default_value_shape_proto;
  default_value_shape_proto.add_dim()->set_size(-1);
  default_value_shape_proto.add_dim()->set_size(-1);
  default_value_shape_proto.add_dim()->set_size(-1);
  TensorShapeProto value_shape_proto;
  value_shape_proto.add_dim()->set_size(-1);
  value_shape_proto.add_dim()->set_size(-1);
  EXPECT_FALSE(
      ValidateDefaultValueShape(default_value_shape_proto, value_shape_proto)
          .ok());
}

TEST(ValidateDefaultValueShape, WrongDimensionSize) {
  TensorShapeProto default_value_shape_proto;
  default_value_shape_proto.add_dim()->set_size(3);
  default_value_shape_proto.add_dim()->set_size(-1);
  TensorShapeProto value_shape_proto;
  value_shape_proto.add_dim()->set_size(5);
  value_shape_proto.add_dim()->set_size(6);
  value_shape_proto.add_dim()->set_size(-1);
  EXPECT_FALSE(
      ValidateDefaultValueShape(default_value_shape_proto, value_shape_proto)
          .ok());
}

// This is the case where broadcast could work, but we throw an error.
TEST(ValidateDefaultValueShape, WrongDimensionSizeBut1) {
  TensorShapeProto default_value_shape_proto;
  default_value_shape_proto.add_dim()->set_size(3);
  default_value_shape_proto.add_dim()->set_size(1);
  TensorShapeProto value_shape_proto;
  value_shape_proto.add_dim()->set_size(5);
  value_shape_proto.add_dim()->set_size(3);
  value_shape_proto.add_dim()->set_size(7);
  TF_EXPECT_OK(
      ValidateDefaultValueShape(default_value_shape_proto, value_shape_proto));
}

}  // namespace
}  // namespace tensorflow

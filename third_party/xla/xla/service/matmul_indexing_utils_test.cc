/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/matmul_indexing_utils.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

TEST(GetNonContractingDimsTest, Valid) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[1,2,3,4,5,6]"));
  EXPECT_THAT(GetNonContractingDims(shape, /*batch_dims=*/{4},
                                    /*contracting_dims=*/{1, 5}),
              absl_testing::IsOkAndHolds(ElementsAre(0, 2, 3)));
}

TEST(DotOperandDimsTest, Basic) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30,40,50,60]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0, 5},
                      /*non_contracting_dims=*/{1, 3},
                      /*contracting_dims=*/{2, 4});
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0, 5));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1, 3));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2, 4));
  EXPECT_EQ(dims.Rank(DotOperandDims::kBatch), 2);
  EXPECT_EQ(dims.Rank(DotOperandDims::kNonContracting), 2);
  EXPECT_EQ(dims.Rank(DotOperandDims::kContracting), 2);
  EXPECT_THAT(dims.Sizes(DotOperandDims::kBatch), ElementsAre(10, 60));
  EXPECT_THAT(dims.Sizes(DotOperandDims::kNonContracting), ElementsAre(20, 40));
  EXPECT_THAT(dims.Sizes(DotOperandDims::kContracting), ElementsAre(30, 50));
}

TEST(DotOperandDimsTest, IntoDotDimensionNumbers) {
  TF_ASSERT_OK_AND_ASSIGN(Shape lhs_shape,
                          ParseShape("f32[10,20,30,40,50,60]"));
  DotOperandDims lhs_dims(lhs_shape, /*batch_dims=*/{0, 5},
                          /*non_contracting_dims=*/{1, 3},
                          /*contracting_dims=*/{2, 4});

  // Batch dims must match in size, contracting dims must match in size.
  TF_ASSERT_OK_AND_ASSIGN(Shape rhs_shape,
                          ParseShape("f32[10,60,70,80,30,50]"));
  DotOperandDims rhs_dims(rhs_shape, /*batch_dims=*/{0, 1},
                          /*non_contracting_dims=*/{2, 3},
                          /*contracting_dims=*/{4, 5});

  TF_ASSERT_OK_AND_ASSIGN(
      DotDimensionNumbers ddn,
      DotOperandDims::CreateDotDimensionNumbers(lhs_dims, rhs_dims));

  EXPECT_THAT(ddn.lhs_batch_dimensions(), ElementsAre(0, 5));
  EXPECT_THAT(ddn.rhs_batch_dimensions(), ElementsAre(0, 1));
  EXPECT_THAT(ddn.lhs_contracting_dimensions(), ElementsAre(2, 4));
  EXPECT_THAT(ddn.rhs_contracting_dimensions(), ElementsAre(4, 5));
}

TEST(DotOperandDimsTest, IntoOutputShape) {
  TF_ASSERT_OK_AND_ASSIGN(Shape lhs_shape,
                          ParseShape("f32[10,20,30,40,50,60]"));
  DotOperandDims lhs_dims(lhs_shape, /*batch_dims=*/{0, 5},
                          /*non_contracting_dims=*/{1, 3},
                          /*contracting_dims=*/{2, 4});
  // Batch dims must match in size, contracting dims must match in size.
  TF_ASSERT_OK_AND_ASSIGN(Shape rhs_shape,
                          ParseShape("f32[10,60,70,80,30,50]"));
  DotOperandDims rhs_dims(rhs_shape, /*batch_dims=*/{0, 1},
                          /*non_contracting_dims=*/{2, 3},
                          /*contracting_dims=*/{4, 5});
  TF_ASSERT_OK_AND_ASSIGN(Shape output_shape,
                          DotOperandDims::ComputeOutputShape(
                              PrimitiveType::F32, lhs_dims, rhs_dims));
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_1,
                          ParseShape("f32[10,60,20,40,70,80]"));
  EXPECT_EQ(output_shape, expected_shape_1);
}

TEST(DotOperandDimsTest, ApplyPermutation) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30,40,50,60]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0, 5},
                      /*non_contracting_dims=*/{1, 3},
                      /*contracting_dims=*/{2, 4});
  dims.ApplyPermutation({4, 0, 2, 1, 3, 5});
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_2,
                          ParseShape("f32[50,10,30,20,40,60]{5,0,4,2,3,1}"));
  EXPECT_EQ(dims.shape(), expected_shape_2);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(1, 5));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(3, 4));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2, 0));
}

TEST(DotOperandDimsTest, CollapseRemoveIfEmpty) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[2,3,1,1,1,6]"));
  DotOperandDims dims(shape, /*batch_dims=*/{2, 3, 4},
                      /*non_contracting_dims=*/{0},
                      /*contracting_dims=*/{1, 5});
  TF_ASSERT_OK(dims.CollapseCategory(DotOperandDims::kBatch,
                                     /*remove_if_empty=*/true));
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_3, ParseShape("f32[2,3,6]"));
  EXPECT_EQ(dims.shape(), expected_shape_3);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre());
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(1, 2));
}

TEST(DotOperandDimsTest, CollapseKeepIfEmpty) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[2,3,1,1,1,6]"));
  DotOperandDims dims(shape, /*batch_dims=*/{2, 3, 4},
                      /*non_contracting_dims=*/{0},
                      /*contracting_dims=*/{1, 5});
  TF_ASSERT_OK(
      dims.CollapseCategory(DotOperandDims::kBatch, /*remove_if_empty=*/false));
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_4, ParseShape("f32[2,3,1,6]"));
  EXPECT_EQ(dims.shape(), expected_shape_4);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(2));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(1, 3));
}

TEST(DotOperandDimsTest, CollapseEmptyKeepIfEmpty) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[2,4,6]"));
  DotOperandDims dims(shape, /*batch_dims=*/{},
                      /*non_contracting_dims=*/{0},
                      /*contracting_dims=*/{1, 2});
  TF_ASSERT_OK(
      dims.CollapseCategory(DotOperandDims::kBatch, /*remove_if_empty=*/false));
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_5, ParseShape("f32[2,4,6]"));
  EXPECT_EQ(dims.shape(), expected_shape_5);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre());
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(1, 2));
}

TEST(DotOperandDimsTest, CollapseNormalCase) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,2,3,4,5,6]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1, 5},
                      /*contracting_dims=*/{2, 3, 4});
  TF_ASSERT_OK(dims.CollapseCategory(DotOperandDims::kContracting,
                                     /*remove_if_empty=*/false));
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_6, ParseShape("f32[10,2,60,6]"));
  EXPECT_EQ(dims.shape(), expected_shape_6);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1, 3));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2));
}

TEST(DotOperandDimsTest, CollapseErrorCases) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30,40,50,60]"));

  // 1. Unsorted consecutive dimensions (returns InvalidArgumentError)
  DotOperandDims unsorted_dims(shape, /*batch_dims=*/{4, 3, 2},
                               /*non_contracting_dims=*/{0, 1},
                               /*contracting_dims=*/{5});
  EXPECT_THAT(unsorted_dims.CollapseCategory(DotOperandDims::kBatch,
                                             /*remove_if_empty=*/false),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // 2. Sorted non-consecutive dimensions (returns InvalidArgumentError)
  DotOperandDims non_consecutive_dims(shape, /*batch_dims=*/{2, 4},
                                      /*non_contracting_dims=*/{0, 1},
                                      /*contracting_dims=*/{3, 5});
  EXPECT_THAT(non_consecutive_dims.CollapseCategory(DotOperandDims::kBatch,
                                                    /*remove_if_empty=*/false),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(DotOperandDimsTest, EraseDimensions) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30,40,50,60]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0, 5},
                      /*non_contracting_dims=*/{1, 3},
                      /*contracting_dims=*/{4, 2});
  TF_ASSERT_OK(dims.EraseDimensions(0, 4));
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_7, ParseShape("f32[50,60]"));
  EXPECT_EQ(dims.shape(), expected_shape_7);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(1));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre());
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(0));
}

TEST(DotOperandDimsTest, RemoveDegenerateDimensions) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[1,2,1,3,1]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0, 1, 2},
                      /*non_contracting_dims=*/{3},
                      /*contracting_dims=*/{4});
  TF_ASSERT_OK(dims.RemoveDegenerateDimensions());
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_8, ParseShape("f32[2,3]"));
  EXPECT_EQ(dims.shape(), expected_shape_8);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre());
}

TEST(DotOperandDimsTest, MergeAdjacentDimensions) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[2,3,5,7]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0, 1},
                      /*non_contracting_dims=*/{2},
                      /*contracting_dims=*/{3});
  TF_ASSERT_OK(dims.MergeAdjacentDimensions());
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_9, ParseShape("f32[6,5,7]"));
  EXPECT_EQ(dims.shape(), expected_shape_9);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2));

  // Case: [10, 20] batch_dims={1,0} - should not merge.
  TF_ASSERT_OK_AND_ASSIGN(Shape shape2, ParseShape("f32[10,20]"));
  DotOperandDims dims2(shape2, /*batch_dims=*/{1, 0},
                       /*non_contracting_dims=*/{},
                       /*contracting_dims=*/{});
  TF_ASSERT_OK(dims2.MergeAdjacentDimensions());
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_10, ParseShape("f32[10,20]"));
  EXPECT_EQ(dims2.shape(), expected_shape_10);
  EXPECT_THAT(dims2.Indices(DotOperandDims::kBatch), ElementsAre(1, 0));

  // Case: [10, 20, 30] batch_dims={0, 2, 1} - should not merge.
  TF_ASSERT_OK_AND_ASSIGN(Shape shape3, ParseShape("f32[10,20,30]"));
  DotOperandDims dims3(shape3, /*batch_dims=*/{0, 2, 1},
                       /*non_contracting_dims=*/{},
                       /*contracting_dims=*/{});
  TF_ASSERT_OK(dims3.MergeAdjacentDimensions());
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_11, ParseShape("f32[10,20,30]"));
  EXPECT_EQ(dims3.shape(), expected_shape_11);
  EXPECT_THAT(dims3.Indices(DotOperandDims::kBatch), ElementsAre(0, 2, 1));
}

TEST(DotOperandDimsTest, IndexWithinCategory) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30,40,50,60]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0, 5},
                      /*non_contracting_dims=*/{1, 3},
                      /*contracting_dims=*/{2, 4});
  EXPECT_THAT(dims.IndexWithinCategory(DotOperandDims::kBatch, 5),
              absl_testing::IsOkAndHolds(1));
  EXPECT_THAT(dims.IndexWithinCategory(DotOperandDims::kNonContracting, 1),
              absl_testing::IsOkAndHolds(0));
  EXPECT_FALSE(dims.IndexWithinCategory(DotOperandDims::kContracting, 0).ok());
}

TEST(DotOperandDimsTest, InsertDimensionIntoEmptyCategory) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30]"));
  DotOperandDims dims(shape, /*batch_dims=*/{},
                      /*non_contracting_dims=*/{0, 1},
                      /*contracting_dims=*/{2});
  TF_ASSERT_OK_AND_ASSIGN(int64_t idx,
                          dims.InsertDimension(DotOperandDims::kBatch, 3, 40));
  EXPECT_EQ(idx, 0);
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_12,
                          ParseShape("f32[10,20,30,40]"));
  EXPECT_EQ(dims.shape(), expected_shape_12);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(3));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(0, 1));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2));
}

TEST(DotOperandDimsTest, InsertDimensionIntoNonEmptyCategory) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{2});
  TF_ASSERT_OK_AND_ASSIGN(int64_t idx,
                          dims.InsertDimension(DotOperandDims::kBatch, 3, 40));
  EXPECT_EQ(idx, 1);
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_13,
                          ParseShape("f32[10,20,30,40]"));
  EXPECT_EQ(dims.shape(), expected_shape_13);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0, 3));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2));
}

TEST(DotOperandDimsTest, InsertDimensionIntoFirstCategory) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{2});
  TF_ASSERT_OK_AND_ASSIGN(int64_t idx,
                          dims.InsertDimension(DotOperandDims::kBatch, 0, 40));
  EXPECT_EQ(idx, 0);
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_14,
                          ParseShape("f32[40,10,20,30]"));
  EXPECT_EQ(dims.shape(), expected_shape_14);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0, 1));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(2));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(3));
}

TEST(DotOperandDimsTest, InsertDimensionIntoLastCategory) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{2});
  TF_ASSERT_OK_AND_ASSIGN(
      int64_t idx, dims.InsertDimension(DotOperandDims::kContracting, 3, 40));
  EXPECT_EQ(idx, 1);
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_15,
                          ParseShape("f32[10,20,30,40]"));
  EXPECT_EQ(dims.shape(), expected_shape_15);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2, 3));
}

TEST(DotOperandDimsTest, InsertDimensionIntoMiddleCategory) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{2});
  TF_ASSERT_OK_AND_ASSIGN(
      int64_t idx,
      dims.InsertDimension(DotOperandDims::kNonContracting, 1, 40));
  EXPECT_EQ(idx, 0);
  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_16,
                          ParseShape("f32[10,40,20,30]"));
  EXPECT_EQ(dims.shape(), expected_shape_16);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1, 2));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(3));
}

TEST(DotOperandDimsTest, InsertDimensionAtSpecificIndex) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{2});
  // Original batch: {0}.
  // Insert at index 0 (before the existing batch dim).
  TF_ASSERT_OK_AND_ASSIGN(
      int64_t idx, dims.InsertDimension(DotOperandDims::kBatch, 3, 40, 0));
  EXPECT_EQ(idx, 0);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(3, 0));
}

TEST(DotOperandDimsTest, MapBackwardThroughTranspose) {
  TF_ASSERT_OK_AND_ASSIGN(Shape operand_shape, ParseShape("f32[10,20,30]"));
  auto param = HloInstruction::CreateParameter(0, operand_shape, "param");

  TF_ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[30,10,20]"));
  auto transpose =
      HloInstruction::CreateTranspose(output_shape, param.get(), {2, 0, 1});

  DotOperandDims dims(output_shape, /*batch_dims=*/{1},
                      /*non_contracting_dims=*/{2},
                      /*contracting_dims=*/{0});

  TF_ASSERT_OK_AND_ASSIGN(auto mapped_dims_opt,
                          dims.MapBackward(transpose.get()));
  ASSERT_TRUE(mapped_dims_opt.has_value());

  EXPECT_THAT(mapped_dims_opt->Indices(DotOperandDims::kBatch), ElementsAre(0));
  EXPECT_THAT(mapped_dims_opt->Indices(DotOperandDims::kNonContracting),
              ElementsAre(1));
  EXPECT_THAT(mapped_dims_opt->Indices(DotOperandDims::kContracting),
              ElementsAre(2));
}

TEST(DotOperandDimsTest, MapBackwardThroughReshape) {
  TF_ASSERT_OK_AND_ASSIGN(Shape operand_shape, ParseShape("f32[4,4,10]"));
  auto param = HloInstruction::CreateParameter(0, operand_shape, "param");

  TF_ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[16,10]"));
  auto reshape = HloInstruction::CreateReshape(output_shape, param.get());

  DotOperandDims dims(output_shape, /*batch_dims=*/{1},
                      /*non_contracting_dims=*/{0},
                      /*contracting_dims=*/{});

  TF_ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.MapBackward(reshape.get()));
  ASSERT_TRUE(mapped_dims.has_value());

  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kBatch), ElementsAre(2));
  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kNonContracting),
              ElementsAre(0, 1));
  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kContracting),
              ElementsAre());
}

TEST(DotOperandDimsTest, MapBackwardThroughReshapeMixedCategories) {
  TF_ASSERT_OK_AND_ASSIGN(Shape operand_shape, ParseShape("f32[16]"));
  auto param = HloInstruction::CreateParameter(0, operand_shape, "param");

  TF_ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[4,4]"));
  auto reshape = HloInstruction::CreateReshape(output_shape, param.get());

  DotOperandDims dims(output_shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{});

  TF_ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.MapBackward(reshape.get()));
  EXPECT_FALSE(mapped_dims.has_value());
}

TEST(DotOperandDimsTest, MapBackwardThroughReshapeInsertSize1DimAllowed) {
  TF_ASSERT_OK_AND_ASSIGN(Shape operand_shape, ParseShape("f32[4]"));
  auto param = HloInstruction::CreateParameter(0, operand_shape, "param");

  TF_ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[1,4]"));
  auto reshape = HloInstruction::CreateReshape(output_shape, param.get());

  // Output dim 0 is Batch (size 1), dim 1 is NonContracting (size 4).
  // Reshape inserts size 1 dim at index 0.
  DotOperandDims dims(output_shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{});

  TF_ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.MapBackward(reshape.get()));
  ASSERT_TRUE(mapped_dims.has_value());

  // The inserted size 1 dim (output dim 0) corresponds to no input dim.
  // The input dim 0 corresponds to output dim 1 (NonContracting).
  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kBatch), ElementsAre());
  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kNonContracting),
              ElementsAre(0));
  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kContracting),
              ElementsAre());
}

TEST(DotOperandDimsTest, GetPermutedBackward) {
  TF_ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[30,10,20]"));
  DotOperandDims dims(output_shape, /*batch_dims=*/{1},
                      /*non_contracting_dims=*/{2},
                      /*contracting_dims=*/{0});

  std::vector<int64_t> permutation = {2, 0, 1};

  auto mapped_dims = dims.GetPermuted(InversePermutation(permutation));

  EXPECT_THAT(mapped_dims.Indices(DotOperandDims::kBatch), ElementsAre(0));
  EXPECT_THAT(mapped_dims.Indices(DotOperandDims::kNonContracting),
              ElementsAre(1));
  EXPECT_THAT(mapped_dims.Indices(DotOperandDims::kContracting),
              ElementsAre(2));
}

TEST(DotOperandDimsTest, GetPermutedForward) {
  TF_ASSERT_OK_AND_ASSIGN(Shape input_shape, ParseShape("f32[10,20,30]"));
  DotOperandDims dims(input_shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{2});

  std::vector<int64_t> permutation = {2, 0, 1};

  auto mapped_dims = dims.GetPermuted(permutation);

  EXPECT_THAT(mapped_dims.Indices(DotOperandDims::kBatch), ElementsAre(1));
  EXPECT_THAT(mapped_dims.Indices(DotOperandDims::kNonContracting),
              ElementsAre(2));
  EXPECT_THAT(mapped_dims.Indices(DotOperandDims::kContracting),
              ElementsAre(0));
}

TEST(DotOperandDimsTest, Reshape) {
  TF_ASSERT_OK_AND_ASSIGN(Shape input_shape, ParseShape("f32[4,4,10]"));
  DotOperandDims dims(input_shape, /*batch_dims=*/{2},
                      /*non_contracting_dims=*/{0, 1},
                      /*contracting_dims=*/{});

  TF_ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[16,10]"));

  TF_ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.Reshape(output_shape));
  ASSERT_TRUE(mapped_dims.has_value());

  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kBatch), ElementsAre(1));
  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kNonContracting),
              ElementsAre(0));
  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kContracting),
              ElementsAre());
}

TEST(DotOperandDimsTest, ReshapeDegenerateAbsorption) {
  // Reshaping [6, 35] batch={0}, contracting={1} into [1,2,1,3,1,5,1,7,1]
  // Expected: batch = {0,1,2,3}, contracting = {4,5,6,7,8}
  TF_ASSERT_OK_AND_ASSIGN(Shape input_shape, ParseShape("f32[6,35]"));
  DotOperandDims dims(input_shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{},
                      /*contracting_dims=*/{1});

  TF_ASSERT_OK_AND_ASSIGN(Shape output_shape,
                          ParseShape("f32[1,2,1,3,1,5,1,7,1]"));

  TF_ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.Reshape(output_shape));
  ASSERT_TRUE(mapped_dims.has_value());

  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kBatch),
              ElementsAre(0, 1, 2, 3));
  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kNonContracting),
              ElementsAre());
  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kContracting),
              ElementsAre(4, 5, 6, 7, 8));
}

TEST(DotOperandDimsTest, ReshapeBoundaryIntersectionError) {
  // Input: [6, 10] batch={0}, contracting={1}
  // Target: [4, 15] -> Product matches but 4 does not divide 6. Boundary
  // intersection.
  // Expected: returns std::nullopt.
  TF_ASSERT_OK_AND_ASSIGN(Shape input_shape, ParseShape("f32[6,10]"));
  DotOperandDims dims(input_shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{},
                      /*contracting_dims=*/{1});

  TF_ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[4,15]"));

  TF_ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.Reshape(output_shape));
  EXPECT_FALSE(mapped_dims.has_value());
}

TEST(DotOperandDimsTest, CreateShapeTrackerToSizeMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(Shape src_shape, ParseShape("f32[2,3,4,5]"));
  DotOperandDims src_dims(src_shape, /*batch_dims=*/{0},
                          /*non_contracting_dims=*/{1, 2},
                          /*contracting_dims=*/{3});

  TF_ASSERT_OK_AND_ASSIGN(Shape dst_shape, ParseShape("f32[2,3,5,5]"));
  DotOperandDims dst_dims(dst_shape, /*batch_dims=*/{0},
                          /*non_contracting_dims=*/{1, 2},
                          /*contracting_dims=*/{3});

  EXPECT_FALSE(src_dims.CreateShapeTrackerTo(dst_dims).ok());
}

TEST(DotOperandDimsTest, ApplyTransformationsFromBasic) {
  TF_ASSERT_OK_AND_ASSIGN(Shape src_before_shape, ParseShape("f32[2,3,4,5]"));
  DotOperandDims src_before(src_before_shape, /*batch_dims=*/{0},
                            /*non_contracting_dims=*/{1, 2},
                            /*contracting_dims=*/{3});

  TF_ASSERT_OK_AND_ASSIGN(Shape src_after_shape, ParseShape("f32[2,5,3,4]"));
  DotOperandDims src_after(src_after_shape, /*batch_dims=*/{0},
                           /*non_contracting_dims=*/{2, 3},
                           /*contracting_dims=*/{1});

  TF_ASSERT_OK_AND_ASSIGN(Shape current_shape, ParseShape("f32[2,7,5]"));
  DotOperandDims current(current_shape, /*batch_dims=*/{0},
                         /*non_contracting_dims=*/{1},
                         /*contracting_dims=*/{2});

  TF_ASSERT_OK_AND_ASSIGN(auto transformed, current.ApplyTransformationsFrom(
                                                src_before, src_after));

  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_17, ParseShape("f32[2,5,7]"));
  EXPECT_EQ(transformed.shape(), expected_shape_17);
  EXPECT_THAT(transformed.Indices(DotOperandDims::kBatch), ElementsAre(0));
  EXPECT_THAT(transformed.Indices(DotOperandDims::kContracting),
              ElementsAre(1));
  EXPECT_THAT(transformed.Indices(DotOperandDims::kNonContracting),
              ElementsAre(2));
}

TEST(DotOperandDimsTest, ApplyTransformationsFromMoreNCDims) {
  TF_ASSERT_OK_AND_ASSIGN(Shape src_before_shape, ParseShape("f32[2,5,3]"));
  DotOperandDims src_before(src_before_shape, /*batch_dims=*/{0},
                            /*non_contracting_dims=*/{2},
                            /*contracting_dims=*/{1});

  TF_ASSERT_OK_AND_ASSIGN(Shape src_after_shape, ParseShape("f32[3,2,5]"));
  DotOperandDims src_after(src_after_shape, /*batch_dims=*/{1},
                           /*non_contracting_dims=*/{0},
                           /*contracting_dims=*/{2});

  TF_ASSERT_OK_AND_ASSIGN(Shape current_shape, ParseShape("f32[2,7,8,5]"));
  DotOperandDims current(current_shape, /*batch_dims=*/{0},
                         /*non_contracting_dims=*/{1, 2},
                         /*contracting_dims=*/{3});

  TF_ASSERT_OK_AND_ASSIGN(auto transformed, current.ApplyTransformationsFrom(
                                                src_before, src_after));

  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape_18, ParseShape("f32[7,2,5,8]"));
  EXPECT_EQ(transformed.shape(), expected_shape_18);
  EXPECT_THAT(transformed.Indices(DotOperandDims::kBatch), ElementsAre(1));
  EXPECT_THAT(transformed.Indices(DotOperandDims::kContracting),
              ElementsAre(2));
  EXPECT_THAT(transformed.Indices(DotOperandDims::kNonContracting),
              ElementsAre(0, 3));
}

TEST(DotOperandDimsTest, ApplyTransformationsFromSizeMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(Shape src_before_shape, ParseShape("f32[2,3,4,5]"));
  DotOperandDims src_before(src_before_shape, /*batch_dims=*/{0},
                            /*non_contracting_dims=*/{1, 2},
                            /*contracting_dims=*/{3});

  TF_ASSERT_OK_AND_ASSIGN(Shape src_after_shape, ParseShape("f32[2,5,3,4]"));
  DotOperandDims src_after(src_after_shape, /*batch_dims=*/{0},
                           /*non_contracting_dims=*/{2, 3},
                           /*contracting_dims=*/{1});

  TF_ASSERT_OK_AND_ASSIGN(Shape current_shape, ParseShape("f32[3,7,5]"));
  DotOperandDims current(current_shape, /*batch_dims=*/{0},
                         /*non_contracting_dims=*/{1},
                         /*contracting_dims=*/{2});

  EXPECT_FALSE(current.ApplyTransformationsFrom(src_before, src_after).ok());
}

TEST(DotOperandDimsTest, ApplyTransformationsFromRankMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(Shape src_before_shape, ParseShape("f32[6,35,1]"));
  DotOperandDims src_before(src_before_shape, /*batch_dims=*/{0},
                            /*non_contracting_dims=*/{2},
                            /*contracting_dims=*/{1});

  TF_ASSERT_OK_AND_ASSIGN(Shape src_after_shape,
                          ParseShape("f32[1,2,5,1,7,3]"));
  DotOperandDims src_after(src_after_shape, /*batch_dims=*/{1, 3, 5},
                           /*non_contracting_dims=*/{0},
                           /*contracting_dims=*/{2, 4});

  TF_ASSERT_OK_AND_ASSIGN(Shape current_shape, ParseShape("f32[6,35]"));
  DotOperandDims current(current_shape, /*batch_dims=*/{0},
                         /*non_contracting_dims=*/{},
                         /*contracting_dims=*/{1});

  TF_ASSERT_OK_AND_ASSIGN(auto transformed, current.ApplyTransformationsFrom(
                                                src_before, src_after));

  TF_ASSERT_OK_AND_ASSIGN(Shape expected_shape, ParseShape("f32[1,2,5,1,7,3]"));
  EXPECT_EQ(transformed.shape(), expected_shape);
  EXPECT_THAT(transformed.Indices(DotOperandDims::kBatch),
              ElementsAre(1, 3, 5));
  EXPECT_THAT(transformed.Indices(DotOperandDims::kContracting),
              ElementsAre(2, 4));
  EXPECT_THAT(transformed.Indices(DotOperandDims::kNonContracting),
              ElementsAre(0));
}

TEST(DotOperandDimsTest, IsConsecutive) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30,40,50,60]"));

  // 1. Empty category (returns true)
  DotOperandDims empty_dims(shape, /*batch_dims=*/{},
                            /*non_contracting_dims=*/{0, 1},
                            /*contracting_dims=*/{2, 3, 4, 5});
  EXPECT_TRUE(empty_dims.IsConsecutive(DotOperandDims::kBatch));

  // 2. Single element (returns true)
  DotOperandDims single_dims(shape, /*batch_dims=*/{2},
                             /*non_contracting_dims=*/{0, 1},
                             /*contracting_dims=*/{3, 4, 5});
  EXPECT_TRUE(single_dims.IsConsecutive(DotOperandDims::kBatch));

  // 3. Sorted and consecutive elements (returns true)
  DotOperandDims sorted_consecutive_dims(shape, /*batch_dims=*/{2, 3, 4},
                                         /*non_contracting_dims=*/{0, 1},
                                         /*contracting_dims=*/{5});
  EXPECT_TRUE(sorted_consecutive_dims.IsConsecutive(DotOperandDims::kBatch));

  // 4. Consecutive but unsorted/descending elements (returns false)
  DotOperandDims unsorted_consecutive_dims(shape, /*batch_dims=*/{4, 3, 2},
                                           /*non_contracting_dims=*/{0, 1},
                                           /*contracting_dims=*/{5});
  EXPECT_FALSE(unsorted_consecutive_dims.IsConsecutive(DotOperandDims::kBatch));

  // 5. Sorted but non-consecutive elements (returns false)
  DotOperandDims sorted_non_consecutive_dims(shape, /*batch_dims=*/{2, 4},
                                             /*non_contracting_dims=*/{0, 1},
                                             /*contracting_dims=*/{3, 5});
  EXPECT_FALSE(
      sorted_non_consecutive_dims.IsConsecutive(DotOperandDims::kBatch));
}

}  // namespace
}  // namespace xla

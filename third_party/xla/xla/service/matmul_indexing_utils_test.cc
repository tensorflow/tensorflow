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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

TEST(GetNonContractingDimsTest, Valid) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[1,2,3,4,5,6]"));
  EXPECT_THAT(GetNonContractingDims(shape, /*batch_dims=*/{4},
                                    /*contracting_dims=*/{1, 5}),
              absl_testing::IsOkAndHolds(ElementsAre(0, 2, 3)));
}

TEST(DotOperandDimsTest, Basic) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30,40,50,60]"));
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

TEST(DotOperandDimsTest, Categories) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30,40,50,60]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0, 5},
                      /*non_contracting_dims=*/{1, 3},
                      /*contracting_dims=*/{2, 4});
  EXPECT_THAT(dims.Categories(),
              ElementsAre(DotOperandDims::kBatch,           // 0
                          DotOperandDims::kNonContracting,  // 1
                          DotOperandDims::kContracting,     // 2
                          DotOperandDims::kNonContracting,  // 3
                          DotOperandDims::kContracting,     // 4
                          DotOperandDims::kBatch            // 5
                          ));
}

TEST(DotOperandDimsTest, IntoDotDimensionNumbers) {
  ASSERT_OK_AND_ASSIGN(Shape lhs_shape, ParseShape("f32[10,20,30,40,50,60]"));
  DotOperandDims lhs_dims(lhs_shape, /*batch_dims=*/{0, 5},
                          /*non_contracting_dims=*/{1, 3},
                          /*contracting_dims=*/{2, 4});

  // Batch dims must match in size, contracting dims must match in size.
  ASSERT_OK_AND_ASSIGN(Shape rhs_shape, ParseShape("f32[10,60,70,80,30,50]"));
  DotOperandDims rhs_dims(rhs_shape, /*batch_dims=*/{0, 1},
                          /*non_contracting_dims=*/{2, 3},
                          /*contracting_dims=*/{4, 5});

  ASSERT_OK_AND_ASSIGN(
      DotDimensionNumbers ddn,
      DotOperandDims::CreateDotDimensionNumbers(lhs_dims, rhs_dims));

  EXPECT_THAT(ddn.lhs_batch_dimensions(), ElementsAre(0, 5));
  EXPECT_THAT(ddn.rhs_batch_dimensions(), ElementsAre(0, 1));
  EXPECT_THAT(ddn.lhs_contracting_dimensions(), ElementsAre(2, 4));
  EXPECT_THAT(ddn.rhs_contracting_dimensions(), ElementsAre(4, 5));
}

TEST(DotOperandDimsTest, IntoOutputShape) {
  ASSERT_OK_AND_ASSIGN(Shape lhs_shape, ParseShape("f32[10,20,30,40,50,60]"));
  DotOperandDims lhs_dims(lhs_shape, /*batch_dims=*/{0, 5},
                          /*non_contracting_dims=*/{1, 3},
                          /*contracting_dims=*/{2, 4});
  // Batch dims must match in size, contracting dims must match in size.
  ASSERT_OK_AND_ASSIGN(Shape rhs_shape, ParseShape("f32[10,60,70,80,30,50]"));
  DotOperandDims rhs_dims(rhs_shape, /*batch_dims=*/{0, 1},
                          /*non_contracting_dims=*/{2, 3},
                          /*contracting_dims=*/{4, 5});
  ASSERT_OK_AND_ASSIGN(Shape output_shape,
                       DotOperandDims::ComputeOutputShape(PrimitiveType::F32,
                                                          lhs_dims, rhs_dims));
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_1,
                       ParseShape("f32[10,60,20,40,70,80]"));
  EXPECT_EQ(output_shape, expected_shape_1);
}

TEST(DotOperandDimsTest, ApplyPermutation) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30,40,50,60]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0, 5},
                      /*non_contracting_dims=*/{1, 3},
                      /*contracting_dims=*/{2, 4});
  dims.ApplyPermutation({4, 0, 2, 1, 3, 5});
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_2,
                       ParseShape("f32[50,10,30,20,40,60]{5,0,4,2,3,1}"));
  EXPECT_EQ(dims.shape(), expected_shape_2);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(1, 5));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(3, 4));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2, 0));
}

TEST(DotOperandDimsTest, CollapseRemoveIfEmpty) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[2,3,1,1,1,6]"));
  DotOperandDims dims(shape, /*batch_dims=*/{2, 3, 4},
                      /*non_contracting_dims=*/{0},
                      /*contracting_dims=*/{1, 5});
  ASSERT_OK(dims.CollapseCategory(DotOperandDims::kBatch,
                                  /*remove_if_empty=*/true));
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_3, ParseShape("f32[2,3,6]"));
  EXPECT_EQ(dims.shape(), expected_shape_3);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre());
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(1, 2));
}

TEST(DotOperandDimsTest, CollapseKeepIfEmpty) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[2,3,1,1,1,6]"));
  DotOperandDims dims(shape, /*batch_dims=*/{2, 3, 4},
                      /*non_contracting_dims=*/{0},
                      /*contracting_dims=*/{1, 5});
  ASSERT_OK(
      dims.CollapseCategory(DotOperandDims::kBatch, /*remove_if_empty=*/false));
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_4, ParseShape("f32[2,3,1,6]"));
  EXPECT_EQ(dims.shape(), expected_shape_4);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(2));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(1, 3));
}

TEST(DotOperandDimsTest, CollapseEmptyKeepIfEmpty) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[2,4,6]"));
  DotOperandDims dims(shape, /*batch_dims=*/{},
                      /*non_contracting_dims=*/{0},
                      /*contracting_dims=*/{1, 2});
  ASSERT_OK(
      dims.CollapseCategory(DotOperandDims::kBatch, /*remove_if_empty=*/false));
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_5, ParseShape("f32[2,4,6]"));
  EXPECT_EQ(dims.shape(), expected_shape_5);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre());
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(1, 2));
}

TEST(DotOperandDimsTest, CollapseNormalCase) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,2,3,4,5,6]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1, 5},
                      /*contracting_dims=*/{2, 3, 4});
  ASSERT_OK(dims.CollapseCategory(DotOperandDims::kContracting,
                                  /*remove_if_empty=*/false));
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_6, ParseShape("f32[10,2,60,6]"));
  EXPECT_EQ(dims.shape(), expected_shape_6);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1, 3));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2));
}

TEST(DotOperandDimsTest, CollapseErrorCases) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30,40,50,60]"));

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
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30,40,50,60]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0, 5},
                      /*non_contracting_dims=*/{1, 3},
                      /*contracting_dims=*/{4, 2});
  ASSERT_OK(dims.EraseDimensions(0, 4));
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_7, ParseShape("f32[50,60]"));
  EXPECT_EQ(dims.shape(), expected_shape_7);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(1));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre());
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(0));
}

TEST(DotOperandDimsTest, RemoveDegenerateDimensions) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[1,2,1,3,1]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0, 1, 2},
                      /*non_contracting_dims=*/{3},
                      /*contracting_dims=*/{4});
  ASSERT_OK(dims.RemoveDegenerateDimensions());
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_8, ParseShape("f32[2,3]"));
  EXPECT_EQ(dims.shape(), expected_shape_8);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre());
}

TEST(DotOperandDimsTest, MergeAdjacentDimensions) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[2,3,5,7]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0, 1},
                      /*non_contracting_dims=*/{2},
                      /*contracting_dims=*/{3});
  ASSERT_OK(dims.MergeAdjacentDimensions());
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_9, ParseShape("f32[6,5,7]"));
  EXPECT_EQ(dims.shape(), expected_shape_9);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2));

  // Case: [10, 20] batch_dims={1,0} - should not merge.
  ASSERT_OK_AND_ASSIGN(Shape shape2, ParseShape("f32[10,20]"));
  DotOperandDims dims2(shape2, /*batch_dims=*/{1, 0},
                       /*non_contracting_dims=*/{},
                       /*contracting_dims=*/{});
  ASSERT_OK(dims2.MergeAdjacentDimensions());
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_10, ParseShape("f32[10,20]"));
  EXPECT_EQ(dims2.shape(), expected_shape_10);
  EXPECT_THAT(dims2.Indices(DotOperandDims::kBatch), ElementsAre(1, 0));

  // Case: [10, 20, 30] batch_dims={0, 2, 1} - should not merge.
  ASSERT_OK_AND_ASSIGN(Shape shape3, ParseShape("f32[10,20,30]"));
  DotOperandDims dims3(shape3, /*batch_dims=*/{0, 2, 1},
                       /*non_contracting_dims=*/{},
                       /*contracting_dims=*/{});
  ASSERT_OK(dims3.MergeAdjacentDimensions());
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_11, ParseShape("f32[10,20,30]"));
  EXPECT_EQ(dims3.shape(), expected_shape_11);
  EXPECT_THAT(dims3.Indices(DotOperandDims::kBatch), ElementsAre(0, 2, 1));
}

TEST(DotOperandDimsTest, IndexWithinCategory) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30,40,50,60]"));
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
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30]"));
  DotOperandDims dims(shape, /*batch_dims=*/{},
                      /*non_contracting_dims=*/{0, 1},
                      /*contracting_dims=*/{2});
  ASSERT_OK_AND_ASSIGN(int64_t idx,
                       dims.InsertDimension(DotOperandDims::kBatch, 3, 40));
  EXPECT_EQ(idx, 0);
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_12, ParseShape("f32[10,20,30,40]"));
  EXPECT_EQ(dims.shape(), expected_shape_12);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(3));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(0, 1));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2));
}

TEST(DotOperandDimsTest, InsertDimensionIntoNonEmptyCategory) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{2});
  ASSERT_OK_AND_ASSIGN(int64_t idx,
                       dims.InsertDimension(DotOperandDims::kBatch, 3, 40));
  EXPECT_EQ(idx, 1);
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_13, ParseShape("f32[10,20,30,40]"));
  EXPECT_EQ(dims.shape(), expected_shape_13);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0, 3));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2));
}

TEST(DotOperandDimsTest, InsertDimensionIntoFirstCategory) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{2});
  ASSERT_OK_AND_ASSIGN(int64_t idx,
                       dims.InsertDimension(DotOperandDims::kBatch, 0, 40));
  EXPECT_EQ(idx, 0);
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_14, ParseShape("f32[40,10,20,30]"));
  EXPECT_EQ(dims.shape(), expected_shape_14);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0, 1));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(2));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(3));
}

TEST(DotOperandDimsTest, InsertDimensionIntoLastCategory) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{2});
  ASSERT_OK_AND_ASSIGN(
      int64_t idx, dims.InsertDimension(DotOperandDims::kContracting, 3, 40));
  EXPECT_EQ(idx, 1);
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_15, ParseShape("f32[10,20,30,40]"));
  EXPECT_EQ(dims.shape(), expected_shape_15);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2, 3));
}

TEST(DotOperandDimsTest, InsertDimensionIntoMiddleCategory) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{2});
  ASSERT_OK_AND_ASSIGN(
      int64_t idx,
      dims.InsertDimension(DotOperandDims::kNonContracting, 1, 40));
  EXPECT_EQ(idx, 0);
  ASSERT_OK_AND_ASSIGN(Shape expected_shape_16, ParseShape("f32[10,40,20,30]"));
  EXPECT_EQ(dims.shape(), expected_shape_16);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1, 2));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(3));
}

TEST(DotOperandDimsTest, InsertDimensionAtSpecificIndex) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{2});
  // Original batch: {0}.
  // Insert at index 0 (before the existing batch dim).
  ASSERT_OK_AND_ASSIGN(int64_t idx,
                       dims.InsertDimension(DotOperandDims::kBatch, 3, 40, 0));
  EXPECT_EQ(idx, 0);
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(3, 0));
}

TEST(DotOperandDimsTest, PermuteToConsecutive) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30,40,50,60]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0, 5},
                      /*non_contracting_dims=*/{1, 3},
                      /*contracting_dims=*/{2, 4});

  auto permutation = dims.PermuteToConsecutive(DotOperandDims::kContracting);
  EXPECT_TRUE(permutation.has_value());
  EXPECT_THAT(*permutation, ElementsAre(0, 1, 3, 5, 2, 4));

  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0, 3));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1, 2));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(4, 5));
}

TEST(DotOperandDimsTest, PermuteToConsecutiveAlreadyConsecutive) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30,40,50,60]"));
  DotOperandDims dims(shape, /*batch_dims=*/{0, 5},
                      /*non_contracting_dims=*/{1, 2},
                      /*contracting_dims=*/{3, 4});

  auto permutation = dims.PermuteToConsecutive(DotOperandDims::kContracting);
  EXPECT_FALSE(permutation.has_value());
}

TEST(DotOperandDimsTest, PermuteToConsecutiveUnsorted) {
  Shape shape = ParseShape("f32[10,20,30,40,50,60]").value();
  DotOperandDims dims(shape, /*batch_dims=*/{0, 5},
                      /*non_contracting_dims=*/{1, 2},
                      /*contracting_dims=*/{4, 3});

  auto permutation = dims.PermuteToConsecutive(DotOperandDims::kContracting);
  ASSERT_TRUE(permutation.has_value());
  EXPECT_THAT(*permutation, ElementsAre(0, 1, 2, 5, 4, 3));

  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0, 3));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1, 2));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(4, 5));
}

TEST(DotOperandDimsTest, MapBackwardThroughTranspose) {
  ASSERT_OK_AND_ASSIGN(Shape operand_shape, ParseShape("f32[10,20,30]"));
  auto param = HloInstruction::CreateParameter(0, operand_shape, "param");

  ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[30,10,20]"));
  auto transpose =
      HloInstruction::CreateTranspose(output_shape, param.get(), {2, 0, 1});

  DotOperandDims dims(output_shape, /*batch_dims=*/{1},
                      /*non_contracting_dims=*/{2},
                      /*contracting_dims=*/{0});

  ASSERT_OK_AND_ASSIGN(auto mapped_dims_opt, dims.MapBackward(transpose.get()));
  ASSERT_TRUE(mapped_dims_opt.has_value());

  EXPECT_THAT(mapped_dims_opt->Indices(DotOperandDims::kBatch), ElementsAre(0));
  EXPECT_THAT(mapped_dims_opt->Indices(DotOperandDims::kNonContracting),
              ElementsAre(1));
  EXPECT_THAT(mapped_dims_opt->Indices(DotOperandDims::kContracting),
              ElementsAre(2));
}

TEST(DotOperandDimsTest, MapBackwardThroughReshape) {
  ASSERT_OK_AND_ASSIGN(Shape operand_shape, ParseShape("f32[4,4,10]"));
  auto param = HloInstruction::CreateParameter(0, operand_shape, "param");

  ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[16,10]"));
  auto reshape = HloInstruction::CreateReshape(output_shape, param.get());

  DotOperandDims dims(output_shape, /*batch_dims=*/{1},
                      /*non_contracting_dims=*/{0},
                      /*contracting_dims=*/{});

  ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.MapBackward(reshape.get()));
  ASSERT_TRUE(mapped_dims.has_value());

  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kBatch), ElementsAre(2));
  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kNonContracting),
              ElementsAre(0, 1));
  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kContracting),
              ElementsAre());
}

TEST(DotOperandDimsTest, MapBackwardThroughReshapeMixedCategories) {
  ASSERT_OK_AND_ASSIGN(Shape operand_shape, ParseShape("f32[16]"));
  auto param = HloInstruction::CreateParameter(0, operand_shape, "param");

  ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[4,4]"));
  auto reshape = HloInstruction::CreateReshape(output_shape, param.get());

  DotOperandDims dims(output_shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{});

  ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.MapBackward(reshape.get()));
  EXPECT_FALSE(mapped_dims.has_value());
}

TEST(DotOperandDimsTest, MapBackwardThroughReshapeInsertSize1DimAllowed) {
  ASSERT_OK_AND_ASSIGN(Shape operand_shape, ParseShape("f32[4]"));
  auto param = HloInstruction::CreateParameter(0, operand_shape, "param");

  ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[1,4]"));
  auto reshape = HloInstruction::CreateReshape(output_shape, param.get());

  // Output dim 0 is Batch (size 1), dim 1 is NonContracting (size 4).
  // Reshape inserts size 1 dim at index 0.
  DotOperandDims dims(output_shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{});

  ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.MapBackward(reshape.get()));
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
  ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[30,10,20]"));
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
  ASSERT_OK_AND_ASSIGN(Shape input_shape, ParseShape("f32[10,20,30]"));
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
  ASSERT_OK_AND_ASSIGN(Shape input_shape, ParseShape("f32[4,4,10]"));
  DotOperandDims dims(input_shape, /*batch_dims=*/{2},
                      /*non_contracting_dims=*/{0, 1},
                      /*contracting_dims=*/{});

  ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[16,10]"));

  ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.Reshape(output_shape));
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
  ASSERT_OK_AND_ASSIGN(Shape input_shape, ParseShape("f32[6,35]"));
  DotOperandDims dims(input_shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{},
                      /*contracting_dims=*/{1});

  ASSERT_OK_AND_ASSIGN(Shape output_shape,
                       ParseShape("f32[1,2,1,3,1,5,1,7,1]"));

  ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.Reshape(output_shape));
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
  ASSERT_OK_AND_ASSIGN(Shape input_shape, ParseShape("f32[6,10]"));
  DotOperandDims dims(input_shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{},
                      /*contracting_dims=*/{1});

  ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[4,15]"));

  ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.Reshape(output_shape));
  EXPECT_FALSE(mapped_dims.has_value());
}

TEST(DotOperandDimsTest, ReshapeConsecutiveButNotSorted) {
  // Input: [10, 20] batch={1, 0} (consecutive but unsorted)
  // Target: [200]
  // Expected: returns std::nullopt (cannot reshape/merge unsorted dims)
  ASSERT_OK_AND_ASSIGN(Shape input_shape, ParseShape("f32[10,20]"));
  DotOperandDims dims(input_shape, /*batch_dims=*/{1, 0},
                      /*non_contracting_dims=*/{},
                      /*contracting_dims=*/{});

  ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[200]"));

  ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.Reshape(output_shape));
  EXPECT_FALSE(mapped_dims.has_value());
}

TEST(DotOperandDimsTest, IsConsecutive) {
  ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("f32[10,20,30,40,50,60]"));

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

TEST(DotOperandDimsTest, MapForwardThroughTranspose) {
  ASSERT_OK_AND_ASSIGN(Shape input_shape, ParseShape("f32[10,20,30]"));
  auto param = HloInstruction::CreateParameter(0, input_shape, "param");

  ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[30,10,20]"));
  auto transpose =
      HloInstruction::CreateTranspose(output_shape, param.get(), {2, 0, 1});

  DotOperandDims dims(input_shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1},
                      /*contracting_dims=*/{2});

  ASSERT_OK_AND_ASSIGN(auto mapped_dims_opt, dims.MapForward(transpose.get()));
  ASSERT_TRUE(mapped_dims_opt.has_value());

  EXPECT_THAT(mapped_dims_opt->Indices(DotOperandDims::kBatch), ElementsAre(1));
  EXPECT_THAT(mapped_dims_opt->Indices(DotOperandDims::kNonContracting),
              ElementsAre(2));
  EXPECT_THAT(mapped_dims_opt->Indices(DotOperandDims::kContracting),
              ElementsAre(0));
}

TEST(DotOperandDimsTest, MapForwardThroughReshape) {
  ASSERT_OK_AND_ASSIGN(Shape input_shape, ParseShape("f32[4,4,10]"));
  auto param = HloInstruction::CreateParameter(0, input_shape, "param");

  ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[16,10]"));
  auto reshape = HloInstruction::CreateReshape(output_shape, param.get());

  DotOperandDims dims(input_shape, /*batch_dims=*/{2},
                      /*non_contracting_dims=*/{0, 1},
                      /*contracting_dims=*/{});

  ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.MapForward(reshape.get()));
  ASSERT_TRUE(mapped_dims.has_value());

  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kBatch), ElementsAre(1));
  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kNonContracting),
              ElementsAre(0));
  EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kContracting),
              ElementsAre());
}

TEST(DotOperandDimsTest, ReshapeScalarToDegenerate) {
  // Scalar input
  {
    ASSERT_OK_AND_ASSIGN(Shape input_shape, ParseShape("f32[]"));
    DotOperandDims dims(input_shape, /*batch_dims=*/{},
                        /*non_contracting_dims=*/{},
                        /*contracting_dims=*/{});

    ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[1,1,1]"));

    ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.Reshape(output_shape));
    ASSERT_TRUE(mapped_dims.has_value());

    EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kBatch), ElementsAre());
    EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kNonContracting),
                ElementsAre(0, 1, 2));
    EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kContracting),
                ElementsAre());
  }

  // 1D degenerate input
  {
    ASSERT_OK_AND_ASSIGN(Shape input_shape, ParseShape("f32[1]"));
    DotOperandDims dims(input_shape, /*batch_dims=*/{0},
                        /*non_contracting_dims=*/{},
                        /*contracting_dims=*/{});

    ASSERT_OK_AND_ASSIGN(Shape output_shape, ParseShape("f32[1,1,1]"));

    ASSERT_OK_AND_ASSIGN(auto mapped_dims, dims.Reshape(output_shape));
    ASSERT_TRUE(mapped_dims.has_value());

    EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kBatch), ElementsAre());
    EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kNonContracting),
                ElementsAre(0, 1, 2));
    EXPECT_THAT(mapped_dims->Indices(DotOperandDims::kContracting),
                ElementsAre());
  }
}

}  // namespace
}  // namespace xla

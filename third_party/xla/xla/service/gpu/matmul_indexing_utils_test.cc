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

#include "xla/service/gpu/matmul_indexing_utils.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/shape.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::tsl::testing::IsOkAndHolds;

TEST(GetNonContractingDimsTest, Valid) {
  Shape shape = ParseShape("f32[1,2,3,4,5,6]").value();
  EXPECT_THAT(GetNonContractingDims(shape, /*batch_dims=*/{4},
                                    /*contracting_dims=*/{1, 5}),
              IsOkAndHolds(ElementsAre(0, 2, 3)));
}

TEST(DotOperandDimsTest, Basic) {
  Shape shape = ParseShape("f32[10,20,30,40,50,60]").value();
  DotOperandDims dims(shape, /*batch_dims=*/{0, 5},
                      /*non_contracting_dims=*/{1, 3},
                      /*contracting_dims=*/{2, 4});
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0, 5));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1, 3));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2, 4));
  EXPECT_EQ(dims.DimensionCount(DotOperandDims::kBatch), 2);
  EXPECT_EQ(dims.DimensionCount(DotOperandDims::kNonContracting), 2);
  EXPECT_EQ(dims.DimensionCount(DotOperandDims::kContracting), 2);
  EXPECT_THAT(dims.DimensionSizes(DotOperandDims::kBatch), ElementsAre(10, 60));
  EXPECT_THAT(dims.DimensionSizes(DotOperandDims::kNonContracting),
              ElementsAre(20, 40));
  EXPECT_THAT(dims.DimensionSizes(DotOperandDims::kContracting),
              ElementsAre(30, 50));
}

TEST(DotOperandDimsTest, IntoDotDimensionNumbers) {
  Shape lhs_shape = ParseShape("f32[10,20,30,40,50,60]").value();
  DotOperandDims lhs_dims(lhs_shape, /*batch_dims=*/{0, 5},
                          /*non_contracting_dims=*/{1, 3},
                          /*contracting_dims=*/{2, 4});

  // Batch dims must match in size, contracting dims must match in size.
  Shape rhs_shape = ParseShape("f32[10,60,70,80,30,50]").value();
  DotOperandDims rhs_dims(rhs_shape, /*batch_dims=*/{0, 1},
                          /*non_contracting_dims=*/{2, 3},
                          /*contracting_dims=*/{4, 5});

  TF_ASSERT_OK_AND_ASSIGN(
      DotDimensionNumbers ddn,
      DotOperandDims::IntoDotDimensionNumbers(lhs_dims, rhs_dims));

  EXPECT_THAT(ddn.lhs_batch_dimensions(), ElementsAre(0, 5));
  EXPECT_THAT(ddn.rhs_batch_dimensions(), ElementsAre(0, 1));
  EXPECT_THAT(ddn.lhs_contracting_dimensions(), ElementsAre(2, 4));
  EXPECT_THAT(ddn.rhs_contracting_dimensions(), ElementsAre(4, 5));
}

TEST(DotOperandDimsTest, IntoOutputShape) {
  Shape lhs_shape = ParseShape("f32[10,20,30,40,50,60]").value();
  DotOperandDims lhs_dims(lhs_shape, /*batch_dims=*/{0, 5},
                          /*non_contracting_dims=*/{1, 3},
                          /*contracting_dims=*/{2, 4});
  // Batch dims must match in size, contracting dims must match in size.
  Shape rhs_shape = ParseShape("f32[10,60,70,80,30,50]").value();
  DotOperandDims rhs_dims(rhs_shape, /*batch_dims=*/{0, 1},
                          /*non_contracting_dims=*/{2, 3},
                          /*contracting_dims=*/{4, 5});
  TF_ASSERT_OK_AND_ASSIGN(
      Shape output_shape,
      DotOperandDims::IntoOutputShape(PrimitiveType::F32, lhs_dims, rhs_dims));
  EXPECT_EQ(output_shape, ParseShape("f32[10,60,20,40,70,80]").value());
}

TEST(DotOperandDimsTest, Permute) {
  Shape shape = ParseShape("f32[10,20,30,40,50,60]").value();
  DotOperandDims dims(shape, /*batch_dims=*/{0, 5},
                      /*non_contracting_dims=*/{1, 3},
                      /*contracting_dims=*/{2, 4});
  dims.Permute({4, 0, 2, 1, 3, 5});
  EXPECT_EQ(dims.shape(),
            ParseShape("f32[50,10,30,20,40,60]{5,0,4,2,3,1}").value());
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(1, 5));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(3, 4));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2, 0));
}

TEST(DotOperandDimsTest, CollapseRemoveIfEmpty) {
  Shape shape = ParseShape("f32[2,3,1,1,1,6]").value();
  DotOperandDims dims(shape, /*batch_dims=*/{3, 2, 4},
                      /*non_contracting_dims=*/{0},
                      /*contracting_dims=*/{1, 5});
  TF_ASSERT_OK(dims.Collapse(DotOperandDims::kBatch, /*remove_if_empty=*/true));
  EXPECT_EQ(dims.shape(), ParseShape("f32[2,3,6]").value());
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre());
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(1, 2));
}

TEST(DotOperandDimsTest, CollapseKeepIfEmpty) {
  Shape shape = ParseShape("f32[2,3,1,1,1,6]").value();
  DotOperandDims dims(shape, /*batch_dims=*/{3, 2, 4},
                      /*non_contracting_dims=*/{0},
                      /*contracting_dims=*/{1, 5});
  TF_ASSERT_OK(
      dims.Collapse(DotOperandDims::kBatch, /*remove_if_empty=*/false));
  EXPECT_EQ(dims.shape(), ParseShape("f32[2,3,1,6]").value());
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(2));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(1, 3));
}

TEST(DotOperandDimsTest, CollapseEmptyKeepIfEmpty) {
  Shape shape = ParseShape("f32[2,4,6]").value();
  DotOperandDims dims(shape, /*batch_dims=*/{},
                      /*non_contracting_dims=*/{0},
                      /*contracting_dims=*/{1, 2});
  TF_ASSERT_OK(
      dims.Collapse(DotOperandDims::kBatch, /*remove_if_empty=*/false));
  EXPECT_EQ(dims.shape(), ParseShape("f32[2,4,6]").value());
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre());
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(1, 2));
}

TEST(DotOperandDimsTest, CollapseNormalCase) {
  Shape shape = ParseShape("f32[10,2,3,4,5,6]").value();
  DotOperandDims dims(shape, /*batch_dims=*/{0},
                      /*non_contracting_dims=*/{1, 5},
                      /*contracting_dims=*/{4, 3, 2});
  TF_ASSERT_OK(
      dims.Collapse(DotOperandDims::kContracting, /*remove_if_empty=*/false));
  EXPECT_EQ(dims.shape(), ParseShape("f32[10,2,60,6]").value());
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(0));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre(1, 3));
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(2));
}

TEST(DotOperandDimsTest, EraseDimensions) {
  Shape shape = ParseShape("f32[10,20,30,40,50,60]").value();
  DotOperandDims dims(shape, /*batch_dims=*/{0, 5},
                      /*non_contracting_dims=*/{1, 3},
                      /*contracting_dims=*/{4, 2});
  TF_ASSERT_OK(dims.EraseDimensions(0, 4));
  EXPECT_EQ(dims.shape(), ParseShape("f32[50,60]").value());
  EXPECT_THAT(dims.Indices(DotOperandDims::kBatch), ElementsAre(1));
  EXPECT_THAT(dims.Indices(DotOperandDims::kNonContracting), ElementsAre());
  EXPECT_THAT(dims.Indices(DotOperandDims::kContracting), ElementsAre(0));
}

TEST(DotOperandDimsTest, LocalIndex) {
  Shape shape = ParseShape("f32[10,20,30,40,50,60]").value();
  DotOperandDims dims(shape, /*batch_dims=*/{0, 5},
                      /*non_contracting_dims=*/{1, 3},
                      /*contracting_dims=*/{2, 4});
  EXPECT_EQ(dims.LocalIndex(DotOperandDims::kBatch, 5).value(), 1);
  EXPECT_EQ(dims.LocalIndex(DotOperandDims::kNonContracting, 1).value(), 0);
  EXPECT_FALSE(dims.LocalIndex(DotOperandDims::kContracting, 0).ok());
}

}  // namespace
}  // namespace gpu
}  // namespace xla

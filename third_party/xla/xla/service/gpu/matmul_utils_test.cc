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

#include "xla/service/gpu/matmul_utils.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/shape.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;

using CanFoldTransposeOperandIntoDotTest = HloHardwareIndependentTestBase;

TEST_F(CanFoldTransposeOperandIntoDotTest, ArgTransposeFoldGemm) {
  const char* hlo_text = R"(
HloModule ArgTransposeFoldGemm

ENTRY AddDotsFunc {
  x = f32[3,2] parameter(0)
  y = f32[3,4] parameter(1)
  x_transposed = f32[2,3] transpose(x), dimensions={1, 0}
  ROOT dot_a = f32[2,4] dot(x_transposed, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  auto dot = module->entry_computation()->root_instruction();
  EXPECT_THAT(CanFoldTransposeOperandIntoDot(*dot, 0), IsOkAndHolds(true));
}

TEST_F(CanFoldTransposeOperandIntoDotTest, BatchedArgRowColTransposeFoldGemm) {
  const char* hlo_text = R"(
HloModule BatchedArgRowColTransposeFoldGemm

ENTRY AddDotsFunc {
  x = f32[5,3,2] parameter(0)
  y = f32[5,3,4] parameter(1)
  x_transposed = f32[5,2,3] transpose(x), dimensions={0, 2, 1}
  ROOT dot_a = f32[5,2,4] dot(x_transposed, y), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
}

)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  auto dot = module->entry_computation()->root_instruction();
  EXPECT_THAT(CanFoldTransposeOperandIntoDot(*dot, 0), IsOkAndHolds(true));
}

TEST_F(CanFoldTransposeOperandIntoDotTest, BatchRowTransposeFoldGemm) {
  const char* hlo_text = R"(
HloModule BatchRowTransposeFoldCheck

ENTRY AddDotsFunc {
  x = f32[2,5,3] parameter(0)
  y = f32[5,3,4] parameter(1)
  x_transposed = f32[5,2,3] transpose(x), dimensions={1, 0, 2}
  ROOT dot_a = f32[5,2,4] dot(x_transposed, y), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
}

)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  auto dot = module->entry_computation()->root_instruction();
  EXPECT_THAT(CanFoldTransposeOperandIntoDot(*dot, 0), IsOkAndHolds(true));
}

TEST_F(CanFoldTransposeOperandIntoDotTest,
       BatchFromMinorDimTransposeDoesntFold) {
  const char* hlo_text = R"(
HloModule BatchFromMinorDimTransposeDoesntFold

ENTRY AddDotsFunc {
  x = f32[3,2,5] parameter(0)
  y = f32[5,3,4] parameter(1)
  x_transposed = f32[5,2,3] transpose(x), dimensions={2, 1, 0}
  ROOT dot_a = f32[5,2,4] dot(x_transposed, y), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
}

)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  auto dot = module->entry_computation()->root_instruction();
  EXPECT_THAT(CanFoldTransposeOperandIntoDot(*dot, 0), IsOkAndHolds(false));
}

TEST_F(CanFoldTransposeOperandIntoDotTest,
       TransposedNonContractingDimsDontFold) {
  const char* hlo_text = R"(
HloModule TransposedNonContractingDimsDontFold

ENTRY AddDotsFunc {
  x = f32[5,3,4]{2,1,0} parameter(1)
  y = f32[5,2,6,3]{3,1,2,0} parameter(0)
  y_transposed = f32[5,6,2,3]{3,2,1,0} transpose(y), dimensions={0, 2, 1, 3}
  ROOT dot_a = f32[5,4,6,2]{3,2,1,0} dot(x, y_transposed), lhs_contracting_dims={1}, rhs_contracting_dims={3}, lhs_batch_dims={0}, rhs_batch_dims={0}
}

)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  auto dot = module->entry_computation()->root_instruction();
  EXPECT_THAT(CanFoldTransposeOperandIntoDot(*dot, 1), IsOkAndHolds(false));
}

struct GetBatchRowColumnShapeTestParams {
  absl::string_view shape;
  std::vector<int64_t> batch_dims;
  std::vector<int64_t> row_dims;
  std::vector<int64_t> col_dims;
  absl::string_view expected_shape;
};

using GetBatchRowColumnShapeTest =
    ::testing::TestWithParam<GetBatchRowColumnShapeTestParams>;

TEST_P(GetBatchRowColumnShapeTest, ValidShape) {
  const GetBatchRowColumnShapeTestParams& params = GetParam();

  Shape shape = ParseShape(params.shape).value();
  EXPECT_THAT(GetBatchRowColumnShape(shape, params.batch_dims, params.row_dims,
                                     params.col_dims),
              IsOkAndHolds(ParseShape(params.expected_shape).value()));
}

INSTANTIATE_TEST_SUITE_P(
    GetBatchRowColumnShapeTests, GetBatchRowColumnShapeTest,
    ::testing::ValuesIn<GetBatchRowColumnShapeTestParams>({
        {"f32[3,4]{1,0}", /*batch_dims=*/{}, /*row_dims=*/{0}, /*col_dims=*/{1},
         "f32[1,3,4]{2,1,0}"},
        {"f32[3,4]{0,1}", {}, {0}, {1}, "f32[1,3,4]{1,2,0}"},
        {"f32[3,4]{1,0}", {}, {1}, {0}, "f32[1,4,3]{1,2,0}"},
        {"f32[3,4,5]{2,1,0}", {0}, {1}, {2}, "f32[3,4,5]{2,1,0}"},
        {"f32[3,4,5]{2,1,0}", {2}, {1}, {0}, "f32[5,4,3]{0,1,2}"},
        {"f32[3,4,5,6,7,8]{5,2,4,1,3,0}",
         {0, 3},
         {1, 4},
         {2, 5},
         "f32[18,28,40]{2,1,0}"},
    }));

TEST(GetBatchRowColumnShapeTest, BatchRowsColsInterleaved) {
  Shape shape = ParseShape("f32[3,4,5,6,7,8]{5,4,3,2,1,0}").value();
  auto result =
      GetBatchRowColumnShape(shape, /*batch_dims=*/{0, 3},
                             /*row_dims=*/{1, 4}, /*col_dims=*/{2, 5});
  EXPECT_FALSE(result.ok());
}

TEST(GetBatchRowColumnShapeTest, WrongPhysicalOrder) {
  Shape shape = ParseShape("f32[3,4,5,6]{3,2,0,1}").value();
  auto result = GetBatchRowColumnShape(shape, /*batch_dims=*/{0, 1},
                                       /*row_dims=*/{2}, /*col_dims=*/{3});
  EXPECT_FALSE(result.ok());
}

using Order = MatrixLayout::Order;

struct GetMatrixLayoutTestParams {
  absl::string_view shape;
  int64_t batch_size;
  int64_t num_rows;
  int64_t num_cols;
  Order order;
  int64_t leading_dim_stride;
  int64_t batch_stride;
};

using GetMatrixLayoutTest = ::testing::TestWithParam<GetMatrixLayoutTestParams>;

TEST_P(GetMatrixLayoutTest, ValidShape) {
  const GetMatrixLayoutTestParams& params = GetParam();

  Shape shape = ParseShape(params.shape).value();
  MatrixLayout result = MatrixLayout::For(shape).value();
  EXPECT_EQ(result.batch_size, params.batch_size);
  EXPECT_EQ(result.num_rows, params.num_rows);
  EXPECT_EQ(result.num_cols, params.num_cols);
  EXPECT_EQ(result.order, params.order);
  EXPECT_EQ(result.leading_dim_stride, params.leading_dim_stride);
  EXPECT_EQ(result.batch_stride, params.batch_stride);
}

INSTANTIATE_TEST_SUITE_P(
    GetMatrixLayoutTests, GetMatrixLayoutTest,
    ::testing::ValuesIn<GetMatrixLayoutTestParams>({
        {"f32[3,4,5]{2,1,0}", /*batch_size=*/3, /*num_rows=*/4, /*num_cols=*/5,
         /*order=*/Order::kRowMajor, /*leading_dim_stride=*/5,
         /*batch_stride=*/20},
        {"f32[3,4,5]{1,2,0}", 3, 4, 5, Order::kColumnMajor, 4, 20},
        {"f32[3,4,5]{2,0,1}", 3, 4, 5, Order::kRowMajor, 15, 5},
        {"f32[3,4,5]{1,0,2}", 3, 4, 5, Order::kColumnMajor, 12, 4},
    }));

TEST(GetMatrixLayoutTest, BatchInMostMinorPhysicalDimension) {
  Shape shape = ParseShape("f32[3,4,5]{0,2,1}").value();
  EXPECT_FALSE(MatrixLayout::For(shape).ok());
}

using GetMatrixSizeRewriteThresholdTest = HloHardwareIndependentTestBase;

TEST_F(GetMatrixSizeRewriteThresholdTest, MatMulTooSmallForRewrite) {
  const char* hlo_text = R"(
HloModule DotFuncModule

ENTRY DotFunc {
  x = f32[100,30,3] parameter(0)
  y = f32[100,3,3] parameter(1)
  ROOT dot = f32[100,30,3] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
}

)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  auto dot = module->entry_computation()->root_instruction();
  EXPECT_THAT(IsMatrixMultiplicationTooSmallForRewriting(*dot, 100),
              IsOkAndHolds(true));
}

TEST_F(GetMatrixSizeRewriteThresholdTest, MatMulSupportedByClassicalEmitters) {
  const char* hlo_text = R"(
HloModule DotFuncModule

ENTRY DotFunc {
  x = f32[100,30,3] parameter(0)
  y = f32[100,3,3] parameter(1)
  ROOT dot = f32[100,30,3] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
}

)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  auto dot = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsDotSupportedByClassicalEmitters(*dot));
}

TEST_F(GetMatrixSizeRewriteThresholdTest,
       MatMulUnsupportedByClassicalEmitters) {
  const char* hlo_text = R"(
HloModule DotFuncModule

ENTRY DotFunc {
  x = s8[100,30,3] parameter(0)
  y = s8[100,3,3] parameter(1)
  ROOT dot = s32[100,30,3] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
}

)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  auto dot = module->entry_computation()->root_instruction();
  EXPECT_FALSE(IsDotSupportedByClassicalEmitters(*dot));
}

TEST_F(GetMatrixSizeRewriteThresholdTest, MatMulLeftLargeEnoughForRewrite) {
  const char* hlo_text = R"(
HloModule DotFuncModule

ENTRY DotFunc {
  x = f32[50,2] parameter(0)
  y = f32[2,2] parameter(1)
  ROOT dot = f32[50,2] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  auto dot = module->entry_computation()->root_instruction();
  EXPECT_THAT(IsMatrixMultiplicationTooSmallForRewriting(*dot, 100),
              IsOkAndHolds(false));
}

TEST_F(GetMatrixSizeRewriteThresholdTest, MatMulRightLargeEnoughForRewrite) {
  const char* hlo_text = R"(
HloModule DotFuncModule

ENTRY DotFunc {
  x = f32[2,2] parameter(0)
  y = f32[2,50] parameter(1)
  ROOT dot = f32[2,50] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  auto dot = module->entry_computation()->root_instruction();
  EXPECT_THAT(IsMatrixMultiplicationTooSmallForRewriting(*dot, 100),
              IsOkAndHolds(false));
}

TEST_F(GetMatrixSizeRewriteThresholdTest, MatMulTogetherLargeEnoughForRewrite) {
  const char* hlo_text = R"(
HloModule DotFuncModule

ENTRY DotFunc {
  x = f32[4,16] parameter(0)
  y = f32[16,4] parameter(1)
  ROOT dot = f32[4,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  auto dot = module->entry_computation()->root_instruction();
  EXPECT_THAT(IsMatrixMultiplicationTooSmallForRewriting(*dot, 100),
              IsOkAndHolds(false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

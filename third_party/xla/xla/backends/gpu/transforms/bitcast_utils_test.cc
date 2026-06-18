/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/bitcast_utils.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

TEST(BitcastUtilsTest, CopyElementType) {
  Shape source_f32 = ShapeUtil::MakeShape(F32, {2, 3});
  Shape destination_s32 = ShapeUtil::MakeShape(S32, {4, 5});
  CopyElementType(source_f32, &destination_s32);
  EXPECT_EQ(destination_s32.element_type(), F32);
}

TEST(BitcastUtilsTest, CopyElementTypeCopiesCustomElementSize) {
  Shape source_custom_size = ShapeUtil::MakeShape(S8, {2, 3});
  source_custom_size.mutable_layout()->set_element_size_in_bits(4);
  Shape destination_custom_size = ShapeUtil::MakeShape(F32, {4, 5});
  CopyElementType(source_custom_size, &destination_custom_size);
  EXPECT_EQ(destination_custom_size.element_type(), S8);
  EXPECT_EQ(destination_custom_size.layout().element_size_in_bits(), 4);
}

TEST(BitcastUtilsTest, CalculateBitcastOfBroadcast) {
  // Example:
  // operand = f32[2, 3]
  // broadcast = f32[2, 3, 4] broadcast(operand), dimensions={0, 1}
  // result = f32[6, 4] bitcast(broadcast)
  //
  // Expected to rewrite as:
  // bitcast = f32[6] (params.new_shape) bitcast(operand)
  // result = f32[6, 4] broadcast(bitcast), dimensions={0} (params.new_dims)
  Shape operand_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  Shape result_shape = ShapeUtil::MakeShape(F32, {6, 4});
  auto param = HloInstruction::CreateParameter(0, operand_shape, "p");
  auto broadcast_inst =
      HloInstruction::CreateBroadcast(broadcast_shape, param.get(), {0, 1});

  ASSERT_OK_AND_ASSIGN(
      BitcastParams params,
      CalculateBitcastOfBroadcast(
          Cast<HloBroadcastInstruction>(broadcast_inst.get()), result_shape));

  EXPECT_TRUE(
      ShapeUtil::Compatible(params.new_shape, ShapeUtil::MakeShape(F32, {6})));
  EXPECT_THAT(params.new_dims, testing::ElementsAre(0));
}

TEST(BitcastUtilsTest, CalculateBitcastOfBroadcastOfScalar) {
  // broadcast = f32[2, 3, 4] broadcast(f32[]), dimensions={0, 1}
  // result = f32[6, 4] bitcast(broadcast)
  // ->
  // bitcast = f32[] bitcast(f32[])
  // result = f32[6, 4] broadcast(bitcast), dimensions={0}
  Shape operand_shape = ShapeUtil::MakeShape(F32, {});
  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  Shape result_shape = ShapeUtil::MakeShape(F32, {6, 4});
  auto param = HloInstruction::CreateParameter(0, operand_shape, "p");
  auto broadcast_inst =
      HloInstruction::CreateBroadcast(broadcast_shape, param.get(), {});

  ASSERT_OK_AND_ASSIGN(
      BitcastParams params,
      CalculateBitcastOfBroadcast(
          Cast<HloBroadcastInstruction>(broadcast_inst.get()), result_shape));

  EXPECT_TRUE(
      ShapeUtil::Compatible(params.new_shape, ShapeUtil::MakeShape(F32, {})));
  EXPECT_THAT(params.new_dims, testing::ElementsAre());
}

TEST(BitcastUtilsTest,
     CalculateBitcastOfBroadcastAcrossOperandDimensionsFails) {
  // We cannot calculate bitcast of broadcast when the bitcast mixes operand
  // and broadcast dimensions.
  // Ex: broadcast = f32[3, 4] broadcast(f32[3] operand), dimensions={0}
  //     result = f32[12] bitcast(broadcast)
  Shape operand_shape = ShapeUtil::MakeShape(F32, {3});
  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {3, 4});
  Shape result_shape = ShapeUtil::MakeShape(F32, {12});
  auto param = HloInstruction::CreateParameter(0, operand_shape, "p");
  auto broadcast_inst =
      HloInstruction::CreateBroadcast(broadcast_shape, param.get(), {0});

  EXPECT_THAT(
      CalculateBitcastOfBroadcast(
          Cast<HloBroadcastInstruction>(broadcast_inst.get()), result_shape),
      ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BitcastUtilsTest, CalculateBroadcastOfBitcast) {
  // Example:
  // bitcast = f32[6] bitcast(operand: f32[2, 3])
  // result = f32[6, 4] broadcast(bitcast), dimensions={0}
  //
  // Expected to rewrite as:
  // broadcast = f32[2, 3, 4] broadcast(operand), dimensions={0, 1}
  // result = f32[6, 4] bitcast(broadcast)

  Shape operand_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape bitcast_shape = ShapeUtil::MakeShape(F32, {6});
  Shape result_shape = ShapeUtil::MakeShape(F32, {6, 4});

  auto param = HloInstruction::CreateParameter(0, operand_shape, "p");
  auto bitcast_inst = HloInstruction::CreateBitcast(bitcast_shape, param.get());
  auto broadcast_inst =
      HloInstruction::CreateBroadcast(result_shape, bitcast_inst.get(), {0});

  ASSERT_OK_AND_ASSIGN(
      BitcastParams params,
      CalculateBroadcastOfBitcast(
          Cast<HloBroadcastInstruction>(broadcast_inst.get()), operand_shape));

  EXPECT_TRUE(ShapeUtil::Compatible(params.new_shape,
                                    ShapeUtil::MakeShape(F32, {2, 3, 4})));
  EXPECT_THAT(params.new_dims, testing::ElementsAre(0, 1));
}

TEST(BitcastUtilsTest,
     CalculateBroadcastOfBitcastNonContiguousDimensionsFails) {
  // Cannot calculate broadcast of bitcast when the broadcast inserts a new
  // dimension between the bitcast dimensions.
  // Ex: bitcast = f32[2, 3] bitcast(f32[6] operand)
  //     broadcast = f32[2, 4, 3] broadcast(bitcast), dimensions={0, 2}
  Shape operand_shape = ShapeUtil::MakeShape(F32, {6});
  Shape bitcast_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape result_shape = ShapeUtil::MakeShape(F32, {2, 4, 3});

  auto param = HloInstruction::CreateParameter(0, operand_shape, "p");
  auto bitcast_inst = HloInstruction::CreateBitcast(bitcast_shape, param.get());
  auto broadcast_inst =
      HloInstruction::CreateBroadcast(result_shape, bitcast_inst.get(), {0, 2});

  EXPECT_THAT(
      CalculateBroadcastOfBitcast(
          Cast<HloBroadcastInstruction>(broadcast_inst.get()), operand_shape),
      ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BitcastUtilsTest, CalculateBitcastOfTranspose) {
  // Example:
  // operand = f32[7, 6]
  // transpose = f32[6, 7] transpose(operand), dimensions={1, 0}
  // result = f32[2, 3, 7] bitcast(transpose)
  //
  // Expected to rewrite as:
  // bitcast = f32[7, 2, 3] bitcast(operand)
  // result = f32[2, 3, 7] transpose(bitcast), dimensions={1, 2, 0}
  Shape operand_shape = ShapeUtil::MakeShape(F32, {7, 6});
  Shape transpose_shape = ShapeUtil::MakeShape(F32, {6, 7});
  Shape result_shape = ShapeUtil::MakeShape(F32, {2, 3, 7});
  auto param = HloInstruction::CreateParameter(0, operand_shape, "p");
  auto transpose_inst =
      HloInstruction::CreateTranspose(transpose_shape, param.get(), {1, 0});

  ASSERT_OK_AND_ASSIGN(
      BitcastParams params,
      CalculateBitcastOfTranspose(
          Cast<HloTransposeInstruction>(transpose_inst.get()), result_shape));

  EXPECT_TRUE(ShapeUtil::Compatible(params.new_shape,
                                    ShapeUtil::MakeShape(F32, {7, 2, 3})));
  EXPECT_EQ(params.new_shape.layout(), result_shape.layout());
  EXPECT_THAT(params.new_dims, testing::ElementsAre(1, 2, 0));
}

TEST(BitcastUtilsTest, CalculateBitcastOfTransposeWithDifferentLayoutsFails) {
  // We cannot calculate bitcast of transpose when layouts of transpose operand
  // and result are different.
  // Ex: operand = f32[7, 6]{1,0}
  //     transpose = f32[6, 7]{0,1} transpose(operand), dimensions={1, 0}
  //     result = f32[2, 3, 7] bitcast(transpose)
  Shape operand_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {7, 6}, {1, 0});
  Shape transpose_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {6, 7}, {0, 1});
  Shape result_shape = ShapeUtil::MakeShape(F32, {2, 3, 7});
  auto param = HloInstruction::CreateParameter(0, operand_shape, "p");
  auto transpose_inst =
      HloInstruction::CreateTranspose(transpose_shape, param.get(), {1, 0});

  EXPECT_THAT(
      CalculateBitcastOfTranspose(
          Cast<HloTransposeInstruction>(transpose_inst.get()), result_shape),
      ::absl_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST(BitcastUtilsTest,
     CalculateBitcastOfTransposeNoncontiguousDimensionsFails) {
  // We cannnot calculate bitcast of transpose when the bitcast dimension mixes
  // dimensions that are physically non-contiguous.
  // Ex: operand = f32[2, 3]
  //     transpose = f32[3, 2] transpose(operand), dimensions={1, 0}
  //     result = f32[6] bitcast(transpose)
  Shape operand_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape transpose_shape = ShapeUtil::MakeShape(F32, {3, 2});
  Shape result_shape = ShapeUtil::MakeShape(F32, {6});
  auto param = HloInstruction::CreateParameter(0, operand_shape, "p");
  auto transpose_inst =
      HloInstruction::CreateTranspose(transpose_shape, param.get(), {1, 0});

  EXPECT_THAT(
      CalculateBitcastOfTranspose(
          Cast<HloTransposeInstruction>(transpose_inst.get()), result_shape),
      ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BitcastUtilsTest, CalculateBitcastOfTransposeHandlesTrivialDimensions) {
  // Size-1 dimensions in the transpose are handled correctly by dropping them
  // from the new bitcast shape and new transpose dimensions.
  // Ex: operand = f32[7, 1, 6]
  //     transpose = f32[1, 6, 7] transpose(operand), dimensions={1, 2, 0}
  //     result = f32[1, 2, 3, 7] bitcast(transpose)
  // ->
  // bitcast = f32[7, 2, 3] bitcast(operand)
  // result = f32[2, 3, 7] transpose(bitcast), dimensions={1, 2, 0}
  Shape operand_shape = ShapeUtil::MakeShape(F32, {7, 1, 6});
  Shape transpose_shape = ShapeUtil::MakeShape(F32, {1, 6, 7});
  Shape result_shape = ShapeUtil::MakeShape(F32, {1, 2, 3, 7});

  auto param = HloInstruction::CreateParameter(0, operand_shape, "p");
  auto transpose_inst =
      HloInstruction::CreateTranspose(transpose_shape, param.get(), {1, 2, 0});

  ASSERT_OK_AND_ASSIGN(
      BitcastParams params,
      CalculateBitcastOfTranspose(
          Cast<HloTransposeInstruction>(transpose_inst.get()), result_shape));

  Shape expected_shape = ShapeUtil::MakeShape(F32, {0, 7, 2, 3});
  EXPECT_EQ(params.new_shape, expected_shape);
  EXPECT_THAT(params.new_dims, testing::ElementsAre(0, 2, 3, 1));
}

TEST(BitcastUtilsTest, CalculateTransposeOfBitcast) {
  // Example:
  // operand = f32[7, 6]
  // bitcast = f32[7, 2, 3]
  // transpose = f32[2, 3, 7] transpose(bitcast), dimensions={1, 2, 0}
  //
  // Expected to rewrite as:
  // transpose = f32[6, 7] transpose(operand: f32[7, 6]), dimensions={1, 0}
  // result = f32[2, 3, 7] bitcast(transpose)
  Shape operand_shape = ShapeUtil::MakeShape(F32, {7, 6});
  Shape bitcast_shape = ShapeUtil::MakeShape(F32, {7, 2, 3});
  Shape transpose_shape = ShapeUtil::MakeShape(F32, {2, 3, 7});
  auto param = HloInstruction::CreateParameter(0, bitcast_shape, "p");
  auto transpose_inst =
      HloInstruction::CreateTranspose(transpose_shape, param.get(), {1, 2, 0});

  ASSERT_OK_AND_ASSIGN(
      BitcastParams params,
      CalculateTransposeOfBitcast(
          Cast<HloTransposeInstruction>(transpose_inst.get()), operand_shape));

  EXPECT_TRUE(ShapeUtil::Compatible(params.new_shape,
                                    ShapeUtil::MakeShape(F32, {6, 7})));
  EXPECT_EQ(params.new_shape.layout(), operand_shape.layout());
  EXPECT_THAT(params.new_dims, testing::ElementsAre(1, 0));
}

struct CommonFactorsTestCase {
  std::vector<int64_t> from, to;
  absl::InlinedVector<std::pair<int64_t, int64_t>, 8> expected;
};

class CommonFactorsMergingTrivialRangesTest
    : public ::testing::TestWithParam<CommonFactorsTestCase> {};

TEST_P(CommonFactorsMergingTrivialRangesTest, Example) {
  const CommonFactorsTestCase& test_case = GetParam();
  EXPECT_EQ(test_case.expected, detail::CommonFactorsMergingTrivialRanges(
                                    test_case.from, test_case.to));
}

INSTANTIATE_TEST_SUITE_P(
    CommonFactorsMergingTrivialRangesTestSuite,
    CommonFactorsMergingTrivialRangesTest,
    ::testing::Values(
        CommonFactorsTestCase{{1}, {}, {{0, 0}, {1, 0}}},
        CommonFactorsTestCase{{}, {1}, {{0, 0}, {0, 1}}},
        CommonFactorsTestCase{{}, {}, {{0, 0}}},
        CommonFactorsTestCase{{1, 2, 0}, {2, 0, 3}, {{0, 0}, {3, 3}}},
        CommonFactorsTestCase{{2, 3, 0}, {1, 0, 1000}, {{0, 0}, {3, 3}}},
        CommonFactorsTestCase{{1, 1, 1}, {1, 1}, {{0, 0}, {1, 1}, {3, 2}}},
        CommonFactorsTestCase{{1, 1, 3}, {3, 1, 1}, {{0, 0}, {3, 3}}},
        CommonFactorsTestCase{{2, 6}, {4, 3}, {{0, 0}, {2, 2}}},
        CommonFactorsTestCase{{1, 2, 6}, {4, 1, 3, 1}, {{0, 0}, {3, 4}}},
        CommonFactorsTestCase{{2, 3, 4, 5}, {6, 20}, {{0, 0}, {2, 1}, {4, 2}}},
        CommonFactorsTestCase{
            {2, 3, 4, 5, 6}, {6, 20, 6}, {{0, 0}, {2, 1}, {4, 2}, {5, 3}}},
        CommonFactorsTestCase{{2, 2, 2, 2}, {4, 4}, {{0, 0}, {2, 1}, {4, 2}}},
        CommonFactorsTestCase{
            {2, 5, 1, 3}, {1, 10, 3, 1}, {{0, 0}, {2, 2}, {4, 4}}}),
    [](const ::testing::TestParamInfo<CommonFactorsTestCase>& info) {
      return absl::StrCat(absl::StrJoin(info.param.from, "_"), "_to_",
                          absl::StrJoin(info.param.to, "_"));
    });

}  // namespace
}  // namespace xla::gpu

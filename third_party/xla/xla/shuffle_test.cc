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

#include "xla/shuffle.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace shuffle {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(ShuffleTest, PermuteSetsTheIndices) {
  const Literal indices = LiteralUtil::CreateR2<int32_t>({{1, 0}, {0, 1}});

  const ShuffleMode mode = Permute(indices);

  EXPECT_EQ(mode.mode_case(), ShuffleMode::kPermute);
  ASSERT_OK_AND_ASSIGN(const Literal mode_indices, GetPermuteIndices(mode));
  EXPECT_EQ(mode_indices, indices);
}

TEST(ShuffleTest, GetPermuteIndicesShapeReturnsTheShapeOfTheIndices) {
  const ShuffleMode mode =
      Permute(LiteralUtil::CreateR2<int8_t>({{1, 0}, {0, 1}}));

  EXPECT_THAT(GetPermuteIndicesShape(mode),
              absl_testing::IsOkAndHolds(ShapeUtil::MakeShape(S8, {2, 2})));
}

TEST(ShuffleTest, RotateSetsTheShifts) {
  const ShuffleMode mode = Rotate({2, 3});

  EXPECT_EQ(mode.mode_case(), ShuffleMode::kRotate);
  EXPECT_THAT(mode.rotate().shifts(), ElementsAre(2, 3));
}

TEST(ShuffleTest, RotateWithoutShiftsIsStillInRotateMode) {
  const ShuffleMode mode = Rotate({});

  EXPECT_EQ(mode.mode_case(), ShuffleMode::kRotate);
  EXPECT_THAT(mode.rotate().shifts(), IsEmpty());
}

TEST(ShuffleTest, MultiRotateSetsTheMultiShifts) {
  const Literal multi_shifts = LiteralUtil::CreateR2<int32_t>({{1, 2}});

  const ShuffleMode mode = MultiRotate(multi_shifts);

  EXPECT_EQ(mode.mode_case(), ShuffleMode::kMultiRotate);
  ASSERT_OK_AND_ASSIGN(const Literal mode_multi_shifts, GetMultiShifts(mode));
  EXPECT_EQ(mode_multi_shifts, multi_shifts);
}

TEST(ShuffleTest, GetMultiShiftsShapeReturnsTheShapeOfTheMultiShifts) {
  const ShuffleMode mode = MultiRotate(LiteralUtil::CreateR2<int8_t>({{1, 2}}));

  EXPECT_THAT(GetMultiShiftsShape(mode),
              absl_testing::IsOkAndHolds(ShapeUtil::MakeShape(S8, {1, 2})));
}

TEST(ShuffleTest, GetMultiRotatePermuteIndicesRotatesEverySliceByItsShift) {
  const ShuffleMode mode =
      MultiRotate(LiteralUtil::CreateR2<int32_t>({{1, 2}}));

  ASSERT_OK_AND_ASSIGN(const Literal indices,
                       GetMultiRotatePermuteIndices(mode, /*dimension=*/0,
                                                    /*dimension_size=*/3));
  EXPECT_EQ(indices, LiteralUtil::CreateR2<int32_t>({{1, 2}, {2, 0}, {0, 1}}));
}

TEST(ShuffleTest, GetMultiRotatePermuteIndicesWrapsShiftsAround) {
  const ShuffleMode mode =
      MultiRotate(LiteralUtil::CreateR2<int32_t>({{-1, 5}}));

  ASSERT_OK_AND_ASSIGN(const Literal indices,
                       GetMultiRotatePermuteIndices(mode, /*dimension=*/0,
                                                    /*dimension_size=*/3));
  EXPECT_EQ(indices, LiteralUtil::CreateR2<int32_t>({{2, 2}, {0, 0}, {1, 1}}));
}

// A shifts array that is of size 1 along a dimension that is not rotated holds
// one shift that every slice along that dimension is rotated by, so the
// indices stay of size 1 there and the shuffle broadcasts them.
TEST(ShuffleTest, GetMultiRotatePermuteIndicesKeepsASharedShiftShared) {
  const ShuffleMode mode = MultiRotate(LiteralUtil::CreateR2<int32_t>({{1}}));

  ASSERT_OK_AND_ASSIGN(const Literal indices,
                       GetMultiRotatePermuteIndices(mode, /*dimension=*/0,
                                                    /*dimension_size=*/3));
  EXPECT_EQ(indices, LiteralUtil::CreateR2<int32_t>({{1}, {2}, {0}}));
}

TEST(ShuffleTest, GetMultiRotatePermuteIndicesWithBroadcastDimension) {
  // shifts has shape [1, 2, 1], rotating dimension 2 of size 4.
  // Dimension 0 has size 1 (broadcast across dimension 0).
  const Literal shifts = LiteralUtil::CreateR3<int32_t>({{{1}, {2}}});
  const ShuffleMode mode = MultiRotate(shifts);

  ASSERT_OK_AND_ASSIGN(const Literal indices,
                       GetMultiRotatePermuteIndices(mode, /*dimension=*/2,
                                                    /*dimension_size=*/4));
  EXPECT_EQ(indices,
            LiteralUtil::CreateR3<int32_t>({{{1, 2, 3, 0}, {2, 3, 0, 1}}}));
}

TEST(ShuffleTest, GetMultiRotatePermuteIndicesOfADimensionOfSizeOneAreZero) {
  const ShuffleMode mode =
      MultiRotate(LiteralUtil::CreateR2<int32_t>({{1, 2}}));

  ASSERT_OK_AND_ASSIGN(const Literal indices,
                       GetMultiRotatePermuteIndices(mode, /*dimension=*/0,
                                                    /*dimension_size=*/1));
  EXPECT_EQ(indices, LiteralUtil::CreateR2<int32_t>({{0, 0}}));
}

TEST(ShuffleTest, GetMultiRotatePermuteIndicesOfAnEmptyDimensionAreEmpty) {
  const ShuffleMode mode =
      MultiRotate(LiteralUtil::CreateR2<int32_t>({{1, 2}}));

  ASSERT_OK_AND_ASSIGN(const Literal indices,
                       GetMultiRotatePermuteIndices(mode, /*dimension=*/0,
                                                    /*dimension_size=*/0));
  EXPECT_EQ(indices.shape(), ShapeUtil::MakeShape(S32, {0, 2}));
}

// The shifts fit in the type they are given, but an index runs up to the size
// of the rotated dimension, which that type need not hold.
TEST(ShuffleTest, GetMultiRotatePermuteIndicesTakeATypeThatHoldsEveryIndex) {
  const ShuffleMode mode = MultiRotate(LiteralUtil::CreateR2<int8_t>({{1, 2}}));

  ASSERT_OK_AND_ASSIGN(const Literal indices,
                       GetMultiRotatePermuteIndices(mode, /*dimension=*/0,
                                                    /*dimension_size=*/300));
  EXPECT_EQ(indices.shape().element_type(), S32);
  EXPECT_EQ(indices.GetIntegralAsS64({299, 1}), 1);
}

TEST(ShuffleTest, NormalizeShiftKeepsShiftsThatAreAlreadyInRange) {
  EXPECT_EQ(NormalizeShift(0, 4), 0);
  EXPECT_EQ(NormalizeShift(1, 4), 1);
  EXPECT_EQ(NormalizeShift(3, 4), 3);
}

TEST(ShuffleTest, NormalizeShiftWrapsShiftsAround) {
  EXPECT_EQ(NormalizeShift(4, 4), 0);
  EXPECT_EQ(NormalizeShift(5, 4), 1);
  EXPECT_EQ(NormalizeShift(9, 4), 1);
}

TEST(ShuffleTest, NormalizeShiftWrapsNegativeShiftsAround) {
  EXPECT_EQ(NormalizeShift(-1, 4), 3);
  EXPECT_EQ(NormalizeShift(-4, 4), 0);
  EXPECT_EQ(NormalizeShift(-5, 4), 3);
}

TEST(ShuffleTest, NormalizeShiftOfADimensionOfSizeOneIsANoOp) {
  EXPECT_EQ(NormalizeShift(3, 1), 0);
  EXPECT_EQ(NormalizeShift(-3, 1), 0);
}

TEST(ShuffleTest, NormalizeShiftOfAnEmptyDimensionIsANoOp) {
  EXPECT_EQ(NormalizeShift(0, 0), 0);
  EXPECT_EQ(NormalizeShift(3, 0), 0);
  EXPECT_EQ(NormalizeShift(-3, 0), 0);
}

}  // namespace
}  // namespace shuffle
}  // namespace xla

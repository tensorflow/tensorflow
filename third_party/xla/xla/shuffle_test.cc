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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/xla_data.pb.h"

namespace xla {
namespace shuffle {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(ShuffleTest, MakeRotateModeSetsTheShifts) {
  const ShuffleMode mode = MakeRotateMode({2, 3});

  EXPECT_EQ(mode.mode_case(), ShuffleMode::kRotate);
  EXPECT_THAT(mode.rotate().shifts(), ElementsAre(2, 3));
}

TEST(ShuffleTest, MakeRotateModeWithoutShiftsIsStillInRotateMode) {
  const ShuffleMode mode = MakeRotateMode({});

  EXPECT_EQ(mode.mode_case(), ShuffleMode::kRotate);
  EXPECT_THAT(mode.rotate().shifts(), IsEmpty());
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

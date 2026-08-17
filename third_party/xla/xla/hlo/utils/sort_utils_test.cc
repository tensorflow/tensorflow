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

#include "xla/hlo/utils/sort_utils.h"

#include <cstdint>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace {

using SortUtilsTest = HloHardwareIndependentTestBase;

TEST_F(SortUtilsTest, MatchSimpleSortComparatorAscending) {
  constexpr char kHlo[] = R"(
HloModule test_module

compare {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT cmp = pred[] compare(p0, p1), direction=LT
}

ENTRY main {
  x = f32[10] parameter(0)
  ROOT sort = f32[10] sort(x), dimensions={0}, to_apply=compare
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const auto* compare = Cast<HloCompareInstruction>(
      module->GetComputationWithName("compare")->root_instruction());

  EXPECT_EQ(MatchSimpleSortComparator(compare),
            std::make_pair(int64_t{0}, int64_t{1}));
  EXPECT_EQ(MatchNumpySortComparator(compare),
            std::make_pair(int64_t{-1}, int64_t{-1}));
}

TEST_F(SortUtilsTest, MatchSimpleSortComparatorReversed) {
  constexpr char kHlo[] = R"(
HloModule test_module

compare {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT cmp = pred[] compare(p1, p0), direction=LT
}

ENTRY main {
  x = f32[10] parameter(0)
  ROOT sort = f32[10] sort(x), dimensions={0}, to_apply=compare
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const auto* compare = Cast<HloCompareInstruction>(
      module->GetComputationWithName("compare")->root_instruction());

  EXPECT_EQ(MatchSimpleSortComparator(compare),
            std::make_pair(int64_t{1}, int64_t{0}));
  EXPECT_EQ(MatchNumpySortComparator(compare),
            std::make_pair(int64_t{-1}, int64_t{-1}));
}

TEST_F(SortUtilsTest, MatchSimpleSortComparatorArgsortFourParams) {
  constexpr char kHlo[] = R"(
HloModule test_module

compare {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  p2 = s32[] parameter(2)
  p3 = s32[] parameter(3)
  ROOT cmp = pred[] compare(p0, p1), direction=LT
}

ENTRY main {
  x = f32[10] parameter(0)
  y = s32[10] parameter(1)
  ROOT sort = (f32[10], s32[10]) sort(x, y), dimensions={0}, to_apply=compare
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const auto* compare = Cast<HloCompareInstruction>(
      module->GetComputationWithName("compare")->root_instruction());

  EXPECT_EQ(MatchSimpleSortComparator(compare),
            std::make_pair(int64_t{0}, int64_t{1}));
  EXPECT_EQ(MatchNumpySortComparator(compare),
            std::make_pair(int64_t{-1}, int64_t{-1}));
}

TEST_F(SortUtilsTest, MatchNumpySortComparatorPreExpansion) {
  constexpr char kHlo[] = R"(
HloModule test_module

compare {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  lhs_is_nan = pred[] compare(lhs, lhs), direction=NE
  c_nan = f32[] constant(nan)
  c_zero = f32[] constant(0)
  lhs_is_zero = pred[] compare(lhs, c_zero), direction=EQ
  lhs_no_neg_zero = f32[] select(lhs_is_zero, c_zero, lhs)
  lhs_canonical = f32[] select(lhs_is_nan, c_nan, lhs_no_neg_zero)
  rhs_is_nan = pred[] compare(rhs, rhs), direction=NE
  rhs_is_zero = pred[] compare(rhs, c_zero), direction=EQ
  rhs_no_neg_zero = f32[] select(rhs_is_zero, c_zero, rhs)
  rhs_canonical = f32[] select(rhs_is_nan, c_nan, rhs_no_neg_zero)
  ROOT cmp = pred[] compare(lhs_canonical, rhs_canonical), direction=LT, type=TOTALORDER
}

ENTRY main {
  x = f32[10] parameter(0)
  ROOT sort = f32[10] sort(x), dimensions={0}, to_apply=compare
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const auto* compare = Cast<HloCompareInstruction>(
      module->GetComputationWithName("compare")->root_instruction());

  EXPECT_EQ(MatchSimpleSortComparator(compare),
            std::make_pair(int64_t{-1}, int64_t{-1}));
  EXPECT_EQ(MatchNumpySortComparator(compare),
            std::make_pair(int64_t{0}, int64_t{1}));
}

TEST_F(SortUtilsTest, MatchNumpySortComparatorPostExpansion) {
  constexpr char kHlo[] = R"(
HloModule test_module

compare {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  lhs_is_nan = pred[] compare(lhs, lhs), direction=NE
  c_nan = f32[] constant(nan)
  c_zero = f32[] constant(0)
  lhs_is_zero = pred[] compare(lhs, c_zero), direction=EQ
  lhs_no_neg_zero = f32[] select(lhs_is_zero, c_zero, lhs)
  lhs_canonical = f32[] select(lhs_is_nan, c_nan, lhs_no_neg_zero)
  lhs_bitcast = s32[] bitcast-convert(lhs_canonical)
  c_s32_zero = s32[] constant(0)
  lhs_is_neg = pred[] compare(lhs_bitcast, c_s32_zero), direction=LT
  c_mask = s32[] constant(2147483647)
  lhs_mapped = s32[] xor(lhs_bitcast, c_mask)
  lhs_expanded = s32[] select(lhs_is_neg, lhs_mapped, lhs_bitcast)

  rhs_is_nan = pred[] compare(rhs, rhs), direction=NE
  rhs_is_zero = pred[] compare(rhs, c_zero), direction=EQ
  rhs_no_neg_zero = f32[] select(rhs_is_zero, c_zero, rhs)
  rhs_canonical = f32[] select(rhs_is_nan, c_nan, rhs_no_neg_zero)
  rhs_bitcast = s32[] bitcast-convert(rhs_canonical)
  rhs_is_neg = pred[] compare(rhs_bitcast, c_s32_zero), direction=LT
  rhs_mapped = s32[] xor(rhs_bitcast, c_mask)
  rhs_expanded = s32[] select(rhs_is_neg, rhs_mapped, rhs_bitcast)

  ROOT cmp = pred[] compare(lhs_expanded, rhs_expanded), direction=LT, type=SIGNED
}

ENTRY main {
  x = f32[10] parameter(0)
  ROOT sort = f32[10] sort(x), dimensions={0}, to_apply=compare
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const auto* compare = Cast<HloCompareInstruction>(
      module->GetComputationWithName("compare")->root_instruction());

  EXPECT_EQ(MatchSimpleSortComparator(compare),
            std::make_pair(int64_t{-1}, int64_t{-1}));
  EXPECT_EQ(MatchNumpySortComparator(compare),
            std::make_pair(int64_t{0}, int64_t{1}));
}

TEST_F(SortUtilsTest, MatchRawTotalOrderIsNotNumpyOrder) {
  constexpr char kHlo[] = R"(
HloModule test_module

compare {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT cmp = pred[] compare(p0, p1), direction=LT, type=TOTALORDER
}

ENTRY main {
  x = f32[10] parameter(0)
  ROOT sort = f32[10] sort(x), dimensions={0}, to_apply=compare
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const auto* compare = Cast<HloCompareInstruction>(
      module->GetComputationWithName("compare")->root_instruction());

  // Raw parameter comparison matches simple sort comparator, but NOT numpy
  // comparator.
  EXPECT_EQ(MatchSimpleSortComparator(compare),
            std::make_pair(int64_t{0}, int64_t{1}));
  EXPECT_EQ(MatchNumpySortComparator(compare),
            std::make_pair(int64_t{-1}, int64_t{-1}));
}

TEST_F(SortUtilsTest, MatchNullCompareInstructionReturnsMinusOne) {
  EXPECT_EQ(MatchSimpleSortComparator(nullptr),
            std::make_pair(int64_t{-1}, int64_t{-1}));
  EXPECT_EQ(MatchNumpySortComparator(nullptr),
            std::make_pair(int64_t{-1}, int64_t{-1}));
}

TEST_F(SortUtilsTest, MatchNonParameterArithmeticReturnsMinusOne) {
  constexpr char kHlo[] = R"(
HloModule test_module

compare {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  c1 = f32[] constant(1.0)
  p0_plus_one = f32[] add(p0, c1)
  ROOT cmp = pred[] compare(p0_plus_one, p1), direction=LT
}

ENTRY main {
  x = f32[10] parameter(0)
  ROOT sort = f32[10] sort(x), dimensions={0}, to_apply=compare
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const auto* compare = Cast<HloCompareInstruction>(
      module->GetComputationWithName("compare")->root_instruction());

  EXPECT_EQ(MatchSimpleSortComparator(compare),
            std::make_pair(int64_t{-1}, int64_t{-1}));
  EXPECT_EQ(MatchNumpySortComparator(compare),
            std::make_pair(int64_t{-1}, int64_t{-1}));
}

TEST_F(SortUtilsTest, MatchMissingNanSelectInNumpyComparatorReturnsMinusOne) {
  constexpr char kHlo[] = R"(
HloModule test_module

compare {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  c_zero = f32[] constant(0)
  lhs_is_zero = pred[] compare(lhs, c_zero), direction=EQ
  lhs_no_neg_zero = f32[] select(lhs_is_zero, c_zero, lhs)
  rhs_is_zero = pred[] compare(rhs, c_zero), direction=EQ
  rhs_no_neg_zero = f32[] select(rhs_is_zero, c_zero, rhs)
  ROOT cmp = pred[] compare(lhs_no_neg_zero, rhs_no_neg_zero), direction=LT, type=TOTALORDER
}

ENTRY main {
  x = f32[10] parameter(0)
  ROOT sort = f32[10] sort(x), dimensions={0}, to_apply=compare
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const auto* compare = Cast<HloCompareInstruction>(
      module->GetComputationWithName("compare")->root_instruction());

  EXPECT_EQ(MatchSimpleSortComparator(compare),
            std::make_pair(int64_t{-1}, int64_t{-1}));
  EXPECT_EQ(MatchNumpySortComparator(compare),
            std::make_pair(int64_t{-1}, int64_t{-1}));
}

TEST_F(SortUtilsTest, MatchInvalidNanConstantInNumpyComparatorReturnsMinusOne) {
  constexpr char kHlo[] = R"(
HloModule test_module

compare {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  lhs_is_nan = pred[] compare(lhs, lhs), direction=NE
  c_not_nan = f32[] constant(42.0)
  c_zero = f32[] constant(0)
  lhs_is_zero = pred[] compare(lhs, c_zero), direction=EQ
  lhs_no_neg_zero = f32[] select(lhs_is_zero, c_zero, lhs)
  lhs_canonical = f32[] select(lhs_is_nan, c_not_nan, lhs_no_neg_zero)

  rhs_is_nan = pred[] compare(rhs, rhs), direction=NE
  rhs_is_zero = pred[] compare(rhs, c_zero), direction=EQ
  rhs_no_neg_zero = f32[] select(rhs_is_zero, c_zero, rhs)
  rhs_canonical = f32[] select(rhs_is_nan, c_not_nan, rhs_no_neg_zero)

  ROOT cmp = pred[] compare(lhs_canonical, rhs_canonical), direction=LT, type=TOTALORDER
}

ENTRY main {
  x = f32[10] parameter(0)
  ROOT sort = f32[10] sort(x), dimensions={0}, to_apply=compare
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const auto* compare = Cast<HloCompareInstruction>(
      module->GetComputationWithName("compare")->root_instruction());

  EXPECT_EQ(MatchSimpleSortComparator(compare),
            std::make_pair(int64_t{-1}, int64_t{-1}));
  EXPECT_EQ(MatchNumpySortComparator(compare),
            std::make_pair(int64_t{-1}, int64_t{-1}));
}

TEST_F(SortUtilsTest, MatchCorruptedPostExpansionPatternReturnsMinusOne) {
  constexpr char kHlo[] = R"(
HloModule test_module

compare {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  lhs_bitcast = s32[] bitcast-convert(lhs)
  c_s32_zero = s32[] constant(0)
  lhs_is_neg = pred[] compare(lhs_bitcast, c_s32_zero), direction=LT
  c_mask = s32[] constant(2147483647)
  lhs_mapped = s32[] xor(lhs_bitcast, c_mask)
  lhs_expanded = s32[] select(lhs_is_neg, lhs_mapped, lhs_bitcast)

  rhs_bitcast = s32[] bitcast-convert(rhs)
  rhs_is_neg = pred[] compare(rhs_bitcast, c_s32_zero), direction=LT
  rhs_mapped = s32[] xor(rhs_bitcast, c_mask)
  rhs_expanded = s32[] select(rhs_is_neg, rhs_mapped, rhs_bitcast)

  ROOT cmp = pred[] compare(lhs_expanded, rhs_expanded), direction=LT, type=SIGNED
}

ENTRY main {
  x = f32[10] parameter(0)
  ROOT sort = f32[10] sort(x), dimensions={0}, to_apply=compare
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const auto* compare = Cast<HloCompareInstruction>(
      module->GetComputationWithName("compare")->root_instruction());

  EXPECT_EQ(MatchSimpleSortComparator(compare),
            std::make_pair(int64_t{-1}, int64_t{-1}));
  EXPECT_EQ(MatchNumpySortComparator(compare),
            std::make_pair(int64_t{-1}, int64_t{-1}));
}

}  // namespace
}  // namespace xla

/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/permutation_sort_expander.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using ::absl_testing::IsOkAndHolds;
using PermutationSortExpanderTest = HloHardwareIndependentTestBase;

TEST_F(PermutationSortExpanderTest, ReplacePermutationSortWithScatter) {
  const char* hlo_string = R"(
    HloModule permutation_sort

    lt_f32 {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, type=TOTALORDER
    }

    lt_s32 {
      p.0.lhs = s32[] parameter(0)
      p.0.rhs = s32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
    }

    ENTRY sort_computation {
      keys = f32[64,8732]{1,0} parameter(0)
      values = s32[64,8732]{1,0} iota(), iota_dimension=1
      sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values), dimensions={1}, to_apply=lt_f32
      gte = s32[64,8732]{1,0} get-tuple-element(sort), index=1
      ROOT sort2 = (s32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(gte, values), dimensions={1}, to_apply=lt_s32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_THAT(PermutationSortExpander().Run(module.get()), IsOkAndHolds(true));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, op::Tuple(op::Iota(),
                      op::Scatter(
                          op::Broadcast(op::Constant()),
                          op::Concatenate(op::Iota(),
                                          op::Reshape(op::GetTupleElement(
                                              op::Sort(), /*tuple_index=*/1))),
                          op::Iota())));
}

TEST_F(PermutationSortExpanderTest, DontReplaceIfWrongComparisonDirection) {
  const char* hlo_string = R"(
    HloModule permutation_sort

    lt_f32 {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, type=TOTALORDER
    }

    lt_s32 {
      p.0.lhs = s32[] parameter(0)
      p.0.rhs = s32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=GT
    }

    ENTRY sort_computation {
      keys = f32[64,8732]{1,0} parameter(0)
      values = s32[64,8732]{1,0} iota(), iota_dimension=1
      sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values), dimensions={1}, to_apply=lt_f32
      gte = s32[64,8732]{1,0} get-tuple-element(sort), index=1
      ROOT sort2 = (s32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(gte, values), dimensions={1}, to_apply=lt_s32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_THAT(PermutationSortExpander().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(PermutationSortExpanderTest, DontReplaceIfComparingWrongParameters) {
  const char* hlo_string = R"(
    HloModule permutation_sort

    lt_f32 {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, type=TOTALORDER
    }

    lt_s32 {
      p.0.lhs = s32[] parameter(0)
      p.0.rhs = s32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.rhs, p.0.lhs), direction=LT
    }

    ENTRY sort_computation {
      keys = f32[64,8732]{1,0} parameter(0)
      values = s32[64,8732]{1,0} iota(), iota_dimension=1
      sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values), dimensions={1}, to_apply=lt_f32
      gte = s32[64,8732]{1,0} get-tuple-element(sort), index=1
      ROOT sort2 = (s32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(gte, values), dimensions={1}, to_apply=lt_s32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_THAT(PermutationSortExpander().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(PermutationSortExpanderTest, DontReplacePermutationSortIfNonIntegral) {
  // Same as ReplacePermutationSortWithScatter except that the iota has F32
  // type.
  const char* hlo_string = R"(
    HloModule permutation_sort

    lt_f32 {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      p.1.lhs = f32[] parameter(2)
      p.1.rhs = f32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, type=TOTALORDER
    }

    ENTRY sort_computation {
      keys = f32[64,8732]{1,0} parameter(0)
      values = f32[64,8732]{1,0} iota(), iota_dimension=1
      sort = (f32[64,8732]{1,0}, f32[64,8732]{1,0}) sort(keys, values), dimensions={1}, to_apply=lt_f32
      gte = f32[64,8732]{1,0} get-tuple-element(sort), index=1
      ROOT sort2 = (f32[64,8732]{1,0}, f32[64,8732]{1,0}) sort(gte, values), dimensions={1}, to_apply=lt_f32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_THAT(PermutationSortExpander().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(PermutationSortExpanderTest, DontReplacePermutationSortWrongDimensions) {
  // Same as ReplacePermutationSortWithScatter except that the sort dimensions
  // don't match.
  const char* hlo_string = R"(
   HloModule permutation_sort

    lt_f32 {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT, type=TOTALORDER
    }

    lt_s32 {
      p.0.lhs = s32[] parameter(0)
      p.0.rhs = s32[] parameter(1)
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
    }

    ENTRY sort_computation {
      keys = f32[64,8732]{1,0} parameter(0)
      values = s32[64,8732]{1,0} iota(), iota_dimension=1
      sort = (f32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(keys, values), dimensions={1}, to_apply=lt_f32
      gte = s32[64,8732]{1,0} get-tuple-element(sort), index=1
      ROOT sort2 = (s32[64,8732]{1,0}, s32[64,8732]{1,0}) sort(gte, values), dimensions={0}, to_apply=lt_s32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  EXPECT_THAT(PermutationSortExpander().Run(module.get()), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla

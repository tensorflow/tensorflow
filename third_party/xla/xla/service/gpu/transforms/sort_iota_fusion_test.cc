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

#include "xla/service/gpu/transforms/sort_iota_fusion.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"

namespace m = ::xla::match;

namespace xla::gpu {
namespace {

using SortIotaFusionTest = HloHardwareIndependentTestBase;

TEST_F(SortIotaFusionTest, FuseIota) {
  auto module = *ParseAndReturnVerifiedModule(R"(
    HloModule module

    sorting_computation {
      %lhs_key = s32[] parameter(0)
      %rhs_key = s32[] parameter(1)
      %lhs_index = s32[] parameter(2)
      %rhs_index = s32[] parameter(3)
      %lhs_index2 = s32[] parameter(4)
      %rhs_index2 = s32[] parameter(5)
      %lt_key = pred[] compare(%lhs_key, %rhs_key), direction=LT
      %gt_key = pred[] compare(%rhs_key, %lhs_key), direction=LT
      %eq_key = pred[] compare(%lt_key, %gt_key), direction=EQ
      %lt_index = pred[] compare(%lhs_index, %rhs_index), direction=LT
      ROOT res = pred[] select(%eq_key, %lt_index, %lt_key)
    }

    ENTRY main {
      p0 = s32[16384]{0} parameter(0)
      neg = s32[16384]{0} negate(p0)
      iota = s32[16384]{0} iota(), iota_dimension=0
      ROOT sort = (s32[16384]{0}, s32[16384]{0}, s32[16384]{0}) sort(neg, iota, iota), dimensions={0}, is_stable=true, to_apply=sorting_computation
    }
  )");
  EXPECT_THAT(SortIotaFusion().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Fusion(&fusion, m::Negate())));
  EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kCustom);
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Sort(m::Parameter(), m::Iota(), m::Iota())));
}

}  // namespace
}  // namespace xla::gpu

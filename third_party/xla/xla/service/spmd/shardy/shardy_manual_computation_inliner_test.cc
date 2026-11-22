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

#include "xla/service/spmd/shardy/shardy_manual_computation_inliner.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::sdy {
namespace {

using ::absl_testing::IsOkAndHolds;
using ShardyManualComputationInlinerTest = HloHardwareIndependentTestBase;

TEST_F(ShardyManualComputationInlinerTest,
       InlinesManualComputationAndPreservesOtherCalls) {
  constexpr char kHlo[] = R"hlo(
HloModule test

xla.sdy.inlinable_manual_computation_body.0 {
  p0 = f32[4] parameter(0)
  ROOT neg = f32[4] negate(p0)
}

regular_callee {
  p0 = f32[4] parameter(0)
  ROOT exp = f32[4] exponential(p0)
}

composite_callee {
  p0 = f32[4] parameter(0)
  ROOT abs = f32[4] abs(p0)
}

ENTRY main {
  x = f32[4] parameter(0)
  manual_call = f32[4] call(x), to_apply=xla.sdy.inlinable_manual_computation_body.0
  regular_call = f32[4] call(manual_call), to_apply=regular_callee
  ROOT composite_call = f32[4] call(regular_call), to_apply=composite_callee,
      is_composite=true,
      frontend_attributes={composite.attributes="{}",composite.name="test.comp",composite.version="1"}
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  EXPECT_THAT(ShardyManualComputationInliner{}.Run(module.get()),
              IsOkAndHolds(true));

  EXPECT_EQ(module->GetComputationWithName(
                "xla.sdy.inlinable_manual_computation_body.0"),
            nullptr);
  HloInstruction* neg = FindInstruction(module.get(), HloOpcode::kNegate);
  ASSERT_NE(neg, nullptr);
  EXPECT_EQ(neg->parent(), module->entry_computation());
  EXPECT_NE(module->GetComputationWithName("regular_callee"), nullptr);
  EXPECT_NE(module->GetComputationWithName("composite_callee"), nullptr);
  EXPECT_NE(FindInstruction(module.get(), "regular_call"), nullptr);
  EXPECT_NE(FindInstruction(module.get(), "composite_call"), nullptr);
}

TEST_F(ShardyManualComputationInlinerTest, InlinesNestedManualComputations) {
  constexpr char kHlo[] = R"hlo(
HloModule test

xla.sdy.inlinable_manual_computation_body.inner {
  p0 = f32[4] parameter(0)
  ROOT neg = f32[4] negate(p0)
}

xla.sdy.inlinable_manual_computation_body.outer {
  p0 = f32[4] parameter(0)
  inner = f32[4] call(p0), to_apply=xla.sdy.inlinable_manual_computation_body.inner
  ROOT add = f32[4] add(inner, p0)
}

ENTRY main {
  x = f32[4] parameter(0)
  ROOT out = f32[4] call(x), to_apply=xla.sdy.inlinable_manual_computation_body.outer
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  EXPECT_THAT(ShardyManualComputationInliner{}.Run(module.get()),
              IsOkAndHolds(true));

  EXPECT_EQ(FindInstruction(module.get(), HloOpcode::kCall), nullptr);
  EXPECT_EQ(module->computation_count(), 1);
  EXPECT_NE(FindInstruction(module.get(), HloOpcode::kNegate), nullptr);
  EXPECT_NE(FindInstruction(module.get(), HloOpcode::kAdd), nullptr);
}

TEST_F(ShardyManualComputationInlinerTest,
       ReturnsFalseWhenNoInlineableManualComputation) {
  constexpr char kHlo[] = R"hlo(
HloModule test

xla.sdy.manual_computation_body.0 {
  p0 = f32[4] parameter(0)
  ROOT neg = f32[4] negate(p0)
}

ENTRY main {
  x = f32[4] parameter(0)
  ROOT out = f32[4] call(x), to_apply=xla.sdy.manual_computation_body.0
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  EXPECT_THAT(ShardyManualComputationInliner{}.Run(module.get()),
              IsOkAndHolds(false));
  EXPECT_NE(FindInstruction(module.get(), HloOpcode::kCall), nullptr);
}

}  // namespace
}  // namespace xla::sdy

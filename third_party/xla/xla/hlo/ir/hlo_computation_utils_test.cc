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

#include "xla/hlo/ir/hlo_computation_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::hlo_computation_utils {
namespace {

using ::absl_testing::IsOkAndHolds;

bool HasSpecialDeviceAttr(const HloInstruction* instr) {
  const auto& map = instr->frontend_attributes().map();
  auto it = map.find("device");
  return it != map.end() && it->second == "special";
}

class HloComputationUtilsTest : public HloHardwareIndependentTestBase {};

TEST_F(HloComputationUtilsTest, CloneComputationDefaultSuffix) {
  constexpr absl::string_view hlo_string = R"hlo(
    // CHECK-NOT: is_scheduled=true
    HloModule test

    // CHECK-LABEL: %callee (
    // CHECK-NEXT: %p0 = f32[4]{0} parameter(0)
    // CHECK-NEXT: %p1 = f32[4]{0} parameter(1)
    // CHECK-NEXT: ROOT %sum = f32[4]{0} add(%p0, %p1)
    callee {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      ROOT sum = f32[4] add(p0, p1)
    }

    // CHECK-LABEL: ENTRY %main (
    ENTRY main {
      a = f32[4] parameter(0)
      b = f32[4] parameter(1)
      ROOT call = f32[4] call(a, b), to_apply=callee
    }

    // CHECK-LABEL: %callee.clone (
    // CHECK-NEXT: %p0.1 = f32[4]{0} parameter(0)
    // CHECK-NEXT: %p1.1 = f32[4]{0} parameter(1)
    // CHECK-NEXT: ROOT %sum.1 = f32[4]{0} add(%p0.1, %p1.1)
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  CloneComputation(FindComputation(module.get(), "callee"));
  EXPECT_THAT(RunFileCheck(module->ToString(), hlo_string), IsOkAndHolds(true));
}

TEST_F(HloComputationUtilsTest, CloneComputationWithEmptySuffix) {
  constexpr absl::string_view hlo_string = R"hlo(
    HloModule test

    // CHECK-LABEL: %callee (
    callee {
      ROOT p0 = f32[4] parameter(0)
    }

    // CHECK-LABEL: ENTRY %main (
    ENTRY main {
      a = f32[4] parameter(0)
      ROOT call = f32[4] call(a), to_apply=callee
    }

    // CHECK-LABEL: %callee.clone (
    // CHECK-NEXT: ROOT %p0.1 = f32[4]{0} parameter(0)
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  CloneComputation(FindComputation(module.get(), "callee"), "");
  EXPECT_THAT(RunFileCheck(module->ToString(), hlo_string), IsOkAndHolds(true));
}

TEST_F(HloComputationUtilsTest, CloneComputationWithCustomSuffix) {
  constexpr absl::string_view hlo_string = R"hlo(
    HloModule test

    // CHECK-LABEL: %callee (
    callee {
      ROOT p0 = f32[4] parameter(0)
    }

    // CHECK-LABEL: ENTRY %main (
    ENTRY main {
      a = f32[4] parameter(0)
      ROOT call = f32[4] call(a), to_apply=callee
    }

    // CHECK-LABEL: %callee.specialized (
    // CHECK-NEXT: ROOT %p0.1 = f32[4]{0} parameter(0)
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  CloneComputation(FindComputation(module.get(), "callee"), "specialized");
  EXPECT_THAT(RunFileCheck(module->ToString(), hlo_string), IsOkAndHolds(true));
}

TEST_F(HloComputationUtilsTest, CloneUnscheduledComputationInScheduledModule) {
  constexpr absl::string_view hlo_string = R"hlo(
    // CHECK: HloModule test, is_scheduled=true
    HloModule test, is_scheduled=true

    // CHECK-LABEL: %callee (
    callee {
      p0 = f32[4] parameter(0)
      neg = f32[4] negate(p0)
      ROOT sum = f32[4] add(neg, neg)
    }, execution_thread="parallel_thread"

    // CHECK-LABEL: ENTRY %main (
    ENTRY main {
      a = f32[4] parameter(0)
      start = ((f32[4]), f32[4], s32[]) async-start(a),
          async_execution_thread="parallel_thread", calls=callee
      ROOT done = f32[4] async-done(start)
    }

    // CHECK-LABEL: %callee.clone (
    // CHECK-NEXT: %p0.1 = f32[4]{0} parameter(0)
    // CHECK-NEXT: %neg.1 = f32[4]{0} negate(%p0.1)
    // CHECK-NEXT: ROOT %sum.1 = f32[4]{0} add(%neg.1, %neg.1)
    // CHECK-NEXT: }, execution_thread="parallel_thread"
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  HloComputation* callee = FindComputation(module.get(), "callee");
  module->schedule().remove_computation(callee);
  CloneComputation(callee);
  EXPECT_THAT(RunFileCheck(module->ToString(), hlo_string), IsOkAndHolds(true));
}

TEST_F(HloComputationUtilsTest, CloneScheduledComputationInScheduledModule) {
  constexpr absl::string_view hlo_string = R"hlo(
    // CHECK: HloModule test, is_scheduled=true
    HloModule test, is_scheduled=true

    // CHECK-LABEL: %callee (
    // CHECK-NEXT: %p0 = f32[4]{0} parameter(0)
    // CHECK-NEXT: %exp = f32[4]{0} exponential(%p0)
    // CHECK-NEXT: %neg = f32[4]{0} negate(%p0)
    // CHECK-NEXT: ROOT %sum = f32[4]{0} add(%neg, %exp)
    callee {
      p0 = f32[4] parameter(0)
      neg = f32[4] negate(p0)
      exp = f32[4] exponential(p0)
      ROOT sum = f32[4] add(neg, exp)
    }

    // CHECK-LABEL: ENTRY %main (
    ENTRY main {
      a = f32[4] parameter(0)
      ROOT call = f32[4] call(a), to_apply=callee
    }

    // CHECK-LABEL: %callee.clone (
    // CHECK-NEXT: %p0.1 = f32[4]{0} parameter(0)
    // CHECK-NEXT: %exp.1 = f32[4]{0} exponential(%p0.1)
    // CHECK-NEXT: %neg.1 = f32[4]{0} negate(%p0.1)
    // CHECK-NEXT: ROOT %sum.1 = f32[4]{0} add(%neg.1, %exp.1)
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  HloComputation* callee = FindComputation(module.get(), "callee");
  module->schedule().set_sequence(callee,
                                  {callee->GetInstructionWithName("p0"),
                                   callee->GetInstructionWithName("exp"),
                                   callee->GetInstructionWithName("neg"),
                                   callee->GetInstructionWithName("sum")});
  CloneComputation(callee);
  EXPECT_THAT(RunFileCheck(module->ToString(), hlo_string), IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(module->schedule().ToString(), R"(
    // CHECK-LABEL: computation callee.clone (thread main):
    // CHECK-NEXT: p0.1
    // CHECK-NEXT: exp.1
    // CHECK-NEXT: neg.1
    // CHECK-NEXT: sum.1
  )"),
              IsOkAndHolds(true));
}

TEST_F(HloComputationUtilsTest,
       SpecializeComputationsNoSharedCallsReturnsFalse) {
  constexpr absl::string_view hlo_string = R"hlo(
    HloModule test

    ordinary_callee {
      ROOT p0 = s32[32] parameter(0)
    }

    special_callee {
      p0 = s32[32] parameter(0)
      ROOT neg = s32[32] negate(p0)
    }

    ENTRY main {
      p0 = s32[32] parameter(0)
      c0 = s32[32] call(p0), to_apply=ordinary_callee
      c1 = s32[32] call(p0), to_apply=special_callee,
          frontend_attributes={device="special"}
      ROOT sum = s32[32] add(c0, c1)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(SpecializeComputations(module.get(), HasSpecialDeviceAttr));
  EXPECT_EQ(module->computation_count(), 3);
}

TEST_F(HloComputationUtilsTest,
       SpecializeComputationsDirectSharedCallDefaultSuffix) {
  constexpr absl::string_view hlo_string = R"hlo(
    HloModule test

    // CHECK-LABEL: %foo.clone (
    // CHECK-NEXT: %foo_p.1 = s32[32]{0} parameter(0)
    // CHECK-NEXT: ROOT %mul.1 = s32[32]{0} multiply(%foo_p.1, %foo_p.1)

    // CHECK-LABEL: %foo (
    // CHECK-NEXT: %foo_p = s32[32]{0} parameter(0)
    // CHECK-NEXT: ROOT %mul = s32[32]{0} multiply(%foo_p, %foo_p)
    foo {
      foo_p = s32[32] parameter(0)
      ROOT mul = s32[32] multiply(foo_p, foo_p)
    }

    // CHECK-LABEL: ENTRY %main (
    // CHECK-NEXT: %main_p = s32[32]{0} parameter(0)
    // CHECK-NEXT: %special_call = s32[32]{0} call(%main_p), to_apply=%foo.clone
    // CHECK-NEXT: %ordinary_call = s32[32]{0} call(%main_p), to_apply=%foo{{$}}
    // CHECK-NEXT: ROOT %sum = s32[32]{0} add(%special_call, %ordinary_call)
    ENTRY main {
      main_p = s32[32] parameter(0)
      special_call = s32[32] call(main_p), to_apply=foo,
          frontend_attributes={device="special"}
      ordinary_call = s32[32] call(main_p), to_apply=foo
      ROOT sum = s32[32] add(special_call, ordinary_call)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(SpecializeComputations(module.get(), HasSpecialDeviceAttr));
  EXPECT_THAT(RunFileCheck(module->ToString(), hlo_string), IsOkAndHolds(true));
}

TEST_F(HloComputationUtilsTest,
       SpecializeComputationsSharedOuterAndNestedCallsCustomSuffix) {
  constexpr absl::string_view hlo_string = R"hlo(
    HloModule test

    // CHECK-LABEL: %bar.sp (
    // CHECK-NEXT: %bar_p.1 = s32[32]{0} parameter(0)
    // CHECK-NEXT: ROOT %mul.1 = s32[32]{0} multiply(%bar_p.1, %bar_p.1)

    // CHECK-LABEL: %foo.sp (
    // CHECK-NEXT: %foo_p.1 = s32[32]{0} parameter(0)
    // CHECK-NEXT: ROOT %bar_call.1 = s32[32]{0} call(%foo_p.1), to_apply=%bar.sp

    // CHECK-LABEL: %bar (
    // CHECK-NEXT: %bar_p = s32[32]{0} parameter(0)
    // CHECK-NEXT: ROOT %mul = s32[32]{0} multiply(%bar_p, %bar_p)
    bar {
      bar_p = s32[32] parameter(0)
      ROOT mul = s32[32] multiply(bar_p, bar_p)
    }

    // CHECK-LABEL: %foo (
    // CHECK-NEXT: %foo_p = s32[32]{0} parameter(0)
    // CHECK-NEXT: ROOT %bar_call = s32[32]{0} call(%foo_p), to_apply=%bar{{$}}
    foo {
      foo_p = s32[32] parameter(0)
      ROOT bar_call = s32[32] call(foo_p), to_apply=bar
    }

    // CHECK-LABEL: ENTRY %main (
    // CHECK-NEXT: %main_p = s32[32]{0} parameter(0)
    // CHECK-NEXT: %special_call = s32[32]{0} call(%main_p), to_apply=%foo.sp
    // CHECK-NEXT: %ordinary_call = s32[32]{0} call(%main_p), to_apply=%foo{{$}}
    // CHECK-NEXT: ROOT %sum = s32[32]{0} add(%special_call, %ordinary_call)
    ENTRY main {
      main_p = s32[32] parameter(0)
      special_call = s32[32] call(main_p), to_apply=foo,
          frontend_attributes={device="special"}
      ordinary_call = s32[32] call(main_p), to_apply=foo
      ROOT sum = s32[32] add(special_call, ordinary_call)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(SpecializeComputations(module.get(), HasSpecialDeviceAttr,
                                     /*suffix=*/"sp"));
  EXPECT_THAT(RunFileCheck(module->ToString(), hlo_string), IsOkAndHolds(true));
}

TEST_F(HloComputationUtilsTest,
       SpecializeComputationsMultiLevelSharedCallsMinimallyClones) {
  constexpr absl::string_view hlo_string = R"hlo(
    HloModule test

    // CHECK-LABEL: %leaf (
    // CHECK-NEXT: %leaf_p = s32[32]{0} parameter(0)
    // CHECK-NEXT: ROOT %mul = s32[32]{0} multiply(%leaf_p, %leaf_p)
    leaf {
      leaf_p = s32[32] parameter(0)
      ROOT mul = s32[32] multiply(leaf_p, leaf_p)
    }

    // CHECK-LABEL: %mid (
    // CHECK-NEXT: %mid_p = s32[32]{0} parameter(0)
    // CHECK-NEXT: ROOT %leaf_call = s32[32]{0} call(%mid_p), to_apply=%leaf{{$}}
    mid {
      mid_p = s32[32] parameter(0)
      ROOT leaf_call = s32[32] call(mid_p), to_apply=leaf
    }

    // CHECK-LABEL: %leaf.sp (
    // CHECK-NEXT: %leaf_p.1 = s32[32]{0} parameter(0)
    // CHECK-NEXT: ROOT %mul.1 = s32[32]{0} multiply(%leaf_p.1, %leaf_p.1)

    // CHECK-LABEL: %mid.sp (
    // CHECK-NEXT: %mid_p.1 = s32[32]{0} parameter(0)
    // CHECK-NEXT: ROOT %leaf_call.1 = s32[32]{0} call(%mid_p.1), to_apply=%leaf.sp

    // CHECK-LABEL: %special_outer (
    // CHECK-NEXT: %outer_p = s32[32]{0} parameter(0)
    // CHECK-NEXT: ROOT %mid_call = s32[32]{0} call(%outer_p), to_apply=%mid.sp
    special_outer {
      outer_p = s32[32] parameter(0)
      ROOT mid_call = s32[32] call(outer_p), to_apply=mid
    }

    // CHECK-NOT: %special_outer.sp
    // CHECK-NOT: %leaf.sp.
    // CHECK-NOT: %mid.sp.

    // CHECK-LABEL: ENTRY %main (
    // CHECK-NEXT: %main_p = s32[32]{0} parameter(0)
    // CHECK-NEXT: %ord_0 = s32[32]{0} call(%main_p), to_apply=%mid{{$}}
    // CHECK-NEXT: %ord_1 = s32[32]{0} call(%ord_0), to_apply=%mid{{$}}
    // CHECK-NEXT: %sp_0 = s32[32]{0} call(%ord_1), to_apply=%mid.sp
    // CHECK-NEXT: %sp_1 = s32[32]{0} call(%ord_1), to_apply=%special_outer
    // CHECK-NEXT: ROOT %sum = s32[32]{0} add(%sp_0, %sp_1)
    ENTRY main {
      main_p = s32[32] parameter(0)
      ord_0 = s32[32] call(main_p), to_apply=mid
      ord_1 = s32[32] call(ord_0), to_apply=mid
      sp_0 = s32[32] call(ord_1), to_apply=mid,
          frontend_attributes={device="special"}
      sp_1 = s32[32] call(ord_1), to_apply=special_outer,
          frontend_attributes={device="special"}
      ROOT sum = s32[32] add(sp_0, sp_1)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(SpecializeComputations(module.get(), HasSpecialDeviceAttr,
                                     /*suffix=*/"sp"));
  EXPECT_THAT(RunFileCheck(module->ToString(), hlo_string), IsOkAndHolds(true));
}

TEST_F(HloComputationUtilsTest,
       SpecializeComputationsSpecialCallInsideOrdinaryFunction) {
  constexpr absl::string_view hlo_string = R"hlo(
    HloModule test

    // CHECK-LABEL: %shared_leaf.clone (
    // CHECK-NEXT: %leaf_p.1 = s32[32]{0} parameter(0)
    // CHECK-NEXT: ROOT %mul.1 = s32[32]{0} multiply(%leaf_p.1, %leaf_p.1)

    // CHECK-LABEL: %ordinary_wrapper (
    // CHECK-NEXT: %wrap_p = s32[32]{0} parameter(0)
    // CHECK-NEXT: ROOT %special_call = s32[32]{0} call(%wrap_p), to_apply=%shared_leaf.clone

    // CHECK-LABEL: %shared_leaf (
    // CHECK-NEXT: %leaf_p = s32[32]{0} parameter(0)
    // CHECK-NEXT: ROOT %mul = s32[32]{0} multiply(%leaf_p, %leaf_p)
    shared_leaf {
      leaf_p = s32[32] parameter(0)
      ROOT mul = s32[32] multiply(leaf_p, leaf_p)
    }

    ordinary_wrapper {
      wrap_p = s32[32] parameter(0)
      ROOT special_call = s32[32] call(wrap_p), to_apply=shared_leaf,
          frontend_attributes={device="special"}
    }

    // CHECK-LABEL: ENTRY %main (
    // CHECK-NEXT: %main_p = s32[32]{0} parameter(0)
    // CHECK-NEXT: %a = s32[32]{0} call(%main_p), to_apply=%ordinary_wrapper{{$}}
    // CHECK-NEXT: %b = s32[32]{0} call(%main_p), to_apply=%shared_leaf{{$}}
    // CHECK-NEXT: ROOT %sum = s32[32]{0} add(%a, %b)
    ENTRY main {
      main_p = s32[32] parameter(0)
      a = s32[32] call(main_p), to_apply=ordinary_wrapper
      b = s32[32] call(main_p), to_apply=shared_leaf
      ROOT sum = s32[32] add(a, b)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(SpecializeComputations(module.get(), HasSpecialDeviceAttr));
  EXPECT_THAT(RunFileCheck(module->ToString(), hlo_string), IsOkAndHolds(true));
}

TEST_F(HloComputationUtilsTest,
       SpecializeComputationsIgnoresUnreachableComputation) {
  constexpr absl::string_view hlo_string = R"hlo(
    HloModule test

    shared_leaf {
      ROOT p0 = s32[32] parameter(0)
    }

    dead_caller {
      p0 = s32[32] parameter(0)
      ROOT dead_special = s32[32] call(p0), to_apply=shared_leaf,
          frontend_attributes={device="special"}
    }

    ENTRY main {
      p0 = s32[32] parameter(0)
      ROOT live_ordinary = s32[32] call(p0), to_apply=shared_leaf
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(SpecializeComputations(module.get(), HasSpecialDeviceAttr));
  EXPECT_EQ(module->computation_count(), 3);
}

}  // namespace
}  // namespace xla::hlo_computation_utils

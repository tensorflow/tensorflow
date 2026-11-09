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
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::hlo_computation_utils {
namespace {

using ::absl_testing::IsOkAndHolds;

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

}  // namespace
}  // namespace xla::hlo_computation_utils

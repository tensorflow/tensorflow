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
#include "xla/hlo/transforms/simplifiers/computation_canonicalizers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ComputationCanonicalizersTest = HloHardwareIndependentTestBase;

TEST_F(ComputationCanonicalizersTest, MoveParametersToFront) {
  const char* hlo = R"(
      HloModule TestModule, is_scheduled=true

      %fused_computation (param_0: s32[], param_1: s32[]) -> s32[] {
        %p0 = s32[] parameter(0)
        %p1 = s32[] parameter(1)
        ROOT %add = s32[] add(s32[] %p0, s32[] %p1)
      }

      %fused_computation.1 (param_0: s32[], param_1: s32[]) -> s32[] {
        %p0 = s32[] parameter(0)
        %p1 = s32[] parameter(1)
        ROOT %add = s32[] add(s32[] %p0, s32[] %p1)
      }

      ENTRY %main (a: s32[], b: s32[], c: s32[]) -> s32[] {
        %a = s32[] parameter(0)
        %b = s32[] parameter(1)
        %fusion = s32[] fusion(s32[] %a, s32[] %b), kind=kLoop, calls=%fused_computation
        %c = s32[] parameter(2)
        ROOT %fusion.1 = s32[] fusion(s32[] %a, s32[] %c), kind=kLoop, calls=%fused_computation.1
      })";

  const char* expected = R"(
// CHECK: ENTRY %main (a: s32[], b: s32[], c: s32[]) -> s32[] {
// CHECK:   %a = s32[] parameter(0)
// CHECK:   %b = s32[] parameter(1)
// CHECK:   %c = s32[] parameter(2)
// CHECK:   %fusion = s32[] fusion(%a, %b), kind=kLoop, calls=%fused_computation
// CHECK:   ROOT %fusion.1 = s32[] fusion(%a, %c), kind=kLoop, calls=%fused_computation.1
// CHECK: })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK(MoveParametersAndConstantsToFront(*module->entry_computation()));
  EXPECT_THAT(
      RunFileCheck(
          module->ToString(HloPrintOptions{}.set_print_operand_shape(false)),
          expected),
      absl_testing::IsOkAndHolds(true));
}

TEST_F(ComputationCanonicalizersTest, MoveGTEsRightAfterTupleDefinition) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m, is_scheduled=true
e {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  t = tuple(a, b)
  x = s32[] add(a, b)
  g0 = s32[] get-tuple-element(t), index=0
  g1 = s32[] get-tuple-element(t), index=1
  r = s32[] multiply(g0, g1)
})"));
  EXPECT_THAT(MoveGTEsRightAfterTupleDefinition(*module->entry_computation()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(module->ToString(),
                           R"(
// CHECK:      parameter
// CHECK-NEXT: parameter
// CHECK-NEXT: tuple
// CHECK-NEXT: get-tuple-element
// CHECK-NEXT: get-tuple-element
// CHECK-NEXT: add
// CHECK-NEXT: multiply
)"),
              absl_testing::IsOkAndHolds(true));
}

}  // namespace
}  // namespace xla

/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/fusion_constant_sinking.h"

#include <memory>
#include <string>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/pattern_matcher.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using FusionConstantSinkingTest = HloHardwareIndependentTestBase;

TEST_F(FusionConstantSinkingTest, SinkConstant) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  
    %fused_computation.slice (param_0.51117: s8[56,4096,4096], param_1: s32[]) -> s8[1,4096,4096] {
      %param_0.51117 = s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} parameter(0)
      p1 = s32[]{:T(128)} parameter(1)
      %constant.85694 = s32[]{:T(128)} constant(0)
      ROOT %dynamic-slice.22040 = s8[1,4096,4096]{2,1,0:T(8,128)(4,1)} dynamic-slice(s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} %param_0.51117, s32[]{:T(128)} p1, s32[]{:T(128)} %constant.85694, s32[]{:T(128)} %constant.85694), dynamic_slice_sizes={1,4096,4096}
    }

    ENTRY main {
      p0 = s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} parameter(0)
      c = s32[]{:T(128)} constant(10)
      ROOT out = s8[1,4096,4096]{2,1,0:T(8,128)(4,1)} fusion(s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} p0, s32[]{:T(128)} c), kind=kLoop, calls=%fused_computation.slice
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  FusionConstantSinking constant_sinking;

  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_sinking, module.get()));

  EXPECT_TRUE(result);
  EXPECT_THAT(
      module->GetComputationWithName("fused_computation.slice")
          ->root_instruction(),
      GmockMatch(match::DynamicSlice(match::Parameter(0), match::Constant(),
                                     match::Constant(), match::Constant())));
}

TEST_F(FusionConstantSinkingTest, SingleOperandFusionNoSink) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  
    %fused_computation (param_1: s8[]) -> s8[1,4096,4096] {
      param0 = s8[] parameter(0)
      ROOT out = s8[1,4096,4096]{2,1,0:T(8,128)(4,1)} broadcast(param0), dimensions={}
    }

    ENTRY main {
      c = s8[]{:T(128)} constant(10)
      ROOT out = s8[1,4096,4096]{2,1,0:T(8,128)(4,1)} fusion(s8[]{:T(128)} c), kind=kLoop, calls=%fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  FusionConstantSinking constant_sinking;

  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_sinking, module.get()));

  EXPECT_FALSE(result);
}

// Fusions with single operands are not considered because the nested
// computation will be left without any parameters
TEST_F(FusionConstantSinkingTest, SingleOperandUserNoSink) {
  std::string hlo_string = R"(
  HloModule SimpleLoop

    %fused_computation.inner (param_1: s32[]) -> s32[] {
      p1 = s32[]{:T(128)} parameter(0)
      %constant.85694 = s32[]{:T(128)} constant(10)
      ROOT out = s32[] add(p1, %constant.85694)
    }

    %fused_computation (param_0.51117: s32[4096,4096], param_1:
    s32[]) -> s32[4096,4096] {
      %param_0.51117 = s32[4096,4096]{1,0:T(8,128)(4,1)} parameter(0)
      p1 = s32[]{:T(128)} parameter(1)
      %inner.fusion = s32[] fusion(s32[]{:T(128)} p1), kind=kLoop, calls=%fused_computation.inner
      %broadcast = s32[4096,4096]{1,0:T(8,128)(4,1)} broadcast(%inner.fusion), dimensions={}
      ROOT out = s32[4096,4096] add(%broadcast, %param_0.51117)
    }

    ENTRY main {
      p0 = s32[4096,4096]{1,0:T(8,128)(4,1)} parameter(0)
      c = s32[]{:T(128)} constant(10)
      ROOT out = s32[4096,4096]{1,0:T(8,128)(4,1)}
      fusion(s32[4096,4096]{1,0:T(8,128)(4,1)} p0, s32[]{:T(128)} c), kind=kLoop, calls=%fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  FusionConstantSinking constant_sinking;

  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_sinking, module.get()));

  EXPECT_FALSE(result);
}

TEST_F(FusionConstantSinkingTest, NonScalarNoSink) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  
    %fused_computation (param_1: s8[2], p1: s8[2,4096,4096]) -> s8[2,4096,4096] {
      param0 = s8[2] parameter(0)
      param1 = s8[2,4096,4096]{2,1,0:T(8,128)(4,1)} parameter(1)
      bcast = s8[2,4096,4096]{2,1,0:T(8,128)(4,1)} broadcast(param0), dimensions={0}
      ROOT out = s8[2,4096,4096]{2,1,0:T(8,128)(4,1)} add(param1, bcast)
    }

    ENTRY main {
      p = s8[2,4096,4096]{2,1,0:T(8,128)(4,1)} parameter(0)
      c = s8[2]{0:T(128)} constant({10,20})
      ROOT out = s8[2,4096,4096]{2,1,0:T(8,128)(4,1)} fusion(s8[2]{0:T(128)} c, p), kind=kLoop, calls=%fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  FusionConstantSinking constant_sinking;

  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_sinking, module.get()));

  EXPECT_FALSE(result);
}

TEST_F(FusionConstantSinkingTest, SinkConstantNested) {
  std::string hlo_string = R"(
  HloModule SimpleLoop

    %fused_computation.inner (param_0.51117: s8[56,4096,4096], param_1:
    s32[]) -> s8[1,4096,4096] {
      %param_0.51117 = s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} parameter(0)
      p1 = s32[]{:T(128)} parameter(1)
      %constant.85694 = s32[]{:T(128)} constant(0)

      ROOT %dynamic-slice.22040 = s8[1,4096,4096]{2,1,0:T(8,128)(4,1)}
      dynamic-slice(s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} %param_0.51117,
      s32[]{:T(128)} p1, s32[]{:T(128)} %constant.85694, s32[]{:T(128)}
      %constant.85694), dynamic_slice_sizes={1,4096,4096}
    }

    %fused_computation (param_0.51117: s8[56,4096,4096], param_1:
    s32[]) -> s8[4096,4096] {
      %param_0.51117 = s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} parameter(0)
      p1 = s32[]{:T(128)} parameter(1)
    
      %inner.fusion = s8[1,4096,4096]{2,1,0:T(8,128)(4,1)} fusion(s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} %param_0.51117, s32[]{:T(128)} p1), kind=kLoop, calls=%fused_computation.inner

      ROOT %bitcast = s8[4096,4096]{1,0:T(8,128)(4,1)} bitcast(s8[1,4096,4096]{2,1,0:T(8,128)(4,1)} %inner.fusion)
    }

    ENTRY main {
      p0 = s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} parameter(0)
      c = s32[]{:T(128)} constant(10)
      ROOT out = s8[4096,4096]{1,0:T(8,128)(4,1)}
      fusion(s8[56,4096,4096]{2,1,0:T(8,128)(4,1)} p0, s32[]{:T(128)} c),
      kind=kLoop, calls=%fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  FusionConstantSinking constant_sinking;

  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_sinking, module.get()));

  EXPECT_TRUE(result);
  EXPECT_THAT(
      module->GetComputationWithName("fused_computation")->num_parameters(), 1);
  EXPECT_THAT(module->GetComputationWithName("fused_computation.inner")
                  ->num_parameters(),
              1);
}

}  // namespace
}  // namespace xla

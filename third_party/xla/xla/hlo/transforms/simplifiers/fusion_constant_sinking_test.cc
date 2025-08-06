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

TEST_F(FusionConstantSinkingTest, SinkNonTopLevelConstant) {
  std::string hlo_string = R"(
  HloModule fusion.2653, entry_computation_layout={(bf16[16,128]{1,0:T(8,128)(2,1)}, u32[]{:T(128)}, bf16[128,16,8,1]{1,2,0,3:T(8,128)(2,1)})->bf16[32,8,2,128]{3,1,2,0:T(8,128)(2,1)}}

fused_computation.4564 {
  param_1.225408 = bf16[128,16,8,1]{1,2,0,3:T(8,128)(2,1)S(1)} parameter(1)
  param_2.150049 = bf16[]{:T(256)} parameter(2)
  pad.29514 = bf16[128,16,8,2]{1,2,0,3:T(8,128)(2,1)} pad(param_1.225408, param_2.150049), padding=0_0x0_0x0_0x0_1
  param_0.180938 = u32[]{:T(128)} parameter(0)
  constant.227383 = u32[]{:T(128)} constant(0)
  ROOT dynamic-slice.51000 = bf16[32,16,8,2]{1,2,0,3:T(8,128)(2,1)} dynamic-slice(pad.29514, param_0.180938, constant.227383, constant.227383, constant.227383), dynamic_slice_sizes={32,16,8,2}
}

fused_computation.4568 {
  param_0.180939 = u32[]{:T(128)} parameter(0)
  param_1.225409 = bf16[128,16,8,1]{1,2,0,3:T(8,128)(2,1)S(1)} parameter(1)
  constant.227393 = bf16[]{:T(256)} constant(-inf)
  fusion.46232 = bf16[32,16,8,2]{1,2,0,3:T(8,128)(2,1)} fusion(param_0.180939, param_1.225409, constant.227393), kind=kLoop, calls=fused_computation.4564
  ROOT copy.63812 = bf16[32,16,8,2]{1,2,3,0:T(8,128)(2,1)} copy(fusion.46232)
}

fused_computation.4579 {
  param_0.12345 = bf16[16,128]{1,0:T(8,128)(2,1)S(1)} parameter(0)
  ROOT bitcast.21843 = bf16[16,128,1,1]{1,0,3,2:T(8,128)(2,1)} bitcast(param_0.12345)
}

fused_computation.4567 {
  param_1.225407 = u32[]{:T(128)} parameter(1)
  param_2.150048 = bf16[128,16,8,1]{1,2,0,3:T(8,128)(2,1)S(1)} parameter(2)
  fusion.2654 = bf16[32,16,8,2]{1,2,3,0:T(8,128)(2,1)} fusion(param_1.225407, param_2.150048), kind=kLoop, calls=fused_computation.4568
  param_0.180937 = bf16[16,128]{1,0:T(8,128)(2,1)S(1)} parameter(0)
  fusion.2661 = bf16[16,128,1,1]{1,0,3,2:T(8,128)(2,1)} fusion(param_0.180937), kind=kLoop, calls=fused_computation.4579
  ROOT convolution.6282 = bf16[32,8,2,128]{3,1,2,0:T(8,128)(2,1)S(1)} convolution(fusion.2654, fusion.2661), window={size=1x1}, dim_labels=0fb1_io01->0b1f, operand_precision={highest,highest}
}

ENTRY fusion.2653 {
  parameter.0 = bf16[16,128]{1,0:T(8,128)(2,1)} parameter(0)
  copy.1 = bf16[16,128]{1,0:T(8,128)(2,1)S(1)} copy(parameter.0)
  parameter.1 = u32[]{:T(128)} parameter(1)
  parameter.2 = bf16[128,16,8,1]{1,2,0,3:T(8,128)(2,1)} parameter(2)
  copy.2 = bf16[128,16,8,1]{1,2,0,3:T(8,128)(2,1)S(1)} copy(parameter.2)
  fusion.2653 = bf16[32,8,2,128]{3,1,2,0:T(8,128)(2,1)S(1)} fusion(copy.1, parameter.1, copy.2), kind=kOutput, calls=fused_computation.4567
  ROOT copy = bf16[32,8,2,128]{3,1,2,0:T(8,128)(2,1)} copy(fusion.2653)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  FusionConstantSinking constant_sinking;

  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&constant_sinking, module.get()));

  EXPECT_TRUE(result);
  EXPECT_THAT(module->GetComputationWithName("fused_computation.4564")
                  ->num_parameters(),
              2);
}

}  // namespace
}  // namespace xla

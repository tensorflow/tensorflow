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

#include "xla/backends/gpu/transforms/fusion_dynamic_memcpy_rewriter.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;

using Parameter = DynamicSliceFusion::Parameter;
using Result = DynamicSliceFusion::Result;
using CstOff = DynamicSliceFusion::ConstantOffset;
using RtOff = DynamicSliceFusion::RuntimeOffset;
using Offsets = std::vector<DynamicSliceFusion::Offset>;

using FusionDynamicMemcpyRewriterTest = HloHardwareIndependentTestBase;

DynamicSliceConfig MakeConfig(int64_t loop_index, int64_t offset,
                              int64_t stride) {
  DynamicSliceConfig config;
  config.set_loop_index(loop_index);
  config.set_byte_offset(offset);
  config.set_byte_stride(stride);
  return config;
}

DynamicSliceConfig MakeStaticConfig(int64_t offset) {
  DynamicSliceConfig config;
  config.set_byte_offset(offset);
  return config;
}

std::optional<std::string> GetCustomFusionName(const HloInstruction* instr) {
  auto config = instr->backend_config<GpuBackendConfig>();
  if (!config.ok()) {
    return std::nullopt;
  }
  const auto& fusion_config = config->fusion_backend_config();
  if (fusion_config.kind() != kCustomFusionKind ||
      !fusion_config.has_custom_fusion_config()) {
    return std::nullopt;
  }
  return fusion_config.custom_fusion_config().name();
}

TEST_F(FusionDynamicMemcpyRewriterTest, RewritesDsToV2Fusion) {
  constexpr char kHlo[] = R"(
    dynamic_slice {
      p0 = s32[4] parameter(0)
      c1 = s32[] constant(1)

      ROOT slice = s32[1] dynamic-slice(p0, c1), dynamic_slice_sizes={1},
          backend_config={"dynamic_slice_config":
              {"byte_offset":"4","byte_stride":"0"}}
    }

    ENTRY main {
      p0 = s32[4] parameter(0)
      ROOT fusion = s32[1] fusion(p0), kind=kLoop, calls=dynamic_slice
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(FusionDynamicMemcpyRewriter().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* fusion =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(fusion->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kCustom);

  auto name = GetCustomFusionName(fusion);
  ASSERT_TRUE(name.has_value()) << module->ToString();
  EXPECT_EQ(*name, kDynamicSliceFusionConfigName);

  // Verify DynamicSliceFusion can parse the rewritten body.
  const HloComputation* body = fusion->fused_instructions_computation();
  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);
  EXPECT_EQ(hero->opcode(), HloOpcode::kCopy);

  ASSERT_OK_AND_ASSIGN(auto parameters,
                       DynamicSliceFusion::ResolveParameters(hero));
  EXPECT_THAT(parameters, ElementsAre(Parameter{
                              0, ShapeUtil::MakeShape(S32, {4}),
                              ShapeUtil::MakeShape(S32, {1}),
                              MakeStaticConfig(4), Offsets{CstOff{0, 0}}}));

  ASSERT_OK_AND_ASSIGN(auto results, DynamicSliceFusion::ResolveResults(hero));
  EXPECT_THAT(results,
              ElementsAre(Result{
                  std::nullopt, 0, ShapeUtil::MakeShape(S32, {1}),
                  ShapeUtil::MakeShape(S32, {1}), std::nullopt, std::nullopt}));
}

TEST_F(FusionDynamicMemcpyRewriterTest, DoesNotRewriteCall) {
  constexpr char kHlo[] = R"(
    dynamic_slice {
      p0 = s32[4] parameter(0)
      c1 = s32[] constant(1)

      ROOT slice = s32[1] dynamic-slice(p0, c1), dynamic_slice_sizes={1}
    }

    ENTRY main {
      p0 = s32[4] parameter(0)
      ROOT call = s32[1] call(p0), to_apply=dynamic_slice
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(FusionDynamicMemcpyRewriter().Run(module.get()),
              absl_testing::IsOkAndHolds(false))
      << module->ToString();
}

TEST_F(FusionDynamicMemcpyRewriterTest, RewritesDusToV2Fusion) {
  constexpr char kHlo[] = R"(
    dynamic_slice {
      p0 = s32[4,8,8] parameter(0)
      p1 = s32[1,1,8] parameter(1)
      p2 = s32[] parameter(2)
      c1 = s32[] constant(1)

      ROOT update-slice = s32[4,8,8] dynamic-update-slice(p0, p1, p2, c1, c1),
          backend_config={"dynamic_slice_config":
              {"byte_offset":"32","byte_stride":"256","loop_index":"0"}}
    }

    body {
      p0 = (s32[], s32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[4,8,8] get-tuple-element(p0), index=1
      val = s32[1,1,8] constant({{{1,2,3,4,5,6,7,8}}})

      updated = s32[4,8,8] fusion(input, val, ivar), kind=kLoop, calls=dynamic_slice
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)

      ROOT result = (s32[], s32[4,8,8])
          tuple(next_ivar, updated)
    }

    condition {
      p0 = (s32[], s32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c6 = s32[] constant(6)
      ROOT cmp = pred[] compare(ivar, c6), direction=LT
    }

    ENTRY main {
      input = s32[4,8,8] parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[4,8,8]) tuple(c0, input)
      ROOT while = (s32[], s32[4,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"6"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(FusionDynamicMemcpyRewriter().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* fusion =
      module->GetComputationWithName("body")->GetInstructionWithName("updated");
  ASSERT_NE(fusion, nullptr);
  ASSERT_EQ(fusion->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kCustom);

  auto name = GetCustomFusionName(fusion);
  ASSERT_TRUE(name.has_value()) << module->ToString();
  EXPECT_EQ(*name, kDynamicSliceFusionConfigName);

  // Verify DynamicSliceFusion can parse the rewritten body.
  const HloComputation* body = fusion->fused_instructions_computation();
  const HloInstruction* hero = DynamicSliceFusion::FindHero(body);
  ASSERT_NE(hero, nullptr);
  EXPECT_EQ(hero->opcode(), HloOpcode::kCopy);

  ASSERT_OK_AND_ASSIGN(auto parameters,
                       DynamicSliceFusion::ResolveParameters(hero));
  EXPECT_THAT(parameters,
              ElementsAre(Parameter{1, ShapeUtil::MakeShape(S32, {1, 1, 8}),
                                    ShapeUtil::MakeShape(S32, {1, 1, 8}),
                                    std::nullopt, std::nullopt}));

  ASSERT_OK_AND_ASSIGN(auto results, DynamicSliceFusion::ResolveResults(hero));
  EXPECT_THAT(results,
              ElementsAre(Result{
                  0, 0, ShapeUtil::MakeShape(S32, {4, 8, 8}),
                  ShapeUtil::MakeShape(S32, {1, 1, 8}), MakeConfig(0, 32, 256),
                  Offsets{RtOff{2, 0}, CstOff{0, 1}, CstOff{0, 2}}}));
}

TEST_F(FusionDynamicMemcpyRewriterTest,
       DoesNotRewriteFusionWithoutDynamicSliceConfig) {
  constexpr char kHlo[] = R"(
    dynamic_slice {
      p0 = s32[4] parameter(0)
      c1 = s32[] constant(1)
      ROOT slice = s32[1] dynamic-slice(p0, c1), dynamic_slice_sizes={1}
    }

    ENTRY main {
      p0 = s32[4] parameter(0)
      ROOT fusion = s32[1] fusion(p0), kind=kLoop, calls=dynamic_slice
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(FusionDynamicMemcpyRewriter().Run(module.get()),
              absl_testing::IsOkAndHolds(false))
      << module->ToString();
}

}  // namespace
}  // namespace xla::gpu

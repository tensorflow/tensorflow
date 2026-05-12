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

#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;

using FusionDynamicMemcpyRewriterTest = HloHardwareIndependentTestBase;

std::optional<DynamicMemcpyConfig> GetMemcpyConfig(
    const HloInstruction* instr) {
  auto config = instr->backend_config<GpuBackendConfig>();
  if (!config.ok()) {
    return std::nullopt;
  }

  const auto& fusion_config = config->fusion_backend_config();
  if (fusion_config.kind() != kDynamicMemcpyFusionKind ||
      !fusion_config.has_dynamic_memcpy_config()) {
    return std::nullopt;
  }
  return fusion_config.dynamic_memcpy_config();
}

constexpr char kSliceMemcpyModule[] = R"(
    // This fusion is technically not a dynamic memcpy. Tests for that are in
    // the unit tests for DynamicMemcpyFusion::GetMemcpyDescriptorForFusion,
    // in copy_test.cc. Here, we just test that the logic triggers as expected.
    dynamic_slice {
      p0 = s32[4] parameter(0)
      c1 = s32[] constant(1)

      ROOT slice = s32[1] dynamic-slice(p0, c1), dynamic_slice_sizes={1}
    }

    ENTRY main {
      p0 = s32[4] parameter(0)
      ROOT fusion = s32[1] fusion(p0), kind=kLoop, calls=dynamic_slice
    })";

TEST_F(FusionDynamicMemcpyRewriterTest, AnnotatesMemcpyFusion) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kSliceMemcpyModule));
  EXPECT_THAT(FusionDynamicMemcpyRewriter().Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  auto config =
      GetMemcpyConfig(module->entry_computation()->root_instruction());
  ASSERT_TRUE(config.has_value()) << module->ToString();

  EXPECT_FALSE(config->depends_on_loop());
  EXPECT_THAT(config->src_offset_bytes(), ElementsAre(4));
  EXPECT_THAT(config->dst_offset_bytes(), ElementsAre(0));
}

constexpr char kSliceCallModule[] = R"(
    dynamic_slice {
      p0 = s32[4] parameter(0)
      c1 = s32[] constant(1)

      ROOT slice = s32[1] dynamic-slice(p0, c1), dynamic_slice_sizes={1}
    }

    ENTRY main {
      p0 = s32[4] parameter(0)
      ROOT call = s32[1] call(p0), to_apply=dynamic_slice
    })";

TEST_F(FusionDynamicMemcpyRewriterTest, DoesNotAnnotateCall) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kSliceCallModule));
  EXPECT_THAT(FusionDynamicMemcpyRewriter().Run(module.get()),
              absl_testing::IsOkAndHolds(false))
      << module->ToString();
  EXPECT_FALSE(GetMemcpyConfig(module->entry_computation()->root_instruction())
                   .has_value())
      << module->ToString();
}

constexpr char kLoopUpdateSliceMemcpyModule[] = R"(
    dynamic_slice {
      p0 = s32[4,8,8] parameter(0)
      p1 = s32[1,1,8] parameter(1)
      p2 = s32[] parameter(2)
      c1 = s32[] constant(1)

      ROOT update-slice = s32[4,8,8] dynamic-update-slice(p0, p1, p2, c1, c1)
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

TEST_F(FusionDynamicMemcpyRewriterTest,
       AnnotatesDusMemcpyFusionWithIterations) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kLoopUpdateSliceMemcpyModule));
  EXPECT_THAT(FusionDynamicMemcpyRewriter().Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  auto config = GetMemcpyConfig(
      module->GetComputationWithName("body")->GetInstructionWithName(
          "updated"));
  ASSERT_TRUE(config.has_value()) << module->ToString();

  EXPECT_TRUE(config->depends_on_loop());
  EXPECT_THAT(config->src_offset_bytes(), ElementsAre(0, 0, 0, 0, 0, 0));
  EXPECT_THAT(config->dst_offset_bytes(),
              ElementsAre(32, 32 + 256 * 1, 32 + 256 * 2, 32 + 256 * 3,
                          32 + 256 * 3, 32 + 256 * 3));
}

constexpr char kLoopDynamicVariableDifferentInitStep[] = R"(
    dynamic_slice_comp {
      p0 = s32[4,8,8] parameter(0)
      p1 = s32[] parameter(1)
      c0 = s32[] constant(0)
      ROOT slice = s32[1,8,8] dynamic-slice(p0, p1, c0, c0),
          dynamic_slice_sizes={1,8,8}
    }

    body {
      p0 = (s32[], s32[4,8,8], s32[]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[4,8,8] get-tuple-element(p0), index=1
      counter = s32[] get-tuple-element(p0), index=2

      sliced = s32[1,8,8] fusion(input, counter), kind=kLoop,
          calls=dynamic_slice_comp

      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      next_counter = s32[] add(counter, c1)

      ROOT result = (s32[], s32[4,8,8], s32[])
          tuple(next_ivar, input, next_counter)
    }

    condition {
      p0 = (s32[], s32[4,8,8], s32[]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c3 = s32[] constant(3)
      ROOT cmp = pred[] compare(ivar, c3), direction=LT
    }

    ENTRY main {
      input = s32[4,8,8] parameter(0)
      c2 = s32[] constant(2)
      c3 = s32[] constant(3)
      tuple = (s32[], s32[4,8,8], s32[]) tuple(c2, input, c3)
      ROOT while = (s32[], s32[4,8,8], s32[]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"3"},
                          "known_init_step":{"init":"2","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"},
                          "dynamic_variables":[{"tuple_index":"2","init":"3","step":"1"}]}
    })";

TEST_F(FusionDynamicMemcpyRewriterTest,
       DynamicVariableUsesPerVariableInitStep) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kLoopDynamicVariableDifferentInitStep));
  EXPECT_THAT(FusionDynamicMemcpyRewriter().Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  auto config = GetMemcpyConfig(
      module->GetComputationWithName("body")->GetInstructionWithName("sliced"));
  ASSERT_TRUE(config.has_value()) << module->ToString();

  EXPECT_TRUE(config->depends_on_loop());
  EXPECT_THAT(config->src_offset_bytes(), ElementsAre(768, 768, 768));
  EXPECT_THAT(config->dst_offset_bytes(), ElementsAre(0, 0, 0));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

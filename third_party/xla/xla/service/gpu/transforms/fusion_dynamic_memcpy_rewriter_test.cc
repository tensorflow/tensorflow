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

#include "xla/service/gpu/transforms/fusion_dynamic_memcpy_rewriter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;

using FusionDynamicMemcpyRewriterTest = HloHardwareIndependentTestBase;

bool IsMemcpyFusion(const HloInstruction* instr) {
  const auto& config = instr->backend_config<GpuBackendConfig>();
  return config.ok() &&
         config->fusion_backend_config().kind() == kDynamicMemcpyFusionKind;
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
              IsOkAndHolds(true));
  EXPECT_TRUE(IsMemcpyFusion(module->entry_computation()->root_instruction()))
      << module->ToString();
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
              IsOkAndHolds(false))
      << module->ToString();
  EXPECT_FALSE(IsMemcpyFusion(module->entry_computation()->root_instruction()))
      << module->ToString();
}

}  // namespace
}  // namespace gpu
}  // namespace xla

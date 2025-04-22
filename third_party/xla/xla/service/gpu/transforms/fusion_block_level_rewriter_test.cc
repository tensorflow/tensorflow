/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/fusion_block_level_rewriter.h"

#include <memory>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;

bool HasTritonBlockLevelFusionConfig(const HloInstruction* fusion) {
  return HloPredicateIsOp<HloOpcode::kFusion>(fusion) &&
         fusion->has_backend_config() &&
         fusion->backend_config<GpuBackendConfig>().ok() &&
         fusion->backend_config<GpuBackendConfig>()
             ->fusion_backend_config()
             .has_block_level_fusion_config() &&
         fusion->backend_config<GpuBackendConfig>()
                 ->fusion_backend_config()
                 .kind() == kTritonFusionKind;
}

class FusionBlockLevelRewriterTest : public HloHardwareIndependentTestBase {
 protected:
  se::DeviceDescription device_info_{TestGpuDeviceInfo::RTXA6000DeviceInfo(
      se::CudaComputeCapability::Ampere())};
};

bool RewriteEverythingPossible(const HloFusionInstruction* fusion) {
  return true;
}

TEST_F(FusionBlockLevelRewriterTest,
       DoesNotRewriteFusionThatIsAlreadyBlockLevel) {
  const absl::string_view hlo_text = R"(
fusion_computation {
  ROOT param_0 = f32[10,10] parameter(0)
}

ENTRY entry {
  param_0 = f32[10,10] parameter(0)
  ROOT fusion = f32[10,10] fusion(param_0), kind=kCustom,
    calls=fusion_computation,
    backend_config={"fusion_backend_config":
      {"kind":"__triton", "block_level_fusion_config":{}}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  EXPECT_THAT(
      FusionBlockLevelRewriter(device_info_, HloCostAnalysis::DefaultShapeSize,
                               RewriteEverythingPossible)
          .Run(module.get()),
      IsOkAndHolds(false));
}

TEST_F(FusionBlockLevelRewriterTest,
       RewritesFusionThatIsNotBlockLevelAndCanBeTiledAndCodegenedCorrectly) {
  const absl::string_view hlo_text = R"(
fusion_computation {
  ROOT param_0 = f32[10,10] parameter(0)
}

ENTRY entry {
  param_0 = f32[10,10] parameter(0)
  ROOT fusion = f32[10,10] fusion(param_0), kind=kLoop,
    calls=fusion_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  EXPECT_THAT(
      FusionBlockLevelRewriter(device_info_, HloCostAnalysis::DefaultShapeSize,
                               RewriteEverythingPossible)
          .Run(module.get()),
      IsOkAndHolds(true));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kCustom);
  EXPECT_TRUE(HasTritonBlockLevelFusionConfig(root));
}

TEST_F(FusionBlockLevelRewriterTest,
       DoesNotRewriteFusionThatIsNotBlockLevelAndCannotBeTiledCorrectly) {
  const absl::string_view hlo_text = R"(
fusion_computation {
  param_0 = f32[10,10] parameter(0)
  ROOT bitcast = f32[25,4] bitcast(param_0)
}

ENTRY entry {
  param_0 = f32[10,10] parameter(0)
  ROOT fusion = f32[25,4] fusion(param_0), kind=kLoop,
    calls=fusion_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  mlir::MLIRContext ctx;

  ASSERT_FALSE(std::holds_alternative<SymbolicTileAnalysis>(
      SymbolicTileAnalysis::AnalyzeComputation(
          *module->GetComputationWithName("fusion_computation"), &ctx)));
  EXPECT_THAT(
      FusionBlockLevelRewriter(device_info_, HloCostAnalysis::DefaultShapeSize,
                               RewriteEverythingPossible)
          .Run(module.get()),
      IsOkAndHolds(false));
}

TEST_F(FusionBlockLevelRewriterTest,
       DoesNotRewriteFusionThatIsNotBlockLevelAndCannotBeCodegenedCorrectly) {
  const absl::string_view hlo_text = R"(
fusion_computation {
  param_0 = f8e4m3fn[10,10] parameter(0)
  ROOT add = f8e4m3fn[10,10] add(param_0, param_0)
}

ENTRY entry {
  param_0 = f8e4m3fn[10,10] parameter(0)
  ROOT fusion = f8e4m3fn[10,10] fusion(param_0), kind=kLoop,
    calls=fusion_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_FALSE(IsTritonSupportedComputation(
      *module->GetComputationWithName("fusion_computation"),
      device_info_.gpu_compute_capability()));
  EXPECT_THAT(
      FusionBlockLevelRewriter(device_info_, HloCostAnalysis::DefaultShapeSize,
                               RewriteEverythingPossible)
          .Run(module.get()),
      IsOkAndHolds(false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

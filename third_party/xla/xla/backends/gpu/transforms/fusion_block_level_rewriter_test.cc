/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License
for the specific language governing permissions and limitations under the
License.
==============================================================================*/

#include "xla/backends/gpu/transforms/fusion_block_level_rewriter.h"

#include <memory>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

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
 public:
  FusionBlockLevelRewriterTest() {
    RegisterSymbolicExprStorage(&mlir_context_);
  }

 protected:
  se::DeviceDescription device_info_{TestGpuDeviceInfo::RTXA6000DeviceInfo(
      se::CudaComputeCapability::Ampere())};

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        HloHardwareIndependentTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_enable_fusion_block_level_rewriter(
        true);
    return debug_options;
  }
  mlir::MLIRContext mlir_context_;
};

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
                               &mlir_context_)
          .Run(module.get()),
      absl_testing::IsOkAndHolds(false));
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
                               &mlir_context_)
          .Run(module.get()),
      absl_testing::IsOkAndHolds(true));
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

  ASSERT_FALSE(std::holds_alternative<SymbolicTileAnalysis>(
      SymbolicTileAnalysis::AnalyzeComputation(
          *module->GetComputationWithName("fusion_computation"),
          &mlir_context_)));
  EXPECT_THAT(
      FusionBlockLevelRewriter(device_info_, HloCostAnalysis::DefaultShapeSize,
                               &mlir_context_)
          .Run(module.get()),
      absl_testing::IsOkAndHolds(false));
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
                               &mlir_context_)
          .Run(module.get()),
      absl_testing::IsOkAndHolds(false));
}

TEST_F(FusionBlockLevelRewriterTest, RewritesS32ReductionFusions) {
  constexpr absl::string_view kHloText = R"(

%scalar_add_computation {
  %scalar_lhs = s32[] parameter(0)
  %scalar_rhs = s32[] parameter(1)
  ROOT %add.1 = s32[] add(%scalar_lhs, %scalar_rhs)
}

%fused_reduce {
  %param.1 = s32[32,4096] parameter(1)
  %broadcast.0 = s32[32,4096,4096] broadcast(%param.1), dimensions={0,2}
  %param.0 = s32[4096,4096] parameter(0)
  %broadcast.1 = s32[32,4096,4096] broadcast(%param.0), dimensions={1,2}
  %multiply.2 = s32[32,4096,4096] multiply(%broadcast.0, %broadcast.1)
  %constant_2 = s32[] constant(0)
  ROOT %reduce.2 = s32[32,4096] reduce(%multiply.2, %constant_2), dimensions={2}, to_apply=%scalar_add_computation
}

ENTRY entry  {
  %param.0 = s32[32,4096] parameter(0)
  %param.1 = s32[4096,4096] parameter(1)
  ROOT %input_reduce_fusion = s32[32,4096]{1,0} fusion(%param.1, %param.0), kind=kInput, calls=%fused_reduce
}

)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  se::DeviceDescription device_info{TestGpuDeviceInfo::RTXA6000DeviceInfo(
      se::CudaComputeCapability::Ampere())};
  FusionBlockLevelRewriter rewriter(
      device_info, HloCostAnalysis::DefaultShapeSize, &mlir_context_);
  EXPECT_THAT(rewriter.Run(module.get()), absl_testing::IsOkAndHolds(true));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kCustom);
  EXPECT_TRUE(HasTritonBlockLevelFusionConfig(root));
}

TEST_F(FusionBlockLevelRewriterTest,
       RewritesLoopTransposeFusionWithSplitDimensions) {
  // This test checks if the rewriter can handle a transpose where dimensions
  // are split in the HLO but logically contiguous.
  // Logical shape: [100, 200, 300] -> [300, 200, 100] (Swap dim 0 and 2).
  // Physical shape: [100, 200, 10, 30] -> [10, 30, 200, 100].
  // The normalized logical transpose shape should recover the logical swap.
  const absl::string_view hlo_text = R"(
fusion_computation {
  p0 = f32[100,200,10,30] parameter(0)
  ROOT transpose = f32[10,30,200,100] transpose(p0), dimensions={2,3,1,0}
}

ENTRY entry {
  p0 = f32[100,200,10,30] parameter(0)
  ROOT fusion = f32[10,30,200,100] fusion(p0), kind=kLoop,
    calls=fusion_computation
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  EXPECT_THAT(
      FusionBlockLevelRewriter(device_info_, HloCostAnalysis::DefaultShapeSize,
                               &mlir_context_)
          .Run(module.get()),
      absl_testing::IsOkAndHolds(true));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kCustom);
  EXPECT_TRUE(HasTritonBlockLevelFusionConfig(root));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

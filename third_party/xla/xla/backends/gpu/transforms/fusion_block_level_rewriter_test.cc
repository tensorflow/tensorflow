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
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/codegen/tiling/tiling_space.h"
#include "xla/codegen/xtile/xtile_config.pb.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/mlir_context_pool.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

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

class FusionBlockLevelRewriterTest : public HloHardwareIndependentTestBase,
                                     public testing::WithParamInterface<bool> {
 public:
  FusionBlockLevelRewriterTest() {
    RegisterSymbolicExprStorage(&mlir_context_);
  }

  bool EnableSameShapeMultiOutputFusion() const { return GetParam(); }

 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        HloHardwareIndependentTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_enable_fusion_block_level_rewriter(
        true);
    debug_options
        .set_xla_gpu_experimental_enable_same_shape_multi_output_fusion(
            EnableSameShapeMultiOutputFusion());
    return debug_options;
  }

  se::DeviceDescription device_info_{TestGpuDeviceInfo::RTXA6000DeviceInfo(
      se::CudaComputeCapability::Ampere())};

  mlir::MLIRContext mlir_context_;
};

INSTANTIATE_TEST_SUITE_P(FusionBlockLevelRewriterTest,
                         FusionBlockLevelRewriterTest, testing::Bool(),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "SameShapeMultiOutputFusion"
                                             : "Default";
                         });

TEST_P(FusionBlockLevelRewriterTest,
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  EXPECT_THAT(
      FusionBlockLevelRewriter(device_info_, HloCostAnalysis::DefaultShapeSize,
                               &mlir_context_)
          .Run(module.get()),
      IsOkAndHolds(false));
}

TEST_P(FusionBlockLevelRewriterTest,
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  EXPECT_THAT(
      FusionBlockLevelRewriter(device_info_, HloCostAnalysis::DefaultShapeSize,
                               &mlir_context_)
          .Run(module.get()),
      IsOkAndHolds(true));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kCustom);
  EXPECT_TRUE(HasTritonBlockLevelFusionConfig(root));
}

TEST_P(FusionBlockLevelRewriterTest,
       DoesNotRewriteFusionThatIsNotBlockLevelAndCannotBeTiledCorrectly) {
  // TODO: b/502910372 - update the test when we support multi-output fusions.
  const absl::string_view hlo_text = R"hlo(
f {
  p0 = f32[10] parameter(0)
  p1 = f32[20] parameter(1)
  ROOT multi_output = (f32[10], f32[20]) tuple(p0, p1)
}

ENTRY entry {
  p0 = f32[10] parameter(0)
  p1 = f32[20] parameter(1)
  ROOT fusion = (f32[10], f32[20]) fusion(p0, p1), kind=kLoop, calls=f
})hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
      HloFusionAdaptor::ForInstruction(
          module->entry_computation()->root_instruction());
  EXPECT_THAT(TilingSpace::Create(*fusion_adaptor, &mlir_context_),
              StatusIs(absl::StatusCode::kUnimplemented,
                       testing::HasSubstr("multiple roots")));
  EXPECT_THAT(
      FusionBlockLevelRewriter(device_info_, HloCostAnalysis::DefaultShapeSize,
                               &mlir_context_)
          .Run(module.get()),
      IsOkAndHolds(false));
}

TEST_P(FusionBlockLevelRewriterTest,
       RewritesMultiOutputFusionWithIdenticalShapes) {
  const absl::string_view hlo_text = R"hlo(
f {
  p0 = f32[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  add = f32[10,10] add(p0, p1)
  sub = f32[10,10] subtract(p0, p1)
  ROOT multi_output = (f32[10,10], f32[10,10]) tuple(add, sub)
}

ENTRY entry {
  p0 = f32[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  ROOT fusion = (f32[10,10], f32[10,10]) fusion(p0, p1), kind=kLoop, calls=f
})hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  absl::StatusOr<bool> result =
      FusionBlockLevelRewriter(device_info_, HloCostAnalysis::DefaultShapeSize,
                               &mlir_context_)
          .Run(module.get());

  if (EnableSameShapeMultiOutputFusion()) {
    EXPECT_THAT(result, IsOkAndHolds(true));
    const HloInstruction* root =
        module->entry_computation()->root_instruction();
    EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
    EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kCustom);
    EXPECT_TRUE(HasTritonBlockLevelFusionConfig(root));
  } else {
    EXPECT_THAT(result, IsOkAndHolds(false));
  }
}

TEST_P(FusionBlockLevelRewriterTest,
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_FALSE(IsTritonSupportedComputation(
      *module->GetComputationWithName("fusion_computation"),
      device_info_.gpu_compute_capability()));
  EXPECT_THAT(
      FusionBlockLevelRewriter(device_info_, HloCostAnalysis::DefaultShapeSize,
                               &mlir_context_)
          .Run(module.get()),
      IsOkAndHolds(false));
}

TEST_P(FusionBlockLevelRewriterTest, RewritesScanFusions) {
  constexpr absl::string_view kHloText = R"(
combiner {
  in = f32[] parameter(0)
  carry = f32[] parameter(1)
  add = f32[] add(carry, in)
  ROOT t = (f32[], f32[]) tuple(add, add)
}

fused_scan {
  input = f32[100]{0} parameter(0)
  init = f32[] parameter(1)
  scan = (f32[100]{0}, f32[]) scan(input, init), dimensions={0}, num_carries=1, to_apply=combiner, is_associative=true
  ROOT gte0 = f32[100]{0} get-tuple-element(scan), index=0
}

ENTRY entry {
  input = f32[100]{0} parameter(0)
  init = f32[] constant(0)
  ROOT fusion = f32[100]{0} fusion(input, init), kind=kCustom, calls=fused_scan,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));

  EXPECT_THAT(
      FusionBlockLevelRewriter(device_info_, HloCostAnalysis::DefaultShapeSize,
                               &mlir_context_)
          .Run(module.get()),
      absl_testing::IsOkAndHolds(true));

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(HasTritonBlockLevelFusionConfig(root));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_THAT(backend_config.fusion_backend_config()
                  .block_level_fusion_config()
                  .sequential_tile_sizes(),
              testing::SizeIs(1));
}

TEST_P(FusionBlockLevelRewriterTest, RewritesFusionWithParallelTilingSearch) {
  const absl::string_view hlo_text = R"(
fusion_computation {
  param_0 = f32[128,128] parameter(0)
  ROOT exp = f32[128,128] exponential(param_0)
}

ENTRY entry {
  param_0 = f32[128,128] parameter(0)
  ROOT fusion = f32[128,128] fusion(param_0), kind=kCustom,
    calls=fusion_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_pool", 4);
  // Same contexts as GpuCompiler pools. They are single-threaded, so the cost
  // model must give each candidate its own context.
  MlirContextPool mlir_context_pool(CreateMlirContext, /*preallocate=*/4);
  EXPECT_THAT(
      FusionBlockLevelRewriter(device_info_, HloCostAnalysis::DefaultShapeSize,
                               &mlir_context_, &thread_pool, &mlir_context_pool)
          .Run(module.get()),
      absl_testing::IsOkAndHolds(true));
}

}  // namespace
}  // namespace xla::gpu

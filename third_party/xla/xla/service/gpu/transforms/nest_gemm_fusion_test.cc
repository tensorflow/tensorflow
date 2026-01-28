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

#include "xla/service/gpu/transforms/nest_gemm_fusion.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;

namespace xla {

namespace gpu {
namespace {

// Wraps a matcher for a fusion instruction's output tile sizes.
// Proto matchers would be nice, but b/229726259 is P2.
MATCHER_P(OutputTileSizesIs, matcher, "") {
  auto backend_config = arg.template backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    *result_listener << "failed to get backend config: "
                     << backend_config.status();
    return false;
  }
  FusionBackendConfig fusion_backend_config =
      backend_config->fusion_backend_config();
  if (!fusion_backend_config.has_block_level_fusion_config()) {
    *result_listener << "has no block level fusion config";
    return false;
  }
  if (fusion_backend_config.kind() != "__triton_nested_gemm_fusion") {
    *result_listener << "fusion kind is not __triton_nested_gemm_fusion";
    return false;
  }
  auto output_tile_sizes =
      fusion_backend_config.block_level_fusion_config().output_tiles(0).sizes();
  return ExplainMatchResult(matcher, output_tile_sizes, result_listener);
}

class NestGemmFusionTest : public HloHardwareIndependentTestBase {
 protected:
  NestGemmFusionTest() { RegisterSymbolicExprStorage(&mlir_context_); }
  const se::DeviceDescription device_description_{
      TestGpuDeviceInfo::RTXA6000DeviceInfo(
          se::GpuComputeCapability{se::CudaComputeCapability::Ampere()})};
  mlir::MLIRContext mlir_context_;

  std::unique_ptr<VerifiedHloModule> RunNestGemmFusion(
      absl::string_view hlo, const bool expect_change = true) {
    std::unique_ptr<VerifiedHloModule> module =
        ParseAndReturnVerifiedModule(hlo).value();
    EXPECT_THAT(
        NestGemmFusion(device_description_, &mlir_context_).Run(module.get()),
        IsOkAndHolds(expect_change));
    EXPECT_OK(verifier().Run(module.get()).status());
    return module;
  }
};

TEST_F(NestGemmFusionTest, BasicTest) {
  absl::string_view hlo = R"(
dot {
  lhs = f32[8192,512] parameter(0)
  rhs = f32[512,512] parameter(1)
  ROOT  dot = f32[8192,512] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = f32[8192,512] parameter(0)
  p1 = f32[512,512] parameter(1)
  ROOT fusion = f32[8192,512] fusion(p0, p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_gemm",  "triton_gemm_config": {
          "block_m":"64", "block_n":"256", "block_k":"32",
          "split_k":"1", "num_stages":"5", "num_warps":"4", "num_ctas":"3"
        }
      }
    }
})";

  std::unique_ptr<VerifiedHloModule> module = RunNestGemmFusion(hlo);
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(match::Fusion(&fusion)));
  EXPECT_THAT(*fusion, OutputTileSizesIs(ElementsAre(64, 256)));

  BlockLevelFusionConfig block_level_fusion_config =
      fusion->backend_config<GpuBackendConfig>()
          ->fusion_backend_config()
          .block_level_fusion_config();
  EXPECT_THAT(block_level_fusion_config.output_tiles(0).sizes(),
              ElementsAre(64, 256));
  EXPECT_THAT(block_level_fusion_config.num_warps(), 4);
  EXPECT_THAT(block_level_fusion_config.num_ctas(), 3);
  EXPECT_THAT(block_level_fusion_config.num_stages(), 5);

  const HloInstruction* lhs = nullptr;
  const HloInstruction* rhs = nullptr;
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(match::Dot(match::Fusion(&lhs), match::Fusion(&rhs))));
  EXPECT_THAT(*lhs, OutputTileSizesIs(ElementsAre(64, 32)));
  EXPECT_THAT(*rhs, OutputTileSizesIs(ElementsAre(32, 256)));

  // The old GEMM config should have been deleted.
  EXPECT_FALSE(fusion->backend_config<GpuBackendConfig>()
                   ->fusion_backend_config()
                   .has_triton_gemm_config());
}

TEST_F(NestGemmFusionTest, ConcatenationsAreHoistedWithinNestedGemmFusions) {
  absl::string_view hlo = R"(
HloModule t

triton_gemm {
  parameter_0 = f32[2,3,10]{2,1,0} parameter(0)
  parameter_1 = f32[2,10,128]{2,1,0} parameter(1)
  parameter_2 = f32[2,10,256]{2,1,0} parameter(2)
  concatenate = f32[2,10,384]{2,1,0} concatenate(parameter_1, parameter_2), dimensions={2}
  ROOT dot = f32[2,3,384]{2,1,0} dot(parameter_0, concatenate),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
}

ENTRY e {
  parameter_0 = f32[2,3,10]{2,1,0} parameter(0)
  parameter_1 = f32[2,10,128]{2,1,0} parameter(1)
  parameter_2 = f32[2,10,256]{2,1,0} parameter(2)
  ROOT dot = f32[2,3,384]{2,1,0} fusion(parameter_0, parameter_1, parameter_2),
    kind=kCustom, calls=triton_gemm,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    triton_gemm_config: {"block_m":16,"block_n":64,"block_k":32,
                         "split_k":1,"num_stages":1,"num_warps":2,
                         "num_ctas":1}}}
})";
  std::unique_ptr<VerifiedHloModule> module = RunNestGemmFusion(hlo);
  HloComputation* fusion_computation = module->entry_computation()
                                           ->root_instruction()
                                           ->fused_instructions_computation();
  HloInstruction* dot_lhs;
  HloInstruction* dot_rhs;
  EXPECT_THAT(
      fusion_computation->root_instruction(),
      GmockMatch(match::Dot(match::Fusion(&dot_lhs), match::Fusion(&dot_rhs))));
  EXPECT_THAT(dot_rhs->fused_instructions_computation()->root_instruction(),
              GmockMatch(match::Concatenate(match::Fusion(), match::Fusion())));
}

TEST_F(NestGemmFusionTest, CreatesTwoNestedFusionsFromSameParameter) {
  absl::string_view hlo = R"(
dot {
  p0 = f32[32] parameter(0)
  lhs = f32[4,8] reshape(p0)
  rhs = f32[8,4] reshape(p0)
  ROOT dot = f32[4,4] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = f32[32] parameter(0)
  ROOT fusion = f32[4,4] fusion(p0),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_gemm",  "triton_gemm_config": {
          "block_m":"4", "block_n":"4", "block_k":"8",
          "split_k":"1", "num_stages":"1", "num_warps":"1", "num_ctas":"1"
        }
      }
    }
}
)";
  std::unique_ptr<VerifiedHloModule> module = RunNestGemmFusion(hlo);
  EXPECT_THAT(
      RunFileCheck(module->ToString(HloPrintOptions::ShortParsable()), R"(
CHECK: {{.*}} {
CHECK-NEXT: {{.*}} f32[32]{0} parameter(0)
CHECK-NEXT: ROOT {{.*}} reshape
CHECK-NEXT: }
CHECK: {{.*}} {
CHECK-NEXT: {{.*}} f32[32]{0} parameter(0)
CHECK-NEXT: ROOT {{.*}} reshape
)"),
      IsOkAndHolds(true));
}

// TODO(b/393299275): update test to use a unsupported operation.
TEST_F(NestGemmFusionTest, DISABLED_UnsupportedComputationsAreNotChanged) {
  // Fusions other than kTritonNestedGemmFusionKind are not supported.
  // In this case pass should only change the supported fusions.
  absl::string_view hlo = R"(
identity {
  ROOT result = f32[128,128]{1,0} parameter(0)
}

unsupported_fusion {
  p0 = f32[128,128]{1,0} parameter(0)
  // This fusion is not supported by nest_gemm_fusion pass.
  cp0 = f32[128,128]{1,0} fusion(p0), kind=kCustom, calls=identity
  p1 = f32[128,128]{1,0} parameter(1)
  ROOT result = f32[128,128]{1,0} dot(cp0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

supported_fusion {
  lhs = f32[8192,512] parameter(0)
  rhs = f32[512,512] parameter(1)
  ROOT dot = f32[8192,512] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[128,128]{1,0} parameter(0)
  p1 = f32[128,128]{1,0} parameter(1)
  r1 = f32[128,128] fusion(p0, p1), kind=kCustom, calls=unsupported_fusion,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
    "triton_gemm_config": {
      "block_m":32,"block_n":16,"block_k":128,
      "split_k":1,"num_stages":1,"num_warps":4, "num_ctas":1}}}
  p2 = f32[8192,512] parameter(2)
  p3 = f32[512,512] parameter(3)
  r2 = f32[8192,512] fusion(p2, p3), kind=kCustom, calls=supported_fusion,
    backend_config={
      "fusion_backend_config": {
        "kind":"__triton_gemm",  "triton_gemm_config": {
          "block_m":"64", "block_n":"256", "block_k":"32",
          "split_k":"1", "num_stages":"5", "num_warps":"4", "num_ctas":"3"
        }
      }
    }
  ROOT result = (f32[128,128], f32[8192,512]) tuple(r1, r2)
}
)";
  std::unique_ptr<VerifiedHloModule> module = RunNestGemmFusion(hlo);
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->operand(0)
                ->backend_config<GpuBackendConfig>()
                ->fusion_backend_config()
                .kind(),
            "__triton_gemm");
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->operand(1)
                ->backend_config<GpuBackendConfig>()
                ->fusion_backend_config()
                .kind(),
            "__triton_nested_gemm_fusion");
}

}  // namespace
}  // namespace gpu
}  // namespace xla

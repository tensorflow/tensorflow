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

#include "xla/service/gpu/transforms/convert_triton_gemm_config.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;

namespace xla::gpu {
namespace {

// Wraps a matcher for a fusion instruction's output tile sizes.
// Proto matchers would be nice, but b/229726259 is P2.
MATCHER_P(ContractionTileSizesIs, matcher, "") {
  auto backend_config = arg.template backend_config<Tile>();
  if (!backend_config.ok()) {
    *result_listener << "failed to get tile sizes: " << backend_config.status();
    return false;
  }
  return ExplainMatchResult(matcher, backend_config->sizes(), result_listener);
}

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

class ConvertTritonGemmConfigTest : public HloHardwareIndependentTestBase {
 protected:
  ConvertTritonGemmConfigTest() { RegisterSymbolicExprStorage(&mlir_context_); }
  const se::DeviceDescription device_description_{
      TestGpuDeviceInfo::RTXA6000DeviceInfo(
          se::GpuComputeCapability{se::CudaComputeCapability::Ampere()})};
  mlir::MLIRContext mlir_context_;

  std::unique_ptr<VerifiedHloModule> RunConvertTritonGemmConfig(
      absl::string_view hlo, const bool expect_change = true) {
    std::unique_ptr<VerifiedHloModule> module =
        ParseAndReturnVerifiedModule(hlo).value();
    EXPECT_THAT(ConvertTritonGemmConfig(device_description_, &mlir_context_)
                    .Run(module.get()),
                IsOkAndHolds(expect_change));
    EXPECT_OK(verifier().Run(module.get()).status());
    return module;
  }
};

TEST_F(ConvertTritonGemmConfigTest, BasicTest) {
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

  std::unique_ptr<VerifiedHloModule> module = RunConvertTritonGemmConfig(hlo);
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

  EXPECT_THAT(*fusion->fused_expression_root(),
              ContractionTileSizesIs(ElementsAre(32)));

  // The old GEMM config should have been deleted.
  EXPECT_FALSE(fusion->backend_config<GpuBackendConfig>()
                   ->fusion_backend_config()
                   .has_triton_gemm_config());
}

}  // namespace
}  // namespace xla::gpu

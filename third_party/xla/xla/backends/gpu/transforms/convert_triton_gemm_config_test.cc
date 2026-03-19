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

#include "xla/backends/gpu/transforms/convert_triton_gemm_config.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
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

namespace xla::gpu {
namespace {

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
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    CHECK: ROOT {{.*}} = f32[8192,512]{1,0} dot(
    CHECK-SAME: backend_config={"sizes":["32"]}
    CHECK: ENTRY
    CHECK: ROOT{{.*}}fusion(
    CHECK-SAME: kind=kCustom
    CHECK-SAME: "kind":"__triton_nested_gemm_fusion"
    CHECK-SAME: "block_level_fusion_config"
    CHECK-SAME: "num_warps":"4"
    CHECK-SAME: "output_tiles":[{"sizes":["64","256"]}]
    CHECK-SAME: "num_ctas":3
    CHECK-SAME: "num_stages":5
)"));
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(match::Fusion(&fusion)));
  // The old GEMM config should have been deleted.
  EXPECT_FALSE(fusion->backend_config<GpuBackendConfig>()
                   ->fusion_backend_config()
                   .has_triton_gemm_config());
}

TEST_F(ConvertTritonGemmConfigTest, ScaledDot) {
  absl::string_view hlo = R"(
scaled_dot {
  lhs = bf16[4,4] parameter(0)
  rhs = bf16[4,4] parameter(1)
  lhs_scale = bf16[1,1] parameter(2)
  rhs_scale = bf16[1,1] parameter(3)
  ROOT dot = bf16[4,4] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY entry {
  p0 = bf16[4,4] parameter(0)
  p1 = bf16[4,4] parameter(1)
  p2 = bf16[1,1] parameter(2)
  p3 = bf16[1,1] parameter(3)
  ROOT fusion = bf16[4,4] fusion(p0, p1, p2, p3),
    kind=kCustom, calls=scaled_dot, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_gemm",
        "triton_gemm_config": {
          "block_m":"16", "block_n":"32", "block_k":"64",
          "split_k":"1", "num_stages":"1", "num_warps":"4", "num_ctas":"1"
        }
      }
    }
})";

  std::unique_ptr<VerifiedHloModule> module =
      ParseAndReturnVerifiedModule(hlo).value();
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_scaled_dot_with_triton(true);
  EXPECT_THAT(ConvertTritonGemmConfig(device_description_, &mlir_context_)
                  .Run(module.get()),
              IsOkAndHolds(true));
  EXPECT_OK(verifier().Run(module.get()).status());
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    CHECK: ROOT {{.*}} = bf16[4,4]{1,0} scaled-dot({{.*}}backend_config={"sizes":["64"]}
    CHECK: ENTRY
    CHECK: ROOT{{.*}}fusion(
    CHECK-SAME: "kind":"__triton_nested_gemm_fusion"
    CHECK-SAME: "output_tiles":[{"sizes":["16","32"]}]
)"));
}

}  // namespace
}  // namespace xla::gpu

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

#include "xla/backends/gpu/autotuner/custom_kernel.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

using CustomKernelBackendConfig = AutotuneResult::CustomKernelFusionKey;

using ::tsl::proto_testing::EqualsProto;
using tsl::testing::IsOk;
using tsl::testing::IsOkAndHolds;
using tsl::testing::StatusIs;

const char kCustomKernelFusionHlo[] = R"(
HloModule extracted

cutlass_gemm {
  p0 = f32[15,19]{1,0} parameter(0)
  p1 = f32[19,17]{1,0} parameter(1)
  ROOT r = f32[15, 17]{1,0} dot(p0, p1), lhs_contracting_dims={1},
  rhs_contracting_dims={0}
}

ENTRY region_198.14436 {
  p.0 = f32[15,19]{1,0} parameter(0)
  p.1 = f32[19,17]{1,0} parameter(1)
  ROOT cutlass_gemm = f32[15,17]{1,0} fusion(p.0, p.1), kind=kCustom,
  calls=cutlass_gemm,
  backend_config={
    "fusion_backend_config":{
      "kind":"__custom_fusion",
      "custom_fusion_config":{
        "name":"cutlass_gemm",
        "kernel_index":-1
      }
    }
  }
})";

const char kTritonFusionHlo[] = R"(
HloModule module

computation {
  p0 = bf16[1024,1024]{1,0} parameter(0)
  convert0 = f32[1024,1024]{1,0} convert(p0)
  p1 = bf16[1024,1024]{1,0} parameter(1)
  convert1 = f32[1024,1024]{1,0} convert(p1)
  ROOT dot = f32[1024,1024]{1,0} dot(convert0, convert1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  p0 = bf16[1024,1024]{1,0} parameter(0)
  p1 = bf16[1024,1024]{1,0} parameter(1)
  ROOT fusion = f32[1024,1024]{1,0} fusion(p0, p1),
    kind=kCustom, calls=computation,
    backend_config={
      "fusion_backend_config": {
        "kind":"__triton_gemm"
      }
    }
})";

class CustomKernelBackendTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions debug_options_;
  NVPTXCompiler compiler_;
  CustomKernelBackend backend_;

  CustomKernelBackendTest()
      : backend_(PlatformUtil::GetDefaultPlatform()
                     .value()
                     ->ExecutorForDevice(0)
                     .value(),
                 &debug_options_, &compiler_) {}

  CustomKernelBackendConfig ExpectedDefaultAlgorithm() {
    auto config = AutotuneResult::CustomKernelFusionKey();
    config.set_kernel_index(0);
    return config;
  }
};

TEST_F(CustomKernelBackendTest, CanCreateCublasBackend) {
  ASSERT_NE(nullptr, &backend_);
}

TEST_F(CustomKernelBackendTest, GetSupportedConfigsFromCustomKernelFusion) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCustomKernelFusionHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsOkAndHolds(testing::SizeIs(testing::Gt(0))));
}

TEST_F(CustomKernelBackendTest,
       GetSupportedConfigsReturnsEmptyVectorForNonCustomKernelFusion) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTritonFusionHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsOkAndHolds(testing::SizeIs(0)));
}

TEST_F(CustomKernelBackendTest, ReturnsDefaultConfig) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kCustomKernelFusionHlo));

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(config, IsOk());
  EXPECT_THAT(static_cast<const CustomKernelBackendConfig&>(*config.value()),
              EqualsProto(ExpectedDefaultAlgorithm()));
}

TEST_F(CustomKernelBackendTest,
       FailsReturningDefaultConfigForNonCustomFusionInstruction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTritonFusionHlo));

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(config, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(CustomKernelBackendTest, ApplyConfig) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kCustomKernelFusionHlo));
  CustomKernelBackendConfig config;
  config.set_kernel_index(2);
  TF_EXPECT_OK(backend_.ApplyConfig(
      *hlo_module->entry_computation()->root_instruction(), config));
  EXPECT_THAT(RunFileCheck(hlo_module->ToString(), "CHECK: \"kernel_index\":2"),
              IsOkAndHolds(true));
}

}  // namespace gpu
}  // namespace xla

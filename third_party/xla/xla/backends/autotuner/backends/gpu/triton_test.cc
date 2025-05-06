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

#include "xla/backends/autotuner/backends/gpu/triton.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::testing::IsOk;
using TritonBackendConfig = AutotuneResult::TritonGemmKey;

const char kHlo[] = R"(
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
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  })";

class TritonBackendTest : public HloHardwareIndependentTestBase {
 protected:
  TritonBackendTest()
      : target_config_([]() {
          se::GpuTargetConfigProto target_config_proto;
          *target_config_proto.mutable_gpu_device_info() =
              TestGpuDeviceInfo().CudaOrRocmDeviceInfo().ToGpuProto();
          return Compiler::TargetConfig(target_config_proto);
        }()),
        backend_(&target_config_, &debug_options_, &compiler_) {}

  DebugOptions debug_options_;
  NVPTXCompiler compiler_;
  Compiler::TargetConfig target_config_;
  TritonBackend backend_;
};

TEST_F(TritonBackendTest, GetSupportedConfigs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));

  std::vector<std::unique_ptr<BackendConfig>> configs =
      backend_.GetSupportedConfigs(
          *(module->entry_computation()->root_instruction()), nullptr);
  EXPECT_GT(configs.size(), 0);
}

TEST_F(TritonBackendTest, GetSupportedConfigsForUnsupportedInstruction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* unsupported_instr = module->entry_computation()
                                          ->root_instruction()
                                          ->called_computations()[0]
                                          ->root_instruction();
  std::vector<std::unique_ptr<BackendConfig>> configs =
      backend_.GetSupportedConfigs(*unsupported_instr, nullptr);
  EXPECT_THAT(configs, ::testing::IsEmpty());
}

TEST_F(TritonBackendTest, GetDefaultConfig) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  TritonBackendConfig expected_config =
      TritonGemmConfig(64, 64, 64, 1, 1, 2, 1).ToProto();

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction()));

  EXPECT_THAT(config, IsOk());
  const TritonBackendConfig& actual_config =
      static_cast<const TritonBackendConfig&>(*config.value());
  EXPECT_THAT(actual_config, EqualsProto(expected_config));
}

TEST_F(TritonBackendTest, GetDefaultConfigForUnsupportedInstruction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* unsupported_instr = module->entry_computation()
                                          ->root_instruction()
                                          ->called_computations()[0]
                                          ->root_instruction();
  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(*unsupported_instr);
  EXPECT_THAT(config.status(),
              ::tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TritonBackendTest, Compile) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction())));
  absl::StatusOr<std::unique_ptr<Executable>> executable = backend_.Compile(
      *(module->entry_computation()->root_instruction()), *config);
  EXPECT_THAT(executable, IsOk());
}

}  // namespace
}  // namespace gpu
}  // namespace xla

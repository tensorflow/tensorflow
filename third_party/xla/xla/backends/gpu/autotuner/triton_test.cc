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

#include "xla/backends/gpu/autotuner/triton.h"

#include <algorithm>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

using absl_testing::IsOk;
using absl_testing::StatusIs;
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

const char kScaledDotHlo[] = R"(
HloModule ScaledDotIsFused, entry_computation_layout={(bf16[4,4]{1,0}, bf16[4,4]{1,0}, bf16[1,1]{1,0}, bf16[1,1]{1,0})->bf16[4,4]{1,0}}

%fusion_dot (parameter_0: bf16[4,4], parameter_1: bf16[4,4], parameter_2: bf16[1,1], parameter_3: bf16[1,1]) -> bf16[4,4] {
  %parameter_0 = bf16[4,4]{1,0} parameter(0)
  %parameter_1 = bf16[4,4]{1,0} parameter(1)
  %parameter_2 = bf16[1,1]{1,0} parameter(2)
  %parameter_3 = bf16[1,1]{1,0} parameter(3)
  ROOT %dot.1 = bf16[4,4]{1,0} scaled-dot(%parameter_0, %parameter_1, %parameter_2, %parameter_3), lhs_contracting_dims={1}, rhs_contracting_dims={1}, metadata={op_name="foo"}
}

ENTRY %entry (lhs: bf16[4,4], rhs: bf16[4,4], lhs_scale: bf16[1,1], rhs_scale: bf16[1,1]) -> bf16[4,4] {
  %lhs = bf16[4,4]{1,0} parameter(0)
  %rhs = bf16[4,4]{1,0} parameter(1)
  %lhs_scale = bf16[1,1]{1,0} parameter(2)
  %rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT %fusion = bf16[4,4]{1,0} fusion(%lhs, %rhs, %lhs_scale, %rhs_scale), kind=kCustom, calls=%fusion_dot, metadata={op_name="foo"}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm"},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
})";

class TritonBackendTest : public HloHardwareIndependentTestBase {
 protected:
  TritonBackendTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()),
        target_config_(stream_executor_),
        backend_(&debug_options_, &compiler_, &target_config_, &mlir_context_) {
    RegisterSymbolicExprStorage(&mlir_context_);
  }

  DebugOptions debug_options_;
  NVPTXCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  TritonBackend backend_;
  mlir::MLIRContext mlir_context_;
};

TEST_F(TritonBackendTest, GetSupportedConfigs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          *(module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOk());
  EXPECT_GT(configs.value().size(), 0);

  if (backend_.target_config()
          .device_description.cuda_compute_capability()
          .IsAtLeastHopper()) {
    // Check that TMA configurations are generated.
    EXPECT_TRUE(std::any_of(configs.value().begin(), configs.value().end(),
                            [](auto& config) {
                              TritonBackendConfig actual_config;
                              if (!config->UnpackTo(&actual_config)) {
                                return false;
                              }
                              return actual_config.is_tma_allowed();
                            }));
  }
}

TEST_F(TritonBackendTest, GetSupportedConfigsForScaledDot) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kScaledDotHlo));
  HloInstruction* fusion_instr =
      module->entry_computation()->root_instruction();
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(*fusion_instr);
  EXPECT_THAT(configs, absl_testing::IsOk());
  EXPECT_GT(configs.value().size(), 0);
}

TEST_F(TritonBackendTest, GetAndApplyConfigForScaledDot) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kScaledDotHlo));
  HloInstruction* fusion_instr =
      module->entry_computation()->root_instruction();
  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(*fusion_instr);
  EXPECT_THAT(config, absl_testing::IsOk());
  EXPECT_THAT(backend_.ApplyConfig(*fusion_instr, *config.value()), IsOk());
}

TEST_F(TritonBackendTest, GetSupportedConfigsRestrictedDefaultSearch) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> default_configs =
      backend_.GetSupportedConfigs(
          *(module->entry_computation()->root_instruction()));
  debug_options_.set_xla_gpu_exhaustive_tiling_search(true);
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
      exhaustive_configs = backend_.GetSupportedConfigs(
          *(module->entry_computation()->root_instruction()));
  EXPECT_THAT(default_configs, IsOk());
  EXPECT_THAT(exhaustive_configs, IsOk());
  EXPECT_GE(exhaustive_configs.value().size(), default_configs.value().size());
}

TEST_F(TritonBackendTest, GetSupportedConfigsForUnsupportedInstruction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* unsupported_instr = module->entry_computation()
                                          ->root_instruction()
                                          ->called_computations()[0]
                                          ->root_instruction();
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(*unsupported_instr);
  EXPECT_THAT(configs, absl_testing::IsOk());
  EXPECT_THAT(configs.value(), testing::IsEmpty());
}

TEST_F(TritonBackendTest, GetDefaultConfig) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction()));

  EXPECT_THAT(config, absl_testing::IsOk());
}

TEST_F(TritonBackendTest, GetDefaultConfigReturnsSplitKOne) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  debug_options_.set_xla_gpu_enable_split_k_autotuning(true);

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction()));

  ASSERT_THAT(config, absl_testing::IsOk());
  TritonBackendConfig triton_config;
  ASSERT_TRUE(config.value()->UnpackTo(&triton_config));
  EXPECT_EQ(triton_config.split_k(), 1);
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
  EXPECT_THAT(config.status(), StatusIs(absl::StatusCode::kInvalidArgument));
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
  EXPECT_THAT(executable, absl_testing::IsOk());
}

}  // namespace
}  // namespace gpu
}  // namespace xla

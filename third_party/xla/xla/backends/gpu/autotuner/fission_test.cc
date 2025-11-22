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

#include "xla/backends/gpu/autotuner/fission.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::SizeIs;

const char kTritonFusionHlo[] = R"(
  HloModule module

  computation {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    convert0 = f32[1024,1024]{1,0} convert(p0)
    p1 = s8[1024,1024]{1,0} parameter(1)
    convert1 = f32[1024,1024]{1,0} convert(p1)
    ROOT dot = f32[1024,1024]{1,0} dot(convert0, convert1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    p1 = s8[1024,1024]{1,0} parameter(1)
    ROOT fusion = f32[1024,1024]{1,0} fusion(p0, p1),
      kind=kCustom, calls=computation,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  })";

class FissionBackendTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions debug_options_;
  NVPTXCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  FissionBackend backend_;
  mlir::MLIRContext mlir_context_;

  FissionBackendTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()),
        target_config_(stream_executor_),
        backend_(stream_executor_, &debug_options_, &compiler_, &target_config_,
                 &mlir_context_) {}
};

TEST_F(FissionBackendTest, CanCreateCublasBackend) {
  ASSERT_NE(nullptr, &backend_);
}

TEST_F(FissionBackendTest, GetSupportedConfigsFromCublasCustomCall) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTritonFusionHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOkAndHolds(SizeIs(testing::Ge(2))));
  // The first config is the cublas config.
  AutotuneResult::GemmKey cublas_config;
  EXPECT_TRUE(configs.value().front()->UnpackTo(&cublas_config));
  EXPECT_EQ(cublas_config.algorithm(), -1);
  // The last config is the custom kernel config.
  AutotuneResult::CustomKernelFusionKey custom_kernel_config;
  EXPECT_TRUE(configs.value().back()->UnpackTo(&custom_kernel_config));
  EXPECT_EQ(custom_kernel_config.kernel_index(), 0);
}

TEST_F(FissionBackendTest, GetSupportedConfigsForUnsupportedInstructionFails) {
  std::string hlo = R"(
    HloModule module

    ENTRY main {
      p0 = f32[1024,1024]{1,0} parameter(0)
      p1 = f32[1024,1024]{1,0} parameter(1)
      ROOT dot = f32[1024,1024]{1,0} dot(p0, p1),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_.GetSupportedConfigs(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(FissionBackendTest, GetDefaultConfigFails) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTritonFusionHlo));

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_.GetDefaultConfig(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(config.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(FissionBackendTest, ApplyCublasConfigToFusionInstruction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kTritonFusionHlo));
  hlo_module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_cublaslt(false);
  AutotuneResult::GemmKey config;
  config.set_algorithm(3);
  google::protobuf::Any any;
  any.PackFrom(config);
  TF_EXPECT_OK(backend_.ApplyConfig(
      *hlo_module->entry_computation()->root_instruction(), any));
  EXPECT_THAT(RunFileCheck(hlo_module->ToString(),
                           "CHECK: \"__cublas$gemm\"\n"
                           "CHECK: \"selected_algorithm\":\"3\""),
              IsOkAndHolds(true));
}

TEST_F(FissionBackendTest, ApplyCublasLtConfigToFusionInstruction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kTritonFusionHlo));
  hlo_module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_cublaslt(true);
  AutotuneResult::GemmKey config;
  config.set_algorithm(3);
  google::protobuf::Any any;
  any.PackFrom(config);
  TF_EXPECT_OK(backend_.ApplyConfig(
      *hlo_module->entry_computation()->root_instruction(), any));
  EXPECT_THAT(
      RunFileCheck(hlo_module->ToString(), "CHECK: \"__cublas$lt$matmul\""),
      IsOkAndHolds(true));
}

TEST_F(FissionBackendTest, ApplyCustomKernelConfigToFusionInstruction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kTritonFusionHlo));
  AutotuneResult::CustomKernelFusionKey config;
  config.set_kernel_index(3);
  google::protobuf::Any any;
  any.PackFrom(config);
  TF_EXPECT_OK(backend_.ApplyConfig(
      *hlo_module->entry_computation()->root_instruction(), any));
  EXPECT_THAT(RunFileCheck(hlo_module->ToString(), "CHECK: \"kernel_index\":3"),
              IsOkAndHolds(true));
}

TEST_F(FissionBackendTest, ApplyCublasConfigToFusionInWhileBody) {
  const char kWhileHlo[] = R"(
HloModule module

fusion_computation {
  fp0 = bf16[1024,1024]{1,0} parameter(0)
  convert0 = f32[1024,1024]{1,0} convert(fp0)
  fp1 = s8[1024,1024]{1,0} parameter(1)
  convert1 = f32[1024,1024]{1,0} convert(fp1)
  ROOT dot = f32[1024,1024]{1,0} dot(convert0, convert1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

while_cond {
  cond_param = (s32[], f32[1024,1024]{1,0}, bf16[1024,1024]{1,0}, s8[1024,1024]{1,0}) parameter(0)
  count = s32[] get-tuple-element(cond_param), index=0
  limit = s32[] constant(1)
  ROOT result = pred[] compare(count, limit), direction=LT
}

while_body {
  body_param = (s32[], f32[1024,1024]{1,0}, bf16[1024,1024]{1,0}, s8[1024,1024]{1,0}) parameter(0)
  count = s32[] get-tuple-element(body_param), index=0
  p0_body = bf16[1024,1024]{1,0} get-tuple-element(body_param), index=2
  p1_body = s8[1024,1024]{1,0} get-tuple-element(body_param), index=3
  fusion = f32[1024,1024]{1,0} fusion(p0_body, p1_body),
    kind=kCustom, calls=fusion_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  one = s32[] constant(1)
  new_count = s32[] add(count, one)
  ROOT result = (s32[], f32[1024,1024]{1,0}, bf16[1024,1024]{1,0}, s8[1024,1024]{1,0}) tuple(new_count, fusion, p0_body, p1_body)
}

ENTRY main {
  p0 = bf16[1024,1024]{1,0} parameter(0)
  p1 = s8[1024,1024]{1,0} parameter(1)
  c0 = s32[] constant(0)
  init_f32 = f32[1024,1024]{1,0} broadcast(f32[] constant(0.0)), dimensions={}
  while_init = (s32[], f32[1024,1024]{1,0}, bf16[1024,1024]{1,0}, s8[1024,1024]{1,0}) tuple(c0, init_f32, p0, p1)
  while_result = (s32[], f32[1024,1024]{1,0}, bf16[1024,1024]{1,0}, s8[1024,1024]{1,0}) while(while_init),
    body=while_body, condition=while_cond
  ROOT result = f32[1024,1024]{1,0} get-tuple-element(while_result), index=1
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kWhileHlo));
  hlo_module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_cublaslt(false);

  HloInstruction* while_instr =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  ASSERT_EQ(while_instr->opcode(), HloOpcode::kWhile);
  HloComputation* body_computation = while_instr->while_body();
  HloInstruction* fusion_instr =
      body_computation->root_instruction()->mutable_operand(1);
  ASSERT_EQ(fusion_instr->opcode(), HloOpcode::kFusion);

  AutotuneResult::GemmKey config;
  config.set_algorithm(3);
  google::protobuf::Any any;
  any.PackFrom(config);
  TF_EXPECT_OK(backend_.ApplyConfig(*fusion_instr, any));
  EXPECT_THAT(
      RunFileCheck(
          hlo_module->ToString(),
          "CHECK: while_body"
          "\nCHECK: custom-call({{.*}}), custom_call_target=\"__cublas$gemm\""
          "\nCHECK: \"selected_algorithm\":\"3\""),
      IsOkAndHolds(true));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

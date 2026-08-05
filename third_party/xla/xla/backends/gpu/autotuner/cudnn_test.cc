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

#include "xla/backends/gpu/autotuner/cudnn.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

using CudnnBackendConfig = stream_executor::dnn::AlgorithmProto;

using ::testing::Gt;
using ::testing::SizeIs;
using ::tsl::proto_testing::EqualsProto;

absl::string_view kCudnnFusionHlo = R"hlo(
  fusion1 {
    p0 = f32[3,28,32] parameter(0)
    p1 = f32[3,28,32] parameter(1)
    ROOT d = f32[3,32,32] dot(p0, p1),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }

  ENTRY e {
    p0 = f32[3,28,32] parameter(0)
    p1 = f32[3,28,32] parameter(1)
    ROOT _ = f32[3,32,32] fusion(p0, p1), kind=kCustom, calls=fusion1,
      backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
  })hlo";

absl::string_view kCudnnConvolutionFusionHlo = R"hlo(
  fusion1 {
    p0 = f32[16,56,56,16] parameter(0)
    p1 = f32[16,3,3,16] parameter(1)
    ROOT c = f32[16,54,54,16] convolution(p0, p1),
      window={size=3x3},
      dim_labels=f01b_i01o->f01b,
      convolution_kind=fprop
  }

  ENTRY e {
    p0 = f32[16,56,56,16] parameter(0)
    p1 = f32[16,3,3,16] parameter(1)
    ROOT _ = f32[16,54,54,16] fusion(p0, p1), kind=kCustom, calls=fusion1,
      backend_config={
        "fusion_backend_config": {
          "kind": "__cudnn$fusion",
        }
      }
  })hlo";

absl::string_view kTritonGemmFusionHlo = R"hlo(
  fusion1 {
    p0 = f32[3,28,32] parameter(0)
    p1 = f32[3,28,32] parameter(1)
    d = f32[3,32,32] dot(p0, p1),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }

  e {
    p0 = f32[3,28,32] parameter(0)
    p1 = f32[3,28,32] parameter(1)
    _ = f32[3,32,32] fusion(p0, p1), kind=kCustom, calls=fusion1,
      backend_config={"fusion_backend_config": {kind: "__triton_gemm"}}
  })hlo";

absl::string_view kScaledDotGemmFusionHlo = R"hlo(
  block_scaled_dot {
    lhs = f8e4m3fn[256,128] parameter(0)
    rhs = f8e4m3fn[384,128] parameter(1)
    lhs_scale = f8e8m0fnu[256,4] parameter(2)
    rhs_scale = f8e8m0fnu[384,4] parameter(3)
    ROOT result = f32[256,384] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }

  ENTRY main {
    lhs = f8e4m3fn[256,128] parameter(0)
    rhs = f8e4m3fn[384,128] parameter(1)
    lhs_scale = f8e8m0fnu[256,4] parameter(2)
    rhs_scale = f8e8m0fnu[384,4] parameter(3)
    ROOT result = f32[256,384] fusion(lhs, rhs, lhs_scale, rhs_scale),
        kind=kCustom, calls=block_scaled_dot,
        backend_config={"fusion_backend_config":{kind:"__triton_gemm"}}
  })hlo";

absl::string_view kCudnnCustomCallHlo = R"hlo(
  HloModule module

  ENTRY %main {
    %arg0 = f32[3,56,56,16]{2,1,0,3} parameter(0)
    %arg1 = f32[3,3,3,64]{2,1,0,3} parameter(1)
    %cudnn-conv = (f32[54,54,16,64]{1,0,3,2}, u8[0]{0})
      custom-call(%arg0, %arg1), custom_call_target="__cudnn$convForward",
      window={size=3x3},
      dim_labels=f01b_i01o->01bf,
      backend_config={
        "cudnn_conv_backend_config":{
          "activation_mode":"kNone",
          "conv_result_scale":1,
          "side_input_scale":0,
          "leakyrelu_alpha":0
        },
      }
    ROOT %get-tuple-element = f32[54,54,16,64]{1,0,3,2} get-tuple-element(%cudnn-conv), index=0
  })hlo";

absl::string_view kUnsupportedHlo = R"hlo(
  computation {
    p0 = s2[3,3] parameter(0)
    p1 = s2[3,3] parameter(1)
    d = s2[3,3] dot(p0, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }

  main {
    p0 = s2[3,3] parameter(0)
    p1 = s2[3,3] parameter(1)
    fusion = s2[3,3] fusion(p0, p1),
      kind=kCustom, calls=computation,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  })hlo";

class CudnnBackendTest : public HloHardwareIndependentTestBase {
 protected:
  CudnnBackendTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()),
        target_config_(stream_executor_),
        debug_options_(
            HloHardwareIndependentTestBase::GetDebugOptionsForTest()) {
    debug_options_.set_xla_gpu_cudnn_gemm_fusion_level(2);
    backend_ = std::make_unique<CudnnBackend>(stream_executor_, &debug_options_,
                                              &compiler_, &target_config_);
  }

  NVPTXCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  DebugOptions debug_options_;
  std::unique_ptr<CudnnBackend> backend_;
};

TEST_F(CudnnBackendTest, CanCreateCublasBackend) {
  ASSERT_NE(nullptr, backend_);
}

TEST_F(CudnnBackendTest, GetSupportedConfigsFromCudnnFusion) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kCudnnFusionHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_->GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOkAndHolds(SizeIs(Gt(0))));
}

TEST_F(CudnnBackendTest, GetSupportedConfigsFromCudnnConvolutionFusion) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kCudnnConvolutionFusionHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_->GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOkAndHolds(SizeIs(Gt(0))));
}

TEST_F(CudnnBackendTest, GetSupportedConfigsFromTritonGemmFusion) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kTritonGemmFusionHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_->GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOkAndHolds(SizeIs(Gt(0))));
}

TEST_F(CudnnBackendTest, GetSupportedConfigsFromScaledDotGemmFusion) {
  se::CudaComputeCapability cc =
      stream_executor_->GetDeviceDescription().cuda_compute_capability();
  if (!cc.IsAtLeastBlackwell()) {
    GTEST_SKIP() << "Block-scaled dot is only supported on Blackwell GPUs.";
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kScaledDotGemmFusionHlo));
  // Test with cudnn_gemm_fusion_level = 0 to ensure it is permanently enabled.
  DebugOptions debug_options = debug_options_;
  debug_options.set_xla_gpu_cudnn_gemm_fusion_level(0);
  CudnnBackend backend(stream_executor_, &debug_options, &compiler_,
                       &target_config_);
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend.GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, absl_testing::IsOkAndHolds(SizeIs(Gt(0))));
}

TEST_F(CudnnBackendTest, GetSupportedConfigsFromCudnnCustomCall) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kCudnnCustomCallHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_->GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()->operand(0)));
  EXPECT_THAT(configs, absl_testing::IsOkAndHolds(SizeIs(Gt(0))));
}

TEST_F(CudnnBackendTest,
       GetSupportedConfigsFromUnsupportedFusionReturnsUnimplementedError) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kUnsupportedHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_->GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs,
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(CudnnBackendTest,
       GetSupportedConfigsReturnsErrorForConvolutionWithNullStreamExecutor) {
  CudnnBackend backend_without_stream_executor(nullptr, &debug_options_,
                                               &compiler_, &target_config_);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kCudnnCustomCallHlo));

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_without_stream_executor.GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()->operand(0)));
  EXPECT_THAT(configs,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(CudnnBackendTest, GetDefaultConfigFromCudnnFusion) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kCudnnFusionHlo));

  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_->GetDefaultConfig(
          (*hlo_module->entry_computation()->root_instruction()));
  ASSERT_OK(config);
  ASSERT_TRUE((*config)->has_algorithm());
  CudnnBackendConfig algorithm_config = (*config)->algorithm();
  EXPECT_GE(algorithm_config.algo_id(), 0);
}

TEST_F(CudnnBackendTest, GetDefaultConfigFromCudnnCustomCall) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kCudnnCustomCallHlo));
  absl::StatusOr<std::unique_ptr<BackendConfig>> config =
      backend_->GetDefaultConfig(
          (*hlo_module->entry_computation()->root_instruction()->operand(0)));
  ASSERT_OK(config);
  ASSERT_TRUE((*config)->has_algorithm());
  CudnnBackendConfig algorithm_config = (*config)->algorithm();
  EXPECT_EQ(algorithm_config.algo_id(), -1);
}

TEST_F(CudnnBackendTest, GetDefaultConfigSucceedsWithNullStreamExecutor) {
  ASSERT_OK_AND_ASSIGN(stream_executor::GpuTargetConfigProto proto,
                       GetGpuTargetConfig(GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(Compiler::GpuTargetConfig target_config,
                       Compiler::GpuTargetConfig::FromProto(proto));
  debug_options_.set_xla_gpu_cudnn_deviceless_compilation_mode(
      DebugOptions::CUDNN_DEVICELESS_COMPILATION_AUTO);
  CudnnBackend backend_without_stream_executor(
      /*stream_executor=*/nullptr, &debug_options_, &compiler_, &target_config);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kCudnnFusionHlo));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_without_stream_executor.GetDefaultConfig(
          (*hlo_module->entry_computation()->root_instruction())));
  ASSERT_TRUE(config->has_algorithm());
}

TEST_F(CudnnBackendTest, CudnnDevicelessCompilationDisabledFails) {
  ASSERT_OK_AND_ASSIGN(stream_executor::GpuTargetConfigProto proto,
                       GetGpuTargetConfig(GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(Compiler::GpuTargetConfig target_config,
                       Compiler::GpuTargetConfig::FromProto(proto));
  debug_options_.set_xla_gpu_cudnn_deviceless_compilation_mode(
      DebugOptions::CUDNN_DEVICELESS_COMPILATION_DISABLED);
  CudnnBackend backend_without_stream_executor(
      /*stream_executor=*/nullptr, &debug_options_, &compiler_, &target_config);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kCudnnFusionHlo));

  EXPECT_THAT(backend_without_stream_executor.GetDefaultConfig(
                  (*hlo_module->entry_computation()->root_instruction())),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(CudnnBackendTest, ApplyConfigToCudnnFusion) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kCudnnFusionHlo));
  CudnnBackendConfig config;
  config.set_algo_id(1);
  HloInstruction* fusion_instr =
      hlo_module->entry_computation()->root_instruction();
  BackendConfig backend_config;
  *backend_config.mutable_algorithm() = config;
  ASSERT_OK(backend_->ApplyConfig(*fusion_instr, backend_config));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion_instr->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().cudnn_fusion_config().plan_id(),
            1);
}

TEST_F(CudnnBackendTest, ApplyConfigToTritonGemmFusionSetsCudnnKind) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kTritonGemmFusionHlo));
  CudnnBackendConfig config;
  config.set_algo_id(1);
  HloInstruction* fusion_instr =
      hlo_module->entry_computation()->root_instruction();
  BackendConfig backend_config;
  *backend_config.mutable_algorithm() = config;
  ASSERT_OK(backend_->ApplyConfig(*fusion_instr, backend_config));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion_instr->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), kCuDnnFusionKind);
  EXPECT_EQ(gpu_config.fusion_backend_config().cudnn_fusion_config().plan_id(),
            1);
}

TEST_F(CudnnBackendTest, ApplyConfigToCudnnCustomCall) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kCudnnCustomCallHlo));
  CudnnBackendConfig config;
  config.set_algo_id(1);
  HloInstruction* instr =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  BackendConfig backend_config;
  *backend_config.mutable_algorithm() = config;
  ASSERT_OK(backend_->ApplyConfig(*instr, backend_config));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       instr->backend_config<GpuBackendConfig>());
  EXPECT_THAT(gpu_config.cudnn_conv_backend_config().algorithm(),
              EqualsProto(config));
}

TEST_F(CudnnBackendTest, ApplyConfigToCudnnCustomCallWithWorkspace) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kCudnnCustomCallHlo));
  CudnnBackendConfig config;
  config.set_algo_id(1);
  config.mutable_workspace_size()->set_value(1024);
  HloInstruction* instr =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  BackendConfig backend_config;
  *backend_config.mutable_algorithm() = config;
  ASSERT_OK(backend_->ApplyConfig(*instr, backend_config));

  auto* replaced_instr =
      hlo_module->entry_computation()->GetInstructionWithName("cudnn-conv");

  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       replaced_instr->backend_config<GpuBackendConfig>());
  EXPECT_THAT(gpu_config.cudnn_conv_backend_config().algorithm(),
              EqualsProto(config));
  EXPECT_EQ(replaced_instr->shape().tuple_shapes(1).dimensions(0), 1024);
}

}  // namespace gpu
}  // namespace xla

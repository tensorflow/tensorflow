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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/fission_backend.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/backends/gpu/transforms/block_scaling_rewriter.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/mlir_context_pool.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_dnn.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

// Returns the first device executor of the default platform.
se::StreamExecutor* GetDefaultStreamExecutor() {
  absl::StatusOr<se::Platform*> platform = PlatformUtil::GetDefaultPlatform();
  CHECK_OK(platform.status());
  absl::StatusOr<se::StreamExecutor*> stream_executor =
      (*platform)->ExecutorForDevice(0);
  CHECK_OK(stream_executor.status());
  return *stream_executor;
}

}  // namespace

using CudnnBackendConfig = stream_executor::dnn::AlgorithmProto;

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AnyOf;
using ::testing::Each;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Property;
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

absl::string_view kLoopDotFusionHlo = R"hlo(
  wrapped_dot_computation {
    p0 = f32[4,4] parameter(0)
    p1 = f32[4,4] parameter(1)
    ROOT d = f32[4,4] dot(p0, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  e {
    p0 = f32[4,4] parameter(0)
    p1 = f32[4,4] parameter(1)
    ROOT wrapped_dot = f32[4,4] fusion(p0, p1), kind=kLoop,
      calls=wrapped_dot_computation
  })hlo";

absl::string_view kF64GemmFusionHlo = R"hlo(
  fusion1 {
    p0 = f64[3,28,32] parameter(0)
    p1 = f64[3,28,32] parameter(1)
    ROOT d = f64[3,32,32] dot(p0, p1),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }

  e {
    p0 = f64[3,28,32] parameter(0)
    p1 = f64[3,28,32] parameter(1)
    ROOT _ = f64[3,32,32] fusion(p0, p1), kind=kCustom, calls=fusion1,
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

// The scales are already swizzled, as emitted by CudnnScaledDotRewriter.
constexpr absl::string_view kScaledDotCudnnFusionHlo = R"hlo(
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
        backend_config={"fusion_backend_config":{kind:"__cudnn$fusion"}}
  })hlo";

// K = 64 requires padding the scales from 2 to 4 columns before swizzling.
constexpr absl::string_view kScaledDotWithEpilogueGemmFusionHlo = R"hlo(
  block_scaled_dot {
    lhs = f8e4m3fn[256,64] parameter(0)
    rhs = f8e4m3fn[384,64] parameter(1)
    lhs_scale = f8e8m0fnu[256,2] parameter(2)
    rhs_scale = f8e8m0fnu[384,2] parameter(3)
    dot = f32[256,384] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1}, rhs_contracting_dims={1}
    ROOT convert = bf16[256,384] convert(dot)
  }

  ENTRY main {
    lhs = f8e4m3fn[256,64] parameter(0)
    rhs = f8e4m3fn[384,64] parameter(1)
    lhs_scale = f8e8m0fnu[256,2] parameter(2)
    rhs_scale = f8e8m0fnu[384,2] parameter(3)
    ROOT result = bf16[256,384] fusion(lhs, rhs, lhs_scale, rhs_scale),
        kind=kCustom, calls=block_scaled_dot,
        backend_config={"fusion_backend_config":{kind:"__triton_gemm"}}
  })hlo";

// Non-K-minor LHS and scale are transposed to K-minor inside the GEMM fusion.
constexpr absl::string_view kScaledDotWithPrologueTransposeGemmFusionHlo =
    R"hlo(
  block_scaled_dot {
    lhs_in = f8e4m3fn[64,256] parameter(0)
    lhs = f8e4m3fn[256,64] transpose(lhs_in), dimensions={1,0}
    rhs = f8e4m3fn[384,64] parameter(1)
    lhs_scale_in = f8e8m0fnu[2,256] parameter(2)
    lhs_scale = f8e8m0fnu[256,2] transpose(lhs_scale_in), dimensions={1,0}
    rhs_scale = f8e8m0fnu[384,2] parameter(3)
    ROOT dot = f32[256,384] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }

  ENTRY main {
    lhs = f8e4m3fn[64,256] parameter(0)
    rhs = f8e4m3fn[384,64] parameter(1)
    lhs_scale = f8e8m0fnu[2,256] parameter(2)
    rhs_scale = f8e8m0fnu[384,2] parameter(3)
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

// A "well-behaved" concatenate (all operands have the same size along the
// concat dimension and all operands are used downstream) is eligible to
// become a cuDNN fusion. Untagged (kLoop), so CudnnBackend must decide
// eligibility on its own rather than trusting an upstream tag.
absl::string_view kWellFormedConcatFusionHlo = R"hlo(
  fused_computation {
    p0 = f32[4,8] parameter(0)
    p1 = f32[4,8] parameter(1)
    ROOT concat = f32[4,16] concatenate(p0, p1), dimensions={1}
  }

  ENTRY e {
    a = f32[4,8] parameter(0)
    b = f32[4,8] parameter(1)
    ROOT fusion = f32[4,16] fusion(a, b), kind=kLoop, calls=fused_computation
  })hlo";

// A fusion already claimed by another custom pass (e.g. a legacy
// custom-fusion match) must not be reclaimed by CudnnBackend even if it would
// otherwise look eligible (single, well-formed concatenate).
absl::string_view kConcatFusionTaggedCustomFusionKindHlo = R"hlo(
  fused_computation {
    p0 = f32[4,8] parameter(0)
    p1 = f32[4,8] parameter(1)
    ROOT concat = f32[4,16] concatenate(p0, p1), dimensions={1}
  }

  ENTRY e {
    a = f32[4,8] parameter(0)
    b = f32[4,8] parameter(1)
    ROOT fusion = f32[4,16] fusion(a, b), kind=kCustom, calls=fused_computation,
      backend_config={"fusion_backend_config": {kind: "__custom_fusion"}}
  })hlo";

// Two concatenates in one fusion: cuDNN reports plans for it but fails to
// lower it, so it must be rejected before any cuDNN query.
absl::string_view kMultiConcatFusionHlo = R"hlo(
  fused_computation {
    p0 = f32[4,8] parameter(0)
    p1 = f32[4,8] parameter(1)
    c0 = f32[4,16] concatenate(p0, p1), dimensions={1}
    c1 = f32[4,32] concatenate(c0, c0), dimensions={1}
    ROOT neg = f32[4,32] negate(c1)
  }

  ENTRY e {
    a = f32[4,8] parameter(0)
    b = f32[4,8] parameter(1)
    ROOT fusion = f32[4,32] fusion(a, b), kind=kLoop, calls=fused_computation
  })hlo";

class CudnnBackendTest : public HloHardwareIndependentTestBase {
 protected:
  CudnnBackendTest()
      : stream_executor_(GetDefaultStreamExecutor()),
        target_config_(stream_executor_),
        debug_options_(
            HloHardwareIndependentTestBase::GetDebugOptionsForTest()) {
    debug_options_.set_xla_gpu_cudnn_gemm_fusion_level(2);
    backend_ = std::make_unique<CudnnBackend>(stream_executor_, &debug_options_,
                                              &compiler_, &target_config_);
  }

  std::unique_ptr<FissionBackend> CreateFissionBackend() {
    auto pipeline = std::make_unique<HloPassPipeline>("cudnn_fission");
    pipeline->AddPass(std::make_unique<CudnnScaledDotRewriter>());
    return std::make_unique<FissionBackend>(
        &debug_options_, &compiler_, &target_config_,
        std::make_unique<CudnnBackend>(stream_executor_, &debug_options_,
                                       &compiler_, &target_config_),
        std::move(pipeline), &alias_info_, &mlir_context_pool_,
        stream_executor_);
  }

  NVPTXCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  DebugOptions debug_options_;
  std::unique_ptr<CudnnBackend> backend_;
  GpuAliasInfo alias_info_{target_config_.device_description};
  MlirContextPool mlir_context_pool_{CreateMlirContext};
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

TEST_F(CudnnBackendTest, GetSupportedConfigsFromF64GemmFusionReturnsEmpty) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kF64GemmFusionHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      backend_->GetSupportedConfigs(
          (*hlo_module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsOkAndHolds(IsEmpty()));
}

TEST_F(CudnnBackendTest, GetSupportedConfigsFromScaledDotCudnnFusion) {
  if (!CudnnScaledDotHelper::IsAvailable(
          stream_executor_->GetDeviceDescription())) {
    GTEST_SKIP() << "cuDNN scaled dot requires Blackwell and cuDNN 9.10+.";
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kScaledDotCudnnFusionHlo));
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

TEST_F(CudnnBackendTest, GetSupportedConfigsFromScaledDotGemmFusionIsEmpty) {
  ASSERT_OK_AND_ASSIGN(stream_executor::GpuTargetConfigProto proto,
                       GetGpuTargetConfig(GpuModel::B200));
  ASSERT_OK_AND_ASSIGN(Compiler::GpuTargetConfig b200_target_config,
                       Compiler::GpuTargetConfig::FromProto(proto));
  CudnnBackend backend(/*stream_executor=*/nullptr, &debug_options_, &compiler_,
                       &b200_target_config);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kScaledDotGemmFusionHlo));
  EXPECT_THAT(backend.GetSupportedConfigs(
                  *hlo_module->entry_computation()->root_instruction()),
              IsOkAndHolds(IsEmpty()));
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
  if (!se::gpu::SupportsDevicelessCudnnCompilation()) {
    GTEST_SKIP() << "Deviceless cuDNN compilation requires cuDNN >= "
                 << se::gpu::kMinDevicelessCudnnVersion;
  }
  ASSERT_OK_AND_ASSIGN(stream_executor::GpuTargetConfigProto proto,
                       GetGpuTargetConfig(GpuModel::H100_SXM));
  ASSERT_OK_AND_ASSIGN(Compiler::GpuTargetConfig target_config,
                       Compiler::GpuTargetConfig::FromProto(proto));
  DebugOptions debug_options = debug_options_;
  debug_options.set_xla_gpu_cudnn_deviceless_compilation_mode(
      DebugOptions::CUDNN_DEVICELESS_COMPILATION_AUTO);
  CudnnBackend backend_without_stream_executor(
      /*stream_executor=*/nullptr, &debug_options, &compiler_, &target_config);
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
  DebugOptions debug_options = debug_options_;
  debug_options.set_xla_gpu_cudnn_deviceless_compilation_mode(
      DebugOptions::CUDNN_DEVICELESS_COMPILATION_DISABLED);
  CudnnBackend backend_without_stream_executor(
      /*stream_executor=*/nullptr, &debug_options, &compiler_, &target_config);
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

TEST_F(CudnnBackendTest, ApplyConfigToLoopFusionSetsCustomFusionKind) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kLoopDotFusionHlo));
  CudnnBackendConfig config;
  config.set_algo_id(1);
  HloInstruction* fusion_instr =
      hlo_module->entry_computation()->root_instruction();
  BackendConfig backend_config;
  *backend_config.mutable_algorithm() = config;
  ASSERT_OK(backend_->ApplyConfig(*fusion_instr, backend_config));
  EXPECT_EQ(fusion_instr->fusion_kind(), HloInstruction::FusionKind::kCustom);
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion_instr->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().kind(), kCuDnnFusionKind);
  EXPECT_EQ(gpu_config.fusion_backend_config().cudnn_fusion_config().plan_id(),
            1);
}

TEST_F(CudnnBackendTest, ApplyConfigToScaledDotCudnnFusionKeepsOperands) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kScaledDotCudnnFusionHlo));
  BackendConfig backend_config;
  backend_config.mutable_algorithm()->set_algo_id(1);
  HloInstruction* fusion_instr =
      hlo_module->entry_computation()->root_instruction();
  ASSERT_OK(backend_->ApplyConfig(*fusion_instr, backend_config));
  ASSERT_EQ(hlo_module->entry_computation()->root_instruction(), fusion_instr);
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       fusion_instr->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.fusion_backend_config().cudnn_fusion_config().plan_id(),
            1);
  EXPECT_THAT(fusion_instr->operands(),
              Each(Property(&HloInstruction::opcode, HloOpcode::kParameter)));
}

TEST_F(CudnnBackendTest, ApplyConfigToScaledDotGemmFusionFails) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kScaledDotGemmFusionHlo));
  BackendConfig backend_config;
  backend_config.mutable_algorithm()->set_algo_id(1);
  EXPECT_THAT(
      backend_->ApplyConfig(
          *hlo_module->entry_computation()->root_instruction(), backend_config),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("CudnnScaledDotRewriter")));
}

TEST_F(CudnnBackendTest, FissionDoesNotSupportFusionWithoutScaledDot) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kTritonGemmFusionHlo));
  std::unique_ptr<FissionBackend> fission_backend = CreateFissionBackend();
  EXPECT_TRUE(fission_backend->CanProduceWrongResults());
  HloInstruction& fusion = *hlo_module->entry_computation()->root_instruction();
  EXPECT_FALSE(fission_backend->IsSupported(fusion));
  EXPECT_THAT(fission_backend->GetSupportedConfigs(fusion),
              IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(fission_backend->ApplyConfig(fusion, BackendConfig{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not support")));
}

TEST_F(CudnnBackendTest, FissionDoesNotSupportScaledDotCudnnFusion) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kScaledDotCudnnFusionHlo));
  std::unique_ptr<FissionBackend> fission_backend = CreateFissionBackend();
  HloInstruction& fusion = *hlo_module->entry_computation()->root_instruction();
  EXPECT_FALSE(fission_backend->IsSupported(fusion));
  EXPECT_THAT(fission_backend->GetSupportedConfigs(fusion),
              IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(fission_backend->ApplyConfig(fusion, BackendConfig{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not support")));
}

TEST_F(CudnnBackendTest, FissionRewritesScaledDotGemmFusionToCudnnFusion) {
  if (!CudnnScaledDotHelper::IsAvailable(
          stream_executor_->GetDeviceDescription())) {
    GTEST_SKIP() << "cuDNN scaled dot requires Blackwell and cuDNN 9.10+.";
  }
  std::unique_ptr<FissionBackend> fission_backend = CreateFissionBackend();
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kScaledDotWithEpilogueGemmFusionHlo));
  HloInstruction* fusion = hlo_module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       fission_backend->GetSupportedConfigs(*fusion));
  ASSERT_THAT(configs, Not(IsEmpty()));
  EXPECT_OK(fission_backend->Compile(*fusion, *configs[0]));

  ASSERT_OK(fission_backend->ApplyConfig(*fusion, *configs[0]));

  // The cuDNN fusion is preserved and the scale padding/swizzle is fused. The
  // convert epilogue can't fuse into the cuDNN fusion and stays standalone.
  const HloComputation* entry = hlo_module->entry_computation();
  EXPECT_THAT(entry->instructions(),
              Each(Property(&HloInstruction::opcode,
                            Not(AnyOf(HloOpcode::kPad, HloOpcode::kTranspose,
                                      HloOpcode::kScaledDot)))));
  const HloInstruction* cudnn_fusion = entry->root_instruction()->operand(0);
  ASSERT_TRUE(IsCudnnFusion(*cudnn_fusion));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       cudnn_fusion->backend_config<GpuBackendConfig>());
  EXPECT_TRUE(
      gpu_config.fusion_backend_config().cudnn_fusion_config().has_plan_id());
}

TEST_F(CudnnBackendTest,
       FissionExtractsTransposePrologueFromScaledDotGemmFusion) {
  if (!CudnnScaledDotHelper::IsAvailable(
          stream_executor_->GetDeviceDescription())) {
    GTEST_SKIP() << "cuDNN scaled dot requires Blackwell and cuDNN 9.10+.";
  }
  std::unique_ptr<FissionBackend> fission_backend = CreateFissionBackend();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(
                           kScaledDotWithPrologueTransposeGemmFusionHlo));
  HloInstruction* fusion = hlo_module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       fission_backend->GetSupportedConfigs(*fusion));
  ASSERT_THAT(configs, Not(IsEmpty()));
  EXPECT_OK(fission_backend->Compile(*fusion, *configs[0]));

  ASSERT_OK(fission_backend->ApplyConfig(*fusion, *configs[0]));

  const HloComputation* entry = hlo_module->entry_computation();
  EXPECT_THAT(
      entry->instructions(),
      Each(Property(&HloInstruction::opcode,
                    Not(AnyOf(HloOpcode::kPad, HloOpcode::kScaledDot)))));
  const HloInstruction* root = entry->root_instruction();
  ASSERT_TRUE(IsCudnnFusion(*root));
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kTranspose);
  ASSERT_EQ(root->operand(2)->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(root->operand(2)->operand(0)->opcode(), HloOpcode::kFusion);
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

TEST_F(CudnnBackendTest,
       IsSupportedForUntaggedConcatFusionIsFalseWithFlagDisabledByDefault) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kWellFormedConcatFusionHlo));
  // debug_options_ leaves xla_gpu_cudnn_non_gemm_fusion_level at its default
  // (0), so an otherwise-eligible untagged fusion must not be claimed.
  EXPECT_FALSE(backend_->IsSupported(
      *hlo_module->entry_computation()->root_instruction()));
}

TEST_F(CudnnBackendTest,
       IsSupportedForFusionAlreadyTaggedCustomFusionKindIsFalse) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kConcatFusionTaggedCustomFusionKindHlo));
  DebugOptions debug_options = debug_options_;
  debug_options.set_xla_gpu_cudnn_non_gemm_fusion_level(1);
  CudnnBackend backend(stream_executor_, &debug_options, &compiler_,
                       &target_config_);
  EXPECT_FALSE(backend.IsSupported(
      *hlo_module->entry_computation()->root_instruction()));
}

TEST_F(CudnnBackendTest,
       IsSupportedForUntaggedConcatFusionIsFalseWithCudnnTooOld) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      ParseAndReturnVerifiedModule(kWellFormedConcatFusionHlo));
  DebugOptions debug_options = debug_options_;
  debug_options.set_xla_gpu_cudnn_non_gemm_fusion_level(1);
  Compiler::GpuTargetConfig target_config = target_config_;
  target_config.device_description.set_dnn_version(
      stream_executor::SemanticVersion(9, 25, 0));
  CudnnBackend backend(stream_executor_, &debug_options, &compiler_,
                       &target_config);
  EXPECT_FALSE(backend.IsSupported(
      *hlo_module->entry_computation()->root_instruction()));
}

TEST_F(CudnnBackendTest, IsSupportedForFusionWithMultipleConcatsIsFalse) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kMultiConcatFusionHlo));
  DebugOptions debug_options = debug_options_;
  debug_options.set_xla_gpu_cudnn_non_gemm_fusion_level(1);
  Compiler::GpuTargetConfig target_config = target_config_;
  target_config.device_description.set_dnn_version(
      stream_executor::SemanticVersion(9, 26, 0));
  target_config.device_description.set_gpu_compute_capability(
      se::CudaComputeCapability::Blackwell());
  CudnnBackend backend(stream_executor_, &debug_options, &compiler_,
                       &target_config);
  EXPECT_FALSE(backend.IsSupported(
      *hlo_module->entry_computation()->root_instruction()));
}

}  // namespace gpu
}  // namespace xla

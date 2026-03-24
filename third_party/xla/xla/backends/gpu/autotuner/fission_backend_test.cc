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

#include "xla/backends/gpu/autotuner/fission_backend.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#if GOOGLE_CUDA
#include "xla/backends/gpu/autotuner/cublas.h"
#elif TENSORFLOW_USE_ROCM
#include "xla/backends/gpu/autotuner/rocblas.h"
#endif
#include "xla/backends/gpu/autotuner/custom_kernel.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/backends/gpu/transforms/custom_kernel_fusion_rewriter.h"
#include "xla/backends/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/backends/gpu/transforms/gemm_rewriter.h"
#include "xla/backends/gpu/transforms/scaled_dot_rewriter.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace {

using absl_testing::IsOk;
using absl_testing::IsOkAndHolds;
using ::testing::Gt;
using ::testing::HasSubstr;
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

const char kHloWithUpcast[] = R"(
  HloModule module, entry_computation_layout={(bf16[1024,1024]{1,0}, bf16[1024,1024]{1,0})->f32[1024,1024]{1,0}}

  %gemm_fusion_r_computation {
    %parameter_0 = bf16[1024,1024]{1,0} parameter(0)
    %convert.2 = f32[1024,1024]{1,0} convert(%parameter_0)
    %parameter_1 = bf16[1024,1024]{1,0} parameter(1)
    %convert.3 = f32[1024,1024]{1,0} convert(%parameter_1)
    ROOT %r.1 = f32[1024,1024]{1,0} dot(%convert.2, %convert.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    %p0 = bf16[1024,1024]{1,0} parameter(0)
    %p1 = bf16[1024,1024]{1,0} parameter(1)
    ROOT %gemm_fusion_r = f32[1024,1024]{1,0} fusion(%p0, %p1), kind=kCustom, calls=gemm_fusion_r_computation, backend_config={"fusion_backend_config":{"kind":"__triton_gemm"},"force_earliest_schedule":false}
  })";

const char kHloWithUpcastPrologueK64[] = R"(
  HloModule module

  %gemm_fusion_r_computation (parameter_0.1: f32[1,256,4,16], parameter_1.1: bf16[1,4,16,4096]) -> f32[256,4096] {
    %parameter_0.1 = f32[1,256,4,16]{3,2,1,0} parameter(0)
    %bitcast.60 = f32[256,64]{1,0} bitcast(f32[1,256,4,16]{3,2,1,0} %parameter_0.1)
    %parameter_1.1 = bf16[1,4,16,4096]{3,2,1,0} parameter(1)
    %bitcast.61 = bf16[64,4096]{1,0} bitcast(bf16[1,4,16,4096]{3,2,1,0} %parameter_1.1)
    %convert.22 = f32[64,4096]{1,0} convert(bf16[64,4096]{1,0} %bitcast.61)
    ROOT r = f32[256,4096]{1,0} dot(f32[256,64]{1,0} %bitcast.60, f32[64,4096]{1,0} %convert.22), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    %p0 = f32[1,256,4,16] parameter(0)
    %p1 = bf16[1,4,16,4096] parameter(1)
    ROOT %gemm_fusion_r = f32[256,4096] fusion(%p0, %p1), kind=kCustom,
    calls=gemm_fusion_r_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"},"force_earliest_schedule":false}
  }
)";

const char kHloWithUpcastPrologueK128[] = R"(
  HloModule module

  %gemm_fusion_r_computation (parameter_0.1: f32[1,256,4,32], parameter_1.1: bf16[1,4,32,4096]) -> f32[256,4096] {
    %parameter_0.1 = f32[1,256,4,32]{3,2,1,0} parameter(0)
    %bitcast.60 = f32[256,128]{1,0} bitcast(f32[1,256,4,32]{3,2,1,0} %parameter_0.1)
    %parameter_1.1 = bf16[1,4,32,4096]{3,2,1,0} parameter(1)
    %bitcast.61 = bf16[128,4096]{1,0} bitcast(bf16[1,4,32,4096]{3,2,1,0} %parameter_1.1)
    %convert.22 = f32[128,4096]{1,0} convert(bf16[128,4096]{1,0} %bitcast.61)
    ROOT r = f32[256,4096]{1,0} dot(f32[256,128]{1,0} %bitcast.60, f32[128,4096]{1,0} %convert.22), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY main {
    %p0 = f32[1,256,4,32] parameter(0)
    %p1 = bf16[1,4,32,4096] parameter(1)
    ROOT %gemm_fusion_r = f32[256,4096] fusion(%p0, %p1), kind=kCustom,
    calls=gemm_fusion_r_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"},"force_earliest_schedule":false}
  }
)";

const char kHloWithUpcastPrologueEpilogueK64[] = R"(
  HloModule module

  %gemm_fusion_r_computation (parameter_0.1: f32[1,256,4,16], parameter_1.1: bf16[1,4,16,4096]) -> bf16[1048576] {
    %parameter_0.1 = f32[1,256,4,16]{3,2,1,0} parameter(0)
    %bitcast.60 = f32[256,64]{1,0} bitcast(f32[1,256,4,16]{3,2,1,0} %parameter_0.1)
    %parameter_1.1 = bf16[1,4,16,4096]{3,2,1,0} parameter(1)
    %bitcast.61 = bf16[64,4096]{1,0} bitcast(bf16[1,4,16,4096]{3,2,1,0} %parameter_1.1)
    %convert.22 = f32[64,4096]{1,0} convert(bf16[64,4096]{1,0} %bitcast.61)
    %dot.5 = f32[256,4096]{1,0} dot(f32[256,64]{1,0} %bitcast.60, f32[64,4096]{1,0} %convert.22), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    %convert.23 = bf16[256,4096]{1,0} convert(f32[256,4096]{1,0} %dot.5)
    %bitcast.62 = bf16[1,256,4096]{2,1,0} bitcast(bf16[256,4096]{1,0} %convert.23)
    %transpose.18 = bf16[1,4096,256]{2,1,0} transpose(bf16[1,256,4096]{2,1,0} %bitcast.62), dimensions={0,2,1}
    ROOT %bitcast.63 = bf16[1048576]{0} bitcast(bf16[1,4096,256]{2,1,0} %transpose.18)
  }

  ENTRY main {
    %p0 = f32[1,256,4,16] parameter(0)
    %p1 = bf16[1,4,16,4096] parameter(1)
    ROOT %gemm_fusion_r = bf16[1048576] fusion(%p0, %p1), kind=kCustom,
    calls=gemm_fusion_r_computation,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"},"force_earliest_schedule":false}
  }
)";

const char kF8TritonFusionHlo[] = R"(
HloModule o

gemm_fusion {
  p0 = f8e4m3fn[64,6144]{1,0} parameter(0)
  p1 = f8e4m3fn[64,6144]{1,0} parameter(1)
  ROOT %dot.0 = f32[64,64]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  p0 = f8e4m3fn[64,6144]{1,0} parameter(0)
  p1 = f8e4m3fn[64,6144]{1,0} parameter(1)
  ROOT %dot.0 = f32[64,64]{1,0} fusion(p0, p1), kind=kCustom, calls=gemm_fusion, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm"},"force_earliest_schedule":false}
})";

const char kScaledDotFusionHlo[] = R"(
HloModule module

fusion_computation {
  p0 = f32[1024,1024] parameter(0)
  p1 = f32[1024,1024] parameter(1)
  p0_scale = f32[1024,8] parameter(2)
  p1_scale = f32[8,1024] parameter(3)
  ROOT r = f32[1024,1024] scaled-dot(p0, p1, p0_scale, p1_scale),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[1024,1024] parameter(0)
  p1 = f32[1024,1024] parameter(1)
  p0_scale = f32[1024,8] parameter(2)
  p1_scale = f32[8,1024] parameter(3)
  ROOT r = f32[1024,1024] fusion(p0, p1, p0_scale, p1_scale),
    kind=kCustom, calls=fusion_computation
})";

const char kUnsupportedFusionHlo[] = R"(
  HloModule module
  computation {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    convert0 = f32[1024,1024]{1,0} convert(p0)
    p1 = s8[1024,1024]{1,0} parameter(1)
    convert1 = f32[1024,1024]{1,0} convert(p1)
    ROOT add = f32[1024,1024]{1,0} add(convert0, convert1)
  }

  ENTRY main {
    p0 = bf16[1024,1024]{1,0} parameter(0)
    p1 = s8[1024,1024]{1,0} parameter(1)
    ROOT fusion = f32[1024,1024]{1,0} fusion(p0, p1),
      kind=kCustom, calls=computation
  })";

std::unique_ptr<HloPassPipeline> GetCublasRewriterPipeline(
    const se::DeviceDescription& device_description) {
  auto pipeline = std::make_unique<HloPassPipeline>("fission_pipeline");
  pipeline->AddPass(std::make_unique<ScaledDotRewriter>());
  pipeline->AddPass(std::make_unique<DotAlgorithmRewriter>());
  for (GemmRewriterOptions::DType dtype :
       {GemmRewriterOptions::DType::kFp8Only,
        GemmRewriterOptions::DType::kNonFp8Only}) {
    auto gemm_rewriter = std::make_unique<GemmRewriter>(
        device_description.gpu_compute_capability(),
        device_description.runtime_version(), GemmRewriterOptions{dtype});
    pipeline->AddPass(std::move(gemm_rewriter));
  }
  return pipeline;
}

std::unique_ptr<HloPassPipeline> GetCustomKernelRewriterPipeline(
    const se::DeviceDescription& device_description) {
  auto pipeline = std::make_unique<HloPassPipeline>("fission_pipeline");
  pipeline->AddPass(
      std::make_unique<CustomKernelFusionRewriter>(&device_description));
  return pipeline;
}

// Static helper to create a BLAS backend (Cublas on CUDA, Rocblas on ROCm).
std::unique_ptr<GpuCodegenBackend> CreateCublasBackend(
    se::StreamExecutor* stream_executor, const DebugOptions* debug_options,
    Compiler* compiler, const Compiler::GpuTargetConfig* target_config) {
#if GOOGLE_CUDA
  return std::make_unique<CublasBackend>(stream_executor, debug_options,
                                         compiler, target_config);
#elif TENSORFLOW_USE_ROCM
  return std::make_unique<RocblasBackend>(stream_executor, debug_options,
                                          compiler, target_config);
#endif
  LOG(FATAL) << "Neither CUDA nor ROCm is enabled.";
}

std::unique_ptr<GpuCodegenBackend> CreateCublasBackendWithF8Fallback(
    se::StreamExecutor* stream_executor, const DebugOptions* debug_options,
    Compiler* compiler, const Compiler::GpuTargetConfig* target_config) {
#if GOOGLE_CUDA
  return std::make_unique<CublasBackend>(stream_executor, debug_options,
                                         compiler, target_config,
                                         /*enable_f8_fallback=*/true);
#elif TENSORFLOW_USE_ROCM
  return std::make_unique<RocblasBackend>(stream_executor, debug_options,
                                          compiler, target_config,
                                          /*fp8_lt_fallback=*/true);
#endif
  LOG(FATAL) << "Neither CUDA nor ROCm is enabled.";
}

std::unique_ptr<GpuCodegenBackend> CreateCustomKernelBackend(
    se::StreamExecutor* stream_executor, const DebugOptions* debug_options,
    Compiler* compiler, const Compiler::GpuTargetConfig* target_config) {
  return std::make_unique<CustomKernelBackend>(stream_executor, debug_options,
                                               compiler, target_config);
}

bool IsRocm(se::StreamExecutor* stream_executor) {
  return stream_executor->GetDeviceDescription()
      .gpu_compute_capability()
      .IsRocm();
}

struct FissionTestParams {
  std::string test_name;
  std::string hlo_string;
  // Factory function to create the rewriter pipeline.
  std::function<std::unique_ptr<HloPassPipeline>(
      const se::DeviceDescription& device_description)>
      pipeline_factory;
  // Factory function to create the underlying codegen backend.
  std::function<std::unique_ptr<GpuCodegenBackend>(
      se::StreamExecutor*, const DebugOptions*, Compiler*,
      const Compiler::GpuTargetConfig*)>
      backend_factory;
  // Substrings expected to be in the module after ApplyConfig.
  std::function<std::vector<std::string>(
      const se::DeviceDescription& device_description)>
      expected_module_substrings_fn;
  std::string expected_backend_name;
};

class FissionTest : public HloHardwareIndependentTestBase,
                    public ::testing::WithParamInterface<FissionTestParams> {
 protected:
  DebugOptions debug_options_;
  se::Platform* platform_;
  se::StreamExecutor* stream_executor_;
  std::unique_ptr<Compiler> compiler_;
  Compiler::GpuTargetConfig target_config_;
  se::DeviceDescription device_description_;
  std::unique_ptr<HloPassPipeline> rewriter_pipeline_;
  std::unique_ptr<GpuCodegenBackend> base_codegen_backend_;
  GpuAliasInfo alias_info_;
  std::unique_ptr<FissionBackend> fission_backend_;
  mlir::MLIRContext mlir_context_;

  FissionTest()
      : platform_(PlatformUtil::GetDefaultPlatform().value()),
        stream_executor_(platform_->ExecutorForDevice(0).value()),
        compiler_(Compiler::GetForPlatform(platform_->id()).value()),
        target_config_(stream_executor_),
        device_description_(stream_executor_->GetDeviceDescription()),
        rewriter_pipeline_(GetParam().pipeline_factory(device_description_)),
        base_codegen_backend_(
            GetParam().backend_factory(stream_executor_, &debug_options_,
                                       compiler_.get(), &target_config_)),
        alias_info_(device_description_),
        fission_backend_(std::make_unique<FissionBackend>(
            &debug_options_, compiler_.get(), &target_config_,
            std::move(base_codegen_backend_), std::move(rewriter_pipeline_),
            &alias_info_, &mlir_context_, stream_executor_)) {}
};

TEST_P(FissionTest, CanCreateFissionBackend) {
  const std::string& test_name = GetParam().test_name;
  if (IsRocm(stream_executor_) && test_name == "TritonFusion_CustomKernel") {
    GTEST_SKIP() << test_name << " is not supported on ROCm";
  }

  std::string expected_name = GetParam().expected_backend_name;
  if (IsRocm(stream_executor_) && expected_name == "CUBLAS_FISSION") {
    expected_name = "ROCBLAS_FISSION";
  }
  EXPECT_EQ(fission_backend_->name(), expected_name);
}

TEST_P(FissionTest, GetSupportedConfigs) {
  const std::string& test_name = GetParam().test_name;
  if (IsRocm(stream_executor_) && test_name == "TritonFusion_CustomKernel") {
    GTEST_SKIP() << test_name << " is not supported on ROCm";
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(GetParam().hlo_string));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      fission_backend_->GetSupportedConfigs(
          (*module->entry_computation()->root_instruction()));
  // ROCm returns multiple algorithm configurations, so we check for at least 1.
  EXPECT_THAT(configs, IsOkAndHolds(testing::SizeIs(testing::Ge(1))));
}

TEST_P(FissionTest, GetSupportedConfigsUnsupportedFusion) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kUnsupportedFusionHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      fission_backend_->GetSupportedConfigs(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsOkAndHolds(testing::IsEmpty()));
}

TEST_P(FissionTest, GetDefaultConfig) {
  const std::string& test_name = GetParam().test_name;
  if (IsRocm(stream_executor_) && (test_name == "TritonFusion_CublasLt_F8" ||
                                   test_name == "TritonFusion_CustomKernel")) {
    GTEST_SKIP() << test_name << " is not supported on ROCm";
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(GetParam().hlo_string));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_THAT(fission_backend_->GetDefaultConfig(*fusion), IsOk());
}

TEST_P(FissionTest, Compile) {
  const std::string& test_name = GetParam().test_name;
  if (IsRocm(stream_executor_) && (test_name == "TritonFusion_CublasLt_F8" ||
                                   test_name == "TritonFusion_CustomKernel")) {
    GTEST_SKIP() << test_name << " is not supported on ROCm";
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(GetParam().hlo_string));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                       fission_backend_->GetDefaultConfig(*fusion));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                       fission_backend_->Compile(*fusion, *config));
  EXPECT_NE(executable, nullptr);
}

TEST_P(FissionTest, ApplyConfig) {
  const std::string& test_name = GetParam().test_name;
  if (IsRocm(stream_executor_) && (test_name == "TritonFusion_CublasLt_F8" ||
                                   test_name == "TritonFusion_CustomKernel")) {
    GTEST_SKIP() << test_name << " is not supported on ROCm";
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(GetParam().hlo_string));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                       fission_backend_->GetDefaultConfig(*fusion));
  EXPECT_THAT(fission_backend_->ApplyConfig(*fusion, *config), IsOk());
  std::string module_str = module->ToString();
  for (const std::string& expected_substr :
       GetParam().expected_module_substrings_fn(device_description_)) {
    EXPECT_THAT(module_str, HasSubstr(expected_substr));
  }
}

INSTANTIATE_TEST_SUITE_P(
    FissionTests, FissionTest,
    ::testing::ValuesIn<FissionTestParams>({
        {"TritonFusion_Cublas", kTritonFusionHlo, &GetCublasRewriterPipeline,
         &CreateCublasBackend,
         /*expected_module_substrings_fn=*/
         [](const se::DeviceDescription& device_description) {
           return std::vector<std::string>{
               "custom_call_target=\"__cublas$gemm\"",
               "\"selected_algorithm\":\"-1\""};
         },
         /*expected_backend_name=*/"CUBLAS_FISSION"},
        {"TritonFusion_CublasLt_F8", kF8TritonFusionHlo,
         &GetCublasRewriterPipeline, &CreateCublasBackendWithF8Fallback,
         /*expected_module_substrings_fn=*/
         [](const se::DeviceDescription& device_description) {
           if (device_description.gpu_compute_capability()
                   .cuda_compute_capability()
                   ->IsAtLeastHopper()) {
             return std::vector<std::string>{
                 "custom_call_target=\"__cublas$lt$matmul$f8\"",
                 "\"selected_algorithm\":\"0\""};
           }
           return std::vector<std::string>{
               "custom_call_target=\"__cublas$gemm\"",
               "\"selected_algorithm\":\"-1\""};
         },
         /*expected_backend_name=*/"CUBLAS_FISSION"},
        {"ScaledDotFusion_Cublas", kScaledDotFusionHlo,
         &GetCublasRewriterPipeline, &CreateCublasBackend,
         /*expected_module_substrings_fn=*/
         [](const se::DeviceDescription& device_description) {
           return std::vector<std::string>{
               "custom_call_target=\"__cublas$gemm\"",
               "\"selected_algorithm\":\"-1\""};
         },
         /*expected_backend_name=*/"CUBLAS_FISSION"},
        {"TritonFusion_CustomKernel", kTritonFusionHlo,
         &GetCustomKernelRewriterPipeline, &CreateCustomKernelBackend,
         /*expected_module_substrings_fn=*/
         [](const se::DeviceDescription& device_description) {
           return std::vector<std::string>{
               "\"kind\":\"__custom_fusion\"",
           };
         },
         /*expected_backend_name=*/"CUSTOM_KERNEL_FISSION"},
    }),
    [](const ::testing::TestParamInfo<FissionTest::ParamType>& info) {
      return info.param.test_name;
    });

class CublasFissionBackendTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions debug_options_;
  se::Platform* platform_;
  se::StreamExecutor* stream_executor_;
  std::unique_ptr<Compiler> compiler_;
  Compiler::GpuTargetConfig target_config_;
  se::DeviceDescription device_description_;
  GpuAliasInfo alias_info_;
  std::unique_ptr<FissionBackend> fission_backend_;
  mlir::MLIRContext mlir_context_;

  CublasFissionBackendTest()
      : platform_(PlatformUtil::GetDefaultPlatform().value()),
        stream_executor_(platform_->ExecutorForDevice(0).value()),
        compiler_(Compiler::GetForPlatform(platform_->id()).value()),
        target_config_(stream_executor_),
        device_description_(stream_executor_->GetDeviceDescription()),
        alias_info_(device_description_),
        fission_backend_(std::make_unique<FissionBackend>(
            &debug_options_, compiler_.get(), &target_config_,
            CreateCublasBackend(stream_executor_, &debug_options_,
                                compiler_.get(), &target_config_),
            GetCublasRewriterPipeline(device_description_), &alias_info_,
            &mlir_context_, stream_executor_)) {}
};

TEST_F(CublasFissionBackendTest, ApplyConfigRemovesComputation) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kTritonFusionHlo));
  EXPECT_EQ(module->computation_count(), 2);
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                       fission_backend_->GetDefaultConfig(*fusion));
  EXPECT_THAT(fission_backend_->ApplyConfig(*fusion, *config), IsOk());
  EXPECT_EQ(module->computation_count(), 1);
}

TEST_F(CublasFissionBackendTest, CublasFallbackForTf32Tf32F32X3Algorithm) {
  if (device_description_.gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "Not supported on ROCm";
  }
  constexpr absl::string_view kHloDotFusionWithAlgorithm = R"(
    HloModule module

    computation {
      p0 = f32[1024,1024] parameter(0)
      p1 = f32[1024,1024] parameter(1)
      ROOT r = f32[1024,1024] dot(p0, p1),
        algorithm=$0,
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }

    ENTRY main {
      p0 = f32[1024,1024] parameter(0)
      p1 = f32[1024,1024] parameter(1)
      ROOT computation = f32[1024,1024] fusion(f32[1024,1024] p0,f32[1024,1024] p1),
        kind=kCustom,
        calls=computation
    }
  )";

  std::string hlo_string =
      absl::Substitute(kHloDotFusionWithAlgorithm, "dot_tf32_tf32_f32_x3");
  auto module_statusor = ParseAndReturnVerifiedModule(hlo_string);
  ASSERT_TRUE(module_statusor.ok());
  auto module = std::move(module_statusor).value();

  auto configs_statusor = fission_backend_->GetSupportedConfigs(
      *module->entry_computation()->root_instruction());
  ASSERT_TRUE(configs_statusor.ok());
  auto configs = std::move(configs_statusor).value();

  auto hasCublasConfig = [&](const auto& configs) {
    for (const auto& config : configs) {
      AutotuneResult::GemmKey gemm_key;
      if (config->UnpackTo(&gemm_key)) {
        return true;
      }
    }
    return false;
  };

  EXPECT_TRUE(hasCublasConfig(configs))
      << "There is dot_algorithm_rewrite that supports fallback to cublas "
         "implementation for dot_tf32_tf32_f32_x3.";
}

TEST_F(CublasFissionBackendTest, CublasFallbackForBf16Bf16F32Algorithm) {
  constexpr absl::string_view kHloDotFusionWithAlgorithm = R"(
    HloModule module

    computation {
      p0 = f32[1024,1024] parameter(0)
      p1 = f32[1024,1024] parameter(1)
      ROOT r = f32[1024,1024] dot(p0, p1),
        algorithm=$0,
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }

    ENTRY main {
      p0 = f32[1024,1024] parameter(0)
      p1 = f32[1024,1024] parameter(1)
      ROOT computation = f32[1024,1024] fusion(f32[1024,1024] p0,f32[1024,1024] p1),
        kind=kCustom,
        calls=computation
    }
  )";

  std::string hlo_string =
      absl::Substitute(kHloDotFusionWithAlgorithm, "dot_bf16_bf16_f32");
  auto module_statusor = ParseAndReturnVerifiedModule(hlo_string);
  ASSERT_TRUE(module_statusor.ok());
  auto module = std::move(module_statusor).value();

  auto configs_statusor = fission_backend_->GetSupportedConfigs(
      *module->entry_computation()->root_instruction());
  ASSERT_TRUE(configs_statusor.ok());
  auto configs = std::move(configs_statusor).value();

  auto hasCublasConfig = [&](const auto& configs) {
    for (const auto& config : configs) {
      AutotuneResult::GemmKey gemm_key;
      if (config->UnpackTo(&gemm_key)) {
        return true;
      }
    }
    return false;
  };

  const auto& comp = device_description_.gpu_compute_capability();

  if (!comp.IsRocm()) {
    auto cc = comp.cuda_compute_capability();
    // Ampere (8.0) and newer should have fallback.
    if (cc->IsAtLeastAmpere()) {
      EXPECT_TRUE(hasCublasConfig(configs))
          << "There should be a cublas fallback for dot_bf16_bf16_f32";
    } else {
      EXPECT_FALSE(hasCublasConfig(configs));
    }
  } else {
    // ROCm
    EXPECT_TRUE(hasCublasConfig(configs));
  }
}

class CustomKernelFissionBackendTest : public HloHardwareIndependentTestBase {
 public:
  // Static helper to create the Custom Kernel rewriter pipeline.
  static std::unique_ptr<HloPassPipeline> GetCustomKernelRewriterPipeline(
      const se::DeviceDescription& device_description) {
    auto pipeline = std::make_unique<HloPassPipeline>("fission_pipeline");
    pipeline->AddPass(
        std::make_unique<CustomKernelFusionRewriter>(&device_description));
    return pipeline;
  }

  // Static helper to create a CustomKernelBackend.
  static std::unique_ptr<GpuCodegenBackend> CreateCustomKernelBackend(
      se::StreamExecutor* stream_executor, const DebugOptions* debug_options,
      Compiler* compiler, const Compiler::GpuTargetConfig* target_config) {
    return std::make_unique<CustomKernelBackend>(stream_executor, debug_options,
                                                 compiler, target_config);
  }

 protected:
  DebugOptions debug_options_;
  se::Platform* platform_;
  se::StreamExecutor* stream_executor_;
  std::unique_ptr<Compiler> compiler_;
  Compiler::GpuTargetConfig target_config_;
  se::DeviceDescription device_description_;
  GpuAliasInfo alias_info_;
  std::unique_ptr<FissionBackend> fission_backend_;
  mlir::MLIRContext mlir_context_;

  CustomKernelFissionBackendTest()
      : platform_(PlatformUtil::GetDefaultPlatform().value()),
        stream_executor_(platform_->ExecutorForDevice(0).value()),
        compiler_(Compiler::GetForPlatform(platform_->id()).value()),
        target_config_(stream_executor_),
        device_description_(stream_executor_->GetDeviceDescription()),
        alias_info_(device_description_),
        fission_backend_(std::make_unique<FissionBackend>(
            &debug_options_, compiler_.get(), &target_config_,
            CreateCustomKernelBackend(stream_executor_, &debug_options_,
                                      compiler_.get(), &target_config_),
            GetCustomKernelRewriterPipeline(device_description_), &alias_info_,
            &mlir_context_, stream_executor_)) {}
};

TEST_F(CustomKernelFissionBackendTest, GetSupportedConfigsForUpcastGemm) {
  if (device_description_.gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloWithUpcast));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      fission_backend_->GetSupportedConfigs(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsOkAndHolds(testing::SizeIs(Gt(0))));
}

TEST_F(CustomKernelFissionBackendTest,
       GeneratesTwoConfigsForUpcastGemmWithPrologue) {
  if (device_description_.gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloWithUpcastPrologueK64));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      fission_backend_->GetSupportedConfigs(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsOkAndHolds(SizeIs(2)));
}

TEST_F(CustomKernelFissionBackendTest,
       GeneratesOneConfigForUpcastGemmWithPrologue) {
  if (device_description_.gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHloWithUpcastPrologueK128));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      fission_backend_->GetSupportedConfigs(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsOkAndHolds(SizeIs(1)));
}

TEST_F(CustomKernelFissionBackendTest,
       GeneratesConfigForUpcastGemmWithPrologueAndEpilogue) {
  if (device_description_.gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHloWithUpcastPrologueEpilogueK64));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      fission_backend_->GetSupportedConfigs(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsOkAndHolds(SizeIs(2)));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

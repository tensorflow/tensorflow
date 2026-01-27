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
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/autotune_results.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/cublas.h"
#include "xla/backends/gpu/autotuner/custom_kernel.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/gpu/transforms/custom_kernel_fusion_rewriter.h"
#include "xla/service/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/service/gpu/transforms/scaled_dot_rewriter.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace {

using absl_testing::IsOk;
using absl_testing::IsOkAndHolds;
using ::testing::HasSubstr;

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
  std::vector<std::string> expected_module_substrings;
  std::string expected_backend_name;
};

class FissionTest : public HloHardwareIndependentTestBase,
                    public ::testing::WithParamInterface<FissionTestParams> {
 public:
  // Static helper to create the Cublas rewriter pipeline.
  static std::unique_ptr<HloPassPipeline> GetCublasRewriterPipeline(
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

  // Static helper to create the Custom Kernel rewriter pipeline.
  static std::unique_ptr<HloPassPipeline> GetCustomKernelRewriterPipeline(
      const se::DeviceDescription& device_description) {
    auto pipeline = std::make_unique<HloPassPipeline>("fission_pipeline");
    pipeline->AddPass(
        std::make_unique<CustomKernelFusionRewriter>(&device_description));
    return pipeline;
  }

  // Static helper to create a CublasBackend.
  static std::unique_ptr<GpuCodegenBackend> CreateCublasBackend(
      se::StreamExecutor* stream_executor, const DebugOptions* debug_options,
      Compiler* compiler, const Compiler::GpuTargetConfig* target_config) {
    return std::make_unique<CublasBackend>(stream_executor, debug_options,
                                           compiler, target_config);
  }

  // Static helper to create a CublasBackend.
  static std::unique_ptr<GpuCodegenBackend> CreateCublasBackendWiithF8Fallback(
      se::StreamExecutor* stream_executor, const DebugOptions* debug_options,
      Compiler* compiler, const Compiler::GpuTargetConfig* target_config) {
    return std::make_unique<CublasBackend>(stream_executor, debug_options,
                                           compiler, target_config,
                                           /*enable_f8_fallback=*/true);
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
  NVPTXCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  se::DeviceDescription device_description_;
  std::unique_ptr<HloPassPipeline> rewriter_pipeline_;
  std::unique_ptr<GpuCodegenBackend> base_codegen_backend_;
  GpuAliasInfo alias_info_;
  std::unique_ptr<FissionBackend> fission_backend_;
  mlir::MLIRContext mlir_context_;

  FissionTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()),
        target_config_(stream_executor_),
        device_description_(stream_executor_->GetDeviceDescription()),
        rewriter_pipeline_(GetParam().pipeline_factory(device_description_)),
        base_codegen_backend_(GetParam().backend_factory(
            stream_executor_, &debug_options_, &compiler_, &target_config_)),
        alias_info_(device_description_),
        fission_backend_(std::make_unique<FissionBackend>(
            &debug_options_, &compiler_, &target_config_,
            std::move(base_codegen_backend_), std::move(rewriter_pipeline_),
            &alias_info_, &mlir_context_, stream_executor_)) {}
};

class CublasFissionBackendTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions debug_options_;
  NVPTXCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  se::DeviceDescription device_description_;
  std::unique_ptr<HloPassPipeline> rewriter_pipeline_;
  std::unique_ptr<GpuCodegenBackend> base_codegen_backend_;
  GpuAliasInfo alias_info_;
  std::unique_ptr<FissionBackend> fission_backend_;
  mlir::MLIRContext mlir_context_;

  CublasFissionBackendTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()),
        target_config_(stream_executor_),
        device_description_(stream_executor_->GetDeviceDescription()),
        rewriter_pipeline_(
            FissionTest::GetCublasRewriterPipeline(device_description_)),
        base_codegen_backend_(FissionTest::CreateCublasBackend(
            stream_executor_, &debug_options_, &compiler_, &target_config_)),
        alias_info_(device_description_),
        fission_backend_(std::make_unique<FissionBackend>(
            &debug_options_, &compiler_, &target_config_,
            std::move(base_codegen_backend_), std::move(rewriter_pipeline_),
            &alias_info_, &mlir_context_, stream_executor_)) {}
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

TEST_P(FissionTest, CanCreateFissionBackend) {
  EXPECT_EQ(fission_backend_->name(), GetParam().expected_backend_name);
}

TEST_P(FissionTest, GetSupportedConfigs) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(GetParam().hlo_string));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      fission_backend_->GetSupportedConfigs(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsOkAndHolds(testing::SizeIs(1)));
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(GetParam().hlo_string));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_THAT(fission_backend_->GetDefaultConfig(*fusion), IsOk());
}

TEST_P(FissionTest, Compile) {
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(GetParam().hlo_string));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                       fission_backend_->GetDefaultConfig(*fusion));
  EXPECT_THAT(fission_backend_->ApplyConfig(*fusion, *config), IsOk());
  std::string module_str = module->ToString();
  for (const std::string& expected_substr :
       GetParam().expected_module_substrings) {
    EXPECT_THAT(module_str, HasSubstr(expected_substr));
  }
}

INSTANTIATE_TEST_SUITE_P(
    FissionTests, FissionTest,
    ::testing::ValuesIn<FissionTestParams>({
        {"TritonFusion_Cublas",
         kTritonFusionHlo,
         &FissionTest::GetCublasRewriterPipeline,
         &FissionTest::CreateCublasBackend,
         /*expected_module_substrings=*/
         {"custom_call_target=\"__cublas$gemm\"",
          "\"selected_algorithm\":\"-1\""},
         /*expected_backend_name=*/"Cublas_fission"},
        {"TritonFusion_CublasLt_F8",
         kF8TritonFusionHlo,
         &FissionTest::GetCublasRewriterPipeline,
         &FissionTest::CreateCublasBackendWiithF8Fallback,
         /*expected_module_substrings=*/
         {"custom_call_target=\"__cublas$lt$matmul$f8\"",
          "\"selected_algorithm\":\"0\""},
         /*expected_backend_name=*/"Cublas_fission"},
        {"TritonFusion_CustomKernel",
         kTritonFusionHlo,
         &FissionTest::GetCustomKernelRewriterPipeline,
         &FissionTest::CreateCustomKernelBackend,
         /*expected_module_substrings=*/
         {
             "\"kind\":\"__custom_fusion\"",
         },
         /*expected_backend_name=*/"CustomKernel_fission"},
        {"ScaledDotFusion_Cublas",
         kScaledDotFusionHlo,
         &FissionTest::GetCublasRewriterPipeline,
         &FissionTest::CreateCublasBackend,
         /*expected_module_substrings=*/
         {"custom_call_target=\"__cublas$gemm\"",
          "\"selected_algorithm\":\"-1\""},
         /*expected_backend_name=*/"Cublas_fission"},
    }),
    [](const ::testing::TestParamInfo<FissionTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace gpu
}  // namespace xla

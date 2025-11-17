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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/cublas.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

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

class CublasFissionTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions debug_options_;
  NVPTXCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  se::DeviceDescription device_description_;
  std::unique_ptr<HloPassPipeline> rewriter_pipeline_;
  std::unique_ptr<GpuCodegenBackend> cublas_backend_;
  std::unique_ptr<FissionBackend> fission_backend_;
  mlir::MLIRContext mlir_context_;
  SymbolicExprContext symbolic_expr_context_{&mlir_context_};

  std::unique_ptr<HloPassPipeline> GetCublasRewriterPipeline() {
    auto pipeline = std::make_unique<HloPassPipeline>("fission_pipeline");
    pipeline->AddPass(std::make_unique<DotAlgorithmRewriter>());
    for (GemmRewriterOptions::DType dtype :
         {GemmRewriterOptions::DType::kFp8Only,
          GemmRewriterOptions::DType::kNonFp8Only}) {
      auto gemm_rewriter = std::make_unique<GemmRewriter>(
          device_description_.gpu_compute_capability(),
          device_description_.runtime_version(), GemmRewriterOptions{dtype});
      pipeline->AddPass(std::move(gemm_rewriter));
    }
    return pipeline;
  }

  CublasFissionTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()),
        target_config_(stream_executor_),
        device_description_(stream_executor_->GetDeviceDescription()),
        rewriter_pipeline_(GetCublasRewriterPipeline()),
        cublas_backend_(std::make_unique<CublasBackend>(
            stream_executor_, &debug_options_, &compiler_, &target_config_)),
        fission_backend_(std::make_unique<FissionBackend>(
            &debug_options_, &compiler_, &target_config_,
            std::move(cublas_backend_), std::move(rewriter_pipeline_),
            &symbolic_expr_context_, stream_executor_)) {}
};

TEST_F(CublasFissionTest, CanCreateFissionBackend) {
  EXPECT_EQ(fission_backend_->name(), "Cublas_fission");
}

TEST_F(CublasFissionTest, GetSupportedConfigs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTritonFusionHlo));
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> configs =
      fission_backend_->GetSupportedConfigs(
          (*module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsOkAndHolds(testing::SizeIs(1)));
}

TEST_F(CublasFissionTest, GetDefaultConfig) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTritonFusionHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  EXPECT_THAT(fission_backend_->GetDefaultConfig(*fusion), IsOk());
}

TEST_F(CublasFissionTest, Compile) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTritonFusionHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          fission_backend_->GetDefaultConfig(*fusion));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                          fission_backend_->Compile(*fusion, *config));
  EXPECT_NE(executable, nullptr);
}

TEST_F(CublasFissionTest, ApplyConfig) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kTritonFusionHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          fission_backend_->GetDefaultConfig(*fusion));
  EXPECT_THAT(fission_backend_->ApplyConfig(*fusion, *config), IsOk());
  std::string module_str = module->ToString();
  EXPECT_THAT(module_str, HasSubstr("custom_call_target=\"__cublas$gemm\""));
  EXPECT_THAT(module_str, HasSubstr("\"selected_algorithm\":\"-1\""));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

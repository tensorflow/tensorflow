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

#include "xla/backends/cpu/autotuner/llvm_kernel_backend.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/cpu/autotuner/cpu_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/backend_config.pb.h"

namespace xla::cpu {
namespace {

constexpr absl::string_view kLlvmKernelHlo = R"(
    HloModule eltwise_f32_0

    llvm_fusion {
      p0 = f32[1024,1024] parameter(0)
      p1 = f32[1024,1024] parameter(1)
      add0 = f32[1024,1024] add(p0, p1)
      mul0 = f32[1024,1024] multiply(add0, add0)
      ROOT sub = f32[1024,1024] subtract(mul0, p0)
    }

    ENTRY e {
      p0 = f32[1024,1024] parameter(0)
      p1 = f32[1024,1024] parameter(1)
      ROOT %result = f32[1024,1024] fusion(%p0, %p1), kind=kLoop, 
        calls=llvm_fusion
    }
  )";

constexpr absl::string_view kLlvmKernelConcatenateHlo = R"(
    HloModule fusion.1

    ENTRY e {
      p0 = f32[3,2] parameter(0)
      p1 = f32[1,2] parameter(1)
      ROOT result = f32[4,2] concatenate(p0, p1), dimensions={0}
    }
)";

class LlvmKernelBackendTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(compiler_, CpuCodegenBackend::CreateBackendCompiler());
    ASSERT_OK_AND_ASSIGN(backend_, LlvmKernelBackend::Create(compiler_.get()));
  }

  std::unique_ptr<CodegenBackend> backend_;
  std::unique_ptr<Compiler> compiler_;
};

TEST_F(LlvmKernelBackendTest, NameTest) {
  EXPECT_THAT(backend_->name(), "llvm_kernel_backend");
}

TEST_F(LlvmKernelBackendTest, IsSupportedTest) {
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        ParseAndReturnVerifiedModule(kLlvmKernelConcatenateHlo));

    ASSERT_OK_AND_ASSIGN(auto configs,
                         backend_->GetSupportedConfigs(
                             *module->entry_computation()->root_instruction()));

    EXPECT_FALSE(configs.empty());
  }

  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(kLlvmKernelHlo));

    ASSERT_OK_AND_ASSIGN(auto configs,
                         backend_->GetSupportedConfigs(
                             *module->entry_computation()->root_instruction()));

    EXPECT_TRUE(configs.empty());
  }
}

TEST_F(LlvmKernelBackendTest, GetDefaultConfigTest) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kLlvmKernelConcatenateHlo));
  ASSERT_OK_AND_ASSIGN(auto config,
                       backend_->GetDefaultConfig(
                           *module->entry_computation()->root_instruction()));
  LlvmKernelBackend::Config llvm_kernel_config;
  ASSERT_TRUE(config->UnpackTo(&llvm_kernel_config));

  EXPECT_FALSE(llvm_kernel_config.disable_loop_unrolling());
  EXPECT_FALSE(llvm_kernel_config.slp_vectorizer_disabled());
  EXPECT_FALSE(llvm_kernel_config.optimize_for_size());
}

TEST_F(LlvmKernelBackendTest, GetSupportedConfigsTest) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kLlvmKernelConcatenateHlo));
  ASSERT_OK_AND_ASSIGN(auto configs,
                       backend_->GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  EXPECT_EQ(configs.size(), 8);
}

TEST_F(LlvmKernelBackendTest, CompileSupportedBackends) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kLlvmKernelConcatenateHlo));
  HloInstruction* instruction = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(auto configs,
                       backend_->GetSupportedConfigs(*instruction));
  for (auto& config : configs) {
    ASSERT_OK_AND_ASSIGN(auto executable,
                         backend_->Compile(*instruction, *config));
  }
}

TEST_F(LlvmKernelBackendTest, EnsureConfigIsApplied) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kLlvmKernelConcatenateHlo));
  HloInstruction* instruction = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(auto configs,
                       backend_->GetSupportedConfigs(*instruction));

  for (const auto& config : configs) {
    LlvmKernelBackend::Config llvm_kernel_config;
    ASSERT_TRUE(config->UnpackTo(&llvm_kernel_config));
    EXPECT_TRUE(backend_->ApplyConfig(*instruction, *config).ok());

    ASSERT_OK_AND_ASSIGN(auto instruction_backend_config,
                         instruction->backend_config<BackendConfig>());

    EXPECT_EQ(instruction_backend_config.llvm_kernel_options()
                  .disable_loop_unrolling(),
              llvm_kernel_config.disable_loop_unrolling());

    EXPECT_EQ(instruction_backend_config.llvm_kernel_options()
                  .slp_vectorizer_disabled(),
              llvm_kernel_config.slp_vectorizer_disabled());

    EXPECT_EQ(
        instruction_backend_config.llvm_kernel_options().optimize_for_size(),
        llvm_kernel_config.optimize_for_size());
  }
}

}  // namespace
}  // namespace xla::cpu

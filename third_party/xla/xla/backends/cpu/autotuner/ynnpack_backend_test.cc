/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/autotuner/ynnpack_backend.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/cpu/autotuner/cpu_codegen_backend.h"
#include "xla/backends/cpu/ynn_fusion_options.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/backend_config.pb.h"

namespace xla::cpu {
namespace {

constexpr absl::string_view kYnnFusionHlo = R"(
    HloModule ynn_eltwise

    ynn_fusion {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      ROOT add = f32[100] add(p0, p1)
    }

    ENTRY e {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      ROOT %fusion = f32[100] fusion(%p0, %p1), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    }
  )";

constexpr absl::string_view kNonYnnFusionHlo = R"(
    HloModule non_ynn_eltwise

    llvm_fusion {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      ROOT add = f32[100] add(p0, p1)
    }

    ENTRY e {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      ROOT %fusion = f32[100] fusion(%p0, %p1), kind=kLoop, calls=llvm_fusion
    }
  )";

class YnnpackBackendTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(compiler_, CpuCodegenBackend::CreateBackendCompiler());
    ASSERT_OK_AND_ASSIGN(backend_, YnnpackBackend::Create(compiler_.get()));
  }

  std::unique_ptr<CodegenBackend> backend_;
  std::unique_ptr<Compiler> compiler_;
};

TEST_F(YnnpackBackendTest, NameTest) {
  EXPECT_THAT(backend_->name(), kYnnpackBackendName);
}

TEST_F(YnnpackBackendTest, GetDefaultConfigTest) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kYnnFusionHlo));
  ASSERT_OK_AND_ASSIGN(auto config,
                       backend_->GetDefaultConfig(
                           *module->entry_computation()->root_instruction()));
  ASSERT_TRUE(config->has_ynn_fusion());
  xla::cpu::YnnFusionOptions ynn_config = config->ynn_fusion();

  EXPECT_TRUE(ynn_config.use_threadpool());
}

TEST_F(YnnpackBackendTest, GetSupportedConfigsTest) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kYnnFusionHlo));
  ASSERT_OK_AND_ASSIGN(auto configs,
                       backend_->GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  EXPECT_EQ(configs.size(), 2);  // use_threadpool=false, use_threadpool=true
}

TEST_F(YnnpackBackendTest, UnsupportedConfigsTest) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kNonYnnFusionHlo));
  ASSERT_OK_AND_ASSIGN(auto configs,
                       backend_->GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  EXPECT_TRUE(configs.empty());
}

TEST_F(YnnpackBackendTest, CompileSupportedBackends) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kYnnFusionHlo));
  HloInstruction* instruction = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(auto configs,
                       backend_->GetSupportedConfigs(*instruction));
  for (auto& config : configs) {
    ASSERT_OK_AND_ASSIGN(auto executable,
                         backend_->Compile(*instruction, *config));
  }
}

TEST_F(YnnpackBackendTest, EnsureConfigIsApplied) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kYnnFusionHlo));
  HloInstruction* instruction = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(auto configs,
                       backend_->GetSupportedConfigs(*instruction));

  for (const auto& config : configs) {
    ASSERT_TRUE(config->has_ynn_fusion());
    xla::cpu::YnnFusionOptions ynn_config = config->ynn_fusion();
    EXPECT_TRUE(backend_->ApplyConfig(*instruction, *config).ok());

    ASSERT_OK_AND_ASSIGN(auto instruction_backend_config,
                         instruction->backend_config<BackendConfig>());

    EXPECT_EQ(instruction_backend_config.fusion_config()
                  .ynn_fusion_options()
                  .use_threadpool(),
              ynn_config.use_threadpool());
  }
}

}  // namespace
}  // namespace xla::cpu

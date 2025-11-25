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

#include "xla/backends/cpu/autotuner/xnnpack_backend.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/cpu/autotuner/cpu_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

constexpr absl::string_view kXnnpackFusionHlo = R"(
    HloModule eltwise_f32_0

    xnn_fusion {
      p0 = f32[1024,1024] parameter(0)
      p1 = f32[1024,1024] parameter(1)
      add0 = f32[1024,1024] add(p0, p1)
      mul0 = f32[1024,1024] multiply(add0, add0)
      ROOT sub = f32[1024,1024] subtract(mul0, p0)
    }

    ENTRY e {
      p0 = f32[1024,1024] parameter(0)
      p1 = f32[1024,1024] parameter(1)
      ROOT %result = f32[1024,1024] fusion(%p0, %p1), kind=kCustom,
        calls=xnn_fusion,
        backend_config={"fusion_config": {"kind": "__xnn_fusion"}}
    }
  )";

class XnnpackBackendTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(compiler_,
                            CpuCodegenBackend::CreateBackendCompiler());
    TF_ASSERT_OK_AND_ASSIGN(backend_, XnnpackBackend::Create(compiler_.get()));
  }
  std::unique_ptr<CodegenBackend> backend_;
  std::unique_ptr<Compiler> compiler_;
};

TEST_F(XnnpackBackendTest, NameTest) {
  EXPECT_THAT(backend_->name(), "xnnpack");
}

TEST_F(XnnpackBackendTest, GetDefaultConfigTest) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kXnnpackFusionHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      auto config, backend_->GetDefaultConfig(
                       *module->entry_computation()->root_instruction()));
  XnnFusionOptions xnn_fusion_options;
  config->UnpackTo(&xnn_fusion_options);

  EXPECT_TRUE(xnn_fusion_options.use_threadpool());
}

TEST_F(XnnpackBackendTest, InvalidFusionKind) {
  constexpr absl::string_view bad_fusion_kind_hlo = R"(
    HloModule eltwise_f32_0

    not_xnn_fusion {
      p0 = f32[1024,1024] parameter(0)
      p1 = f32[1024,1024] parameter(1)
      add0 = f32[1024,1024] add(p0, p1)
      mul0 = f32[1024,1024] multiply(add0, add0)
      ROOT sub = f32[1024,1024] subtract(mul0, p0)
    }

    ENTRY e {
      p0 = f32[1024,1024] parameter(0)
      p1 = f32[1024,1024] parameter(1)
      ROOT %result = f32[1024,1024] fusion(%p0, %p1), kind=kCustom,
        calls=not_xnn_fusion,
        backend_config={fusion_config: {kind: "__not_xnn_fusion"}}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(bad_fusion_kind_hlo));
  auto config = backend_->GetDefaultConfig(
      *module->entry_computation()->root_instruction());

  EXPECT_THAT(config,
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Backend kind __not_xnn_fusion doesn't "
                                     "match expected kind __xnn_fusion.")));
}

TEST_F(XnnpackBackendTest, GetSupportedConfigsTest) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kXnnpackFusionHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      auto configs, backend_->GetSupportedConfigs(
                        *module->entry_computation()->root_instruction()));

  EXPECT_EQ(configs.size(), 2);
  XnnFusionOptions xnn_fusion_options0;
  configs[0]->UnpackTo(&xnn_fusion_options0);
  EXPECT_TRUE(xnn_fusion_options0.use_threadpool());
  XnnFusionOptions xnn_fusion_options1;
  configs[1]->UnpackTo(&xnn_fusion_options1);
  EXPECT_FALSE(xnn_fusion_options1.use_threadpool());
}

TEST_F(XnnpackBackendTest, CompileSupportedBackends) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kXnnpackFusionHlo));
  HloInstruction* fusion_instruction =
      module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(auto configs,
                          backend_->GetSupportedConfigs(*fusion_instruction));
  for (auto& config : configs) {
    TF_ASSERT_OK_AND_ASSIGN(auto executable,
                            backend_->Compile(*fusion_instruction, *config));
  }
}

TEST_F(XnnpackBackendTest, EnsureConfigIsApplied) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kXnnpackFusionHlo));
  HloInstruction* fusion_instruction =
      module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(auto configs,
                          backend_->GetSupportedConfigs(*fusion_instruction));

  for (const auto& config : configs) {
    XnnFusionOptions xnn_fusion_options;
    config->UnpackTo(&xnn_fusion_options);
    EXPECT_TRUE(backend_->ApplyConfig(*fusion_instruction, *config).ok());

    TF_ASSERT_OK_AND_ASSIGN(
        auto instruction_backend_config,
        fusion_instruction->backend_config<BackendConfig>());

    EXPECT_EQ(instruction_backend_config.fusion_config()
                  .xnn_fusion_options()
                  .use_threadpool(),
              xnn_fusion_options.use_threadpool());
  }
}

}  // namespace
}  // namespace xla::cpu

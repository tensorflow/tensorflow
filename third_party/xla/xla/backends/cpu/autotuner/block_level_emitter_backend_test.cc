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

#include "xla/backends/cpu/autotuner/block_level_emitter_backend.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/cpu/autotuner/cpu_codegen_backend.h"
#include "xla/backends/cpu/custom_fusion_configs.h"
#include "xla/codegen/xtile/xtile_config.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/backend_config.pb.h"

namespace xla::cpu {
namespace {

constexpr absl::string_view kFusionHlo = R"(
    HloModule eltwise_f32

    fusion_comp {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      ROOT add = f32[100] add(p0, p1)
    }

    ENTRY e {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      ROOT %fusion = f32[100] fusion(%p0, %p1), kind=kLoop, calls=fusion_comp
    }
  )";

constexpr absl::string_view kNonFusionHlo = R"(
    HloModule non_fusion

    ENTRY e {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      ROOT add = f32[100] add(p0, p1)
    }
  )";

constexpr absl::string_view kPreExistingConfigFusionHlo = R"(
    HloModule pre_existing_fusion

    fusion_comp {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      ROOT add = f32[100] add(p0, p1)
    }

    ENTRY e {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      ROOT %fusion = f32[100] fusion(%p0, %p1), kind=kCustom, calls=fusion_comp,
        backend_config={"fusion_config": {
            kind: "__xtile_fusion",
            block_level_fusion_config: {"num_warps": 4}}}
    }
  )";

class BlockLevelEmitterBackendTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(compiler_, CpuCodegenBackend::CreateBackendCompiler());
    ASSERT_OK_AND_ASSIGN(backend_,
                         BlockLevelEmitterBackend::Create(compiler_.get()));
  }

  std::unique_ptr<CodegenBackend> backend_;
  std::unique_ptr<Compiler> compiler_;
};

TEST_F(BlockLevelEmitterBackendTest, NameTest) {
  EXPECT_THAT(backend_->name(), kCpuBlockLevelEmitterBackendName);
}

TEST_F(BlockLevelEmitterBackendTest, BackendTest) {
  EXPECT_EQ(backend_->backend(), autotuner::Backend::BLOCK_LEVEL_EMITTER_CPU);
}

TEST_F(BlockLevelEmitterBackendTest, GetDefaultConfigTest) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kFusionHlo));
  ASSERT_OK_AND_ASSIGN(auto config,
                       backend_->GetDefaultConfig(
                           *module->entry_computation()->root_instruction()));
  ASSERT_TRUE(config->has_block_level());
  const xtile::BlockLevelFusionConfig& block_level_config =
      config->block_level();

  EXPECT_EQ(block_level_config.num_warps(), 1);
}

TEST_F(BlockLevelEmitterBackendTest, GetSupportedConfigsTest) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kFusionHlo));
  ASSERT_OK_AND_ASSIGN(auto configs,
                       backend_->GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  EXPECT_EQ(configs.size(), 1);
  ASSERT_TRUE(configs[0]->has_block_level());
  EXPECT_EQ(configs[0]->block_level().num_warps(), 1);
}

TEST_F(BlockLevelEmitterBackendTest, GetNonFusionConfigTest) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kNonFusionHlo));
  ASSERT_OK_AND_ASSIGN(auto configs,
                       backend_->GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  EXPECT_EQ(configs.size(), 0);
}

TEST_F(BlockLevelEmitterBackendTest, PreExistingConfigTest) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kPreExistingConfigFusionHlo));
  ASSERT_OK_AND_ASSIGN(auto configs,
                       backend_->GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  ASSERT_EQ(configs.size(), 1);
  ASSERT_TRUE(configs[0]->has_block_level());
  EXPECT_EQ(configs[0]->block_level().num_warps(), 4);
}

TEST_F(BlockLevelEmitterBackendTest, CompileSupportedBackends) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kFusionHlo));
  HloInstruction* instruction = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(auto configs,
                       backend_->GetSupportedConfigs(*instruction));
  for (auto& config : configs) {
    ASSERT_OK_AND_ASSIGN(auto executable,
                         backend_->Compile(*instruction, *config));
  }
}

TEST_F(BlockLevelEmitterBackendTest, EnsureConfigIsApplied) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kFusionHlo));
  HloInstruction* instruction = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(auto configs,
                       backend_->GetSupportedConfigs(*instruction));

  for (const auto& config : configs) {
    ASSERT_TRUE(config->has_block_level());
    const xtile::BlockLevelFusionConfig& block_level_config =
        config->block_level();
    EXPECT_TRUE(backend_->ApplyConfig(*instruction, *config).ok());

    ASSERT_OK_AND_ASSIGN(auto instruction_backend_config,
                         instruction->backend_config<BackendConfig>());

    EXPECT_EQ(instruction_backend_config.fusion_config().kind(),
              kXtileFusionKind);
    ASSERT_TRUE(instruction_backend_config.fusion_config()
                    .has_block_level_fusion_config());
    EXPECT_EQ(instruction_backend_config.fusion_config()
                  .block_level_fusion_config()
                  .num_warps(),
              block_level_config.num_warps());
  }
}

}  // namespace
}  // namespace xla::cpu

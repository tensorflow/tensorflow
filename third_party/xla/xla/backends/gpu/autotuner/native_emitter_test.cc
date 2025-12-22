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

#include "xla/backends/gpu/autotuner/native_emitter.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

const char kReductionFusionHlo[] = R"(
HloModule m

%func (lhs: f32[], rhs: f32[]) -> f32[] {
  %rhs = f32[] parameter(1)
  %lhs = f32[] parameter(0)
  ROOT %sum = f32[] add(%lhs, %rhs)
}

%fused_reduce.clone (param_0: f32[32,4096,2048]) -> f32[32,2048] {
  %param_0 = f32[32,4096,2048]{2,1,0} parameter(0)
  %c0 = f32[] constant(0)
  ROOT %reduce = f32[32,2048]{1,0} reduce(%param_0, %c0), dimensions={1},
    to_apply=%func
}

ENTRY %entry_computation (p0: f32[32,4096,2048]) -> f32[32,2048] {
  %p0 = f32[32,4096,2048]{2,1,0} parameter(0)
  ROOT %reduce_fusion = f32[32,2048]{1,0} fusion(%p0), kind=kInput,
    calls=%fused_reduce.clone
})";

const char kCustomFusionHlo[] = R"(
HloModule m

%fused_add_and_sub (p0: f32[32,16], p1: f32[32,16]) -> (f32[32,16], f32[32,16]) {
  %p0 = f32[32,16]{1,0} parameter(0)
  %p1 = f32[32,16]{1,0} parameter(1)
  %add = f32[32,16]{1,0} add(%p0, %p1)
  %sub = f32[32,16]{1,0} subtract(%p0, %p1)
  ROOT %tuple = (f32[32,16]{1,0}, f32[32,16]{1,0}) tuple(%add, %sub)
}

ENTRY %entry_computation (p0: f32[32,16], p1: f32[32,16]) -> (f32[32,16], f32[32,16]) {
  %p0 = f32[32,16]{1,0} parameter(0)
  %p1 = f32[32,16]{1,0} parameter(1)
  ROOT %reduce_fusion = (f32[32,16]{1,0}, f32[32,16]{1,0}) fusion(%p0, %p1), kind=kCustom,
    calls=%fused_add_and_sub,
    backend_config={ "fusion_backend_config": {
      "kind":"__triton",
      "block_level_fusion_config":{
        "num_warps":"1","output_tiles":[{"sizes":["1","4"]}],
        "num_ctas":1,"num_stages":1,"is_tma_allowed":false
      }
    }}
})";

class NativeEmitterBackendTest : public HloHardwareIndependentTestBase {
 protected:
  NativeEmitterBackendTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()),
        target_config_(stream_executor_),
        backend_(&debug_options_, &compiler_, &target_config_) {}

  DebugOptions debug_options_;
  NVPTXCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  NativeEmitterBackend backend_;
};

TEST_F(NativeEmitterBackendTest, GetDefaultConfig) {
  TF_ASSERT_OK_AND_ASSIGN(auto reduction_module,
                          ParseAndReturnVerifiedModule(kReductionFusionHlo));
  auto fusion = reduction_module->entry_computation()->root_instruction();
  // Call GetDefaultConfig on the fusion instruction.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*(fusion)));
  // Verify the returned config is a native emitter config.
  NativeEmitterBackendConfig native_emitter_config;
  ASSERT_TRUE(config->UnpackTo(&native_emitter_config));
}

TEST_F(NativeEmitterBackendTest, GetSupportedConfigs) {
  TF_ASSERT_OK_AND_ASSIGN(auto reduction_module,
                          ParseAndReturnVerifiedModule(kReductionFusionHlo));
  auto fusion = reduction_module->entry_computation()->root_instruction();
  // Call GetSupportedConfigs on the fusion instruction.
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*(fusion)));
  // There should only be a single config for the native emitter backend.
  ASSERT_EQ(configs.size(), 1);
  // Verify the returned config is a native emitter config.
  NativeEmitterBackendConfig native_emitter_config;
  ASSERT_TRUE(configs[0]->UnpackTo(&native_emitter_config));
}

TEST_F(NativeEmitterBackendTest,
       GetSupportedConfigsDoesNotSupportKCustomFusions) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kCustomFusionHlo));
  auto fusion_instruction = module->entry_computation()->root_instruction();
  // Call GetSupportedConfigs on the fusion instruction.
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                          backend_.GetSupportedConfigs(*(fusion_instruction)));
  // GetSupportedConfigs should return an empty vector as it doesn't support the
  // fusion.
  ASSERT_TRUE(configs.empty());
}

TEST_F(NativeEmitterBackendTest, ApplyConfig) {
  TF_ASSERT_OK_AND_ASSIGN(auto reduction_module,
                          ParseAndReturnVerifiedModule(kReductionFusionHlo));
  auto fusion = reduction_module->entry_computation()->root_instruction();
  // Call ApplyConfig on the fusion instruction.
  NativeEmitterBackendConfig native_emitter_config;
  BackendConfig config;
  config.PackFrom(native_emitter_config);
  ASSERT_THAT(backend_.ApplyConfig(*(fusion), config), absl_testing::IsOk());
  // Verify the fusion instruction is now a kInput fusion.
  ASSERT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kInput);
  // Verify the fusion instruction has a native emitter backend config.
  ASSERT_TRUE(fusion->has_backend_config());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_backend_config,
                          fusion->backend_config<GpuBackendConfig>());
  ASSERT_TRUE(gpu_backend_config.has_native_emitter_backend_config());
}

TEST_F(NativeEmitterBackendTest, ApplyConfigFailsForUnsupportedConfig) {
  TF_ASSERT_OK_AND_ASSIGN(auto reduction_module,
                          ParseAndReturnVerifiedModule(kReductionFusionHlo));
  auto fusion = reduction_module->entry_computation()->root_instruction();
  BlockLevelFusionConfig block_level_fusion_config;
  BackendConfig config;
  config.PackFrom(block_level_fusion_config);
  ASSERT_THAT(backend_.ApplyConfig(*(fusion), config),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(NativeEmitterBackendTest, CompileForDefaultConfig) {
  TF_ASSERT_OK_AND_ASSIGN(auto reduction_module,
                          ParseAndReturnVerifiedModule(kReductionFusionHlo));
  auto fusion = reduction_module->entry_computation()->root_instruction();
  // Call GetDefaultConfig on the fusion instruction.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend_.GetDefaultConfig(*(fusion)));
  // Attempt to compile the fusion using the retrieved backend config.
  auto maybe_executable = backend_.Compile(*fusion, *config);
  // Verify that compilation succeeded and returned a valid executable.
  EXPECT_THAT(maybe_executable, absl_testing::IsOk());
}

class MockCompiler : public Compiler {
 public:
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Executable>>, RunBackend,
              (std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
               const CompileOptions& options),
              (override));
  MOCK_METHOD(se::Platform::Id, PlatformId, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<HloModule>>, RunHloPasses,
              (std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
               const CompileOptions& options),
              (override));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<Executable>>>, Compile,
              (std::unique_ptr<HloModule> hlo_module,
               std::vector<se::StreamExecutor*> stream_execs,
               const CompileOptions& options),
              (override));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>>,
              CompileAheadOfTime,
              (std::unique_ptr<HloModule> hlo_module,
               const AotCompilationOptions& options),
              (override));
  MOCK_METHOD(HloCostAnalysis::ShapeSizeFunction, ShapeSizeBytesFunction, (),
              (const, override));
};

TEST_F(NativeEmitterBackendTest, CompileSetsIsAutotuningCompilationOption) {
  TF_ASSERT_OK_AND_ASSIGN(auto reduction_module,
                          ParseAndReturnVerifiedModule(kReductionFusionHlo));
  auto fusion = reduction_module->entry_computation()->root_instruction();
  MockCompiler mock_compiler;
  NativeEmitterBackend backend(&debug_options_, &mock_compiler,
                               &target_config_);
  // Call GetDefaultConfig on the fusion instruction.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                          backend.GetDefaultConfig(*(fusion)));
  EXPECT_CALL(
      mock_compiler,
      RunBackend(
          testing::_, testing::_,
          testing::Field(&Compiler::CompileOptions::embed_hlo_module, false)))
      .WillOnce(testing::Return(std::unique_ptr<Executable>()));
  // Attempt to compile the fusion using the retrieved backend config.
  EXPECT_THAT(backend.Compile(*fusion, *config), absl_testing::IsOk());
}

}  // namespace
}  // namespace gpu
}  // namespace xla

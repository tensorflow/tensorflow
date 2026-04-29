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

#include "xla/backends/gpu/autotuner/block_level_emitter.h"

#include <algorithm>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

using ::tsl::proto_testing::EqualsProto;

// Checks if any config has is_tma_allowed set to true.
bool AnyTmaAllowed(const std::vector<std::unique_ptr<BackendConfig>>& configs) {
  return std::any_of(configs.begin(), configs.end(), [](auto& config) {
    BlockLevelFusionConfig actual_config;
    if (!config->UnpackTo(&actual_config)) {
      return false;
    }
    return actual_config.is_tma_allowed();
  });
}

// Test fixture for the TritonBlockLevelFusionEmitterBackend.
//
// Inherits from HloHardwareIndependentTestBase to use XLA utilities like
// module parsing and verification. Sets up the backend with a mock compiler
// and default platform executor.
class TritonBlockLevelFusionEmitterBackendTest
    : public HloHardwareIndependentTestBase {
 protected:
  TritonBlockLevelFusionEmitterBackendTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()),
        target_config_(stream_executor_),
        backend_(&debug_options_, &compiler_,
                 compiler_.ShapeSizeBytesFunction(), &target_config_) {}

  DebugOptions debug_options_;
  NVPTXCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  BlockLevelEmitterBackend backend_;
};

// Verifies that GetDefaultConfig correctly parses and returns the
// BlockLevelFusionConfig embedded in the backend_config of a fusion
// instruction.
TEST_F(TritonBlockLevelFusionEmitterBackendTest, GetDefaultConfig_FromHlo) {
  // Parse an HLO module containing a kCustom Triton fusion with a backend
  // config that includes block-level tiling parameters.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m
%wrapped_transpose_computation {
  %param_0 = f32[16,64]{1,0} parameter(0)
  ROOT %transpose.3.1 = f32[64,16]{1,0} transpose(%param_0), dimensions={1,0}
}

ENTRY %main {
  %p0 = f32[16,64]{1,0} parameter(0), metadata={op_name="a"}
  ROOT %wrapped_transpose = f32[64,16]{1,0} fusion(%p0), kind=kCustom,
  calls=%wrapped_transpose_computation,
  metadata={op_name="a"},
  backend_config={
  "fusion_backend_config": {
    "kind": "__triton",
    "block_level_fusion_config": {
      "output_tiles": [
        {"sizes": ["4","16"]}
      ],
      "num_warps": "2",
      "num_ctas": 1,
      "num_stages": 1
    }
  }}
}
)"));

  // Call GetDefaultConfig on the root instruction (the fusion op).
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction())));
  // Verify that the returned config is indeed a BlockLevelFusionConfig.
  BlockLevelFusionConfig block_level_fusion_config;
  ASSERT_TRUE(config->UnpackTo(&block_level_fusion_config));
  // Check that the config matches the proto embedded in the instruction.
  EXPECT_THAT(block_level_fusion_config, EqualsProto(R"pb(
                output_tiles { sizes: 4 sizes: 16 }
                num_warps: 2
                num_ctas: 1
                num_stages: 1
              )pb"));
}

// Tests that GetDefaultConfig falls back to generating a default
// BlockLevelFusionConfig when the backend config does not specify
// a block_level_fusion_config.
//
// We do not test the exact contents of the config here as this is a call to the
// cost model, which has its own tests.
TEST_F(TritonBlockLevelFusionEmitterBackendTest, GetDefaultConfig_Fallback) {
  // Parse an HLO module with a fusion instruction lacking any backend config.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m
%wrapped_transpose_computation {
  %param_0 = f32[16,1,64]{2,1,0} parameter(0)
  ROOT %transpose.3.1 = f32[64,1,16]{2,1,0} transpose(%param_0), dimensions={2,1,0}
}

ENTRY %main {
  %p0 = f32[16,1,64]{2,1,0} parameter(0), metadata={op_name="a"}
  ROOT %wrapped_transpose = f32[64,1,16]{2,1,0} fusion(%p0), kind=kInput,
  calls=%wrapped_transpose_computation
}
)"));

  // Call GetDefaultConfig on the root instruction (the fusion op).
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction())));
  // Verify that the returned config is indeed a BlockLevelFusionConfig.
  BlockLevelFusionConfig block_level_fusion_config;
  ASSERT_TRUE(config->UnpackTo(&block_level_fusion_config));
  // Verify the config is reasonable.
  EXPECT_GE(block_level_fusion_config.output_tiles_size(), 1);
  EXPECT_GE(block_level_fusion_config.num_warps(), 1);
  EXPECT_GE(block_level_fusion_config.num_ctas(), 1);
  EXPECT_GE(block_level_fusion_config.num_stages(), 1);
}

// Tests that `ApplyConfig` correctly attaches a generated default
// BlockLevelFusionConfig to a fusion instruction.
TEST_F(TritonBlockLevelFusionEmitterBackendTest, ApplyConfig) {
  // Build and verify a simple HLO module containing a 2D transpose fusion.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m
%wrapped_transpose_computation {
  %param_0 = f32[16,64]{1,0} parameter(0)
  ROOT %transpose.3.1 = f32[64,16]{1,0} transpose(%param_0), dimensions={1,0}
}

ENTRY %main {
  %p0 = f32[16,64]{1,0} parameter(0), metadata={op_name="a"}
  ROOT %wrapped_transpose = f32[64,16]{1,0} fusion(%p0), kind=kInput,
  calls=%wrapped_transpose_computation
}
)"));

  HloInstruction* instr = module->entry_computation()->root_instruction();
  // Ask the backend to generate a default block-level fusion configuration
  // for this fusion operation.
  // Call GetDefaultConfig on the root instruction (the fusion op).
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction())));
  // Verify that the returned config is indeed a BlockLevelFusionConfig.
  BlockLevelFusionConfig block_level_fusion_config;
  ASSERT_TRUE(config->UnpackTo(&block_level_fusion_config));

  // Apply the generated config to the fusion instruction.
  EXPECT_THAT(backend_.ApplyConfig(*instr, *config), absl_testing::IsOk());
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_backend_config,
                          instr->backend_config<GpuBackendConfig>());
  // Ensure that the backend config on the instruction matches what was applied.
  EXPECT_THAT(
      gpu_backend_config.fusion_backend_config().block_level_fusion_config(),
      EqualsProto(block_level_fusion_config));
  EXPECT_EQ(gpu_backend_config.fusion_backend_config().kind(),
            kTritonFusionKind);
  EXPECT_EQ(instr->fusion_kind(), HloInstruction::FusionKind::kCustom);
}

TEST_F(TritonBlockLevelFusionEmitterBackendTest, Compile) {
  // Parse an HLO module containing a kCustom Triton fusion with a backend
  // config that includes block-level tiling parameters.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m
%wrapped_transpose_computation {
  %param_0 = f32[16,64]{1,0} parameter(0)
  ROOT %transpose.3.1 = f32[64,16]{1,0} transpose(%param_0), dimensions={1,0}
}

ENTRY %main {
  %p0 = f32[16,64]{1,0} parameter(0), metadata={op_name="a"}
  ROOT %wrapped_transpose = f32[64,16]{1,0} fusion(%p0), kind=kCustom,
  calls=%wrapped_transpose_computation,
  metadata={op_name="a"},
  backend_config={
  "fusion_backend_config": {
    "kind": "__triton",
    "block_level_fusion_config": {
      "output_tiles": [
        {"sizes": ["4","16"]}
      ],
      "num_warps": "2",
      "num_ctas": 1,
      "num_stages": 1
    }
  }}
}
)"));
  // Call GetDefaultConfig on the root instruction (the fusion op).
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction())));
  // Attempt to compile the root instruction using the retrieved backend config.
  absl::StatusOr<std::unique_ptr<Executable>> executable = backend_.Compile(
      *(module->entry_computation()->root_instruction()), *config);
  // Verify that compilation succeeded and returned a valid executable.
  EXPECT_THAT(executable, absl_testing::IsOk());
}

TEST_F(TritonBlockLevelFusionEmitterBackendTest,
       CompileThroughCostModelConfig) {
  // Parse an HLO module without any assigned backend config.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m
%wrapped_transpose_computation {
  %param_0 = f32[16,64]{1,0} parameter(0)
  ROOT %transpose.3.1 = f32[64,16]{1,0} transpose(%param_0), dimensions={1,0}
}

ENTRY %main {
  %p0 = f32[16,64]{1,0} parameter(0)
  ROOT %wrapped_transpose = f32[64,16]{1,0} fusion(%p0), kind=kCustom,
  calls=%wrapped_transpose_computation
}
)"));
  // Call GetDefaultConfig on the root instruction (the fusion op).
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction())));
  // Attempt to compile the root instruction using the retrieved backend config.
  absl::StatusOr<std::unique_ptr<Executable>> executable = backend_.Compile(
      *(module->entry_computation()->root_instruction()), *config);
  // Verify that compilation succeeded and returned a valid executable.
  EXPECT_THAT(executable, absl_testing::IsOk());
}

}  // namespace gpu
}  // namespace xla

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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

using ::tsl::proto_testing::EqualsProto;

// Test fixture for the TritonBlockLevelFusionEmitterBackend.
//
// Inherits from HloHardwareIndependentTestBase to use XLA utilities like
// module parsing and verification. Sets up the backend with a mock compiler
// and default platform executor.
class TritonBlockLevelFusionEmitterBackendTest
    : public HloHardwareIndependentTestBase {
 protected:
  TritonBlockLevelFusionEmitterBackendTest()
      : backend_(PlatformUtil::GetDefaultPlatform()
                     .value()
                     ->ExecutorForDevice(0)
                     .value(),
                 &debug_options_, &compiler_) {}

  DebugOptions debug_options_;
  NVPTXCompiler compiler_;
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
  ASSERT_EQ(config->GetDescriptor(), BlockLevelFusionConfig::GetDescriptor())
      << "Config is not a BlockLevelFusionConfig";
  // TODO: Use DownCastMessage when :protobuf_lite target is available in OSS.
  const BlockLevelFusionConfig* block_level_fusion_config =
      dynamic_cast<const BlockLevelFusionConfig*>(config.get());
  ASSERT_NE(block_level_fusion_config, nullptr);
  // Check that the config matches the proto embedded in the instruction.
  EXPECT_THAT(*block_level_fusion_config, EqualsProto(R"pb(
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
// The HLO module contains a fusion instruction with a Triton backend config,
// but without the detailed block_level_fusion_config settings. This test
// verifies that the backend creates a reasonable default config.
TEST_F(TritonBlockLevelFusionEmitterBackendTest, GetDefaultConfig_Fallback) {
  // Parse an HLO module with a fusion instruction having a Triton backend
  // config that lacks an explicit block_level_fusion_config.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m
%wrapped_transpose_computation {
  %param_0 = f32[16,1,64]{2,1,0} parameter(0)
  ROOT %transpose.3.1 = f32[64,1,16]{2,1,0} transpose(%param_0), dimensions={2,1,0}
}

ENTRY %main {
  %p0 = f32[16,1,64]{2,1,0} parameter(0), metadata={op_name="a"}
  ROOT %wrapped_transpose = f32[64,1,16]{2,1,0} fusion(%p0), kind=kCustom,
  calls=%wrapped_transpose_computation,
  metadata={op_name="a"},
  backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)"));

  // Call GetDefaultConfig on the root instruction (the fusion op).
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction())));
  // Verify that the returned config is indeed a BlockLevelFusionConfig.
  ASSERT_EQ(config->GetDescriptor(), BlockLevelFusionConfig::GetDescriptor())
      << "Config is not a BlockLevelFusionConfig";
  const BlockLevelFusionConfig* block_level_fusion_config =
      dynamic_cast<const BlockLevelFusionConfig*>(config.get());
  ASSERT_NE(block_level_fusion_config, nullptr);
  // Verify that the default config contains default tiles sizes for the
  // dimensions of the input.
  EXPECT_THAT(*block_level_fusion_config, EqualsProto(R"pb(
    output_tiles { sizes: 16 sizes: 1 sizes: 16 }
    num_warps: 1
    num_ctas: 1
    num_stages: 1
  )pb"));
}

// Tests that GetDefaultConfig correctly handles shapes containing zero-sized
// dimensions.
//
// The HLO module defines a fusion instruction with an input tensor that has a
// zero-sized dimension (dimension size 0). The backend config specifies a
// Triton fusion kind but does not include a block-level fusion config. This
// test verifies that the default config is generated correctly and handles
// zero-sized dimensions by preserving them in the output tile sizes.
TEST_F(TritonBlockLevelFusionEmitterBackendTest,
       GetDefaultConfig_Fallback_ZeroDim) {
  // Parse an HLO module with a fusion instruction that has a zero-sized
  // dimension in the input shape.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m
%wrapped_transpose_computation {
  %param_0 = f32[5,0,10]{2,1,0} parameter(0)
  ROOT %transpose.3.1 = f32[10,0,5]{2,1,0} transpose(%param_0), dimensions={2,1,0}
}

ENTRY %main {
  %p0 = f32[5,0,10]{2,1,0} parameter(0), metadata={op_name="a"}
  ROOT %wrapped_transpose = f32[10,0,5]{2,1,0} fusion(%p0), kind=kCustom,
  calls=%wrapped_transpose_computation,
  metadata={op_name="a"},
  backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)"));

  // Call GetDefaultConfig on the root instruction (the fusion op).
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction())));
  // Verify that the returned config is indeed a BlockLevelFusionConfig.
  ASSERT_EQ(config->GetDescriptor(), BlockLevelFusionConfig::GetDescriptor())
      << "Config is not a BlockLevelFusionConfig";
  const BlockLevelFusionConfig* block_level_fusion_config =
      dynamic_cast<const BlockLevelFusionConfig*>(config.get());
  ASSERT_NE(block_level_fusion_config, nullptr);
  // Verify the default output tile sizes:
  // - The tile size for the dimension with size 10 is 16
  // - The zero-sized dimension remains zero
  // - The tile size for the dimension with size 5 is 8.
  // Also verify default tuning parameters: 1 warp, 1 CTA, 1 stage.
  EXPECT_THAT(*block_level_fusion_config, EqualsProto(R"pb(
    output_tiles { sizes: 16 sizes: 0 sizes: 8 }
    num_warps: 1
    num_ctas: 1
    num_stages: 1
  )pb"));
}

// Tests that GetDefaultConfig correctly generates default block-level fusion
// configurations for a fusion instruction that returns a tuple of two array
// shapes.
TEST_F(TritonBlockLevelFusionEmitterBackendTest,
       GetDefaultConfig_Fallback_tuple2) {
  // Parse and verify an HLO module with a fusion instruction that returns a
  // tuple of two array shapes.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m
%wrapped_transpose_computation {
  %param_0 = f32[16,64]{1,0} parameter(0)
  %param_1 = f32[32,12]{1,0} parameter(1)
  %transpose.3.1 = f32[64,16]{1,0} transpose(%param_0), dimensions={1,0}
  %transpose.4.1 = f32[12,32]{1,0} transpose(%param_1), dimensions={1,0}
  ROOT %tu = (f32[64,16]{1,0}, f32[12,32]{1,0}) tuple(%transpose.3.1, %transpose.4.1)
}

ENTRY %main {
  %p0 = f32[16,64]{1,0} parameter(0), metadata={op_name="a"}
  %p1 = f32[32,12]{1,0} parameter(1), metadata={op_name="b"}
  ROOT %wrapped_transpose = (f32[64,16]{1,0}, f32[12,32]{1,0}) fusion(%p0, %p1), kind=kCustom,
  calls=%wrapped_transpose_computation,
  metadata={op_name="ab"},
  backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)"));

  // Call GetDefaultConfig on the root instruction (the fusion op).
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction())));
  // Verify that the returned config is indeed a BlockLevelFusionConfig.
  ASSERT_EQ(config->GetDescriptor(), BlockLevelFusionConfig::GetDescriptor())
      << "Config is not a BlockLevelFusionConfig";
  const BlockLevelFusionConfig* block_level_fusion_config =
      dynamic_cast<const BlockLevelFusionConfig*>(config.get());
  ASSERT_NE(block_level_fusion_config, nullptr);
  // Check that the config correctly includes tiling info for both tuple
  // elements
  EXPECT_THAT(*block_level_fusion_config, EqualsProto(R"pb(
    output_tiles { sizes: 16 sizes: 16 }
    output_tiles { sizes: 16 sizes: 16 }
    num_warps: 1
    num_ctas: 1
    num_stages: 1
  )pb"));
}

}  // namespace gpu
}  // namespace xla

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

#include "xla/backends/gpu/autotuner/tensor_ir.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Not;

constexpr char kElementwiseFusionHlo[] = R"(
HloModule m

%fused_add (p0: f32[32,16], p1: f32[32,16]) -> f32[32,16] {
  %p0 = f32[32,16]{1,0} parameter(0)
  %p1 = f32[32,16]{1,0} parameter(1)
  ROOT %add = f32[32,16]{1,0} add(%p0, %p1)
}

ENTRY %entry_computation (p0: f32[32,16], p1: f32[32,16]) -> f32[32,16] {
  %p0 = f32[32,16]{1,0} parameter(0)
  %p1 = f32[32,16]{1,0} parameter(1)
  ROOT %fusion = f32[32,16]{1,0} fusion(%p0, %p1), kind=kLoop, calls=%fused_add
})";

constexpr char kReductionFusionHlo[] = R"(
HloModule m

%func (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %sum = f32[] add(%lhs, %rhs)
}

%fused_reduce (p0: f32[32,4096]) -> f32[32] {
  %p0 = f32[32,4096]{1,0} parameter(0)
  %c0 = f32[] constant(0)
  ROOT %reduce = f32[32]{0} reduce(%p0, %c0), dimensions={1}, to_apply=%func
}

ENTRY %entry_computation (p0: f32[32,4096]) -> f32[32] {
  %p0 = f32[32,4096]{1,0} parameter(0)
  ROOT %fusion = f32[32]{0} fusion(%p0), kind=kInput, calls=%fused_reduce
})";

// Integer fusions exercise the signedness bridge: StableHLO uses signless
// `i32`, while nv_tensor_ir requires signed `si32`.
constexpr char kElementwiseS32FusionHlo[] = R"(
HloModule m

%fused_mul (p0: s32[32,16], p1: s32[32,16]) -> s32[32,16] {
  %p0 = s32[32,16]{1,0} parameter(0)
  %p1 = s32[32,16]{1,0} parameter(1)
  ROOT %mul = s32[32,16]{1,0} multiply(%p0, %p1)
}

ENTRY %entry_computation (p0: s32[32,16], p1: s32[32,16]) -> s32[32,16] {
  %p0 = s32[32,16]{1,0} parameter(0)
  %p1 = s32[32,16]{1,0} parameter(1)
  ROOT %fusion = s32[32,16]{1,0} fusion(%p0, %p1), kind=kLoop, calls=%fused_mul
})";

// `pad` isn't part of the opcode set the TensorIR emitter supports.
constexpr char kUnsupportedOpFusionHlo[] = R"(
HloModule m

%fused_pad (p0: f32[4]) -> f32[6] {
  %p0 = f32[4]{0} parameter(0)
  %c0 = f32[] constant(0)
  ROOT %pad = f32[6]{0} pad(%p0, %c0), padding=1_1
}

ENTRY %entry_computation (p0: f32[4]) -> f32[6] {
  %p0 = f32[4]{0} parameter(0)
  ROOT %fusion = f32[6]{0} fusion(%p0), kind=kLoop, calls=%fused_pad
})";

class TensorIrBackendTest : public HloHardwareIndependentTestBase {
 protected:
  TensorIrBackendTest()
      : platform_(PlatformUtil::GetDefaultPlatform().value()),
        stream_executor_(platform_->ExecutorForDevice(0).value()),
        target_config_(stream_executor_),
        compiler_(Compiler::GetForPlatform(platform_->id()).value()),
        backend_(&debug_options_, compiler_.get(), &target_config_) {
    // The backend only supports Hopper and newer. Pin the target so that these
    // tests do not depend on which GPU the test machine happens to have;
    // `IsSupportedReturnsFalseForPreHopper` covers the other side.
    target_config_.device_description.set_cuda_compute_capability(
        se::CudaComputeCapability::Hopper());
  }

  DebugOptions debug_options_;
  se::Platform* platform_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  std::unique_ptr<Compiler> compiler_;
  TensorIrBackend backend_;
};

TEST_F(TensorIrBackendTest, IsSupportedReturnsTrueForSupportedFusion) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kElementwiseFusionHlo));
  EXPECT_TRUE(
      backend_.IsSupported(*module->entry_computation()->root_instruction()));
}

TEST_F(TensorIrBackendTest, IsSupportedReturnsFalseForNonFusion) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule m
ENTRY %entry_computation (p0: f32[4], p1: f32[4]) -> f32[4] {
  %p0 = f32[4]{0} parameter(0)
  %p1 = f32[4]{0} parameter(1)
  ROOT %add = f32[4]{0} add(%p0, %p1)
})"));
  EXPECT_FALSE(
      backend_.IsSupported(*module->entry_computation()->root_instruction()));
}

TEST_F(TensorIrBackendTest, IsSupportedReturnsFalseForPreHopper) {
  target_config_.device_description.set_cuda_compute_capability(
      se::CudaComputeCapability::Ampere());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kElementwiseFusionHlo));
  EXPECT_FALSE(
      backend_.IsSupported(*module->entry_computation()->root_instruction()));
}

TEST_F(TensorIrBackendTest, IsSupportedReturnsFalseForUnsupportedOpcode) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kUnsupportedOpFusionHlo));
  EXPECT_FALSE(
      backend_.IsSupported(*module->entry_computation()->root_instruction()));
}

TEST_F(TensorIrBackendTest, GetDefaultConfigReturnsTensorIrConfig) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kElementwiseFusionHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                       backend_.GetDefaultConfig(
                           *module->entry_computation()->root_instruction()));

  // GetDefaultConfig returns the top-ranked config from GetSupportedConfigs.
  ASSERT_TRUE(config->has_tensor_ir());
  EXPECT_THAT(config->tensor_ir().tile_size(), Not(IsEmpty()));
}

TEST_F(TensorIrBackendTest, GetDefaultConfigFailsForUnsupportedInstruction) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kUnsupportedOpFusionHlo));
  EXPECT_THAT(backend_.GetDefaultConfig(
                  *module->entry_computation()->root_instruction()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TensorIrBackendTest, GetSupportedConfigsReturnsRankedTilings) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kReductionFusionHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  // GetSupportedConfigs asks SelectBestTilings for at most 10 configs.
  EXPECT_THAT(configs, Not(IsEmpty()));
  EXPECT_LE(configs.size(), 10);
  for (const std::unique_ptr<BackendConfig>& config : configs) {
    ASSERT_TRUE(config->has_tensor_ir());
    EXPECT_THAT(config->tensor_ir().tile_size(), Not(IsEmpty()));
  }
}

// If the signless `i32` to signed `si32` bridge regressed, legalization would
// fail and `GetSupportedConfigs` would return an error instead of configs.
TEST_F(TensorIrBackendTest, GetSupportedConfigsHandlesSignedIntegers) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kElementwiseS32FusionHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));

  EXPECT_THAT(configs, Not(IsEmpty()));
  for (const std::unique_ptr<BackendConfig>& config : configs) {
    ASSERT_TRUE(config->has_tensor_ir());
    EXPECT_THAT(config->tensor_ir().tile_size(), Not(IsEmpty()));
  }
}

TEST_F(TensorIrBackendTest, GetSupportedConfigsReturnsEmptyForUnsupported) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kUnsupportedOpFusionHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));
  EXPECT_THAT(configs, IsEmpty());
}

TEST_F(TensorIrBackendTest, ApplyConfigSetsFusionBackendConfig) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kElementwiseFusionHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  BackendConfig config;
  config.mutable_tensor_ir()->add_tile_size(8);
  config.mutable_tensor_ir()->add_tile_size(16);
  config.mutable_tensor_ir()->set_reduction_tile_size(64);

  ASSERT_THAT(backend_.ApplyConfig(*fusion, config), IsOk());

  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_backend_config,
                       fusion->backend_config<GpuBackendConfig>());
  const FusionBackendConfig& backend_config =
      gpu_backend_config.fusion_backend_config();
  EXPECT_EQ(backend_config.kind(), kTensorIrFusionKind);
  EXPECT_THAT(backend_config.tensor_ir_fusion_config().tile_size(),
              ElementsAre(8, 16));
  EXPECT_EQ(backend_config.tensor_ir_fusion_config().reduction_tile_size(), 64);
  EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kCustom);
}

TEST_F(TensorIrBackendTest, ApplyConfigFailsForWrongConfigType) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kElementwiseFusionHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  BackendConfig config;
  config.mutable_gemm()->set_algorithm(1);

  EXPECT_THAT(backend_.ApplyConfig(*fusion, config),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TensorIrBackendTest, ApplyConfigFailsForUnsupportedInstruction) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kUnsupportedOpFusionHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  BackendConfig config;
  config.mutable_tensor_ir()->add_tile_size(8);

  EXPECT_THAT(backend_.ApplyConfig(*fusion, config),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla::gpu

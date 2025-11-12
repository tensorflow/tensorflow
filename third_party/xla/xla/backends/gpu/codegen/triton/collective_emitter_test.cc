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

#include "xla/backends/gpu/codegen/triton/collective_emitter.h"

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/codegen/triton/fusion.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::testing::Optional;
using ::tsl::proto_testing::EqualsProto;

struct ModuleWithFusion {
  std::unique_ptr<HloModule> module;

  const HloFusionInstruction* FusionInstr() const {
    return Cast<HloFusionInstruction>(
        module->entry_computation()->root_instruction());
  }
  HloFusionInstruction* MutableFusionInstr() {
    return Cast<HloFusionInstruction>(
        module->entry_computation()->root_instruction());
  }
};

struct ModuleWithEmitter : public ModuleWithFusion {
  mlir::MLIRContext mlir_context;
  SymbolicExprContext symbolic_expr_context{&mlir_context};
  std::optional<HloFusionAnalysis> analysis;
  std::unique_ptr<TritonFusion> emitter;

  explicit ModuleWithEmitter(std::unique_ptr<HloModule> module_arg)
      : ModuleWithFusion{std::move(module_arg)} {}
};

class CollectiveBlockLevelConfigTest : public HloHardwareIndependentTestBase {
 public:
  CollectiveBlockLevelConfigTest()
      : device_info_{TestGpuDeviceInfo::RTXH100SXMDeviceInfo()} {}

  absl::StatusOr<ModuleWithFusion> BuildModuleWithFusion(
      const Shape& shape) const {
    const std::string module_str = GetModuleStr(shape);
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                        ParseAndReturnVerifiedModule(module_str));
    const HloInstruction* instr = hlo_query::GetFirstInstructionWithOpcode(
        *module->entry_computation(), HloOpcode::kAllReduceStart);
    std::unique_ptr<HloModule> module_with_fusion =
        NewModuleWithFusion(instr, HloInstruction::FusionKind::kLoop);
    module_with_fusion->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_unsupported_use_all_reduce_one_shot_kernel(true);
    return ModuleWithFusion{std::move(module_with_fusion)};
  }

 protected:
  static std::string GetModuleStr(const Shape& shape) {
    return absl::StrFormat(R"(
      HloModule test
      apply_op {
        x = f32[] parameter(0)
        y = f32[] parameter(1)
        ROOT apply_op = f32[] add(x, y)
      }

      ENTRY test_computation {
        param_0 = %1$s parameter(0)
        all-reduce-start = %1$s all-reduce-start(param_0), to_apply=apply_op, replica_groups={{0,1}}
        ROOT all-reduce-done = %1$s all-reduce-done(all-reduce-start)
      }
    )",
                           shape.ToString());
  }

  const se::DeviceDescription device_info_;
};

class CollectiveEmitterTest : public CollectiveBlockLevelConfigTest {
 public:
  absl::StatusOr<std::unique_ptr<ModuleWithEmitter>> BuildModuleWithEmitter(
      const Shape& shape, const se::DeviceDescription& device_info) const {
    TF_ASSIGN_OR_RETURN(ModuleWithFusion module_with_fusion,
                        BuildModuleWithFusion(shape));
    TF_ASSIGN_OR_RETURN(
        bool collective_fusion_config_set,
        TrySetGpuBackendConfigForCollective(
            device_info_, module_with_fusion.MutableFusionInstr()));
    if (!collective_fusion_config_set) {
      return absl::InternalError(
          "Failed to set collective fusion config. "
          "TrySetGpuBackendConfigForCollective returned false.");
    }
    auto result = std::make_unique<ModuleWithEmitter>(
        std::move(module_with_fusion.module));
    result->analysis =
        HloFusionAnalysis::Create(*result->FusionInstr(), device_info);
    std::unique_ptr<FusionInterface> fusion_emitter =
        GetFusionEmitter(PreBufferAssignmentFusionInfo{*result->analysis},
                         &result->symbolic_expr_context);
    TritonFusion* triton_emitter =
        dynamic_cast<TritonFusion*>(fusion_emitter.get());
    TF_RET_CHECK(triton_emitter != nullptr);
    fusion_emitter.release();
    result->emitter = absl::WrapUnique(triton_emitter);
    return std::move(result);
  }
};

struct AllReduceBlockLevelConfigTestCase {
  std::string test_name;
  Shape shape;
  std::string expected_proto;

  // Teach gTest how to print the test case.
  [[maybe_unused]] friend void PrintTo(
      const AllReduceBlockLevelConfigTestCase& test_case, std::ostream* os) {
    *os << "{test_name: " << test_case.test_name
        << " shape: " << test_case.shape.ToString()
        << " expected_proto: " << test_case.expected_proto << "}";
  }
};

class CollectiveEmitterParameterizedTest
    : public CollectiveBlockLevelConfigTest,
      public ::testing::WithParamInterface<AllReduceBlockLevelConfigTestCase> {
};

TEST_P(CollectiveEmitterParameterizedTest, AllReduceBlockLevelConfig) {
  const auto& param = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(const auto module_with_fusion,
                          BuildModuleWithFusion(param.shape));
  TF_ASSERT_OK_AND_ASSIGN(const auto block_level_config,
                          GetCollectiveBlockLevelFusionConfig(
                              device_info_, module_with_fusion.FusionInstr()));
  EXPECT_THAT(block_level_config, Optional(EqualsProto(param.expected_proto)));
}

INSTANTIATE_TEST_SUITE_P(
    CollectiveEmitterParameterizedTestInstantiation,
    CollectiveEmitterParameterizedTest,
    ::testing::Values(AllReduceBlockLevelConfigTestCase{
                          /* .test_name = */ "F32_65536",
                          /* .shape = */ ShapeUtil::MakeShape(F32, {65536}),
                          /* .expected_proto = */ R"pb(
                            num_warps: 16
                            num_ctas: 1
                            num_stages: 1
                            output_tiles { sizes: 4096 }
                          )pb"},
                      AllReduceBlockLevelConfigTestCase{
                          /* .test_name= */ "F32_200_100",
                          /* .shape= */ ShapeUtil::MakeShape(F32, {200, 100}),
                          /* .expected_proto= */ R"pb(
                            num_warps: 16
                            num_ctas: 1
                            num_stages: 1
                            output_tiles { sizes: 256 sizes: 16 }
                          )pb"}),
    [](const ::testing::TestParamInfo<
        CollectiveEmitterParameterizedTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST_F(CollectiveEmitterTest, AllReduceWithTritonGetLaunchConfig) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleWithEmitter> result_ptr,
      BuildModuleWithEmitter(ShapeUtil::MakeShape(F32, {65536}), device_info_));
  auto& result = *result_ptr;
  const TritonFusion* triton_fusion = result.emitter.get();
  ASSERT_NE(triton_fusion, nullptr);
  auto const launch_config = triton_fusion->GetLaunchConfig();
  ASSERT_NE(launch_config, std::nullopt);
  EXPECT_EQ(launch_config->launch_dimensions.num_blocks(), 16);
  EXPECT_EQ(launch_config->launch_dimensions.num_threads_per_block(), 512);
}

}  // namespace

}  // namespace xla::gpu

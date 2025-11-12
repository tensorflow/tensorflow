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
#include <ostream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
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
}  // namespace

}  // namespace xla::gpu

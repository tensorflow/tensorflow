/* Copyright 2023 The OpenXLA Authors.
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

#include "xla/hlo/experimental/auto_sharding/auto_sharding.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_option.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_util.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/buffer_value.h"
#include "xla/service/hlo_alias_analysis.h"
#include "xla/service/hlo_memory_scheduler.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/hlo_value.h"
#include "xla/statusor.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace spmd {
namespace {
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::FieldsAre;
using ::testing::IsEmpty;
using ::testing::IsFalse;
using ::testing::IsTrue;
using ::testing::Pair;
using ::testing::ResultOf;
using ::testing::UnorderedElementsAre;

using DummyAutoShardingTest = HloTestBase;

TEST_F(DummyAutoShardingTest, ReplicatedShardingDummy) {
  constexpr absl::string_view kHloString = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0)
  %param1 = f32[5,7,11,13]{3,2,1,0} parameter(1)
  %add = f32[5,7,11,13]{3,2,1,0} add(%param0, %param1)
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%add)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, DummyAutoSharding().Run(module.get()));
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "param0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{replicated}"));
}

class AutoShardingTest : public HloTestBase {
 protected:
  const absl::string_view kDotHloString = R"(
HloModule module
ENTRY matmul {
  parameter.1 = f32[32,64]{1,0} parameter(0)
  parameter.2 = f32[64,128]{1,0} parameter(1)
  ROOT root = f32[32,128]{1,0} dot(parameter.1, parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  const absl::string_view kAddHloString = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[16,32,64]{2,1,0} parameter(0)
  %param1 = f32[16,32,64]{2,1,0} parameter(1)
  ROOT root = f32[16,32,64]{2,1,0} add(%param0, %param1)
})";
  void RunMatMulAutoShardingWithOptions(
      AutoShardingOption option, size_t expected_num_tiles,
      size_t expected_sharded_dimensions = 1) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                            ParseAndReturnVerifiedModule(kDotHloString));
    RunAutoShardingWithOptions(module.get(), option, expected_num_tiles,
                               expected_sharded_dimensions);
  }

  void RunAddAutoShardingWithOptions(AutoShardingOption option,
                                     size_t expected_num_tiles,
                                     size_t expected_sharded_dimensions = 1) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                            ParseAndReturnVerifiedModule(kAddHloString));
    RunAutoShardingWithOptions(module.get(), option, expected_num_tiles,
                               expected_sharded_dimensions);
  }

  void RunAutoShardingWithOptions(HloModule* module, AutoShardingOption option,
                                  size_t expected_num_tiles,
                                  size_t expected_sharded_dimensions = 1) {
    TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module));
    EXPECT_TRUE(changed);
    // To simplify the test, only checking the sharding of root.
    auto* root = FindInstruction(module, "root");
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->sharding().NumTiles(), expected_num_tiles);
    EXPECT_EQ(VectorGreaterThanOneElementCount(
                  root->sharding().tile_assignment().dimensions(),
                  root->sharding().ReplicateOnLastTileDim()),
              expected_sharded_dimensions);
  }

  void RunMatMulAutoShardingWithOptionsExpectFail(AutoShardingOption option) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                            ParseAndReturnVerifiedModule(kDotHloString));
    RunAutoShardingWithOptionsExpectFail(module.get(), option);
  }

  void RunAutoShardingWithOptionsExpectFail(HloModule* module,
                                            AutoShardingOption option) {
    EXPECT_FALSE(AutoSharding(option).Run(module).ok());
  }

  void RunMatMulAutoShardingWithOptionsNoDeviceIds(
      AutoShardingOption option, std::vector<int64_t> expected_tile,
      bool expeted_last_dim_replicate = false) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                            ParseAndReturnVerifiedModule(kDotHloString));
    RunAutoShardingWithOptionsNoDeviceIds(module.get(), option, expected_tile,
                                          expeted_last_dim_replicate);
  }

  void RunAutoShardingWithOptionsNoDeviceIds(HloModule* module,
                                             AutoShardingOption option,
                                             std::vector<int64_t> expected_tile,
                                             bool expeted_last_dim_replicate) {
    TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module));
    EXPECT_TRUE(changed);
    // To simplify the test, only checking the sharding of root.
    HloInstruction* root = FindInstruction(module, "root");
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->sharding().ReplicateOnLastTileDim(),
              expeted_last_dim_replicate);
    EXPECT_THAT(root->sharding().tile_assignment().dimensions(),
                ElementsAreArray(expected_tile));
  }
};

TEST_F(AutoShardingTest, MemoryBudgetTest) {
  auto compute_memory_budget_lower_bound =
      [](const HloModule& module, int64_t num_devices,
         const absl::flat_hash_map<std::string, std::vector<HloSharding>>&
             preserved_shardings = {}) -> absl::StatusOr<int64_t> {
    auto size_fn = [](const BufferValue& buffer) {
      return spmd::GetBytes(buffer.shape());
    };
    TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                        ScheduleModule(&module, size_fn,
                                       ComputationSchedulerToModuleScheduler(
                                           DFSMemoryScheduler),
                                       /* execution_threads */ {}));
    const HloComputation* entry_computation = module.entry_computation();
    std::unique_ptr<HloAliasAnalysis> alias_analysis =
        HloAliasAnalysis::Run(&module).value();

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloLiveRange> hlo_live_range,
        HloLiveRange::Run(schedule, *alias_analysis, entry_computation));
    absl::flat_hash_map<const HloValue*, HloLiveRange::TimeBound>&
        buffer_live_ranges = hlo_live_range->buffer_live_ranges();
    spmd::LivenessSet liveness_set(hlo_live_range->schedule_end_time() + 1);
    for (const auto& [hlo_value, live_range] : buffer_live_ranges) {
      for (spmd::LivenessIdx i = live_range.start; i <= live_range.end; ++i) {
        liveness_set[i].push_back(hlo_value);
      }
    }
    absl::flat_hash_set<const HloInstruction*> instructions_to_shard(
        module.entry_computation()->instructions().begin(),
        module.entry_computation()->instructions().end());
    return spmd::MemoryBudgetLowerBound(module, instructions_to_shard,
                                        liveness_set, *alias_analysis,
                                        num_devices, preserved_shardings);
  };

  constexpr absl::string_view kHloString = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[16384,16384]{0,1} parameter(0)
  %param1 = f32[16384,16384]{0,1} parameter(1)
  %add = f32[16384,16384]{0,1} add(%param0, %param1)
  ROOT %copy = f32[16384,16384]{0,1} copy(%add)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding partial_sharding,
                          ParseSharding("{devices=[64,1]<=[64]}"));
  TF_ASSERT_OK_AND_ASSIGN(
      int64_t partial_mesh_64x1_budget_lower_bound,
      compute_memory_budget_lower_bound(*module, /* num_devices */ 64));
  for (HloInstruction* ins : module->entry_computation()->instructions()) {
    ins->set_sharding(partial_sharding);
  }
  TF_ASSERT_OK_AND_ASSIGN(
      int64_t full_mesh_64x8_budget_lower_bound,
      compute_memory_budget_lower_bound(*module, /* num_devices */ 512));
  CHECK_LT(full_mesh_64x8_budget_lower_bound,
           partial_mesh_64x1_budget_lower_bound)
      << "The memory budget lower bound per device should be lower with a "
         "larger number of devices. Instead, the bound was "
      << partial_mesh_64x1_budget_lower_bound << " bytes for 64 devices and "
      << full_mesh_64x8_budget_lower_bound << " bytes for 512 devices.";
}

TEST_F(AutoShardingTest, DISABLED_ElementWiseOperator) {
  constexpr absl::string_view kHloString = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[128,128]{0,1} parameter(0)
  %param1 = f32[128,128]{0,1} parameter(1)
  %add = f32[128,128]{0,1} add(%param0, %param1)
  ROOT %copy = f32[128,128]{0,1} copy(%add)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "param0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,2]0,2,1,3}"));
}

TEST_F(AutoShardingTest, Unsupported3DShardingTest) {
  constexpr absl::string_view kHloString = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[32,32,32,32] parameter(0)
  %param1 = f32[32,32,32,32] parameter(1)
  %add = f32[32,32,32,32] add(%param0, %param1), sharding={devices=[2,2,1,2]<=[8]}
  ROOT %copy = f32[32,32,32,32] copy(%add)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  // The case of a fleet HLO when run with try_multiple_mesh_shapes = true
  option.device_mesh_shape = {2, 4};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  EXPECT_DEATH(auto status = AutoSharding(option).Run(module.get()),
               ".*too many axes.*");
}

TEST_F(AutoShardingTest, NDIterativeSolveTest) {
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %elementwise {
  param = s32[512,3084]{1,0} parameter(0), sharding={devices=[256,1]0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23,8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255}
  sharding_call = s32[512,3084]{1,0} custom-call(param), custom_call_target="Sharding", sharding={devices=[256,1]<=[256]}
  ROOT slice = s32[512,2048]{1,0} slice(sharding_call), slice={[0:512], [0:2048]}
})";

  AutoShardingOption option;
  option.enable = true;
  option.solve_nd_sharding_iteratively = true;
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings;
  option.device_mesh_shape = {16, 16};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
  HloInstruction* slice = FindInstruction(module.get(), "slice");
  EXPECT_NE(slice, nullptr);
  EXPECT_THAT(slice, op::Sharding("{devices=[256,1]<=[256]}"));
}

TEST_F(AutoShardingTest, SliceDeviceMeshTest) {
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %elementwise {
  param = s32[512,3084]{1,0} parameter(0)
  slice = s32[512,2048]{1,0} slice(param), slice={[0:512], [0:2048]}
  ROOT copy = s32[512,2048]{1,0} copy(slice)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, AutoSharding(/* option */ AutoShardingOption{
                                     .enable = true,
                                     .solve_nd_sharding_iteratively = true,
                                     .device_mesh_shape = {2, 2},
                                     .device_mesh_alpha = {1.0, 1.0},
                                     .device_mesh_beta = {0.01, 1.0}})
                        .Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
  const HloInstruction* slice = FindInstruction(module.get(), "slice");
  ASSERT_NE(slice, nullptr);
  EXPECT_THAT(
      slice,
      AnyOf(op::Sharding("{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}"),
            op::Sharding("{devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}")));
}

TEST_F(AutoShardingTest, SliceMixedUserShardingTest) {
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %elementwise {
  param = s32[512,3084]{1,0} parameter(0), sharding={devices=[4,1]0,2,1,3}
  slice = s32[512,2048]{1,0} slice(param), slice={[0:512], [0:2048]}
  ROOT copy = s32[512,2048]{1,0} copy(slice)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      AutoSharding(
          /* option */ {
              .enable = true,
              .preserve_shardings =
                  AutoShardingOption::PreserveShardingsType::kKeepAllShardings,
              .solve_nd_sharding_iteratively = true,
              .device_mesh_shape = {2, 2},
              .device_mesh_ids = {0, 2, 1, 3},
              .device_mesh_alpha = {1.0, 1.0},
              .device_mesh_beta = {0.01, 1.0}})
          .Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);

  std::vector<HloInstruction*> instructions =
      module->entry_computation()->MakeInstructionPostOrder();
  EXPECT_THAT(instructions,
              Each(ResultOf(
                  [](const HloInstruction* ins) { return ins->has_sharding(); },
                  IsTrue())));
  EXPECT_THAT(instructions, Each(op::Sharding("{devices=[4,1]0,2,1,3}")));
}

TEST_F(AutoShardingTest, UserShardingTest) {
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %elementwise {
  concatenate.76306 = bf16[1,4096,8,256]{3,2,1,0} parameter(0)
  constant.15158 = bf16[] constant(0)
  pad.70 = bf16[1,4352,8,256]{3,2,1,0} pad(concatenate.76306, constant.15158), padding=0_0x0_256x0_0x0_0, sharding={devices=[1,1,128,1]<=[128]}
  ROOT copy.45 = bf16[1,4352,8,256]{3,2,1,0} copy(pad.70)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      AutoSharding(
          /* option */ AutoShardingOption{
              .enable = true,
              .preserve_shardings =
                  AutoShardingOption::PreserveShardingsType::kKeepAllShardings,
              .device_mesh_shape = {128, 1},
              .device_mesh_alpha = {1.0, 1.0},
              .device_mesh_beta = {0.01, 1.0}})
          .Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
}

TEST_F(AutoShardingTest, RngBitGeneratorArrayInput) {
  constexpr absl::string_view kHloString = R"(
HloModule rng_bit_generator

ENTRY %RngBitGenerator (p0: u64[2]) -> (u64[2], u32[16,16]) {
  %p0 = u64[2]{0} parameter(0)
  ROOT %rand = (u64[2]{0}, u32[16,16]{1,0}) rng-bit-generator(u64[2]{0} %p0), algorithm=rng_three_fry
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {1.0, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "p0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{replicated}"));
}

TEST_F(AutoShardingTest, SPMDShardToFullShapeTest) {
  constexpr absl::string_view kHloString = R"(
HloModule rng_bit_generator

add.6.clone {
  y.13 = bf16[]{:T(256)} parameter(1)
  x.13 = bf16[]{:T(256)} parameter(0)
  ROOT add.9011 = bf16[]{:T(256)} add(x.13, y.13)
}

ENTRY main {
  input.1 = bf16[512,512]{1,0} parameter(0)
  custom-call.1 = bf16[512,512]{1,0} custom-call(input.1), custom_call_target="Sharding", sharding={devices=[4,4]<=[16]}
  custom-call.2 = bf16[128,128]{1,0} custom-call(custom-call.1), custom_call_target="SPMDFullToShardShape", sharding={manual}
  all-reduce.1 = bf16[128,128]{1,0} all-reduce(custom-call.2), channel_id=621, replica_groups={{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15}}, use_global_device_ids=true, to_apply=add.6.clone, frontend_attributes={from-cross-replica-sharding="true"}, backend_config={"flag_configs":[],"barrier_config":{"barrier_type":"CUSTOM","id":"9"},"scoped_memory_configs":[],"compute_type":"COMPUTE_TYPE_DEFAULT","device_type":"DEVICE_TYPE_INVALID","used_scoped_memory_configs":[]}
  custom-call.3 = bf16[512,512]{1,0} custom-call(all-reduce.1), custom_call_target="SPMDShardToFullShape", sharding={devices=[4,1,4]<=[16]last_tile_dim_replicate}
  ROOT copy.1 = copy(custom-call.3)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  // Check that custom call shardings are preserved despite us dropped user
  // shardings
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kRemoveAllShardings;
  option.enable = true;
  option.device_mesh_shape = {4, 4};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {1.0, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);

  const HloInstruction* custom_call2 =
      FindInstruction(module.get(), "custom-call.2");
  ASSERT_NE(custom_call2, nullptr);
  EXPECT_THAT(custom_call2, op::Sharding("{manual}"));

  const HloInstruction* custom_call3 =
      FindInstruction(module.get(), "custom-call.3");
  ASSERT_NE(custom_call3, nullptr);
  EXPECT_THAT(custom_call3,
              op::Sharding("{devices=[4,1,4]<=[16]last_tile_dim_replicate}"));

  // Auto-sharding rewrites Sharding custom-calls
  const HloInstruction* custom_call1 = custom_call2->operand(0);
  ASSERT_NE(custom_call1, nullptr);
  EXPECT_THAT(custom_call1, op::Sharding("{devices=[4,4]<=[16]}"));
}

TEST_F(AutoShardingTest, SPMDShardToFullShapeMultipleValidMeshShapeTest) {
  constexpr absl::string_view kHloString = R"(
HloModule rng_bit_generator

add.6.clone {
  y.13 = bf16[]{:T(256)} parameter(1)
  x.13 = bf16[]{:T(256)} parameter(0)
  ROOT add.9011 = bf16[]{:T(256)} add(x.13, y.13)
}

ENTRY main {
  input.1 = bf16[512,512]{1,0} parameter(0)
  custom-call.1 = bf16[512,512]{1,0} custom-call(input.1), custom_call_target="Sharding", sharding={devices=[4,4]<=[16]}
  custom-call.2 = bf16[128,128]{1,0} custom-call(custom-call.1), custom_call_target="SPMDFullToShardShape", sharding={manual}
  all-reduce.1 = bf16[128,128]{1,0} all-reduce(custom-call.2), channel_id=621, replica_groups={{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15}}, use_global_device_ids=true, to_apply=add.6.clone, frontend_attributes={from-cross-replica-sharding="true"}, backend_config={"flag_configs":[],"barrier_config":{"barrier_type":"CUSTOM","id":"9"},"scoped_memory_configs":[],"compute_type":"COMPUTE_TYPE_DEFAULT","device_type":"DEVICE_TYPE_INVALID","used_scoped_memory_configs":[]}
  reshape.1 = bf16[64,2,128]{2,1,0} reshape(bf16[128,128]{1,0} all-reduce.1)
  reshape.2 = bf16[64,256]{1,0} reshape(bf16[64,2,128]{2,1,0} reshape.1)
  custom-call.3 = bf16[512,512]{1,0} custom-call(reshape.2), custom_call_target="SPMDShardToFullShape", sharding={devices=[8,2]<=[16]}
  ROOT copy.1 = copy(custom-call.3)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kRemoveAllShardings;
  option.enable = true;
  option.try_multiple_mesh_shapes = false;
  option.device_mesh_shape = {4, 4};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {1.0, 1.0};
  EXPECT_DEATH(auto status = AutoSharding(option).Run(module.get()),
               "Auto-sharding cannot infer a single appropriate mesh shape for "
               "this HLO, and AutoShardingption::try_multiple_mesh_shapes is "
               "set to false. Please re-run with the option set to true.");
}

TEST_F(AutoShardingTest, RngBitGeneratorTupleInput) {
  constexpr absl::string_view kHloString = R"(
HloModule rng_bit_generator

ENTRY %RngBitGenerator {
  param.0 = u32[2]{0:T(128)} parameter(0)
  param.1 = u32[2]{0:T(128)} parameter(1)
  tuple.3 = (u32[2]{0:T(128)}, u32[2]{0:T(128)}) tuple(param.0, param.1)
  ROOT rng-bit-generator = u32[100,100]{1,0:T(8,128)} rng-bit-generator(tuple.3), algorithm=rng_default
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
  auto* param0 = FindInstruction(module.get(), "param.0");
  auto* param1 = FindInstruction(module.get(), "param.1");
  ASSERT_NE(param0, nullptr);
  ASSERT_NE(param0, nullptr);
  EXPECT_THAT(param0, op::Sharding("{replicated}"));
  EXPECT_THAT(param1, op::Sharding("{replicated}"));
}

TEST_F(AutoShardingTest, DotLHSTwoNonContractingDims) {
  constexpr absl::string_view kHloString = R"(
HloModule module
ENTRY %entry {
  %param0 = f32[4,256,64]{2,1,0} parameter(0)
  %param1 = f32[64,32]{0,1} parameter(1)
  %dot = f32[4,256,32]{2,1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[64,32]{0,1} %param1), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  ROOT %copy = f32[4,256,32]{2,1,0} copy(%dot)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(2) << module->ToString();
  EXPECT_TRUE(changed);
  auto* param0 = FindInstruction(module.get(), "param0");
  auto* param1 = FindInstruction(module.get(), "param1");
  auto* dot = FindInstruction(module.get(), "dot");
  ASSERT_NE(param0, nullptr);
  ASSERT_NE(param1, nullptr);
  ASSERT_NE(dot, nullptr);
  EXPECT_THAT(
      std::make_tuple(param0, param1, dot),
      AnyOf(
          FieldsAre(
              op::Sharding(
                  "{devices=[1,2,1,2]0,1,2,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[1,2,2]0,2,1,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[1,2,2]0,1,2,3}")),
          FieldsAre(
              op::Sharding(
                  "{devices=[1,2,1,2]0,2,1,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[1,2,2]0,2,1,3}")),
          FieldsAre(
              op::Sharding(
                  "{devices=[2,1,1,2]0,1,2,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[1,2,2]0,2,1,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[2,1,2]0,1,2,3}")),
          FieldsAre(
              op::Sharding(
                  "{devices=[2,1,1,2]0,2,1,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[2,1,2]0,2,1,3}"))));
}

TEST_F(AutoShardingTest, DotRHSTwoNonContractingDims) {
  constexpr absl::string_view kHloString = R"(
HloModule module
ENTRY %entry {
  %param0 = f32[4,256,32]{2,1,0} parameter(0)
  %param1 = f32[4,256,4,8]{1,3,2,0} parameter(1)
  %dot = f32[32,4,8]{2,1,0} dot(f32[4,256,32]{2,1,0} %param0, f32[4,256,4,8]{1,3,2,0} %param1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
  ROOT %copy = f32[32,4,8]{2,1,0} copy(%dot)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(2) << module->ToString();
  EXPECT_TRUE(changed);
  auto* param0 = FindInstruction(module.get(), "param0");
  auto* param1 = FindInstruction(module.get(), "param1");
  auto* dot = FindInstruction(module.get(), "dot");
  ASSERT_NE(param0, nullptr);
  ASSERT_NE(param1, nullptr);
  ASSERT_NE(dot, nullptr);
  EXPECT_THAT(
      std::make_tuple(param0, param1, dot),
      AnyOf(
          FieldsAre(op::Sharding(
                        "{devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}"),
                    op::Sharding(
                        "{devices=[1,1,2,1,2]0,2,1,3 last_tile_dim_replicate}"),
                    op::Sharding("{devices=[2,2,1]0,1,2,3}")),
          FieldsAre(op::Sharding(
                        "{devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}"),
                    op::Sharding(
                        "{devices=[1,1,1,2,2]0,2,1,3 last_tile_dim_replicate}"),
                    op::Sharding("{devices=[2,1,2]0,1,2,3}")),
          FieldsAre(op::Sharding(
                        "{devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}"),
                    op::Sharding(
                        "{devices=[1,1,1,2,2]0,1,2,3 last_tile_dim_replicate}"),
                    op::Sharding("{devices=[2,1,2]0,2,1,3}")),
          FieldsAre(op::Sharding(
                        "{devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}"),
                    op::Sharding(
                        "{devices=[1,1,2,1,2]0,1,2,3 last_tile_dim_replicate}"),
                    op::Sharding("{devices=[2,2,1]0,2,1,3}"))));
}

TEST_F(AutoShardingTest, DotTwoContractingDims) {
  constexpr absl::string_view kHloString = R"(
HloModule module
ENTRY %entry {
  %param0 = f32[4,256,64]{2,1,0} parameter(0)
  %param1 = f32[4,256,32]{2,1,0} parameter(1)
  %dot = f32[64,32]{1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[4,256,32]{2,1,0} %param1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
  ROOT %copy = f32[64,32]{1,0} copy(%dot)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(2) << module->ToString();
  EXPECT_TRUE(changed);
  auto* param0 = FindInstruction(module.get(), "param0");
  auto* param1 = FindInstruction(module.get(), "param1");
  auto* dot = FindInstruction(module.get(), "dot");
  ASSERT_NE(param0, nullptr);
  ASSERT_NE(param1, nullptr);
  ASSERT_NE(dot, nullptr);
  EXPECT_THAT(
      std::make_tuple(param0, param1, dot),
      AnyOf(FieldsAre(op::Sharding(
                          "{devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}"),
                      op::Sharding(
                          "{devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}"),
                      op::Sharding("{devices=[2,2]0,2,1,3}")),
            FieldsAre(op::Sharding(
                          "{devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}"),
                      op::Sharding(
                          "{devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}"),
                      op::Sharding("{devices=[2,2]0,1,2,3}"))));
}

TEST_F(AutoShardingTest, TwoMatmul) {
  constexpr absl::string_view kHloString = R"(
HloModule module
ENTRY twomatmul {
  parameter.1 = f32[64,64]{1,0} parameter(0)
  parameter.2 = f32[64,128]{1,0} parameter(1)
  dot.4 = f32[64,128]{1,0} dot(parameter.1, parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  parameter.3 = f32[128,64]{1,0} parameter(2)
  ROOT dot.5 = f32[64,64]{1,0} dot(dot.4, parameter.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.allow_recompute_heavy_op = false;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
  auto* param1 = FindInstruction(module.get(), "parameter.1");
  ASSERT_NE(param1, nullptr);
  EXPECT_THAT(param1,
              op::Sharding("{devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}"));
  auto* param2 = FindInstruction(module.get(), "parameter.2");
  ASSERT_NE(param2, nullptr);
  EXPECT_THAT(param2,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"));
  auto* param3 = FindInstruction(module.get(), "parameter.3");
  ASSERT_NE(param3, nullptr);
  EXPECT_THAT(param3,
              op::Sharding("{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}"));
  auto* dot4 = FindInstruction(module.get(), "dot.4");
  ASSERT_NE(dot4, nullptr);
  EXPECT_THAT(dot4, op::Sharding("{devices=[2,2]0,2,1,3}"));
  auto* dot5 = FindInstruction(module.get(), "dot.5");
  ASSERT_NE(dot5, nullptr);
  EXPECT_THAT(dot5,
              op::Sharding("{devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}"));

  // Test with replicated strategies on for dot
  TF_ASSERT_OK_AND_ASSIGN(module, ParseAndReturnVerifiedModule(kHloString));
  option.enable = true;
  option.allow_recompute_heavy_op = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(changed, AutoSharding(option).Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
  param1 = FindInstruction(module.get(), "parameter.1");
  param2 = FindInstruction(module.get(), "parameter.2");
  param3 = FindInstruction(module.get(), "parameter.3");
  dot4 = FindInstruction(module.get(), "dot.4");
  dot5 = FindInstruction(module.get(), "dot.5");
  ASSERT_NE(param1, nullptr);
  ASSERT_NE(param2, nullptr);
  ASSERT_NE(param3, nullptr);
  ASSERT_NE(dot4, nullptr);
  ASSERT_NE(dot5, nullptr);
  EXPECT_THAT(
      std::make_tuple(param1, param2, param3, dot4, dot5),
      AnyOf(
          FieldsAre(
              op::Sharding("{replicated}"), op::Sharding("{replicated}"),
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"),
              op::Sharding("{replicated}"),
              op::Sharding("{devices=[2,2]0,2,1,3}")),
          FieldsAre(
              op::Sharding("{replicated}"), op::Sharding("{replicated}"),
              op::Sharding("{devices=[1,2,2]0,2,1,3 last_tile_dim_replicate}"),
              op::Sharding("{replicated}"),
              op::Sharding("{devices=[2,2]0,1,2,3}"))));
}

TEST_F(AutoShardingTest, ProcessCustomCallShardings) {
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[6,3] parameter(0)
  %copy = f32[6,3] copy(%param0)
  %annotate = f32[6,3] custom-call(%copy), custom_call_target="Sharding",
    backend_config="unspecified_dims=[1]",
    sharding={devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}
  %copy.2 = f32[6,3] copy(%annotate)
  ROOT %copy.3 = f32[6,3] copy(%copy.2)
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  // %annotate's sharding is moved to %copy.
  auto* copy = FindInstruction(module.get(), "copy");
  ASSERT_NE(copy, nullptr);
  EXPECT_TRUE(copy->has_sharding());
  EXPECT_THAT(copy,
              op::Sharding("{devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}"));
}

TEST_F(AutoShardingTest, SaveAndRemoveShardingAnnotationKeepAll) {
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %entry (param0: f32[4,256,64], param1: f32[4,256,32]) -> f32[64,32] {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}
  %param1 = f32[4,256,32]{2,1,0} parameter(1), sharding={devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}
  %dot = f32[64,32]{1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[4,256,32]{2,1,0} %param1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}, sharding={devices=[2,2]0,1,2,3}
  ROOT %copy = f32[64,32]{1,0} copy(f32[64,32]{1,0} %dot), sharding={devices=[2,2]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  // Keep all user shardings
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings;
  absl::flat_hash_set<const HloInstruction*> instructions_to_shard(
      module->entry_computation()->instructions().begin(),
      module->entry_computation()->instructions().end());
  std::pair<absl::flat_hash_map<std::string, std::vector<HloSharding>>, bool>
      saved_shardings_result =
          AutoShardingImplementation(option).SaveAndRemoveShardingAnnotation(
              module.get(), instructions_to_shard,
              /* replicated_small_tensors */ {},
              /* execution_threads */ {});
  absl::flat_hash_map<std::string, std::vector<HloSharding>> saved_shardings =
      saved_shardings_result.first;
  bool changed = saved_shardings_result.second;
  EXPECT_FALSE(changed);
  std::vector<HloInstruction*> instructions =
      module->entry_computation()->MakeInstructionPostOrder();
  EXPECT_THAT(instructions,
              Each(ResultOf(
                  [](const HloInstruction* ins) { return ins->has_sharding(); },
                  IsTrue())));

  auto verified_parse_sharding = [](const absl::string_view sharding_str) {
    absl::StatusOr<HloSharding> sharding = ParseSharding(sharding_str);
    CHECK_OK(sharding);
    return *sharding;
  };

  EXPECT_THAT(
      saved_shardings,
      UnorderedElementsAre(
          Pair("param0",
               ElementsAre(verified_parse_sharding(
                   "{devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}"))),
          Pair("param1",
               ElementsAre(verified_parse_sharding(
                   "{devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}"))),
          Pair("dot",
               ElementsAre(verified_parse_sharding("{devices=[2,2]0,1,2,3}"))),
          Pair("copy", ElementsAre(verified_parse_sharding(
                           "{devices=[2,2]0,1,2,3}")))));
}

TEST_F(AutoShardingTest,
       SaveAndRemoveShardingAnnotationKeepInputOutputSmallTensor) {
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %entry (param0: f32[4,256,64], param1: f32[4,256,32]) -> f32[64,32] {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[2,2,1]0,1,2,3}
  %param1 = f32[4,256,32]{2,1,0} parameter(1), sharding={devices=[2,2,1]0,1,2,3}
  %dot = f32[64,32]{1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[4,256,32]{2,1,0} %param1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}, sharding={replicated}
  ROOT %copy = f32[64,32]{1,0} copy(f32[64,32]{1,0} %dot), sharding={devices=[2,2]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  // Keep all user shardings
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepInputOutputShardings;
  absl::flat_hash_set<const HloInstruction*> instructions_to_shard(
      module->entry_computation()->instructions().begin(),
      module->entry_computation()->instructions().end());
  std::pair<absl::flat_hash_map<std::string, std::vector<HloSharding>>, bool>
      saved_shardings_result =
          AutoShardingImplementation(option).SaveAndRemoveShardingAnnotation(
              module.get(), instructions_to_shard,
              /* replicated_small_tensors */ {"dot"},
              /* execution_threads */ {});
  absl::flat_hash_map<std::string, std::vector<HloSharding>> saved_shardings =
      saved_shardings_result.first;
  bool changed = saved_shardings_result.second;
  EXPECT_FALSE(changed);
  std::vector<HloInstruction*> instructions =
      module->entry_computation()->MakeInstructionPostOrder();
  EXPECT_THAT(instructions,
              Each(ResultOf(
                  [](const HloInstruction* ins) { return ins->has_sharding(); },
                  IsTrue())));

  auto verified_parse_sharding = [](const absl::string_view sharding_str) {
    absl::StatusOr<HloSharding> sharding = ParseSharding(sharding_str);
    CHECK_OK(sharding);
    return *sharding;
  };

  EXPECT_THAT(
      saved_shardings,
      UnorderedElementsAre(
          Pair("param0", ElementsAre(verified_parse_sharding(
                             "{devices=[2,2,1]0,1,2,3}"))),
          Pair("param1", ElementsAre(verified_parse_sharding(
                             "{devices=[2,2,1]0,1,2,3}"))),
          Pair("dot", ElementsAre(verified_parse_sharding("{replicated}"))),
          Pair("copy", ElementsAre(verified_parse_sharding(
                           "{devices=[2,2]0,1,2,3}")))));
}

TEST_F(AutoShardingTest, SaveAndRemoveShardingAnnotationKeepInputOutput) {
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %entry (param0: f32[4,256,64], param1: f32[4,256,32]) -> f32[64,32] {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}
  %param1 = f32[4,256,32]{2,1,0} parameter(1), sharding={devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}
  %param0_copy = f32[4,256,64]{2,1,0} copy(param0), sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}
  %param1_copy = f32[4,256,32]{2,1,0} copy(param1), sharding={devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}
  %dot = f32[64,32]{1,0} dot(f32[4,256,64]{2,1,0} %param0_copy, f32[4,256,32]{2,1,0} %param1_copy), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}, sharding={devices=[2,2]0,1,2,3}
  ROOT %copy = f32[64,32]{1,0} copy(f32[64,32]{1,0} %dot), sharding={devices=[2,2]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepInputOutputShardings;
  absl::flat_hash_set<const HloInstruction*> instructions_to_shard(
      module->entry_computation()->instructions().begin(),
      module->entry_computation()->instructions().end());
  std::pair<absl::flat_hash_map<std::string, std::vector<HloSharding>>, bool>
      saved_shardings_result =
          AutoShardingImplementation(option).SaveAndRemoveShardingAnnotation(
              module.get(), instructions_to_shard,
              /* replicated_small_tensors */ {},
              /* execution_threads */ {});
  absl::flat_hash_map<std::string, std::vector<HloSharding>> saved_shardings =
      saved_shardings_result.first;
  bool changed = saved_shardings_result.second;
  EXPECT_TRUE(changed);

  // Dot does not have shardings anymore.
  const HloInstruction* dot = FindInstruction(module.get(), "dot");
  ASSERT_NE(dot, nullptr);
  EXPECT_FALSE(dot->has_sharding());

  // params and copies still have shardings.
  const HloInstruction* param0 = FindInstruction(module.get(), "param0");
  ASSERT_NE(param0, nullptr);
  EXPECT_TRUE(param0->has_sharding());
  EXPECT_THAT(
      param0,
      op::Sharding("{devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}"));

  const HloInstruction* param0_copy =
      FindInstruction(module.get(), "param0_copy");
  ASSERT_NE(param0_copy, nullptr);
  EXPECT_TRUE(param0_copy->has_sharding());
  EXPECT_THAT(
      param0_copy,
      op::Sharding("{devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}"));

  const HloInstruction* param1 = FindInstruction(module.get(), "param1");
  ASSERT_NE(param1, nullptr);
  EXPECT_TRUE(param1->has_sharding());
  EXPECT_THAT(
      param1,
      op::Sharding("{devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}"));

  const HloInstruction* param1_copy =
      FindInstruction(module.get(), "param1_copy");
  ASSERT_NE(param1_copy, nullptr);
  EXPECT_TRUE(param1_copy->has_sharding());
  EXPECT_THAT(
      param1_copy,
      op::Sharding("{devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}"));

  // Root still has sharding
  const HloInstruction* copy = FindInstruction(module.get(), "copy");
  ASSERT_NE(copy, nullptr);
  EXPECT_TRUE(copy->has_sharding());
  EXPECT_THAT(copy, op::Sharding("{devices=[2,2]0,1,2,3}"));

  EXPECT_THAT(
      saved_shardings,
      UnorderedElementsAre(Pair("param0", ElementsAre(param0->sharding())),
                           Pair("param0_copy", ElementsAre(param0->sharding())),
                           Pair("param1", ElementsAre(param1->sharding())),
                           Pair("param1_copy", ElementsAre(param1->sharding())),
                           Pair("copy", ElementsAre(copy->sharding()))));
}

TEST_F(AutoShardingTest, SaveAndRemoveShardingAnnotationRemoveAll) {
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %entry (param0: f32[4,256,64], param1: f32[4,256,32]) -> f32[64,32] {
  %param0 = f32[4,256,64]{2,1,0} parameter(0),
  sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate} %param1 =
  f32[4,256,32]{2,1,0} parameter(1), sharding={devices=[1,1,2,2]0,2,1,3
  last_tile_dim_replicate} %dot = f32[64,32]{1,0} dot(f32[4,256,64]{2,1,0}
  %param0, f32[4,256,32]{2,1,0} %param1), lhs_contracting_dims={0,1},
  rhs_contracting_dims={0,1}, sharding={devices=[2,2]0,1,2,3} ROOT %copy =
  f32[64,32]{1,0} copy(f32[64,32]{1,0} %dot), sharding={devices=[2,2]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  // Remove all user shardings
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kRemoveAllShardings;
  absl::flat_hash_set<const HloInstruction*> instructions_to_shard(
      module->entry_computation()->instructions().begin(),
      module->entry_computation()->instructions().end());
  std::pair<absl::flat_hash_map<std::string, std::vector<HloSharding>>, bool>
      saved_shardings_result =
          AutoShardingImplementation(option).SaveAndRemoveShardingAnnotation(
              module.get(), instructions_to_shard,
              /* replicated_small_tensors */ {},
              /* execution_threads */ {});
  absl::flat_hash_map<std::string, std::vector<HloSharding>> saved_shardings =
      saved_shardings_result.first;
  bool changed = saved_shardings_result.second;
  EXPECT_TRUE(changed);
  EXPECT_THAT(saved_shardings, IsEmpty());
  std::vector<HloInstruction*> instructions =
      module->entry_computation()->MakeInstructionPostOrder();
  EXPECT_THAT(instructions,
              Each(ResultOf(
                  [](const HloInstruction* ins) { return ins->has_sharding(); },
                  IsFalse())));
}

TEST_F(AutoShardingTest, SaveAndRemoveShardingAnnotationRemoveAllSmallTensor) {
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %entry (param0: f32[4,256,64], param1: f32[4,256,32]) -> f32[64,32] {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[2,2,1]0,1,2,3}
  %param1 = f32[4,256,32]{2,1,0} parameter(1), sharding={devices=[2,2,1]0,1,2,3}
  %dot = f32[64,32]{1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[4,256,32]{2,1,0} %param1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}, sharding={replicated}
  ROOT %copy = f32[64,32]{1,0} copy(f32[64,32]{1,0} %dot), sharding={replicated}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  // Remove all user shardings
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kRemoveAllShardings;
  absl::flat_hash_set<const HloInstruction*> instructions_to_shard(
      module->entry_computation()->instructions().begin(),
      module->entry_computation()->instructions().end());
  std::pair<absl::flat_hash_map<std::string, std::vector<HloSharding>>, bool>
      saved_shardings_result =
          AutoShardingImplementation(option).SaveAndRemoveShardingAnnotation(
              module.get(), instructions_to_shard,
              /* replicated_small_tensors */ {"dot", "copy"},
              /* execution_threads */ {});
  absl::flat_hash_map<std::string, std::vector<HloSharding>> saved_shardings =
      saved_shardings_result.first;
  bool changed = saved_shardings_result.second;
  EXPECT_TRUE(changed);

  // params have no shardings.
  const HloInstruction* param0 = FindInstruction(module.get(), "param0");
  ASSERT_NE(param0, nullptr);
  EXPECT_FALSE(param0->has_sharding());

  const HloInstruction* param1 = FindInstruction(module.get(), "param1");
  ASSERT_NE(param1, nullptr);
  EXPECT_FALSE(param1->has_sharding());

  // Dot and copy have shardings as they are specified as replicated small
  // tensors.
  const HloInstruction* dot = FindInstruction(module.get(), "dot");
  ASSERT_NE(dot, nullptr);
  EXPECT_TRUE(dot->has_sharding());
  EXPECT_TRUE(dot->sharding().IsReplicated());

  const HloInstruction* copy = FindInstruction(module.get(), "copy");
  ASSERT_NE(copy, nullptr);
  EXPECT_TRUE(copy->has_sharding());
  EXPECT_TRUE(copy->sharding().IsReplicated());

  EXPECT_THAT(
      saved_shardings,
      UnorderedElementsAre(Pair("dot", ElementsAre(dot->sharding())),
                           Pair("copy", ElementsAre(copy->sharding()))));
}

TEST_F(AutoShardingTest, TupleReduceTest) {
  constexpr absl::string_view kHloString = R"(
HloModule module
%func (lhs_value: f32[], lhs_index: s32[], rhs_value: f32[], rhs_index: s32[]) -> (f32[], s32[]) {
  %lhs_value = f32[] parameter(0)
  %rhs_value = f32[] parameter(2)
  %compare.a = pred[] compare(f32[] %lhs_value, f32[] %rhs_value), direction=GE
  %select.a = f32[] select(pred[] %compare.a, f32[] %lhs_value, f32[] %rhs_value)
  %compare.b = pred[] compare(f32[] %lhs_value, f32[] %rhs_value), direction=EQ
  %lhs_index = s32[] parameter(1)
  %rhs_index = s32[] parameter(3)
  %minimum = s32[] minimum(s32[] %lhs_index, s32[] %rhs_index)
  %select.b = s32[] select(pred[] %compare.a, s32[] %lhs_index, s32[] %rhs_index)
  %select.c = s32[] select(pred[] %compare.b, s32[] %minimum, s32[] %select.b)
  ROOT %tuple = (f32[], s32[]) tuple(f32[] %select.a, s32[] %select.c)
}

ENTRY %entry {
  %param0 = f32[1,16,40]{2,1,0} parameter(0)
  %iota = s32[1,16,40]{2,1,0} iota(), iota_dimension=2
  %constant.a = f32[] constant(-inf)
  %constant.b = s32[] constant(0)
  %reduce = (f32[1,16]{1,0}, s32[1,16]{1,0}) reduce(f32[1,16,40]{2,1,0} %param0, s32[1,16,40]{2,1,0} %iota, f32[] %constant.a, s32[] %constant.b), dimensions={2}, to_apply=%func
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  auto* reduce = FindInstruction(module.get(), "reduce");
  ASSERT_NE(reduce, nullptr);
  EXPECT_THAT(
      reduce,
      AnyOf(op::Sharding("{{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}, "
                         "{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}}"),
            op::Sharding("{{devices=[1,2,2]0,2,1,3 last_tile_dim_replicate}, "
                         "{devices=[1,2,2]0,2,1,3 last_tile_dim_replicate}}")));
  auto sharding = reduce->sharding();
  TF_EXPECT_OK(sharding.Validate(reduce->shape(), 4));
}

TEST_F(AutoShardingTest, ReduceTest) {
  constexpr absl::string_view kHloString = R"(
HloModule module

%func (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

ENTRY %entry {
  %param0 = f32[1,16,128]{2,1,0} parameter(0)
  %param1 = f32[] parameter(1)
  %reduce = f32[1,16]{1,0} reduce(f32[1,16,128]{2,1,0} %param0, f32[] %param1), dimensions={2}, to_apply=%func
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  auto* reduce = FindInstruction(module.get(), "reduce");
  auto* param0 = FindInstruction(module.get(), "param0");
  ASSERT_NE(reduce, nullptr);
  auto reduce_matcher1 =
      op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}");
  auto param0_matcher1 =
      op::Sharding("{devices=[1,2,1,2]0,1,2,3 last_tile_dim_replicate}");
  auto reduce_matcher2 =
      op::Sharding("{devices=[1,2,2]0,2,1,3 last_tile_dim_replicate}");
  auto param0_matcher2 =
      op::Sharding("{devices=[1,2,1,2]0,2,1,3 last_tile_dim_replicate}");
  EXPECT_TRUE(
      (Matches(param0_matcher1)(param0) && Matches(reduce_matcher1)(reduce)) ||
      (Matches(param0_matcher2)(param0) && Matches(reduce_matcher2)(reduce)));
  auto sharding = reduce->sharding();
  TF_EXPECT_OK(sharding.Validate(reduce->shape(), 4));
}

TEST_F(AutoShardingTest, ScatterTest2D) {
  constexpr absl::string_view kHloString = R"(
HloModule module

region {
  Arg_0 = s32[] parameter(0)
  ROOT Arg_1 = s32[] parameter(1)
}

ENTRY %Scatter {
  call = s32[4,128]{1,0} parameter(0)
  clamp = s32[4,2]{1,0} parameter(1)
  broadcast = s32[4,8]{1,0} parameter(2)
  ROOT scatter = s32[4,128]{1,0} scatter(call, clamp, broadcast), update_window_dims={1}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0,1}, index_vector_dim=1, indices_are_sorted=true, unique_indices=true, to_apply=region
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  // Memory budget lower than what would be required if the largest tensors are
  // sharded only 2-ways
  option.memory_budget_per_device = 4 * 2 * (4 * 128 / 2) - 1;

  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
  auto* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(scatter, AnyOf(op::Sharding("{devices=[2,2]0,2,1,3}"),
                             op::Sharding("{devices=[2,2]0,1,2,3}")));
  auto scatter_sharding = scatter->sharding();
  TF_EXPECT_OK(scatter_sharding.Validate(scatter->shape(), 4));
}

TEST_F(AutoShardingTest, ScatterTest3D) {
  constexpr absl::string_view kHloString = R"(
HloModule module

region {
  Arg_0 = f32[] parameter(0)
  ROOT Arg_1 = f32[] parameter(1)
}

ENTRY %Scatter {
  call = f32[4,128,128]{2,1,0} parameter(0)
  clamp = s32[4,3]{1,0} parameter(1)
  multiply = f32[4,8,8]{2,1,0} parameter(2)
  ROOT scatter = f32[4,128,128]{2,1,0} scatter(call, clamp, multiply), update_window_dims={1,2}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0,1,2}, index_vector_dim=1, indices_are_sorted=true, unique_indices=true, to_apply=region
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  // Memory budget lower than what would be required if the largest tensors are
  // sharded only 2-ways
  option.memory_budget_per_device = 4 * 2 * (4 * 128 * 128 / 2) - 1;

  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
  auto* scatter = FindInstruction(module.get(), "scatter");
  ASSERT_NE(scatter, nullptr);
  EXPECT_THAT(scatter, AnyOf(op::Sharding("{devices=[2,2,1]0,2,1,3}"),
                             op::Sharding("{devices=[2,2,1]0,1,2,3}"),
                             op::Sharding("{devices=[2,1,2]0,2,1,3}"),
                             op::Sharding("{devices=[2,1,2]0,1,2,3}"),
                             op::Sharding("{devices=[1,2,2]0,1,2,3}"),
                             op::Sharding("{devices=[1,2,2]0,2,1,3}")));
  auto scatter_sharding = scatter->sharding();
  TF_EXPECT_OK(scatter_sharding.Validate(scatter->shape(), 4));
}

TEST_F(AutoShardingTest, GatherTest) {
  constexpr absl::string_view kHloString = R"(
HloModule module
ENTRY %entry {
  %param0 = f32[256,1024]{0,1} parameter(0)
  %param1 = s32[128,512,1]{2,1,0} parameter(1)
  ROOT %gather = f32[128,512,1024]{2,1,0} gather(f32[256,1024]{0,1} %param0, s32[128,512,1]{2,1,0} %param1), offset_dims={2}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=2, slice_sizes={1,1024}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  auto* gather = FindInstruction(module.get(), "gather");
  ASSERT_NE(gather, nullptr);
  EXPECT_THAT(
      gather,
      AnyOf(
          op::Sharding("{devices=[1,2,1,2]0,1,2,3 last_tile_dim_replicate}"),
          op::Sharding("{devices=[1,2,1,2]0,2,1,3 last_tile_dim_replicate}"),
          op::Sharding("{devices=[2,1,1,2]0,1,2,3 last_tile_dim_replicate}"),
          op::Sharding("{devices=[2,1,1,2]0,2,1,3 last_tile_dim_replicate}")));
  auto gather_sharding = gather->sharding();
  TF_EXPECT_OK(gather_sharding.Validate(gather->shape(), 4));
}

TEST_F(AutoShardingTest, GatherTestNoReshard) {
  constexpr absl::string_view kHloString = R"(
HloModule module
ENTRY %entry {
  get-tuple-element = s8[1000,128]{1,0} parameter(0)
  reshape = s32[8,1,1]{2,1,0} parameter(1)
  gather = s8[8,1,128]{2,1,0} gather(get-tuple-element, reshape), offset_dims={2}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=2, slice_sizes={1,128}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 1, 8};
  option.device_mesh_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  option.device_mesh_alpha = {1.0, 1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
  auto* gather = FindInstruction(module.get(), "gather");
  auto* param0 = FindInstruction(module.get(), "get-tuple-element");
  ASSERT_NE(gather, nullptr);
  ASSERT_NE(param0, nullptr);
  EXPECT_THAT(gather, op::Sharding("{devices=[8,1,1]0,1,2,3,4,5,6,7}"));
  EXPECT_THAT(param0, AnyOf(op::Sharding("{devices=[1,8]0,1,2,3,4,5,6,7}"),
                            op::Sharding("{devices=[8,1]0,1,2,3,4,5,6,7}")));
  TF_EXPECT_OK(gather->sharding().Validate(gather->shape(), 8));
  // Ensure no resharding op is created for operand 0 of gather in this case.
  EXPECT_EQ(param0, gather->operand(0));
}

TEST_F(AutoShardingTest, GatherConvTest) {
  constexpr absl::string_view kHloString = R"(
HloModule module
ENTRY %entry {
  %param0 = f32[1024,1024]{0,1} parameter(0)
  %param1 = s32[128,1024,1]{2,1,0} parameter(1)
  %gather = f32[128,1024,1024]{2,1,0} gather(f32[1024,1024]{0,1} %param0, s32[128,1024,1]{2,1,0} %param1),
  offset_dims={2}, collapsed_slice_dims={0}, start_index_map={0},
  index_vector_dim=2, slice_sizes={1,1024}
  %param2 = f32[1024,1024]{1,0} parameter(2), sharding={replicated}
  %reshape = f32[1024,1024,1]{2,1,0} reshape(param2)
  ROOT convolution = f32[128,1024,1024]{2,1,0} convolution(gather, reshape),
  window={size=1}, dim_labels=b0f_io0->b0f
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {4, 1, 1};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  auto* gather = FindInstruction(module.get(), "gather");
  auto* conv = FindInstruction(module.get(), "convolution");
  ASSERT_NE(gather, nullptr);
  ASSERT_NE(conv, nullptr);
  EXPECT_THAT(gather, op::Sharding("{devices=[1,4,1]0,1,2,3}"));
  EXPECT_THAT(conv, op::Sharding("{devices=[1,4,1]0,1,2,3}"));
  auto gather_sharding = gather->sharding();
  TF_EXPECT_OK(gather_sharding.Validate(gather->shape(), 4));
  auto conv_sharding = conv->sharding();
  TF_EXPECT_OK(conv_sharding.Validate(conv->shape(), 4));
}

TEST_F(AutoShardingTest, MatmulMeshShape1DMeshShape) {
  AutoShardingOption option;
  option.enable = true;
  // Only provide device_mesh_shape
  option.device_mesh_shape = {4};
  RunMatMulAutoShardingWithOptions(option, 4);
  option.device_mesh_shape = {8};
  RunMatMulAutoShardingWithOptions(option, 8);
}

TEST_F(AutoShardingTest, MatmulMeshShape1DMeshShapeIds) {
  AutoShardingOption option;
  option.enable = true;

  // Add mesh_ids
  option.device_mesh_shape = {4};
  option.device_mesh_ids = {0, 1, 2, 3};
  RunMatMulAutoShardingWithOptions(option, 4);

  option.device_mesh_shape = {8};
  option.device_mesh_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  RunMatMulAutoShardingWithOptions(option, 8);
}

TEST_F(AutoShardingTest, MatmulMeshShape1DAllOptions) {
  AutoShardingOption option;
  option.enable = true;
  // Add alpha and beta
  option.device_mesh_shape = {4};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0};
  option.device_mesh_beta = {1.0};
  RunMatMulAutoShardingWithOptions(option, 4);

  option.device_mesh_shape = {8};
  option.device_mesh_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  option.device_mesh_alpha = {1.0};
  option.device_mesh_beta = {1.0};
  RunMatMulAutoShardingWithOptions(option, 8);
}

TEST_F(AutoShardingTest, MatmulMeshShape2DAllOptions) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  RunMatMulAutoShardingWithOptions(option, 4, 2);

  option.enable = true;
  option.device_mesh_shape = {1, 4};
  RunMatMulAutoShardingWithOptions(option, 4);

  option.enable = true;
  option.device_mesh_shape = {4, 1};
  RunMatMulAutoShardingWithOptions(option, 4);
}

TEST_F(AutoShardingTest, MatmulMeshShape2DNoAlphaBeta) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  RunMatMulAutoShardingWithOptions(option, 4, 2);

  option.enable = true;
  option.device_mesh_shape = {1, 4};
  RunMatMulAutoShardingWithOptions(option, 4);

  // Specifying all mesh_* options.
  option.enable = true;
  option.device_mesh_shape = {4, 1};
  RunMatMulAutoShardingWithOptions(option, 4);
}

TEST_F(AutoShardingTest, MatmulMeshShape2DNoAlphaBetaMeshIds) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  RunMatMulAutoShardingWithOptions(option, 4, 2);

  option.enable = true;
  option.device_mesh_shape = {1, 4};
  RunMatMulAutoShardingWithOptions(option, 4);

  // Specifying all mesh_* options.
  option.enable = true;
  option.device_mesh_shape = {4, 1};
  RunMatMulAutoShardingWithOptions(option, 4);
}

TEST_F(AutoShardingTest, MatmulMeshShape2DNoMeshIds) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  RunMatMulAutoShardingWithOptions(option, 4, 2);

  option.enable = true;
  option.device_mesh_shape = {1, 4};
  RunMatMulAutoShardingWithOptions(option, 4);

  // Specifying all mesh_* options.
  option.enable = true;
  option.device_mesh_shape = {4, 1};
  RunMatMulAutoShardingWithOptions(option, 4);
}

TEST_F(AutoShardingTest, DISABLED_MatmulMeshShape3DAllOptions) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2, 2};
  option.device_mesh_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  option.device_mesh_alpha = {1.0, 1.0, 1.0};
  option.device_mesh_beta = {0.01, 0.5, 1.0};
  RunMatMulAutoShardingWithOptionsNoDeviceIds(option, {2, 2, 2}, true);
}

TEST_F(AutoShardingTest, Matmul3DMeshShape2DSharding) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 2, 2};
  RunMatMulAutoShardingWithOptions(option, 4, 2);

  option.device_mesh_shape = {2, 1, 2};
  RunMatMulAutoShardingWithOptions(option, 4, 2);

  option.device_mesh_shape = {2, 2, 1};
  RunMatMulAutoShardingWithOptions(option, 4, 2);
}

TEST_F(AutoShardingTest, DISABLED_AddMeshShape3DAllOptions) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 2, 4};
  option.device_mesh_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  option.device_mesh_alpha = {1.0, 1.0, 1.0};
  option.device_mesh_beta = {0.01, 0.5, 1.0};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {4, 1, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {1, 4, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);
}

TEST_F(AutoShardingTest, DISABLED_AddMeshShape3DNoAlphaBeta) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 2, 4};
  option.device_mesh_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {4, 1, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {1, 4, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);
}

TEST_F(AutoShardingTest, DISABLED_AddMeshShape3DNoAlphaBetaMeshIds) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 2, 4};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {4, 1, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {1, 4, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);
}

TEST_F(AutoShardingTest, DISABLED_AddMeshShape3DNoMeshIds) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 2, 4};
  option.device_mesh_alpha = {1.0, 1.0, 1.0};
  option.device_mesh_beta = {0.01, 0.5, 1.0};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {4, 1, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {1, 4, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);
}

TEST_F(AutoShardingTest, DISABLED_MatMulMeshShape2D) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  RunMatMulAutoShardingWithOptions(option, 4, 2);
}

TEST_F(AutoShardingTest, DISABLED_AddMeshShape2D) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  RunAddAutoShardingWithOptions(option, 4, 2);
}

TEST_F(AutoShardingTest, AddMeshShape3D) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2, 2};
  option.device_mesh_alpha = {1.0, 1.0, 1.0};
  option.device_mesh_beta = {0.01, 0.5, 1.0};
  RunAddAutoShardingWithOptions(option, 2);
}

TEST_F(AutoShardingTest, LargeSize) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 2, 4, 7};
  option.device_mesh_alpha = {1.0, 1.0, 1.0, 1.0};
  option.device_mesh_beta = {1.0, 1.0, 1.0, 1.0};
  RunMatMulAutoShardingWithOptions(option, 8, 2);
}

TEST_F(AutoShardingTest, InvalidOptions) {
  // Sizes do not match.
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 2, 4};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 0.5};
  EXPECT_FALSE(option.CheckAndSetup().ok());
  RunMatMulAutoShardingWithOptionsExpectFail(option);

  // device_mesh_shape is empty.
  AutoShardingOption empty_option;
  empty_option.enable = true;
  EXPECT_FALSE(empty_option.CheckAndSetup().ok());
  RunMatMulAutoShardingWithOptionsExpectFail(empty_option);

  // Non-positive values in device_mesh_shape.
  AutoShardingOption option_with_non_positive_mesh;
  option_with_non_positive_mesh.enable = true;
  option_with_non_positive_mesh.device_mesh_shape = {0, 4};
  EXPECT_FALSE(option_with_non_positive_mesh.CheckAndSetup().ok());
  RunMatMulAutoShardingWithOptionsExpectFail(option_with_non_positive_mesh);
  option_with_non_positive_mesh.device_mesh_shape = {-1, 4};
  EXPECT_FALSE(option_with_non_positive_mesh.CheckAndSetup().ok());
  RunMatMulAutoShardingWithOptionsExpectFail(option_with_non_positive_mesh);

  // device_mesh_shape and device_mesh_ids are not compatible.
  AutoShardingOption option_not_compatible;
  option_not_compatible.enable = true;
  option_not_compatible.device_mesh_shape = {4, 8};
  option_not_compatible.device_mesh_ids = {1, 2, 3, 4};
  EXPECT_FALSE(option_not_compatible.CheckAndSetup().ok());
  RunMatMulAutoShardingWithOptionsExpectFail(option_not_compatible);
}

TEST_F(AutoShardingTest, AutoShardingKeepUserShardingInputOutput) {
  // An HLO Module with sharding for all instructions.
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %entry (param0: f32[4,256,64], param1: f32[4,256,32]) -> f32[64,32] {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}
  %param1 = f32[4,256,32]{2,1,0} parameter(1), sharding={devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}
  %dot = f32[64,32]{1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[4,256,32]{2,1,0} %param1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}, sharding={devices=[2,2]0,1,2,3}
  ROOT %copy = f32[64,32]{1,0} copy(f32[64,32]{1,0} %dot), sharding={devices=[2,2]0,1,2,3}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  // Remove the sharding in dot
  auto* dot = FindInstruction(module.get(), "dot");
  dot->clear_sharding();
  EXPECT_FALSE(dot->has_sharding());
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepInputOutputShardings;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  auto* dot_after = FindInstruction(module.get(), "dot");
  ASSERT_NE(dot_after, nullptr);
  EXPECT_THAT(dot_after, op::Sharding("{devices=[2,2]0,1,2,3}"));
  auto sharding = dot_after->sharding();
  TF_EXPECT_OK(sharding.Validate(dot_after->shape(), 4));
}

TEST_F(AutoShardingTest, AutoShardingKeepUserShardingAdd) {
  // An HLO Module with sharding for all instructions.
  constexpr absl::string_view kHloString = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[128,128]{0,1} parameter(0)
  %param1 = f32[128,128]{0,1} parameter(1)
  %add = f32[128,128]{0,1} add(%param0, %param1), sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %copy = f32[128,128]{0,1} copy(%add)
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  // Run AutoSharding
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  auto* param0_after = FindInstruction(module.get(), "param0");
  ASSERT_NE(param0_after, nullptr);
  EXPECT_THAT(param0_after,
              op::Sharding("{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}"));
  auto* param1_after = FindInstruction(module.get(), "param1");
  ASSERT_NE(param1_after, nullptr);
  EXPECT_THAT(param1_after,
              op::Sharding("{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}"));
  auto* add_after = FindInstruction(module.get(), "add");
  ASSERT_NE(add_after, nullptr);
  EXPECT_THAT(add_after,
              op::Sharding("{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}"));
}

TEST_F(AutoShardingTest, AutoShardingKeepUserShardingDot) {
  // An HLO Module with sharding for all instructions.
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %entry (param0: f32[4,256,64], param1: f32[4,256,32]) -> f32[64,32] {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}
  %param1 = f32[4,256,32]{2,1,0} parameter(1), sharding={devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}
  %dot = f32[64,32]{1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[4,256,32]{2,1,0} %param1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}, sharding={devices=[2,2]0,1,2,3}
  ROOT %copy = f32[64,32]{1,0} copy(f32[64,32]{1,0} %dot), sharding={devices=[2,2]0,1,2,3}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  // Remove the sharding in param0, param1 and copy
  auto* param0 = FindInstruction(module.get(), "param0");
  param0->clear_sharding();
  EXPECT_FALSE(param0->has_sharding());
  auto* param1 = FindInstruction(module.get(), "param1");
  param1->clear_sharding();
  EXPECT_FALSE(param1->has_sharding());
  auto* copy = FindInstruction(module.get(), "copy");
  copy->clear_sharding();
  EXPECT_FALSE(copy->has_sharding());
  // Run AutoSharding
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  auto* param0_after = FindInstruction(module.get(), "param0");
  ASSERT_NE(param0_after, nullptr);
  EXPECT_THAT(
      param0_after,
      op::Sharding("{devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}"));
  auto* param1_after = FindInstruction(module.get(), "param1");
  ASSERT_NE(param1_after, nullptr);
  EXPECT_THAT(
      param1_after,
      op::Sharding("{devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}"));
  auto* copy_after = FindInstruction(module.get(), "copy");
  ASSERT_NE(copy_after, nullptr);
  EXPECT_THAT(copy_after, op::Sharding("{devices=[2,2]0,1,2,3}"));
}

TEST_F(AutoShardingTest, DISABLED_AutoShardingKeepUserShardingTupleReduce) {
  constexpr absl::string_view kHloString = R"(
HloModule module
%func (lhs_value: f32[], lhs_index: s32[], rhs_value: f32[], rhs_index: s32[]) -> (f32[], s32[]) {
  %lhs_value = f32[] parameter(0)
  %rhs_value = f32[] parameter(2)
  %compare.a = pred[] compare(f32[] %lhs_value, f32[] %rhs_value), direction=GE
  %select.a = f32[] select(pred[] %compare.a, f32[] %lhs_value, f32[] %rhs_value)
  %compare.b = pred[] compare(f32[] %lhs_value, f32[] %rhs_value), direction=EQ
  %lhs_index = s32[] parameter(1)
  %rhs_index = s32[] parameter(3)
  %minimum = s32[] minimum(s32[] %lhs_index, s32[] %rhs_index)
  %select.b = s32[] select(pred[] %compare.a, s32[] %lhs_index, s32[] %rhs_index)
  %select.c = s32[] select(pred[] %compare.b, s32[] %minimum, s32[] %select.b)
  ROOT %tuple = (f32[], s32[]) tuple(f32[] %select.a, s32[] %select.c)
}

ENTRY %entry {
  %param0 = f32[1,16,40]{2,1,0} parameter(0)
  %iota = s32[1,16,40]{2,1,0} iota(), iota_dimension=2
  %constant.a = f32[] constant(-inf)
  %constant.b = s32[] constant(0)
  %reduce = (f32[1,16]{1,0}, s32[1,16]{1,0}) reduce(f32[1,16,40]{2,1,0} %param0, s32[1,16,40]{2,1,0} %iota, f32[] %constant.a, s32[] %constant.b), dimensions={2}, to_apply=%func,
    sharding={{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}, {devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  // Keep all users shardings.
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  auto* reduce = FindInstruction(module.get(), "reduce");
  ASSERT_NE(reduce, nullptr);
  EXPECT_THAT(reduce, op::Sharding(
                          "{{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}, "
                          "{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}}"));
  auto sharding = reduce->sharding();
  TF_EXPECT_OK(sharding.Validate(reduce->shape(), 4));
  auto* param0 = FindInstruction(module.get(), "param0");
  ASSERT_NE(param0, nullptr);
  // There are multiple valid shardings, and we only
  EXPECT_FALSE(param0->sharding().IsReplicated());
}

TEST_F(AutoShardingTest, GetTupleElementUserShardingsParameter) {
  constexpr absl::string_view kHloString = R"(
HloModule module
ENTRY %tupleparameter {
  %param0 = f32[32,64]{1,0} parameter(0)
  %param1 = f32[32,64]{1,0} parameter(1), sharding={devices=[2,2]<=[4]}
  %tuple1 = (f32[32,64]{1,0}, f32[32,64]{1,0}) tuple(f32[32,64]{1,0} %param0, f32[32,64]{1,0} %param1)
  %first = f32[32,64]{1,0} get-tuple-element((f32[32,64]{1,0}, f32[32,64]{1,0}) %tuple1), index=0
  %second = f32[32,64]{1,0} get-tuple-element((f32[32,64]{1,0}, f32[32,64]{1,0}) %tuple1), index=1, sharding={devices=[4,1]<=[4]}
  ROOT root = f32[32,64]{1,0} add(%first, %second)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
  const HloInstruction* param1 = FindInstruction(module.get(), "param1");
  ASSERT_NE(param1, nullptr);
  EXPECT_THAT(param1, op::Sharding("{devices=[2,2]<=[4]}"));

  const HloInstruction* second = FindInstruction(module.get(), "root");
  ASSERT_NE(second, nullptr);
  EXPECT_THAT(second, op::Sharding("{devices=[4,1]<=[4]}"));
}

TEST_F(AutoShardingTest, DISABLED_TupleParameter) {
  constexpr absl::string_view kHloString = R"(
HloModule module
ENTRY %tupleparameter {
  %tuple_param = (f32[16,32,64]{2,1,0}, f32[16,32,64]{2,1,0}) parameter(0)
  %first = f32[16,32,64]{2,1,0} get-tuple-element((f32[16,32,64]{2,1,0}, f32[16,32,64]{2,1,0}) %tuple_param), index=0
  %second = f32[16,32,64]{2,1,0} get-tuple-element((f32[16,32,64]{2,1,0}, f32[16,32,64]{2,1,0}) %tuple_param), index=1
  ROOT root = f32[16,32,64]{2,1,0} add(%first, %second)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
  auto* tuple_param = FindInstruction(module.get(), "tuple_param");
  ASSERT_NE(tuple_param, nullptr);
  EXPECT_THAT(
      tuple_param,
      op::Sharding("{{devices=[2,2,1]0,2,1,3}, {devices=[2,2,1]0,2,1,3}}"));
  TF_EXPECT_OK(tuple_param->sharding().Validate(tuple_param->shape(), 4));
}

// CRASHES
TEST_F(AutoShardingTest, DISABLED_GetTupleElementWithUserShardingTest) {
  constexpr absl::string_view kHloString = R"(
HloModule module

%while_cond {
  %param0 = (u32[],f32[16,256,256]{2,1,0},f32[16,256,256]{2,1,0}) parameter(0)
  %count = u32[] get-tuple-element((u32[],f32[16,256,256]{2,1,0},f32[16,256,256]{2,1,0}) %param0), index=0
  %limit = u32[] constant(2)
  ROOT %lt = pred[] compare(%count, %limit), direction=LT
}

%while_body {
  %param0 = (u32[],f32[16,256,256]{2,1,0},f32[16,256,256]{2,1,0}) parameter(0)
  %count = u32[] get-tuple-element((u32[],f32[16,256,256]{2,1,0},f32[16,256,256]{2,1,0}) %param0), index=0
  %v1 = f32[16,256,256]{2,1,0} get-tuple-element((u32[],f32[16,256,256]{2,1,0},f32[16,256,256]{2,1,0}) %param0), index=1
  %v2 = f32[16,256,256]{2,1,0} get-tuple-element((u32[],f32[16,256,256]{2,1,0},f32[16,256,256]{2,1,0}) %param0), index=2

  %dot = f32[16,256,256]{2,1,0} dot(f32[16,256,256]{2,1,0} %v1, f32[16,256,256]{2,1,0} %v2), lhs_contracting_dims={2}, rhs_contracting_dims={2}, lhs_batch_dims={0}, rhs_batch_dims={0}
  %dot_tanh = f32[16,256,256]{2,1,0} tanh(f32[16,256,256]{2,1,0} %dot)
  %dot_cos = f32[16,256,256]{2,1,0} cosine(f32[16,256,256]{2,1,0} %dot)
  ROOT %result = (u32[],f32[16,256,256]{2,1,0},f32[16,256,256]{2,1,0}) tuple(%count, %dot_tanh, %dot_cos)
}

ENTRY %entry (param0: f32[16,256,256], param1: f32[16,256,256]) -> f32[16,256,256] {
  %param0 = f32[16,256,256]{2,1,0} parameter(0), sharding={devices=[2,1,2]0,1,2,3}
  %param1 = f32[16,256,256]{2,1,0} parameter(1), sharding={devices=[2,1,2]0,1,2,3}

  %zero = u32[] constant(0)
  %init = (u32[], f32[16,256,256], f32[16,256,256]) tuple(%zero, %param0, %param1)
  %while.1 = (u32[],f32[16,256,256]{2,1,0},f32[16,256,256]{2,1,0})  while(%init), body=%while_body, condition=%while_cond
  %tuple1 = f32[16,256,256]{2,1,0} get-tuple-element((u32[], f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}) %while.1), index=1, sharding={devices=[2,2,1]0,2,1,3}
  ROOT %tanh = f32[16,256,256]{2,1,0} tanh(f32[16,256,256]{2,1,0} %tuple1)
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings;
  option.enable = true;
  option.device_mesh_shape = {2, 1, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0, 1.0};
  auto changed_or = AutoSharding(option).Run(module.get());
  EXPECT_FALSE(changed_or.ok());
}

TEST_F(AutoShardingTest, While) {
  constexpr absl::string_view kHloString = R"(
HloModule module

%cond {
  %vars.cond = (u32[], bf16[2,2048,768], bf16[128,512,2048], bf16[128,512,768], s32[]) parameter(0)
  %count.cond = u32[] get-tuple-element(%vars.cond), index=0
  %limit = u32[] constant(2)
  ROOT %lt = pred[] compare(%count.cond, %limit), direction=LT
}

%body {
  %param = (u32[], bf16[2,2048,768], bf16[128,512,2048], bf16[128,512,768], s32[]) parameter(0)
  %i0 = s32[] constant(0)
  %count = u32[] get-tuple-element(%param), index=0
  %gte0 = bf16[2,2048,768]{2,1,0} get-tuple-element(%param), index=1
  %index = s32[] get-tuple-element(%param), index=4
  %ds = bf16[1,2048,768]{2,1,0} dynamic-slice(%gte0, s32[] %index, s32[] %i0, s32[] %i0), dynamic_slice_sizes={1,2048,768}
  %rhs = bf16[2048,768]{1,0} reshape(%ds)
  %lhs = bf16[128,512,2048]{2,1,0} get-tuple-element(%param), index=2
  %dot = bf16[128,512,768]{2,1,0} dot(bf16[128,512,2048]{2,1,0} %lhs, bf16[2048,768]{1,0} %rhs), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  ROOT %tuple = (u32[], bf16[2,2048,768], bf16[128,512,2048], bf16[128,512,768], s32[]) tuple(%count, %gte0, %lhs, %dot, index)
}

ENTRY %entry {
  %p0 = bf16[2048,768] parameter(0)
  %p1 = bf16[128,512,2048] parameter(1)
  %p2 = bf16[128,512,768] parameter(2)
  %reshape0 = bf16[1,2048,768] reshape(%p0)
  %concat0 = bf16[2,2048,768] concatenate(%reshape0, %reshape0), dimensions={0}
  %zero = u32[] constant(0)
  %p3 = s32[] parameter(3)
  %init = (u32[], bf16[2,2048,768], bf16[128,512,2048], bf16[128,512,768], s32[]) tuple(%zero, %concat0, %p1, %p2, %p3)
  %while = (u32[], bf16[2, 2048, 768], bf16[128,512,2048], bf16[128,512,768], s32[]) while(%init), body=%body, condition=%cond
  ROOT %result = bf16[128,512,768] get-tuple-element(%while), index=3
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(0) << module->ToString();
  EXPECT_TRUE(changed);
  auto* while_op = FindInstruction(module.get(), "while");
  ASSERT_NE(while_op, nullptr);
  // There could be multiple valid sharding for this test. So we only check
  // that the shardings are the same for while op related instructions.
  for (size_t i = 0; i < while_op->while_body()
                             ->root_instruction()
                             ->sharding()
                             .tuple_elements()
                             .size();
       i++) {
    const HloSharding& root_sharding = while_op->while_body()
                                           ->root_instruction()
                                           ->sharding()
                                           .tuple_elements()
                                           .at(i);
    EXPECT_EQ(while_op->while_body()
                  ->parameter_instruction(0)
                  ->sharding()
                  .tuple_elements()
                  .at(i)
                  .ToString(),
              root_sharding.ToString());
    EXPECT_EQ(while_op->while_condition()
                  ->parameter_instruction(0)
                  ->sharding()
                  .tuple_elements()
                  .at(i)
                  .ToString(),
              root_sharding.ToString());
  }
}

TEST_F(AutoShardingTest, DynamicSlice) {
  constexpr absl::string_view kHloString = R"(
HloModule module
ENTRY %entry {
  %param0 = s32[] parameter(0)
  %arg_tuple = (s32[], f32[4,256,1024]{2,1,0}, f32[2]{0}, f32[2]{0}, f32[2]{0}, /*index=5*/f32[2]{0}, f32[2]{0}, f32[2]{0}, f32[2,4,256,1024]{3,2,1,0}, f32[2,4096]{1,0}, /*index=10*/f32[2,1024,4096]{2,1,0}, f32[2,1024]{1,0}, f32[2,4096,1024]{2,1,0}, f32[2,1024]{1,0}, f32[2,1024]{1,0}, /*index=15*/f32[2,1024]{1,0}, f32[2,1024]{1,0}, f32[2,4,256]{2,1,0}, f32[2,1024,4,256]{3,2,1,0}, f32[2,256]{1,0}, /*index=20*/f32[2,1024]{1,0}, f32[2,1024,4,256]{3,2,1,0}, f32[2,4,256]{2,1,0}, f32[2,1024,4,256]{3,2,1,0}, f32[2,4,256]{2,1,0}, /*index=25*/f32[2,1024,4,256]{3,2,1,0}, f32[2,4096]{1,0}, f32[2,1024,4096]{2,1,0}, f32[2,1024]{1,0}, f32[2,4096,1024]{2,1,0}, /*index=30*/f32[2,1024]{1,0}, f32[2,1024]{1,0}, f32[2,1024]{1,0}, f32[2,1024]{1,0}, f32[2,4,256]{2,1,0}, /*index=35*/f32[2,1024,4,256]{3,2,1,0}, f32[2,256]{1,0}, f32[2,1024]{1,0}, f32[2,1024,4,256]{3,2,1,0}, f32[2,4,256]{2,1,0}, /*index=40*/f32[2,1024,4,256]{3,2,1,0}, f32[2,4,256]{2,1,0}, f32[2,1024,4,256]{3,2,1,0}, f32[4,1,256,256]{3,2,1,0}, f32[4,256,1]{2,1,0}, /*index=45*/f32[4,256,1]{2,1,0}, f32[4,256,1]{2,1,0}, f32[4,256,1]{2,1,0}, f32[4,256,1]{2,1,0}, f32[], /*index=50*/f32[], f32[4,256,1]{2,1,0}, f32[], f32[]) parameter(1)
  %constant.a = s32[] constant(2)
  %constant.b = s32[] constant(0)
  %compare = pred[] compare(s32[] %param0, s32[] %constant.b), direction=LT
  %add = s32[] add(s32[] %param0, s32[] %constant.a)
  %select = s32[] select(pred[] %compare, s32[] %add, s32[] %param0)
  %get-tuple-element = f32[2,1024]{1,0} get-tuple-element((s32[], f32[4,256,1024]{2,1,0}, f32[2]{0}, f32[2]{0}, f32[2]{0}, /*index=5*/f32[2]{0}, f32[2]{0}, f32[2]{0}, f32[2,4,256,1024]{3,2,1,0}, f32[2,4096]{1,0}, /*index=10*/f32[2,1024,4096]{2,1,0}, f32[2,1024]{1,0}, f32[2,4096,1024]{2,1,0}, f32[2,1024]{1,0}, f32[2,1024]{1,0}, /*index=15*/f32[2,1024]{1,0}, f32[2,1024]{1,0}, f32[2,4,256]{2,1,0}, f32[2,1024,4,256]{3,2,1,0}, f32[2,256]{1,0}, /*index=20*/f32[2,1024]{1,0}, f32[2,1024,4,256]{3,2,1,0}, f32[2,4,256]{2,1,0}, f32[2,1024,4,256]{3,2,1,0}, f32[2,4,256]{2,1,0}, /*index=25*/f32[2,1024,4,256]{3,2,1,0}, f32[2,4096]{1,0}, f32[2,1024,4096]{2,1,0}, f32[2,1024]{1,0}, f32[2,4096,1024]{2,1,0}, /*index=30*/f32[2,1024]{1,0}, f32[2,1024]{1,0}, f32[2,1024]{1,0}, f32[2,1024]{1,0}, f32[2,4,256]{2,1,0}, /*index=35*/f32[2,1024,4,256]{3,2,1,0}, f32[2,256]{1,0}, f32[2,1024]{1,0}, f32[2,1024,4,256]{3,2,1,0}, f32[2,4,256]{2,1,0}, /*index=40*/f32[2,1024,4,256]{3,2,1,0}, f32[2,4,256]{2,1,0}, f32[2,1024,4,256]{3,2,1,0}, f32[4,1,256,256]{3,2,1,0}, f32[4,256,1]{2,1,0}, /*index=45*/f32[4,256,1]{2,1,0}, f32[4,256,1]{2,1,0}, f32[4,256,1]{2,1,0}, f32[4,256,1]{2,1,0}, f32[], /*index=50*/f32[], f32[4,256,1]{2,1,0}, f32[], f32[]) %arg_tuple), index=16
  ROOT %dynamic-slice = f32[1,1024]{1,0} dynamic-slice(f32[2,1024]{1,0} %get-tuple-element, s32[] %select, s32[] %constant.b), dynamic_slice_sizes={1,1024}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(0) << module->ToString();
  EXPECT_TRUE(changed);
}

TEST_F(AutoShardingTest, Alias) {
  constexpr absl::string_view kHloString = R"(
HloModule module, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias), {2}: (2, {}, may-alias), {3}: (3, {}, may-alias)}

ENTRY %entry {
  param.0 = u32[] parameter(0)
  param.1 = f32[32]{0} parameter(1)
  param.2 = f32[32]{0} parameter(2)
  param.3 = f32[1000]{0} parameter(3)
  ROOT tuple = (u32[], f32[32]{0}, f32[32]{0}, f32[1000]{0}) tuple(param.0, param.1, param.2, param.3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(0) << module->ToString();
  EXPECT_TRUE(changed);
}

TEST_F(AutoShardingTest, AliasTupleParameter) {
  constexpr absl::string_view kHloString = R"(
HloModule module, input_output_alias={ {0}: (0, {0}, may-alias), {1}: (0, {1}, may-alias), {2}: (0, {2}, may-alias), {3}: (0, {3}, may-alias)}

ENTRY %entry {
  arg_tuple.1 = (u32[], f32[32]{0}, f32[32]{0}, f32[1000]{0}) parameter(0)
  get-tuple-element.0 = u32[] get-tuple-element(arg_tuple.1), index=0
  get-tuple-element.1 = f32[32]{0} get-tuple-element(arg_tuple.1), index=1
  get-tuple-element.2 = f32[32]{0} get-tuple-element(arg_tuple.1), index=2
  get-tuple-element.3 = f32[1000]{0} get-tuple-element(arg_tuple.1), index=3
  ROOT tuple = (u32[], f32[32]{0}, f32[32]{0}, f32[1000]{0}) tuple(get-tuple-element.0, get-tuple-element.1, get-tuple-element.2, get-tuple-element.3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(0) << module->ToString();
  EXPECT_TRUE(changed);
}

TEST_F(AutoShardingTest, JaxRandomUniform) {
  constexpr absl::string_view kHloString = R"(
HloModule module
clone {
  lhs.1 = u32[] parameter(0)
  rhs.1 = u32[] parameter(2)
  or.2 = u32[] or(lhs.1, rhs.1)
  lhs.0 = u32[] parameter(1)
  rhs.0 = u32[] parameter(3)
  or.3 = u32[] or(lhs.0, rhs.0)
  ROOT tuple.23 = (u32[], u32[]) tuple(or.2, or.3)
}

ENTRY %entry {
  shift-left = u32[2,2]{1,0} parameter(0)
  select = u32[2,2]{1,0} parameter(1)
  constant.a = u32[] parameter(2)
  reduce = (u32[2]{0}, u32[2]{0}) reduce(shift-left, select, constant.a, constant.a), dimensions={1}, to_apply=clone
  rng-bit-generator = u32[8,512]{1,0} rng-bit-generator(reduce), algorithm=rng_default
  constant.b = u32[] constant(9)
  broadcast.a = u32[8,512]{1,0} broadcast(constant.b), dimensions={}, sharding={replicated}
  shift-right-logical = u32[8,512]{1,0} shift-right-logical(rng-bit-generator, broadcast.a)
  constant.c = u32[] constant(1065353216)
  broadcast.b = u32[8,512]{1,0} broadcast(constant.c), dimensions={}, sharding={replicated}
  or = u32[8,512]{1,0} or(shift-right-logical, broadcast.b)
  bitcast-convert = f32[8,512]{1,0} bitcast-convert(or)
  constant.d = f32[] constant(1)
  broadcast.c = f32[8,512]{1,0} broadcast(constant.d), dimensions={}, sharding={replicated}
  subtract = f32[8,512]{1,0} subtract(bitcast-convert, broadcast.c)
  constant.e = f32[] constant(0)
  broadcast.d = f32[8,512]{1,0} broadcast(constant.e), dimensions={}, sharding={replicated}
  ROOT maximum = f32[8,512]{1,0} maximum(subtract, broadcast.d)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(0) << module->ToString();
  EXPECT_TRUE(changed);
  EXPECT_TRUE(module->entry_computation()->root_instruction()->has_sharding());
  auto* tuple_operand = FindInstruction(module.get(), "reduce");
  ASSERT_NE(tuple_operand, nullptr);
  EXPECT_THAT(tuple_operand, op::Sharding("{{replicated}, {replicated}}"));
}

TEST_F(AutoShardingTest, Reshape) {
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %entry {
  %param.0 = bf16[24,2048,2048]{2,1,0} parameter(0)
  %param.1 = s32[] parameter(1)
  %param.2 = bf16[512,1024,2048]{2,1,0} parameter(2)
  %constant = s32[] constant(0)
  %dynamic-slice = bf16[1,2048,2048]{2,1,0} dynamic-slice(bf16[24,2048,2048]{2,1,0} %param.0, s32[] %param.1, s32[] %constant, s32[] %constant), dynamic_slice_sizes={1,2048,2048}
  %reshape = bf16[2048,16,128]{2,1,0} reshape(bf16[1,2048,2048]{2,1,0} %dynamic-slice)
  %dot = bf16[512,1024,16,128]{3,2,1,0} dot(bf16[512,1024,2048]{2,1,0} %param.2, bf16[2048,16,128]{2,1,0} %reshape), lhs_contracting_dims={2}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {64, 1};
  option.device_mesh_ids.resize(64);
  std::iota(option.device_mesh_ids.begin(), option.device_mesh_ids.end(), 0);
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(1) << module->ToString();
  EXPECT_TRUE(changed);
}

TEST_F(AutoShardingTest, ReshapeWithInvalidUserSharding) {
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %entry {
  %param.0 = bf16[24,16,16]{2,1,0} parameter(0), sharding={devices=[32,1,1]<=[32]}
  %reshape = bf16[1,24,16,16]{3,2,1,0} reshape(%param.0)
  %copy = bf16[1,24,16,16]{3,2,1,0} copy(%reshape)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {32, 1};
  option.device_mesh_ids.resize(32);
  std::iota(option.device_mesh_ids.begin(), option.device_mesh_ids.end(), 0);
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  VLOG(1) << module->ToString();
  HloInstruction* reshape = FindInstruction(module.get(), "reshape");
  EXPECT_THAT(reshape, op::Sharding("{devices=[1,32,1,1]<=[32]}"));
}

TEST_F(AutoShardingTest, Broadcast) {
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %entry {
  %param.0 = s32[32]{0} parameter(0)
  ROOT broadcast = s32[512,1024,1024,32]{3,2,1,0} broadcast(s32[32]{0} %param.0), dimensions={3}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 1, 64};
  option.memory_budget_per_device = 1025 * 1024 * 1024;  // 1025MB
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(1) << module->ToString();
  EXPECT_TRUE(changed);
}

TEST_F(AutoShardingTest, TestReshardingCostsForUserAnnotatedSharding) {
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[256,256] parameter(0)
  %param1 = f32[256,256] parameter(1)
  %dot = f32[256,256] dot(%param0, %param1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  ROOT %result = f32[256,256] tanh(%dot), sharding={devices=[1,4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_beta = {1, 1};
  option.device_mesh_alpha = {1, 1};
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings;
  AutoSharding pass(option);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);
  LOG(INFO) << module->ToString();
  EXPECT_GT(pass.GetSolverOptimalObjectiveValue(), 0);
}

TEST_F(AutoShardingTest, AllowAliasToFollowerConversion) {
  constexpr absl::string_view kHloString = R"(
HloModule module, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias), {2}: (2, {}, may-alias), {3}: (3, {}, may-alias)}

ENTRY %entry {
  param.0 = u32[] parameter(0)
  param.1 = f32[32]{0} parameter(1)
  param.2 = f32[32]{0} parameter(2)
  param.3 = f32[32000]{0} parameter(3)
  ROOT tuple.61 = (u32[], f32[32]{0}, f32[32]{0}, f32[32000]{0}) tuple(param.0, param.1, param.2, param.3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  option.allow_alias_to_follower_conversion = true;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(0) << module->ToString();
  EXPECT_TRUE(changed);
}

TEST_F(AutoShardingTest, DisallowAliasToFollowerConversion) {
  constexpr absl::string_view kHloString = R"(
HloModule module, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias), {2}: (2, {}, may-alias), {3}: (3, {}, may-alias)}

ENTRY %entry {
  param.0 = u32[] parameter(0)
  param.1 = f32[32]{0} parameter(1)
  param.2 = f32[32]{0} parameter(2)
  param.3 = f32[32000]{0} parameter(3)
  ROOT tuple.61 = (u32[], f32[32]{0}, f32[32]{0}, f32[32000]{0}) tuple(param.0, param.1, param.2, param.3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  option.allow_alias_to_follower_conversion = false;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(0) << module->ToString();
  EXPECT_TRUE(changed);
}

TEST_F(AutoShardingTest, BufferDonorConfigPreservation) {
  constexpr absl::string_view kHloString = R"(
HloModule Module, buffer_donor={ (0, {0}), (0, {1}) }

ENTRY entry {
  %p = (f32[], f32[]) parameter(0)
  %p0 = f32[] get-tuple-element((f32[], f32[]) %p), index=0
  %p1 = f32[] get-tuple-element((f32[], f32[]) %p), index=1
  ROOT %out = (f32[], f32[]) tuple(%p0, %p1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  // Creating an explicit copy here to ensure that it is not modified during
  // auto-sharding
  const HloBufferDonorConfig buffer_donor_config_before =
      module->buffer_donor_config();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  const HloBufferDonorConfig& buffer_donor_config_after =
      module->buffer_donor_config();
  EXPECT_EQ(buffer_donor_config_before.ToString(),
            buffer_donor_config_after.ToString());
}

TEST_F(AutoShardingTest, InputOutputAliasConfigPreservation) {
  constexpr absl::string_view kHloString = R"(
HloModule Module, input_output_alias={ {0}: (0, {0}, must-alias), {1}: (0, {1}) }

ENTRY entry {
  %p = (f32[], f32[]) parameter(0)
  %p0 = f32[] get-tuple-element((f32[], f32[]) %p), index=0
  %p1 = f32[] get-tuple-element((f32[], f32[]) %p), index=1
  ROOT %out = (f32[], f32[]) tuple(%p0, %p1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  // Creating an explicit copy here to ensure that it is not modified during
  // auto-sharding
  const HloInputOutputAliasConfig input_output_alias_config_before =
      module->input_output_alias_config();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  const HloInputOutputAliasConfig& input_output_alias_config_after =
      module->input_output_alias_config();
  EXPECT_EQ(input_output_alias_config_before.ToString(),
            input_output_alias_config_after.ToString());
}

}  // namespace
}  // namespace spmd
}  // namespace xla

/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/profile_guided_latency_estimator.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::Property;
using ::testing::UnorderedElementsAre;
using ::tsl::testing::StatusIs;

int GetIndexByName(absl::Span<HloInstruction* const> instruction_sequence,
                   absl::string_view hlo_name) {
  return absl::c_find_if(instruction_sequence,
                         [hlo_name](HloInstruction* instruction) {
                           return instruction->name() == hlo_name;
                         }) -
         instruction_sequence.begin();
}

// TODO(b/346918304): Separate relevant tests from gpu_hlo_schedule_test.cc
// into broader GPU scheduling related tests vs. tests related to components of
// GPU LHS.

class GpuLatencyHidingSchedulerBaseTest : public HloTestBase {
 protected:
  absl::StatusOr<HloModule*> ScheduleModule(
      HloModule* module, int64_t num_parallel_resources = 1) {
    auto& test_backend = backend();
    const auto& gpu_device_info =
        test_backend.default_stream_executor()->GetDeviceDescription();
    HloModuleConfig config(module->config());
    DebugOptions dboptions(config.debug_options());
    dboptions.set_xla_gpu_enable_pgle_accuracy_checker(true);
    dboptions.set_xla_gpu_experimental_parallel_collective_overlap_limit(
        num_parallel_resources);
    config.set_debug_options(dboptions);
    module->set_config(config);
    TF_RETURN_IF_ERROR(
        ScheduleGpuModule(module, /*pointer_size=*/8, gpu_device_info)
            .status());
    return module;
  }

  HloModuleConfig GetModuleConfig(absl::string_view fdo_profile) {
    HloModuleConfig config;
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_latency_hiding_scheduler(true);
    debug_options.set_xla_gpu_lhs_enable_gpu_async_tracker(true);
    config.set_debug_options(debug_options);
    *config.mutable_fdo_profile() = fdo_profile;
    return config;
  }
};

TEST_F(GpuLatencyHidingSchedulerBaseTest,
       GPUProfileStatisticsAggregatorDoesNotCountMissingNoops) {
  GPUProfileStatisticsAggregator aggregator;
  ProfileStatisticsAggregator::Statistics before_stats = aggregator.GetStats();

  ASSERT_EQ(before_stats.missing_instructions.size(), 0);
  ASSERT_EQ(before_stats.found_instructions_count, 0);

  absl::string_view kFdoProfile = "";
  absl::string_view kHloModule = R"(
    HloModule m

    ENTRY main {
      parameter0 = f32[] parameter(0)
      parameter1 = f32[32] parameter(1)
      const0 = f32[] constant(42)
      bitcast0 = f32[2,16] bitcast(parameter1)
      partition-id0 = u32[] partition-id()
      replica-id0 = u32[] replica-id()
      tuple0 = (f32[], f32[2,16], u32[], u32[]) tuple(parameter0, bitcast0, partition-id0, replica-id0)
      opt-barrier = (f32[], f32[2,16], u32[], u32[]) opt-barrier(tuple0)
      ROOT _ = get-tuple-element(opt-barrier), index=0
    }
  )";

  auto config = GetModuleConfig(kFdoProfile);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloModule, config));

  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    aggregator.HandleMissingInstructionCost(*instr);

    ProfileStatisticsAggregator::Statistics after_stats = aggregator.GetStats();
    EXPECT_EQ(after_stats.missing_instructions.size(), 0);
    EXPECT_EQ(after_stats.found_instructions_count, 0);
  }
}

TEST_F(GpuLatencyHidingSchedulerBaseTest,
       GPUProfileStatisticsAggregatorCountsMissingInstruction) {
  GPUProfileStatisticsAggregator aggregator;
  ProfileStatisticsAggregator::Statistics before_stats = aggregator.GetStats();

  ASSERT_EQ(before_stats.missing_instructions.size(), 0);
  ASSERT_EQ(before_stats.found_instructions_count, 0);

  absl::string_view kFdoProfile = R"pb(
    costs { name: "dot0" cost_us: 100.0 }
  )pb";
  absl::string_view kHloModule = R"(
    HloModule m

    ENTRY main {
      parameter0 = f32[] parameter(0)
      parameter1 = f32[32] parameter(1)
      const0 = f32[] constant(42)
      add0 = f32[] add(parameter0, const0)
      bitcast0 = f32[2,16] bitcast(parameter1)
      tuple0 = (f32[], f32[2,16]) tuple(add0, bitcast0)
      ROOT _ = get-tuple-element(tuple0), index=0
    }
  )";

  auto config = GetModuleConfig(kFdoProfile);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloModule, config));

  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    aggregator.HandleMissingInstructionCost(*instr);
  }
  ProfileStatisticsAggregator::Statistics after_stats = aggregator.GetStats();
  EXPECT_EQ(after_stats.missing_instructions.size(), 1);
  EXPECT_EQ((*after_stats.missing_instructions.begin())->opcode(),
            HloOpcode::kAdd);
  EXPECT_EQ(after_stats.found_instructions_count, 0);
}

TEST_F(GpuLatencyHidingSchedulerBaseTest,
       GPUProfileStatisticsAggregatorCountsMissingAsyncPairs) {
  GPUProfileStatisticsAggregator aggregator;
  ProfileStatisticsAggregator::Statistics before_stats = aggregator.GetStats();

  ASSERT_EQ(before_stats.missing_instructions.size(), 0);
  ASSERT_EQ(before_stats.found_instructions_count, 0);

  absl::string_view kFdoProfile = "";
  absl::string_view kHloModule = R"(
    HloModule m

    reduce {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT _ = f32[] add(x, y)
    }

    ENTRY main {
      p0 = f32[] parameter(0)
      p1 = f32[2] parameter(1)
      ar_0 = f32[] all-reduce-start(p0), to_apply=reduce
      ar_1 = f32[] all-reduce-done(ar_0)
      rs_0 = ((f32[2]), f32[1]) reduce-scatter-start(p1), to_apply=reduce, dimensions={0}
      rs_1 = f32[1] reduce-scatter-done(rs_0)
      ag_0 = (f32[2], f32[4]) all-gather-start(p1), replica_groups={{0,1}}, dimensions={0}
      ag_1 = f32[4] all-gather-done(ag_0)
      ROOT _ = (f32[], f32[1], f32[4]) tuple(ar_1, rs_1, ag_1)
    }
  )";

  auto config = GetModuleConfig(kFdoProfile);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloModule, config));

  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    for (const HloInstruction* user : instr->users()) {
      aggregator.HandleMissingInstructionLatency(*instr, *user);
    }
  }
  ProfileStatisticsAggregator::Statistics after_stats = aggregator.GetStats();
  EXPECT_EQ(after_stats.found_instructions_count, 0);
  EXPECT_EQ(after_stats.missing_instructions.size(), 3);
  EXPECT_THAT(
      after_stats.missing_instructions,
      UnorderedElementsAre(
          Property(&HloInstruction::opcode, HloOpcode::kAllReduceStart),
          Property(&HloInstruction::opcode, HloOpcode::kAsyncStart),
          Property(&HloInstruction::opcode, HloOpcode::kAllGatherStart)));
}

TEST_F(GpuLatencyHidingSchedulerBaseTest,
       ScheduleGpuModuleErrorsOutOnMissingInstrucitonsForAWhileLoopBody) {
  absl::string_view kFdoProfile = R"pb(
    costs { name: "dot0" cost_us: 100.0 }
  )pb";
  absl::string_view kHloModule = R"(
    HloModule m

    loop_body {
      p = (u32[], f32[1]) parameter(0)
      t0 = u32[] get-tuple-element(p), index=0
      t1 = f32[1] get-tuple-element(p), index=1
      add0 = f32[1] add(t1, t1)
      ROOT _ = (u32[],f32[1]) tuple(t0,t1)
    }

    loop_cond {
      p1 = (u32[], f32[1]) parameter(0)
      count = u32[] get-tuple-element(p1), index=0
      ub = u32[] constant(2)
      ROOT _ = pred[] compare(count, ub), direction=LT
    }

    ENTRY main {
      p2 = f32[1] parameter(0)
      ind = u32[] constant(1)
      t = (u32[],f32[1]) tuple(ind,p2)
      w = (u32[],f32[1]) while(t), body=loop_body, condition=loop_cond
      ROOT _ = f32[1] get-tuple-element(w), index=1
    }
  )";
  auto config = GetModuleConfig(kFdoProfile);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule, config));

  EXPECT_THAT(ScheduleModule(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(GpuLatencyHidingSchedulerBaseTest,
       ScheduleGpuModuleErrorsOutOnMissingInstrucitonsForAnEntryComputation) {
  absl::string_view kFdoProfile = R"pb(
    costs { name: "dot0" cost_us: 100.0 }
  )pb";
  absl::string_view kHloModule = R"(
    HloModule m

    ENTRY main {
      p0 = f32[1] parameter(0)
      ROOT add0 = f32[1] add(p0,p0)
    }
  )";
  auto config = GetModuleConfig(kFdoProfile);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule, config));

  EXPECT_THAT(ScheduleModule(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(GpuLatencyHidingSchedulerBaseTest,
       ScheduleGpuModulePassesOnFullFDOProfile) {
  absl::string_view kFdoProfile = R"pb(
    costs { name: "add0" cost_us: 100.0 }
  )pb";
  absl::string_view kHloModule = R"(
    HloModule m

    ENTRY main {
      p0 = f32[1] parameter(0)
      ROOT add0 = f32[1] add(p0,p0)
    }
  )";
  auto config = GetModuleConfig(kFdoProfile);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule, config));

  TF_EXPECT_OK(ScheduleModule(module.get()));
}

TEST_F(GpuLatencyHidingSchedulerBaseTest,
       MultipleParallelResourceShouldOverlapCollectives) {
  absl::string_view kFdoProfile = R"pb(
    costs { name: "add_0" cost_us: 100000.0 }
    costs { name: "ar_0" cost_us: 10.0 }
    costs { name: "rs_0" cost_us: 10.0 }
  )pb";
  ;
  absl::string_view kHloModule = R"(
    HloModule m

    reduce {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT _ = f32[] add(x, y)
    }

    ENTRY main {
      p0 = f32[] parameter(0)
      p1 = f32[2] parameter(1)
      p2 = f32[2] parameter(2)
      ar_0 = f32[] all-reduce-start(p0), to_apply=reduce
      ar_1 = f32[] all-reduce-done(ar_0)
      rs_0 = ((f32[2]), f32[1]) reduce-scatter-start(p1), to_apply=reduce, dimensions={0}
      rs_1 = f32[1] reduce-scatter-done(rs_0)
      add_0 = f32[2] add(p1, p2)
      ROOT _ = (f32[], f32[1], f32[2]) tuple(ar_1, rs_1, add_0)
    }
  )";

  auto config = GetModuleConfig(kFdoProfile);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloModule, config));

  TF_EXPECT_OK(ScheduleModule(module.get(), /*num_parallel_resources=*/2));
  auto schedule = module->schedule();
  std::vector<HloInstruction*> instruction_sequence =
      schedule.sequence(module->entry_computation()).instructions();
  // Since we allow 2 collectives in-flight, we should expect this pattern:
  // ar(rs)-start -> rs(ar)-start -> add -> ar(rs)-done -> ar(rs)-done
  EXPECT_TRUE(GetIndexByName(instruction_sequence, "ar_0") <
                  GetIndexByName(instruction_sequence, "rs_1") &&
              GetIndexByName(instruction_sequence, "rs_0") <
                  GetIndexByName(instruction_sequence, "ar_1"));
  EXPECT_TRUE(GetIndexByName(instruction_sequence, "add_0") >
                  GetIndexByName(instruction_sequence, "ar_0") &&
              GetIndexByName(instruction_sequence, "add_0") >
                  GetIndexByName(instruction_sequence, "rs_0") &&
              GetIndexByName(instruction_sequence, "add_0") <
                  GetIndexByName(instruction_sequence, "ar_1") &&
              GetIndexByName(instruction_sequence, "add_0") <
                  GetIndexByName(instruction_sequence, "rs_1"));
}

}  // namespace
}  // namespace xla::gpu

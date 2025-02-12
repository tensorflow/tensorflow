/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/latency_hiding_scheduler.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/transforms/collectives/async_collective_creator.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/legalize_scheduling_annotations.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

namespace {

constexpr int kMaxConcurrentAsyncCollectivePermutes = 5;

int PositionInVector(absl::Span<HloInstruction* const> vec,
                     const HloInstruction* element) {
  return std::distance(vec.begin(), std::find(vec.begin(), vec.end(), element));
}

bool MaxConcurrentCollectivePermutesBelowThreshold(
    absl::Span<HloInstruction* const> instruction_sequence) {
  int max_concurrent_collective_permutes = 0;
  int num_concurrent_collective_permutes = 0;
  for (HloInstruction* instruction : instruction_sequence) {
    if (instruction->opcode() == HloOpcode::kCollectivePermuteStart) {
      num_concurrent_collective_permutes += 1;
      max_concurrent_collective_permutes =
          std::max(max_concurrent_collective_permutes,
                   num_concurrent_collective_permutes);
    }
    if (instruction->opcode() == HloOpcode::kCollectivePermuteDone) {
      num_concurrent_collective_permutes -= 1;
    }
  }
  int max_num_collective_permutes_threshold =
      kMaxConcurrentAsyncCollectivePermutes;
  return max_concurrent_collective_permutes <=
         max_num_collective_permutes_threshold;
}

int GetIndex(absl::Span<HloInstruction* const> instruction_sequence,
             absl::string_view hlo_name) {
  return absl::c_find_if(instruction_sequence,
                         [hlo_name](HloInstruction* instruction) {
                           return instruction->name() == hlo_name;
                         }) -
         instruction_sequence.begin();
}

int GetOpcodeIndexUsingMetaData(
    HloOpcode opcode, absl::Span<HloInstruction* const> instruction_sequence,
    absl::string_view metadata_name) {
  return absl::c_find_if(instruction_sequence,
                         [metadata_name, opcode](HloInstruction* instruction) {
                           return instruction->metadata().op_name() ==
                                      metadata_name &&
                                  instruction->opcode() == opcode;
                         }) -
         instruction_sequence.begin();
}

SchedulerConfig GetDefaultSchedConfig() {
  SchedulerConfig sched_cfg;
  sched_cfg.collective_permute_overlap_limit =
      kMaxConcurrentAsyncCollectivePermutes;
  sched_cfg.send_recv_overlap_limit = INT32_MAX;
  return sched_cfg;
}

class TestLatencyEstimator : public LatencyEstimator {
 public:
  TimeCost GetLatencyBetween(const HloGraphNode& from,
                             const HloGraphNode& target) const override {
    static constexpr TimeCost kLowLatency = 1.0;
    if (from.GetInstr().opcode() == HloOpcode::kCollectivePermuteStart &&
        target.GetInstr().opcode() == HloOpcode::kCollectivePermuteDone) {
      return kLowLatency *
             ShapeUtil::ElementsIn(from.GetInstr().operand(0)->shape());
    }
    return kLowLatency;
  }
  TimeCost NodeCost(const HloInstruction* instr) const override {
    if (instr->IsLoopFusion()) {
      return instr->shape().IsTuple()
                 ? kMediumCost
                 : kLowCost * ShapeUtil::ElementsIn(instr->shape());
    }
    if (instr->IsOutputFusion() || instr->opcode() == HloOpcode::kConvolution) {
      return instr->shape().IsTuple()
                 ? kHighCost
                 : kMediumCost * ShapeUtil::ElementsIn(instr->shape());
    }
    return kLowCost;
  }
  int CyclesPerMicrosecond() const override { return 1; }

 public:
  static constexpr TimeCost kLowCost = 1.0;
  static constexpr TimeCost kMediumCost = 1000.0;
  static constexpr TimeCost kHighCost = 5000.0;
};

absl::StatusOr<bool> RunScheduler(
    HloModule* module, SchedulerConfig sched_config = GetDefaultSchedConfig(),
    std::unique_ptr<LatencyEstimator> latency_estimator =
        std::make_unique<ApproximateLatencyEstimator>(),
    std::unique_ptr<AsyncTracker> async_tracker = nullptr) {
  AsyncCollectiveCreator::CollectiveCreatorConfig config{
      /*convert_all_reduce=*/HloPredicateTrue,
      /*convert_all_gather=*/HloPredicateTrue,
      /*convert_collective_broadcast=*/HloPredicateTrue,
      /*convert_collective_permute=*/HloPredicateTrue};
  TF_ASSIGN_OR_RETURN(bool value,
                      AsyncCollectiveCreator(std::move(config)).Run(module));
  TF_ASSIGN_OR_RETURN(value, LegalizeSchedulingAnnotations(
                                 LegalizeSchedulingAnnotations::Config())
                                 .Run(module));
  HloCostAnalysis::ShapeSizeFunction shape_size_bytes =
      [&shape_size_bytes](const Shape& shape) -> int64_t {
    int64_t shape_size = 0;
    if (shape.IsToken()) {
      return 0;
    }
    if (shape.IsTuple()) {
      for (auto& sub_shape : shape.tuple_shapes()) {
        shape_size += shape_size_bytes(sub_shape);
      }
      return shape_size;
    }
    return ShapeUtil::ByteSizeOfElements(shape);
  };
  if (!async_tracker) {
    async_tracker = std::make_unique<AsyncTracker>(sched_config);
  }
  auto scheduler_core = std::make_unique<DefaultSchedulerCore>(
      shape_size_bytes, async_tracker.get(), latency_estimator.get(),
      sched_config);
  TF_ASSIGN_OR_RETURN(
      value, LatencyHidingScheduler(std::move(latency_estimator),
                                    std::move(async_tracker),
                                    std::move(scheduler_core), shape_size_bytes)
                 .Run(module));

  return value;
}

}  // namespace

class LatencyHidingSchedulerTest : public HloTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> ParseHloText(
      absl::string_view hlo_string) {
    return ParseAndReturnVerifiedModule(hlo_string, GetModuleConfigForTest());
  }
};

TEST_F(LatencyHidingSchedulerTest, AllGatherAsyncSimple) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true


ENTRY %module {
  %constant.19 = u32[] constant(0)
  %replica_id = u32[]{:T(128)} replica-id()
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %color_operand.1 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(
    f32[]{:T(128)} %convert), dimensions={}
  %ag-start = (f32[8,256,256], f32[16,256,256]) all-gather-start(
    f32[8,256,256] %color_operand.1), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-done = f32[16,256,256] all-gather-done(
    (f32[8,256,256], f32[16,256,256]) %ag-start),
    metadata={op_type="AllGather" op_name="ag0"}
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT a2 = f32[16,256,256]{2,1,0} add(%ag-done, c0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  EXPECT_EQ(GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherDone,
                                        new_instruction_sequence, "ag0"),
            GetIndex(new_instruction_sequence, "a2") - 1);
}

TEST_F(LatencyHidingSchedulerTest, AllGatherAsyncBalance) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY %module {
  %constant.19 = u32[] constant(0)
  %replica_id = u32[]{:T(128)} replica-id()
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %color_operand.1 = f32[1,8,256,256]{3,2,1,0:T(8,128)} broadcast(
    f32[]{:T(128)} %convert), dimensions={}
  %ag-start = (f32[1,8,256,256], f32[2,8,256,256]) all-gather-start(
    f32[1,8,256,256] %color_operand.1), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-done = f32[2,8,256,256] all-gather-done(
    (f32[1,8,256,256], f32[2,8,256,256]) %ag-start),
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-done-bc = f32[16,256,256] bitcast(f32[2,8,256,256] %ag-done),
    metadata={op_type="Bitcast" op_name="ag0"}
  %ag-start.2 = (f32[1,8,256,256], f32[2,8,256,256]) all-gather-start(
    f32[1,8,256,256] %color_operand.1), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag1"}
  %ag-done.2 = f32[2,8,256,256] all-gather-done(
    (f32[1,8,256,256], f32[2,8,256,256]) %ag-start.2),
    metadata={op_type="AllGather" op_name="ag1"}
  %ag-done-bc.2 = f32[16,256,256] bitcast(f32[2,8,256,256] %ag-done.2),
    metadata={op_type="Bitcast" op_name="ag1"}
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
    metadata={op_type="AllGather" op_name="c0"}
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
    metadata={op_type="AllGather" op_name="c1"}
  a2 = f32[16,256,256]{2,1,0} add(c1, c0)
  ROOT t = (f32[16,256,256], f32[16,256,256], f32[16,256,256]) tuple(a2, %ag-done-bc.2, %ag-done-bc)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // We expect that the scheduling would look like this:
  //   %ag-start = all-gather-start()
  //   %c0 = convolution()
  //   %ag-done = all-gather-done()
  //   %ag-start.2 = all-gather-start()
  //   %c1 = convolution()
  //   %ag-done.2 = f32[2,8,256,256]{3,2,1,0} all-gather-done()
  // This means that the all-gathers are balanced over the two convolutions
  // rather than being unbalanced (one of the two all-gathers overlaps with
  // both the convolutons and the other with nothing).
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kConvolution,
                                        new_instruction_sequence, "c0"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherDone,
                                        new_instruction_sequence, "ag0"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kConvolution,
                                        new_instruction_sequence, "c0"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherStart,
                                        new_instruction_sequence, "ag0"));
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kConvolution,
                                        new_instruction_sequence, "c1"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherDone,
                                        new_instruction_sequence, "ag1"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kConvolution,
                                        new_instruction_sequence, "c1"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherStart,
                                        new_instruction_sequence, "ag1"));
}

TEST_F(LatencyHidingSchedulerTest, AllGatherAsyncReshaped) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true


ENTRY %module {
  %constant.19 = u32[] constant(0)
  %replica_id = u32[]{:T(128)} replica-id()
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %color_operand.1 = f32[1,8,256,256]{3,2,1,0:T(8,128)} broadcast(
    f32[]{:T(128)} %convert), dimensions={}
  %ag-start = (f32[1,8,256,256], f32[2,8,256,256]) all-gather-start(
    f32[1,8,256,256] %color_operand.1), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-done = f32[2,8,256,256] all-gather-done(
    (f32[1,8,256,256], f32[2,8,256,256]) %ag-start),
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-done-bc = f32[16,256,256] bitcast(f32[2,8,256,256] %ag-done),
    metadata={op_type="Bitcast" op_name="ag0"}
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT a2 = f32[16,256,256]{2,1,0} add(%ag-done-bc, c0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  EXPECT_EQ(GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherDone,
                                        new_instruction_sequence, "ag0"),
            GetIndex(new_instruction_sequence, "a2") - 2);
  EXPECT_EQ(GetOpcodeIndexUsingMetaData(HloOpcode::kBitcast,
                                        new_instruction_sequence, "ag0"),
            GetIndex(new_instruction_sequence, "a2") - 1);
}

TEST_F(LatencyHidingSchedulerTest, AllGatherAsyncOverlapped) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true


ENTRY %module {
  %constant.19 = u32[] constant(1)
  %replica_id = u32[]{:T(128)} replica-id()
  %add.1 = u32[]{:T(128)} add(replica_id, constant.19)
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %convert.1 = f32[]{:T(128)} convert(u32[]{:T(128)} %add.1)
  %color_operand.1 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(f32[]{:T(128)} %convert), dimensions={}
  %color_operand.2 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(f32[]{:T(128)} %convert.1), dimensions={}
  %ag-start = (f32[8,256,256], f32[16,256,256]) all-gather-start(f32[8,256,256] %color_operand.1), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-start.2 = (f32[8,256,256], f32[16,256,256]) all-gather-start(f32[8,256,256] %color_operand.2), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag1"}
  %ag-done = f32[16,256,256] all-gather-done((f32[8,256,256], f32[16,256,256]) %ag-start),
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-done.2 = f32[16,256,256] all-gather-done((f32[8,256,256], f32[16,256,256]) %ag-start.2),
    metadata={op_type="AllGather" op_name="ag1"}
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT a2 = f32[16,256,256]{2,1,0} add(%ag-done, %ag-done.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherDone,
                                        new_instruction_sequence, "ag0"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherStart,
                                        new_instruction_sequence, "ag1"));
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherStart,
                                        new_instruction_sequence, "ag0"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherDone,
                                        new_instruction_sequence, "ag0"));
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherStart,
                                        new_instruction_sequence, "ag1"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherDone,
                                        new_instruction_sequence, "ag1"));
}

TEST_F(LatencyHidingSchedulerTest, AllGatherAsyncOverlapped2) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true


ENTRY %module {
  %constant.19 = u32[] constant(1)
  %replica_id = u32[]{:T(128)} replica-id()
  %add.1 = u32[]{:T(128)} add(replica_id, constant.19)
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %convert.1 = f32[]{:T(128)} convert(u32[]{:T(128)} %add.1)
  %color_operand.1 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(f32[]{:T(128)} %convert), dimensions={}
  %color_operand.2 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(f32[]{:T(128)} %convert.1), dimensions={}
  %ag-start = (f32[8,256,256], f32[16,256,256]) all-gather-start(f32[8,256,256] %color_operand.1), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-start.2 = (f32[8,256,256], f32[16,256,256]) all-gather-start(f32[8,256,256] %color_operand.2), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag1"}
  %ag-done = f32[16,256,256] all-gather-done((f32[8,256,256], f32[16,256,256]) %ag-start),
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-done.2 = f32[16,256,256] all-gather-done((f32[8,256,256], f32[16,256,256]) %ag-start.2),
    metadata={op_type="AllGather" op_name="ag1"}
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  c0 = f32[16,256,256]{2,1,0} convolution(ag-done, ag-done.2),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT a2 = f32[16,256,256]{2,1,0} add(%c0, %c1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherDone,
                                        new_instruction_sequence, "ag0"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherStart,
                                        new_instruction_sequence, "ag1"));
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherStart,
                                        new_instruction_sequence, "ag0"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherDone,
                                        new_instruction_sequence, "ag0"));
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherStart,
                                        new_instruction_sequence, "ag1"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherDone,
                                        new_instruction_sequence, "ag1"));
}

TEST_F(LatencyHidingSchedulerTest, AllGatherAsyncOverlapped3) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true


ENTRY %module {
  %constant.19 = u32[] constant(1)
  %replica_id = u32[]{:T(128)} replica-id()
  %add.1 = u32[]{:T(128)} add(replica_id, constant.19)
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %convert.1 = f32[]{:T(128)} convert(u32[]{:T(128)} %add.1)
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  %color_operand.1 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(f32[]{:T(128)} %convert), dimensions={}
  %color_operand.2 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(f32[]{:T(128)} %convert.1), dimensions={}
  %ag-start = (f32[8,256,256], f32[16,256,256]) all-gather-start(f32[8,256,256] %color_operand.1), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-start.2 = (f32[8,256,256], f32[16,256,256]) all-gather-start(f32[8,256,256] %color_operand.2), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag1"}
  %ag-done = f32[16,256,256] all-gather-done((f32[8,256,256], f32[16,256,256]) %ag-start),
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-done.2 = f32[16,256,256] all-gather-done((f32[8,256,256], f32[16,256,256]) %ag-start.2),
    metadata={op_type="AllGather" op_name="ag1"}
  c0 = f32[16,256,256]{2,1,0} convolution(ag-done, ag-done.2),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT a2 = f32[16,256,256]{2,1,0} add(%ag-done, %ag-done.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherDone,
                                        new_instruction_sequence, "ag0"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherStart,
                                        new_instruction_sequence, "ag1"));
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherStart,
                                        new_instruction_sequence, "ag0"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherDone,
                                        new_instruction_sequence, "ag0"));
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherStart,
                                        new_instruction_sequence, "ag1"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllGatherDone,
                                        new_instruction_sequence, "ag1"));
}

TEST_F(LatencyHidingSchedulerTest, AllReduceAsyncBalance) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

%add {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %a = f32[] add(p0, p1)
}

ENTRY %module {
  %constant.19 = u32[] constant(0)
  %replica_id = u32[]{:T(128)} replica-id()
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %color_operand.1 = f32[2,8,256,256]{3,2,1,0:T(8,128)} broadcast(
    f32[]{:T(128)} %convert), dimensions={}
  %color_operand.2 = f32[2,8,256,256]{3,2,1,0:T(8,128)} broadcast(
    f32[]{:T(128)} %convert), dimensions={}
  %ar-start = f32[2,8,256,256] all-reduce-start(
    f32[2,8,256,256] %color_operand.1), replica_groups={{0,1}}, to_apply=%add,
    metadata={op_type="AllReduce" op_name="ar0"}
  %ar-start.2 = f32[2,8,256,256] all-reduce-start(
    f32[2,8,256,256] %color_operand.2), replica_groups={{0,1}}, to_apply=%add,
    metadata={op_type="AllReduce" op_name="ar1"}
  %ar-done = f32[2,8,256,256] all-reduce-done(
    f32[2,8,256,256] %ar-start),
    metadata={op_type="AllReduce" op_name="ar0"}
  %ar-done-bc = f32[16,256,256] bitcast(f32[2,8,256,256] %ar-done),
    metadata={op_type="Bitcast" op_name="ar0"}
  %ar-done.2 = f32[2,8,256,256] all-reduce-done(
    f32[2,8,256,256] %ar-start.2),
    metadata={op_type="AllReduce" op_name="ar1"}
  %ar-done-bc.2 = f32[16,256,256] bitcast(f32[2,8,256,256] %ar-done.2),
    metadata={op_type="Bitcast" op_name="ar1"}
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
    metadata={op_type="AllReduce" op_name="c0"}
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
    metadata={op_type="AllReduce" op_name="c1"}
  a2 = f32[16,256,256]{2,1,0} add(c1, c0)
  ROOT t = (f32[16,256,256], f32[16,256,256], f32[16,256,256]) tuple(a2, %ar-done-bc.2, %ar-done-bc)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // We expect that the scheduling would look like this:
  //   %ar-start = all-reduce-start()
  //   %c0 = convolution()
  //   %ar-done = all-reduce-done()
  //   %ar-start.2 = all-reduce-start()
  //   %c1 = convolution()
  //   %ar-done.2 = f32[2,8,256,256]{3,2,1,0} all-reduce-done()
  // This means that the all-reduces are balanced over the two convolutions
  // rather than being unbalanced (one of the two all-reduces overlaps with
  // both the convolutons and the other with nothing).
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kConvolution,
                                        new_instruction_sequence, "c0"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllReduceDone,
                                        new_instruction_sequence, "ar0"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kConvolution,
                                        new_instruction_sequence, "c0"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllReduceStart,
                                        new_instruction_sequence, "ar0"));
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kConvolution,
                                        new_instruction_sequence, "c1"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllReduceDone,
                                        new_instruction_sequence, "ar1"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kConvolution,
                                        new_instruction_sequence, "c1"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAllReduceStart,
                                        new_instruction_sequence, "ar1"));
}

TEST_F(LatencyHidingSchedulerTest, WhileLoopAliasingBug) {
  // Test for the bug in b/185524709. The second collective permute should be
  // scheduled after add0 since gte0, bitcast, and the destination buffer for
  // the second collective permute alias with each other.
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

while_cond {
  param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
  ROOT gte = pred[] get-tuple-element(param), index=2
}

while_body {
  param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
  gte0 = bf16[8]{0} get-tuple-element(param), index=0
  gte1 = pred[] get-tuple-element(param), index=2
  bitcast = bf16[8]{0} bitcast(gte0)
  collective-permute.1 = bf16[8]{0} collective-permute(gte0), source_target_pairs={{0,1},{1,2},{2,3}}
  add0 = bf16[8]{0} add(collective-permute.1, bitcast)
  negate = bf16[8]{0} negate(add0)
  collective-permute.2 = bf16[8]{0} collective-permute(collective-permute.1), source_target_pairs={{1,0},{0,3},{3,2}}
  ROOT tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.2, negate, gte1)
}

ENTRY entry {
  p0 = bf16[8]{0} parameter(0)
  p1 = bf16[8]{0} parameter(1)
  p2 = pred[] parameter(2)
  tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(p0, p1, p2)
  while = (bf16[8]{0}, bf16[8]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
  gte0 = bf16[8]{0} get-tuple-element(while), index=0
  gte1 = bf16[8]{0} get-tuple-element(while), index=1
  ROOT add = bf16[8]{0} add(gte0, gte1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* while_body = hlo_module->GetComputationWithName("while_body");

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(while_body).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
  // Find the collective permute start that was converted from
  // collective-permute.2 and ensure it was scheduled after add0. Otherwise,
  // gte0 will be clobbered by the async collective permute.
  const HloInstruction* cp_start =
      while_body->root_instruction()->operand(0)->operand(0);
  EXPECT_EQ(cp_start->opcode(), HloOpcode::kCollectivePermuteStart);
  EXPECT_LT(GetIndex(new_instruction_sequence, "add0"),
            GetIndex(new_instruction_sequence, cp_start->name()));
}

TEST_F(LatencyHidingSchedulerTest, WhileLoopAliasingBug2) {
  // Like WhileLoopAliasingBug above, but this time the input buffer of the
  // first collective permute aliases with the output buffer of the second
  // collective permute..
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

while_cond {
  param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
  ROOT gte = pred[] get-tuple-element(param), index=2
}

while_body {
  param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
  gte0 = bf16[8]{0} get-tuple-element(param), index=0
  gte1 = bf16[8]{0} get-tuple-element(param), index=1
  gte2 = pred[] get-tuple-element(param), index=2
  negate1 = bf16[8]{0} negate(gte1)
  collective-permute.1 = bf16[8]{0} collective-permute(gte0), source_target_pairs={{0,1},{1,2},{2,3}}
  negate0 = bf16[8]{0} negate(collective-permute.1)
  collective-permute.2 = bf16[8]{0} collective-permute(negate1), source_target_pairs={{1,0},{0,3},{3,2}}
  ROOT tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.2, negate0, gte2)
}

ENTRY entry {
  p0 = bf16[8]{0} parameter(0)
  p1 = bf16[8]{0} parameter(1)
  p2 = pred[] parameter(2)
  tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(p0, p1, p2)
  while = (bf16[8]{0}, bf16[8]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
  gte0 = bf16[8]{0} get-tuple-element(while), index=0
  gte1 = bf16[8]{0} get-tuple-element(while), index=1
  ROOT add = bf16[8]{0} add(gte0, gte1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* while_body = hlo_module->GetComputationWithName("while_body");

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(while_body).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
  const HloInstruction* cp_start_2 =
      while_body->root_instruction()->operand(0)->operand(0);
  EXPECT_EQ(cp_start_2->opcode(), HloOpcode::kCollectivePermuteStart);
  const HloInstruction* cp_done_1 =
      while_body->root_instruction()->operand(1)->operand(0);
  EXPECT_EQ(cp_done_1->opcode(), HloOpcode::kCollectivePermuteDone);
  EXPECT_LT(GetIndex(new_instruction_sequence, cp_done_1->name()),
            GetIndex(new_instruction_sequence, cp_start_2->name()));
}

TEST_F(LatencyHidingSchedulerTest, SingleCollectivePermuteTest) {
  absl::string_view hlo_string = R"(
    HloModule single_collective_permute_test, is_scheduled=true
    ENTRY after_optimizations_test {
    %parameter.1 = bf16[8]{0} parameter(0), sharding={replicated}
    ROOT %collective-permute.1 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.1), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
  EXPECT_EQ(new_instruction_sequence.size(), 3);
  EXPECT_EQ(new_instruction_sequence[1]->opcode(),
            HloOpcode::kCollectivePermuteStart);
  EXPECT_EQ(new_instruction_sequence[2]->opcode(),
            HloOpcode::kCollectivePermuteDone);
}

TEST_F(LatencyHidingSchedulerTest, InplaceUpdateCPTest) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

%fused_computation.1 (param_0.1: f32[4,4,128], param_1.2: u32[]) -> f32[4,4,128] {
  %param_0.1 = f32[4,4,128]{2,1,0:T(4,128)} parameter(0)
  %constant.15 = f32[]{:T(128)} constant(1)
  %broadcast.2 = f32[2,4,128]{2,1,0:T(4,128)} broadcast(f32[]{:T(128)} %constant.15), dimensions={}
  %param_1.2 = u32[] parameter(1)
  %constant.14 = u32[] constant(0)
  ROOT %dynamic-update-slice.1 = f32[4,4,128]{2,1,0:T(4,128)} dynamic-update-slice(f32[4,4,128]{2,1,0:T(4,128)} %param_0.1, f32[2,4,128]{2,1,0:T(4,128)} %broadcast.2, u32[] %param_1.2, u32[] %constant.14, u32[] %constant.14)
}

ENTRY %module_spmd () -> f32[4,4,128] {
  %constant.8 = u32[] constant(0)
  %constant.5 = u32[] constant(2)
  %tuple.1 = (u32[], u32[], u32[]) tuple(u32[] %constant.8, u32[] %constant.8, u32[] %constant.8)
  %tuple = (u32[], u32[], u32[]) tuple(u32[] %constant.5, u32[] %constant.8, u32[] %constant.8)
  %custom-call = f32[4,4,128]{2,1,0:T(4,128)} custom-call(), custom_call_target="AllocateBuffer"
  %fusion.1 = f32[4,4,128]{2,1,0:T(4,128)} fusion(f32[4,4,128]{2,1,0:T(4,128)} %custom-call, u32[] %constant.5), kind=kLoop, calls=%fused_computation.1
  %collective-permute = f32[4,4,128]{2,1,0:T(4,128)} collective-permute(f32[4,4,128]{2,1,0:T(4,128)} %fusion.1, f32[4,4,128]{2,1,0:T(4,128)} %fusion.1, (u32[], u32[], u32[]) %tuple, (u32[], u32[], u32[]) %tuple.1), channel_id=958, source_target_pairs={{0,4},{4,0},{1,5},{5,1},{2,6},{6,2},{3,7},{7,3}}, slice_sizes={{2,4,128}}, backend_config="{\"flag_configs\":[],\"barrier_config\":{\"barrier_type\":\"CUSTOM\",\"id\":\"0\"},\"scoped_memory_configs\":[]}"
  ROOT %copy.3 = f32[4,4,128]{2,1,0:T(4,128)} copy(f32[4,4,128]{2,1,0:T(4,128)} %collective-permute)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  EXPECT_EQ(new_instruction_sequence.size(),
            original_instruction_sequence.size() + 1);
}

TEST_F(LatencyHidingSchedulerTest, InplaceUpdateCPTest2) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

%sum (x.336: f32[], y.336: f32[]) -> f32[] {
  %x.336 = f32[]{:T(128)} parameter(0)
  %y.336 = f32[]{:T(128)} parameter(1)
  ROOT %add.5252 = f32[]{:T(128)} add(f32[]{:T(128)} %x.336, f32[]{:T(128)} %y.336)
}

ENTRY %module () -> f32[33708,1024] {
  %constant.19 = u32[] constant(0)
  %replica_id = u32[]{:T(128)} replica-id()
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %color_operand.1 = f32[2128,8,128]{2,1,0:T(8,128)} broadcast(f32[]{:T(128)} %convert), dimensions={}
  %all-gather.1 = f32[4256,8,128]{2,1,0:T(8,128)} all-gather(f32[2128,8,128]{2,1,0:T(8,128)} %color_operand.1), replica_groups={{0,6},{2,4},{3,5},{1,7}}, dimensions={0}
  %custom-call = f32[33712,8,128]{2,1,0:T(8,128)} custom-call(), custom_call_target="AllocateBuffer"
  %dynamic-update-slice = f32[33712,8,128]{2,1,0:T(8,128)} dynamic-update-slice(f32[33712,8,128]{2,1,0:T(8,128)} %custom-call, f32[4256,8,128]{2,1,0:T(8,128)} %all-gather.1, u32[] %constant.19, u32[] %constant.19, u32[] %constant.19)
  %tuple.7 = (u32[], u32[], u32[]) tuple(u32[] %constant.19, u32[] %constant.19, u32[] %constant.19)
  %constant.20 = u32[] constant(4256)
  %tuple.8 = (u32[], u32[], u32[]) tuple(u32[] %constant.20, u32[] %constant.19, u32[] %constant.19)
  %collective-permute.3 = f32[33712,8,128]{2,1,0:T(8,128)} collective-permute(f32[33712,8,128]{2,1,0:T(8,128)} %dynamic-update-slice, f32[33712,8,128]{2,1,0:T(8,128)} %dynamic-update-slice, (u32[], u32[], u32[]) %tuple.7, (u32[], u32[], u32[]) %tuple.8), source_target_pairs={{0,2},{2,3},{3,1},{1,0},{6,4},{4,5},{5,7},{7,6}}, slice_sizes={{4256,8,128}}
  %tuple.9 = (u32[], u32[], u32[]) tuple(u32[] %constant.20, u32[] %constant.19, u32[] %constant.19)
  %constant.21 = u32[] constant(8512)
  %tuple.10 = (u32[], u32[], u32[]) tuple(u32[] %constant.21, u32[] %constant.19, u32[] %constant.19)
  %collective-permute.4 = f32[33712,8,128]{2,1,0:T(8,128)} collective-permute(f32[33712,8,128]{2,1,0:T(8,128)} %collective-permute.3, f32[33712,8,128]{2,1,0:T(8,128)} %collective-permute.3, (u32[], u32[], u32[]) %tuple.9, (u32[], u32[], u32[]) %tuple.10), source_target_pairs={{0,2},{2,3},{3,1},{1,0},{6,4},{4,5},{5,7},{7,6}}, slice_sizes={{4256,8,128}}
  %tuple.11 = (u32[], u32[], u32[]) tuple(u32[] %constant.21, u32[] %constant.19, u32[] %constant.19)
  %constant.22 = u32[] constant(12768)
  %tuple.12 = (u32[], u32[], u32[]) tuple(u32[] %constant.22, u32[] %constant.19, u32[] %constant.19)
  %collective-permute.5 = f32[33712,8,128]{2,1,0:T(8,128)} collective-permute(f32[33712,8,128]{2,1,0:T(8,128)} %collective-permute.4, f32[33712,8,128]{2,1,0:T(8,128)} %collective-permute.4, (u32[], u32[], u32[]) %tuple.11, (u32[], u32[], u32[]) %tuple.12), source_target_pairs={{0,2},{2,3},{3,1},{1,0},{6,4},{4,5},{5,7},{7,6}}, slice_sizes={{4256,8,128}}
  ROOT %bitcast.16 = f32[33708,1024]{1,0:T(8,128)} bitcast(f32[33712,8,128]{2,1,0:T(8,128)} %collective-permute.5)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
  EXPECT_EQ(new_instruction_sequence.size(),
            original_instruction_sequence.size() + 4);
}

TEST_F(LatencyHidingSchedulerTest, TwoCollectivePermuteTypesOverlap) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, f32[16,128,256]{2,1,0}) parameter(0)
  gte0 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=0
  gte1 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=1
  cp0 = f32[16,64,256]{2,1,0} collective-permute(gte0),
    source_target_pairs={{0,1},{1,0}},
    metadata={op_type="CollectivePermute" op_name="cp0"}
  cp1 = f32[16,64,256]{2,1,0} collective-permute(cp0),
    source_target_pairs={{0,1},{1,0}},
    metadata={op_type="CollectivePermute" op_name="cp1"}
  c0 = f32[16,256,256]{2,1,0} convolution(gte0, gte1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  cp2 = f32[16,64,256]{2,1,0} collective-permute(gte1),
    source_target_pairs={{0,1},{1,0}},
    metadata={op_type="CollectivePermute" op_name="cp2"}
  c1 = f32[16,256,256]{2,1,0} convolution(cp0, gte1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  cp3 = f32[16,64,256]{2,1,0} collective-permute(cp2),
    source_target_pairs={{0,1},{1,0}},
    metadata={op_type="CollectivePermute" op_name="cp3"}
  gte2 = f32[16,128,256]{2,1,0} get-tuple-element(param), index=2
  const0 = u32[] constant(0)
  const1 = u32[] constant(8)
  tuple0 = (u32[], u32[], u32[]) tuple(u32[] const0, u32[] const0, u32[] const0)
  tuple1 = (u32[], u32[], u32[]) tuple(u32[] const1, u32[] const0, u32[] const0)
  cp4 = f32[16,128,256]{2,1,0} collective-permute(gte2, gte2, tuple0, tuple1),
    source_target_pairs={{2,3},{3,2}},
    slice_sizes={{8,128,256}},
    metadata={op_type="CollectivePermute" op_name="cp4"}
  cp5 = f32[16,128,256]{2,1,0} collective-permute(cp4, cp4, tuple0, tuple1),
    source_target_pairs={{2,3},{3,2}},
    slice_sizes={{8,128,256}},
    metadata={op_type="CollectivePermute" op_name="cp5"}
  ROOT tuple = (f32[16,256,256]{2,1,0}, f32[16,64,256]{2,1,0}, f32[16,128,256]{2,1,0}) tuple(c1, cp3, cp5)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  EXPECT_EQ(new_instruction_sequence.size(),
            original_instruction_sequence.size() + 6);

  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteStart,
                                        new_instruction_sequence, "cp0"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteStart,
                                        new_instruction_sequence, "cp2"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteStart,
                                        new_instruction_sequence, "cp4"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteDone,
                                        new_instruction_sequence, "cp0"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteDone,
                                        new_instruction_sequence, "cp2"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteDone,
                                        new_instruction_sequence, "cp4"),
            GetIndex(new_instruction_sequence, "c0"));

  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteStart,
                                        new_instruction_sequence, "cp1"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteStart,
                                        new_instruction_sequence, "cp3"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteStart,
                                        new_instruction_sequence, "cp5"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteDone,
                                        new_instruction_sequence, "cp1"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteDone,
                                        new_instruction_sequence, "cp3"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteDone,
                                        new_instruction_sequence, "cp5"),
            GetIndex(new_instruction_sequence, "c1"));
}

TEST_F(LatencyHidingSchedulerTest, SerialCollectivePermutesTest) {
  absl::string_view hlo_string = R"(
    HloModule serial_collective_permute_test, is_scheduled=true
    ENTRY after_optimizations_test {
    %parameter.1 = bf16[8]{0} parameter(0)
    %collective-permute.2 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.1), source_target_pairs={{0,1},{1,2},{2,3}}
    %add.3 = bf16[8]{0} add(%parameter.1, %parameter.1)
    %add.4 = bf16[8]{0} add(%add.3, parameter.1)
    %add.5 = bf16[8]{0} add(%collective-permute.2, %add.4)
    %collective-permute.6 = bf16[8]{0} collective-permute(bf16[8]{0} add.5), source_target_pairs={{1,0},{0,3},{3,2}}
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  EXPECT_EQ(original_instruction_sequence.size(), 6);
  EXPECT_EQ(new_instruction_sequence.size(), 8);
  // The new instruction sequence preserves the scheduling order of
  // non-collective-permute instructions.
  EXPECT_LT(PositionInVector(new_instruction_sequence,
                             original_instruction_sequence[0]),
            PositionInVector(new_instruction_sequence,
                             original_instruction_sequence[2]));
  EXPECT_LT(PositionInVector(new_instruction_sequence,
                             original_instruction_sequence[2]),
            PositionInVector(new_instruction_sequence,
                             original_instruction_sequence[3]));
  EXPECT_LT(PositionInVector(new_instruction_sequence,
                             original_instruction_sequence[3]),
            PositionInVector(new_instruction_sequence,
                             original_instruction_sequence[4]));
  EXPECT_EQ(original_instruction_sequence[0]->user_count(), 3);
  EXPECT_EQ(original_instruction_sequence[0]->users()[0]->opcode(),
            HloOpcode::kCollectivePermuteStart);
  HloInstruction* collective_permute_start_1 =
      original_instruction_sequence[0]->users()[0];
  // Collective-permute-start immediately follows its operand.
  EXPECT_EQ(
      PositionInVector(new_instruction_sequence,
                       original_instruction_sequence[0]) +
          1,
      PositionInVector(new_instruction_sequence, collective_permute_start_1));
  EXPECT_EQ(collective_permute_start_1->user_count(), 1);
  EXPECT_EQ(collective_permute_start_1->users()[0]->opcode(),
            HloOpcode::kCollectivePermuteDone);
  HloInstruction* collective_permute_done_1 =
      collective_permute_start_1->users()[0];
  // Collective-permute-done immediately leads one of its users.
  EXPECT_TRUE(
      (PositionInVector(new_instruction_sequence, collective_permute_done_1) +
           1 ==
       PositionInVector(new_instruction_sequence,
                        collective_permute_done_1->users()[0])) ||
      (PositionInVector(new_instruction_sequence, collective_permute_done_1) +
           1 ==
       PositionInVector(new_instruction_sequence,
                        collective_permute_done_1->users()[1])));
  // Collective-permute-done is scheduled before its users.
  EXPECT_TRUE(
      (PositionInVector(new_instruction_sequence, collective_permute_done_1) <
       PositionInVector(new_instruction_sequence,
                        collective_permute_done_1->users()[0])));
  // The second collective-permute starts after the first one is done.
  EXPECT_EQ(new_instruction_sequence[7]->opcode(),
            HloOpcode::kCollectivePermuteDone);
  EXPECT_GT(
      PositionInVector(new_instruction_sequence,
                       new_instruction_sequence[7]->operand(0)),
      PositionInVector(new_instruction_sequence, collective_permute_done_1));
}

TEST_F(LatencyHidingSchedulerTest, BackToBackCollectivePerGmutesTest) {
  absl::string_view hlo_string = R"(
    HloModule back_to_back_collective_permute_test, is_scheduled=true
    ENTRY after_optimizations_test {
    %parameter.1 = bf16[8]{0} parameter(0)
    %collective-permute.2 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.1), source_target_pairs={{0,1},{1,2},{2,3}}
    %collective-permute.6 = bf16[8]{0} collective-permute(bf16[8]{0} collective-permute.2), source_target_pairs={{1,0},{0,3},{3,2}}
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  EXPECT_EQ(original_instruction_sequence.size(), 3);
  EXPECT_EQ(new_instruction_sequence.size(), 5);
  EXPECT_EQ(original_instruction_sequence[0]->user_count(), 1);
  EXPECT_EQ(original_instruction_sequence[0]->users()[0]->opcode(),
            HloOpcode::kCollectivePermuteStart);
  HloInstruction* collective_permute_start_1 =
      original_instruction_sequence[0]->users()[0];
  // Collective-permute-start immediately follows its operand.
  EXPECT_EQ(
      PositionInVector(new_instruction_sequence,
                       original_instruction_sequence[0]) +
          1,
      PositionInVector(new_instruction_sequence, collective_permute_start_1));
  EXPECT_EQ(collective_permute_start_1->user_count(), 1);
  EXPECT_EQ(collective_permute_start_1->users()[0]->opcode(),
            HloOpcode::kCollectivePermuteDone);
  HloInstruction* collective_permute_done_1 =
      collective_permute_start_1->users()[0];
  // Collective-permute-done immediately leads one of its users.
  EXPECT_TRUE(
      (PositionInVector(new_instruction_sequence, collective_permute_done_1) +
           1 ==
       PositionInVector(new_instruction_sequence,
                        collective_permute_done_1->users()[0])) ||
      (PositionInVector(new_instruction_sequence, collective_permute_done_1) +
           1 ==
       PositionInVector(new_instruction_sequence,
                        collective_permute_done_1->users()[1])));
  // Collective-permute-done is scheduled before its users.
  EXPECT_TRUE(
      (PositionInVector(new_instruction_sequence, collective_permute_done_1) <
       PositionInVector(new_instruction_sequence,
                        collective_permute_done_1->users()[0])));
  // The second collective-permute starts after the first one is done.
  EXPECT_EQ(new_instruction_sequence[4]->opcode(),
            HloOpcode::kCollectivePermuteDone);
  EXPECT_GT(
      PositionInVector(new_instruction_sequence,
                       new_instruction_sequence[4]->operand(0)),
      PositionInVector(new_instruction_sequence, collective_permute_done_1));
}

TEST_F(LatencyHidingSchedulerTest, ParallelCollectivePermutesTest) {
  absl::string_view hlo_string = R"(
    HloModule single_collective_permute_test, is_scheduled=true
    ENTRY after_optimizations_test {
    %parameter.1 = bf16[8]{0} parameter(0)
    %collective-permute.2 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.1), source_target_pairs={{0,1},{1,2},{2,3}}
    %constant.3 = bf16[] constant(1)
    %broadcast.4 = bf16[8]{0} broadcast(bf16[] %constant.3), dimensions={}
    %add.5 = bf16[8]{0} add(bf16[8]{0} %collective-permute.2, bf16[8]{0} %broadcast.4)
    %collective-permute.6 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.1), source_target_pairs={{1,0},{0,3},{3,2}}
    %add.6 = bf16[8]{0} add(bf16[8]{0} %collective-permute.6, bf16[8]{0} %add.5)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // The new instruction sequence preserves the scheduling order of
  // non-collective-permute instructions.
  EXPECT_LT(PositionInVector(new_instruction_sequence,
                             original_instruction_sequence[0]),
            PositionInVector(new_instruction_sequence,
                             original_instruction_sequence[2]));
  EXPECT_LT(PositionInVector(new_instruction_sequence,
                             original_instruction_sequence[2]),
            PositionInVector(new_instruction_sequence,
                             original_instruction_sequence[3]));
  EXPECT_LT(PositionInVector(new_instruction_sequence,
                             original_instruction_sequence[3]),
            PositionInVector(new_instruction_sequence,
                             original_instruction_sequence[4]));
  EXPECT_LT(PositionInVector(new_instruction_sequence,
                             original_instruction_sequence[4]),
            PositionInVector(new_instruction_sequence,
                             original_instruction_sequence[6]));
  EXPECT_EQ(original_instruction_sequence[0]->user_count(), 2);
  EXPECT_EQ(original_instruction_sequence[0]->users()[0]->opcode(),
            HloOpcode::kCollectivePermuteStart);
  EXPECT_EQ(original_instruction_sequence[0]->users()[1]->opcode(),
            HloOpcode::kCollectivePermuteStart);

  int collective_permute_1_pos = PositionInVector(
      new_instruction_sequence, original_instruction_sequence[0]->users()[0]);
  int collective_permute_2_pos = PositionInVector(
      new_instruction_sequence, original_instruction_sequence[0]->users()[1]);
  // The two collective-permutes are conducted in parallel.
  EXPECT_TRUE((collective_permute_1_pos == collective_permute_2_pos + 1) ||
              (collective_permute_1_pos + 1 == collective_permute_2_pos));
}

TEST_F(LatencyHidingSchedulerTest, MaxConcurrentCollectivePermutesTest) {
  absl::string_view hlo_string = R"(
    HloModule single_collective_permute_test, is_scheduled=true
    ENTRY after_optimizations_test {
    %parameter.1 = bf16[8]{0} parameter(0)
    %parameter.2 = bf16[8]{0} parameter(1)
    %parameter.3 = bf16[8]{0} parameter(2)
    %collective-permute.4 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.1), source_target_pairs={{0,1},{1,2},{2,3}}
    %collective-permute.5 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.1), source_target_pairs={{1,0},{0,3},{3,2}}
    %collective-permute.6 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.2), source_target_pairs={{0,1},{1,2},{2,3}}
    %collective-permute.7 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.2), source_target_pairs={{1,0},{0,3},{3,2}}
    %collective-permute.8 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.3), source_target_pairs={{0,1},{1,2},{2,3}}
    %collective-permute.9 = bf16[8]{0} collective-permute(bf16[8]{0} parameter.3), source_target_pairs={{1,0},{0,3},{3,2}}
    %add.10 = bf16[8]{0} add(bf16[8]{0} %collective-permute.8, bf16[8]{0} %collective-permute.9)
    %add.11 = bf16[8]{0} add(bf16[8]{0} %collective-permute.7, bf16[8]{0} %add.10)
    %add.12 = bf16[8]{0} add(bf16[8]{0} %collective-permute.6, bf16[8]{0} %add.11)
    %add.13 = bf16[8]{0} add(bf16[8]{0} %collective-permute.5, bf16[8]{0} %add.12)
    ROOT %add.14 = bf16[8]{0} add(bf16[8]{0} %collective-permute.4, bf16[8]{0} %add.13)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
  EXPECT_TRUE(
      MaxConcurrentCollectivePermutesBelowThreshold(new_instruction_sequence));
}

TEST_F(LatencyHidingSchedulerTest, BalanceChainedCollectivePermutesNoOverlap) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[8]{0} parameter(0)
  collective-permute.1 = bf16[8]{0} collective-permute(param), source_target_pairs={{0,1},{1,2},{2,3}}
  copy.2 = bf16[8]{0} copy(collective-permute.1)
  ROOT collective-permute.2 = bf16[8]{0} collective-permute(copy.2), source_target_pairs={{1,0},{0,3},{3,2}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
}

TEST_F(LatencyHidingSchedulerTest, ExistingSingleCollectivePermuteAsyncTest) {
  absl::string_view hlo_string = R"(
    HloModule single_collective_permute_test, is_scheduled=true
    ENTRY after_optimizations_test {
    p0 = f32[16,64,256]{2,1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,256,256]{2,1,0} parameter(2)
    c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
      window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
    %collective-permute-start.1 = (f32[16,256,256]{2,1,0},
      f32[16,256,256]{2,1,0}, u32[], u32[]) collective-permute-start(
      f32[16,256,256]{2,1,0} p2), source_target_pairs={{0,1},{1,2},{2,3}},
      channel_id=1, metadata={op_type="CollectivePermute" op_name="cp0"}
    %collective-permute-done.1 = f32[16,256,256]{2,1,0} collective-permute-done(
      (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0},
      u32[], u32[]) collective-permute-start.1),
      metadata={op_type="CollectivePermute" op_name="cp0"}
    ROOT a = f32[16,256,256]{2,1,0} add(c0, collective-permute-done.1)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteStart,
                                        new_instruction_sequence, "cp0"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_GE(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteDone,
                                        new_instruction_sequence, "cp0"),
            GetIndex(new_instruction_sequence, "c0"));
}

TEST_F(LatencyHidingSchedulerTest, BalanceChainExtended) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  cp0 = f32[16,256,256]{2,1,0} collective-permute(p2),
    source_target_pairs={{0,1},{1,0}},
    metadata={op_type="CollectivePermute" op_name="cp0"}
  cp1 = f32[16,256,256]{2,1,0} collective-permute(p3),
    source_target_pairs={{0,1},{1,0}},
    metadata={op_type="CollectivePermute" op_name="cp1"}
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  t0 = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}) tuple(cp0, cp1)
  gte0 = f32[16,256,256]{2,1,0} get-tuple-element(t0), index=0
  gte1 = f32[16,256,256]{2,1,0} get-tuple-element(t0), index=1
  cp2 = f32[16,256,256]{2,1,0} collective-permute(gte0),
    source_target_pairs={{0,1},{1,0}},
    metadata={op_type="CollectivePermute" op_name="cp2"}
  a2 = f32[16,256,256]{2,1,0} add(cp2, c0)
  cp3 = f32[16,256,256]{2,1,0} collective-permute(gte1),
    source_target_pairs={{0,1},{1,0}},
    metadata={op_type="CollectivePermute" op_name="cp3"}
  ROOT tuple = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}) tuple(a2, cp3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteStart,
                                        new_instruction_sequence, "cp0"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteStart,
                                        new_instruction_sequence, "cp1"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteDone,
                                        new_instruction_sequence, "cp0"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteDone,
                                        new_instruction_sequence, "cp1"),
            GetIndex(new_instruction_sequence, "c0"));

  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteStart,
                                        new_instruction_sequence, "cp2"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteStart,
                                        new_instruction_sequence, "cp3"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteDone,
                                        new_instruction_sequence, "cp2"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kCollectivePermuteDone,
                                        new_instruction_sequence, "cp3"),
            GetIndex(new_instruction_sequence, "c1"));
}

TEST_F(LatencyHidingSchedulerTest,
       BalanceChainedCollectivePermutesLoopedEinsum) {
  std::string hlo_string = R"(
HloModule module, is_scheduled=true

%fused_computation.1793 (param_0.4944: s32[16], param_1.5648: u32[], param_2.3959: u32[], param_3.3338: u32[], param_4.2302: u32[]) -> (s32[1], s32[1], s32[1], s32[1]) {
  %param_0.4944 = s32[16]{0:T(128)} parameter(0)
  %param_1.5648 = u32[]{:T(128)} parameter(1)
  %dynamic-slice.1806 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4944, u32[]{:T(128)} %param_1.5648), dynamic_slice_sizes={1}
  %param_2.3959 = u32[]{:T(128)} parameter(2)
  %dynamic-slice.1807 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4944, u32[]{:T(128)} %param_2.3959), dynamic_slice_sizes={1}
  %param_3.3338 = u32[]{:T(128)} parameter(3)
  %dynamic-slice.1808 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4944, u32[]{:T(128)} %param_3.3338), dynamic_slice_sizes={1}
  %param_4.2302 = u32[]{:T(128)} parameter(4)
  %dynamic-slice.1809 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4944, u32[]{:T(128)} %param_4.2302), dynamic_slice_sizes={1}
  ROOT %tuple.1384 = (s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) tuple(s32[1]{0:T(128)} %dynamic-slice.1806, s32[1]{0:T(128)} %dynamic-slice.1807, s32[1]{0:T(128)} %dynamic-slice.1808, s32[1]{0:T(128)} %dynamic-slice.1809)
}

%fused_computation.109 (param_0.225: bf16[8,1024,1,20,256,1,1]) -> bf16[8,1024,1,20,256,1,1,1] {
  %param_0.225 = bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} parameter(0)
  ROOT %bitcast.713 = bf16[8,1024,1,20,256,1,1,1]{4,1,7,3,2,0,6,5:T(8,128)(2,1)} bitcast(bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} %param_0.225)
}

%fused_computation.110.clone (param_0.251: s32[], param_1.277: bf16[1,20,256,1,16,4,288,1], param_2.190: s32[]) -> bf16[1,20,256,2,1,4,288,1] {
  %param_1.277 = bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} parameter(1)
  %constant.6014 = bf16[]{:T(256)} constant(-inf)
  %pad.370 = bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} pad(bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %param_1.277, bf16[]{:T(256)} %constant.6014), padding=0_0x0_0x0_0x0_1x0_0x0_0x0_0x0_0
  %constant.6004 = s32[]{:T(128)} constant(0)
  %param_0.251 = s32[]{:T(128)} parameter(0)
  %dynamic-slice.1503 = bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} dynamic-slice(bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %pad.370, s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004, /*index=5*/s32[]{:T(128)} %param_0.251, s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004), dynamic_slice_sizes={1,20,256,2,1,4,288,1}
  %pad.369 = bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} pad(bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %param_1.277, bf16[]{:T(256)} %constant.6014), padding=0_0x0_0x0_0x1_0x0_0x0_0x0_0x0_0
  %param_2.190 = s32[]{:T(128)} parameter(2)
  %dynamic-slice.1502 = bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} dynamic-slice(bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %pad.369, s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004, /*index=5*/s32[]{:T(128)} %param_2.190, s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004, s32[]{:T(128)} %constant.6004), dynamic_slice_sizes={1,20,256,2,1,4,288,1}
  ROOT %maximum.513 = bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} maximum(bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %dynamic-slice.1503, bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %dynamic-slice.1502)
}

%fused_computation.108 (param_0.235: bf16[8,1024,1,20,256,1,1], param_1.276: s32[], param_2.187: bf16[1,20,256,1,16,4,288,1], param_3.145: s32[]) -> bf16[2,1,4,288,8,1024,1,1] {
  %param_1.276 = s32[]{:T(128)} parameter(1)
  %param_2.187 = bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} parameter(2)
  %param_3.145 = s32[]{:T(128)} parameter(3)
  %fusion.132 = bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} fusion(s32[]{:T(128)} %param_1.276, bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %param_2.187, s32[]{:T(128)} %param_3.145), kind=kLoop, calls=%fused_computation.110.clone
  %param_0.235 = bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} parameter(0)
  %fusion.129 = bf16[8,1024,1,20,256,1,1,1]{4,1,7,3,2,0,6,5:T(8,128)(2,1)} fusion(bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} %param_0.235), kind=kLoop, calls=%fused_computation.109
  ROOT %convolution.170 = bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} convolution(bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %fusion.132, bf16[8,1024,1,20,256,1,1,1]{4,1,7,3,2,0,6,5:T(8,128)(2,1)} %fusion.129), window={size=1x1x8x1x20x1 pad=0_0x0_0x7_7x0_0x0_0x0_0 rhs_reversal=0x0x1x0x0x0}, dim_labels=34f501b2_2o34i015->501b2f34
}

%fused_computation.117 (param_0.248: bf16[1,4,288,8,1024,1,1], param_1.273: bf16[2,1,4,288,8,1024,1,1]) -> bf16[1,4,288,8,1024,1,1] {
  %param_0.248 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} parameter(0)
  %param_1.273 = bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} parameter(1)
  %slice.1252 = bf16[1,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} slice(bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} %param_1.273), slice={[0:1], [0:1], [0:4], [0:288], [0:8], [0:1024], [0:1], [0:1]}
  %bitcast.719 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} bitcast(bf16[1,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} %slice.1252)
  ROOT %add.3083 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} add(bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} %param_0.248, bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} %bitcast.719)
}

%fused_computation.107 (param_0.223: bf16[8,1024,1,20,256,1,1]) -> bf16[8,1024,1,20,256,1,1,1] {
  %param_0.223 = bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} parameter(0)
  ROOT %bitcast.711 = bf16[8,1024,1,20,256,1,1,1]{4,1,7,3,2,0,6,5:T(8,128)(2,1)} bitcast(bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} %param_0.223)
}

%fused_computation.111.clone (param_0.250: s32[], param_1.275: bf16[1,20,256,1,16,4,288,1], param_2.189: s32[]) -> bf16[1,20,256,2,1,4,288,1] {
  %param_1.275 = bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} parameter(1)
  %constant.6009 = bf16[]{:T(256)} constant(-inf)
  %pad.374 = bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} pad(bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %param_1.275, bf16[]{:T(256)} %constant.6009), padding=0_0x0_0x0_0x0_1x0_0x0_0x0_0x0_0
  %constant.5999 = s32[]{:T(128)} constant(0)
  %param_0.250 = s32[]{:T(128)} parameter(0)
  %dynamic-slice.1507 = bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} dynamic-slice(bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %pad.374, s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999, /*index=5*/s32[]{:T(128)} %param_0.250, s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999), dynamic_slice_sizes={1,20,256,2,1,4,288,1}
  %pad.373 = bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} pad(bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %param_1.275, bf16[]{:T(256)} %constant.6009), padding=0_0x0_0x0_0x1_0x0_0x0_0x0_0x0_0
  %param_2.189 = s32[]{:T(128)} parameter(2)
  %dynamic-slice.1506 = bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} dynamic-slice(bf16[1,20,256,2,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %pad.373, s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999, /*index=5*/s32[]{:T(128)} %param_2.189, s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999, s32[]{:T(128)} %constant.5999), dynamic_slice_sizes={1,20,256,2,1,4,288,1}
  ROOT %maximum.514 = bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} maximum(bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %dynamic-slice.1507, bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %dynamic-slice.1506)
}

%fused_computation.106 (param_0.239: bf16[8,1024,1,20,256,1,1], param_1.274: s32[], param_2.185: bf16[1,20,256,1,16,4,288,1], param_3.144: s32[]) -> bf16[2,1,4,288,8,1024,1,1] {
  %param_1.274 = s32[]{:T(128)} parameter(1)
  %param_2.185 = bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} parameter(2)
  %param_3.144 = s32[]{:T(128)} parameter(3)
  %fusion.133 = bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} fusion(s32[]{:T(128)} %param_1.274, bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %param_2.185, s32[]{:T(128)} %param_3.144), kind=kLoop, calls=%fused_computation.111.clone
  %param_0.239 = bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} parameter(0)
  %fusion.127 = bf16[8,1024,1,20,256,1,1,1]{4,1,7,3,2,0,6,5:T(8,128)(2,1)} fusion(bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} %param_0.239), kind=kLoop, calls=%fused_computation.107
  ROOT %convolution.169 = bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} convolution(bf16[1,20,256,2,1,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %fusion.133, bf16[8,1024,1,20,256,1,1,1]{4,1,7,3,2,0,6,5:T(8,128)(2,1)} %fusion.127), window={size=1x1x8x1x20x1 pad=0_0x0_0x7_7x0_0x0_0x0_0 rhs_reversal=0x0x1x0x0x0}, dim_labels=34f501b2_2o34i015->501b2f34
}

%fused_computation.115 (param_0.244: bf16[1,4,288,8,1024,1,1], param_1.270: bf16[2,1,4,288,8,1024,1,1]) -> bf16[1,4,288,8,1024,1,1] {
  %param_0.244 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} parameter(0)
  %param_1.270 = bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} parameter(1)
  %slice.1249 = bf16[1,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} slice(bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} %param_1.270), slice={[0:1], [0:1], [0:4], [0:288], [0:8], [0:1024], [0:1], [0:1]}
  %bitcast.716 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} bitcast(bf16[1,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} %slice.1249)
  ROOT %add.3082 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} add(bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} %param_0.244, bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} %bitcast.716)
}

%fused_computation.113 (param_0.241: bf16[1,4,288,8,1024,1,1], param_1.267: bf16[2,4,288,8,1024,1,1]) -> bf16[1,4,288,8,1024,1,1] {
  %param_0.241 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} parameter(0)
  %param_1.267 = bf16[2,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} parameter(1)
  %slice.1246 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} slice(bf16[2,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %param_1.267), slice={[1:2], [0:4], [0:288], [0:8], [0:1024], [0:1], [0:1]}
  ROOT %add.3081 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} add(bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %param_0.241, bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %slice.1246)
}

%fused_computation.112 (param_0.240: bf16[1,4,288,8,1024,1,1], param_1.265: bf16[2,4,288,8,1024,1,1]) -> bf16[1,4,288,8,1024,1,1] {
  %param_0.240 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} parameter(0)
  %param_1.265 = bf16[2,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} parameter(1)
  %slice.1245 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} slice(bf16[2,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %param_1.265), slice={[1:2], [0:4], [0:288], [0:8], [0:1024], [0:1], [0:1]}
  ROOT %add.3080 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} add(bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %param_0.240, bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %slice.1245)
}
)";
  hlo_string += R"(

ENTRY entry {
  %param.163 = (bf16[1,20,256,16,4,288,1]{2,5,1,4,3,6,0:T(8,128)(2,1)}, bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)}, bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)}, bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)}, u32[]{:T(128)}) parameter(0)
  %get-tuple-element.20289 = bf16[1,20,256,16,4,288,1]{2,5,1,4,3,6,0:T(8,128)(2,1)} get-tuple-element((bf16[1,20,256,16,4,288,1]{2,5,1,4,3,6,0:T(8,128)(2,1)}, bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)}, bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)}, bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)}, u32[]{:T(128)}) %param.163), index=0
  %get-tuple-element.20290 = bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} get-tuple-element((bf16[1,20,256,16,4,288,1]{2,5,1,4,3,6,0:T(8,128)(2,1)}, bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)}, bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)}, bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)}, u32[]{:T(128)}) %param.163), index=1
  %get-tuple-element.20291 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} get-tuple-element((bf16[1,20,256,16,4,288,1]{2,5,1,4,3,6,0:T(8,128)(2,1)}, bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)}, bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)}, bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)}, u32[]{:T(128)}) %param.163), index=2
  %collective-permute.8 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} collective-permute(bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} %get-tuple-element.20291), channel_id=22, source_target_pairs={{0,15},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6},{8,7},{9,8},{10,9},{11,10},{12,11},{13,12},{14,13},{15,14}}, backend_config="{\"flag_configs\":[],\"barrier_config\":{\"barrier_type\":\"CUSTOM\",\"id\":\"0\"}}"
  %iota.36 = s32[16]{0:T(128)} iota(), iota_dimension=0
  %constant.3283 = u32[1024]{0:T(1024)} constant({...})
  %partition-id.6 = u32[]{:T(128)} partition-id()
  %dynamic-slice.254 = u32[1]{0:T(128)} dynamic-slice(u32[1024]{0:T(1024)} %constant.3283, u32[]{:T(128)} %partition-id.6), dynamic_slice_sizes={1}
  %bitcast.55 = u32[]{:T(128)} bitcast(u32[1]{0:T(128)} %dynamic-slice.254)
  %constant.5148 = u32[]{:T(128)} constant(8)
  %add.2615 = u32[]{:T(128)} add(u32[]{:T(128)} %bitcast.55, u32[]{:T(128)} %constant.5148)
  %get-tuple-element.20293 = u32[]{:T(128)} get-tuple-element((bf16[1,20,256,16,4,288,1]{2,5,1,4,3,6,0:T(8,128)(2,1)}, bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)}, bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)}, bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)}, u32[]{:T(128)}) %param.163), index=4
  %copy.2385 = u32[]{:T(128)} copy(u32[]{:T(128)} %get-tuple-element.20293)
  %constant.3305 = u32[]{:T(128)} constant(1)
  %add.1503 = u32[]{:T(128)} add(u32[]{:T(128)} %copy.2385, u32[]{:T(128)} %constant.3305)
  %subtract.200 = u32[]{:T(128)} subtract(u32[]{:T(128)} %add.2615, u32[]{:T(128)} %add.1503)
  %constant.4875 = u32[]{:T(128)} constant(15)
  %and.29 = u32[]{:T(128)} and(u32[]{:T(128)} %subtract.200, u32[]{:T(128)} %constant.4875)
  %add.1504 = u32[]{:T(128)} add(u32[]{:T(128)} %add.1503, u32[]{:T(128)} %bitcast.55)
  %constant.3285 = u32[]{:T(128)} constant(9)
  %add.1506 = u32[]{:T(128)} add(u32[]{:T(128)} %add.1504, u32[]{:T(128)} %constant.3285)
  %and.28 = u32[]{:T(128)} and(u32[]{:T(128)} %add.1506, u32[]{:T(128)} %constant.4875)
  %subtract.198 = u32[]{:T(128)} subtract(u32[]{:T(128)} %add.2615, u32[]{:T(128)} %copy.2385)
  %and.27 = u32[]{:T(128)} and(u32[]{:T(128)} %subtract.198, u32[]{:T(128)} %constant.4875)
  %add.1498 = u32[]{:T(128)} add(u32[]{:T(128)} %copy.2385, u32[]{:T(128)} %bitcast.55)
  %add.1500 = u32[]{:T(128)} add(u32[]{:T(128)} %add.1498, u32[]{:T(128)} %constant.3285)
  %and.26 = u32[]{:T(128)} and(u32[]{:T(128)} %add.1500, u32[]{:T(128)} %constant.4875)
  %fusion.1987 = (s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) fusion(s32[16]{0:T(128)} %iota.36, u32[]{:T(128)} %and.29, u32[]{:T(128)} %and.28, u32[]{:T(128)} %and.27, u32[]{:T(128)} %and.26), kind=kLoop, calls=%fused_computation.1793
  %get-tuple-element.19793 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) %fusion.1987), index=3
  %bitcast.56 = s32[]{:T(128)} bitcast(s32[1]{0:T(128)} %get-tuple-element.19793)
  %bitcast.54 = bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} bitcast(bf16[1,20,256,16,4,288,1]{2,5,1,4,3,6,0:T(8,128)(2,1)} %get-tuple-element.20289)
  %get-tuple-element.19792 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) %fusion.1987), index=2
  %bitcast.57 = s32[]{:T(128)} bitcast(s32[1]{0:T(128)} %get-tuple-element.19792)
  %fusion.128 = bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} fusion(bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} %get-tuple-element.20290, s32[]{:T(128)} %bitcast.56, bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %bitcast.54, s32[]{:T(128)} %bitcast.57), kind=kOutput, calls=%fused_computation.108
  %fusion.139 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} fusion(bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} %collective-permute.8, bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} %fusion.128), kind=kLoop, calls=%fused_computation.117
  %collective-permute.10 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} collective-permute(bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} %fusion.139), channel_id=24, source_target_pairs={{0,15},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6},{8,7},{9,8},{10,9},{11,10},{12,11},{13,12},{14,13},{15,14}}, backend_config="{\"flag_configs\":[],\"barrier_config\":{\"barrier_type\":\"CUSTOM\",\"id\":\"0\"}}"
  %get-tuple-element.19791 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) %fusion.1987), index=1
  %bitcast.60 = s32[]{:T(128)} bitcast(s32[1]{0:T(128)} %get-tuple-element.19791)
  %get-tuple-element.19790 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) %fusion.1987), index=0
  %bitcast.61 = s32[]{:T(128)} bitcast(s32[1]{0:T(128)} %get-tuple-element.19790)
  %fusion.126 = bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} fusion(bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} %get-tuple-element.20290, s32[]{:T(128)} %bitcast.60, bf16[1,20,256,1,16,4,288,1]{2,6,3,1,0,7,5,4:T(8,128)(2,1)} %bitcast.54, s32[]{:T(128)} %bitcast.61), kind=kOutput, calls=%fused_computation.106
  %fusion.137 = bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} fusion(bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} %collective-permute.10, bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} %fusion.126), kind=kLoop, calls=%fused_computation.115
  %get-tuple-element.20292 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} get-tuple-element((bf16[1,20,256,16,4,288,1]{2,5,1,4,3,6,0:T(8,128)(2,1)}, bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)}, bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)}, bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)}, u32[]{:T(128)}) %param.163), index=3
  %collective-permute.9 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} collective-permute(bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %get-tuple-element.20292), channel_id=23, source_target_pairs={{0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,8},{8,9},{9,10},{10,11},{11,12},{12,13},{13,14},{14,15},{15,0}}, backend_config="{\"flag_configs\":[],\"barrier_config\":{\"barrier_type\":\"CUSTOM\",\"id\":\"1\"}}"
  %bitcast.63 = bf16[2,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} bitcast(bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} %fusion.128)
  %fusion.135 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} fusion(bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %collective-permute.9, bf16[2,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %bitcast.63), kind=kLoop, calls=%fused_computation.113
  %collective-permute.11 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} collective-permute(bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %fusion.135), channel_id=25, source_target_pairs={{0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,8},{8,9},{9,10},{10,11},{11,12},{12,13},{13,14},{14,15},{15,0}}, backend_config="{\"flag_configs\":[],\"barrier_config\":{\"barrier_type\":\"CUSTOM\",\"id\":\"1\"}}"
  %bitcast.64 = bf16[2,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} bitcast(bf16[2,1,4,288,8,1024,1,1]{5,3,0,7,6,4,2,1:T(8,128)(2,1)} %fusion.126)
  %fusion.134 = bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} fusion(bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %collective-permute.11, bf16[2,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %bitcast.64), kind=kLoop, calls=%fused_computation.112
  %constant.5023 = u32[]{:T(128)} constant(2)
  %add.1925 = u32[]{:T(128)} add(u32[]{:T(128)} %copy.2385, u32[]{:T(128)} %constant.5023)
  ROOT %tuple.1457 = (bf16[1,20,256,16,4,288,1]{2,5,1,4,3,6,0:T(8,128)(2,1)}, bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)}, bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)}, bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)}, u32[]{:T(128)}) tuple(bf16[1,20,256,16,4,288,1]{2,5,1,4,3,6,0:T(8,128)(2,1)} %get-tuple-element.20289, bf16[8,1024,1,20,256,1,1]{4,1,3,0,6,5,2:T(8,128)(2,1)} %get-tuple-element.20290, bf16[1,4,288,8,1024,1,1]{4,2,3,1,6,5,0:T(8,128)(2,1)} %fusion.137, bf16[1,4,288,8,1024,1,1]{4,2,0,3,1,6,5:T(8,128)(2,1)} %fusion.134, u32[]{:T(128)} %add.1925)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start"),
            GetIndex(new_instruction_sequence, "fusion.128"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.2"),
            GetIndex(new_instruction_sequence, "fusion.128"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done"),
            GetIndex(new_instruction_sequence, "fusion.128"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done.2"),
            GetIndex(new_instruction_sequence, "fusion.128"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.1"),
            GetIndex(new_instruction_sequence, "fusion.126"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.3"),
            GetIndex(new_instruction_sequence, "fusion.126"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done.1"),
            GetIndex(new_instruction_sequence, "fusion.126"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done.3"),
            GetIndex(new_instruction_sequence, "fusion.126"));
}

TEST_F(LatencyHidingSchedulerTest, MoveCentainConv) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  cp0 = f32[16,256,256]{2,1,0} collective-permute(p2),
    source_target_pairs={{0,1},{1,0}}
  cp1 = f32[16,256,256]{2,1,0} collective-permute(p3),
    source_target_pairs={{0,1},{1,0}}
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  a0 = f32[16,256,256]{2,1,0} add(cp0, c1)
  cp2 = f32[16,256,256]{2,1,0} collective-permute(a0),
    source_target_pairs={{0,1},{1,0}}
  a2 = f32[16,256,256]{2,1,0} add(cp2, c0)
  a1 = f32[16,256,256]{2,1,0} add(cp1, c1)
  cp3 = f32[16,256,256]{2,1,0} collective-permute(a1),
    source_target_pairs={{0,1},{1,0}}
  ROOT tuple = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}) tuple(a2, cp3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.1"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done.1"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.2"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.3"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done.2"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done.3"),
            GetIndex(new_instruction_sequence, "c0"));
}

TEST_F(LatencyHidingSchedulerTest,
       BalanceChainedCollectivePermutesLoopedEinsum2) {
  std::string hlo_string = R"(
HloModule module, is_scheduled=true

%fused_computation.1851 (param_0.5170: s32[32], param_1.5848: u32[], param_2.4103: u32[], param_3.3513: u32[], param_4.2356: u32[]) -> (s32[1], s32[1], s32[1], s32[1]) {
  %param_0.5170 = s32[32]{0:T(128)} parameter(0)
  %param_1.5848 = u32[]{:T(128)} parameter(1)
  %dynamic-slice.1636 = s32[1]{0:T(128)} dynamic-slice(s32[32]{0:T(128)} %param_0.5170, u32[]{:T(128)} %param_1.5848), dynamic_slice_sizes={1}
  %param_2.4103 = u32[]{:T(128)} parameter(2)
  %dynamic-slice.1637 = s32[1]{0:T(128)} dynamic-slice(s32[32]{0:T(128)} %param_0.5170, u32[]{:T(128)} %param_2.4103), dynamic_slice_sizes={1}
  %param_3.3513 = u32[]{:T(128)} parameter(3)
  %dynamic-slice.1638 = s32[1]{0:T(128)} dynamic-slice(s32[32]{0:T(128)} %param_0.5170, u32[]{:T(128)} %param_3.3513), dynamic_slice_sizes={1}
  %param_4.2356 = u32[]{:T(128)} parameter(4)
  %dynamic-slice.1639 = s32[1]{0:T(128)} dynamic-slice(s32[32]{0:T(128)} %param_0.5170, u32[]{:T(128)} %param_4.2356), dynamic_slice_sizes={1}
  ROOT %tuple.1297 = (s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) tuple(s32[1]{0:T(128)} %dynamic-slice.1636, s32[1]{0:T(128)} %dynamic-slice.1637, s32[1]{0:T(128)} %dynamic-slice.1638, s32[1]{0:T(128)} %dynamic-slice.1639)
}

%fused_computation.117 (param_0.249: bf16[16,1024,1,10,256,1]) -> bf16[16,1024,1,10,256,1,1] {
  %param_0.249 = bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} parameter(0)
  ROOT %bitcast.672 = bf16[16,1024,1,10,256,1,1]{4,1,6,3,2,0,5:T(8,128)(2,1)} bitcast(bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %param_0.249)
}

%fused_computation.124.clone (param_0.277: s32[], param_1.330: bf16[1,10,256,1,32,576,1], param_2.233: s32[]) -> bf16[1,10,256,2,1,576,1] {
  %param_1.330 = bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} parameter(1)
  %constant.5658 = bf16[]{:T(256)} constant(-inf)
  %pad.357 = bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} pad(bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %param_1.330, bf16[]{:T(256)} %constant.5658), padding=0_0x0_0x0_0x0_1x0_0x0_0x0_0
  %constant.5648 = s32[]{:T(128)} constant(0)
  %param_0.277 = s32[]{:T(128)} parameter(0)
  %dynamic-slice.1327 = bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} dynamic-slice(bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %pad.357, s32[]{:T(128)} %constant.5648, s32[]{:T(128)} %constant.5648, s32[]{:T(128)} %constant.5648, s32[]{:T(128)} %constant.5648, /*index=5*/s32[]{:T(128)} %param_0.277, s32[]{:T(128)} %constant.5648, s32[]{:T(128)} %constant.5648), dynamic_slice_sizes={1,10,256,2,1,576,1}
  %pad.363 = bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} pad(bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %param_1.330, bf16[]{:T(256)} %constant.5658), padding=0_0x0_0x0_0x1_0x0_0x0_0x0_0
  %param_2.233 = s32[]{:T(128)} parameter(2)
  %dynamic-slice.1333 = bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} dynamic-slice(bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %pad.363, s32[]{:T(128)} %constant.5648, s32[]{:T(128)} %constant.5648, s32[]{:T(128)} %constant.5648, s32[]{:T(128)} %constant.5648, /*index=5*/s32[]{:T(128)} %param_2.233, s32[]{:T(128)} %constant.5648, s32[]{:T(128)} %constant.5648), dynamic_slice_sizes={1,10,256,2,1,576,1}
  ROOT %maximum.510 = bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} maximum(bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %dynamic-slice.1327, bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %dynamic-slice.1333)
}

%fused_computation.116 (param_0.264: bf16[16,1024,1,10,256,1], param_1.329: s32[], param_2.230: bf16[1,10,256,1,32,576,1], param_3.197: s32[]) -> bf16[2,1,576,16,1024,1,1] {
  %param_1.329 = s32[]{:T(128)} parameter(1)
  %param_2.230 = bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} parameter(2)
  %param_3.197 = s32[]{:T(128)} parameter(3)
  %fusion.155 = bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} fusion(s32[]{:T(128)} %param_1.329, bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %param_2.230, s32[]{:T(128)} %param_3.197), kind=kLoop, calls=%fused_computation.124.clone
  %param_0.264 = bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} parameter(0)
  %fusion.147 = bf16[16,1024,1,10,256,1,1]{4,1,6,3,2,0,5:T(8,128)(2,1)} fusion(bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %param_0.264), kind=kLoop, calls=%fused_computation.117
  ROOT %convolution.168 = bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} convolution(bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %fusion.155, bf16[16,1024,1,10,256,1,1]{4,1,6,3,2,0,5:T(8,128)(2,1)} %fusion.147), window={size=1x16x1x10x1 pad=0_0x15_15x0_0x0_0x0_0 rhs_reversal=0x1x0x0x0}, dim_labels=23f40b1_1o23i04->40b1f23
}

%fused_computation.123 (param_0.258: bf16[1,576,16,1024,1,1], param_1.306: bf16[2,1,576,16,1024,1,1]) -> bf16[1,576,16,1024,1,1] {
  %param_0.258 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} parameter(0)
  %param_1.306 = bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} parameter(1)
  %slice.1132 = bf16[1,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} slice(bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} %param_1.306), slice={[0:1], [0:1], [0:576], [0:16], [0:1024], [0:1], [0:1]}
  %bitcast.678 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} bitcast(bf16[1,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} %slice.1132)
  ROOT %add.3125 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} add(bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} %param_0.258, bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} %bitcast.678)
}

%fused_computation.115 (param_0.247: bf16[16,1024,1,10,256,1]) -> bf16[16,1024,1,10,256,1,1] {
  %param_0.247 = bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} parameter(0)
  ROOT %bitcast.670 = bf16[16,1024,1,10,256,1,1]{4,1,6,3,2,0,5:T(8,128)(2,1)} bitcast(bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %param_0.247)
}

%fused_computation.125.clone (param_0.276: s32[], param_1.328: bf16[1,10,256,1,32,576,1], param_2.232: s32[]) -> bf16[1,10,256,2,1,576,1] {
  %param_1.328 = bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} parameter(1)
  %constant.5653 = bf16[]{:T(256)} constant(-inf)
  %pad.360 = bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} pad(bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %param_1.328, bf16[]{:T(256)} %constant.5653), padding=0_0x0_0x0_0x0_1x0_0x0_0x0_0
  %constant.5643 = s32[]{:T(128)} constant(0)
  %param_0.276 = s32[]{:T(128)} parameter(0)
  %dynamic-slice.1330 = bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} dynamic-slice(bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %pad.360, s32[]{:T(128)} %constant.5643, s32[]{:T(128)} %constant.5643, s32[]{:T(128)} %constant.5643, s32[]{:T(128)} %constant.5643, /*index=5*/s32[]{:T(128)} %param_0.276, s32[]{:T(128)} %constant.5643, s32[]{:T(128)} %constant.5643), dynamic_slice_sizes={1,10,256,2,1,576,1}
  %pad.366 = bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} pad(bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %param_1.328, bf16[]{:T(256)} %constant.5653), padding=0_0x0_0x0_0x1_0x0_0x0_0x0_0
  %param_2.232 = s32[]{:T(128)} parameter(2)
  %dynamic-slice.1336 = bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} dynamic-slice(bf16[1,10,256,2,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %pad.366, s32[]{:T(128)} %constant.5643, s32[]{:T(128)} %constant.5643, s32[]{:T(128)} %constant.5643, s32[]{:T(128)} %constant.5643, /*index=5*/s32[]{:T(128)} %param_2.232, s32[]{:T(128)} %constant.5643, s32[]{:T(128)} %constant.5643), dynamic_slice_sizes={1,10,256,2,1,576,1}
  ROOT %maximum.512 = bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} maximum(bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %dynamic-slice.1330, bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %dynamic-slice.1336)
}

%fused_computation.114 (param_0.269: bf16[16,1024,1,10,256,1], param_1.327: s32[], param_2.228: bf16[1,10,256,1,32,576,1], param_3.196: s32[]) -> bf16[2,1,576,16,1024,1,1] {
  %param_1.327 = s32[]{:T(128)} parameter(1)
  %param_2.228 = bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} parameter(2)
  %param_3.196 = s32[]{:T(128)} parameter(3)
  %fusion.157 = bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} fusion(s32[]{:T(128)} %param_1.327, bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %param_2.228, s32[]{:T(128)} %param_3.196), kind=kLoop, calls=%fused_computation.125.clone
  %param_0.269 = bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} parameter(0)
  %fusion.145 = bf16[16,1024,1,10,256,1,1]{4,1,6,3,2,0,5:T(8,128)(2,1)} fusion(bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %param_0.269), kind=kLoop, calls=%fused_computation.115
  ROOT %convolution.167 = bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} convolution(bf16[1,10,256,2,1,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %fusion.157, bf16[16,1024,1,10,256,1,1]{4,1,6,3,2,0,5:T(8,128)(2,1)} %fusion.145), window={size=1x16x1x10x1 pad=0_0x15_15x0_0x0_0x0_0 rhs_reversal=0x1x0x0x0}, dim_labels=23f40b1_1o23i04->40b1f23
}

%fused_computation.121 (param_0.254: bf16[1,576,16,1024,1,1], param_1.303: bf16[2,1,576,16,1024,1,1]) -> bf16[1,576,16,1024,1,1] {
  %param_0.254 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} parameter(0)
  %param_1.303 = bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} parameter(1)
  %slice.1129 = bf16[1,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} slice(bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} %param_1.303), slice={[0:1], [0:1], [0:576], [0:16], [0:1024], [0:1], [0:1]}
  %bitcast.675 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} bitcast(bf16[1,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} %slice.1129)
  ROOT %add.3124 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} add(bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} %param_0.254, bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} %bitcast.675)
}

%fused_computation.119 (param_0.251: bf16[1,576,16,1024,1,1], param_1.300: bf16[2,576,16,1024,1,1]) -> bf16[1,576,16,1024,1,1] {
  %param_0.251 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} parameter(0)
  %param_1.300 = bf16[2,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} parameter(1)
  %slice.1126 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} slice(bf16[2,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %param_1.300), slice={[1:2], [0:576], [0:16], [0:1024], [0:1], [0:1]}
  ROOT %add.3123 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} add(bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %param_0.251, bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %slice.1126)
}

%fused_computation.118 (param_0.250: bf16[1,576,16,1024,1,1], param_1.298: bf16[2,576,16,1024,1,1]) -> bf16[1,576,16,1024,1,1] {
  %param_0.250 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} parameter(0)
  %param_1.298 = bf16[2,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} parameter(1)
  %slice.1125 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} slice(bf16[2,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %param_1.298), slice={[1:2], [0:576], [0:16], [0:1024], [0:1], [0:1]}
  ROOT %add.3122 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} add(bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %param_0.250, bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %slice.1125)
}
)";
  hlo_string += R"(
ENTRY entry {
  %constant.4782 = u32[]{:T(128)} constant(16)
  %constant.4661 = u32[]{:T(128)} constant(2)
  %constant.4517 = u32[]{:T(128)} constant(31)
  %constant.3078 = u32[]{:T(128)} constant(1)
  %constant.3060 = u32[]{:T(128)} constant(17)
  %partition-id.6 = u32[]{:T(128)} partition-id()
  %param.139 = (bf16[1,10,256,32,576,1]{2,4,1,3,5,0:T(8,128)(2,1)}, bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)}, bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)}, bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)}, u32[]{:T(128)}) parameter(0)
  %get-tuple-element.16007 = u32[]{:T(128)} get-tuple-element((bf16[1,10,256,32,576,1]{2,4,1,3,5,0:T(8,128)(2,1)}, bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)}, bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)}, bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)}, u32[]{:T(128)}) %param.139), index=4
  %copy.1385 = u32[]{:T(128)} copy(u32[]{:T(128)} %get-tuple-element.16007)
  %add.1492 = u32[]{:T(128)} add(u32[]{:T(128)} %copy.1385, u32[]{:T(128)} %constant.3078)
  %add.1938 = u32[]{:T(128)} add(u32[]{:T(128)} %copy.1385, u32[]{:T(128)} %constant.4661)
  %get-tuple-element.16004 = bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} get-tuple-element((bf16[1,10,256,32,576,1]{2,4,1,3,5,0:T(8,128)(2,1)}, bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)}, bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)}, bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)}, u32[]{:T(128)}) %param.139), index=1
  %get-tuple-element.16003 = bf16[1,10,256,32,576,1]{2,4,1,3,5,0:T(8,128)(2,1)} get-tuple-element((bf16[1,10,256,32,576,1]{2,4,1,3,5,0:T(8,128)(2,1)}, bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)}, bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)}, bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)}, u32[]{:T(128)}) %param.139), index=0
  %bitcast.58 = bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} bitcast(bf16[1,10,256,32,576,1]{2,4,1,3,5,0:T(8,128)(2,1)} %get-tuple-element.16003)
  %get-tuple-element.16005 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} get-tuple-element((bf16[1,10,256,32,576,1]{2,4,1,3,5,0:T(8,128)(2,1)}, bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)}, bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)}, bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)}, u32[]{:T(128)}) %param.139), index=2
  %get-tuple-element.16006 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} get-tuple-element((bf16[1,10,256,32,576,1]{2,4,1,3,5,0:T(8,128)(2,1)}, bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)}, bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)}, bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)}, u32[]{:T(128)}) %param.139), index=3
  %constant.3058 = u32[1024]{0:T(1024)} constant({...})
  %dynamic-slice.218 = u32[1]{0:T(128)} dynamic-slice(u32[1024]{0:T(1024)} %constant.3058, u32[]{:T(128)} %partition-id.6), dynamic_slice_sizes={1}
  %bitcast.59 = u32[]{:T(128)} bitcast(u32[1]{0:T(128)} %dynamic-slice.218)
  %add.1493 = u32[]{:T(128)} add(u32[]{:T(128)} %add.1492, u32[]{:T(128)} %bitcast.59)
  %add.1495 = u32[]{:T(128)} add(u32[]{:T(128)} %add.1493, u32[]{:T(128)} %constant.3060)
  %and.28 = u32[]{:T(128)} and(u32[]{:T(128)} %add.1495, u32[]{:T(128)} %constant.4517)
  %add.2636 = u32[]{:T(128)} add(u32[]{:T(128)} %bitcast.59, u32[]{:T(128)} %constant.4782)
  %subtract.200 = u32[]{:T(128)} subtract(u32[]{:T(128)} %add.2636, u32[]{:T(128)} %add.1492)
  %and.29 = u32[]{:T(128)} and(u32[]{:T(128)} %subtract.200, u32[]{:T(128)} %constant.4517)
  %subtract.198 = u32[]{:T(128)} subtract(u32[]{:T(128)} %add.2636, u32[]{:T(128)} %copy.1385)
  %and.27 = u32[]{:T(128)} and(u32[]{:T(128)} %subtract.198, u32[]{:T(128)} %constant.4517)
  %add.1487 = u32[]{:T(128)} add(u32[]{:T(128)} %copy.1385, u32[]{:T(128)} %bitcast.59)
  %add.1489 = u32[]{:T(128)} add(u32[]{:T(128)} %add.1487, u32[]{:T(128)} %constant.3060)
  %and.26 = u32[]{:T(128)} and(u32[]{:T(128)} %add.1489, u32[]{:T(128)} %constant.4517)
  %iota.60 = s32[32]{0:T(128)} iota(), iota_dimension=0
  %fusion.2068 = (s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) fusion(s32[32]{0:T(128)} %iota.60, u32[]{:T(128)} %and.29, u32[]{:T(128)} %and.28, u32[]{:T(128)} %and.27, u32[]{:T(128)} %and.26), kind=kLoop, calls=%fused_computation.1851
  %get-tuple-element.15499 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) %fusion.2068), index=3
  %bitcast.60 = s32[]{:T(128)} bitcast(s32[1]{0:T(128)} %get-tuple-element.15499)
  %get-tuple-element.15498 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) %fusion.2068), index=2
  %bitcast.61 = s32[]{:T(128)} bitcast(s32[1]{0:T(128)} %get-tuple-element.15498)
  %get-tuple-element.15497 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) %fusion.2068), index=1
  %bitcast.64 = s32[]{:T(128)} bitcast(s32[1]{0:T(128)} %get-tuple-element.15497)
  %get-tuple-element.15496 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) %fusion.2068), index=0
  %bitcast.65 = s32[]{:T(128)} bitcast(s32[1]{0:T(128)} %get-tuple-element.15496)
  %collective-permute.9 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} collective-permute(bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %get-tuple-element.16006), channel_id=23, source_target_pairs={{0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,8},{8,9},{9,10},{10,11},{11,12},{12,13},{13,14},{14,15},{15,16},{16,17},{17,18},{18,19},{19,20},{20,21},{21,22},{22,23},{23,24},{24,25},{25,26},{26,27},{27,28},{28,29},{29,30},{30,31},{31,0}}, backend_config="{\"flag_configs\":[],\"barrier_config\":{\"barrier_type\":\"CUSTOM\",\"id\":\"1\"}}"
  %collective-permute.8 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} collective-permute(bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} %get-tuple-element.16005), channel_id=22, source_target_pairs={{0,31},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6},{8,7},{9,8},{10,9},{11,10},{12,11},{13,12},{14,13},{15,14},{16,15},{17,16},{18,17},{19,18},{20,19},{21,20},{22,21},{23,22},{24,23},{25,24},{26,25},{27,26},{28,27},{29,28},{30,29},{31,30}}, backend_config="{\"flag_configs\":[],\"barrier_config\":{\"barrier_type\":\"CUSTOM\",\"id\":\"0\"}}"
  %fusion.144 = bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} fusion(bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %get-tuple-element.16004, s32[]{:T(128)} %bitcast.64, bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %bitcast.58, s32[]{:T(128)} %bitcast.65), kind=kOutput, calls=%fused_computation.114
  %bitcast.68 = bf16[2,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} bitcast(bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} %fusion.144)
  %fusion.146 = bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} fusion(bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %get-tuple-element.16004, s32[]{:T(128)} %bitcast.60, bf16[1,10,256,1,32,576,1]{2,5,3,1,0,6,4:T(8,128)(2,1)} %bitcast.58, s32[]{:T(128)} %bitcast.61), kind=kOutput, calls=%fused_computation.116
  %fusion.153 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} fusion(bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} %collective-permute.8, bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} %fusion.146), kind=kLoop, calls=%fused_computation.123
  %collective-permute.10 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} collective-permute(bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} %fusion.153), channel_id=24, source_target_pairs={{0,31},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6},{8,7},{9,8},{10,9},{11,10},{12,11},{13,12},{14,13},{15,14},{16,15},{17,16},{18,17},{19,18},{20,19},{21,20},{22,21},{23,22},{24,23},{25,24},{26,25},{27,26},{28,27},{29,28},{30,29},{31,30}}, backend_config="{\"flag_configs\":[],\"barrier_config\":{\"barrier_type\":\"CUSTOM\",\"id\":\"0\"}}"
  %fusion.151 = bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} fusion(bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} %collective-permute.10, bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} %fusion.144), kind=kLoop, calls=%fused_computation.121
  %bitcast.67 = bf16[2,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} bitcast(bf16[2,1,576,16,1024,1,1]{4,2,0,6,5,3,1:T(8,128)(2,1)} %fusion.146)
  %fusion.149 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} fusion(bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %collective-permute.9, bf16[2,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %bitcast.67), kind=kLoop, calls=%fused_computation.119
  %collective-permute.11 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} collective-permute(bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %fusion.149), channel_id=25, source_target_pairs={{0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,8},{8,9},{9,10},{10,11},{11,12},{12,13},{13,14},{14,15},{15,16},{16,17},{17,18},{18,19},{19,20},{20,21},{21,22},{22,23},{23,24},{24,25},{25,26},{26,27},{27,28},{28,29},{29,30},{30,31},{31,0}}, backend_config="{\"flag_configs\":[],\"barrier_config\":{\"barrier_type\":\"CUSTOM\",\"id\":\"1\"}}"
  %fusion.148 = bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} fusion(bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %collective-permute.11, bf16[2,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %bitcast.68), kind=kLoop, calls=%fused_computation.118
  ROOT %tuple.1373 = (bf16[1,10,256,32,576,1]{2,4,1,3,5,0:T(8,128)(2,1)}, bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)}, bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)}, bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)}, u32[]{:T(128)}) tuple(bf16[1,10,256,32,576,1]{2,4,1,3,5,0:T(8,128)(2,1)} %get-tuple-element.16003, bf16[16,1024,1,10,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %get-tuple-element.16004, bf16[1,576,16,1024,1,1]{3,1,2,5,4,0:T(8,128)(2,1)} %fusion.151, bf16[1,576,16,1024,1,1]{3,1,0,2,5,4:T(8,128)(2,1)} %fusion.148, u32[]{:T(128)} %add.1938)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start"),
            GetIndex(new_instruction_sequence, "fusion.146"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.1"),
            GetIndex(new_instruction_sequence, "fusion.146"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done"),
            GetIndex(new_instruction_sequence, "fusion.146"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done.1"),
            GetIndex(new_instruction_sequence, "fusion.146"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.2"),
            GetIndex(new_instruction_sequence, "fusion.144"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.3"),
            GetIndex(new_instruction_sequence, "fusion.144"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done.2"),
            GetIndex(new_instruction_sequence, "fusion.144"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done.3"),
            GetIndex(new_instruction_sequence, "fusion.144"));
}

TEST_F(LatencyHidingSchedulerTest,
       BalanceChainedCollectivePermutesLoopedEinsum3) {
  std::string hlo_string = R"(
HloModule module, is_scheduled=true

%fused_computation.1799 (param_0.4926: s32[16], param_1.5709: u32[], param_2.3976: u32[], param_3.3386: u32[], param_4.2299: u32[]) -> (s32[1], s32[1], s32[1], s32[1]) {
  %param_0.4926 = s32[16]{0:T(128)} parameter(0)
  %param_1.5709 = u32[]{:T(128)} parameter(1)
  %dynamic-slice.1611 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4926, u32[]{:T(128)} %param_1.5709), dynamic_slice_sizes={1}
  %param_2.3976 = u32[]{:T(128)} parameter(2)
  %dynamic-slice.1612 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4926, u32[]{:T(128)} %param_2.3976), dynamic_slice_sizes={1}
  %param_3.3386 = u32[]{:T(128)} parameter(3)
  %dynamic-slice.1613 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4926, u32[]{:T(128)} %param_3.3386), dynamic_slice_sizes={1}
  %param_4.2299 = u32[]{:T(128)} parameter(4)
  %dynamic-slice.1614 = s32[1]{0:T(128)} dynamic-slice(s32[16]{0:T(128)} %param_0.4926, u32[]{:T(128)} %param_4.2299), dynamic_slice_sizes={1}
  ROOT %tuple.1346 = (s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) tuple(s32[1]{0:T(128)} %dynamic-slice.1611, s32[1]{0:T(128)} %dynamic-slice.1612, s32[1]{0:T(128)} %dynamic-slice.1613, s32[1]{0:T(128)} %dynamic-slice.1614)
}

%fused_computation.243 (param_0.505: bf16[8,2048,2,576,1,1], param_1.586: bf16[8,2048,2,576,1,1]) -> bf16[8,2048,4,576,1,1] {
  %param_1.586 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} parameter(1)
  %constant.5838 = bf16[]{:T(256)} constant(-inf)
  %pad.368 = bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} pad(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %param_1.586, bf16[]{:T(256)} %constant.5838), padding=0_0x0_0x0_2x0_0x0_0x0_0
  %param_0.505 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} parameter(0)
  %pad.367 = bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} pad(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %param_0.505, bf16[]{:T(256)} %constant.5838), padding=0_0x0_0x2_0x0_0x0_0x0_0
  ROOT %maximum.528 = bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} maximum(bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %pad.368, bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %pad.367)
}

%fused_computation.244 (param_0.507: bf16[8,2048,2,576,1,1], param_1.585: bf16[8,2048,2,576,1,1]) -> bf16[8,2048,4,576,1,1] {
  %param_1.585 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} parameter(1)
  %constant.5832 = bf16[]{:T(256)} constant(-inf)
  %pad.370 = bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} pad(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %param_1.585, bf16[]{:T(256)} %constant.5832), padding=0_0x0_0x0_2x0_0x0_0x0_0
  %param_0.507 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} parameter(0)
  %pad.369 = bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} pad(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %param_0.507, bf16[]{:T(256)} %constant.5832), padding=0_0x0_0x2_0x0_0x0_0x0_0
  ROOT %maximum.529 = bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} maximum(bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %pad.370, bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %pad.369)
}

%fused_computation.247 (param_0.511: bf16[8,2048,2,2,576,1,1]) -> bf16[8,2048,2,2,576,1,1] {
  %param_0.511 = bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} parameter(0)
  ROOT %copy.2292 = bf16[8,2048,2,2,576,1,1]{1,4,2,3,6,5,0:T(8,128)(2,1)} copy(bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} %param_0.511)
}

%fused_computation.248.clone (param_0.526: s32[], param_1.589: bf16[1,32,576,1,36,256,1], param_2.400: s32[]) -> bf16[2,2,576,1,36,256,1] {
  %param_1.589 = bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} parameter(1)
  %constant.5843 = bf16[]{:T(256)} constant(-inf)
  %pad.378 = bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} pad(bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %param_1.589, bf16[]{:T(256)} %constant.5843), padding=0_1x0_0x0_0x0_0x0_0x0_0x0_0
  %constant.5853 = s32[]{:T(128)} constant(0)
  %param_0.526 = s32[]{:T(128)} parameter(0)
  %dynamic-slice.1382 = bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} dynamic-slice(bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %pad.378, s32[]{:T(128)} %constant.5853, s32[]{:T(128)} %param_0.526, s32[]{:T(128)} %constant.5853, s32[]{:T(128)} %constant.5853, /*index=5*/s32[]{:T(128)} %constant.5853, s32[]{:T(128)} %constant.5853, s32[]{:T(128)} %constant.5853), dynamic_slice_sizes={2,2,576,1,36,256,1}
  %pad.377 = bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} pad(bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %param_1.589, bf16[]{:T(256)} %constant.5843), padding=1_0x0_0x0_0x0_0x0_0x0_0x0_0
  %param_2.400 = s32[]{:T(128)} parameter(2)
  %dynamic-slice.1381 = bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} dynamic-slice(bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %pad.377, s32[]{:T(128)} %constant.5853, s32[]{:T(128)} %param_2.400, s32[]{:T(128)} %constant.5853, s32[]{:T(128)} %constant.5853, /*index=5*/s32[]{:T(128)} %constant.5853, s32[]{:T(128)} %constant.5853, s32[]{:T(128)} %constant.5853), dynamic_slice_sizes={2,2,576,1,36,256,1}
  ROOT %maximum.532 = bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} maximum(bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %dynamic-slice.1382, bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %dynamic-slice.1381)
}

%fused_computation.246 (param_0.521: bf16[8,2048,2,2,576,1,1], param_1.588: s32[], param_2.399: bf16[1,32,576,1,36,256,1], param_3.247: s32[]) -> bf16[8,2048,1,36,256,1,1] {
  %param_0.521 = bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} parameter(0)
  %fusion.268 = bf16[8,2048,2,2,576,1,1]{1,4,2,3,6,5,0:T(8,128)(2,1)} fusion(bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} %param_0.521), kind=kLoop, calls=%fused_computation.247
  %param_1.588 = s32[]{:T(128)} parameter(1)
  %param_2.399 = bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} parameter(2)
  %param_3.247 = s32[]{:T(128)} parameter(3)
  %fusion.271 = bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} fusion(s32[]{:T(128)} %param_1.588, bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %param_2.399, s32[]{:T(128)} %param_3.247), kind=kLoop, calls=%fused_computation.248.clone
  ROOT %convolution.172 = bf16[8,2048,1,36,256,1,1]{4,1,6,5,3,2,0:T(8,128)(2,1)} convolution(bf16[8,2048,2,2,576,1,1]{1,4,2,3,6,5,0:T(8,128)(2,1)} %fusion.268, bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %fusion.271), window={size=1x1x36x2x2 pad=0_0x0_0x35_35x0_0x0_0 rhs_reversal=0x1x1x0x0}, dim_labels=0b43f12_43i12o0->0b12f34
}

%fused_computation.245 (param_0.508: bf16[8,2048,2,2,576,1,1]) -> bf16[8,2048,2,2,576,1,1] {
  %param_0.508 = bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} parameter(0)
  ROOT %copy.2290 = bf16[8,2048,2,2,576,1,1]{1,4,2,3,6,5,0:T(8,128)(2,1)} copy(bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} %param_0.508)
}

%fused_computation.249.clone (param_0.525: s32[], param_1.587: bf16[1,32,576,1,36,256,1], param_2.398: s32[]) -> bf16[2,2,576,1,36,256,1] {
  %param_1.587 = bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} parameter(1)
  %constant.5837 = bf16[]{:T(256)} constant(-inf)
  %pad.382 = bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} pad(bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %param_1.587, bf16[]{:T(256)} %constant.5837), padding=0_1x0_0x0_0x0_0x0_0x0_0x0_0
  %constant.5848 = s32[]{:T(128)} constant(0)
  %param_0.525 = s32[]{:T(128)} parameter(0)
  %dynamic-slice.1386 = bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} dynamic-slice(bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %pad.382, s32[]{:T(128)} %constant.5848, s32[]{:T(128)} %param_0.525, s32[]{:T(128)} %constant.5848, s32[]{:T(128)} %constant.5848, /*index=5*/s32[]{:T(128)} %constant.5848, s32[]{:T(128)} %constant.5848, s32[]{:T(128)} %constant.5848), dynamic_slice_sizes={2,2,576,1,36,256,1}
  %pad.381 = bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} pad(bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %param_1.587, bf16[]{:T(256)} %constant.5837), padding=1_0x0_0x0_0x0_0x0_0x0_0x0_0
  %param_2.398 = s32[]{:T(128)} parameter(2)
  %dynamic-slice.1385 = bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} dynamic-slice(bf16[2,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %pad.381, s32[]{:T(128)} %constant.5848, s32[]{:T(128)} %param_2.398, s32[]{:T(128)} %constant.5848, s32[]{:T(128)} %constant.5848, /*index=5*/s32[]{:T(128)} %constant.5848, s32[]{:T(128)} %constant.5848, s32[]{:T(128)} %constant.5848), dynamic_slice_sizes={2,2,576,1,36,256,1}
  ROOT %maximum.533 = bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} maximum(bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %dynamic-slice.1386, bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %dynamic-slice.1385)
}

%fused_computation.241 (param_0.503: bf16[8,2048,1,36,256,1], param_1.561: bf16[8,2048,1,36,256,1,1], param_2.397: bf16[8,2048,2,2,576,1,1], param_3.246: s32[], param_4.127: bf16[1,32,576,1,36,256,1], param_5.55: s32[]) -> bf16[8,2048,1,36,256,1] {
  %param_0.503 = bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} parameter(0)
  %param_1.561 = bf16[8,2048,1,36,256,1,1]{4,1,6,5,3,2,0:T(8,128)(2,1)} parameter(1)
  %bitcast.599 = bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} bitcast(bf16[8,2048,1,36,256,1,1]{4,1,6,5,3,2,0:T(8,128)(2,1)} %param_1.561)
  %add.3146 = bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} add(bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %param_0.503, bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %bitcast.599)
  %param_2.397 = bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} parameter(2)
  %fusion.266 = bf16[8,2048,2,2,576,1,1]{1,4,2,3,6,5,0:T(8,128)(2,1)} fusion(bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} %param_2.397), kind=kLoop, calls=%fused_computation.245
  %param_3.246 = s32[]{:T(128)} parameter(3)
  %param_4.127 = bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} parameter(4)
  %param_5.55 = s32[]{:T(128)} parameter(5)
  %fusion.272 = bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} fusion(s32[]{:T(128)} %param_3.246, bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %param_4.127, s32[]{:T(128)} %param_5.55), kind=kLoop, calls=%fused_computation.249.clone
  %convolution.171 = bf16[8,2048,1,36,256,1,1]{4,1,6,5,3,2,0:T(8,128)(2,1)} convolution(bf16[8,2048,2,2,576,1,1]{1,4,2,3,6,5,0:T(8,128)(2,1)} %fusion.266, bf16[2,2,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %fusion.272), window={size=1x1x36x2x2 pad=0_0x0_0x35_35x0_0x0_0 rhs_reversal=0x1x1x0x0}, dim_labels=0b43f12_43i12o0->0b12f34
  %bitcast.596 = bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} bitcast(bf16[8,2048,1,36,256,1,1]{4,1,6,5,3,2,0:T(8,128)(2,1)} %convolution.171)
  ROOT %add.3143 = bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} add(bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %add.3146, bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %bitcast.596)
}
)";
  hlo_string += R"(
ENTRY entry {
  %constant.4735 = u32[]{:T(128)} constant(2)
  %constant.4598 = u32[]{:T(128)} constant(15)
  %constant.3341 = u32[]{:T(128)} constant(1)
  %partition-id.16 = u32[]{:T(128)} partition-id()
  %param.149 = (bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}, bf16[32,576,1,36,256,1]{4,1,0,3,5,2:T(8,128)(2,1)}, bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)}, bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}, u32[]{:T(128)}) parameter(0)
  %get-tuple-element.21127 = u32[]{:T(128)} get-tuple-element((bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}, bf16[32,576,1,36,256,1]{4,1,0,3,5,2:T(8,128)(2,1)}, bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)}, bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}, u32[]{:T(128)}) %param.149), index=4
  %copy.2357 = u32[]{:T(128)} copy(u32[]{:T(128)} %get-tuple-element.21127)
  %add.1530 = u32[]{:T(128)} add(u32[]{:T(128)} %copy.2357, u32[]{:T(128)} %constant.3341)
  %add.1943 = u32[]{:T(128)} add(u32[]{:T(128)} %copy.2357, u32[]{:T(128)} %constant.4735)
  %get-tuple-element.21124 = bf16[32,576,1,36,256,1]{4,1,0,3,5,2:T(8,128)(2,1)} get-tuple-element((bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}, bf16[32,576,1,36,256,1]{4,1,0,3,5,2:T(8,128)(2,1)}, bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)}, bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}, u32[]{:T(128)}) %param.149), index=1
  %bitcast.98 = bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} bitcast(bf16[32,576,1,36,256,1]{4,1,0,3,5,2:T(8,128)(2,1)} %get-tuple-element.21124)
  %get-tuple-element.21123 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} get-tuple-element((bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}, bf16[32,576,1,36,256,1]{4,1,0,3,5,2:T(8,128)(2,1)}, bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)}, bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}, u32[]{:T(128)}) %param.149), index=0
  %get-tuple-element.21125 = bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} get-tuple-element((bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}, bf16[32,576,1,36,256,1]{4,1,0,3,5,2:T(8,128)(2,1)}, bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)}, bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}, u32[]{:T(128)}) %param.149), index=2
  %get-tuple-element.21126 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} get-tuple-element((bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}, bf16[32,576,1,36,256,1]{4,1,0,3,5,2:T(8,128)(2,1)}, bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)}, bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}, u32[]{:T(128)}) %param.149), index=3
  %constant.3344 = s32[16]{0:T(128)} constant({...})
  %constant.3339 = u32[256]{0:T(256)} constant({...})
  %dynamic-slice.312 = u32[1]{0:T(128)} dynamic-slice(u32[256]{0:T(256)} %constant.3339, u32[]{:T(128)} %partition-id.16), dynamic_slice_sizes={1}
  %bitcast.99 = u32[]{:T(128)} bitcast(u32[1]{0:T(128)} %dynamic-slice.312)
  %add.1531 = u32[]{:T(128)} add(u32[]{:T(128)} %add.1530, u32[]{:T(128)} %bitcast.99)
  %and.40 = u32[]{:T(128)} and(u32[]{:T(128)} %add.1531, u32[]{:T(128)} %constant.4598)
  %add.2637 = u32[]{:T(128)} add(u32[]{:T(128)} %bitcast.99, u32[]{:T(128)} %constant.4598)
  %subtract.216 = u32[]{:T(128)} subtract(u32[]{:T(128)} %add.2637, u32[]{:T(128)} %add.1530)
  %and.41 = u32[]{:T(128)} and(u32[]{:T(128)} %subtract.216, u32[]{:T(128)} %constant.4598)
  %subtract.214 = u32[]{:T(128)} subtract(u32[]{:T(128)} %add.2637, u32[]{:T(128)} %copy.2357)
  %and.39 = u32[]{:T(128)} and(u32[]{:T(128)} %subtract.214, u32[]{:T(128)} %constant.4598)
  %add.1527 = u32[]{:T(128)} add(u32[]{:T(128)} %copy.2357, u32[]{:T(128)} %bitcast.99)
  %and.38 = u32[]{:T(128)} and(u32[]{:T(128)} %add.1527, u32[]{:T(128)} %constant.4598)
  %fusion.1974 = (s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) fusion(s32[16]{0:T(128)} %constant.3344, u32[]{:T(128)} %and.41, u32[]{:T(128)} %and.40, u32[]{:T(128)} %and.39, u32[]{:T(128)} %and.38), kind=kLoop, calls=%fused_computation.1799
  %get-tuple-element.20616 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) %fusion.1974), index=3
  %bitcast.100 = s32[]{:T(128)} bitcast(s32[1]{0:T(128)} %get-tuple-element.20616)
  %get-tuple-element.20615 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) %fusion.1974), index=2
  %bitcast.101 = s32[]{:T(128)} bitcast(s32[1]{0:T(128)} %get-tuple-element.20615)
  %get-tuple-element.20614 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) %fusion.1974), index=1
  %bitcast.104 = s32[]{:T(128)} bitcast(s32[1]{0:T(128)} %get-tuple-element.20614)
  %get-tuple-element.20613 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}, s32[1]{0:T(128)}) %fusion.1974), index=0
  %bitcast.105 = s32[]{:T(128)} bitcast(s32[1]{0:T(128)} %get-tuple-element.20613)
  %copy.2356 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} copy(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %get-tuple-element.21126)
  %collective-permute.23 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} collective-permute(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %copy.2356), channel_id=51, source_target_pairs={{0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,8},{8,9},{9,10},{10,11},{11,12},{12,13},{13,14},{14,15},{15,0}}, backend_config="{\"flag_configs\":[],\"barrier_config\":{\"barrier_type\":\"CUSTOM\",\"id\":\"1\"},\"scoped_memory_configs\":[]}"
  %copy.2354 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} copy(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %get-tuple-element.21123)
  %collective-permute.22 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} collective-permute(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %copy.2354), channel_id=50, source_target_pairs={{0,15},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6},{8,7},{9,8},{10,9},{11,10},{12,11},{13,12},{14,13},{15,14}}, backend_config="{\"flag_configs\":[],\"barrier_config\":{\"barrier_type\":\"CUSTOM\",\"id\":\"0\"},\"scoped_memory_configs\":[]}"
  %fusion.264 = bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} fusion(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %copy.2356, bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %copy.2354), kind=kLoop, calls=%fused_computation.243
  %bitcast.97 = bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} bitcast(bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %fusion.264)
  %collective-permute.24 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} collective-permute(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %collective-permute.22), channel_id=52, source_target_pairs={{0,15},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6},{8,7},{9,8},{10,9},{11,10},{12,11},{13,12},{14,13},{15,14}}, backend_config="{\"flag_configs\":[],\"barrier_config\":{\"barrier_type\":\"CUSTOM\",\"id\":\"0\"},\"scoped_memory_configs\":[]}"
  %fusion.265 = bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} fusion(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %collective-permute.23, bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %collective-permute.22), kind=kLoop, calls=%fused_computation.244
  %collective-permute.25 = bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} collective-permute(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %collective-permute.23), channel_id=53, source_target_pairs={{0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,8},{8,9},{9,10},{10,11},{11,12},{12,13},{13,14},{14,15},{15,0}}, backend_config="{\"flag_configs\":[],\"barrier_config\":{\"barrier_type\":\"CUSTOM\",\"id\":\"1\"},\"scoped_memory_configs\":[]}"
  %bitcast.103 = bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} bitcast(bf16[8,2048,4,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %fusion.265)
  %fusion.267 = bf16[8,2048,1,36,256,1,1]{4,1,6,5,3,2,0:T(8,128)(2,1)} fusion(bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} %bitcast.97, s32[]{:T(128)} %bitcast.100, bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %bitcast.98, s32[]{:T(128)} %bitcast.101), kind=kOutput, calls=%fused_computation.246
  %fusion.262 = bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} fusion(bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %get-tuple-element.21125, bf16[8,2048,1,36,256,1,1]{4,1,6,5,3,2,0:T(8,128)(2,1)} %fusion.267, bf16[8,2048,2,2,576,1,1]{1,4,6,5,3,2,0:T(8,128)(2,1)} %bitcast.103, s32[]{:T(128)} %bitcast.104, bf16[1,32,576,1,36,256,1]{5,2,0,1,4,3,6:T(8,128)(2,1)} %bitcast.98, /*index=5*/s32[]{:T(128)} %bitcast.105), kind=kOutput, calls=%fused_computation.241
  ROOT %tuple.1419 = (bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}, bf16[32,576,1,36,256,1]{4,1,0,3,5,2:T(8,128)(2,1)}, bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)}, bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)}, u32[]{:T(128)}) tuple(bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %collective-permute.24, bf16[32,576,1,36,256,1]{4,1,0,3,5,2:T(8,128)(2,1)} %get-tuple-element.21124, bf16[8,2048,1,36,256,1]{4,1,3,0,5,2:T(8,128)(2,1)} %fusion.262, bf16[8,2048,2,576,1,1]{1,3,2,0,5,4:T(8,128)(2,1)} %collective-permute.25, u32[]{:T(128)} %add.1943)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start"),
            GetIndex(new_instruction_sequence, "fusion.267"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.1"),
            GetIndex(new_instruction_sequence, "fusion.267"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done"),
            GetIndex(new_instruction_sequence, "fusion.267"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done.1"),
            GetIndex(new_instruction_sequence, "fusion.267"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.2"),
            GetIndex(new_instruction_sequence, "fusion.262"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.3"),
            GetIndex(new_instruction_sequence, "fusion.262"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done.2"),
            GetIndex(new_instruction_sequence, "fusion.262"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done.3"),
            GetIndex(new_instruction_sequence, "fusion.262"));
}

TEST_F(LatencyHidingSchedulerTest, MoveCentainConv2) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,64]{2,1,0} parameter(3)
  cp0 = f32[16,64,256]{2,1,0} collective-permute(p0),
    source_target_pairs={{0,1},{1,0}}
  cp1 = f32[16,64,256]{2,1,0} collective-permute(p1),
    source_target_pairs={{0,1},{1,0}}
  cp2 = f32[16,64,256]{2,1,0} collective-permute(cp0),
    source_target_pairs={{0,1},{1,0}}
  cp3 = f32[16,64,256]{2,1,0} collective-permute(cp1),
    source_target_pairs={{0,1},{1,0}}
  a0 = f32[16,64,256]{2,1,0} add(cp0, cp1)
  c0 = f32[16,64,256]{2,1,0} convolution(p2, p3),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(a0, c0),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, f32[16,256,256]{2,1,0}) tuple(cp2, cp3, c1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.1"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done.1"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.2"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.3"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done.2"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-done.3"),
            GetIndex(new_instruction_sequence, "c1"));
}

TEST_F(LatencyHidingSchedulerTest, WhileOverlapLimit) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

while_cond {
  param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
  ROOT gte = pred[] get-tuple-element(param), index=2
}

while_body {
  param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
  gte0 = bf16[8]{0} get-tuple-element(param), index=0
  gte1 = pred[] get-tuple-element(param), index=2
  bitcast = bf16[8]{0} bitcast(gte0)
  collective-permute.1 = bf16[8]{0} collective-permute(gte0), source_target_pairs={{0,1},{1,2},{2,3}}
  add0 = bf16[8]{0} add(collective-permute.1, bitcast)
  negate = bf16[8]{0} negate(add0)
  collective-permute.2 = bf16[8]{0} collective-permute(collective-permute.1), source_target_pairs={{1,0},{0,3},{3,2}}
  ROOT tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.2, negate, gte1)
}

ENTRY entry {
  p0 = bf16[8]{0} parameter(0)
  p1 = bf16[8]{0} parameter(1)
  p2 = pred[] parameter(2)
  tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(p0, p1, p2)
  while = (bf16[8]{0}, bf16[8]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
  collective-permute.3 = bf16[8]{0} collective-permute(p1), source_target_pairs={{0,1},{1,2},{2,3}}
  gte0 = bf16[8]{0} get-tuple-element(while), index=0
  gte1 = bf16[8]{0} get-tuple-element(while), index=1
  add = bf16[8]{0} add(gte0, gte1)
  ROOT add2 = bf16[8]{0} add(add, collective-permute.3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.collective_permute_overlap_limit = 2;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // Do not overlap if the sum of collectives inside the loop + the collective
  // we are trying to overlap would go beyond the overlap limit.
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-start.2"),
            GetIndex(new_instruction_sequence, "while"));
}

TEST_F(LatencyHidingSchedulerTest, WhileNestedOverlapLimit) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

while_cond {
  param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
  ROOT gte = pred[] get-tuple-element(param), index=2
}

while_body {
  param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
  gte0 = bf16[8]{0} get-tuple-element(param), index=0
  gte1 = pred[] get-tuple-element(param), index=2
  bitcast = bf16[8]{0} bitcast(gte0)
  collective-permute.1 = bf16[8]{0} collective-permute(gte0), source_target_pairs={{0,1},{1,2},{2,3}}
  add0 = bf16[8]{0} add(collective-permute.1, bitcast)
  negate = bf16[8]{0} negate(add0)
  ROOT tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.1, negate, gte1)
}

while_cond2 {
  param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
  ROOT gte = pred[] get-tuple-element(param), index=2
}

while_body2 {
  param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
  while.1 = (bf16[8]{0}, bf16[8]{0}, pred[]) while(param), condition=while_cond, body=while_body
  gte0 = bf16[8]{0} get-tuple-element(while.1), index=0
  gte1 = pred[] get-tuple-element(while.1), index=2
  bitcast = bf16[8]{0} bitcast(gte0)
  negate = bf16[8]{0} negate(bitcast)
  collective-permute.2 = bf16[8]{0} collective-permute(negate), source_target_pairs={{1,0},{0,3},{3,2}}
  ROOT tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.2, negate, gte1)
}

ENTRY entry {
  p0 = bf16[8]{0} parameter(0)
  p1 = bf16[8]{0} parameter(1)
  p2 = pred[] parameter(2)
  tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(p0, p1, p2)
  while = (bf16[8]{0}, bf16[8]{0}, pred[]) while(tuple), condition=while_cond2, body=while_body2
  collective-permute.3 = bf16[8]{0} collective-permute(p1), source_target_pairs={{0,1},{1,2},{2,3}}
  gte0 = bf16[8]{0} get-tuple-element(while), index=0
  gte1 = bf16[8]{0} get-tuple-element(while), index=1
  add = bf16[8]{0} add(gte0, gte1)
  ROOT add2 = bf16[8]{0} add(add, collective-permute.3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.collective_permute_overlap_limit = 2;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // Do not overlap if the sum of collectives inside the loop + the collective
  // we are trying to overlap would go beyond the overlap limit.
  EXPECT_GT(GetIndex(new_instruction_sequence, "collective-permute-start.2"),
            GetIndex(new_instruction_sequence, "while"));
}

TEST_F(LatencyHidingSchedulerTest, WhileOverlapUnderLimit) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

while_cond {
  param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
  ROOT gte = pred[] get-tuple-element(param), index=2
}

while_body {
  param = (bf16[8]{0}, bf16[8]{0}, pred[]) parameter(0)
  gte0 = bf16[8]{0} get-tuple-element(param), index=0
  gte1 = pred[] get-tuple-element(param), index=2
  bitcast = bf16[8]{0} bitcast(gte0)
  collective-permute.1 = bf16[8]{0} collective-permute(gte0), source_target_pairs={{0,1},{1,2},{2,3}}
  add0 = bf16[8]{0} add(collective-permute.1, bitcast)
  negate = bf16[8]{0} negate(add0)
  collective-permute.2 = bf16[8]{0} collective-permute(collective-permute.1), source_target_pairs={{1,0},{0,3},{3,2}}
  ROOT tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.2, negate, gte1)
}

ENTRY entry {
  p0 = bf16[8]{0} parameter(0)
  p1 = bf16[8]{0} parameter(1)
  p2 = pred[] parameter(2)
  tuple = (bf16[8]{0}, bf16[8]{0}, pred[]) tuple(p0, p1, p2)
  while = (bf16[8]{0}, bf16[8]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
  collective-permute.3 = bf16[8]{0} collective-permute(p1), source_target_pairs={{0,1},{1,2},{2,3}}
  gte0 = bf16[8]{0} get-tuple-element(while), index=0
  gte1 = bf16[8]{0} get-tuple-element(while), index=1
  add = bf16[8]{0} add(gte0, gte1)
  ROOT add2 = bf16[8]{0} add(add, collective-permute.3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.collective_permute_overlap_limit = 3;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // Do not overlap if the sum of collectives inside the loop + the collective
  // we are trying to overlap would go beyond the overlap limit.
  EXPECT_LT(GetIndex(new_instruction_sequence, "collective-permute-start.2"),
            GetIndex(new_instruction_sequence, "while"));
}

TEST_F(LatencyHidingSchedulerTest, WhileOverlapLimitAllGather) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

while_cond {
  param = (bf16[4]{0}, bf16[8]{0}, pred[]) parameter(0)
  ROOT gte = pred[] get-tuple-element(param), index=2
}

while_body {
  param = (bf16[4]{0}, bf16[8]{0}, pred[]) parameter(0)
  gte0 = bf16[4]{0} get-tuple-element(param), index=0
  gte1 = bf16[8]{0} get-tuple-element(param), index=1
  gte2 = pred[] get-tuple-element(param), index=2
  bitcast = bf16[8]{0} bitcast(gte0)
  all-gather.1 = bf16[8]{0} all-gather(gte0), replica_groups={{0,1},{2,3}}, dimensions={0}, channel_id=1
  add0 = bf16[8]{0} add(all-gather.1, bitcast)
  negate = bf16[8]{0} negate(add0)
  collective-permute.2 = bf16[4]{0} collective-permute(gte0), source_target_pairs={{1,0},{0,3},{3,2}}
  ROOT tuple = (bf16[4]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.2, negate, gte2)
}

ENTRY entry {
  p0 = bf16[4]{0} parameter(0)
  p1 = bf16[8]{0} parameter(1)
  p2 = pred[] parameter(2)
  tuple = (bf16[4]{0}, bf16[8]{0}, pred[]) tuple(p0, p1, p2)
  while = (bf16[4]{0}, bf16[8]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
  all-gather.2 = bf16[8]{0} all-gather(p0), replica_groups={{0,1},{2,3}}, dimensions={0}, channel_id=2
  gte0 = bf16[4]{0} get-tuple-element(while), index=0
  gte1 = bf16[8]{0} get-tuple-element(while), index=1
  ROOT tuple.2 = (bf16[4]{0}, bf16[8]{0}, bf16[8]{0}) tuple(gte0, gte1, all-gather.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.collective_permute_overlap_limit = 2;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // Do not overlap if the sum of collectives inside the loop + the collective
  // we are trying to overlap would go beyond the overlap limit.
  EXPECT_GT(GetIndex(new_instruction_sequence, "all-gather-start.1"),
            GetIndex(new_instruction_sequence, "while"));
}

TEST_F(LatencyHidingSchedulerTest, WhileOverlapUnderLimitAllGather) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

while_cond {
  param = (bf16[4]{0}, bf16[8]{0}, pred[]) parameter(0)
  ROOT gte = pred[] get-tuple-element(param), index=2
}

while_body {
  param = (bf16[4]{0}, bf16[8]{0}, pred[]) parameter(0)
  gte0 = bf16[4]{0} get-tuple-element(param), index=0
  gte1 = bf16[8]{0} get-tuple-element(param), index=1
  gte2 = pred[] get-tuple-element(param), index=2
  bitcast = bf16[8]{0} bitcast(gte0)
  all-gather.1 = bf16[8]{0} all-gather(gte0), replica_groups={{0,1},{2,3}}, dimensions={0}, channel_id=1
  add0 = bf16[8]{0} add(all-gather.1, bitcast)
  negate = bf16[8]{0} negate(add0)
  collective-permute.2 = bf16[4]{0} collective-permute(gte0), source_target_pairs={{1,0},{0,3},{3,2}}
  ROOT tuple = (bf16[4]{0}, bf16[8]{0}, pred[]) tuple(collective-permute.2, negate, gte2)
}

ENTRY entry {
  p0 = bf16[4]{0} parameter(0)
  p1 = bf16[8]{0} parameter(1)
  p2 = pred[] parameter(2)
  tuple = (bf16[4]{0}, bf16[8]{0}, pred[]) tuple(p0, p1, p2)
  while = (bf16[4]{0}, bf16[8]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
  all-gather.2 = bf16[8]{0} all-gather(p0), replica_groups={{0,1},{2,3}}, dimensions={0}, channel_id=2
  gte0 = bf16[4]{0} get-tuple-element(while), index=0
  gte1 = bf16[8]{0} get-tuple-element(while), index=1
  ROOT tuple.2 = (bf16[4]{0}, bf16[8]{0}, bf16[8]{0}) tuple(gte0, gte1, all-gather.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.collective_permute_overlap_limit = 2;
  sched_config.all_gather_overlap_limit = 2;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // Do not overlap if the sum of collectives inside the loop + the collective
  // we are trying to overlap would go beyond the overlap limit.
  EXPECT_LT(GetIndex(new_instruction_sequence, "all-gather-start.1"),
            GetIndex(new_instruction_sequence, "while"));
}

TEST_F(LatencyHidingSchedulerTest, AllToAllAsyncBalance) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

async_computation {
  p = f32[2,8,256,256] parameter(0)
  ROOT ata = f32[2,8,256,256] all-to-all(p), dimensions={0}, replica_groups={{0,1}}
}

async_computation.2 {
  p.2 = f32[2,8,256,256] parameter(0)
  ROOT ata.1 = f32[2,8,256,256] all-to-all(p.2), dimensions={0}, replica_groups={{0,1}}
}


ENTRY %module {
  %constant.19 = u32[] constant(0)
  %replica_id = u32[]{:T(128)} replica-id()
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %color_operand.1 = f32[2,8,256,256]{3,2,1,0} broadcast(
    f32[]{:T(128)} %convert), dimensions={}
  %color_operand.2 = f32[2,8,256,256]{3,2,1,0} broadcast(
    f32[]{:T(128)} %convert), dimensions={}
  %ata-start = ((f32[2,8,256,256]), f32[2,8,256,256], u32[], u32[]) async-start(
    f32[2,8,256,256] %color_operand.1), calls=async_computation,
    metadata={op_type="AllToAll" op_name="ata0"}
  %ata-start.2 = ((f32[2,8,256,256]), f32[2,8,256,256], u32[], u32[]) async-start(
    f32[2,8,256,256] %color_operand.2), calls=async_computation.2,
    metadata={op_type="AllToAll" op_name="ata1"}
  %ata-done = f32[2,8,256,256] async-done(%ata-start), calls=async_computation,
    metadata={op_type="AllToAll" op_name="ata0"}
  %ata-done-bc = f32[16,256,256] bitcast(f32[2,8,256,256] %ata-done),
    metadata={op_type="Bitcast" op_name="ata0"}
  %ata-done.2 = f32[2,8,256,256] async-done(%ata-start.2), calls=async_computation.2,
    metadata={op_type="AllToAll" op_name="ata1"}
  %ata-done-bc.2 = f32[16,256,256] bitcast(f32[2,8,256,256] %ata-done.2),
    metadata={op_type="Bitcast" op_name="ata1"}
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
    metadata={op_type="AllToAll" op_name="c0"}
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
    metadata={op_type="AllToAll" op_name="c1"}
  a2 = f32[16,256,256]{2,1,0} add(c1, c0)
  ROOT t = (f32[16,256,256], f32[16,256,256], f32[16,256,256]) tuple(a2, %ata-done-bc.2, %ata-done-bc)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  TF_EXPECT_OK(RunScheduler(hlo_module.get()));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
  // We expect that the scheduling would look like this:
  //   %ar-start = async-start()
  //   %c0 = convolution()
  //   %ar-done = async-done()
  //   %ar-start.2 = all-reduce-start()
  //   %c1 = convolution()
  //   %ar-done.2 = f32[2,8,256,256]{3,2,1,0} async-done()
  // This means that the asyncs are balanced over the two convolutions
  // rather than being unbalanced (one of the two asyncs overlaps with
  // both the convolutons and the other with nothing).
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kConvolution,
                                        new_instruction_sequence, "c0"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAsyncDone,
                                        new_instruction_sequence, "ata0"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kConvolution,
                                        new_instruction_sequence, "c0"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAsyncStart,
                                        new_instruction_sequence, "ata0"));
  EXPECT_LT(GetOpcodeIndexUsingMetaData(HloOpcode::kConvolution,
                                        new_instruction_sequence, "c1"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAsyncDone,
                                        new_instruction_sequence, "ata1"));
  EXPECT_GT(GetOpcodeIndexUsingMetaData(HloOpcode::kConvolution,
                                        new_instruction_sequence, "c1"),
            GetOpcodeIndexUsingMetaData(HloOpcode::kAsyncStart,
                                        new_instruction_sequence, "ata1"));
}

TEST_F(LatencyHidingSchedulerTest, ReleaseOneThatStallsLessFirst) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[1024,2048,2048]{2,1,0} parameter(2)
  p3 = f32[2048,2048,2048]{2,1,0} parameter(3)
  cp1s = (f32[1024,2048,2048]{2,1,0}, f32[1024,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0},{0,3},{3,2}}
  cp2s = (f32[2048,2048,2048]{2,1,0}, f32[2048,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p3), source_target_pairs={{1,0},{0,3},{3,2}}
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb,
    metadata={op_type="AllToAll" op_name="c0"}
  cp1d = f32[1024,2048,2048]{2,1,0} collective-permute-done(cp1s)
  cp2d = f32[2048,2048,2048]{2,1,0} collective-permute-done(cp2s)
  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[1024,2048,2048]{2,1,0}, f32[2048,2048,2048]{2,1,0}) tuple(c0, cp1d, cp2d)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.collective_permute_overlap_limit = 2;
  sched_config.all_gather_overlap_limit = 2;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config,
                            std::make_unique<TestLatencyEstimator>()));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // Make sure that between two instructions that are not ready we first emit
  // the one that causes less stall. This allows to potentially expose more
  // opportunities for the other to overlap.
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp2s"),
            GetIndex(new_instruction_sequence, "cp1s"));
}

TEST_F(LatencyHidingSchedulerTest, ReleaseStartWhenLatencyDue) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[128,2048,2048]{2,1,0} parameter(1)
  p2 = f32[512,2048,2048]{2,1,0} parameter(2)
  cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0},{0,3},{3,2}}
  cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s)
  cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0},{0,3},{3,2}}
  cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s)
  cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp2d), source_target_pairs={{1,0},{0,3},{3,2}}
  cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s)
  slice = f32[16,64,256]{2,1,0} slice(f32[512,2048,2048]{2,1,0} cp1d), slice={[0:16], [0:64], [0:256]}
  c0 = f32[16,256,256]{2,1,0} convolution(p0, slice),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(p0, slice),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, c1, cp2d, cp3d)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.aggressive_scheduling_policies = true;
  sched_config.enable_release_start_policy = true;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config,
                            std::make_unique<TestLatencyEstimator>()));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // Make sure that c0 and c1 are latency-hiding cp2 and cp3 respectively,
  // instead of being scheduled only within the latency of one of cp2 or cp3.
  // Note that the cost of c0 and c1 is larger than the latencies of cp2 and cp3
  // so the best strategy is indeed to distribute them among both cp2 and cp3.
  // When aggressive_scheduling_policies = true, this is achieved thanks to the
  // "kScheduleStart" policy, in absence of which the "kAsyncDepth" prevails
  // in scheduling both c0 and c1 within the latency of cp3.
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp2s"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "c0"),
            GetIndex(new_instruction_sequence, "cp2d"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp2d"),
            GetIndex(new_instruction_sequence, "cp3s"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp3s"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "c1"),
            GetIndex(new_instruction_sequence, "cp3d"));
}

TEST_F(LatencyHidingSchedulerTest, AsyncTrackerTestForTargetDefinedResources) {
  // Extend AsyncTracker for a fake target with one target-defined resource
  class AsyncTrackerForMyTarget : public AsyncTracker {
    enum class MyTargetResourceType {
      kTargetResource0 =
          ResourceTypeToIndex(ResourceType::kTargetDefinedResourceTypeBegin),
      kTargetResourceTypeEnd,
    };

   public:
    explicit AsyncTrackerForMyTarget(const SchedulerConfig& config,
                                     int64_t target_resource0_limit = 3)
        : AsyncTracker(config),
          target_resource0_limit_(target_resource0_limit) {}

    absl::string_view GetResourceName(int64_t resource_type) const override {
      CHECK_LT(
          resource_type,
          ResourceTypeToIndex(MyTargetResourceType::kTargetResourceTypeEnd));
      if (resource_type < GetTargetDefinedResourceTypeBegin()) {
        return AsyncTracker::GetResourceName(resource_type);
      }
      switch (resource_type) {
        case static_cast<int64_t>(MyTargetResourceType::kTargetResource0):
          return "kTargetResource0";
        default:
          return "";
      }
    }

    ResourceHazardType GetResourceHazardType(
        int64_t resource_type) const override {
      CHECK_LT(
          resource_type,
          ResourceTypeToIndex(MyTargetResourceType::kTargetResourceTypeEnd));
      if (resource_type < GetTargetDefinedResourceTypeBegin()) {
        return AsyncTracker::GetResourceHazardType(resource_type);
      }
      switch (resource_type) {
        case static_cast<int64_t>(MyTargetResourceType::kTargetResource0):
          return ResourceHazardType::kShareable;
        default:
          return ResourceHazardType::kUnshareable;
      }
    }

    int64_t GetNumTargetDefinedResources() const override {
      return ResourceTypeToIndex(MyTargetResourceType::kTargetResourceTypeEnd) -
             ResourceTypeToIndex(ResourceType::kTargetDefinedResourceTypeBegin);
    }

    int64_t GetNumAvailableResources(int64_t resource_type) const override {
      CHECK_LT(
          resource_type,
          ResourceTypeToIndex(MyTargetResourceType::kTargetResourceTypeEnd));
      if (resource_type < GetTargetDefinedResourceTypeBegin()) {
        return AsyncTracker::GetNumAvailableResources(resource_type);
      }
      switch (resource_type) {
        case (static_cast<int64_t>(MyTargetResourceType::kTargetResource0)):
          return target_resource0_limit_;
        default:
          return 1;
      }
    }

   private:
    const int64_t target_resource0_limit_;
  };
  // Create an AsyncTrackerForMyTarget object with an overlap limit of 5 for
  // target-defined resource "kTargetResource0"
  const int64_t target_resource0_overlap_limit = 5;
  AsyncTrackerForMyTarget async_tracker_for_my_target(
      SchedulerConfig(), target_resource0_overlap_limit);
  // Check the number of target-defined resources
  CHECK_EQ(async_tracker_for_my_target.GetNumTargetDefinedResources(), 1);
  // Get the index of the target-defined resource
  const int64_t target_resource0_index =
      AsyncTracker::GetTargetDefinedResourceTypeBegin();
  // Check the name of the target-defined resource
  CHECK_EQ(async_tracker_for_my_target.GetResourceName(target_resource0_index),
           "kTargetResource0");
  // Check the hazard type of the target-defined resource
  CHECK_EQ(
      static_cast<int64_t>(async_tracker_for_my_target.GetResourceHazardType(
          target_resource0_index)),
      static_cast<int64_t>(ResourceHazardType::kShareable));
  // Check the number of available resources (overlap limit) for the
  // target-defined resource
  CHECK_EQ(async_tracker_for_my_target.GetNumAvailableResources(
               target_resource0_index),
           target_resource0_overlap_limit);
}

TEST_F(LatencyHidingSchedulerTest, AddDeleteOccupierForSharedResource) {
  std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>> occupiers;
  std::function<bool(std::vector<double>)> check_eq = [&occupiers](
                                                          std::vector<double>
                                                              times) {
    if (times.size() != occupiers.size()) {
      return false;
    }
    int64_t i = 0;
    for (auto it = occupiers.begin(); it != occupiers.end(); ++it) {
      if (std::abs(times[i] - it->second) > 0.0001) {
        VLOG(1)
            << "PFT in occupier list does not match the given value (at index "
            << i << "): " << it->second << " vs " << times[i];
        return false;
      }
      i++;
    }
    return true;
  };

  //============================== Additions Only ==============================
  HloEdge edge1(3, nullptr);
  HloEdge edge2(3, nullptr);
  HloEdge edge3(1, nullptr);

  DefaultSchedulerCore::AddOccupierToResource(0, edge1, occupiers);
  CHECK(check_eq({3}));
  DefaultSchedulerCore::AddOccupierToResource(1, edge2, occupiers);
  CHECK(check_eq({5, 6}));
  DefaultSchedulerCore::AddOccupierToResource(1, edge3, occupiers);
  CHECK(check_eq({4, 6, 7}));

  occupiers.clear();
  edge1.SetOriginalLatency(1);
  edge2.SetOriginalLatency(2);
  edge3.SetOriginalLatency(3);

  DefaultSchedulerCore::AddOccupierToResource(0, edge1, occupiers);
  CHECK(check_eq({1}));
  DefaultSchedulerCore::AddOccupierToResource(0, edge2, occupiers);
  CHECK(check_eq({2, 3}));
  DefaultSchedulerCore::AddOccupierToResource(0, edge3, occupiers);
  CHECK(check_eq({3, 5, 6}));

  occupiers.clear();
  DefaultSchedulerCore::AddOccupierToResource(0, edge1, occupiers);
  CHECK(check_eq({1}));
  DefaultSchedulerCore::AddOccupierToResource(0, edge3, occupiers);
  CHECK(check_eq({2, 4}));
  DefaultSchedulerCore::AddOccupierToResource(0, edge2, occupiers);
  CHECK(check_eq({3, 5, 6}));

  occupiers.clear();
  DefaultSchedulerCore::AddOccupierToResource(0, edge2, occupiers);
  CHECK(check_eq({2}));
  DefaultSchedulerCore::AddOccupierToResource(0, edge1, occupiers);
  CHECK(check_eq({2, 3}));
  DefaultSchedulerCore::AddOccupierToResource(0, edge3, occupiers);
  CHECK(check_eq({3, 5, 6}));

  occupiers.clear();
  DefaultSchedulerCore::AddOccupierToResource(0, edge2, occupiers);
  CHECK(check_eq({2}));
  DefaultSchedulerCore::AddOccupierToResource(0, edge3, occupiers);
  CHECK(check_eq({4, 5}));
  DefaultSchedulerCore::AddOccupierToResource(0, edge1, occupiers);
  CHECK(check_eq({3, 5, 6}));

  occupiers.clear();
  DefaultSchedulerCore::AddOccupierToResource(0, edge3, occupiers);
  CHECK(check_eq({3}));
  DefaultSchedulerCore::AddOccupierToResource(0, edge1, occupiers);
  CHECK(check_eq({2, 4}));
  DefaultSchedulerCore::AddOccupierToResource(0, edge2, occupiers);
  CHECK(check_eq({3, 5, 6}));

  occupiers.clear();
  DefaultSchedulerCore::AddOccupierToResource(0, edge3, occupiers);
  CHECK(check_eq({3}));
  DefaultSchedulerCore::AddOccupierToResource(0, edge2, occupiers);
  CHECK(check_eq({4, 5}));
  DefaultSchedulerCore::AddOccupierToResource(0, edge1, occupiers);
  CHECK(check_eq({3, 5, 6}));

  occupiers.clear();
  DefaultSchedulerCore::AddOccupierToResource(0, edge1, occupiers);
  CHECK(check_eq({1}));
  DefaultSchedulerCore::AddOccupierToResource(1, edge2, occupiers);
  CHECK(check_eq({1, 3}));
  DefaultSchedulerCore::AddOccupierToResource(2, edge3, occupiers);
  CHECK(check_eq({1, 4, 6}));

  HloEdge edge0(0.5, nullptr);
  DefaultSchedulerCore::AddOccupierToResource(2, edge0, occupiers);
  CHECK(check_eq({1, 3.5, 4.5, 6.5}));

  //========================== Additions & Deletions ===========================
  occupiers.clear();
  edge1.SetOriginalLatency(1);
  edge2.SetOriginalLatency(2);
  edge3.SetOriginalLatency(3);

  DefaultSchedulerCore::AddOccupierToResource(0, edge1, occupiers);
  DefaultSchedulerCore::AddOccupierToResource(0, edge2, occupiers);
  DefaultSchedulerCore::AddOccupierToResource(0, edge3, occupiers);
  CHECK(check_eq({3, 5, 6}));
  auto res =
      DefaultSchedulerCore::DeleteOccupierFromResource(0, edge0, occupiers);
  CHECK(!res);

  DefaultSchedulerCore::DeleteOccupierFromResource(0, edge1, occupiers);
  CHECK(check_eq({4, 5}));

  DefaultSchedulerCore::AddOccupierToResource(0, edge1, occupiers);
  CHECK(check_eq({3, 5, 6}));
  DefaultSchedulerCore::DeleteOccupierFromResource(0, edge2, occupiers);
  CHECK(check_eq({2, 4}));

  DefaultSchedulerCore::AddOccupierToResource(0, edge2, occupiers);
  CHECK(check_eq({3, 5, 6}));
  DefaultSchedulerCore::DeleteOccupierFromResource(0, edge3, occupiers);
  CHECK(check_eq({2, 3}));

  DefaultSchedulerCore::AddOccupierToResource(0, edge3, occupiers);
  CHECK(check_eq({3, 5, 6}));

  // Deletions at irregular current times
  DefaultSchedulerCore::DeleteOccupierFromResource(1, edge1, occupiers);
  CHECK(check_eq({4.3333333, 5.3333333}));

  occupiers.clear();
  DefaultSchedulerCore::AddOccupierToResource(0, edge1, occupiers);
  DefaultSchedulerCore::AddOccupierToResource(0, edge2, occupiers);
  DefaultSchedulerCore::AddOccupierToResource(0, edge3, occupiers);
  DefaultSchedulerCore::DeleteOccupierFromResource(4, edge1, occupiers);
  CHECK(check_eq({5, 6}));

  occupiers.clear();
  DefaultSchedulerCore::AddOccupierToResource(0, edge1, occupiers);
  DefaultSchedulerCore::AddOccupierToResource(0, edge2, occupiers);
  DefaultSchedulerCore::AddOccupierToResource(0, edge3, occupiers);
  DefaultSchedulerCore::DeleteOccupierFromResource(4, edge2, occupiers);
  CHECK(check_eq({3, 5.5}));

  occupiers.clear();
  DefaultSchedulerCore::AddOccupierToResource(0, edge1, occupiers);
  DefaultSchedulerCore::AddOccupierToResource(0, edge2, occupiers);
  DefaultSchedulerCore::AddOccupierToResource(0, edge3, occupiers);
  DefaultSchedulerCore::DeleteOccupierFromResource(4, edge3, occupiers);
  CHECK(check_eq({3, 4.5}));
}

TEST_F(LatencyHidingSchedulerTest, DepthPressureReduction) {
  absl::string_view hlo_string = R"(
    HloModule serial_collective_permute_test, is_scheduled=true
    ENTRY after_optimizations_test {
    %parameter.1 = bf16[8]{0} parameter(0)
    %parameter.2 = bf16[8]{0} parameter(1)
    %parameter.3 = bf16[8]{0} parameter(2)
    %parameter.4 = bf16[8]{0} parameter(3)
    %collective-permute.2 = bf16[8]{0} collective-permute(parameter.1), source_target_pairs={{0,1},{1,2},{2,3}}
    %a = bf16[8]{0} add(collective-permute.2, parameter.2)
    %b = bf16[8]{0} add(a, parameter.3)
    %c = bf16[8]{0} add(b, parameter.4)
    %d = bf16[8]{0} add(c, parameter.4)
    %c1 = bf16[8]{0} copy(d)
    %e = bf16[8]{0} add(d, parameter.3)
    %c0 = bf16[8]{0} copy(e)
    %f = bf16[8]{0} add(e, parameter.2)
    %h = bf16[8]{0} add(c0, b)
    %g = bf16[8]{0} add(c1, c)
    %i = bf16[8]{0} add(f, a)
    ROOT %t = (bf16[8]{0}, bf16[8]{0}, bf16[8]{0}, bf16[8]{0}) tuple(f, g, h, i)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  auto sched_config = GetDefaultSchedConfig();
  sched_config.memory_limit = 0;
  sched_config.depth_based_memory_pressure_reduction = true;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  const HloInstruction* f = FindInstruction(hlo_module.get(), "f");
  const HloInstruction* g = FindInstruction(hlo_module.get(), "g");
  EXPECT_LT(PositionInVector(new_instruction_sequence, g),
            PositionInVector(new_instruction_sequence, f));
}

TEST_F(LatencyHidingSchedulerTest, RerunWithSmallerMemoryLimit) {
  absl::string_view hlo_string = R"(
    HloModule rerun_scheduler_test, is_scheduled=true
    ENTRY main {
     p0 = bf16[8]{0} parameter(0)
     c = bf16[] constant(0)
     b = bf16[43]{0} broadcast(c), dimensions={}
     s = bf16[1]{0} slice(b), slice={[0:1]}
     cp = bf16[8]{0} collective-permute(p0), source_target_pairs={{0,1},{1,2},{2,3}}
    ROOT tuple = (bf16[8]{0}, bf16[1]{0}) tuple(cp, s)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  auto sched_config = GetDefaultSchedConfig();
  sched_config.memory_limit = 110;
  sched_config.rerun = 1;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config));
  // LatencyHidingScheduler runs an additional "rerun" iteration because the
  // peak memory usage after the first run was 136 bytes (> 110 bytes), so it
  // sets the new limit to 99 and obtains a peak memory usage of 88 bytes at
  // the end of the rerun. In the first run, collective-permute overlaps the
  // slice op, whereas in the rerun, it does not overlap anything.
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
  const HloInstruction* s = FindInstruction(hlo_module.get(), "s");
  const HloInstruction* cps =
      FindInstruction(hlo_module.get(), "collective-permute-start");
  EXPECT_LT(PositionInVector(new_instruction_sequence, s),
            PositionInVector(new_instruction_sequence, cps));
}

TEST_F(LatencyHidingSchedulerTest, MultipleAsyncDoneOperationsDoNotCreateLoop) {
  absl::string_view hlo_string = R"(
HloModule multiple_async_done_scheduler_test, is_scheduled=true

called_computation {
  ROOT %param = s32[<=4096]{0:T(8)M(1024)} parameter(0)
}

ENTRY main {
  %while_body_forward_pass_input_tuple = (s32[<=4096]{0:T(8)M(1024)}, s32[<=4096]{0:T(8)M(1024)}, s32[<=4096]{0:T(8)M(1024)}) parameter(0), backend_config={"flag_configs":[],"scoped_memory_configs":[],"compute_type":"COMPUTE_TYPE_SCALAR"}

  %get-tuple-element.0 = s32[<=4096]{0:T(8)M(1024)} get-tuple-element(
      (s32[<=4096]{0:T(8)M(1024)}, s32[<=4096]{0:T(8)M(1024)}, s32[<=4096]{0:T(8)M(1024)}) %while_body_forward_pass_input_tuple),
      index=0, backend_config={"flag_configs":[],"scoped_memory_configs":[],"compute_type":"COMPUTE_TYPE_SCALAR"}

  %get-tuple-element.1 = s32[<=4096]{0:T(8)M(1024)} get-tuple-element(
      (s32[<=4096]{0:T(8)M(1024)}, s32[<=4096]{0:T(8)M(1024)}, s32[<=4096]{0:T(8)M(1024)}) %while_body_forward_pass_input_tuple),
      index=1, backend_config={"flag_configs":[],"scoped_memory_configs":[],"compute_type":"COMPUTE_TYPE_SCALAR"}

  %call-start.1 = ((s32[<=4096]{0:T(8)M(1024)}), s32[<=4096]{0:T(8)M(1024)}, u32[]{:T(8)S(8)})
    call-start(s32[<=4096]{0:T(8)M(1024)} %get-tuple-element.1),
      async_execution_thread="sparsecore", to_apply=%called_computation

  %call-done.1 = s32[<=4096]{0:T(8)M(1024)}
    call-done(((s32[<=4096]{0:T(8)M(1024)}), s32[<=4096]{0:T(8)M(1024)}, u32[]{:T(8)S(8)}) %call-start.1)

  %call-start.2 = ((s32[<=4096]{0:T(8)M(1024)}), s32[<=4096]{0:T(8)M(1024)}, u32[]{:T(8)S(8)})
    call-start(s32[<=4096]{0:T(8)M(1024)} %call-done.1),
      async_execution_thread="sparsecore", to_apply=%called_computation

  %call-done.2 = s32[<=4096]{0:T(8)M(1024)}
    call-done(((s32[<=4096]{0:T(8)M(1024)}), s32[<=4096]{0:T(8)M(1024)}, u32[]{:T(8)S(8)}) %call-start.2)

  %call-start.3 = ((s32[<=4096]{0:T(8)M(1024)}), s32[<=4096]{0:T(8)M(1024)}, u32[]{:T(8)S(8)})
    call-start(s32[<=4096]{0:T(8)M(1024)} %get-tuple-element.0),
      async_execution_thread="sparsecore", to_apply=%called_computation

  %call-done.3 = s32[<=4096]{0:T(8)M(1024)}
    call-done(((s32[<=4096]{0:T(8)M(1024)}), s32[<=4096]{0:T(8)M(1024)}, u32[]{:T(8)S(8)}) %call-start.3)

  ROOT %tuple.6 = (s32[<=4096]{0:T(8)M(1024)}, s32[<=4096]{0:T(8)M(1024)})
    tuple(s32[<=4096]{0:T(8)M(1024)} %call-done.2, s32[<=4096]{0:T(8)M(1024)} %call-done.3),
      backend_config={"flag_configs":[],"scoped_memory_configs":[],"compute_type":"COMPUTE_TYPE_SCALAR"}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  auto sched_config = GetDefaultSchedConfig();
  // The double indirection of the buffer aliasing in the module above should
  // not create a failure of scheduling by the async done checks.
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config));
}

TEST_F(LatencyHidingSchedulerTest, CopyScheduling) {
  absl::string_view hlo_string = R"(
HloModule EinsumTest, is_scheduled=true
ENTRY AddR2 {
  y_host = bf16[12800,12800]{1,0:T(8,128)(2,1)} parameter(1)
  z = bf16[12800,12800]{1,0:T(8,128)(2,1)} parameter(2)
  x = bf16[12800,12800]{1,0:T(8,128)(2,1)} parameter(0)
  convolution = bf16[12800,12800]{1,0:T(8,128)(2,1)} convolution(x, z), dim_labels=bf_io->bf
  copy-start = (bf16[12800,12800]{1,0:T(8,128)(2,1)}, bf16[12800,12800]{1,0:T(8,128)(2,1)}, u32[]{:S(2)}) copy-start(y_host)
  copy-done = bf16[12800,12800]{1,0:T(8,128)(2,1)} copy-done(copy-start)
  ROOT convolution.1 = bf16[12800,12800]{1,0:T(8,128)(2,1)} convolution(convolution, copy-done), dim_labels=bf_io->bf
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  auto sched_config = GetDefaultSchedConfig();
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config));
  const HloInstruction* conv = FindInstruction(hlo_module.get(), "convolution");
  const HloInstruction* cps = FindInstruction(hlo_module.get(), "copy-start");
  const HloInstruction* cpd = FindInstruction(hlo_module.get(), "copy-done");
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  EXPECT_LT(PositionInVector(new_instruction_sequence, cps),
            PositionInVector(new_instruction_sequence, conv));
  EXPECT_LT(PositionInVector(new_instruction_sequence, conv),
            PositionInVector(new_instruction_sequence, cpd));
  XLA_VLOG_LINES(1, hlo_module->ToString());
}

TEST_F(LatencyHidingSchedulerTest, MaxCopyScheduling) {
  absl::string_view hlo_string = R"(
HloModule EinsumTest, is_scheduled=true
ENTRY AddR2 {
  y_host = bf16[12800,12800]{1,0:T(8,128)(2,1)} parameter(1)
  q_host = bf16[12800,12800]{1,0:T(8,128)(2,1)} parameter(3)
  z = bf16[12800,12800]{1,0:T(8,128)(2,1)} parameter(2)
  x = bf16[12800,12800]{1,0:T(8,128)(2,1)} parameter(0)
  convolution = bf16[12800,12800]{1,0:T(8,128)(2,1)} convolution(x, z), dim_labels=bf_io->bf
  copy-start = (bf16[12800,12800]{1,0:T(8,128)(2,1)}, bf16[12800,12800]{1,0:T(8,128)(2,1)}, u32[]{:S(2)}) copy-start(y_host)
  copy-done = bf16[12800,12800]{1,0:T(8,128)(2,1)} copy-done(copy-start)
  copy-start2 = (bf16[12800,12800]{1,0:T(8,128)(2,1)}, bf16[12800,12800]{1,0:T(8,128)(2,1)}, u32[]{:S(2)}) copy-start(q_host)
  copy-done2 = bf16[12800,12800]{1,0:T(8,128)(2,1)} copy-done(copy-start2)
  ROOT t = (bf16[12800,12800]{1,0:T(8,128)(2,1)}, bf16[12800,12800]{1,0:T(8,128)(2,1)})  tuple(copy-done2, copy-done)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  auto sched_config = GetDefaultSchedConfig();
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config));
  const HloInstruction* conv = FindInstruction(hlo_module.get(), "convolution");
  const HloInstruction* cps = FindInstruction(hlo_module.get(), "copy-start");
  const HloInstruction* cps2 = FindInstruction(hlo_module.get(), "copy-start2");
  const HloInstruction* cpd2 = FindInstruction(hlo_module.get(), "copy-done2");
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  EXPECT_LT(PositionInVector(new_instruction_sequence, cps2),
            PositionInVector(new_instruction_sequence, conv));
  EXPECT_LT(PositionInVector(new_instruction_sequence, conv),
            PositionInVector(new_instruction_sequence, cpd2));
  EXPECT_LT(PositionInVector(new_instruction_sequence, cps),
            PositionInVector(new_instruction_sequence, cpd2));
  XLA_VLOG_LINES(1, hlo_module->ToString());
}

TEST_F(LatencyHidingSchedulerTest, ScheduleLoopPeeledSendDoneBeforeWhile) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

while_cond {
  param = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, pred[]) parameter(0)
  ROOT gte = pred[] get-tuple-element(param), index=2
}

while_body {
  param = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, pred[]) parameter(0)
  gte0 = bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)} get-tuple-element(param), index=0
  gte1 = bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)} get-tuple-element(param), index=1
  add.0 = bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)} add(gte0, gte1)
  gte2 = pred[] get-tuple-element(param), index=2
  ROOT tuple = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, pred[]) tuple(add.0, gte1, gte2)
}

ENTRY %entry {
  p0 = bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)} parameter(0)
  p1 = bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)} parameter(1)
  after-all = token[] after-all()
  send = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, u32[], token[]) send(p0, after-all), channel_id=1246
  recv = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, u32[], token[]) recv(after-all), channel_id=1247
  recv-done = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, token[]) recv-done(recv), channel_id=1247
  get-tuple-element = bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)} get-tuple-element(recv-done), index=0
  send-done = token[] send-done(send), channel_id=1246, control-predecessors={recv-done}
  p2 = pred[] parameter(2)
  tuple = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, pred[]) tuple(get-tuple-element, p1, p2)
  while = (bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)}, pred[]) while(tuple), condition=while_cond, body=while_body
  ROOT gte0 = bf16[1,1,4096,1344]{2,3,1,0:T(8,128)(2,1)} get-tuple-element(while), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.collective_permute_overlap_limit = 2;
  sched_config.all_gather_overlap_limit = 2;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  // Check that 'send-done' is scheduled before 'while'.
  EXPECT_LT(GetIndex(new_instruction_sequence, "send-done"),
            GetIndex(new_instruction_sequence, "while"));
}

// This test simulates a sample target where all-gathers contain non-extendable
// and selective resources.
TEST_F(LatencyHidingSchedulerTest, AllGatherWithSelectiveOverlap) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY %module {
  %constant.19 = u32[] constant(0)
  %replica_id = u32[]{:T(128)} replica-id()
  %convert = f32[]{:T(128)} convert(u32[]{:T(128)} %replica_id)
  %color_operand.1 = f32[8,256,256]{2,1,0:T(8,128)} broadcast(
    f32[]{:T(128)} %convert), dimensions={}
  %ag-start = (f32[8,256,256], f32[16,256,256]) all-gather-start(
    f32[8,256,256] %color_operand.1), replica_groups={{0,1}}, dimensions={0},
    metadata={op_type="AllGather" op_name="ag0"}
  %ag-done = f32[16,256,256] all-gather-done(
    (f32[8,256,256], f32[16,256,256]) %ag-start),
    metadata={op_type="AllGather" op_name="ag0"}
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[16,64,256]{2,1,0} parameter(1)
  p2 = f32[16,256,256]{2,1,0} parameter(2)
  p3 = f32[16,256,256]{2,1,0} parameter(3)
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c2 = f32[16,256,256]{2,1,0} convolution(p0, p1),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT a2 = f32[16,256,256]{2,1,0} add(%ag-done, c0)
}
)";

  // Extend AsyncTracker for a fake target where all-gather contains
  // non-extendable and selective resources.
  class SelectiveOverlapAsyncTracker : public AsyncTracker {
   public:
    explicit SelectiveOverlapAsyncTracker(const SchedulerConfig& sched_config)
        : AsyncTracker(sched_config) {}

    ResourceHazardType GetResourceHazardType(
        int64_t resource_type) const override {
      if (resource_type == ResourceTypeToIndex(ResourceType::kAllGather)) {
        return ResourceHazardType::kSelective;
      }
      // The first target defined resource is defined as non-extendable.
      if (resource_type == AsyncTracker::GetTargetDefinedResourceTypeBegin()) {
        return ResourceHazardType::kNonextendable;
      }
      return AsyncTracker::GetResourceHazardType(resource_type);
    }

    ResourcesVector GetResourcesFromInstructionImpl(
        const HloInstruction& hlo) const override {
      ResourcesVector result =
          AsyncTracker::GetResourcesFromInstructionImpl(hlo);
      // There is only one target defined resource (which is non-extendable).
      if (hlo.opcode() == HloOpcode::kAllGatherStart) {
        result.push_back({AsyncTracker::GetTargetDefinedResourceTypeBegin(),
                          ResourceUsageType::kResourceRelease});
      } else if (hlo.opcode() == HloOpcode::kAllGatherDone) {
        result.push_back({AsyncTracker::GetTargetDefinedResourceTypeBegin(),
                          ResourceUsageType::kResourceOccupy});
      }
      return result;
    }
    int64_t GetNumTargetDefinedResources() const override { return 1; }
    void SetConcurrentResourceLimits(
        absl::flat_hash_map<int64_t, int64_t>& max_concurrent_resource)
        const override {
      max_concurrent_resource[ResourceTypeToIndex(ResourceType::kAllGather)] =
          1;
      max_concurrent_resource[GetTargetDefinedResourceTypeBegin()] = 1;
    }
    absl::InlinedVector<int64_t, 1> GetReleasedNonextendableResourcesFromVector(
        const ResourcesVector& resources) const override {
      absl::InlinedVector<int64_t, 1> non_extendable_resources;
      for (const ResourcePair& resource : resources) {
        if (GetResourceHazardType(resource.first) ==
            ResourceHazardType::kNonextendable) {
          non_extendable_resources.push_back({resource.first});
        }
      }
      return non_extendable_resources;
    }

    void PostProcessScheduleGraph(
        HloScheduleGraph* schedule_graph,
        const LatencyEstimator* latency_estimator) const override {
      // Mark c2 as not valuable for selective overlap.
      for (const HloInstruction* instr :
           schedule_graph->GetOriginalInstrList()) {
        if (instr->name() == "c2") {
          schedule_graph->GetNode(instr).SetValuableForSelectiveOverlap(false);
        }
      }
    }
  };
  SchedulerConfig sched_config = GetDefaultSchedConfig();
  sched_config.enable_selective_resources = true;
  std::unique_ptr<AsyncTracker> async_tracker =
      std::make_unique<SelectiveOverlapAsyncTracker>(sched_config);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config,
                            std::make_unique<ApproximateLatencyEstimator>(),
                            std::move(async_tracker)));
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  // Without selective async tracker, we would expect all-gather to only overlap
  // with c2 as c2 has a cost of 5000 which fully covers the latency of
  // all-gather. However, as c2 is not valuable for selective overlap, we expect
  // all-gather overlap with c1 and c2 (c2 is effectively ignored from a latency
  // hiding perspective).
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
  int c0_index = GetIndex(new_instruction_sequence, "c0");
  int c1_index = GetIndex(new_instruction_sequence, "c1");
  int c2_index = GetIndex(new_instruction_sequence, "c2");
  int ag_start_index = GetIndex(new_instruction_sequence, "ag-start");
  int ag_done_index = GetIndex(new_instruction_sequence, "ag-done");
  EXPECT_LT(c0_index, ag_start_index);
  EXPECT_LT(ag_start_index, c1_index);
  EXPECT_LT(c1_index, c2_index);
  EXPECT_LT(c2_index, ag_done_index);
}

TEST_F(LatencyHidingSchedulerTest, AnnotationFirstDataIndependentConv) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[128,2048,2048]{2,1,0} parameter(1)
  p2 = f32[512,2048,2048]{2,1,0} parameter(2)
  cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0},{0,3},{3,2}}
  cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s)
  cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0},{0,3},{3,2}}, frontend_attributes={_scheduling_group_id="0"}
  cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s), frontend_attributes={_scheduling_group_id="0"}
  cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp2d), source_target_pairs={{1,0},{0,3},{3,2}}
  cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s)
  slice = f32[16,64,256]{2,1,0} slice(cp1d), slice={[0:16], [0:64], [0:256]}
  c0 = f32[16,256,256]{2,1,0} convolution(p0, slice),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
  c1 = f32[16,256,256]{2,1,0} convolution(p0, slice),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, c1, cp2d, cp3d)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.aggressive_scheduling_policies = true;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config,
                            std::make_unique<TestLatencyEstimator>()));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // Without annotations, cp3 would overlap both c0 and c1. With annotations,
  // cp3 only overlaps with c1, and cp2 overlaps with c0.
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp2s"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "c0"),
            GetIndex(new_instruction_sequence, "cp2d"));
}

TEST_F(LatencyHidingSchedulerTest, AnnotationSecondDataIndependentConv) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[128,2048,2048]{2,1,0} parameter(1)
  p2 = f32[512,2048,2048]{2,1,0} parameter(2)
  cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0},{0,3},{3,2}}
  cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s)
  cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0},{0,3},{3,2}}, frontend_attributes={_scheduling_group_id="0"}
  cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s), frontend_attributes={_scheduling_group_id="0"}
  cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp2d), source_target_pairs={{1,0},{0,3},{3,2}}
  cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s)
  slice = f32[16,64,256]{2,1,0} slice(cp1d), slice={[0:16], [0:64], [0:256]}
  c0 = f32[16,256,256]{2,1,0} convolution(p0, slice),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[16,256,256]{2,1,0} convolution(p0, slice),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, c1, cp2d, cp3d)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.aggressive_scheduling_policies = true;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config,
                            std::make_unique<TestLatencyEstimator>()));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // Without annotations, cp3 would overlap both c0 and c1. With annotations,
  // cp3 only overlaps with c0, and cp2 overlaps with c1.
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp2s"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "c1"),
            GetIndex(new_instruction_sequence, "cp2d"));
}

TEST_F(LatencyHidingSchedulerTest, AnnotationBothDataIndependentConvs) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[128,2048,2048]{2,1,0} parameter(1)
  p2 = f32[512,2048,2048]{2,1,0} parameter(2)
  cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0},{0,3},{3,2}}
  cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s)
  cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0},{0,3},{3,2}}, frontend_attributes={_scheduling_group_id="0"}
  cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s), frontend_attributes={_scheduling_group_id="0"}
  cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp2d), source_target_pairs={{1,0},{0,3},{3,2}}
  cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s)
  slice = f32[16,64,256]{2,1,0} slice(cp1d), slice={[0:16], [0:64], [0:256]}
  c0 = f32[16,256,256]{2,1,0} convolution(p0, slice),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
  c1 = f32[16,256,256]{2,1,0} convolution(p0, slice),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, c1, cp2d, cp3d)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.aggressive_scheduling_policies = true;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config,
                            std::make_unique<TestLatencyEstimator>()));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // Without annotations, cp3 would overlap both c0 and c1. With annotations,
  // cp3 only overlaps with c0, and cp2 overlaps with c1.
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp2s"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp2s"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "c0"),
            GetIndex(new_instruction_sequence, "cp2d"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "c1"),
            GetIndex(new_instruction_sequence, "cp2d"));
}

TEST_F(LatencyHidingSchedulerTest, AnnotationFirstDataDependentConv) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[128,2048,2048]{2,1,0} parameter(1)
  p2 = f32[512,2048,2048]{2,1,0} parameter(2)
  cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0},{0,3},{3,2}}
  cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s)
  cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0},{0,3},{3,2}}, frontend_attributes={_scheduling_group_id="0"}
  cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s), frontend_attributes={_scheduling_group_id="0"}
  cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp2d), source_target_pairs={{1,0},{0,3},{3,2}}
  cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s)
  slice = f32[16,64,256]{2,1,0} slice(cp1d), slice={[0:16], [0:64], [0:256]}
  c0 = f32[16,256,256]{2,1,0} convolution(p0, slice),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
  c1 = f32[1,256,256]{2,1,0} convolution(c0, c0),
    window={size=16 stride=15}, dim_labels=0fb_0io->0fb
  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[1,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, c1, cp2d, cp3d)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.aggressive_scheduling_policies = true;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config,
                            std::make_unique<TestLatencyEstimator>()));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // Without annotations, cp3 would overlap both c0 and c1. With annotations,
  // cp2 overlaps both c0 and cp3 overlaps with c1.
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp2s"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "c0"),
            GetIndex(new_instruction_sequence, "cp2d"));
}

TEST_F(LatencyHidingSchedulerTest, AnnotationSecondDataDependentConv) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[128,2048,2048]{2,1,0} parameter(1)
  p2 = f32[512,2048,2048]{2,1,0} parameter(2)
  cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0},{0,3},{3,2}}
  cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s)
  cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0},{0,3},{3,2}}, frontend_attributes={_scheduling_group_id="0"}
  cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s), frontend_attributes={_scheduling_group_id="0"}
  cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp2d), source_target_pairs={{1,0},{0,3},{3,2}}
  cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s)
  slice = f32[16,64,256]{2,1,0} slice(cp1d), slice={[0:16], [0:64], [0:256]}
  c0 = f32[16,256,256]{2,1,0} convolution(p0, slice),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  c1 = f32[1,256,256]{2,1,0} convolution(c0, c0),
    window={size=16 stride=15}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[1,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, c1, cp2d, cp3d)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.aggressive_scheduling_policies = true;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config,
                            std::make_unique<TestLatencyEstimator>()));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // Without annotations, cp3 would overlap both c0 and c1. With annotations,
  // cp2 overlaps both c1 and cp3 overlaps c0.
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp2s"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "c1"),
            GetIndex(new_instruction_sequence, "cp2d"));
}

TEST_F(LatencyHidingSchedulerTest, AnnotationBothDataDependentConvs) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[128,2048,2048]{2,1,0} parameter(1)
  p2 = f32[512,2048,2048]{2,1,0} parameter(2)
  cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0},{0,3},{3,2}}
  cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s)
  cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0},{0,3},{3,2}}, frontend_attributes={_scheduling_group_id="0"}
  cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s), frontend_attributes={_scheduling_group_id="0"}
  cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp2d), source_target_pairs={{1,0},{0,3},{3,2}}
  cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s)
  slice = f32[16,64,256]{2,1,0} slice(cp1d), slice={[0:16], [0:64], [0:256]}
  c0 = f32[16,256,256]{2,1,0} convolution(p0, slice),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
  c1 = f32[1,256,256]{2,1,0} convolution(c0, c0),
    window={size=16 stride=15}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[1,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, c1, cp2d, cp3d)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.aggressive_scheduling_policies = true;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config,
                            std::make_unique<TestLatencyEstimator>()));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // Without annotations, cp3 would overlap both c0 and c1. With annotations,
  // cp2 overlaps both c0 and c1.
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp2s"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp2s"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "c0"),
            GetIndex(new_instruction_sequence, "cp2d"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "c1"),
            GetIndex(new_instruction_sequence, "cp2d"));
}

TEST_F(LatencyHidingSchedulerTest, AnnotationWithTwoAsyncOps) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[128,2048,2048]{2,1,0} parameter(1)
  p2 = f32[512,2048,2048]{2,1,0} parameter(2)
  cp1s = (f32[512,2048,2048]{2,1,0}, f32[512,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0},{0,3},{3,2}}, frontend_attributes={_scheduling_group_id="0"}
  cp1d = f32[512,2048,2048]{2,1,0} collective-permute-done(cp1s), frontend_attributes={_scheduling_group_id="0"}
  cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0},{0,3},{3,2}}, frontend_attributes={_scheduling_group_id="0"}
  cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s), frontend_attributes={_scheduling_group_id="0"}
  cp3s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp2d), source_target_pairs={{1,0},{0,3},{3,2}}
  cp3d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp3s)
  c0 = f32[16,256,256]{2,1,0} convolution(p0, p0),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[512,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c0, cp1d, cp3d)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.aggressive_scheduling_policies = true;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config,
                            std::make_unique<TestLatencyEstimator>()));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // Without annotations, cp3 would overlap both c0. With annotations, both cp1
  // and cp2 overlap c0.
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp1s"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp2s"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "c0"),
            GetIndex(new_instruction_sequence, "cp1d"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "c0"),
            GetIndex(new_instruction_sequence, "cp2d"));
}

TEST_F(LatencyHidingSchedulerTest, SchedulingAnnotationMakesAnotherGroupReady) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

fused_computation {
  param0 = f32[16,64,256]{2,1,0} parameter(0)
  param1 = f32[16,64,256]{2,1,0} parameter(1)
  ROOT c0 = f32[16,256,256]{2,1,0} convolution(param0, param1), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
}

fused_computation.1 {
  param0.1 = f32[16,256,256]{2,1,0} parameter(0)
  param1.1 = f32[16,256,256]{2,1,0} parameter(1)
  ROOT c1 = f32[1,256,256]{2,1,0} convolution(param0.1, param1.1), window={size=16 stride=15}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
}

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[128,2048,2048]{2,1,0} parameter(1)
  cp0s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0},{0,3},{3,2}}, frontend_attributes={_scheduling_group_id="0"}
  cp0d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp0s), frontend_attributes={_scheduling_group_id="0"}
  cp1s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(cp0d), source_target_pairs={{1,0},{0,3},{3,2}}, frontend_attributes={_scheduling_group_id="1"}
  cp1d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp1s), frontend_attributes={_scheduling_group_id="1"}
  f0 = f32[16,256,256]{2,1,0} fusion(p0, p0), kind=kOutput, calls=fused_computation, frontend_attributes={_scheduling_group_id="0"}
  f1 = f32[1,256,256]{2,1,0} fusion(f0, f0), kind=kOutput, calls=fused_computation.1, frontend_attributes={_scheduling_group_id="1"}
  ROOT tuple = (f32[128,2048,2048]{2,1,0}, f32[1,256,256]{2,1,0}) tuple(cp1d, f1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.aggressive_scheduling_policies = true;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config,
                            std::make_unique<TestLatencyEstimator>()));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // cp0 and cp1 overlap f0 and f1, respectively.
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp0s"),
            GetIndex(new_instruction_sequence, "f0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "f0"),
            GetIndex(new_instruction_sequence, "cp0d"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp1s"),
            GetIndex(new_instruction_sequence, "f1"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "f1"),
            GetIndex(new_instruction_sequence, "cp1d"));
}

TEST_F(LatencyHidingSchedulerTest, AnnotatedRoot) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

fused_computation {
  param0 = f32[16,64,256]{2,1,0} parameter(0)
  param1 = f32[16,64,256]{2,1,0} parameter(1)
  ROOT c0 = f32[16,256,256]{2,1,0} convolution(param0, param1), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
}

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[128,2048,2048]{2,1,0} parameter(1)
  cp0s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0},{0,3},{3,2}}, frontend_attributes={_scheduling_group_id="0"}
  cp0d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp0s), frontend_attributes={_scheduling_group_id="0"}
  ROOT f0 = f32[16,256,256]{2,1,0} fusion(p0, p0), kind=kOutput, calls=fused_computation, frontend_attributes={_scheduling_group_id="0"}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.aggressive_scheduling_policies = true;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config,
                            std::make_unique<TestLatencyEstimator>()));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // cp0 overlaps f0.
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp0s"),
            GetIndex(new_instruction_sequence, "f0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "f0"),
            GetIndex(new_instruction_sequence, "cp0d"));
}

TEST_F(LatencyHidingSchedulerTest, AnnotatedNoOp) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

fused_computation {
  param0 = f32[128,2048]{1,0} parameter(0)
  param1 = f32[8,2048]{1,0} parameter(1)
  constant0 = s32[] constant(0)
  dynamic-update-slice = f32[128,2048]{1,0} dynamic-update-slice(param0, param1, constant0, constant0)
  ROOT tuple = (f32[128,2048]{1,0}, f32[128,2048]{1,0}) tuple(dynamic-update-slice, param0)
}

ENTRY entry {
  p0 = f32[128,2048]{1,0} parameter(0)
  p1 = f32[8,2048]{1,0} parameter(1)
  p2 = f32[128,2048]{1,0} parameter(2)
  cps = (f32[128,2048]{1,0}, f32[128,2048]{1,0}, u32[], u32[]) collective-permute-start(p2), source_target_pairs={{1,0},{0,3},{3,2}}, frontend_attributes={_scheduling_group_id="0"}
  cpd = f32[128,2048]{1,0} collective-permute-done(cps), frontend_attributes={_scheduling_group_id="0"}
  fusion = (f32[128,2048]{1,0}, f32[128,2048]{1,0}) fusion(p0, p1), kind=kLoop, calls=fused_computation, frontend_attributes={_scheduling_group_id="0"}
  gte = f32[128,2048]{1,0} get-tuple-element(fusion), index=0, frontend_attributes={_scheduling_group_id="0"}
  ROOT add = f32[128,2048]{1,0} add(gte, cpd)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config,
                            std::make_unique<TestLatencyEstimator>()));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // cp overlaps fusion and gte
  EXPECT_LT(GetIndex(new_instruction_sequence, "cps"),
            GetIndex(new_instruction_sequence, "fusion"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "gte"),
            GetIndex(new_instruction_sequence, "cpd"));
}

TEST_F(LatencyHidingSchedulerTest, OutOfOrderStartAndDone) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

while_condition {
  tuple = ((f32[16,16], u32[], token[]), f32[16,16], u32[]) parameter(0)
  i = get-tuple-element(tuple), index=2
  n = u32[] constant(2)
  ROOT predicate = pred[] compare(i, n), direction=LT
}

while_body {
  tuple = ((f32[16,16], u32[], token[]), f32[16,16], u32[]) parameter(0)
  gte = get-tuple-element(tuple), index=0
  param = get-tuple-element(tuple), index=1
  i = get-tuple-element(tuple), index=2
  dot = f32[16,16] dot(param, param), lhs_contracting_dims={0}, rhs_contracting_dims={1}
  recv_done = (f32[16], token[]) recv-done(gte), frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
  after_all = token[] after-all()
  recv = (f32[16,16], u32[], token[]) recv(after_all), frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}, control-predecessors={recv_done}
  c1 = u32[] constant(1)
  add = add(i, c1)
  ROOT tuple_ = ((f32[16,16], u32[], token[]), f32[16,16], u32[]) tuple(recv, dot, add)
}

ENTRY main {
  param0 = f32[16,16] parameter(0)
  after_all0 = token[] after-all()
  recv0 = (f32[16,16], u32[], token[]) recv(after_all0), frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
  c0 = u32[] constant(0)
  tuple = ((f32[16,16], u32[], token[]), f32[16,16], u32[]) tuple(recv0, param0, c0)
  while = ((f32[16,16], u32[], token[]), f32[16,16], u32[]) while(tuple), body=while_body, condition=while_condition
  gte0 = (f32[16,16], u32[], token[]) get-tuple-element(while), index=0
  ROOT recv_done0 = (f32[16], token[]) recv-done(gte0), frontend_attributes={_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3}}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.schedule_send_recvs = true;
  sched_config.send_recv_host_overlap_limit = 2;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config,
                            std::make_unique<TestLatencyEstimator>()));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
}

TEST_F(LatencyHidingSchedulerTest, SchedulingAnnotationCrossesOverlapLimit) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[16,64,256]{2,1,0} parameter(0)
  p1 = f32[128,2048,2048]{2,1,0} parameter(1)
  cp1s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0},{0,3},{3,2}}, frontend_attributes={_scheduling_group_id="0"}
  cp1d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp1s), frontend_attributes={_scheduling_group_id="0"}
  cp2s = (f32[128,2048,2048]{2,1,0}, f32[128,2048,2048]{2,1,0}, u32[], u32[]) collective-permute-start(p1), source_target_pairs={{1,0},{0,3},{3,2}}
  cp2d = f32[128,2048,2048]{2,1,0} collective-permute-done(cp2s)
  slice = f32[16,64,256]{2,1,0} slice(cp1d), slice={[0:16], [0:64], [0:256]}
  c1 = f32[16,256,256]{2,1,0} convolution(p0, p0),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="0"}
  c2 = f32[16,256,256]{2,1,0} convolution(p0, slice),
    window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT tuple.2 = (f32[16,256,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[128,2048,2048]{2,1,0}) tuple(c1, c2, cp2d)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.collective_permute_overlap_limit = 1;
  TF_EXPECT_OK(RunScheduler(hlo_module.get(), sched_config,
                            std::make_unique<TestLatencyEstimator>()));
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  // With the overlap limit of 1 on collective permutes, we cannot schedule the
  // scheduling group with annotation 0 right after it becomes ready, because
  // cp2's overlap would be open at that moment. cp1 can be scheduled only after
  // cp2 is closed (in the reverse order).
  EXPECT_LT(GetIndex(new_instruction_sequence, "cp1d"),
            GetIndex(new_instruction_sequence, "cp2s"));
}

TEST_F(LatencyHidingSchedulerTest, CrossComputationAnnotation) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    param = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(param), index=2
  }

  while_body {
    param = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) parameter(0)
    gte0 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=0
    gte1 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=1
    gte2 = pred[] get-tuple-element(param), index=2
    cps1 = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, u32[], u32[]) collective-permute-start(gte1), source_target_pairs={{0,1},{1,2},{2,3},{3,0}}, frontend_attributes={_scheduling_group_id="1"}
    cpd1 = f32[16,64,256]{2,1,0} collective-permute-done(cps1), frontend_attributes={_scheduling_group_id="1"}
    c1 = f32[16,256,256]{2,1,0} convolution(gte0, gte0), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    slice = f32[16,64,256]{2,1,0} slice(c1), slice={[0:16], [0:64], [0:256]}
    add = f32[16,64,256]{2,1,0} add(gte0, slice)
    ROOT tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) tuple(add, cpd1, gte2)
  }

  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    p3 = pred[] parameter(3)
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"}
    tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) tuple(p1, p2, p3)
    while = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) while(tuple), condition=while_cond, body=while_body
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="1"}
    gte = f32[16,64,256]{2,1,0} get-tuple-element(while), index=0
    ROOT tuple1 = (f32[16,64,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}) tuple(gte, c0, agd0)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  EXPECT_TRUE(RunScheduler(hlo_module.get(), sched_config,
                           std::make_unique<TestLatencyEstimator>())
                  .ok());
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  EXPECT_LT(GetIndex(new_instruction_sequence, "ags0"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "c0"),
            GetIndex(new_instruction_sequence, "agd0"));
  const HloInstruction* while_inst = FindInstruction(hlo_module.get(), "while");
  std::vector<HloInstruction*> loop_instruction_sequence =
      module_schedule.sequence(while_inst->while_body()).instructions();
  EXPECT_LT(GetIndex(loop_instruction_sequence, "cps1"),
            GetIndex(loop_instruction_sequence, "c1"));
  EXPECT_LT(GetIndex(loop_instruction_sequence, "c1"),
            GetIndex(loop_instruction_sequence, "cpd1"));
}

TEST_F(LatencyHidingSchedulerTest, RaggedAllToAll) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  async_computation {
        input = f32[8,128,1024]{2,1,0:T(8,128)} parameter(0)
        output = f32[8,128,1024]{2,1,0:T(8,128)} parameter(1)
        input_offsets = s32[8]{0} parameter(2)
        send_sizes = s32[8]{0} parameter(3)
        output_offsets = s32[8]{0} parameter(4)
        recv_sizes = s32[8]{0} parameter(5)
        ROOT ra2a = f32[8,128,1024]{2,1,0:T(8,128)} ragged-all-to-all(input, output,input_offsets, send_sizes, output_offsets, recv_sizes), replica_groups={{0,1,2,3,4,5,6,7}}
      }

  ENTRY RA2A {
    p0 = f32[8,128,1024]{2,1,0:T(8,128)} parameter(0)
    c0 = f32[] constant(0)
    output = f32[8,128,1024]{2,1,0:T(8,128)} broadcast(c0), dimensions={}
    p1 = s32[8]{0} parameter(1)
    p2 = s32[8]{0} parameter(2)
    p3 = s32[8]{0} parameter(3)
    p4 = s32[8]{0} parameter(4)
    p5 = f32[1024, 1024]{1,0:T(8,128)} parameter(5)
    input = f32[8,128,1024]{2,1,0:T(8,128)} copy(p0)
    input_offsets = s32[8]{0} copy(p1)
    send_sizes = s32[8]{0} copy(p2)
    output_offsets = s32[8]{0} copy(p3)
    recv_sizes = s32[8]{0} copy(p4)
    ra2a-start = ((f32[8,128,1024]{2,1,0:T(8,128)}, f32[8,128,1024]{2,1,0:T(8,128)}, s32[8]{0}, s32[8]{0}, s32[8]{0}, s32[8]{0}),
                     f32[8,128,1024]{2,1,0:T(8,128)}, u32[]{:S(2)}, u32[]{:S(2)}) async-start(input, output, input_offsets, send_sizes, output_offsets, recv_sizes), calls=async_computation
    ra2a-done = f32[8,128,1024]{2,1,0:T(8,128)} async-done(ra2a-start), calls=async_computation
    d = f32[1024,1024]{1,0:T(8,128)} dot(p5, p5), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[8,128,1024]{2,1,0:T(8,128)}, f32[1024,1024]{1,0:T(8,128)}) tuple(ra2a-done, d)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  EXPECT_TRUE(RunScheduler(hlo_module.get(), sched_config).ok());
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  EXPECT_LT(GetIndex(new_instruction_sequence, "ra2a-start"),
            GetIndex(new_instruction_sequence, "d"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "d"),
            GetIndex(new_instruction_sequence, "ra2a-done"));
}

TEST_F(LatencyHidingSchedulerTest, InvalidAnnotationOverlap) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    param = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(param), index=2
  }

  while_body {
    param = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) parameter(0)
    gte0 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=0
    gte1 = f32[16,64,256]{2,1,0} get-tuple-element(param), index=1
    gte2 = pred[] get-tuple-element(param), index=2
    cps1 = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, u32[], u32[]) collective-permute-start(gte1), source_target_pairs={{0,1},{1,2},{2,3},{3,0}}, frontend_attributes={_scheduling_group_id="1"}
    cpd1 = f32[16,64,256]{2,1,0} collective-permute-done(cps1), frontend_attributes={_scheduling_group_id="1"}
    c1 = f32[16,256,256]{2,1,0} convolution(gte0, gte0), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    slice = f32[16,64,256]{2,1,0} slice(c1), slice={[0:16], [0:64], [0:256]}
    add = f32[16,64,256]{2,1,0} add(gte0, slice)
    ROOT tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) tuple(add, cpd1, gte2)
  }

  ENTRY entry {
    p0 = f32[256,1024]{1,0} parameter(0)
    p1 = f32[16,64,256]{2,1,0} parameter(1)
    p2 = f32[16,64,256]{2,1,0} parameter(2)
    p3 = pred[] parameter(3)
    c0 = f32[16,256,256]{2,1,0} convolution(p1, p2), window={size=16 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb, frontend_attributes={_scheduling_group_id="1"}
    ags0 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"}
    ags1 = (f32[256,1024]{1,0}, f32[1024,1024]{1,0}) all-gather-start(p0), replica_groups={{0,1,2,3}}, dimensions={0}, frontend_attributes={_scheduling_group_id="1"}
    tuple = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) tuple(p1, p2, p3)
    while = (f32[16,64,256]{2,1,0}, f32[16,64,256]{2,1,0}, pred[]) while(tuple), condition=while_cond, body=while_body
    agd1 = f32[1024,1024]{1,0} all-gather-done(ags1), frontend_attributes={_scheduling_group_id="1"}
    agd0 = f32[1024,1024]{1,0} all-gather-done(ags0), frontend_attributes={_scheduling_group_id="1"}
    gte = f32[16,64,256]{2,1,0} get-tuple-element(while), index=0
    ROOT tuple1 = (f32[16,64,256]{2,1,0}, f32[16,256,256]{2,1,0}, f32[1024,1024]{1,0}, f32[1024,1024]{1,0}) tuple(gte, c0, agd0, agd1)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  auto sched_config = GetDefaultSchedConfig();
  sched_config.all_gather_overlap_limit = 1;
  EXPECT_TRUE(RunScheduler(hlo_module.get(), sched_config,
                           std::make_unique<TestLatencyEstimator>())
                  .ok());
  EXPECT_TRUE(hlo_module->has_entry_computation());

  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(hlo_module->entry_computation()).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }

  EXPECT_LT(GetIndex(new_instruction_sequence, "ags0"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "c0"),
            GetIndex(new_instruction_sequence, "agd0"));
  EXPECT_TRUE((GetIndex(new_instruction_sequence, "ags0") >
               GetIndex(new_instruction_sequence, "agd1")) ||
              (GetIndex(new_instruction_sequence, "ags1") >
               GetIndex(new_instruction_sequence, "agd0")));
}

}  // namespace xla

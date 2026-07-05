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

#include "xla/service/memory_space_assignment/allocation.h"

#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla::memory_space_assignment {
namespace {

class AllocationTest : public HloHardwareIndependentTestBase {
 protected:
  void RunAnalysis(HloModule* module,
                   const std::vector<absl::string_view>& inst_names,
                   std::unique_ptr<HloLiveRange>& live_range,
                   std::unique_ptr<HloAliasAnalysis>& alias_analysis) {
    HloSchedule schedule(module);
    HloInstructionSequence sequence;
    for (auto name : inst_names) {
      sequence.push_back(FindInstruction(module, name));
    }
    schedule.set_sequence(module->entry_computation(), sequence);

    AliasInfo alias_info;
    TF_ASSERT_OK_AND_ASSIGN(alias_analysis,
                            HloAliasAnalysis::Run(module, &alias_info));
    TF_ASSERT_OK_AND_ASSIGN(live_range,
                            HloLiveRange::Run(schedule, *alias_analysis,
                                              module->entry_computation()));
  }
};

TEST_F(AllocationTest, CopyAllocationProcessSimple) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[2,3]{1,0} parameter(0)
  p1 = f32[2,3]{1,0} parameter(1)
  p1_negate = f32[2,3]{1,0} negate(p1)
  add = f32[2,3]{1,0} add(p0, p1_negate)
  ROOT tuple = tuple(add, p0)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  std::unique_ptr<HloLiveRange> hlo_live_range;
  std::unique_ptr<HloAliasAnalysis> alias_analysis;
  RunAnalysis(module.get(), {"p0", "p1", "p1_negate", "add", "tuple"},
              hlo_live_range, alias_analysis);
  // HloComputation* computation = module->entry_computation();
  HloInstruction* add = FindInstruction(module.get(), "add");
  HloInstruction* p1_negate = FindInstruction(module.get(), "p1_negate");

  HeapSimulator::Chunk p1_negate_chunk =
      HeapSimulator::Chunk::FromOffsetSize(0, 24);

  PinnedAllocation p1_negate_pinned(HloPosition{p1_negate, {}},
                                    MemorySpace::kDefault, p1_negate_chunk,
                                    /*start_time=*/0,
                                    /*end_time=*/5);
  CopyAllocation copy_allocation(p1_negate_pinned, MemorySpace::kAlternate,
                                 std::nullopt,
                                 /*copy_start_schedule_after_time=*/2,
                                 /*copy_done_schedule_before_time=*/3,
                                 /*end_time=*/5, std::nullopt,
                                 /*sync_mem_op=*/nullptr);

  // Use the correct instruction and operand numbers for the add instruction
  copy_allocation.AddUse(HloUse{add, 1});  // Use of p1_negate in add
  BitcastSplitFn split_fn = nullptr;
  TF_ASSERT_OK(
      copy_allocation.Process(split_fn, *hlo_live_range, *alias_analysis));

  // Check copy_start and copy_done instructions.
  HloInstruction* copy_start = copy_allocation.copy_start();
  ASSERT_NE(copy_start, nullptr);
  EXPECT_EQ(copy_start->opcode(), HloOpcode::kCopyStart);
  EXPECT_EQ(copy_start->operand(0), p1_negate);

  HloInstruction* copy_done = copy_allocation.copy_done();
  ASSERT_NE(copy_done, nullptr);
  EXPECT_EQ(copy_done->opcode(), HloOpcode::kCopyDone);
  EXPECT_EQ(copy_done->operand(0), copy_start);

  // Check that uses are updated.
  EXPECT_EQ(add->operand(1), copy_done);

  // Check defining position
  EXPECT_EQ(copy_allocation.defining_position().instruction, copy_done);
}

TEST_F(AllocationTest, EvictedSplitShape) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[2,3]{1,0} parameter(0)
  p1 = f32[2,3]{1,0} parameter(1)
  p1_negate = f32[2,3]{1,0:S(1)SC(0:1)} negate(p1)
  add = f32[2,3]{1,0} add(p0, p1_negate)
  ROOT tuple = tuple(add, p0)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  std::unique_ptr<HloLiveRange> hlo_live_range;
  std::unique_ptr<HloAliasAnalysis> alias_analysis;
  RunAnalysis(module.get(), {"p0", "p1", "p1_negate", "add", "tuple"},
              hlo_live_range, alias_analysis);
  // HloComputation* computation = module->entry_computation();
  HloInstruction* add = FindInstruction(module.get(), "add");
  HloInstruction* p1_negate = FindInstruction(module.get(), "p1_negate");

  HeapSimulator::Chunk p1_negate_chunk =
      HeapSimulator::Chunk::FromOffsetSize(0, 24);

  PinnedAllocation p1_negate_pinned(HloPosition{p1_negate, {}},
                                    MemorySpace::kAlternate, p1_negate_chunk,
                                    /*start_time=*/0,
                                    /*end_time=*/5);
  CopyAllocation copy_allocation(p1_negate_pinned, MemorySpace::kDefault,
                                 std::nullopt,
                                 /*copy_start_schedule_after_time=*/2,
                                 /*copy_done_schedule_before_time=*/3,
                                 /*end_time=*/5, std::nullopt,
                                 /*sync_mem_op=*/nullptr);

  // Use the correct instruction and operand numbers for the add instruction
  copy_allocation.AddUse(HloUse{add, 1});  // Use of p1_negate in add
  BitcastSplitFn split_fn = nullptr;
  TF_ASSERT_OK(
      copy_allocation.Process(split_fn, *hlo_live_range, *alias_analysis));

  // Check copy_start and copy_done instructions.
  HloInstruction* copy_start = copy_allocation.copy_start();
  ASSERT_NE(copy_start, nullptr);
  EXPECT_EQ(copy_start->opcode(), HloOpcode::kCopyStart);
  EXPECT_EQ(copy_start->operand(0), p1_negate);

  HloInstruction* copy_done = copy_allocation.copy_done();
  ASSERT_NE(copy_done, nullptr);
  EXPECT_EQ(copy_done->opcode(), HloOpcode::kCopyDone);
  EXPECT_EQ(copy_done->operand(0), copy_start);
  EXPECT_EQ(copy_done->shape().layout().split_configs().size(), 0);

  // Check that uses are updated.
  EXPECT_EQ(add->operand(1), copy_done);

  // Check defining position
  EXPECT_EQ(copy_allocation.defining_position().instruction, copy_done);
}

TEST_F(AllocationTest, CopyAllocationProcessReplaceSyncSlice) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = f32[1,3]{1,0} parameter(0)
  p1 = f32[2,3]{1,0} parameter(1)
  p1_negate = f32[2,3]{1,0} negate(p1)
  slice = f32[1,3]{1,0} slice(p1_negate), slice={[0:1], [0:3]}
  add = f32[1,3]{1,0} add(p0, slice)
  ROOT tuple = tuple(add, p0)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  std::unique_ptr<HloLiveRange> hlo_live_range;
  std::unique_ptr<HloAliasAnalysis> alias_analysis;
  RunAnalysis(module.get(), {"p0", "p1", "p1_negate", "slice", "add", "tuple"},
              hlo_live_range, alias_analysis);
  // HloComputation* computation = module->entry_computation();
  HloInstruction* add = FindInstruction(module.get(), "add");
  HloInstruction* p1_negate = FindInstruction(module.get(), "p1_negate");
  HloInstruction* slice = FindInstruction(module.get(), "slice");

  HeapSimulator::Chunk p1_negate_chunk =
      HeapSimulator::Chunk::FromOffsetSize(0, 24);

  PinnedAllocation p1_negate_pinned(HloPosition{p1_negate, {}},
                                    MemorySpace::kAlternate, p1_negate_chunk,
                                    /*start_time=*/0,
                                    /*end_time=*/5);
  CopyAllocation copy_allocation(p1_negate_pinned, MemorySpace::kAlternate,
                                 std::nullopt,
                                 /*copy_start_schedule_after_time=*/2,
                                 /*copy_done_schedule_before_time=*/3,
                                 /*end_time=*/5, std::nullopt,
                                 /*sync_mem_op=*/slice);

  // Use the correct instruction and operand numbers for the add instruction
  copy_allocation.AddUse(HloUse{add, 1});  // Use of p1_negate in add
  BitcastSplitFn split_fn = nullptr;
  TF_ASSERT_OK(
      copy_allocation.Process(split_fn, *hlo_live_range, *alias_analysis));

  // Check copy_start and copy_done instructions.
  HloInstruction* slice_start = copy_allocation.copy_start();
  ASSERT_NE(slice_start, nullptr);
  EXPECT_EQ(slice_start->opcode(), HloOpcode::kAsyncStart);
  EXPECT_EQ(slice_start->operand(0), p1_negate);

  HloInstruction* slice_done = copy_allocation.copy_done();
  ASSERT_NE(slice_done, nullptr);
  EXPECT_EQ(slice_done->opcode(), HloOpcode::kAsyncDone);
  EXPECT_EQ(slice_done->operand(0), slice_start);

  // Check the shapes.
  EXPECT_EQ(slice_done->shape(), slice->shape());

  // Check that uses are updated.
  EXPECT_EQ(add->operand(1), slice_done);

  // Check defining position
  EXPECT_EQ(copy_allocation.defining_position().instruction, slice_done);
}

TEST_F(AllocationTest, SkipTupleReconstructionForAsyncCollective) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  new_buffer = f32[2,3]{1,0} parameter(0)
  cp-start = (f32[2,3]{1,0}, f32[2,3]{1,0}, u32[], u32[]) collective-permute-start(new_buffer), channel_id=1, source_target_pairs={{0,1}}
  cp-done = f32[2,3]{1,0} collective-permute-done(cp-start)
  ROOT tuple = tuple(cp-done)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  std::unique_ptr<HloLiveRange> hlo_live_range;
  std::unique_ptr<HloAliasAnalysis> alias_analysis;
  RunAnalysis(module.get(), {"new_buffer", "cp-start", "cp-done", "tuple"},
              hlo_live_range, alias_analysis);

  HloInstruction* cp_start = FindInstruction(module.get(), "cp-start");
  HloInstruction* cp_done = FindInstruction(module.get(), "cp-done");
  HloInstruction* new_buffer = FindInstruction(module.get(), "new_buffer");

  HeapSimulator::Chunk chunk = HeapSimulator::Chunk::FromOffsetSize(0, 24);

  PinnedAllocation pinned(HloPosition{new_buffer, {}}, MemorySpace::kAlternate,
                          chunk, 0, 5);
  pinned.AddUse(HloUse{cp_done, 0, {0}});

  BitcastSplitFn split_fn = nullptr;
  TF_ASSERT_OK(pinned.Process(split_fn, *hlo_live_range, *alias_analysis));

  EXPECT_EQ(cp_done->operand(0), cp_start);
}

}  // namespace
}  // namespace xla::memory_space_assignment

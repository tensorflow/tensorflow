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

#include <gmock/gmock.h>
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
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

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
    ASSERT_OK_AND_ASSIGN(alias_analysis,
                         HloAliasAnalysis::Run(module, &alias_info));
    ASSERT_OK_AND_ASSIGN(live_range,
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
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

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
  ASSERT_OK(
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
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

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
  ASSERT_OK(
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
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

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
  ASSERT_OK(
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
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

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
  ASSERT_OK(pinned.Process(split_fn, *hlo_live_range, *alias_analysis));

  EXPECT_EQ(cp_done->operand(0), cp_start);
}

TEST_F(AllocationTest, MirroredAllocationDelegatesChunkToOriginalAllocation) {
  HloComputation::Builder builder("entry");
  HloInstruction* p0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {2, 3}), "p0"));
  PinnedAllocation original_allocation(
      HloPosition{p0, {}}, MemorySpace::kAlternate,
      HeapSimulator::Chunk::FromOffsetSize(-1, 64),
      /*start_time=*/0, /*end_time=*/5);
  MirroredAllocation mirrored_allocation(original_allocation, /*time=*/2);
  original_allocation.set_offset(128);

  const Allocation& alloc = mirrored_allocation;
  ASSERT_TRUE(alloc.maybe_chunk().has_value());
  EXPECT_EQ(alloc.maybe_chunk()->offset, 128);
  EXPECT_EQ(alloc.chunk().offset, 128);
}

TEST_F(AllocationTest, ParentAllocationPropagatesOriginalValue) {
  absl::string_view hlo_string = R"(
HloModule module

while_body {
  param = (f32[2,3]{1,0}, f32[2,3]{1,0}) parameter(0), origin={({"bp0"}, {"bp1"})}
  gte0 = f32[2,3]{1,0} get-tuple-element(param), index=0
  gte1 = f32[2,3]{1,0} get-tuple-element(param), index=1
  add = f32[2,3]{1,0} add(gte0, gte1)
  ROOT root = (f32[2,3]{1,0}, f32[2,3]{1,0}) tuple(add, gte1), origin={({"br0"}, {"br1"})}
}

while_condition {
  param = (f32[2,3]{1,0}, f32[2,3]{1,0}) parameter(0), origin={({"cp0"}, {"cp1"})}
  ROOT cond = pred[] constant(true)
}

ENTRY entry {
  p0 = f32[2,3]{1,0} parameter(0), origin={{"p0"}}
  p1 = f32[2,3]{1,0} parameter(1), origin={{"p1"}}
  p2 = f32[2,3]{1,0} parameter(2), origin={{"p2"}}
  init = (f32[2,3]{1,0}, f32[2,3]{1,0}) tuple(p0, p1), origin={({"p0"}, {"p1"})}
  while = (f32[2,3]{1,0}, f32[2,3]{1,0}) while(init), condition=while_condition, body=while_body, origin={({"w0"}, {"w1"}),["while#$"]}
  entry_gte0 = f32[2,3]{1,0} get-tuple-element(while), index=0
  ROOT out = (f32[2,3]{1,0}) tuple(entry_gte0)
}
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  std::unique_ptr<HloLiveRange> hlo_live_range;
  std::unique_ptr<HloAliasAnalysis> alias_analysis;
  RunAnalysis(module.get(),
              {"p0", "p1", "p2", "init", "while", "entry_gte0", "out"},
              hlo_live_range, alias_analysis);

  HloInstruction* p2 = FindInstruction(module.get(), "p2");
  HloInstruction* while_instr = FindInstruction(module.get(), "while");
  HloInstruction* entry_gte0 = FindInstruction(module.get(), "entry_gte0");

  HeapSimulator::Chunk chunk = HeapSimulator::Chunk::FromOffsetSize(0, 24);
  PinnedAllocation p2_pinned(HloPosition{p2, {}}, MemorySpace::kDefault, chunk,
                             /*start_time=*/0, /*end_time=*/10);

  ParentAllocation parent_alloc(
      p2_pinned, while_instr,
      HloPosition{while_instr->while_body()->parameter_instruction(0), {}},
      /*time=*/5);

  BitcastSplitFn split_fn = nullptr;
  ASSERT_OK(parent_alloc.Process(split_fn, *hlo_live_range, *alias_analysis));
  ASSERT_OK(parent_alloc.PostProcess());

  ASSERT_NE(while_instr->while_init()->original_value(), nullptr);
  EXPECT_EQ(while_instr->while_init()->original_value()->ToString(),
            R"(({"p0"}, {"p1"}, {"p2"}))");

  ASSERT_NE(while_instr->original_value(), nullptr);
  EXPECT_EQ(while_instr->original_value()->ToString(),
            R"(({"w0"}, {"w1"}, {"p2"}),["while#$"])");

  ASSERT_NE(
      while_instr->while_body()->parameter_instruction(0)->original_value(),
      nullptr);
  EXPECT_EQ(while_instr->while_body()
                ->parameter_instruction(0)
                ->original_value()
                ->ToString(),
            R"(({"bp0"}, {"bp1"}, {"p2"}))");

  ASSERT_NE(while_instr->while_condition()
                ->parameter_instruction(0)
                ->original_value(),
            nullptr);
  EXPECT_EQ(while_instr->while_condition()
                ->parameter_instruction(0)
                ->original_value()
                ->ToString(),
            R"(({"cp0"}, {"cp1"}, {"p2"}))");

  ASSERT_NE(while_instr->while_body()->root_instruction()->original_value(),
            nullptr);
  EXPECT_EQ(while_instr->while_body()
                ->root_instruction()
                ->original_value()
                ->ToString(),
            R"(({"br0"}, {"br1"}, {"p2"}))");

  const HloInstruction* tuple_with_old_shape = entry_gte0->operand(0);
  ASSERT_NE(tuple_with_old_shape->original_value(), nullptr);
  EXPECT_EQ(tuple_with_old_shape->original_value()->ToString(),
            R"(({"w0"}, {"w1"}),["while#$"])");
}

}  // namespace
}  // namespace xla::memory_space_assignment

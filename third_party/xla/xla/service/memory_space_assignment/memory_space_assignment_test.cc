/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/memory_space_assignment/memory_space_assignment.h"

#include <stdbool.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/service/cost_modelling/op_cost.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/algorithm.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/allocation_value.h"
#include "xla/service/memory_space_assignment/buffer_interval_comparator.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/service/memory_space_assignment/memory_space_assignment_test_base.h"
#include "xla/service/memory_space_assignment/options.h"
#include "xla/service/memory_space_assignment/prefetch_interval_picker.h"
#include "xla/service/memory_space_assignment/repacking.h"
#include "xla/service/memory_space_assignment/slice.h"
#include "xla/service/memory_space_assignment/testing_utils.h"
#include "xla/service/memory_space_assignment/utils.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/statusor.h"

namespace xla {
namespace memory_space_assignment {
namespace {

namespace op = xla::testing::opcode_matchers;
using Chunk = HeapSimulator::Chunk;
using ::testing::_;
using ::testing::Return;
using ::testing::UnorderedElementsAre;

constexpr float kBytesPerSecond = 100;

const auto& ShapeSize = HloCostAnalysis::DefaultShapeSize;

using MemorySpaceAssignmentTest = MemorySpaceAssignmentTestBase;

TEST_F(MemorySpaceAssignmentTest, ParameterOnly) {
  // A module consisting of a single parameter. Inputs/outputs are currently
  // excluded from memory space assignment.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
}

TEST_F(MemorySpaceAssignmentTest, Simple) {
  // A simple module with a few simple instructions. Expect this to be
  // transformed with CopyStart and CopyDone instructions inserted after inputs
  // and before outputs.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, p1));
  HloInstruction* sub = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kSubtract, p0, p1));
  HloInstruction* mul = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, add, sub));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, add, sub, mul});
  TF_CHECK_OK(module->set_schedule(schedule));

  auto preset_assignments = AssignMemorySpace(module.get());

  // Inputs and outputs are currently placed in the default memory. Everything
  // else should be in the alternate memory.
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0},
      /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1,
      /*element_size_in_bits=*/0, kAlternateMemorySpace);
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  EXPECT_THAT(mul, op::ShapeWithLayout(shape));
  EXPECT_THAT(add, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(sub, op::ShapeWithLayout(shape_in_alternate_mem));

  // Make sure the preset assignments is sane.
  EXPECT_EQ(preset_assignments->chunks().size(), 3);
  EXPECT_EQ(preset_assignments->assignment_informations().size(), 1);
  // Ensure the offset assigned to add and sub are different.
  EXPECT_NE(preset_assignments->chunks()[0].second.offset,
            preset_assignments->chunks()[1].second.offset);
}

TEST_F(MemorySpaceAssignmentTest, BasicSplit) {
  constexpr int64_t kSplitDimension = 1;
  constexpr int64_t kSplitIndex = 256;
  SplitConfig split_config(/*dimension=*/kSplitDimension, {kSplitIndex});
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {256, 512});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, p1));
  HloInstruction* sub = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kSubtract, p0, p1));
  HloInstruction* mul = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, add, sub));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, add, sub, mul});
  CHECK_OK(module->set_schedule(schedule));

  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 256 * 512 * 16;
  options.init_split_tree_fn =
      [&options](const HloInstruction* instruction,
                 absl::flat_hash_map<const HloInstruction*, ShapeTree<int64_t>>*
                     split_map) {
        if (split_map != nullptr) {
          auto result = split_map->find(instruction);
          if (result != split_map->end()) {
            return result->second;
          }
        }
        return ShapeTree<int64_t>(instruction->shape(),
                                  options.any_split_dimension);
      };

  options.determine_split_dimension_fn =
      [&split_config](
          const HloValue& hlo_value,
          absl::flat_hash_map<const HloInstruction*, ShapeTree<int64_t>>*
              split_map) -> std::optional<SplitConfig> {
    if (hlo_value.instruction()->opcode() == HloOpcode::kAdd ||
        hlo_value.instruction()->opcode() == HloOpcode::kSubtract) {
      return split_config;
    }
    return std::nullopt;
  };

  auto preset_assignments = AssignMemorySpace(module.get(), options);

  // Inputs and outputs are currently placed in the default memory. Everything
  // else should be in the alternate memory.
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {256, 512},
      /*minor_to_major=*/{1, 0},
      /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1,
      /*element_size_in_bits=*/0, kAlternateMemorySpace);
  shape_in_alternate_mem.mutable_layout()->add_split_configs(split_config);
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  EXPECT_THAT(mul, op::ShapeWithLayout(shape));
  EXPECT_THAT(add, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(sub, op::ShapeWithLayout(shape_in_alternate_mem));

  // Make sure the preset assignments is sane.
  EXPECT_EQ(preset_assignments->chunks().size(), 3);
  EXPECT_EQ(preset_assignments->assignment_informations().size(), 1);
  // Ensure the offset assigned to add and sub are different.
  EXPECT_NE(preset_assignments->chunks()[0].second.offset,
            preset_assignments->chunks()[1].second.offset);
}

TEST_F(MemorySpaceAssignmentTest, NegateChain) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  EXPECT_THAT(add, op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(1))));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0},
      /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1,
      /*element_size_in_bits=*/0, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
  // Ensure the CopyStart/CopyDone schedules.
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  EXPECT_THAT(sequence.instructions()[0], op::Parameter(0));
  EXPECT_THAT(sequence.instructions()[1], op::Parameter(1));
  EXPECT_THAT(sequence.instructions()[2], op::CopyStart());
  EXPECT_THAT(sequence.instructions()[10], op::CopyDone());
}

TEST_F(MemorySpaceAssignmentTest, PinnedDefaultMemorySpace) {
  absl::string_view hlo_string = R"(
  HloModule NegateChain, is_scheduled=true, entry_computation_layout={(f32[2,3]{1,0}, f32[2,3]{1,0:S(2)})->f32[2,3]{1,0}}

  ENTRY %NegateChain (p0: f32[2,3], p1: f32[2,3]) -> f32[2,3] {
    %p0 = f32[2,3]{1,0} parameter(0)
    %p1 = f32[2,3]{1,0:S(2)} parameter(1)
    %negate = f32[2,3]{1,0:S(2)} negate(f32[2,3]{1,0} %p0)
    %negate.1 = f32[2,3]{1,0:S(2)} negate(f32[2,3]{1,0:S(2)} %negate)
    %negate.2 = f32[2,3]{1,0:S(2)} negate(f32[2,3]{1,0:S(2)} %negate.1)
    %negate.3 = f32[2,3]{1,0} negate(f32[2,3]{1,0:S(2)} %negate.2)
    %negate.4 = f32[2,3]{1,0:S(2)} negate(f32[2,3]{1,0} %negate.3)
    %negate.5 = f32[2,3]{1,0:S(2)} negate(f32[2,3]{1,0:S(2)} %negate.4)
    %negate.6 = f32[2,3]{1,0:S(2)} negate(f32[2,3]{1,0:S(2)} %negate.5)
    ROOT %add = f32[2,3]{1,0} add(f32[2,3]{1,0:S(2)} %negate.6, f32[2,3]{1,0:S(2)} %p1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
  XLA_VLOG_LINES(1, module->ToString());
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* p1 = FindInstruction(module.get(), "p1");
  HloInstruction* negate = FindInstruction(module.get(), "negate");
  HloInstruction* negate_1 = FindInstruction(module.get(), "negate.1");
  HloInstruction* negate_2 = FindInstruction(module.get(), "negate.2");
  HloInstruction* negate_3 = FindInstruction(module.get(), "negate.3");
  HloInstruction* negate_4 = FindInstruction(module.get(), "negate.4");
  HloInstruction* negate_5 = FindInstruction(module.get(), "negate.5");
  HloInstruction* negate_6 = FindInstruction(module.get(), "negate.6");
  HloInstruction* add = FindInstruction(module.get(), "add");
  std::vector<const HloInstruction*> pinned_hbm_instructions = {
      p1, negate, negate_1, negate_2, negate_4, negate_5, negate_6};
  for (const HloInstruction* instruction : pinned_hbm_instructions) {
    EXPECT_EQ(instruction->shape().layout().memory_space(),
              kPinnedDefaultMemorySpace);
  }
  // Check p0 and add are in the default memory space.
  EXPECT_EQ(p0->shape().layout().memory_space(), kDefaultMemorySpace);
  EXPECT_EQ(add->shape().layout().memory_space(), kDefaultMemorySpace);
  // Check negate_3 is in pinned to alternate memory space.
  EXPECT_EQ(negate_3->shape().layout().memory_space(), kAlternateMemorySpace);
  // Check that p1 is only used once at the add instruction. ie, the there is no
  // copy/prefetch.
  CHECK_EQ(p1->users().size(), 1);
  EXPECT_EQ(p1->users()[0], add);
}

// A simple case where the synchronous copy is actually redundant, because its
// operand ends up getting prefetched and the its output is only used once, so
// we remove the sync copy.
TEST_F(MemorySpaceAssignmentTest,
       SyncCopyReplacementRedundantCopyAfterPrefetch) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[2,3]{1,0} parameter(0)
  p1 = f32[2,3]{1,0} parameter(1)
  negate0 = f32[2,3]{1,0} negate(p1)
  negate1 = f32[2,3]{1,0} negate(negate0)
  negate2 = f32[2,3]{1,0} negate(negate1)
  negate3 = f32[2,3]{1,0} negate(negate2)
  negate4 = f32[2,3]{1,0} negate(negate3)
  negate5 = f32[2,3]{1,0} negate(negate4)
  negate6 = f32[2,3]{1,0} negate(negate5)
  negate7 = f32[2,3]{1,0} negate(negate6)
  p0_copy = f32[2,3]{1,0} copy(p0)
  ROOT add0 = f32[2,3]{1,0} add(p0_copy, negate7)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.enable_sync_copy_replacement = true;
  AssignMemorySpace(module.get(), options);
  HloInstruction* add0 = FindInstruction(module.get(), "add0");
  ASSERT_NE(add0, nullptr);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  ASSERT_NE(p0, nullptr);
  EXPECT_THAT(add0->operand(0),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace, p0));
}

TEST_F(MemorySpaceAssignmentTest, SyncCopyReplacementWithControlPredecessor) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[2,3]{1,0} parameter(0)
  p1 = f32[2,3]{1,0} parameter(1)
  negate0 = f32[2,3]{1,0} negate(p1)
  negate1 = f32[2,3]{1,0} negate(negate0)
  negate2 = f32[2,3]{1,0} negate(negate1)
  negate3 = f32[2,3]{1,0} negate(negate2)
  negate4 = f32[2,3]{1,0} negate(negate3)
  negate5 = f32[2,3]{1,0} negate(negate4)
  negate6 = f32[2,3]{1,0} negate(negate5)
  negate7 = f32[2,3]{1,0} negate(negate6)
  p0_copy = f32[2,3]{1,0} copy(p0)
  negate8 = f32[2,3]{1,0} negate(negate7)
  negate9 = f32[2,3]{1,0} negate(negate8), control-predecessors={p0_copy}
  ROOT add0 = f32[2,3]{1,0} add(p0_copy, negate9)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.enable_sync_copy_replacement = true;
  AssignMemorySpace(module.get(), options);
  HloInstruction* add0 = FindInstruction(module.get(), "add0");
  ASSERT_NE(add0, nullptr);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  ASSERT_NE(p0, nullptr);
  EXPECT_THAT(add0->operand(0),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace, p0));
  HloInstruction* negate9 = FindInstruction(module.get(), "negate9");
  ASSERT_NE(negate9, nullptr);
  const HloInstruction* copy_done = add0->operand(0);
  const HloInstructionSequence& sequence =
      module->schedule().sequence(module->entry_computation());
  auto find_index = [&](const HloInstruction* instruction) {
    return std::distance(sequence.instructions().begin(),
                         std::find(sequence.instructions().begin(),
                                   sequence.instructions().end(), instruction));
  };
  int64_t copy_done_time = find_index(copy_done);
  int64_t negate9_time = find_index(negate9);
  // The negate9 instruction should be scheduled after the copy done, because of
  // the control dependency constraint.
  EXPECT_GT(negate9_time, copy_done_time);
}

// This is a case where we p0_copy uses and and p0 uses after copy(p0) are not
// allowed to use the same async CopyAllocation. While p0 can be prefetched at
// p0_copy, but we may clobber the data if we use the same async copy used for
// prefetching to replace the sync copy p0_copy. The pattern here is that the
// sync copy operand (p0) shows up at more than one position.
TEST_F(MemorySpaceAssignmentTest,
       SyncCopyReplacementWouldNeedMoreThanOneAsyncCopy) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[2,3]{1,0} parameter(0)
  p1 = f32[2,3]{1,0} parameter(1)
  negate0 = f32[2,3]{1,0} negate(p1)
  negate1 = f32[2,3]{1,0} negate(negate0)
  negate2 = f32[2,3]{1,0} negate(negate1)
  negate3 = f32[2,3]{1,0} negate(negate2)
  negate4 = f32[2,3]{1,0} negate(negate3)
  negate5 = f32[2,3]{1,0} negate(negate4)
  negate6 = f32[2,3]{1,0} negate(negate5)
  negate7 = f32[2,3]{1,0} negate(negate6)
  p0_copy0 = f32[2,3]{1,0} copy(p0)
  p0_copy1 = f32[2,3]{1,0} copy(p0)
  negate8 = f32[2,3]{1,0} negate(negate7)
  negate9 = f32[2,3]{1,0} negate(negate8)
  negate10 = f32[2,3]{1,0} negate(negate9)
  negate11 = f32[2,3]{1,0} negate(negate10)
  negate12 = f32[2,3]{1,0} negate(negate11)
  constant.1 = f32[] constant(0)
  broadcast = f32[2,1] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  dynamic-update-slice.0 = f32[2,3] dynamic-update-slice(p0_copy0, broadcast, constant.3, constant.3)
  dynamic-update-slice.1 = f32[2,3] dynamic-update-slice(p0_copy1, broadcast, constant.3, constant.3)
  ROOT tuple0 = tuple(negate12, dynamic-update-slice.0, dynamic-update-slice.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.enable_sync_copy_replacement = true;
  AssignMemorySpace(module.get(), options);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");

  HloInstruction* dynamic_update_slice_0 =
      FindInstruction(module.get(), "dynamic-update-slice.0");
  HloInstruction* dynamic_update_slice_1 =
      FindInstruction(module.get(), "dynamic-update-slice.1");
  const HloInstruction* p0_copy0_replacement =
      dynamic_update_slice_0->operand(0);
  const HloInstruction* p0_copy1_replacement =
      dynamic_update_slice_1->operand(0);
  EXPECT_THAT(p0_copy0_replacement,
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace, p0));
  EXPECT_THAT(p0_copy1_replacement,
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace, p0));
  ASSERT_NE(p0_copy0_replacement, p0_copy1_replacement);
}

// All uses of the sync copy operand that are scheduled before the replaced sync
// copy share the allocation in alternate memory (if any).
TEST_F(MemorySpaceAssignmentTest, SyncCopyReplacementOperandHasMultipleUses) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[2,3]{1,0} parameter(0)
  p1 = f32[2,3]{1,0} parameter(1)
  negate0 = negate(p1)
  negate1 = negate(negate0)
  negate2 = negate(negate1)
  negate3 = negate(negate2)
  negate4 = negate(negate3)
  negate5 = negate(negate4)
  p0_negate0 = negate(p0)
  p0_negate1 = negate(p0)
  negate6 = negate(negate5)
  negate7 = negate(negate6)
  p0_copy = f32[2,3]{1,0} copy(p0)
  add0 = add(p0_copy, p0_negate0)
  ROOT tuple = tuple(negate7, add0, p0_negate1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.enable_sync_copy_replacement = true;
  AssignMemorySpace(module.get(), options);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* add0 = FindInstruction(module.get(), "add0");
  HloInstruction* p0_negate0 = FindInstruction(module.get(), "p0_negate0");
  HloInstruction* p0_negate1 = FindInstruction(module.get(), "p0_negate1");

  EXPECT_THAT(add0->operand(0),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace, p0));
  EXPECT_THAT(p0_negate0->operand(0),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace, p0));
  EXPECT_THAT(p0_negate1->operand(0),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace, p0));
  ASSERT_EQ(p0_negate0->operand(0), p0_negate1->operand(0));
  ASSERT_NE(add0->operand(0), p0_negate0->operand(0));
}

TEST_F(MemorySpaceAssignmentTest, SyncSliceReplacementAfterPrefetch) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[10,2,3]{2,1,0} parameter(0)
    p1 = f32[10,2,3]{2,1,0} parameter(1)
    negate0 = negate(p1)
    negate1 = negate(negate0)
    negate2 = negate(negate1)
    negate3 = negate(negate2)
    negate4 = negate(negate3)
    negate5 = negate(negate4)
    negate6 = negate(negate5)
    negate7 = negate(negate6)
    slice = f32[1,2,3] slice(p0), slice={[0:1], [0:2], [0:3]}
    concat = f32[11,2,3] concatenate(negate7, slice), dimensions={0}
    ROOT root = negate(concat)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 512;
  options.enable_sync_copy_replacement = false;
  options.enable_sync_slice_replacement = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };
  AssignMemorySpace(module.get(), options);
  HloInstruction* concat = FindInstruction(module.get(), "concat");
  ASSERT_NE(concat, nullptr);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  ASSERT_NE(p0, nullptr);
  EXPECT_THAT(concat->operand(1), op::AsyncDone(op::AsyncStart(p0)));
}

TEST_F(MemorySpaceAssignmentTest, SyncDynamicSliceReplacementAfterPrefetch) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[10,2,3]{2,1,0} parameter(0)
    p1 = f32[10,2,3]{2,1,0} parameter(1)
    index = s32[] parameter(2)
    zero = s32[] constant(0)
    negate0 = negate(p1)
    negate1 = negate(negate0)
    negate2 = negate(negate1)
    negate3 = negate(negate2)
    negate4 = negate(negate3)
    negate5 = negate(negate4)
    negate6 = negate(negate5)
    negate7 = negate(negate6)
    dynamic_slice = f32[1,2,3] dynamic-slice(p0, index, zero, zero), dynamic_slice_sizes={1,2,3}
    concat = f32[11,2,3] concatenate(negate7, dynamic_slice), dimensions={0}
    ROOT root = negate(concat)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 512;
  options.enable_sync_copy_replacement = false;
  options.enable_sync_slice_replacement = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };
  AssignMemorySpace(module.get(), options);
  HloInstruction* concat = FindInstruction(module.get(), "concat");
  ASSERT_NE(concat, nullptr);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* index = FindInstruction(module.get(), "index");
  HloInstruction* zero = FindInstruction(module.get(), "zero");
  ASSERT_NE(p0, nullptr);
  EXPECT_THAT(concat->operand(1),
              op::AsyncDone(op::AsyncStart(p0, index, zero, zero)));
}

TEST_F(MemorySpaceAssignmentTest, SyncSliceReplacementIgnoredTrivials) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[10,2,3]{2,1,0} parameter(0)
    p1 = f32[10,2,3]{2,1,0} parameter(1)
    negate0 = negate(p1)
    negate1 = negate(negate0)
    negate2 = negate(negate1)
    negate3 = negate(negate2)
    negate4 = negate(negate3)
    negate5 = negate(negate4)
    negate6 = negate(negate5)
    negate7 = negate(negate6)
    slice = f32[1,2,3] slice(p0), slice={[0:1], [0:2], [0:3]}
    bitcast0 = f32[1,3,2] bitcast(slice)
    bitcast1 = f32[10,3,2] bitcast(negate7)
    concat = f32[11,3,2] concatenate(bitcast1, bitcast0), dimensions={0}
    ROOT root = negate(concat)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 512;
  options.enable_sync_copy_replacement = false;
  options.enable_sync_slice_replacement = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };
  AssignMemorySpace(module.get(), options);
  HloInstruction* concat = FindInstruction(module.get(), "concat");
  ASSERT_NE(concat, nullptr);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  ASSERT_NE(p0, nullptr);
  EXPECT_THAT(concat->operand(1),
              op::Bitcast(op::AsyncDone(op::AsyncStart(p0))));
}

TEST_F(MemorySpaceAssignmentTest, SyncDynamicSliceReplacementIgnoredTrivials) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[10,2,3]{2,1,0} parameter(0)
    p1 = f32[10,2,3]{2,1,0} parameter(1)
    index = s32[] parameter(2)
    zero = s32[] constant(0)
    negate0 = negate(p1)
    negate1 = negate(negate0)
    negate2 = negate(negate1)
    negate3 = negate(negate2)
    negate4 = negate(negate3)
    negate5 = negate(negate4)
    negate6 = negate(negate5)
    negate7 = negate(negate6)
    dynamic_slice = f32[1,2,3] dynamic-slice(p0, index, zero, zero), dynamic_slice_sizes={1,2,3}
    bitcast0 = f32[1,3,2] bitcast(dynamic_slice)
    bitcast1 = f32[10,3,2] bitcast(negate7)
    concat = f32[11,3,2] concatenate(bitcast1, bitcast0), dimensions={0}
    ROOT root = negate(concat)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 512;
  options.enable_sync_copy_replacement = false;
  options.enable_sync_slice_replacement = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };
  AssignMemorySpace(module.get(), options);
  HloInstruction* concat = FindInstruction(module.get(), "concat");
  ASSERT_NE(concat, nullptr);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* index = FindInstruction(module.get(), "index");
  HloInstruction* zero = FindInstruction(module.get(), "zero");
  ASSERT_NE(p0, nullptr);
  EXPECT_THAT(
      concat->operand(1),
      op::Bitcast(op::AsyncDone(op::AsyncStart(p0, index, zero, zero))));
}

TEST_F(MemorySpaceAssignmentTest, SyncSliceReplacementAfterEviction) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[8,4,2]{2,1,0} parameter(0)
    p1 = f32[4,4,2]{2,1,0} parameter(1)
    negate_p0 = negate(p0)
    slice0 = f32[1,4,2] slice(negate_p0), slice={[0:1], [0:4], [0:2]}
    negate0 = negate(p1)
    negate1 = negate(negate0)
    negate2 = negate(negate1)
    negate3 = negate(negate2)
    negate4 = negate(negate3)
    negate5 = negate(negate4)
    negate6 = negate(negate5)
    negate7 = negate(negate6)
    negate8 = negate(negate7)
    negate9 = negate(negate8)
    negate10 = negate(negate9)
    slice1 = f32[1,4,2] slice(negate10), slice={[0:1], [0:4], [0:2]}
    concat = f32[2,4,2] concatenate(slice0, slice1), dimensions={0}
    ROOT root = negate(concat)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 400;
  options.enable_sync_copy_replacement = false;
  options.enable_sync_slice_replacement = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };

  AssignMemorySpace(module.get(), options);

  HloInstruction* negate_p0 = FindInstruction(module.get(), "negate_p0");
  ASSERT_NE(negate_p0, nullptr);
  HloInstruction* concat = FindInstruction(module.get(), "concat");
  ASSERT_NE(concat, nullptr);
  EXPECT_THAT(concat->operand(0),
              op::AsyncDone(op::AsyncStart(op::AsyncCopy(
                  kDefaultMemorySpace, kAlternateMemorySpace, negate_p0))));
}

TEST_F(MemorySpaceAssignmentTest, SyncDynamicSliceReplacementAfterEviction) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[8,4,2]{2,1,0} parameter(0)
    p1 = f32[4,4,2]{2,1,0} parameter(1)
    index = s32[] parameter(2)
    zero = s32[] constant(0)
    negate_p0 = negate(p0)
    dynamic_slice = f32[1,4,2] dynamic-slice(negate_p0, index, zero, zero), dynamic_slice_sizes={1,4,2}
    negate0 = negate(p1)
    negate1 = negate(negate0)
    negate2 = negate(negate1)
    negate3 = negate(negate2)
    negate4 = negate(negate3)
    negate5 = negate(negate4)
    negate6 = negate(negate5)
    negate7 = negate(negate6)
    negate8 = negate(negate7)
    negate9 = negate(negate8)
    negate10 = negate(negate9)
    dynamic_slice1 = f32[1,4,2] dynamic-slice(negate10, index, zero, zero), dynamic_slice_sizes={1,4,2}
    concat = f32[2,4,2] concatenate(dynamic_slice, dynamic_slice1), dimensions={0}
    ROOT root = negate(concat)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 400;
  options.enable_sync_copy_replacement = false;
  options.enable_sync_slice_replacement = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };

  AssignMemorySpace(module.get(), options);
  HloInstruction* negate_p0 = FindInstruction(module.get(), "negate_p0");
  ASSERT_NE(negate_p0, nullptr);
  HloInstruction* concat = FindInstruction(module.get(), "concat");
  ASSERT_NE(concat, nullptr);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* index = FindInstruction(module.get(), "index");
  HloInstruction* zero = FindInstruction(module.get(), "zero");
  EXPECT_THAT(concat->operand(0),
              op::AsyncDone(op::AsyncStart(negate_p0, index, zero, zero)));
  EXPECT_THAT(negate_p0, op::Negate(op::AsyncCopy(kAlternateMemorySpace,
                                                  kDefaultMemorySpace, p0)));
}

TEST_F(MemorySpaceAssignmentTest, SyncSliceReplacementTwoSlices) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[10,2,3]{2,1,0} parameter(0)
    p1 = f32[10,2,3]{2,1,0} parameter(1)
    negate0 = negate(p1)
    negate1 = negate(negate0)
    negate2 = negate(negate1)
    negate3 = negate(negate2)
    negate4 = negate(negate3)
    negate5 = negate(negate4)
    negate6 = negate(negate5)
    negate7 = negate(negate6)
    slice.1 = f32[1,2,3] slice(p0), slice={[0:1], [0:2], [0:3]}
    slice.2 = f32[1,2,3] slice(p0), slice={[1:2], [0:2], [0:3]}
    add = f32[1,2,3] add(slice.1, slice.2)
    ROOT concat = f32[11,2,3] concatenate(negate7, add), dimensions={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 512;
  options.enable_sync_copy_replacement = false;
  options.enable_sync_slice_replacement = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };
  AssignMemorySpace(module.get(), options);
  HloInstruction* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  ASSERT_NE(p0, nullptr);
  ASSERT_NE(add->operand(0), add->operand(1));
  EXPECT_THAT(add->operand(0), op::AsyncDone(op::AsyncStart(p0)));
  EXPECT_THAT(add->operand(1), op::AsyncDone(op::AsyncStart(p0)));
}

TEST_F(MemorySpaceAssignmentTest, SyncDynamicSliceReplacementTwoSlices) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[10,2,3]{2,1,0} parameter(0)
    p1 = f32[10,2,3]{2,1,0} parameter(1)
    index = s32[] parameter(2)
    zero = s32[] constant(0)
    negate0 = negate(p1)
    negate1 = negate(negate0)
    negate2 = negate(negate1)
    negate3 = negate(negate2)
    negate4 = negate(negate3)
    negate5 = negate(negate4)
    negate6 = negate(negate5)
    negate7 = negate(negate6)
    dynamic_slice.1 = f32[1,2,3] dynamic-slice(p0, index, zero, zero), dynamic_slice_sizes={1,2,3}
    dynamic_slice.2 = f32[1,2,3] dynamic-slice(p0, index, zero, zero), dynamic_slice_sizes={1,2,3}
    add = f32[1,2,3] add(dynamic_slice.1, dynamic_slice.2)
    ROOT concat = f32[11,2,3] concatenate(negate7, add), dimensions={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 512;
  options.enable_sync_copy_replacement = false;
  options.enable_sync_slice_replacement = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };
  AssignMemorySpace(module.get(), options);
  HloInstruction* add = FindInstruction(module.get(), "add");
  ASSERT_NE(add, nullptr);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* index = FindInstruction(module.get(), "index");
  HloInstruction* zero = FindInstruction(module.get(), "zero");
  ASSERT_NE(p0, nullptr);
  ASSERT_NE(add->operand(0), add->operand(1));
  EXPECT_THAT(add->operand(0),
              op::AsyncDone(op::AsyncStart(p0, index, zero, zero)));
  EXPECT_THAT(add->operand(1),
              op::AsyncDone(op::AsyncStart(p0, index, zero, zero)));
}

TEST_F(MemorySpaceAssignmentTest, SyncSliceReplacementNestedSlices) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[10,2,3]{2,1,0} parameter(0)
    p1 = f32[10,2,3]{2,1,0} parameter(1)
    negate0 = negate(p1)
    negate1 = negate(negate0)
    negate2 = negate(negate1)
    negate3 = negate(negate2)
    negate4 = negate(negate3)
    negate5 = negate(negate4)
    negate6 = negate(negate5)
    negate7 = negate(negate6)
    slice0 = f32[9,2,3] slice(p0), slice={[0:9], [0:2], [0:3]}
    negate8 = f32[10,2,3] negate(negate7)
    negate9 = f32[10,2,3] negate(negate8)
    negate10 = f32[10,2,3] negate(negate9)
    negate11 = f32[10,2,3] negate(negate10)
    negate12 = f32[10,2,3] negate(negate11)
    slice1 = f32[1,2,3] slice(slice0), slice={[0:1], [0:2], [0:3]}
    concat = f32[11,2,3] concatenate(negate12, slice1), dimensions={0}
    ROOT root = negate(concat)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 300;
  options.enable_sync_copy_replacement = false;
  options.enable_sync_slice_replacement = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };
  AssignMemorySpace(module.get(), options);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  ASSERT_NE(p0, nullptr);
  HloInstruction* concat = FindInstruction(module.get(), "concat");
  ASSERT_NE(concat, nullptr);
  EXPECT_THAT(concat->operand(1), op::Slice(op::AsyncDone(op::AsyncStart(p0))));
}

TEST_F(MemorySpaceAssignmentTest, SyncDynamicSliceReplacementNestedSlices) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[10,2,3]{2,1,0} parameter(0)
    p1 = f32[10,2,3]{2,1,0} parameter(1)
    index = s32[] parameter(2)
    zero = s32[] constant(0)
    negate0 = negate(p1)
    negate1 = negate(negate0)
    negate2 = negate(negate1)
    negate3 = negate(negate2)
    negate4 = negate(negate3)
    negate5 = negate(negate4)
    negate6 = negate(negate5)
    negate7 = negate(negate6)
    dynamic_slice.0 = f32[9,2,3] dynamic-slice(p0, index, zero, zero), dynamic_slice_sizes={9,2,3}
    negate8 = f32[10,2,3] negate(negate7)
    negate9 = f32[10,2,3] negate(negate8)
    negate10 = f32[10,2,3] negate(negate9)
    negate11 = f32[10,2,3] negate(negate10)
    negate12 = f32[10,2,3] negate(negate11)
    slice1 = f32[1,2,3] slice(dynamic_slice.0), slice={[0:1], [0:2], [0:3]}
    concat = f32[11,2,3] concatenate(negate12, slice1), dimensions={0}
    ROOT root = negate(concat)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 300;
  options.enable_sync_copy_replacement = false;
  options.enable_sync_slice_replacement = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };
  AssignMemorySpace(module.get(), options);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* index = FindInstruction(module.get(), "index");
  HloInstruction* zero = FindInstruction(module.get(), "zero");
  ASSERT_NE(p0, nullptr);
  HloInstruction* concat = FindInstruction(module.get(), "concat");
  ASSERT_NE(concat, nullptr);
  EXPECT_THAT(concat->operand(1),
              op::Slice(op::AsyncDone(op::AsyncStart(p0, index, zero, zero))));
}

TEST_F(MemorySpaceAssignmentTest, SyncSliceReplacementOneFails) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[10,2,3]{2,1,0} parameter(0)
    p1 = f32[1,2,3]{2,1,0} parameter(1)
    negate0 = f32[1,2,3]{2,1,0:S(1)} negate(p1)
    negate1 = f32[1,2,3]{2,1,0:S(1)} negate(negate0)
    negate2 = f32[1,2,3]{2,1,0:S(1)} negate(negate1)
    negate3 = f32[1,2,3]{2,1,0:S(1)} negate(negate2)
    negate4 = f32[1,2,3]{2,1,0:S(1)} negate(negate3)
    negate5 = f32[1,2,3]{2,1,0:S(1)} negate(negate4)
    negate6 = f32[1,2,3]{2,1,0:S(1)} negate(negate5)
    negate7 = f32[1,2,3]{2,1,0:S(1)} negate(negate6)
    slice.0 = f32[8,2,3] slice(p0), slice={[0:8], [0:2], [0:3]}
    slice.1 = f32[1,2,3] slice(p0), slice={[8:9], [0:2], [0:3]}
    slice.2 = f32[1,2,3] slice(p0), slice={[9:10], [0:2], [0:3]}
    add.0 = f32[1,2,3] add(slice.1, slice.2)
    concat.0 = f32[9,2,3] concatenate(slice.0, add.0), dimensions={0}
    ROOT concat.1 = f32[10,2,3] concatenate(negate7, concat.0), dimensions={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 72;
  options.enable_sync_copy_replacement = false;
  options.enable_sync_slice_replacement = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };
  AssignMemorySpace(module.get(), options);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  ASSERT_NE(p0, nullptr);
  HloInstruction* add0 = FindInstruction(module.get(), "add.0");
  ASSERT_NE(add0, nullptr);
  EXPECT_THAT(add0->operand(0), op::AsyncDone(op::AsyncStart(p0)));
  EXPECT_THAT(add0->operand(1), op::AsyncDone(op::AsyncStart(p0)));
  HloInstruction* slice0 = FindInstruction(module.get(), "slice.0");
  ASSERT_NE(slice0, nullptr);
}

TEST_F(MemorySpaceAssignmentTest, SyncDynamicSliceReplacementOneFails) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[10,2,3]{2,1,0} parameter(0)
    p1 = f32[1,2,3]{2,1,0} parameter(1)
    index = s32[] parameter(2)
    zero = s32[] constant(0)
    negate0 = f32[1,2,3]{2,1,0:S(1)} negate(p1)
    negate1 = f32[1,2,3]{2,1,0:S(1)} negate(negate0)
    negate2 = f32[1,2,3]{2,1,0:S(1)} negate(negate1)
    negate3 = f32[1,2,3]{2,1,0:S(1)} negate(negate2)
    negate4 = f32[1,2,3]{2,1,0:S(1)} negate(negate3)
    negate5 = f32[1,2,3]{2,1,0:S(1)} negate(negate4)
    negate6 = f32[1,2,3]{2,1,0:S(1)} negate(negate5)
    negate7 = f32[1,2,3]{2,1,0:S(1)} negate(negate6)
    dynamic_slice.0 = f32[8,2,3] dynamic-slice(p0, index, zero, zero), dynamic_slice_sizes={8,2,3}
    dynamic_slice.1 = f32[1,2,3] dynamic-slice(p0, index, zero, zero), dynamic_slice_sizes={1,2,3}
    dynamic_slice.2 = f32[1,2,3] dynamic-slice(p0, index, zero, zero), dynamic_slice_sizes={1,2,3}
    add.0 = f32[1,2,3] add(dynamic_slice.1, dynamic_slice.2)
    concat.0 = f32[9,2,3] concatenate(dynamic_slice.0, add.0), dimensions={0}
    ROOT concat.1 = f32[10,2,3] concatenate(negate7, concat.0), dimensions={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 72;
  options.enable_sync_copy_replacement = false;
  options.enable_sync_slice_replacement = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };
  AssignMemorySpace(module.get(), options);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* index = FindInstruction(module.get(), "index");
  HloInstruction* zero = FindInstruction(module.get(), "zero");
  ASSERT_NE(p0, nullptr);
  HloInstruction* add0 = FindInstruction(module.get(), "add.0");
  ASSERT_NE(add0, nullptr);
  EXPECT_THAT(add0->operand(0),
              op::AsyncDone(op::AsyncStart(p0, index, zero, zero)));
  EXPECT_THAT(add0->operand(1),
              op::AsyncDone(op::AsyncStart(p0, index, zero, zero)));
  HloInstruction* slice0 = FindInstruction(module.get(), "dynamic_slice.0");
  ASSERT_NE(slice0, nullptr);
}

// The prefetch logic has to correctly distinguish the output shape of an async
// copy vs an async slice. In this test, a prefetch of p0 would not fit into the
// memory, while prefetching a slice of p0 is feasible.
TEST_F(MemorySpaceAssignmentTest, SyncSliceReplacementTheSlicedOneFits) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[10,2,3]{2,1,0} parameter(0)
  p1 = f32[1,2,3]{2,1,0} parameter(1)
  negate0 = negate(p1)
  negate1 = negate(negate0)
  negate2 = negate(negate1)
  negate3 = negate(negate2)
  negate4 = negate(negate3)
  negate5 = negate(negate4)
  negate6 = negate(negate5)
  negate7 = negate(negate6)
  slice = f32[1,2,3] slice(p0), slice={[0:1], [0:2], [0:3]}
  concat = f32[2,2,3] concatenate(negate7, slice), dimensions={0}
  ROOT root = negate(concat)
  }
  )";

  Options options = DefaultMemorySpaceOptions();
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };
  options.max_size_in_bytes = 64;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module1,
                          ParseAndReturnVerifiedModule(hlo_string));
  options.enable_sync_slice_replacement = false;
  AssignMemorySpace(module1.get(), options);
  HloInstruction* p0 = FindInstruction(module1.get(), "p0");
  ASSERT_NE(p0, nullptr);
  HloInstruction* concat = FindInstruction(module1.get(), "concat");
  ASSERT_NE(concat, nullptr);
  EXPECT_THAT(concat->operand(1), op::Slice(p0));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module2,
                          ParseAndReturnVerifiedModule(hlo_string));
  options.enable_sync_slice_replacement = true;
  AssignMemorySpace(module2.get(), options);
  p0 = FindInstruction(module2.get(), "p0");
  ASSERT_NE(p0, nullptr);
  concat = FindInstruction(module2.get(), "concat");
  ASSERT_NE(concat, nullptr);
  EXPECT_THAT(concat->operand(1), op::AsyncDone(op::AsyncStart(p0)));
}

TEST_F(MemorySpaceAssignmentTest, SyncReplacementMultipleOpTypes) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[10,2,3]{2,1,0} parameter(0)
  p1 = f32[10,2,3]{2,1,0} parameter(1)
  p0_copy = copy(p0)
  negate0 = negate(p1)
  negate1 = negate(negate0)
  negate2 = negate(negate1)
  negate3 = negate(negate2)
  negate4 = negate(negate3)
  negate5 = negate(negate4)
  negate6 = negate(negate5)
  negate7 = negate(negate6)
  slice = f32[1,2,3] slice(p0_copy), slice={[0:1], [0:2], [0:3]}
  concat = f32[11,2,3] concatenate(negate7, slice), dimensions={0}
  ROOT root = negate(concat)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 512;
  options.enable_sync_copy_replacement = true;
  options.enable_sync_slice_replacement = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };
  AssignMemorySpace(module.get(), options);
  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  ASSERT_NE(p0, nullptr);
  HloInstruction* concat = FindInstruction(module.get(), "concat");
  ASSERT_NE(concat, nullptr);
  EXPECT_THAT(
      concat->operand(1),
      op::Slice(op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace, p0)));
}

// This test is for the redundant aliasing bug (b/374902759) introduced between
// different operands of the same instruction while converting sync copy to
// async ones.
TEST_F(MemorySpaceAssignmentTest, SyncReplacementAliasingBug) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true, entry_computation_layout={(f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, pred[])->f32[10,2,3]{2,1,0}}

%while_body (p0.1: (f32[10,2,3], f32[10,2,3], f32[10,2,3], pred[])) -> (f32[10,2,3], f32[10,2,3], f32[10,2,3], pred[]) {
  %p0.1 = (f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, pred[]) parameter(0)
  %gte0 = f32[10,2,3]{2,1,0} get-tuple-element((f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, pred[]) %p0.1), index=0
  %gte1 = f32[10,2,3]{2,1,0} get-tuple-element((f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, pred[]) %p0.1), index=1
  %gte2 = f32[10,2,3]{2,1,0} get-tuple-element((f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, pred[]) %p0.1), index=2
  %gte3 = pred[] get-tuple-element((f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, pred[]) %p0.1), index=3
  %neg0 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %gte2)
  %neg1 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %neg0)
  %neg2 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %neg1)
  %neg3 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %neg2)
  %neg4 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %neg3)
  %neg5 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %neg4)
  %neg6 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %neg5)
  %neg7 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %neg6)
  ROOT %tuple = (f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, pred[]) tuple(f32[10,2,3]{2,1,0} %gte0, f32[10,2,3]{2,1,0} %gte1, f32[10,2,3]{2,1,0} %neg7, pred[] %gte3)
}

%while_cond (p0: (f32[10,2,3], f32[10,2,3], f32[10,2,3], pred[])) -> pred[] {
  %p0 = (f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, pred[]) parameter(0)
  ROOT %gte = pred[] get-tuple-element((f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, pred[]) %p0), index=3
}

ENTRY %entry (p0.2: f32[10,2,3], p1: f32[10,2,3], p2: pred[]) -> f32[10,2,3] {
  %p0.2 = f32[10,2,3]{2,1,0} parameter(0)
  %p1 = f32[10,2,3]{2,1,0} parameter(1)
  %p2 = pred[] parameter(2)
  p0_copy = f32[10,2,3]{2,1,0} copy(f32[10,2,3]{2,1,0} %p0.2)
  %p0_copy_copy = f32[10,2,3]{2,1,0} copy(f32[10,2,3]{2,1,0} p0_copy)
  %negate0 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %p1)
  %negate1 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %negate0)
  %negate2 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %negate1)
  %negate3 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %negate2)
  %negate4 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %negate3)
  %negate5 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %negate4)
  %negate6 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %negate5)
  %negate7 = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %negate6)
  %tuple.3 = (f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, pred[]) tuple(f32[10,2,3]{2,1,0} %negate7, f32[10,2,3]{2,1,0} %p0_copy, f32[10,2,3]{2,1,0} %p0_copy_copy, pred[] %p2)
  while.1 = (f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, pred[]) while((f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, pred[]) %tuple.3), condition=%while_cond, body=%while_body
  %gte.1 = f32[10,2,3]{2,1,0} get-tuple-element((f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, f32[10,2,3]{2,1,0}, pred[]) while.1), index=2
  ROOT %negate = f32[10,2,3]{2,1,0} negate(f32[10,2,3]{2,1,0} %gte.1)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 1024;
  options.enable_sync_copy_replacement = true;
  options.enable_sync_slice_replacement = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };
  AssignMemorySpace(module.get(), options);
  HloInstruction* while_instruction = FindInstruction(module.get(), "while.1");
  ASSERT_NE(while_instruction, nullptr);
  const HloInstruction* tuple = while_instruction->operand(0);
  HloInstruction* p0_copy = FindInstruction(module.get(), "p0_copy");
  ASSERT_NE(p0_copy, nullptr);
  EXPECT_THAT(tuple->operand(1), op::AsyncCopy(kDefaultMemorySpace,
                                               kAlternateMemorySpace, p0_copy));
  EXPECT_THAT(tuple->operand(2),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            tuple->operand(1)));
}

// Added for b/376344953 that was introduced when we tried to
// convert a sync copy that was used by a conditional into an async copy.
TEST_F(MemorySpaceAssignmentTest, ConditionalCopyReplacement) {
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg1 = f32[3]{0} negate(gte)
  }

  false_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg2 = f32[3]{0} negate(gte)
  }

  ENTRY entry {
    p0_main = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0_main)
    tuple = (f32[3]{0}) tuple(copy)
    ROOT conditional = f32[3]{0} conditional(p1, tuple, tuple), true_computation=true_computation, false_computation=false_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.enable_sync_copy_replacement = true;
  AssignMemorySpace(module.get(), options);
  auto conditional =
      module->GetComputationWithName("entry")->GetInstructionWithName(
          "conditional");
  CHECK_NE(conditional, nullptr);
  auto p0 = module->GetComputationWithName("entry")->GetInstructionWithName(
      "p0_main");
  CHECK_NE(p0, nullptr);
  auto copy = conditional->operand(1)->operand(0);
  CHECK_NE(copy, nullptr);
  EXPECT_THAT(copy,
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace, p0));
}

TEST_F(MemorySpaceAssignmentTest, AllocationRequestAndResultModifierTest) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[2,3]{1,0} parameter(0)
  p1 = f32[2,3]{1,0} parameter(1)
  negate0 = f32[2,3]{1,0} negate(p1)
  negate1 = f32[2,3]{1,0} negate(negate0)
  negate2 = f32[2,3]{1,0} negate(negate1)
  negate3 = f32[2,3]{1,0} negate(negate2)
  negate4 = f32[2,3]{1,0} negate(negate3)
  negate5 = f32[2,3]{1,0} negate(negate4)
  negate6 = f32[2,3]{1,0} negate(negate5)
  negate7 = f32[2,3]{1,0} negate(negate6)
  ROOT add0 = f32[2,3]{1,0} add(p0, negate7)
  }
  )";
  // The baseline behavior is to prefetch p0 at add0.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> baseline_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  AssignMemorySpace(baseline_module.get(), options);
  HloInstruction* add0 = FindInstruction(baseline_module.get(), "add0");
  ASSERT_NE(add0, nullptr);
  HloInstruction* p0 = FindInstruction(baseline_module.get(), "p0");
  ASSERT_NE(p0, nullptr);
  EXPECT_THAT(add0->operand(0),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace, p0));

  // We should be able to prevent prefetching p0 at add0 using
  // allocation_result_modifier_testing_fn.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> result_modifier_module,
      ParseAndReturnVerifiedModule(hlo_string));
  options.max_retries = 1;
  options.allocation_request_modifier_testing_fn = nullptr;
  options.allocation_result_modifier_testing_fn =
      [](const AllocationRequest& request, AllocationResult& result) {
        if (request.allocation_value_to_update->defining_instruction()
                    ->name() == "p0" &&
            request.use->hlo_use.instruction->name() == "add0") {
          result = AllocationResult::kFailRequiresUncommit;
        }
      };
  AssignMemorySpace(result_modifier_module.get(), options);
  add0 = FindInstruction(result_modifier_module.get(), "add0");
  ASSERT_NE(add0, nullptr);
  p0 = FindInstruction(result_modifier_module.get(), "p0");
  ASSERT_NE(p0, nullptr);
  EXPECT_EQ(add0->operand(0), p0);

  // We should be able to enforce an earlier prefetch of p0 at add0 using
  // allocation_request_modifier_testing_fn.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> request_modifier_module,
      ParseAndReturnVerifiedModule(hlo_string));
  options.max_retries = 1;
  options
      .allocation_request_modifier_testing_fn = [](AllocationRequest& request) {
    if (request.allocation_value_to_update->defining_instruction()->name() ==
            "p0" &&
        request.use->hlo_use.instruction->name() == "add0") {
      // Schedule the copy-done before negate4 (scheduled at 6).
      request.latest_prefetch_time = 6;
    }
  };
  options.allocation_result_modifier_testing_fn = nullptr;
  AssignMemorySpace(request_modifier_module.get(), options);
  add0 = FindInstruction(request_modifier_module.get(), "add0");
  CHECK_NE(add0, nullptr);
  p0 = FindInstruction(request_modifier_module.get(), "p0");
  CHECK_NE(p0, nullptr);
  EXPECT_THAT(add0->operand(0),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace, p0));
  // The copy-done should have been scheduled before negate4.
  HloInstruction* negate4 =
      FindInstruction(request_modifier_module.get(), "negate4");
  CHECK_NE(negate4, nullptr);
  const HloInstructionSequence& sequence =
      request_modifier_module->schedule().sequence(
          request_modifier_module->entry_computation());
  auto find_index = [&](const HloInstruction* instruction) {
    return std::distance(sequence.instructions().begin(),
                         std::find(sequence.instructions().begin(),
                                   sequence.instructions().end(), instruction));
  };

  int negate4_index = find_index(negate4);
  int copy_done_index = find_index(add0->operand(0));
  EXPECT_LT(copy_done_index, negate4_index);
}

// Added for b/372277844#comment15 that was introduced when the allocation
// failed while trying to convert a sync slice to an async one, but not due to
// the conversion itself. In this case, associated buffer with the slice
// (p0_copy) is too large to fit in alternate memory. Hence, the
// allocation_values will be empty in retries, previously causing a crash in
// MsaAlgorithm::GetInefficientAllocationSites().
TEST_F(MemorySpaceAssignmentTest, SyncReplacementLargeBuffers) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[10,2,3]{2,1,0} parameter(0)
  p1 = f32[10,2,3]{2,1,0} parameter(1)
  p0_copy = f32[10,2,3]{2,1,0} copy(p0)
  negate0 = negate(p1)
  negate1 = negate(negate0)
  negate2 = negate(negate1)
  negate3 = negate(negate2)
  negate4 = negate(negate3)
  negate5 = negate(negate4)
  slice = f32[1,2,3] slice(p0_copy), slice={[0:1], [0:2], [0:3]}
  ROOT concat = f32[11,2,3] concatenate(negate5, slice), dimensions={0}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 64;
  options.max_retries = 2;
  options.enable_sync_copy_replacement = false;
  options.enable_sync_slice_replacement = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };
  // Force the allocation of p0_copy to fail for the concat use with only
  // AllocationResult::kFailRequiresUncommit. This means that while the slice
  // replacement was successful, the algorithm must retry one more time without
  // sync slice conversion target, so that maybe other constraints of the
  // allocation can be satisfied.
  options.allocation_result_modifier_testing_fn =
      [](const AllocationRequest& request, AllocationResult& result) {
        if (request.allocation_value->defining_instruction()->name() ==
                "p0_copy" &&
            request.use->hlo_use.instruction->name() == "concat") {
          result = AllocationResult::kFailRequiresUncommit;
        }
      };
  // options.inefficient_use_to_copy_ratio must be greater than 0 and the cost
  // model must be set to trigger the inefficient allocation site logic.
  options.inefficient_use_to_copy_ratio = 1.0;
  AssignMemorySpaceUsingCostAnalysis(module.get(), options);

  HloInstruction* p0_copy = FindInstruction(module.get(), "p0_copy");
  ASSERT_NE(p0_copy, nullptr);
  HloInstruction* concat = FindInstruction(module.get(), "concat");
  ASSERT_NE(concat, nullptr);
  EXPECT_THAT(concat->operand(1), op::Slice(p0_copy));
}

// Added for b/376869021, which surfaced when we tried to convert a sync slice
// that had to extend the allocation of its operand in the alternate memory. In
// this test, we expect the slice0 operand (p0_copy) maintain a valid allocation
// in the alternate memory, until it gets transferred by the async replacement
// of slice0. We hence stress-test such validity by delaying the allocation of
// slice0 by 3 steps.
TEST_F(MemorySpaceAssignmentTest, SyncReplacementAllocationExtensionBug) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[2,2,3]{2,1,0} parameter(0)
  p1 = f32[4,2,3]{2,1,0} parameter(1)
  p0_copy = f32[2,2,3]{2,1,0} copy(p0)
  negate0 = negate(p1)
  negate1 = negate(negate0)
  negate2 = negate(negate1)
  p0_copy0_negate = negate(p0_copy)
  copy_negate2 = copy(negate2)
  slice0 = f32[1,2,3] slice(p0_copy), slice={[0:1], [0:2], [0:3]}
  negate3 = negate(copy_negate2)
  negate4 = negate(negate3)
  negate5 = negate(negate4)
  negate6 = negate(negate5)
  negate7 = negate(negate6)
  neg_slice0 = negate(slice0)
  ROOT tuple = tuple(negate7, neg_slice0)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.enable_sync_copy_replacement = false;
  options.enable_sync_slice_replacement = true;
  options.verify = true;
  options.is_async_slice_implemented_fn =
      [](const HloInstruction* instruction) { return true; };
  options.max_size_in_bytes = 96;
  options.is_position_allowed_in_alternate_mem_fn =
      [](const HloPosition& position) {
        return position.instruction->name() != "p0_copy";
      };
  // Delay the allocation of slice0 by 3 steps to allow copy_negate2 to be
  // allocated in alternate memory.
  options.allocation_request_modifier_testing_fn =
      [](AllocationRequest& request) {
        if (request.only_extend_existing_allocation) {
          request.inclusive_start_time += 3;
          request.end_time += 3;
        }
      };
  const std::string text_proto = R"pb(
    overrides {
      hlo_position_matcher { instruction_name_regex: "copy_negate2|p0_copy" }
      override_options { assign_first: true }
    })pb";
  TF_ASSERT_OK_AND_ASSIGN(auto msa_sort_order_overrides,
                          ParseTextProto<MsaSortOrderOverrides>(text_proto));
  auto preset_assignments = AssignMemorySpaceUsingCostAnalysis(
      module.get(), options,
      /*cost_analysis_options_override=*/std::nullopt,
      /*hlo_cost_options_override=*/std::nullopt,
      /*optional_msa_sort_order_overrides=*/msa_sort_order_overrides);
  HloInstruction* p0_copy = FindInstruction(module.get(), "p0_copy");
  ASSERT_NE(p0_copy, nullptr);
  HloInstruction* neg_slice0 = FindInstruction(module.get(), "neg_slice0");
  ASSERT_NE(neg_slice0, nullptr);
  EXPECT_THAT(neg_slice0->operand(0), op::AsyncDone(op::AsyncStart(p0_copy)));
}

TEST_F(MemorySpaceAssignmentTest, AlwaysSpillJitPrefetchTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  // For buffers produced in alternate memory spill to default and prefetch
  // just in time for uses other than immediate use (if any) and make all
  // prefetches single use for first use and create new prefetches for all
  // subsequent uses.

  // We expect MSA to start prefetching p1 immediately the parameter(1)
  // instruction and to finish immediately before add. The
  // always_spill_to_default_memory option will move the start of the prefetch
  // from just after parameter(1) to just before its completion.
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[2,3]{1,0} parameter(0)
  p1 = f32[2,3]{1,0} parameter(1)
  negate0 = f32[2,3]{1,0} negate(p0)
  negate1 = f32[2,3]{1,0} negate(negate0)
  negate2 = f32[2,3]{1,0} negate(negate1)
  negate3 = f32[2,3]{1,0} negate(negate2)
  negate4 = f32[2,3]{1,0} negate(negate3)
  negate5 = f32[2,3]{1,0} negate(negate4)
  negate6 = f32[2,3]{1,0} negate(negate5)
  ROOT add = f32[2,3]{1,0} add(negate6, p1)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.always_spill_to_default_memory = true;
  AssignMemorySpace(module.get(), options);
  const HloInstructionSequence& sequence =
      module->schedule().sequence(module->entry_computation());
  for (int i = 0; i < sequence.instructions().size(); ++i) {
    VLOG(2) << i << " " << sequence.instructions()[i]->ToString();
  }
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                          HloAliasAnalysis::Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloLiveRange> live_range,
                          HloLiveRange::Run(module->schedule(), *alias_analysis,
                                            module->entry_computation()));
  const HloInstruction* add = FindInstruction(module.get(), "add");
  const HloInstruction* cd = add->operand(1);
  // Check copy made just in time for use and copy is a prefetch.
  EXPECT_THAT(cd, op::CopyDone());
  EXPECT_EQ(live_range->instruction_schedule().at(add),
            live_range->instruction_schedule().at(cd) + 1);
  const HloInstruction* cs = cd->operand(0);
  EXPECT_THAT(cs, op::CopyStart());
  EXPECT_EQ(live_range->instruction_schedule().at(add),
            live_range->instruction_schedule().at(cs) + 2);
  EXPECT_THAT(add, op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(1))));
}

TEST_F(MemorySpaceAssignmentTest, AlwaysSpillPrefetchForSecondUseTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  //
  // Setting always_spill_to_default_memory option to true makes sure the
  // negate0 buffer is copied to default memory between negate0 and negate1,
  // so that version can be prefetched just before it is used at add0.
  // Additionally, we leave a copy of negate0 in alternate memory for use at
  // negate1.
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[2,3]{1,0} parameter(0)
  p1 = f32[2,3]{1,0} parameter(1)
  negate0 = f32[2,3]{1,0} negate(p0)
  negate1 = f32[2,3]{1,0} negate(negate0)
  negate2 = f32[2,3]{1,0} negate(negate1)
  negate3 = f32[2,3]{1,0} negate(negate2)
  negate4 = f32[2,3]{1,0} negate(negate3)
  negate5 = f32[2,3]{1,0} negate(negate4)
  add0 = f32[2,3]{1,0} add(negate5, negate0)
  ROOT add1 = f32[2,3]{1,0} add(add0, p1)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.always_spill_to_default_memory = true;
  AssignMemorySpace(module.get(), options);
  const HloInstructionSequence& sequence =
      module->schedule().sequence(module->entry_computation());
  for (int i = 0; i < sequence.instructions().size(); ++i) {
    VLOG(2) << i << " " << sequence.instructions()[i]->ToString();
  }
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                          HloAliasAnalysis::Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloLiveRange> live_range,
                          HloLiveRange::Run(module->schedule(), *alias_analysis,
                                            module->entry_computation()));
  // Check copies are made just in time for use and copies are prefetches.
  const HloInstruction* add1 = FindInstruction(module.get(), "add1");
  const HloInstruction* cd1 = add1->operand(1);
  EXPECT_THAT(cd1, op::CopyDone());
  EXPECT_EQ(live_range->instruction_schedule().at(add1),
            live_range->instruction_schedule().at(cd1) + 1);
  const HloInstruction* cs1 = cd1->operand(0);
  EXPECT_THAT(cs1, op::CopyStart());
  EXPECT_EQ(live_range->instruction_schedule().at(add1),
            live_range->instruction_schedule().at(cs1) + 2);
  EXPECT_EQ(cd1->shape().layout().memory_space(), kAlternateMemorySpace);
  const HloInstruction* add0 = FindInstruction(module.get(), "add0");

  const HloInstruction* cd0 = add0->operand(1);
  EXPECT_THAT(cd0, op::CopyDone());
  EXPECT_EQ(live_range->instruction_schedule().at(add0),
            live_range->instruction_schedule().at(cd0) + 1);
  const HloInstruction* cs0 = cd0->operand(0);
  EXPECT_THAT(cs0, op::CopyStart());
  EXPECT_EQ(live_range->instruction_schedule().at(add0),
            live_range->instruction_schedule().at(cs0) + 2);
  EXPECT_EQ(cd0->shape().layout().memory_space(), kAlternateMemorySpace);
  // Check prefetch was made from an eviction.
  const HloInstruction* eviction_done = cs0->operand(0);
  EXPECT_EQ(eviction_done->shape().layout().memory_space(),
            kDefaultMemorySpace);
  const HloInstruction* evection_start = eviction_done->operand(0);
  const HloInstruction* negate0 = evection_start->operand(0);
  // Check eviction was immediate.
  EXPECT_EQ(live_range->instruction_schedule().at(evection_start),
            live_range->instruction_schedule().at(negate0) + 1);
  EXPECT_EQ(live_range->instruction_schedule().at(eviction_done),
            live_range->instruction_schedule().at(negate0) + 2);
  EXPECT_EQ(negate0->name(), "negate0");
}

TEST_F(MemorySpaceAssignmentTest, AlwaysSpillEvictionTest) {
  // tanh0 buffer is produced in alternate memory and it has two uses that are
  // sufficiently far apart for an eviction to be scheduled. When the
  // always_spill_to_default_memory option is not true, the buffer stays in
  // alternate memory to serve the first use, is evicted and prefetched again
  // for second use. Setting always_spill_to_default_memory option to true makes
  // the eviction immediate, right after tanh0, the first use at add5 and second
  // use at tuple are served from separate, just-in-time prefetches that copy
  // from the eviction that previously occurred.
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[4,3]{1,0} parameter(0)
  tanh0 = f32[4,3]{1,0} tanh(p0)
  add0 = f32[4,3]{1,0} add(p0, p0)
  add1 = f32[4,3]{1,0} add(add0, p0)
  add2 = f32[4,3]{1,0} add(add1, p0)
  add3 = f32[4,3]{1,0} add(add2, p0)
  add4 = f32[4,3]{1,0} add(add3, p0)
  add5 = f32[4,3]{1,0} add(add4, tanh0)
  negate0 = f32[4,3]{1,0} negate(add5)
  tanh1 = f32[4,3]{1,0} tanh(negate0)
  negate1 = f32[4,3]{1,0} negate(negate0)
  tanh2 = f32[4,3]{1,0} tanh(tanh1)
  negate2 = f32[4,3]{1,0} negate(negate1)
  ROOT tuple = tuple(tanh0, tanh2, negate2)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.always_spill_to_default_memory = true;
  AssignMemorySpace(module.get(), options);
  const HloInstructionSequence& sequence =
      module->schedule().sequence(module->entry_computation());
  for (int i = 0; i < sequence.instructions().size(); ++i) {
    VLOG(2) << i << " " << sequence.instructions()[i]->ToString();
  }
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                          HloAliasAnalysis::Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloLiveRange> live_range,
                          HloLiveRange::Run(module->schedule(), *alias_analysis,
                                            module->entry_computation()));
  // 1. Check tanh0 buffer is short lived.
  // 2. Check tanh0 eviction is immediate.
  // 3. Check tuple is served from eviction.
  // 4. Check add5 is served from a prefetch.
  // 5. Check prefetch comes from the immediate eviction.
  const HloInstruction* tuple = FindInstruction(module.get(), "tuple");
  const HloInstruction* tanh0_eviction_done = tuple->operand(0);
  const HloInstruction* tanh0_eviction_start = tanh0_eviction_done->operand(0);
  const HloInstruction* tanh0 = tanh0_eviction_start->operand(0);
  EXPECT_EQ(tanh0->name(), "tanh0");
  EXPECT_EQ(tanh0_eviction_done->shape().layout().memory_space(),
            kDefaultMemorySpace);
  EXPECT_EQ(live_range->instruction_schedule().at(tanh0_eviction_start),
            live_range->instruction_schedule().at(tanh0) + 1);
  EXPECT_EQ(live_range->instruction_schedule().at(tanh0_eviction_done),
            live_range->instruction_schedule().at(tanh0) + 2);
  const HloInstruction* add5 = FindInstruction(module.get(), "add5");
  const HloInstruction* tanh0_prefetch_done = add5->operand(1);
  const HloInstruction* tanh0_prefetch_start = tanh0_prefetch_done->operand(0);
  EXPECT_EQ(tanh0_prefetch_done->shape().layout().memory_space(),
            kAlternateMemorySpace);
  EXPECT_EQ(live_range->instruction_schedule().at(add5),
            live_range->instruction_schedule().at(tanh0_prefetch_done) + 1);
  EXPECT_EQ(live_range->instruction_schedule().at(add5),
            live_range->instruction_schedule().at(tanh0_prefetch_start) + 2);
  EXPECT_EQ(tanh0_eviction_done, tanh0_prefetch_start->operand(0));
}

TEST_F(MemorySpaceAssignmentTest, FilterUpdatePreferredPrefetchTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options = DefaultMemorySpaceOptions();

  const std::string text_proto = R"pb(
    overrides {
      hlo_operand_filter { size_lte: 24 size_gte: 24 }
      override_options { prefetch_eagerness: 0.5 }
    })pb";
  TF_ASSERT_OK_AND_ASSIGN(
      options.preferred_prefetch_overrides,
      ParseTextProto<PreferredPrefetchOverrides>(text_proto));

  AssignMemorySpace(module.get(), options);

  EXPECT_THAT(add, op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(1))));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0},
      /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1,
      /*element_size_in_bits=*/0, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
  // Ensure the CopyStart/CopyDone schedules.
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  EXPECT_THAT(sequence.instructions()[0], op::Parameter(0));
  EXPECT_THAT(sequence.instructions()[1], op::Parameter(1));
  EXPECT_THAT(sequence.instructions()[6], op::CopyStart());
  EXPECT_THAT(sequence.instructions()[10], op::CopyDone());
}

TEST_F(MemorySpaceAssignmentTest, FilterUpdateConfigExactMatchBeforeTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options = DefaultMemorySpaceOptions();

  const std::string text_proto = R"pb(
    overrides {
      hlo_operand_filter { instruction_name_regex: "add" operand_number: 1 }
      override_options {
        before_instruction: { instruction_regex: "%?negate.3 =.*" }
      }
    })pb";
  TF_ASSERT_OK_AND_ASSIGN(
      options.preferred_prefetch_overrides,
      ParseTextProto<PreferredPrefetchOverrides>(text_proto));

  AssignMemorySpace(module.get(), options);

  EXPECT_THAT(add, op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(1))));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0},
      /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1,
      /*element_size_in_bits=*/0, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
  // Ensure the CopyStart/CopyDone schedules.
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  EXPECT_THAT(sequence.instructions()[0], op::Parameter(0));
  EXPECT_THAT(sequence.instructions()[1], op::Parameter(1));
  EXPECT_THAT(sequence.instructions()[5], op::CopyStart());
  EXPECT_THAT(sequence.instructions()[10], op::CopyDone());
}

TEST_F(MemorySpaceAssignmentTest, FilterUpdateConfigExactMatchAfterTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options = DefaultMemorySpaceOptions();

  const std::string text_proto = R"pb(
    overrides {
      hlo_operand_filter { instruction_name_regex: "add" operand_number: 1 }
      override_options {
        after_instruction: { instruction_regex: "%?negate.1 =.*" }
      }
    })pb";
  TF_ASSERT_OK_AND_ASSIGN(
      options.preferred_prefetch_overrides,
      ParseTextProto<PreferredPrefetchOverrides>(text_proto));

  AssignMemorySpace(module.get(), options);

  EXPECT_THAT(add, op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(1))));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0},
      /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1,
      /*element_size_in_bits=*/0, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
  // Ensure the CopyStart/CopyDone schedules.
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  EXPECT_THAT(sequence.instructions()[0], op::Parameter(0));
  EXPECT_THAT(sequence.instructions()[1], op::Parameter(1));
  EXPECT_THAT(sequence.instructions()[4], op::CopyStart());
  EXPECT_THAT(sequence.instructions()[10], op::CopyDone());
}

TEST_F(MemorySpaceAssignmentTest, FilterUpdateConfigExactMatchTooLateTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options = DefaultMemorySpaceOptions();

  const std::string text_proto = R"pb(
    overrides {
      hlo_operand_filter { instruction_name_regex: "add" operand_number: 1 }
      override_options {
        after_instruction: { instruction_name_regex: "%?negate.5" }
      }
    })pb";
  TF_ASSERT_OK_AND_ASSIGN(
      options.preferred_prefetch_overrides,
      ParseTextProto<PreferredPrefetchOverrides>(text_proto));

  AssignMemorySpace(module.get(), options);

  // Ensure the Async copy is not scheduled.
  EXPECT_THAT(add, op::Add(op::Negate(), op::Parameter(1)));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0},
      /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1,
      /*element_size_in_bits=*/0, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
}

TEST_F(MemorySpaceAssignmentTest, FilterUpdateConfigPrecedenceTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options = DefaultMemorySpaceOptions();

  const std::string text_proto = R"pb(
    overrides {
      hlo_operand_filter { size_lte: 24 size_gte: 24 }
      override_options { prefetch_eagerness: 0.5 }
    }
    overrides {
      hlo_operand_filter { instruction_name_regex: "add" operand_number: 1 }
      override_options {
        after_instruction: { instruction_name_regex: "%?negate.1" }
      }
    })pb";
  TF_ASSERT_OK_AND_ASSIGN(
      options.preferred_prefetch_overrides,
      ParseTextProto<PreferredPrefetchOverrides>(text_proto));

  AssignMemorySpace(module.get(), options);

  EXPECT_THAT(add, op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(1))));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0},
      /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1,
      /*element_size_in_bits=*/0, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
  // Ensure the CopyStart/CopyDone schedules.
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  EXPECT_THAT(sequence.instructions()[0], op::Parameter(0));
  EXPECT_THAT(sequence.instructions()[1], op::Parameter(1));
  EXPECT_THAT(sequence.instructions()[6], op::CopyStart());
  EXPECT_THAT(sequence.instructions()[10], op::CopyDone());
}

TEST_F(MemorySpaceAssignmentTest, FilterUpdateConfigExactMatchPrecedenceTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options = DefaultMemorySpaceOptions();

  const std::string text_proto = R"pb(
    overrides {
      hlo_operand_filter { instruction_name_regex: "add" operand_number: 1 }
      override_options {
        after_instruction: { instruction_name_regex: "%?negate.1" }
      }
    }
    overrides {
      hlo_operand_filter { size_lte: 24 size_gte: 24 }
      override_options { prefetch_eagerness: 0.5 }
    }
  )pb";
  TF_ASSERT_OK_AND_ASSIGN(
      options.preferred_prefetch_overrides,
      ParseTextProto<PreferredPrefetchOverrides>(text_proto));

  AssignMemorySpace(module.get(), options);

  EXPECT_THAT(add, op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(1))));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0},
      /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1,
      /*element_size_in_bits=*/0, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
  // Ensure the CopyStart/CopyDone schedules.
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  EXPECT_THAT(sequence.instructions()[0], op::Parameter(0));
  EXPECT_THAT(sequence.instructions()[1], op::Parameter(1));
  EXPECT_THAT(sequence.instructions()[4], op::CopyStart());
  EXPECT_THAT(sequence.instructions()[10], op::CopyDone());
}

TEST_F(MemorySpaceAssignmentTest, FilterUpdatePreferredPrefetchNoMatchTest) {
  // The negate chain is long enough for asynchronous copy to be inserted
  // between p1 and add.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options = DefaultMemorySpaceOptions();

  const std::string text_proto = R"pb(
    overrides {
      hlo_operand_filter { size_lte: 24 size_gte: 25 }
      override_options { prefetch_eagerness: 0.5 }
    }
  )pb";
  TF_ASSERT_OK_AND_ASSIGN(
      options.preferred_prefetch_overrides,
      ParseTextProto<PreferredPrefetchOverrides>(text_proto));

  AssignMemorySpace(module.get(), options);

  EXPECT_THAT(add, op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(1))));
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0},
      /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1,
      /*element_size_in_bits=*/0, kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
  // Ensure the CopyStart/CopyDone schedules.
  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);
  EXPECT_THAT(sequence.instructions()[0], op::Parameter(0));
  EXPECT_THAT(sequence.instructions()[1], op::Parameter(1));
  EXPECT_THAT(sequence.instructions()[2], op::CopyStart());
  EXPECT_THAT(sequence.instructions()[10], op::CopyDone());
}

TEST_F(MemorySpaceAssignmentTest, EvictAndPrefetch) {
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get());

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Add(op::Add(),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::AsyncCopy(kDefaultMemorySpace,
                                          kAlternateMemorySpace, op::Tanh()))));
}

TEST_F(MemorySpaceAssignmentTest, EvictAndPrefetchLimitAsyncCopies0) {
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get(),
                    UpdateMaxAsyncCopies(DefaultMemorySpaceOptions(), 0));

  EXPECT_LE(CountMaximumOutstandingAsyncCopies(*module).max_prefetches, 0);
  EXPECT_LE(CountMaximumOutstandingAsyncCopies(*module).max_evictions, 0);
}

TEST_F(MemorySpaceAssignmentTest, EvictAndPrefetchLimitAsyncCopies1) {
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get(),
                    UpdateMaxAsyncCopies(DefaultMemorySpaceOptions(), 1));

  EXPECT_LE(CountMaximumOutstandingAsyncCopies(*module).max_prefetches, 1);
  EXPECT_LE(CountMaximumOutstandingAsyncCopies(*module).max_evictions, 1);
}

TEST_F(MemorySpaceAssignmentTest, EvictAndPrefetchLimitAsyncCopies2) {
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get(),
                    UpdateMaxAsyncCopies(DefaultMemorySpaceOptions(), 2));

  EXPECT_LE(CountMaximumOutstandingAsyncCopies(*module).max_prefetches, 2);
  EXPECT_LE(CountMaximumOutstandingAsyncCopies(*module).max_evictions, 2);
}

// TODO(berkin): This test is broken with some prefetch timing improvements.
TEST_F(MemorySpaceAssignmentTest,
       DISABLED_DontEvictWhenThereIsDefaultMemAllocation) {
  // This test is the same as EvictAndPrefetchLimitAsyncCopies1, except we check
  // that there is no eviction if not necessary (due to an existing allocation
  // in default memory).
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* tanh = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  // tanh should be placed in the alternate memory since there isn't much
  // contention in the beginning. However, tanh has another consumer at the end.
  // So it should be kicked out to default memory and prefetched back in.  The
  // graph below is meant to increase the contention to force eviction/prefetch
  // behavior.
  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, tanh));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kSubtract, p0, p1));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, p0, p1));
  HloInstruction* d = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kSubtract, p0, p1));
  HloInstruction* e = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, a, b));
  HloInstruction* f = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, a, c));
  HloInstruction* g = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, a, d));
  HloInstruction* h = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, b, c));
  HloInstruction* i = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, b, d));
  HloInstruction* j = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, c, d));
  HloInstruction* k = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, e, f));
  HloInstruction* l = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, g, h));
  HloInstruction* m = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, i, j));
  HloInstruction* n = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, k, l));
  HloInstruction* o = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, n, m));
  // tanh is being used at the root instruction, and this should be
  // prefetched.
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, o, tanh));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, tanh, a, b, c, d, e, f, g, h, i,
                                      j, k, l, m, n, o, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(),
                    UpdateMaxAsyncCopies(DefaultMemorySpaceOptions(), 1));

  // We expect the second argument to multiply is prefetched c.
  EXPECT_THAT(f, op::Multiply(op::Add(), op::CopyDone()));
  // We make sure that the second argument to this multiply is not evicted
  // CopyDone but is the original c.
  EXPECT_THAT(h, op::Multiply(op::Subtract(), op::Multiply()));
}

TEST_F(MemorySpaceAssignmentTest, EvictAndPrefetchAndPrefetch) {
  // Test for a memory corruption bug involving evict/prefetch/prefetch pattern,
  // where the last prefetch copied from the original buffer in alternate buffer
  // instead of evicted buffer.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* tanh = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, tanh));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kSubtract, p0, p1));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, p0, p1));
  HloInstruction* d = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kSubtract, p0, p1));
  HloInstruction* e = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, a, b));
  HloInstruction* f = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, a, c));
  HloInstruction* g = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, a, d));
  HloInstruction* h = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, b, c));
  HloInstruction* i = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, b, d));
  HloInstruction* j = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, c, d));
  HloInstruction* k = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, e, f));
  HloInstruction* l = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, g, h));
  HloInstruction* m = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, i, j));
  HloInstruction* n = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, k, l));
  HloInstruction* o = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, n, m));
  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, o, tanh));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, add0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* negate7 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate6));
  HloInstruction* negate8 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate7));
  HloInstruction* negate9 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate8));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate9, tanh));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation,
      {p0,      p1,      tanh,    a,       b,       c,       d,       e,
       f,       g,       h,       i,       j,       k,       l,       m,
       n,       o,       add0,    negate0, negate1, negate2, negate3, negate4,
       negate5, negate6, negate7, negate8, negate9, add1});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  // Check that both prefetches (add0 and add1) prefetch from the eviction
  // instead of tanh, which will be placed in the alternate memory directly.
  EXPECT_THAT(
      add0,
      op::Add(op::Add(),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::AsyncCopy(kDefaultMemorySpace,
                                          kAlternateMemorySpace, op::Tanh()))));
  EXPECT_THAT(
      add1,
      op::Add(op::Negate(),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::AsyncCopy(kDefaultMemorySpace,
                                          kAlternateMemorySpace, op::Tanh()))));
}

TEST_F(MemorySpaceAssignmentTest, While) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(xla::F32, {2, 3});
  Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, scalar_shape});

  auto cond_builder = HloComputation::Builder("WhileCond");
  // Tuple param: 24 bytes (each elem has 8 byte pointer, 4 byte element)
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_param"));
  HloInstruction* cond_iter = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, cond_param, 1));
  HloInstruction* cond_limit = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(50.f)));
  // Free cond_param[] (16 bytes), Alloc PRED[] (1 byte)
  HloInstruction* cond_lt = cond_builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), cond_iter,
                                    cond_limit, ComparisonDirection::kLt));
  HloComputation* cond_computation =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto body_builder = HloComputation::Builder("WhileBody");
  // Tuple param: 24 bytes (each elem has 8 byte pointer, 4 byte element)
  HloInstruction* body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_param"));
  HloInstruction* body_iter = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, body_param, 1));
  HloInstruction* body_data = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 0));
  HloInstruction* body_iter_increment = body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.f)));
  HloInstruction* body_iter_next =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          scalar_shape, HloOpcode::kAdd, body_iter, body_iter_increment));
  HloInstruction* body_data_increment =
      body_builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<float>({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}})));
  HloInstruction* body_data_mul =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kMultiply, body_data, body_data));
  HloInstruction* body_data_add =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, body_data, body_data_increment));
  HloInstruction* body_data_next =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, body_data_add, body_data_mul));
  HloInstruction* body_out = body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_data_next, body_iter_next}));
  HloComputation* body_computation =
      module->AddEmbeddedComputation(body_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* data = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param_iter"));
  HloInstruction* iter = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "param_data"));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({data, iter}));
  HloInstruction* while_op = builder.AddInstruction(HloInstruction::CreateWhile(
      tuple_shape, cond_computation, body_computation, tuple));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(cond_computation,
                        {cond_param, cond_iter, cond_limit, cond_lt});
  schedule.set_sequence(body_computation,
                        {body_param, body_iter, body_data, body_iter_increment,
                         body_iter_next, body_data_increment, body_data_mul,
                         body_data_add, body_data_next, body_out});
  schedule.set_sequence(entry_computation, {iter, data, tuple, while_op});
  TF_CHECK_OK(module->set_schedule(schedule));

  LOG(INFO) << module->ToString(HloPrintOptions::ShortParsable());

  AssignMemorySpace(module.get());

  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
  EXPECT_THAT(body_data_mul, op::ShapeWithLayout(shape_in_alternate_mem));
}

TEST_F(MemorySpaceAssignmentTest, Tuple) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape inner_tuple_shape = ShapeUtil::MakeTupleShape({shape});
  Shape tuple_shape =
      ShapeUtil::MakeTupleShape({shape, shape, inner_tuple_shape});
  HloInstruction* p = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p, 0));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p, 1));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));
  HloInstruction* p2 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(inner_tuple_shape, p, 2));
  HloInstruction* p2_0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p2, 0));
  HloInstruction* mul = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, add, p2_0));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation, {p, p0, negate0, negate1, negate2, negate3, negate4, negate5,
                    negate6, p1, add, p2, p2_0, mul});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  EXPECT_THAT(
      mul,
      op::Multiply(op::Add(op::Negate(), op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::GetTupleElement())),
                   op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                 op::GetTupleElement(op::GetTupleElement()))));
}

TEST_F(MemorySpaceAssignmentTest, Bitcast) {
  // Bitcasts can cause the position in the alternate memory to appear multiple
  // times in the preset assignments. This test ensure the preset assignments
  // refer to unique positions.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape param_shape = ShapeUtil::MakeShape(F32, {6});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "p1"));
  HloInstruction* negate = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* bitcast = builder.AddInstruction(
      HloInstruction::CreateBitcast(param_shape, negate));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(param_shape, HloOpcode::kAdd, bitcast, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate, bitcast, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  bitcast = add->mutable_operand(0);
  EXPECT_EQ(bitcast->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(bitcast->shape().layout().memory_space(), kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, Bitcast2) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape param_shape = ShapeUtil::MakeShape(F32, {6});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape, p1));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, bitcast, negate4));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, bitcast, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  EXPECT_EQ(add->operand(0)->shape().layout().memory_space(),
            kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, Bitcast3) {
  HloComputation::Builder builder(TestName());
  Shape shape1 = ShapeUtil::MakeShape(F32, {2, 3});
  Shape shape2 = ShapeUtil::MakeShape(F32, {3, 2});
  Shape shape3 = ShapeUtil::MakeShape(F32, {1, 6});
  Shape param_shape = ShapeUtil::MakeShape(F32, {6});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape1, "p0"));
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate3));
  HloInstruction* bitcast1 =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape1, p1));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape1, HloOpcode::kAdd, bitcast1, negate4));
  HloInstruction* bitcast2 =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape3, p1));
  HloInstruction* bitcast3 =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape2, bitcast2));
  HloInstruction* bitcast4 =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape2, add));
  HloInstruction* mul = builder.AddInstruction(HloInstruction::CreateBinary(
      shape2, HloOpcode::kMultiply, bitcast3, bitcast4));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation,
                        {p0, p1, negate0, negate1, negate2, negate3, negate4,
                         bitcast1, add, bitcast2, bitcast3, bitcast4, mul});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  // We expect one bitcast on the LHS of multiply since bitcast(bitcast(foo)) is
  // converted to bitcast(foo).
  EXPECT_THAT(
      mul,
      op::Multiply(
          op::Bitcast(op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                    op::Parameter(1))),
          op::Bitcast(op::Add(
              op::Bitcast(op::AsyncCopy(kAlternateMemorySpace,
                                        kDefaultMemorySpace, op::Parameter(1))),
              op::Negate()))));
  EXPECT_EQ(add->operand(0)->shape().layout().memory_space(),
            kAlternateMemorySpace);
  EXPECT_EQ(add->shape().layout().memory_space(), kAlternateMemorySpace);
  // bitcast2 will no longer have a consumer and should get DCE'd, so we don't
  // care about its memory space.
  EXPECT_EQ(mul->operand(0)->shape().layout().memory_space(),
            kAlternateMemorySpace);
  EXPECT_EQ(mul->operand(1)->shape().layout().memory_space(),
            kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, BitcastTuple) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape param_shape = ShapeUtil::MakeShape(F32, {6});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});

  auto module = CreateNewVerifiedModule();
  HloComputation::Builder fusion_builder("fusion");
  HloInstruction* fusion_param = fusion_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* fusion_element0 = fusion_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion_param, 0));
  HloInstruction* fusion_element1 = fusion_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion_param, 1));
  fusion_builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, fusion_element0, fusion_element1));
  HloComputation* fusion_computation =
      module->AddEmbeddedComputation(fusion_builder.Build());

  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape, p1));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({bitcast, p0}));
  HloInstruction* fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kCustom, {tuple}, fusion_computation));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation,
                        {p0, p1, negate0, negate1, negate2, negate3, negate4,
                         bitcast, tuple, fusion});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, BitcastGetTupleElementTuple) {
  // This test pattern was encountered in
  // //third_party/tensorflow/compiler/xla/tests:slice_test and was causing a
  // breakage when there is a GetTupleElement(Tuple(Bitcast())) pattern. Also
  // added a GetTupleElement(GetTupleElement(Tuple(Tuple(Bitcast())))) pattern.
  absl::string_view hlo_string = R"(
  HloModule DoIt_S64_10_0_5_1.3, is_scheduled=true

  ENTRY %DoIt_S64_10_0_5_1.3 (p0.1: (u32[10], u32[10])) -> (u32[5], u32[5]) {
    %p0.1 = (u32[10]{0:T(128)}, u32[10]{0:T(128)}) parameter(0)
    %get-tuple-element.1 = u32[10]{0:T(128)} get-tuple-element((u32[10]{0:T(128)}, u32[10]{0:T(128)}) %p0.1), index=1
    %bitcast.1 = u32[5]{0:T(128)} bitcast(u32[10]{0:T(128)} %get-tuple-element.1)
    %get-tuple-element = u32[10]{0:T(128)} get-tuple-element((u32[10]{0:T(128)}, u32[10]{0:T(128)}) %p0.1), index=0
    %bitcast = u32[5]{0:T(128)} bitcast(u32[10]{0:T(128)} %get-tuple-element)
    %tuple.1 = (u32[5]{0:T(128)}, u32[5]{0:T(128)}) tuple(u32[5]{0:T(128)} %bitcast, u32[5]{0:T(128)} %bitcast.1)
    %tuple.3 = ((u32[5]{0:T(128)}, u32[5]{0:T(128)}), (u32[5]{0:T(128)}, u32[5]{0:T(128)})) tuple(%tuple.1, %tuple.1)
    %get-tuple-element.4 = u32[5]{0:T(128)} get-tuple-element((u32[5]{0:T(128)}, u32[5]{0:T(128)}) %tuple.1), index=0
    %get-tuple-element.5 = (u32[5]{0:T(128)}, u32[5]{0:T(128)}) get-tuple-element(%tuple.3), index=0
    %get-tuple-element.6 = u32[5]{0:T(128)} get-tuple-element((u32[5]{0:T(128)}, u32[5]{0:T(128)}) %get-tuple-element.5), index=1
    %copy.2 = u32[5]{0:T(128)} copy(u32[5]{0:T(128)} %get-tuple-element.4)
    %copy.3 = u32[5]{0:T(128)} copy(u32[5]{0:T(128)} %get-tuple-element.6)
    ROOT %tuple.2 = (u32[5]{0:T(128)}, u32[5]{0:T(128)}) tuple(u32[5]{0:T(128)} %copy.2, u32[5]{0:T(128)} %copy.3)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, GetSimplifiedOperandBug) {
  // Test case for a bug finding Bitcasts in GTE(Tuple(...)) pattern.
  absl::string_view hlo_string = R"(
  HloModule sort.16, is_scheduled=true

  ENTRY %sort.16 (param.0.1: s32[1], param.1.2: f32[1], param.2.3: u32[1], param.3.4: s32[1]) -> (s32[1], f32[1], u32[1], s32[1]) {
    %param.3.4 = s32[1]{0:T(128)} parameter(3)
    %param.2.3 = u32[1]{0:T(128)} parameter(2)
    %param.1.2 = f32[1]{0:T(128)} parameter(1)
    %param.0.1 = s32[1]{0:T(128)} parameter(0)
    %tuple.1 = (s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) tuple(s32[1]{0:T(128)} %param.0.1, f32[1]{0:T(128)} %param.1.2, u32[1]{0:T(128)} %param.2.3, s32[1]{0:T(128)} %param.3.4)
    %get-tuple-element.4 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=0
    %get-tuple-element.5 = f32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=1
    %get-tuple-element.6 = u32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=2
    %get-tuple-element.7 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=3
    %copy.4 = s32[1]{0:T(128)} copy(s32[1]{0:T(128)} %get-tuple-element.4)
    %copy.5 = f32[1]{0:T(128)} copy(f32[1]{0:T(128)} %get-tuple-element.5)
    %copy.6 = u32[1]{0:T(128)} copy(u32[1]{0:T(128)} %get-tuple-element.6)
    %copy.7 = s32[1]{0:T(128)} copy(s32[1]{0:T(128)} %get-tuple-element.7)
    ROOT %tuple.2 = (s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) tuple(s32[1]{0:T(128)} %copy.4, f32[1]{0:T(128)} %copy.5, u32[1]{0:T(128)} %copy.6, s32[1]{0:T(128)} %copy.7)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, BitcastMultiUse) {
  // When there is a pattern where a bitcast has multiple uses (negate0 and add)
  // and one is in the default memory and the other is in alternate memory, they
  // both need their own bitcast.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape param_shape = ShapeUtil::MakeShape(F32, {6});
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "p1"));
  HloInstruction* bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape, p0));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, bitcast));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, bitcast, negate4));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, bitcast, negate0, negate1, negate2,
                                      negate3, negate4, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
  EXPECT_THAT(negate0->operand(0), op::ShapeWithLayout(shape));
  EXPECT_THAT(add->operand(0), op::ShapeWithLayout(shape_in_alternate_mem));
}

TEST_F(MemorySpaceAssignmentTest, BitcastMultiUseTuple) {
  // Same as BitcastMultUse but the second use is a tuple.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape param_shape = ShapeUtil::MakeShape(F32, {6});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});

  auto module = CreateNewVerifiedModule();
  HloComputation::Builder fusion_builder("fusion");
  HloInstruction* fusion_param = fusion_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* fusion_element0 = fusion_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion_param, 0));
  HloInstruction* fusion_element1 = fusion_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion_param, 1));
  fusion_builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, fusion_element0, fusion_element1));
  HloComputation* fusion_computation =
      module->AddEmbeddedComputation(fusion_builder.Build());

  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "p1"));
  HloInstruction* bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape, p0));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, bitcast));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({bitcast, negate4}));
  HloInstruction* fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kCustom, {tuple}, fusion_computation));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, bitcast, negate0, negate1, negate2,
                                      negate3, negate4, tuple, fusion});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
  EXPECT_THAT(negate0->operand(0), op::ShapeWithLayout(shape));
  EXPECT_THAT(fusion->operand(0)->operand(0),
              op::ShapeWithLayout(shape_in_alternate_mem));
}

TEST_F(MemorySpaceAssignmentTest, BitcastScheduleBug) {
  // Bitcasts can force asynchronous copies to be scheduled too early, possibly
  // leading to memory corruption.
  //  Bug:
  //    p0------------------>neg-->neg-->neg ... -->neg-->neg-->neg->add
  //                                                                 /
  //    p1->cs->cd->bitcast-----------------------------------------+
  //
  //  Expected:
  //    p0-->neg-->neg-->neg ... -->neg-->neg-->neg------------->add
  //                                                             /
  //    p1--------------------->cs----------------->cd->bitcast-+
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape param_shape = ShapeUtil::MakeShape(F32, {6});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "p1"));
  HloInstruction* bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape, p1));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* negate7 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate6));
  HloInstruction* negate8 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate7));
  HloInstruction* negate9 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate8));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, bitcast, negate9));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation, {p0, p1, bitcast, negate0, negate1, negate2, negate3,
                    negate4, negate5, negate6, negate7, negate8, negate9, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), DefaultMemorySpaceOptions(),
                    /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/4);

  EXPECT_EQ(add->operand(0)->shape().layout().memory_space(),
            kAlternateMemorySpace);
  const auto& instructions =
      module->schedule().sequence(module->entry_computation()).instructions();
  for (int i = 0; i < instructions.size(); ++i) {
    // Expect that there is a negate before and after the CopyStart and there is
    // a negate before CopyDone.
    if (instructions.at(i)->opcode() == HloOpcode::kCopyStart) {
      EXPECT_EQ(instructions.at(i - 1)->opcode(), HloOpcode::kNegate);
      EXPECT_EQ(instructions.at(i + 1)->opcode(), HloOpcode::kNegate);
    } else if (instructions.at(i)->opcode() == HloOpcode::kCopyDone) {
      EXPECT_EQ(instructions.at(i - 1)->opcode(), HloOpcode::kNegate);
    }
  }
}

TEST_F(MemorySpaceAssignmentTest, AddDependency) {
  // Make sure add-dependency is not optimized away.
  absl::string_view hlo_string = R"(
  HloModule AddDependency, is_scheduled=true

  ENTRY %AddDependency (p: f32[3]) -> f32[3] {
    %p = f32[3]{0} parameter(0)
    %neg0 = f32[3]{0} negate(f32[3]{0} %p)
    %neg1 = f32[3]{0} negate(f32[3]{0} %neg0)
    %neg2 = f32[3]{0} negate(f32[3]{0} %neg1)
    %neg3 = f32[3]{0} negate(f32[3]{0} %neg2)
    %neg4 = f32[3]{0} negate(f32[3]{0} %neg3)
    %neg5 = f32[3]{0} negate(f32[3]{0} %neg4)
    %neg6 = f32[3]{0} negate(f32[3]{0} %neg5)
    %token0 = token[] after-all()
    %add_dep = f32[3]{0} add-dependency(f32[3]{0} %p, token[] %token0)
    ROOT %add = f32[3]{0} add(f32[3]{0} %add_dep, f32[3]{0} %neg6)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Add(op::AddDependency(), op::Negate()));
}

TEST_F(MemorySpaceAssignmentTest, WhileAllocationBug) {
  // This test is carefully crafted to include two multiply ops sized [4,3] in a
  // while body. For testing purposes, we have provided a BufferIntervalCompare
  // such that first multiply, then tanh, then other HloValues will be
  // allocated. The memory is sized just enough to fit two [4,3] buffers.
  // Because the multiplies in the while body are going to be allocated in the
  // alternate memory first, the tanh that is fed inside the while loop should
  // not be placed in the alternate memory. Otherwise, we will corrupt memory.
  absl::string_view hlo_string = R"(
  HloModule WhileAllocationBug, is_scheduled=true

  %WhileBody (body_param: (f32[4,3], f32[])) -> (f32[4,3], f32[]) {
    %body_param = (f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element.1 = f32[] get-tuple-element((f32[4,3]{1,0}, f32[]) %body_param), index=1
    %get-tuple-element.2 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[]) %body_param), index=0
    %constant.1 = f32[] constant(1)
    %add = f32[] add(f32[] %get-tuple-element.1, f32[] %constant.1)
    %constant.2 = f32[4,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 }, { 1, 2, 3 }, { 4, 5, 6 } })
    %multiply = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %get-tuple-element.2, f32[4,3]{1,0} %get-tuple-element.2)
    %multiply2 = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %multiply, f32[4,3]{1,0} %multiply)
    %add.1 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.2, f32[4,3]{1,0} %constant.2)
    %add.2 = f32[4,3]{1,0} add(f32[4,3]{1,0} %add.1, f32[4,3]{1,0} %multiply2)
    ROOT %tuple = (f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} %add.2, f32[] %add)
  }

  %WhileCond (cond_param: (f32[4,3], f32[])) -> pred[] {
    %cond_param = (f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element = f32[] get-tuple-element((f32[4,3]{1,0}, f32[]) %cond_param), index=1
    %constant = f32[] constant(50)
    ROOT %compare = pred[] compare(f32[] %get-tuple-element, f32[] %constant), direction=LT
  }

  ENTRY %Entry (param_iter: f32[4,3], param_data: f32[], p2: f32[4,3]) -> f32[4,3] {
    %param_data = f32[] parameter(1)
    %param_iter = f32[4,3]{1,0} parameter(0)
    %p2 = f32[4,3]{1,0} parameter(2)
    %tanh = f32[4,3]{1,0} tanh(f32[4,3]{1,0} %param_iter)
    %neg0 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %p2)
    %neg1 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg0)
    %neg2 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg1)
    %neg3 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg2)
    %neg4 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg3)
    %neg5 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg4)
    %neg6 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg5)
    %add.4 = f32[4,3]{1,0} add(f32[4,3]{1,0} %neg6, f32[4,3]{1,0} %tanh)
    %tuple.1 = (f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} %tanh, f32[] %param_data)
    %while = (f32[4,3]{1,0}, f32[]) while((f32[4,3]{1,0}, f32[]) %tuple.1), condition=%WhileCond, body=%WhileBody
    %get-tuple-element.3 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[]) %while), index=0
    ROOT %add.3 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.3, f32[4,3]{1,0} %add.4)
  }
  )";

  MsaBufferIntervalCompare buffer_interval_compare =
      [](const MsaBufferInterval& a, const MsaBufferInterval& b) {
        bool a_is_mul =
            a.buffer->defining_instruction()->opcode() == HloOpcode::kMultiply;
        bool b_is_mul =
            b.buffer->defining_instruction()->opcode() == HloOpcode::kMultiply;
        if (a_is_mul && !b_is_mul) {
          return true;
        }
        if (!a_is_mul && b_is_mul) {
          return false;
        }
        bool a_is_tanh =
            a.buffer->defining_instruction()->opcode() == HloOpcode::kTanh;
        bool b_is_tanh =
            b.buffer->defining_instruction()->opcode() == HloOpcode::kTanh;
        if (a_is_tanh && !b_is_tanh) {
          return true;
        }
        if (!a_is_tanh && b_is_tanh) {
          return false;
        }
        return a.buffer->id() < b.buffer->id();
      };
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 10);
  AssignMemorySpace(module.get(), DefaultMemorySpaceOptions(),
                    buffer_interval_compare, &prefetch_interval_picker);

  for (const HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kWhile) {
      const Shape& while_subshape =
          ShapeUtil::GetSubshape(instruction->shape(), {0});
      // We expect shape {0} to either be in default memory for the entire while
      // loop or there has to be an eviction within the while loop.
      if (while_subshape.layout().memory_space() == kAlternateMemorySpace) {
        const HloInstruction* body_param =
            instruction->while_body()->parameter_instruction(0);
        const HloInstruction* gte = nullptr;
        for (const HloInstruction* user : body_param->users()) {
          if (user->opcode() == HloOpcode::kGetTupleElement &&
              user->tuple_index() == 0) {
            gte = user;
            break;
          }
        }
        EXPECT_NE(gte, nullptr);
        const HloInstruction* copy_start = nullptr;
        for (const HloInstruction* user : gte->users()) {
          if (user->opcode() == HloOpcode::kCopyStart) {
            copy_start = user;
            break;
          }
        }
        EXPECT_NE(copy_start, nullptr);
        const Shape& copy_start_subshape =
            ShapeUtil::GetSubshape(copy_start->shape(), {0});

        EXPECT_NE(copy_start_subshape.layout().memory_space(),
                  kAlternateMemorySpace);
      }
    }
  }
}

TEST_F(MemorySpaceAssignmentTest, ConsecutiveWhileLoops) {
  absl::string_view hlo_string = R"(
  HloModule WhileAllocationBug, is_scheduled=true

  %WhileBody (body_param: (f32[4,3], f32[4,3], f32[])) -> (f32[4,3], f32[4,3], f32[]) {
    %body_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element.1 = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=2
    %get-tuple-element.2 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=0
    %get-tuple-element.3 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=1
    %constant.1 = f32[] constant(1)
    %add = f32[] add(f32[] %get-tuple-element.1, f32[] %constant.1)
    %constant.2 = f32[4,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 }, { 1, 2, 3 }, { 4, 5, 6 } })
    %multiply = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %get-tuple-element.2, f32[4,3]{1,0} %get-tuple-element.3)
    %multiply2 = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %multiply, f32[4,3]{1,0} %multiply)
    %add.1 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.2, f32[4,3]{1,0} %constant.2)
    %add.2 = f32[4,3]{1,0} add(f32[4,3]{1,0} %add.1, f32[4,3]{1,0} %multiply2)
    ROOT %tuple = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} %add.2, f32[4,3]{1,0} %get-tuple-element.3, f32[] %add)
  }

  %WhileCond (cond_param: (f32[4,3], f32[4,3], f32[])) -> pred[] {
    %cond_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %cond_param), index=2
    %constant = f32[] constant(50)
    ROOT %compare = pred[] compare(f32[] %get-tuple-element, f32[] %constant), direction=LT
  }

  %WhileBody2 (body_param: (f32[4,3], f32[4,3], f32[])) -> (f32[4,3], f32[4,3], f32[]) {
    %body_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element.1 = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=2
    %get-tuple-element.2 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=0
    %get-tuple-element.3 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=1
    %constant.1 = f32[] constant(1)
    %add = f32[] add(f32[] %get-tuple-element.1, f32[] %constant.1)
    %constant.2 = f32[4,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 }, { 1, 2, 3 }, { 4, 5, 6 } })
    %multiply = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %get-tuple-element.2, f32[4,3]{1,0} %get-tuple-element.3)
    %multiply2 = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %multiply, f32[4,3]{1,0} %multiply)
    %add.1 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.2, f32[4,3]{1,0} %constant.2)
    %add.2 = f32[4,3]{1,0} add(f32[4,3]{1,0} %add.1, f32[4,3]{1,0} %multiply2)
    ROOT %tuple = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} %add.2, f32[4,3]{1,0} %get-tuple-element.3, f32[] %add)
  }

  %WhileCond2 (cond_param: (f32[4,3], f32[4,3], f32[])) -> pred[] {
    %cond_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %cond_param), index=2
    %constant = f32[] constant(50)
    ROOT %compare = pred[] compare(f32[] %get-tuple-element, f32[] %constant), direction=LT
  }

  ENTRY %Entry (param_data: f32[4,3], param_iter: f32[], p2: f32[4,3]) -> f32[4,3] {
    %param_iter = f32[] parameter(1)
    %param_data = f32[4,3]{1,0} parameter(0)
    %p2 = f32[4,3]{1,0} parameter(2)
    %neg0 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %p2)
    %neg1 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg0)
    %neg2 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg1)
    %neg3 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg2)
    %neg4 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg3)
    %neg5 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg4)
    %neg6 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg5)
    %add.4 = f32[4,3]{1,0} add(f32[4,3]{1,0} %neg6, f32[4,3]{1,0} %p2)
    %tuple.1 = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} add.4, f32[4,3]{1,0} param_data, f32[] %param_iter)
    %while = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) while((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %tuple.1), condition=%WhileCond, body=%WhileBody
    %get-tuple-element.4 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while), index=0
    %add.3 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.4, f32[4,3]{1,0} %add.4)
    %get-tuple-element.5 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while), index=1
    %tuple.2 = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} add.3, f32[4,3]{1,0} get-tuple-element.5, f32[] %param_iter)
    %while.1 = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) while((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %tuple.2), condition=%WhileCond2, body=%WhileBody2
    %get-tuple-element.6 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while.1), index=0
    ROOT %add.5 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.6, f32[4,3]{1,0} %add.3)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, WhileLiveRangeBug) {
  // Tests against while live ranges being incorrect and the verifier
  // complaining about a conflict.
  absl::string_view hlo_string = R"(
  HloModule WhileAllocationBug, is_scheduled=true

  %WhileBody (body_param: (f32[4,3], f32[4,3], f32[])) -> (f32[4,3], f32[4,3], f32[]) {
    %body_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element.1 = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=2
    %get-tuple-element.2 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=0
    %get-tuple-element.3 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=1
    %neg10 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %get-tuple-element.2)
    %neg11 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg10)
    %neg12 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg11)
    %neg13 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg12)
    %neg14 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg13)
    %neg15 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg14)
    %neg16 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg15)
    %neg17 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg16)
    %neg18 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg17)
    %neg19 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg18)
    %neg20 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg19)
    %constant.1 = f32[] constant(1)
    %add = f32[] add(f32[] %get-tuple-element.1, f32[] %constant.1)
    %constant.2 = f32[4,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 }, { 1, 2, 3 }, { 4, 5, 6 } })
    %multiply = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %neg20, f32[4,3]{1,0} %neg20)
    %multiply2 = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %multiply, f32[4,3]{1,0} %multiply)
    %add.1 = f32[4,3]{1,0} add(f32[4,3]{1,0} get-tuple-element.3, f32[4,3]{1,0} %constant.2)
    %add.2 = f32[4,3]{1,0} add(f32[4,3]{1,0} %add.1, f32[4,3]{1,0} %multiply2)
    ROOT %tuple = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} %add.2, f32[4,3]{1,0} %get-tuple-element.3, f32[] %add)
  }

  %WhileCond (cond_param: (f32[4,3], f32[4,3], f32[])) -> pred[] {
    %cond_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %cond_param), index=2
    %constant = f32[] constant(50)
    ROOT %compare = pred[] compare(f32[] %get-tuple-element, f32[] %constant), direction=LT
  }

  ENTRY %Entry (param_data: f32[4,3], param_iter: f32[], p2: f32[4,3]) -> f32[4,3] {
    %param_iter = f32[] parameter(1)
    %param_data = f32[4,3]{1,0} parameter(0)
    %p2 = f32[4,3]{1,0} parameter(2)
    %neg0 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %p2)
    %neg1 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg0)
    %neg2 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg1)
    %neg3 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg2)
    %neg4 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg3)
    %neg5 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg4)
    %neg6 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg5)
    %add.4 = f32[4,3]{1,0} add(f32[4,3]{1,0} %neg6, f32[4,3]{1,0} %p2)
    %tuple.1 = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} add.4, f32[4,3]{1,0} param_data, f32[] %param_iter)
    %while = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) while((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %tuple.1), condition=%WhileCond, body=%WhileBody
    %get-tuple-element.4 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while), index=0
    %get-tuple-element.5 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while), index=1
    %add.3 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.4, f32[4,3]{1,0} %add.4)
    ROOT %add.5 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.5, f32[4,3]{1,0} %add.3)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, ConsecutiveWhileLoopsOneBuffer) {
  // Tests against a bug when there are consecutive while loops with one buffer
  // (the value doesn't change in the buffer), the parameter can be colored in
  // the alternate memory space.
  absl::string_view hlo_string = R"(
  HloModule WhileAllocationBug, is_scheduled=true

  %WhileBody (body_param: (f32[4,3], f32[4,3], f32[])) -> (f32[4,3], f32[4,3], f32[]) {
    %body_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element.1 = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=2
    %get-tuple-element.2 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=0
    %get-tuple-element.3 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=1
    %neg10 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %get-tuple-element.2)
    %neg11 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg10)
    %neg12 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg11)
    %neg13 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg12)
    %neg14 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg13)
    %neg15 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg14)
    %neg16 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg15)
    %neg17 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg16)
    %neg18 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg17)
    %neg19 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg18)
    %neg20 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg19)
    %constant.1 = f32[] constant(1)
    %add = f32[] add(f32[] %get-tuple-element.1, f32[] %constant.1)
    %constant.2 = f32[4,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 }, { 1, 2, 3 }, { 4, 5, 6 } })
    %multiply = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %neg20, f32[4,3]{1,0} %neg20)
    %multiply2 = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %multiply, f32[4,3]{1,0} %multiply)
    %add.1 = f32[4,3]{1,0} add(f32[4,3]{1,0} get-tuple-element.3, f32[4,3]{1,0} %constant.2)
    %add.2 = f32[4,3]{1,0} add(f32[4,3]{1,0} %add.1, f32[4,3]{1,0} %multiply2)
    ROOT %tuple = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} %add.2, f32[4,3]{1,0} %get-tuple-element.3, f32[] %add)
  }

  %WhileCond (cond_param: (f32[4,3], f32[4,3], f32[])) -> pred[] {
    %cond_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %cond_param), index=2
    %constant = f32[] constant(50)
    ROOT %compare = pred[] compare(f32[] %get-tuple-element, f32[] %constant), direction=LT
  }

  %WhileBody2 (body_param: (f32[4,3], f32[4,3], f32[])) -> (f32[4,3], f32[4,3], f32[]) {
    %body_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element.1 = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=2
    %get-tuple-element.2 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=0
    %get-tuple-element.3 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %body_param), index=1
    %neg10 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %get-tuple-element.2)
    %neg11 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg10)
    %neg12 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg11)
    %neg13 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg12)
    %neg14 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg13)
    %neg15 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg14)
    %neg16 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg15)
    %neg17 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg16)
    %neg18 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg17)
    %neg19 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg18)
    %neg20 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg19)
    %constant.1 = f32[] constant(1)
    %add = f32[] add(f32[] %get-tuple-element.1, f32[] %constant.1)
    %constant.2 = f32[4,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 }, { 1, 2, 3 }, { 4, 5, 6 } })
    %multiply = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %neg20, f32[4,3]{1,0} %neg20)
    %multiply2 = f32[4,3]{1,0} multiply(f32[4,3]{1,0} %multiply, f32[4,3]{1,0} %multiply)
    %add.1 = f32[4,3]{1,0} add(f32[4,3]{1,0} get-tuple-element.3, f32[4,3]{1,0} %constant.2)
    %add.2 = f32[4,3]{1,0} add(f32[4,3]{1,0} %add.1, f32[4,3]{1,0} %multiply2)
    ROOT %tuple = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} %add.2, f32[4,3]{1,0} %get-tuple-element.3, f32[] %add)
  }

  %WhileCond2 (cond_param: (f32[4,3], f32[4,3], f32[])) -> pred[] {
    %cond_param = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element = f32[] get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %cond_param), index=2
    %constant = f32[] constant(50)
    ROOT %compare = pred[] compare(f32[] %get-tuple-element, f32[] %constant), direction=LT
  }

  ENTRY %Entry (param_data: f32[4,3], param_iter: f32[], p2: f32[4,3]) -> f32[4,3] {
    %param_iter = f32[] parameter(1)
    %param_data = f32[4,3]{1,0} parameter(0)
    %p2 = f32[4,3]{1,0} parameter(2)
    %neg0 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %p2)
    %neg1 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg0)
    %neg2 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg1)
    %neg3 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg2)
    %neg4 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg3)
    %neg5 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg4)
    %neg6 = f32[4,3]{1,0} negate(f32[4,3]{1,0} %neg5)
    %add.4 = f32[4,3]{1,0} add(f32[4,3]{1,0} %neg6, f32[4,3]{1,0} %p2)
    %tuple.1 = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} add.4, f32[4,3]{1,0} param_data, f32[] %param_iter)
    %while = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) while((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %tuple.1), condition=%WhileCond, body=%WhileBody
    %get-tuple-element.4 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while), index=0
    %add.3 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.4, f32[4,3]{1,0} %add.4)
    %tuple.2 = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) tuple(f32[4,3]{1,0} add.3, f32[4,3]{1,0} param_data, f32[] %param_iter)
    %while.1 = (f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) while((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %tuple.2), condition=%WhileCond2, body=%WhileBody2
    %get-tuple-element.5 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while.1), index=0
    %get-tuple-element.6 = f32[4,3]{1,0} get-tuple-element((f32[4,3]{1,0}, f32[4,3]{1,0}, f32[]) %while.1), index=1
    ROOT %add.5 = f32[4,3]{1,0} add(f32[4,3]{1,0} %get-tuple-element.5, f32[4,3]{1,0} %get-tuple-element.6)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, WhileCondAliasBug) {
  // While loop is the root of the entry computation. We should ensure the
  // output of the entry computation remains to be in default memory space.
  // Test from //third_party/tensorflow/compiler/xla/tests:while_test
  // WhileTest.WhileWithPrngScalarResult.
  absl::string_view hlo_string = R"(
  HloModule WhileWithPrngScalarResult.18, is_scheduled=true

  %fused_computation (param_0.1: s32[6], param_1.3: s32[1], param_2.3: s32[5]) -> s32[6] {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    ROOT %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
  }

  %body.3 (prev.4: s32[6]) -> s32[6] {
    %constant.7 = s32[]{:T(128)} constant(100)
    %constant.6 = s32[]{:T(128)} constant(0)
    %constant.5 = s32[1]{0:T(128)} constant({1})
    %prev.4 = s32[6]{0:T(128)} parameter(0)
    %rng.8 = s32[5]{0:T(128)} rng(s32[]{:T(128)} %constant.6, s32[]{:T(128)} %constant.7), distribution=rng_uniform
    %neg = s32[1]{0:T(128)} negate(s32[1]{0:T(128)} %constant.5)
    ROOT %fusion = s32[6]{0:T(128)} fusion(s32[6]{0:T(128)} %prev.4, s32[1]{0:T(128)} %neg, s32[5]{0:T(128)} %rng.8), kind=kLoop, calls=%fused_computation
  }

  %WhileWithPrngScalarResult.11 (prev.12: s32[6]) -> pred[] {
    %constant.15 = s32[]{:T(128)} constant(1)
    %prev.12 = s32[6]{0:T(128)} parameter(0)
    %bitcast.1 = s32[1]{0:T(128)} bitcast(s32[6]{0:T(128)} %prev.12)
    %bitcast = s32[]{:T(128)} bitcast(s32[1]{0:T(128)} %bitcast.1)
    ROOT %compare.16 = pred[]{:T(128)} compare(s32[]{:T(128)} %constant.15, s32[]{:T(128)} %bitcast), direction=GT
  }

  ENTRY %WhileWithPrngScalarResult.18 () -> s32[6] {
    %constant.1 = s32[]{:T(128)} constant(0)
    %broadcast.2 = s32[6]{0:T(128)} broadcast(s32[]{:T(128)} %constant.1), dimensions={}
    ROOT %while.17 = s32[6]{0:T(128)} while(s32[6]{0:T(128)} %broadcast.2), condition=%WhileWithPrngScalarResult.11, body=%body.3
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, WhileInPlaceBuffer) {
  // Ensure that a dynamic update slice within a while loop is able to get an
  // alternate memory allocation.
  absl::string_view hlo_string = R"(
  HloModule Module, is_scheduled=true

  fused_computation {
    param0 = f32[2,3] parameter(0)
    constant.1 = f32[] constant(0)
    broadcast = f32[2,1] broadcast(constant.1), dimensions={}
    constant.3 = s32[] constant(0)
    ROOT dynamic-update-slice.5 = f32[2,3] dynamic-update-slice(param0, broadcast, constant.3, constant.3)
  }

  %WhileBody (body_param: (f32[2,3], f32[2,3], f32[])) -> (f32[2,3], f32[2,3], f32[]) {
    %body_param = (f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element.1 = f32[] get-tuple-element((f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) %body_param), index=2
    %get-tuple-element.2 = f32[2,3]{1,0} get-tuple-element((f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) %body_param), index=0
    %get-tuple-element.3 = f32[2,3]{1,0} get-tuple-element((f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) %body_param), index=1
    %fusion = f32[2,3]{1,0} fusion(get-tuple-element.3), kind=kLoop, calls=fused_computation
    %multiply = f32[2,3]{1,0} multiply(f32[2,3]{1,0} %get-tuple-element.2, f32[2,3]{1,0} %fusion)
    ROOT %tuple = (f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) tuple(f32[2,3]{1,0} %multiply, f32[2,3]{1,0} %fusion, f32[] %get-tuple-element.1)
  }

  %WhileCond (cond_param: (f32[2,3], f32[2,3], f32[])) -> pred[] {
    %cond_param = (f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) parameter(0)
    %get-tuple-element = f32[] get-tuple-element((f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) %cond_param), index=2
    %constant = f32[] constant(50)
    ROOT %compare = pred[] compare(f32[] %get-tuple-element, f32[] %constant), direction=LT
  }

  ENTRY %Entry (param_data: f32[2,3], param_iter: f32[], p2: f32[2,3]) -> f32[2,3] {
    %param_iter = f32[] parameter(1)
    %param_data = f32[2,3]{1,0} parameter(0)
    %p2 = f32[2,3]{1,0} parameter(2)
    %copy1 = f32[2,3]{1,0} copy(param_data)
    %copy2 = f32[2,3]{1,0} copy(p2)
    %tuple.1 = (f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) tuple(f32[2,3]{1,0} copy1, f32[2,3]{1,0} copy2, f32[] %param_iter)
    %while = (f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) while((f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) %tuple.1), condition=%WhileCond, body=%WhileBody
    %get-tuple-element.4 = f32[2,3]{1,0} get-tuple-element((f32[2,3]{1,0}, f32[2,3]{1,0}, f32[]) %while), index=0
    ROOT %copy3 = f32[2,3]{1,0} copy(get-tuple-element.4)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
  const HloInstruction* while_op =
      module->entry_computation()->GetInstructionWithName("while");
  EXPECT_EQ(
      ShapeUtil::GetSubshape(while_op->shape(), {1}).layout().memory_space(),
      kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, WhileSharedBufferVerificationBug) {
  // Tests a spurious verification failure when a while has the same value
  // passed in twice (copy0) and that value is evicted within the while loop.
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=3
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = f32[3]{0} get-tuple-element(p0), index=2
    gte3 = pred[] get-tuple-element(p0), index=3
    add = f32[3]{0} add(gte0, gte0)
    negate0 = f32[3]{0} negate(add)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    negate15 = f32[3]{0} negate(negate14)
    negate16 = f32[3]{0} negate(negate15)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) tuple(gte0, gte0, negate16, gte3)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy0 = f32[3]{0} copy(p0)
    copy1 = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) tuple(copy0, copy0, copy1, p1)
    while = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    ROOT gte = f32[3]{0} get-tuple-element(while), index=2
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, b228599972) {
  absl::string_view hlo_string = R"(
HloModule entry, is_scheduled=true

fused_computation {
  %p0 = f32[2,3]{1,0} parameter(0)
  %result0 = f32[2,3]{1,0} copy(%p0)
  %result1 = f32[2,3]{1,0} copy(%p0)
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3]{1,0}) tuple(%result0, %result1)
}

ENTRY entry {
  %p0 = f32[2,3]{1,0} parameter(0)
  %p1 = f32[2,3]{1,0} parameter(1)
  %unused = (f32[2,3]{1,0}, f32[2,3]{1,0}) fusion(%p0), kind=kLoop, calls=%fused_computation
  %unused.0 = f32[2,3]{1,0} get-tuple-element(%unused), index=0
  %unused.1 = f32[2,3]{1,0} get-tuple-element(%unused), index=1
  %negate.0 = f32[2,3]{1,0} negate(f32[2,3]{1,0} %unused.0)
  %negate.1 = f32[2,3]{1,0} negate(f32[2,3]{1,0} %unused.1)

  ROOT %result = f32[2,3]{1,0} negate(%p1)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, b172243149) {
  // Tests for the failure in b/172243149, where if we skip processing
  // non-copy allocations that are in default memory can actually cause
  // failures. In this case, the problem tensor is copy0, where it is fed to
  // both negate, while, and add0. The copy0->negate dependency can be allocated
  // in the alternate memory. Then the algorithm attempts to place the
  // copy0->while edge in the alternate memory, but since this value isn't used
  // in the while loop, it won't get an alternate memory allocation. Finally for
  // the copy0->add0 edge, the algorithm will actually replace it with
  // while{0}->add0, since this is equivalent and while is defined later than
  // copy0. However, if we actually skip processing this while{0}->add0
  // allocation, we won't replace this edge, and will end up with the
  // copy0->add0 edge, which illegally extends the lifetime of the alternate
  // memory buffer in copy0.
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=3
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = f32[3]{0} get-tuple-element(p0), index=2
    gte3 = pred[] get-tuple-element(p0), index=3
    add = f32[3]{0} add(gte1, gte2)
    negate0 = f32[3]{0} negate(add)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    negate15 = f32[3]{0} negate(negate14)
    negate16 = f32[3]{0} negate(negate15)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) tuple(gte0, add, negate16, gte3)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy0 = f32[3]{0} copy(p0)
    copy1 = f32[3]{0} copy(p0)
    copy2 = f32[3]{0} copy(p0)
    negate = f32[3]{0} negate(copy0)
    tuple = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) tuple(copy0, copy1, copy2, p1)
    while = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    gte = f32[3]{0} get-tuple-element(while), index=2
    add0 = f32[3]{0} add(negate, copy0)
    ROOT add1 = f32[3]{0} add(add0, gte)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, ControlPredecessorsBug) {
  // Having control_predecessors on an HLO was preventing us from DCEing an op
  // that doesn't have any users (tuple.1). The scheduler assumes the graph is
  // fully DCEed, which causes some instructions not to be scheduled.
  absl::string_view hlo_string = R"(
  HloModule sort.16, is_scheduled=true

  ENTRY %sort.16 (param.0.1: s32[1], param.1.2: f32[1], param.2.3: u32[1], param.3.4: s32[1]) -> (s32[1], f32[1], u32[1], s32[1]) {
    %param.3.4 = s32[1]{0:T(128)} parameter(3)
    %param.2.3 = u32[1]{0:T(128)} parameter(2)
    %param.1.2 = f32[1]{0:T(128)} parameter(1)
    %param.0.1 = s32[1]{0:T(128)} parameter(0)
    %tuple.1 = (s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) tuple(s32[1]{0:T(128)} %param.0.1, f32[1]{0:T(128)} %param.1.2, u32[1]{0:T(128)} %param.2.3, s32[1]{0:T(128)} %param.3.4), control-predecessors={%param.0.1}
    %get-tuple-element.4 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=0
    %get-tuple-element.5 = f32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=1
    %get-tuple-element.6 = u32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=2
    %get-tuple-element.7 = s32[1]{0:T(128)} get-tuple-element((s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) %tuple.1), index=3
    %copy.4 = s32[1]{0:T(128)} copy(s32[1]{0:T(128)} %get-tuple-element.4)
    %copy.5 = f32[1]{0:T(128)} copy(f32[1]{0:T(128)} %get-tuple-element.5)
    %copy.6 = u32[1]{0:T(128)} copy(u32[1]{0:T(128)} %get-tuple-element.6)
    %copy.7 = s32[1]{0:T(128)} copy(s32[1]{0:T(128)} %get-tuple-element.7)
    ROOT %tuple.2 = (s32[1]{0:T(128)}, f32[1]{0:T(128)}, u32[1]{0:T(128)}, s32[1]{0:T(128)}) tuple(s32[1]{0:T(128)} %copy.4, f32[1]{0:T(128)} %copy.5, u32[1]{0:T(128)} %copy.6, s32[1]{0:T(128)} %copy.7)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, ConditionalShouldBeAllocatedInAlternateMem) {
  // Checks if simple conditionals get alternate memory allocations.
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg1 = f32[3]{0} negate(gte)
  }

  false_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg2 = f32[3]{0} negate(gte)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}) tuple(copy)
    ROOT conditional = f32[3]{0} conditional(p1, tuple, tuple), true_computation=true_computation, false_computation=false_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Check that copy and gtes got alternate memory allocations.
  auto copy =
      module->GetComputationWithName("entry")->GetInstructionWithName("copy");
  EXPECT_EQ(copy->shape().layout().memory_space(), kAlternateMemorySpace);
  auto neg1 = module->GetComputationWithName("true_computation")
                  ->GetInstructionWithName("neg1");
  auto neg1_operand = neg1->operand(0);
  EXPECT_EQ(neg1_operand->shape().layout().memory_space(),
            kAlternateMemorySpace);
  auto neg2 = module->GetComputationWithName("false_computation")
                  ->GetInstructionWithName("neg2");
  auto neg2_operand = neg2->operand(0);
  EXPECT_EQ(neg2_operand->shape().layout().memory_space(),
            kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, ConditionalAvoidsUnnecessaryPrefetch) {
  // Checks if we avoid unnecessary allocation in alternate memory if the input
  // won't be used in the computation for a long time.
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation {
    p0 = (f32[3]{0}, f32[3]{0}) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    neg0 = f32[3]{0} negate(gte0)
    neg1 = f32[3]{0} negate(neg0)
    neg2 = f32[3]{0} negate(neg1)
    neg3 = f32[3]{0} negate(neg2)
    neg4 = f32[3]{0} negate(neg3)
    neg5 = f32[3]{0} negate(neg4)
    neg6 = f32[3]{0} negate(neg5)
    neg7 = f32[3]{0} negate(neg6)
    neg8 = f32[3]{0} negate(neg7)
    neg9 = f32[3]{0} negate(neg8)
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    ROOT add = f32[3]{0} add(neg9, gte1)
  }

  false_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg = f32[3]{0} negate(gte)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy0 = f32[3]{0} copy(p0)
    copy1 = f32[3]{0} copy(p0)
    tuple0 = (f32[3]{0}, f32[3]{0}) tuple(copy0, copy1)
    tuple1 = (f32[3]{0}) tuple(copy0)
    ROOT conditional = f32[3]{0} conditional(p1, tuple0, tuple1), true_computation=true_computation, false_computation=false_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Check that copy1 doesn't get unnecessarily allocated in alternate mem
  // (due to long negate chain in true_computation) but is prefetched before
  // add.
  auto copy0 =
      module->GetComputationWithName("entry")->GetInstructionWithName("copy0");
  EXPECT_EQ(copy0->shape().layout().memory_space(), kAlternateMemorySpace);
  auto copy1 =
      module->GetComputationWithName("entry")->GetInstructionWithName("copy1");
  EXPECT_EQ(copy1->shape().layout().memory_space(), kDefaultMemorySpace);
  auto add = module->GetComputationWithName("true_computation")
                 ->GetInstructionWithName("add");
  auto add_operand = add->operand(1);
  EXPECT_EQ(add_operand->shape().layout().memory_space(),
            kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, ConditionalMultiUse) {
  // Make sure there is an evict when there is a conditional use followed by
  // another use.
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation {
    p0 = (f32[3]{0}, f32[3]{0}) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    add0 = f32[3]{0} add(gte0, gte1)
    neg0 = f32[3]{0} negate(add0)
    neg1 = f32[3]{0} negate(neg0)
    neg2 = f32[3]{0} negate(neg1)
    neg3 = f32[3]{0} negate(neg2)
    neg4 = f32[3]{0} negate(neg3)
    neg5 = f32[3]{0} negate(neg4)
    neg6 = f32[3]{0} negate(neg5)
    neg7 = f32[3]{0} negate(neg6)
    neg8 = f32[3]{0} negate(neg7)
    ROOT neg9 = f32[3]{0} negate(neg8)
  }

  false_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg = f32[3]{0} negate(gte)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy0 = f32[3]{0} copy(p0)
    copy1 = f32[3]{0} copy(p0)
    tuple0 = (f32[3]{0}, f32[3]{0}) tuple(copy0, copy1)
    tuple1 = (f32[3]{0}) tuple(copy0)
    conditional = f32[3]{0} conditional(p1, tuple0, tuple1), true_computation=true_computation, false_computation=false_computation
    ROOT add1 = f32[3]{0} add(copy1, conditional)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Make sure the copy1->add edge is in alternate memory. Before conditional,
  // this should be evicted to default memory and neg uses the input from
  // default memory.
  auto copy1 =
      module->GetComputationWithName("entry")->GetInstructionWithName("copy1");
  EXPECT_EQ(copy1->shape().layout().memory_space(), kAlternateMemorySpace);
  auto add0 = module->GetComputationWithName("true_computation")
                  ->GetInstructionWithName("add0");
  auto add0_operand = add0->operand(1);
  EXPECT_EQ(add0_operand->shape().layout().memory_space(),
            kAlternateMemorySpace);
  auto add1 =
      module->GetComputationWithName("entry")->GetInstructionWithName("add1");
  auto add1_operand = add1->operand(0);
  EXPECT_EQ(add1_operand->shape().layout().memory_space(), kDefaultMemorySpace);
  EXPECT_EQ(add1_operand->opcode(), HloOpcode::kCopyDone);
}

TEST_F(MemorySpaceAssignmentTest, ConditionalMultiUseInWhile) {
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg1 = f32[3]{0} negate(gte)
  }

  false_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg2 = f32[3]{0} negate(gte)
  }

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = pred[] get-tuple-element(p0), index=2
    cond_tuple = (f32[3]{0}) tuple(gte0)
    conditional = f32[3]{0} conditional(gte2, cond_tuple, cond_tuple), true_computation=true_computation, false_computation=false_computation
    add = f32[3]{0} add(conditional, gte1)
    neg0 = f32[3]{0} negate(add)
    neg1 = f32[3]{0} negate(neg0)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(gte0, neg1, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy0 = f32[3]{0} copy(p0)
    copy1 = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy0, copy1, p1)
    while = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    ROOT gte = f32[3]{0} get-tuple-element(while), index=1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
  // Make sure copy1/while{0}/cond_tuple{0} gets alternate memory allocation.
  // This will force an eviction and a prefetch for while body root.
  auto copy0 =
      module->GetComputationWithName("entry")->GetInstructionWithName("copy0");
  EXPECT_EQ(copy0->shape().layout().memory_space(), kAlternateMemorySpace);
  auto conditional = module->GetComputationWithName("while_body")
                         ->GetInstructionWithName("conditional");
  auto conditional_operand = conditional->operand(1);
  EXPECT_EQ(ShapeUtil::GetSubshape(conditional_operand->shape(), {0})
                .layout()
                .memory_space(),
            kAlternateMemorySpace);
  auto while_root =
      module->GetComputationWithName("while_body")->root_instruction();
  auto while_root_operand = while_root->operand(0);
  EXPECT_THAT(
      while_root_operand,
      op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                    op::AsyncCopy(kDefaultMemorySpace, kAlternateMemorySpace,
                                  op::GetTupleElement(op::Parameter(0)))));
}

TEST_F(MemorySpaceAssignmentTest, NestedConditional) {
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation2 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg1 = f32[3]{0} negate(gte)
  }

  false_computation2 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg2 = f32[3]{0} negate(gte)
  }

  true_computation1 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    slice = f32[1]{0} slice(gte), slice={[0:1]}
    bitcast = f32[] bitcast(slice)
    constant = f32[] constant(0.0)
    compare = pred[] compare(bitcast, constant), direction=GT
    ROOT conditional = f32[3]{0} conditional(compare, p0, p0), true_computation=true_computation2, false_computation=false_computation2
  }

  false_computation1 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg3 = f32[3]{0} negate(gte)
  }


  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}) tuple(copy)
    ROOT conditional = f32[3]{0} conditional(p1, tuple, tuple), true_computation=true_computation1, false_computation=false_computation1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Make sure alternate memory allocation gets propagated into both levels of
  // conditional.
  auto copy =
      module->GetComputationWithName("entry")->GetInstructionWithName("copy");
  EXPECT_EQ(copy->shape().layout().memory_space(), kAlternateMemorySpace);
  auto neg1_operand = module->GetComputationWithName("true_computation2")
                          ->GetInstructionWithName("neg1")
                          ->operand(0);
  auto neg2_operand = module->GetComputationWithName("false_computation2")
                          ->GetInstructionWithName("neg2")
                          ->operand(0);
  auto neg3_operand = module->GetComputationWithName("false_computation1")
                          ->GetInstructionWithName("neg3")
                          ->operand(0);
  EXPECT_EQ(neg1_operand->shape().layout().memory_space(),
            kAlternateMemorySpace);
  EXPECT_EQ(neg2_operand->shape().layout().memory_space(),
            kAlternateMemorySpace);
  EXPECT_EQ(neg3_operand->shape().layout().memory_space(),
            kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, NestedConditionalBufferReuseVerificationBug) {
  // Tests a spurious verification failure when there are nested conditionals
  // and the innermost conditional computation reuses the buffer. Here, both the
  // parameter of true_computation2 and neg2 will get the same buffer. Make sure
  // that verification doesn't claim a failure in this case.
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation2 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    neg1 = f32[3]{0} negate(gte)
    neg2 = f32[3]{0} negate(neg1)
    ROOT neg3 = f32[3]{0} negate(neg2)
  }

  false_computation2 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg4 = f32[3]{0} negate(gte)
  }

  true_computation1 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    slice = f32[1]{0} slice(gte), slice={[0:1]}
    bitcast = f32[] bitcast(slice)
    constant = f32[] constant(0.0)
    compare = pred[] compare(bitcast, constant), direction=GT
    tuple = (f32[3]{0}) tuple(gte)
    ROOT conditional = f32[3]{0} conditional(compare, tuple, tuple), true_computation=true_computation2, false_computation=false_computation2
  }

  false_computation1 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg5 = f32[3]{0} negate(gte)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}) tuple(copy)
    ROOT conditional = f32[3]{0} conditional(p1, tuple, tuple), true_computation=true_computation1, false_computation=false_computation1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, WhileInsideNestedConditionalVerificationBug) {
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}) parameter(0)
    ROOT constant = pred[] constant(true)
  }

  while_body {
    p0 = (f32[3]{0}) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    negate0 = f32[3]{0} negate(gte0)
    ROOT tuple = (f32[3]{0}) tuple(negate0)
  }

  true_computation2 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    tuple = (f32[3]{0}) tuple(gte)
    while = (f32[3]{0}) while(tuple), condition=while_cond, body=while_body
    while_gte0 = f32[3]{0} get-tuple-element(while), index=0
    ROOT root = f32[3]{0} negate(while_gte0)
  }

  false_computation2 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg3 = f32[3]{0} negate(gte)
  }

  true_computation1 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    constant = pred[] constant(true)
    tuple = (f32[3]{0}) tuple(gte)
    ROOT conditional = f32[3]{0} conditional(constant, tuple, tuple), true_computation=true_computation2, false_computation=false_computation2
  }

  false_computation1 {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg3 = f32[3]{0} negate(gte)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}) tuple(copy)
    ROOT conditional = f32[3]{0} conditional(p1, tuple, tuple), true_computation=true_computation1, false_computation=false_computation1
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest,
       ConditionalComputationBufferOverlapBeforeParam) {
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation {
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT neg2 = f32[3]{0} negate(gte)
  }

  false_computation {
    c = f32[3]{0} constant({0.0, 1.0, 2.0})
    neg0 = f32[3]{0} negate(c)
    neg1 = f32[3]{0} negate(neg0)
    p0 = (f32[3]{0}) parameter(0)
    gte = f32[3]{0} get-tuple-element(p0), index=0
    ROOT add = f32[3]{0} add(gte, neg1)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}) tuple(copy)
    ROOT conditional = f32[3]{0} conditional(p1, tuple, tuple), true_computation=true_computation, false_computation=false_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(module.get());

  auto get_offset = [&](absl::string_view hlo_name) {
    for (const auto& chunk : preset_assignments->chunks()) {
      if (chunk.first.instruction->name() == hlo_name) {
        return chunk.second.offset;
      }
    }
    return static_cast<int64_t>(-1);
  };

  int64_t copy_offset = get_offset("copy");
  int64_t neg0_offset = get_offset("neg0");
  EXPECT_NE(copy_offset, -1);
  EXPECT_NE(neg0_offset, -1);
  EXPECT_NE(copy_offset, neg0_offset);
}

TEST_F(MemorySpaceAssignmentTest,
       RequestIdentifierShouldNotBeAllocatedInAlternateMem) {
  // Ensure that request identifier returned by Send/Recv HLOs are not allocated
  // in the alternate memory.
  absl::string_view hlo_string = R"(
  HloModule SendRecv, is_scheduled=true

  ENTRY %AddDependency (p: f32[3]) -> f32[3] {
    %p = f32[3]{0} parameter(0)
    %after-all = token[] after-all()
    %recv.4 = (f32[3]{0}, u32[], token[]) recv(token[] %after-all), channel_id=7
    %recv-done.4 = (f32[3]{0}, token[]) recv-done((f32[3]{0}, u32[], token[]) %recv.4), channel_id=7
    %token.1 = token[] get-tuple-element((f32[3]{0}, token[]) %recv-done.4), index=1
    %data = f32[3]{0} get-tuple-element((f32[3]{0}, token[]) %recv-done.4), index=0
    %send = (f32[3]{0}, u32[], token[]) send(f32[3]{0} %data, token[] %token.1), channel_id=2
    %send-done = token[] send-done((f32[3]{0}, u32[], token[]) %send), channel_id=2
    ROOT %add = f32[3]{0} add(f32[3]{0} %p, f32[3]{0} %data)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  for (const HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kSend ||
        instruction->opcode() == HloOpcode::kRecv) {
      const Shape& request_identifier_shape =
          ShapeUtil::GetSubshape(instruction->shape(), {1});
      EXPECT_NE(request_identifier_shape.layout().memory_space(),
                kAlternateMemorySpace);
    }
  }
}

TEST_F(MemorySpaceAssignmentTest, SendDoneShouldHaveSendOperand) {
  // Ensure that SendDone has only a Send operand.
  absl::string_view hlo_string = R"(
  HloModule SendRecv, is_scheduled=true

  ENTRY %AddDependency (p: f32[3]) -> f32[3] {
    %p0 = f32[3]{0} parameter(0)
    %p1 = f32[3]{0} parameter(1)
    %neg0 = f32[3]{0} negate(f32[3]{0} %p1)
    %neg1 = f32[3]{0} negate(f32[3]{0} %neg0)
    %neg2 = f32[3]{0} negate(f32[3]{0} %neg1)
    %neg3 = f32[3]{0} negate(f32[3]{0} %neg2)
    %neg4 = f32[3]{0} negate(f32[3]{0} %neg3)
    %neg5 = f32[3]{0} negate(f32[3]{0} %neg4)
    %neg6 = f32[3]{0} negate(f32[3]{0} %neg5)
    %after-all = token[] after-all()
    %send = (f32[3]{0}, u32[], token[]) send(f32[3]{0} %p0, token[] %after-all), channel_id=2
    %send-done = token[] send-done((f32[3]{0}, u32[], token[]) %send), channel_id=2
    ROOT %add = f32[3]{0} add(f32[3]{0} %p0, f32[3]{0} %neg6)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, SendAndSendDoneShouldGetSameAllocation) {
  // Ensure that Send and SendDone have the same allocation.
  absl::string_view hlo_string = R"(
  HloModule SendRecv, is_scheduled=true

  ENTRY %AddDependency (p: f32[3]) -> f32[3] {
    %p0 = f32[3]{0} parameter(0)
    %p1 = f32[3]{0} parameter(1)
    %after-all = token[] after-all()
    %send = (f32[3]{0}, u32[], token[]) send(f32[3]{0} %p0, token[] %after-all), channel_id=2
    %neg0 = f32[3]{0} negate(f32[3]{0} %p1)
    %neg1 = f32[3]{0} negate(f32[3]{0} %neg0)
    %neg2 = f32[3]{0} negate(f32[3]{0} %neg1)
    %neg3 = f32[3]{0} negate(f32[3]{0} %neg2)
    %neg4 = f32[3]{0} negate(f32[3]{0} %neg3)
    %neg5 = f32[3]{0} negate(f32[3]{0} %neg4)
    %neg6 = f32[3]{0} negate(f32[3]{0} %neg5)
    %send-done = token[] send-done((f32[3]{0}, u32[], token[]) %send), channel_id=2
    ROOT %add = f32[3]{0} add(f32[3]{0} %p0, f32[3]{0} %neg6)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get(), DefaultMemorySpaceOptions(),
                    /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/4);
}

TEST_F(MemorySpaceAssignmentTest, LastUseOpt) {
  // Test that checks the last use optimization. It uses two buffers that should
  // be placed in alternate memory.
  //
  //      +-------+
  //     /         \
  // add1--->sub1   +-------->mul2
  //              mul1===>add2
  //
  // Without the last use optimization, the mul1 buffer will be assigned first
  // (because it is larger) to offset 0. Then, add1 will be scheduled for the
  // add1 to sub1 segment. Because offset 0 is available, it will get that
  // offset. But because offset 0 is not available in the sub1 to mul2 offset,
  // it will end up in unnecessary copies. With the last use optimization, these
  // copies can be optimized away.
  HloComputation::Builder builder(TestName());
  Shape shape1 = ShapeUtil::MakeShape(F32, {2, 3});
  Shape shape2 = ShapeUtil::MakeShape(F32, {2, 4});
  PaddingConfig padding_config = MakeEdgePaddingConfig({{0, 0}, {0, 1}});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape1, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape2, "p1"));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape1, HloOpcode::kAdd, p0, p0));
  HloInstruction* sub1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape1, HloOpcode::kSubtract, p0, add1));
  HloInstruction* mul1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape2, HloOpcode::kMultiply, p1, p1));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape2, HloOpcode::kAdd, mul1, p1));
  HloInstruction* mul2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape1, HloOpcode::kMultiply, add1, sub1));
  HloInstruction* padding_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(F32)));
  HloInstruction* padded_mul2 = builder.AddInstruction(
      HloInstruction::CreatePad(shape2, mul2, padding_value, padding_config));
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape2, HloOpcode::kAdd, add2, padded_mul2));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, add1, sub1, mul1, add2, mul2,
                                      padding_value, padded_mul2, add3});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  EXPECT_THAT(
      mul2,
      op::Multiply(
          op::Add(op::Parameter(0), op::Parameter(0)),
          op::Subtract(op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                     op::Parameter(0)),
                       op::Add(op::Parameter(0), op::Parameter(0)))));
}

TEST_F(MemorySpaceAssignmentTest, NonEntryComputationSchedule1) {
  // Test to ensure CopyStart/CopyDone is placed only in the entry computation.
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(xla::F32, {2, 3});
  Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, scalar_shape});

  auto cond_builder = HloComputation::Builder("WhileCond");
  // Tuple param: 24 bytes (each elem has 8 byte pointer, 4 byte element)
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_param"));
  HloInstruction* cond_iter = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, cond_param, 1));
  HloInstruction* cond_limit = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(50.f)));
  // Free cond_param[] (16 bytes), Alloc PRED[] (1 byte)
  HloInstruction* cond_lt = cond_builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), cond_iter,
                                    cond_limit, ComparisonDirection::kLt));
  HloComputation* cond_computation =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto body_builder = HloComputation::Builder("WhileBody");
  // Tuple param: 24 bytes (each elem has 8 byte pointer, 4 byte element)
  HloInstruction* body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_param"));
  HloInstruction* body_iter = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, body_param, 1));
  HloInstruction* body_data = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 0));
  HloInstruction* body_iter_increment = body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.f)));
  HloInstruction* body_iter_next =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          scalar_shape, HloOpcode::kAdd, body_iter, body_iter_increment));
  HloInstruction* body_data_increment =
      body_builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<float>({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}})));
  HloInstruction* body_data_mul =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kMultiply, body_data, body_data));
  HloInstruction* body_data_add =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, body_data, body_data_increment));
  HloInstruction* body_data_next =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, body_data_add, body_data_mul));
  HloInstruction* body_out = body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_data_next, body_iter_next}));
  HloComputation* body_computation =
      module->AddEmbeddedComputation(body_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* data = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param_iter"));
  HloInstruction* iter = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "param_data"));
  HloInstruction* p2 =
      builder.AddInstruction(HloInstruction::CreateParameter(2, shape, "p2"));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({data, iter}));
  HloInstruction* while_op = builder.AddInstruction(HloInstruction::CreateWhile(
      tuple_shape, cond_computation, body_computation, tuple));
  HloInstruction* while_data = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, while_op, 0));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, while_data, p2));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(cond_computation,
                        {cond_param, cond_iter, cond_limit, cond_lt});
  schedule.set_sequence(body_computation,
                        {body_param, body_iter, body_data, body_iter_increment,
                         body_iter_next, body_data_increment, body_data_mul,
                         body_data_add, body_data_next, body_out});
  schedule.set_sequence(entry_computation,
                        {iter, data, p2, tuple, while_op, while_data, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), DefaultMemorySpaceOptions(), 50);
}

TEST_F(MemorySpaceAssignmentTest, NonEntryComputationSchedule2) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(xla::F32, {2, 3});
  Shape shape2 = ShapeUtil::MakeShape(xla::F32, {3, 3});

  auto call_builder = HloComputation::Builder("Call");
  HloInstruction* call_param = call_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "call_param"));
  HloInstruction* call_param2 = call_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape2, "call_param2"));
  HloInstruction* slice = call_builder.AddInstruction(
      HloInstruction::CreateSlice(shape, call_param2, {0, 0}, {2, 3}, {1, 1}));
  HloInstruction* mul =
      call_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kMultiply, call_param, slice));
  HloInstruction* negate0 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, mul));
  HloInstruction* negate1 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* negate7 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate6));
  HloInstruction* add0 =
      call_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, call_param, negate7));
  HloComputation* call_computation =
      module->AddEmbeddedComputation(call_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape2, "p1"));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, p0));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, add1, p0));
  HloInstruction* negate8 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape2, HloOpcode::kNegate, p1));
  HloInstruction* call = builder.AddInstruction(
      HloInstruction::CreateCall(shape, {add1, negate8}, call_computation));
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, add1));
  HloInstruction* add4 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, call, add3));
  HloInstruction* add5 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, add2, add4));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      call_computation,
      {call_param, call_param2, slice, mul, negate0, negate1, negate2, negate3,
       negate4, negate5, negate6, negate7, add0});
  schedule.set_sequence(entry_computation,
                        {p0, p1, add1, add2, negate8, call, add3, add4, add5});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), DefaultMemorySpaceOptions(), 5);
}

TEST_F(MemorySpaceAssignmentTest, NonEntryComputationSchedule3) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(xla::F32, {2, 3});
  Shape shape2 = ShapeUtil::MakeShape(xla::F32, {3, 3});

  auto call_builder = HloComputation::Builder("Call");
  HloInstruction* call_param = call_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "call_param"));
  // Use shape2 here which is larger (scheduled earlier) to occupy alternate
  // memory at the beginning. This should cause a situation where the prefetch
  // of add1 later in the function body gets the wrong offset which cannot be
  // communicated to the outside the function.
  HloInstruction* iota =
      call_builder.AddInstruction(HloInstruction::CreateIota(shape2, 0));
  HloInstruction* slice = call_builder.AddInstruction(
      HloInstruction::CreateSlice(shape, iota, {0, 0}, {2, 3}, {1, 1}));
  HloInstruction* mul =
      call_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kMultiply, call_param, slice));
  HloInstruction* negate0 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, mul));
  HloInstruction* negate1 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* negate7 = call_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate6));
  HloInstruction* add0 =
      call_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, call_param, negate7));
  HloComputation* call_computation =
      module->AddEmbeddedComputation(call_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, p0));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, add1, p0));
  HloInstruction* call = builder.AddInstruction(
      HloInstruction::CreateCall(shape, {add1}, call_computation));
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, call, add1));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      call_computation,
      {call_param, iota, slice, mul, negate0, negate1, negate2, negate3,
       negate4, negate5, negate6, negate7, add0});
  schedule.set_sequence(entry_computation, {p0, add1, add2, call, add3});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), DefaultMemorySpaceOptions(), 5);
}

// TODO(berkin): This might be an incorrect input graph, investigate.
TEST_F(MemorySpaceAssignmentTest, DISABLED_NonEntryComputationSchedule4) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(xla::F32, {2, 3});
  Shape shape2 = ShapeUtil::MakeShape(xla::F32, {3, 3});

  auto true_builder = HloComputation::Builder("True");
  HloInstruction* true_param = true_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "true_param"));
  HloInstruction* iota =
      true_builder.AddInstruction(HloInstruction::CreateIota(shape2, 0));
  HloInstruction* slice = true_builder.AddInstruction(
      HloInstruction::CreateSlice(shape, iota, {0, 0}, {2, 3}, {1, 1}));
  HloInstruction* mul =
      true_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kMultiply, true_param, slice));
  HloInstruction* negate0 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, mul));
  HloInstruction* negate1 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* negate7 = true_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate6));
  HloInstruction* add0 =
      true_builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, true_param, negate7));
  HloComputation* true_computation =
      module->AddEmbeddedComputation(true_builder.Build());

  auto false_builder = HloComputation::Builder("False");
  HloInstruction* false_param = false_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "false_param"));
  HloComputation* false_computation =
      module->AddEmbeddedComputation(false_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, p0));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, add1, p0));
  HloInstruction* pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloInstruction* conditional =
      builder.AddInstruction(HloInstruction::CreateConditional(
          shape, pred, add1, true_computation, add2, false_computation));
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, conditional, add1));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      true_computation,
      {true_param, iota, slice, mul, negate0, negate1, negate2, negate3,
       negate4, negate5, negate6, negate7, add0});
  schedule.set_sequence(false_computation, {false_param});
  schedule.set_sequence(entry_computation,
                        {p0, add1, add2, pred, conditional, add3});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), DefaultMemorySpaceOptions(), 5);
}

TEST_F(MemorySpaceAssignmentTest, NonEntryComputationSchedule5) {
  // This test reproduces the failure in b/143288178.  Given a graph like the
  // following:
  //
  // ... = foo(a)
  // tuple = tuple((..., a)
  // ... = while(tuple) {
  //   p = param(0)
  //   a1 = get-tuple-element(p), index=n-1
  //   ...
  //   ROOT tuple((..., a1))
  // }
  //
  // If a copy to alternate memory is inserted before foo, and if the size of
  // the while body is less than max prefetch interval so that the copy-done is
  // kept in the alternate memory, then we end up referring to the copy-done in
  // the root instruction of the while loop body. I.e.,
  //
  // cs = copy-start(a)
  // ...
  // cd = copy-done(cs)
  // ... = foo(cd)
  // tuple = tuple((..., cd)
  // ... = while(tuple) {
  //   p = param(0)
  //   a1 = get-tuple-element(p), index=n-1
  //   ...
  //   ROOT tuple((..., cd))  <-- Error: cd belongs to outside computation.
  // }
  //
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(xla::F32, {2, 3});
  Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape =
      ShapeUtil::MakeTupleShape({shape, scalar_shape, scalar_shape});

  auto cond_builder = HloComputation::Builder("WhileCond");
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_param"));
  HloInstruction* cond_iter = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, cond_param, 1));
  HloInstruction* cond_limit = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(50.f)));
  HloInstruction* cond_lt = cond_builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), cond_iter,
                                    cond_limit, ComparisonDirection::kLt));
  HloComputation* cond_computation =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto body_builder = HloComputation::Builder("WhileBody");
  HloInstruction* body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_param"));
  HloInstruction* body_iter = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, body_param, 1));
  HloInstruction* body_data = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 0));
  HloInstruction* body_iter_increment = body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.f)));
  HloInstruction* body_iter_next =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          scalar_shape, HloOpcode::kAdd, body_iter, body_iter_increment));
  HloInstruction* body_data2 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, body_param, 2));
  HloInstruction* body_out = body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_data, body_iter_next, body_data2}));
  HloComputation* body_computation =
      module->AddEmbeddedComputation(body_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* data = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param_data"));
  HloInstruction* iter = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "param_iter"));
  HloInstruction* data2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, scalar_shape, "param_data2"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, data));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* negate7 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate6));
  HloInstruction* sub = builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kSubtract, iter, data2));
  HloInstruction* tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({negate7, iter, data2}));
  HloInstruction* while_op = builder.AddInstruction(HloInstruction::CreateWhile(
      tuple_shape, cond_computation, body_computation, tuple));
  HloInstruction* while_data = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, while_op, 1));
  HloInstruction* root =
      builder.AddInstruction(HloInstruction::CreateTuple({while_data, sub}));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(cond_computation,
                        {cond_param, cond_iter, cond_limit, cond_lt});
  schedule.set_sequence(body_computation,
                        {body_param, body_iter, body_data, body_iter_increment,
                         body_iter_next, body_data2, body_out});
  schedule.set_sequence(
      entry_computation,
      {iter, data, data2, negate0, negate1, negate2, negate3, negate4, negate5,
       negate6, negate7, sub, tuple, while_op, while_data, root});
  TF_CHECK_OK(module->set_schedule(schedule));

  // Set a large max prefetch interval so that the buffer can be kept in
  // alternate memory.
  AssignMemorySpace(module.get(), DefaultMemorySpaceOptions(), 20);
}

TEST_F(MemorySpaceAssignmentTest, NonEntryComputationSchedule6) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(xla::F32, {2, 3});
  Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, scalar_shape, shape});

  auto cond_builder = HloComputation::Builder("WhileCond");
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_param"));
  HloInstruction* cond_iter = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, cond_param, 1));
  HloInstruction* cond_limit = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(50.f)));
  HloInstruction* cond_lt = cond_builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), cond_iter,
                                    cond_limit, ComparisonDirection::kLt));
  HloComputation* cond_computation =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto body_builder = HloComputation::Builder("WhileBody");
  HloInstruction* body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_param"));
  HloInstruction* body_iter = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, body_param, 1));
  HloInstruction* body_data = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 0));
  HloInstruction* body_negate0 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_data));
  HloInstruction* body_negate1 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_negate0));
  HloInstruction* body_negate2 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_negate1));
  HloInstruction* body_negate3 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_negate2));
  HloInstruction* body_negate4 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_negate3));
  HloInstruction* body_negate5 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_negate4));
  HloInstruction* body_negate6 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_negate5));
  HloInstruction* body_negate7 = body_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, body_negate6));
  HloInstruction* body_iter_increment = body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.f)));
  HloInstruction* body_iter_next =
      body_builder.AddInstruction(HloInstruction::CreateBinary(
          scalar_shape, HloOpcode::kAdd, body_iter, body_iter_increment));
  HloInstruction* body_out = body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_data, body_iter_next, body_negate7}));
  HloComputation* body_computation =
      module->AddEmbeddedComputation(body_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* data = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param_data"));
  HloInstruction* iter = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "param_iter"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, data));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* negate7 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate6));
  HloInstruction* tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({data, iter, negate7}));
  HloInstruction* while_op = builder.AddInstruction(HloInstruction::CreateWhile(
      tuple_shape, cond_computation, body_computation, tuple));
  HloInstruction* while_data = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, while_op, 0));
  HloInstruction* while_data2 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, while_op, 2));
  HloInstruction* root = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, while_data, while_data2));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(cond_computation,
                        {cond_param, cond_iter, cond_limit, cond_lt});
  schedule.set_sequence(
      body_computation,
      {body_param, body_iter, body_data, body_negate0, body_negate1,
       body_negate2, body_negate3, body_negate4, body_negate5, body_negate6,
       body_negate7, body_iter_increment, body_iter_next, body_out});
  schedule.set_sequence(
      entry_computation,
      {iter, data, negate0, negate1, negate2, negate3, negate4, negate5,
       negate6, negate7, tuple, while_op, while_data, while_data2, root});
  TF_CHECK_OK(module->set_schedule(schedule));

  // Pick a large max prefetch interval to ensure all the while inputs are
  // allocated in the alternate memory.
  AssignMemorySpace(module.get(), DefaultMemorySpaceOptions(),
                    /*max_prefetch_interval=*/25);

  // Index {0} of the while loop argument is not written inside the while loop,
  // so it can be trivially placed in the alternate memory space.
  *ShapeUtil::GetMutableSubshape(&tuple_shape, {0})->mutable_layout() =
      LayoutUtil::MakeLayout(
          /*minor_to_major=*/{1, 0}, /*dim_level_types=*/{}, /*dim_unique=*/{},
          /*dim_ordered=*/{}, /*tiles=*/{},
          /*tail_padding_alignment_in_elements=*/1,
          /*index_primitive_type=*/PRIMITIVE_TYPE_INVALID,
          /*pointer_primitive_type=*/PRIMITIVE_TYPE_INVALID,
          /*element_size_in_bits=*/0, kAlternateMemorySpace);
  // Index {1} is a scalar, so it is always placed in the default memory.
  *ShapeUtil::GetMutableSubshape(&tuple_shape, {1})->mutable_layout() =
      LayoutUtil::MakeLayout(
          /*minor_to_major=*/{}, /*dim_level_types=*/{}, /*dim_unique=*/{},
          /*dim_ordered=*/{}, /*tiles=*/{},
          /*tail_padding_alignment_in_elements=*/1,
          /*index_primitive_type=*/PRIMITIVE_TYPE_INVALID,
          /*pointer_primitive_type=*/PRIMITIVE_TYPE_INVALID,
          /*element_size_in_bits=*/0, kDefaultMemorySpace);
  // Index {2} of the while loop is placed in the default memory.
  *ShapeUtil::GetMutableSubshape(&tuple_shape, {2})->mutable_layout() =
      LayoutUtil::MakeLayout(
          /*minor_to_major=*/{1, 0}, /*dim_level_types=*/{}, /*dim_unique=*/{},
          /*dim_ordered=*/{}, /*tiles=*/{},
          /*tail_padding_alignment_in_elements=*/1,
          /*index_primitive_type=*/PRIMITIVE_TYPE_INVALID,
          /*pointer_primitive_type=*/PRIMITIVE_TYPE_INVALID,
          /*element_size_in_bits=*/0, kDefaultMemorySpace);

  // Expect the layout for the while loop and its aliased buffers.
  EXPECT_THAT(while_op, op::ShapeWithLayout(tuple_shape));
  EXPECT_THAT(while_op->operand(0), op::ShapeWithLayout(tuple_shape));
  EXPECT_THAT(cond_param, op::ShapeWithLayout(tuple_shape));
  EXPECT_THAT(body_param, op::ShapeWithLayout(tuple_shape));
  EXPECT_THAT(body_out, op::ShapeWithLayout(tuple_shape));
}

TEST_F(MemorySpaceAssignmentTest, DanglingCopy) {
  // This situation was encountered in vss, where there is a mismatch in the
  // memory space in preset assignments and the output graph.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});

  HloInstruction* p = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p, 0));
  HloInstruction* p1a = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p, 1));
  HloInstruction* copy = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCopy, p1a));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* p1b = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p, 1));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1b));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation, {p, p0, negate0, negate1, negate2, negate3, negate4, negate5,
                    negate6, p1a, copy, p1b, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, MultiOutputFusion) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder fusion_builder("fusion");
  HloInstruction* fusion_param0 = fusion_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion_param1 = fusion_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "p1"));
  fusion_builder.AddInstruction(
      HloInstruction::CreateTuple({fusion_param0, fusion_param1}));
  HloComputation* fusion_computation =
      module->AddEmbeddedComputation(fusion_builder.Build());

  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      tuple_shape, HloInstruction::FusionKind::kCustom, {p0, p0},
      fusion_computation));
  HloInstruction* element0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion, 0));
  HloInstruction* element1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion, 1));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, element0, element1));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, fusion, element0, element1, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, TupleInput) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder fusion_builder("fusion");
  HloInstruction* fusion_param = fusion_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* fusion_element0 = fusion_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion_param, 0));
  HloInstruction* fusion_element1 = fusion_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion_param, 1));
  fusion_builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, fusion_element0, fusion_element1));
  HloComputation* fusion_computation =
      module->AddEmbeddedComputation(fusion_builder.Build());

  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p1));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({negate0, negate1}));
  HloInstruction* fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kCustom, {tuple}, fusion_computation));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, tuple, fusion});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, TupleToTuple1) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder fusion0_builder("fusion0");
  HloInstruction* fusion0_param0 = fusion0_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion0_param1 = fusion0_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "p1"));
  fusion0_builder.AddInstruction(
      HloInstruction::CreateTuple({fusion0_param0, fusion0_param1}));
  HloComputation* fusion0_computation =
      module->AddEmbeddedComputation(fusion0_builder.Build());

  HloComputation::Builder fusion1_builder("fusion1");
  HloInstruction* fusion1_param = fusion1_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* fusion1_element0 = fusion1_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion1_param, 0));
  HloInstruction* fusion1_element1 = fusion1_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion1_param, 1));
  fusion1_builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, fusion1_element0, fusion1_element1));
  HloComputation* fusion1_computation =
      module->AddEmbeddedComputation(fusion1_builder.Build());

  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion0 = builder.AddInstruction(HloInstruction::CreateFusion(
      tuple_shape, HloInstruction::FusionKind::kCustom, {p0, p0},
      fusion0_computation));
  HloInstruction* element0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion0, 0));
  HloInstruction* element1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion0, 1));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, element0, element1));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, add0, negate6));
  HloInstruction* fusion1 = builder.AddInstruction(
      HloInstruction::CreateFusion(shape, HloInstruction::FusionKind::kCustom,
                                   {fusion0}, fusion1_computation));
  HloInstruction* mul = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, add1, fusion1));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation,
      {p0, fusion0, element0, element1, negate0, negate1, negate2, negate3,
       negate4, negate5, negate6, add0, add1, fusion1, mul});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), DefaultMemorySpaceOptions(), 5);
  EXPECT_THAT(fusion1,
              op::Fusion(op::Tuple(
                  op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                op::GetTupleElement(op::Fusion(), 0)),
                  op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                op::GetTupleElement(op::Fusion(), 1)))));
}

TEST_F(MemorySpaceAssignmentTest, TupleToTuple2) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});
  Shape nested_tuple_shape = ShapeUtil::MakeTupleShape({shape, tuple_shape});
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder fusion0_builder("fusion0");
  HloInstruction* fusion0_param0 = fusion0_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion0_param1 = fusion0_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* fusion0_tuple = fusion0_builder.AddInstruction(
      HloInstruction::CreateTuple({fusion0_param0, fusion0_param1}));
  fusion0_builder.AddInstruction(
      HloInstruction::CreateTuple({fusion0_param0, fusion0_tuple}));
  HloComputation* fusion0_computation =
      module->AddEmbeddedComputation(fusion0_builder.Build());

  HloComputation::Builder fusion1_builder("fusion1");
  HloInstruction* fusion1_param = fusion1_builder.AddInstruction(
      HloInstruction::CreateParameter(0, nested_tuple_shape, "p"));
  HloInstruction* fusion1_element0 = fusion1_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion1_param, 0));
  HloInstruction* fusion1_element1 = fusion1_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(tuple_shape, fusion1_param, 1));
  HloInstruction* fusion1_element2 = fusion1_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion1_element1, 1));
  fusion1_builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, fusion1_element0, fusion1_element2));
  HloComputation* fusion1_computation =
      module->AddEmbeddedComputation(fusion1_builder.Build());

  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion0 = builder.AddInstruction(HloInstruction::CreateFusion(
      nested_tuple_shape, HloInstruction::FusionKind::kCustom, {p0, p0},
      fusion0_computation));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* fusion1 = builder.AddInstruction(
      HloInstruction::CreateFusion(shape, HloInstruction::FusionKind::kCustom,
                                   {fusion0}, fusion1_computation));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation, {p0, fusion0, negate0, negate1, negate2, negate3, negate4,
                    negate5, negate6, fusion1});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), DefaultMemorySpaceOptions(), 5);

  EXPECT_THAT(
      fusion1,
      op::Fusion(op::Tuple(
          op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                        op::GetTupleElement(op::Fusion(), 0)),
          op::Tuple(
              op::AsyncCopy(
                  kAlternateMemorySpace, kDefaultMemorySpace,
                  op::GetTupleElement(op::GetTupleElement(op::Fusion(), 1), 0)),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::GetTupleElement(
                                op::GetTupleElement(op::Fusion(), 1), 1))))));
}

TEST_F(MemorySpaceAssignmentTest, TupleToTuple3) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder fusion0_builder("fusion0");
  HloInstruction* fusion0_param0 = fusion0_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion0_param1 = fusion0_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "p1"));
  fusion0_builder.AddInstruction(
      HloInstruction::CreateTuple({fusion0_param0, fusion0_param1}));
  HloComputation* fusion0_computation =
      module->AddEmbeddedComputation(fusion0_builder.Build());

  HloComputation::Builder fusion1_builder("fusion1");
  HloInstruction* fusion1_param = fusion1_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* fusion1_element0 = fusion1_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion1_param, 0));
  HloInstruction* fusion1_element1 = fusion1_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion1_param, 1));
  fusion1_builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, fusion1_element0, fusion1_element1));
  HloComputation* fusion1_computation =
      module->AddEmbeddedComputation(fusion1_builder.Build());

  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* fusion0 = builder.AddInstruction(HloInstruction::CreateFusion(
      tuple_shape, HloInstruction::FusionKind::kCustom, {p0, p0},
      fusion0_computation));
  HloInstruction* fusion1 = builder.AddInstruction(
      HloInstruction::CreateFusion(shape, HloInstruction::FusionKind::kCustom,
                                   {fusion0}, fusion1_computation));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, fusion0, fusion1});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());
  EXPECT_THAT(fusion1, op::Fusion(op::Fusion()));
}

TEST_F(MemorySpaceAssignmentTest, InputOutputAlias) {
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});
  HloInstruction* p = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p, 0));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p, 1));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));
  HloInstruction* negate7 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, add));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({p0, add}));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation, {p, p0, negate0, negate1, negate2, negate3, negate4, negate5,
                    negate6, p1, add, negate7, tuple});
  TF_CHECK_OK(module->set_schedule(schedule));

  // Make input {0} alias with output {0} and input {1} alias with output {1}.
  TF_CHECK_OK(module->input_output_alias_config().SetUpAlias({0}, 0, {0}));
  TF_CHECK_OK(module->input_output_alias_config().SetUpAlias({1}, 0, {1}));

  AssignMemorySpace(module.get());

  // Make sure the input is in the default memory space.
  EXPECT_EQ(p->shape().tuple_shapes(0).layout().memory_space(),
            kDefaultMemorySpace);
  EXPECT_EQ(p->shape().tuple_shapes(1).layout().memory_space(),
            kDefaultMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, CostAnalysis) {
  // This is mostly a smoke test since it's difficult and brittle to work out
  // the cost of the HLO instructions.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, negate6, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate0, negate1, negate2,
                                      negate3, negate4, negate5, negate6, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpaceUsingCostAnalysis(module.get());
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  // Negate instructions are in the alternate memory space (1).
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
}

TEST_F(MemorySpaceAssignmentTest, MemoryBoundednessBufferIntervalCompare) {
  // This test is carefully crafted to force only negates to be allocated to the
  // alternate memory. The graph consists of interleaving negate and tanh
  // operations:
  //
  //        +------+      +-------+      +-----
  //       /        \    /         \    /
  //  negate  tanh  negate  tanh   negate  tanh
  //             \          /  \           /
  //              +--------+    +---------+
  //
  // The alternate memory is sized to fit only two f32[4,3] tensors at a time.
  // Also, transcendentals are made to be lower bandwidth than FLOPs. So, the
  // MemoryBoundednessBufferIntervalCompare should prioritize the negates, which
  // are more memory bound.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* tanh0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p1));
  HloInstruction* tanh1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, tanh0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* tanh2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, tanh1));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* tanh3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, tanh2));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* tanh4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, tanh3));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({tanh4, negate4}));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation,
                        {p0, p1, tanh0, negate0, tanh1, negate1, tanh2, negate2,
                         tanh3, negate3, tanh4, negate4, tuple});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpaceUsingCostAnalysis(module.get());
  // Parameters are in the default memory space.
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  Shape shape_in_default_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {4, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      kDefaultMemorySpace);
  // Expect only negates to be in alternate memory space. Not all might fit but
  // make sure at least one does.
  std::vector<HloInstruction*> negate_instructions = {negate0, negate1, negate2,
                                                      negate3, negate4};
  int64_t num_negates_in_alternate_mem = absl::c_count_if(
      negate_instructions, [&](const HloInstruction* instruction) {
        return instruction->shape().layout().memory_space() ==
               kAlternateMemorySpace;
      });
  EXPECT_GE(num_negates_in_alternate_mem, 1);
  EXPECT_THAT(tanh0, op::ShapeWithLayout(shape_in_default_mem));
  EXPECT_THAT(tanh1, op::ShapeWithLayout(shape_in_default_mem));
  EXPECT_THAT(tanh2, op::ShapeWithLayout(shape_in_default_mem));
  EXPECT_THAT(tanh3, op::ShapeWithLayout(shape_in_default_mem));
  EXPECT_THAT(tanh4, op::ShapeWithLayout(shape_in_default_mem));
}

TEST_F(MemorySpaceAssignmentTest,
       MemoryBoundednessOverrideSortOrderAssignFirst) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[3,4]{1,0} parameter(0)
    p1 = f32[3,4]{1,0} parameter(1)
    tanh0 = f32[3,4]{1,0} tanh(p0)
    negate0 = f32[3,4]{1,0} negate(p1)
    tanh1 = f32[3,4]{1,0} tanh(tanh0)
    negate1 = f32[3,4]{1,0} negate(negate0)
    tanh2 = f32[3,4]{1,0} tanh(tanh1)
    negate2 = f32[3,4]{1,0} negate(negate1)
    tanh3 = f32[3,4]{1,0} tanh(tanh2)
    negate3 = f32[3,4]{1,0} negate(negate2)
    tanh4 = f32[3,4]{1,0} tanh(tanh3)
    negate4 = f32[3,4]{1,0} negate(negate3)
    ROOT tuple = (f32[3,4]{1,0}, f32[3,4]{1,0}) tuple(tanh4, negate4)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Override MSA sort order and try to assign all negates to alternate memory
  // first. Alternate memory size is enough to fit 2 f32[4,3] tensors at a time.
  const std::string text_proto = R"pb(
    overrides {
      hlo_position_matcher { instruction_name_regex: "negate(.*)" }
      override_options { assign_first: true }
    })pb";
  TF_ASSERT_OK_AND_ASSIGN(auto msa_sort_order_overrides,
                          ParseTextProto<MsaSortOrderOverrides>(text_proto));

  AssignMemorySpaceUsingCostAnalysis(
      module.get(), /*memory_space_options_override=*/std::nullopt,
      /*cost_analysis_options_override=*/std::nullopt,
      /*hlo_cost_options_override=*/std::nullopt,
      /*optional_msa_sort_order_overrides=*/msa_sort_order_overrides);
  // Parameters are in the default memory space.
  const HloInstruction* p0 = FindInstruction(module.get(), "p0");
  EXPECT_EQ(p0->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* p1 = FindInstruction(module.get(), "p1");
  EXPECT_EQ(p1->shape().layout().memory_space(), kDefaultMemorySpace);
  // Check that all negates are in alternate memory space except negate4.
  // negate4 is a program output, so it has to land in default memory.
  HloInstruction* negate0 = FindInstruction(module.get(), "negate0");
  EXPECT_EQ(negate0->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate1 = FindInstruction(module.get(), "negate1");
  EXPECT_EQ(negate1->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate2 = FindInstruction(module.get(), "negate2");
  EXPECT_EQ(negate2->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate3 = FindInstruction(module.get(), "negate3");
  EXPECT_EQ(negate3->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate4 = FindInstruction(module.get(), "negate4");
  EXPECT_EQ(negate4->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh0 = FindInstruction(module.get(), "tanh0");
  EXPECT_EQ(tanh0->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh1 = FindInstruction(module.get(), "tanh1");
  EXPECT_EQ(tanh1->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh2 = FindInstruction(module.get(), "tanh2");
  EXPECT_EQ(tanh2->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh3 = FindInstruction(module.get(), "tanh3");
  EXPECT_EQ(tanh3->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh4 = FindInstruction(module.get(), "tanh4");
  EXPECT_EQ(tanh4->shape().layout().memory_space(), kDefaultMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest,
       MemoryBoundednessOverrideSortOrderAssignLast) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[3,4]{1,0} parameter(0)
    p1 = f32[3,4]{1,0} parameter(1)
    tanh0 = f32[3,4]{1,0} tanh(p0)
    negate0 = f32[3,4]{1,0} negate(p1)
    tanh1 = f32[3,4]{1,0} tanh(tanh0)
    negate1 = f32[3,4]{1,0} negate(negate0)
    tanh2 = f32[3,4]{1,0} tanh(tanh1)
    negate2 = f32[3,4]{1,0} negate(negate1)
    tanh3 = f32[3,4]{1,0} tanh(tanh2)
    negate3 = f32[3,4]{1,0} negate(negate2)
    tanh4 = f32[3,4]{1,0} tanh(tanh3)
    negate4 = f32[3,4]{1,0} negate(negate3)
    ROOT tuple = (f32[3,4]{1,0}, f32[3,4]{1,0}) tuple(tanh4, negate4)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Override MSA sort order and try to assign all tanhs to alternate memory
  // last. Alternate memory size is enough to fit 2 f32[4,3] tensors at a time.
  const std::string text_proto = R"pb(
    overrides {
      hlo_position_matcher { instruction_name_regex: "tanh(.*)" }
      override_options { assign_last: true }
    }
  )pb";
  TF_ASSERT_OK_AND_ASSIGN(auto msa_sort_order_overrides,
                          ParseTextProto<MsaSortOrderOverrides>(text_proto));

  AssignMemorySpaceUsingCostAnalysis(
      module.get(), /*memory_space_options_override=*/std::nullopt,
      /*cost_analysis_options_override=*/std::nullopt,
      /*hlo_cost_options_override=*/std::nullopt,
      /*optional_msa_sort_order_overrides=*/msa_sort_order_overrides);
  // Parameters are in the default memory space.
  const HloInstruction* p0 = FindInstruction(module.get(), "p0");
  EXPECT_EQ(p0->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* p1 = FindInstruction(module.get(), "p1");
  EXPECT_EQ(p1->shape().layout().memory_space(), kDefaultMemorySpace);
  HloInstruction* negate0 = FindInstruction(module.get(), "negate0");
  EXPECT_EQ(negate0->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate1 = FindInstruction(module.get(), "negate1");
  EXPECT_EQ(negate1->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate2 = FindInstruction(module.get(), "negate2");
  EXPECT_EQ(negate2->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate3 = FindInstruction(module.get(), "negate3");
  EXPECT_EQ(negate3->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate4 = FindInstruction(module.get(), "negate4");
  // negate4 is a program output, so it has to land in default memory.
  EXPECT_EQ(negate4->shape().layout().memory_space(), kDefaultMemorySpace);
  // Check that all tanhs are in default memory space.
  const HloInstruction* tanh0 = FindInstruction(module.get(), "tanh0");
  EXPECT_EQ(tanh0->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh1 = FindInstruction(module.get(), "tanh1");
  EXPECT_EQ(tanh1->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh2 = FindInstruction(module.get(), "tanh2");
  EXPECT_EQ(tanh2->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh3 = FindInstruction(module.get(), "tanh3");
  EXPECT_EQ(tanh3->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh4 = FindInstruction(module.get(), "tanh4");
  EXPECT_EQ(tanh4->shape().layout().memory_space(), kDefaultMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest,
       MemoryBoundednessOverrideSortOrderBySizeLteAssignFirst) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[3,4]{1,0} parameter(0)
    p1 = f32[5,4]{1,0} parameter(1)
    tanh0 = f32[3,4]{1,0} tanh(p0)
    negate0 = f32[5,4]{1,0} negate(p1)
    tanh1 = f32[3,4]{1,0} tanh(tanh0)
    negate1 = f32[5,4]{1,0} negate(negate0)
    tanh2 = f32[3,4]{1,0} tanh(tanh1)
    negate2 = f32[5,4]{1,0} negate(negate1)
    tanh3 = f32[3,4]{1,0} tanh(tanh2)
    negate3 = f32[5,4]{1,0} negate(negate2)
    tanh4 = f32[3,4]{1,0} tanh(tanh3)
    negate4 = f32[5,4]{1,0} negate(negate3)
    ROOT tuple = (f32[3,4]{1,0}, f32[5,4]{1,0}) tuple(tanh4, negate4)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Override MSA sort order and try to assign all buffers with size lesser
  // than or equal to 48 bytes to alternate memory first.
  const std::string text_proto = R"pb(
    overrides {
      hlo_position_matcher { size_lte: 48 }
      override_options { assign_first: true }
    }
  )pb";
  TF_ASSERT_OK_AND_ASSIGN(auto msa_sort_order_overrides,
                          ParseTextProto<MsaSortOrderOverrides>(text_proto));

  Options memory_space_options = DefaultMemorySpaceOptions();
  // Set max size to 120 bytes, such that 2 f32[4,3] tensors can fit in
  // alternate memory at the same time but not 1 f32[4,3] tensor and 1
  // f32[4,5] tensor. If the max size was 128 bytes, negate3 would be assigned
  // to alternate memory.
  memory_space_options.max_size_in_bytes = 120;
  AssignMemorySpaceUsingCostAnalysis(
      module.get(), memory_space_options,
      /*cost_analysis_options_override=*/std::nullopt,
      /*hlo_cost_options_override=*/std::nullopt,
      /*optional_msa_sort_order_overrides=*/msa_sort_order_overrides);
  // Parameters are in the default memory space.
  const HloInstruction* p0 = FindInstruction(module.get(), "p0");
  EXPECT_EQ(p0->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* p1 = FindInstruction(module.get(), "p1");
  EXPECT_EQ(p1->shape().layout().memory_space(), kDefaultMemorySpace);
  HloInstruction* negate0 = FindInstruction(module.get(), "negate0");
  EXPECT_EQ(negate0->shape().layout().memory_space(), kDefaultMemorySpace);
  HloInstruction* negate1 = FindInstruction(module.get(), "negate1");
  EXPECT_EQ(negate1->shape().layout().memory_space(), kDefaultMemorySpace);
  HloInstruction* negate2 = FindInstruction(module.get(), "negate2");
  EXPECT_EQ(negate2->shape().layout().memory_space(), kDefaultMemorySpace);
  HloInstruction* negate3 = FindInstruction(module.get(), "negate3");
  EXPECT_EQ(negate3->shape().layout().memory_space(), kDefaultMemorySpace);
  HloInstruction* negate4 = FindInstruction(module.get(), "negate4");
  EXPECT_EQ(negate4->shape().layout().memory_space(), kDefaultMemorySpace);
  // Check that all tanhs are in alternate memory space except tanh4. tanh4
  // is a program output, so it has to land in default memory.
  const HloInstruction* tanh0 = FindInstruction(module.get(), "tanh0");
  EXPECT_EQ(tanh0->shape().layout().memory_space(), kAlternateMemorySpace);
  const HloInstruction* tanh1 = FindInstruction(module.get(), "tanh1");
  EXPECT_EQ(tanh1->shape().layout().memory_space(), kAlternateMemorySpace);
  const HloInstruction* tanh2 = FindInstruction(module.get(), "tanh2");
  EXPECT_EQ(tanh2->shape().layout().memory_space(), kAlternateMemorySpace);
  const HloInstruction* tanh3 = FindInstruction(module.get(), "tanh3");
  EXPECT_EQ(tanh3->shape().layout().memory_space(), kAlternateMemorySpace);
  const HloInstruction* tanh4 = FindInstruction(module.get(), "tanh4");
  EXPECT_EQ(tanh4->shape().layout().memory_space(), kDefaultMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest,
       MemoryBoundednessOverrideSortOrderBySizeGteAssignFirst) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[3,4]{1,0} parameter(0)
    p1 = f32[5,4]{1,0} parameter(1)
    tanh0 = f32[3,4]{1,0} tanh(p0)
    negate0 = f32[5,4]{1,0} negate(p1)
    tanh1 = f32[3,4]{1,0} tanh(tanh0)
    negate1 = f32[5,4]{1,0} negate(negate0)
    tanh2 = f32[3,4]{1,0} tanh(tanh1)
    negate2 = f32[5,4]{1,0} negate(negate1)
    tanh3 = f32[3,4]{1,0} tanh(tanh2)
    negate3 = f32[5,4]{1,0} negate(negate2)
    tanh4 = f32[3,4]{1,0} tanh(tanh3)
    negate4 = f32[5,4]{1,0} negate(negate3)
    ROOT tuple = (f32[3,4]{1,0}, f32[5,4]{1,0}) tuple(tanh4, negate4)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Override MSA sort order and try to assign all buffers with size greater
  // than or equal to 80 bytes to alternate memory first.
  const std::string text_proto = R"pb(
    overrides {
      hlo_position_matcher { size_gte: 80 }
      override_options { assign_first: true }
    }
  )pb";
  TF_ASSERT_OK_AND_ASSIGN(auto msa_sort_order_overrides,
                          ParseTextProto<MsaSortOrderOverrides>(text_proto));

  Options memory_space_options = DefaultMemorySpaceOptions();
  // Set max size to 160 bytes to allow 2 f32[4,5] tensors to fit in alternate
  // memory at the same time. tanh3 would not be prefetched because negate2 and
  // negate3 would be in alternate memory at the same time leaving no space for
  // tanh3.
  memory_space_options.max_size_in_bytes = 160;
  AssignMemorySpaceUsingCostAnalysis(
      module.get(), memory_space_options,
      /*cost_analysis_options_override=*/std::nullopt,
      /*hlo_cost_options_override=*/std::nullopt,
      /*optional_msa_sort_order_overrides=*/msa_sort_order_overrides);
  // Parameters are in the default memory space.
  const HloInstruction* p0 = FindInstruction(module.get(), "p0");
  EXPECT_EQ(p0->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* p1 = FindInstruction(module.get(), "p1");
  EXPECT_EQ(p1->shape().layout().memory_space(), kDefaultMemorySpace);
  // Check that all negates are in alternate memory space except negate4.
  // negate4 is a program output, so it has to land in default memory.
  HloInstruction* negate0 = FindInstruction(module.get(), "negate0");
  EXPECT_EQ(negate0->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate1 = FindInstruction(module.get(), "negate1");
  EXPECT_EQ(negate1->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate2 = FindInstruction(module.get(), "negate2");
  EXPECT_EQ(negate2->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate3 = FindInstruction(module.get(), "negate3");
  EXPECT_EQ(negate3->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate4 = FindInstruction(module.get(), "negate4");
  EXPECT_EQ(negate4->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh0 = FindInstruction(module.get(), "tanh0");
  EXPECT_EQ(tanh0->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh1 = FindInstruction(module.get(), "tanh1");
  EXPECT_EQ(tanh1->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh2 = FindInstruction(module.get(), "tanh2");
  EXPECT_EQ(tanh2->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh3 = FindInstruction(module.get(), "tanh3");
  EXPECT_EQ(tanh3->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh4 = FindInstruction(module.get(), "tanh4");
  EXPECT_EQ(tanh4->shape().layout().memory_space(), kDefaultMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest,
       MemoryBoundednessOverrideSortOrderByUseAssignFirst) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[3,4]{1,0} parameter(0)
    p1 = f32[3,4]{1,0} parameter(1)
    tanh0 = f32[3,4]{1,0} tanh(p0)
    negate0 = f32[3,4]{1,0} negate(p1)
    tanh1 = f32[3,4]{1,0} tanh(tanh0)
    negate1 = f32[3,4]{1,0} negate(negate0)
    tanh2 = f32[3,4]{1,0} tanh(tanh1)
    negate2 = f32[3,4]{1,0} negate(negate1)
    tanh3 = f32[3,4]{1,0} tanh(tanh2)
    negate3 = f32[3,4]{1,0} negate(negate2)
    tanh4 = f32[3,4]{1,0} tanh(tanh3)
    negate4 = f32[3,4]{1,0} negate(negate3)
    ROOT tuple = (f32[3,4]{1,0}, f32[3,4]{1,0}) tuple(tanh4, negate4)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Override MSA sort order and try to assign all negates to alternate memory
  // first. Alternate memory size is enough to fit 2 f32[4,3] tensors at a time.
  const std::string text_proto = R"pb(
    overrides {
      hlo_position_matcher {
        hlo_use_filter { instruction_name_regex: "negate(.*)" }
      }
      override_options { assign_first: true }
    })pb";
  TF_ASSERT_OK_AND_ASSIGN(auto msa_sort_order_overrides,
                          ParseTextProto<MsaSortOrderOverrides>(text_proto));

  AssignMemorySpaceUsingCostAnalysis(
      module.get(), /*memory_space_options_override=*/std::nullopt,
      /*cost_analysis_options_override=*/std::nullopt,
      /*hlo_cost_options_override=*/std::nullopt,
      /*optional_msa_sort_order_overrides=*/msa_sort_order_overrides);
  // Parameters are in the default memory space.
  const HloInstruction* p0 = FindInstruction(module.get(), "p0");
  EXPECT_EQ(p0->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* p1 = FindInstruction(module.get(), "p1");
  EXPECT_EQ(p1->shape().layout().memory_space(), kDefaultMemorySpace);
  // Check that all negates are in alternate memory space except negate4.
  // negate4 is a program output, so it has to land in default memory.
  HloInstruction* negate0 = FindInstruction(module.get(), "negate0");
  EXPECT_EQ(negate0->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate1 = FindInstruction(module.get(), "negate1");
  EXPECT_EQ(negate1->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate2 = FindInstruction(module.get(), "negate2");
  EXPECT_EQ(negate2->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate3 = FindInstruction(module.get(), "negate3");
  EXPECT_EQ(negate3->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* negate4 = FindInstruction(module.get(), "negate4");
  EXPECT_EQ(negate4->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh0 = FindInstruction(module.get(), "tanh0");
  EXPECT_EQ(tanh0->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh1 = FindInstruction(module.get(), "tanh1");
  EXPECT_EQ(tanh1->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh2 = FindInstruction(module.get(), "tanh2");
  EXPECT_EQ(tanh2->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh3 = FindInstruction(module.get(), "tanh3");
  EXPECT_EQ(tanh3->shape().layout().memory_space(), kDefaultMemorySpace);
  const HloInstruction* tanh4 = FindInstruction(module.get(), "tanh4");
  EXPECT_EQ(tanh4->shape().layout().memory_space(), kDefaultMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, SimpleWhileTupleTest) {
  Shape s32 = ShapeUtil::MakeShape(xla::S32, {});
  Shape f32v1 = ShapeUtil::MakeShape(F32, {1});
  Shape t_s32_f32v1 = ShapeUtil::MakeTupleShape({s32, f32v1});
  auto module = CreateNewVerifiedModule("SimpleWhile");
  HloSchedule schedule(module.get());

  // A simple compare-to-limit (x < 4) computation for a While.
  //
  // condition:
  //   const4[s32] -----------------------------------\
  //                                                   \
  //   param[(s32,f32[4])] --- get-tuple-element[0] --- less-than
  //
  HloComputation* cond_computation;
  {
    auto builder = HloComputation::Builder("WhileCond");
    auto const4 = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(4)));
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, t_s32_f32v1, "x"));
    auto index = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(const4->shape(), param, 0));
    auto compare = builder.AddInstruction(
        HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), index,
                                      const4, ComparisonDirection::kLt));
    cond_computation = module->AddEmbeddedComputation(builder.Build());
    schedule.set_sequence(cond_computation, {const4, param, index, compare});
  }

  // Builds a simple body computation for a While.
  //
  // body:
  //   constv[f32[1]] --------------------------------------\
  //                                                         \
  //                           /--- get-tuple-elementv[1] --- addv ---\
  //   param[(s32,f32[1])] ---|                                    tuple
  //                           \--- get-tuple-elementc[0] --- addc ---/
  //                                                         /
  //   const1[s32] -----------------------------------------/
  //
  HloComputation* body_computation;
  {
    auto builder = HloComputation::Builder("WhileBody");
    auto const1 = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(1)));
    auto constv = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({1.1f})));
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, t_s32_f32v1, "x"));
    auto indexc = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(const1->shape(), param, 0));
    auto addc = builder.AddInstruction(HloInstruction::CreateBinary(
        indexc->shape(), HloOpcode::kAdd, indexc, const1));
    auto indexv = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(constv->shape(), param, 1));
    auto addv = builder.AddInstruction(HloInstruction::CreateBinary(
        constv->shape(), HloOpcode::kAdd, indexv, constv));
    auto tuple =
        builder.AddInstruction(HloInstruction::CreateTuple({addc, addv}));
    body_computation = module->AddEmbeddedComputation(builder.Build());
    schedule.set_sequence(body_computation, {const1, constv, param, indexc,
                                             addc, indexv, addv, tuple});
  }

  // This tests a simple while loop where the parameters are aliased with the
  // output buffers.
  auto builder = HloComputation::Builder("SimpleWhile");
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, t_s32_f32v1, "param"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(s32, param, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32v1, param, 1));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({gte0, gte1}));
  auto while0 = builder.AddInstruction(HloInstruction::CreateWhile(
      t_s32_f32v1, cond_computation, body_computation, tuple));

  HloComputation* computation = module->AddEntryComputation(builder.Build());
  schedule.set_sequence(computation, {param, gte0, gte1, tuple, while0});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get(), DefaultMemorySpaceOptions(),
                    /*max_prefetch_interval=*/50);

  // Ensure all parameters and while are placed in default memory.
  Shape shape_in_default_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {4, 6},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      kDefaultMemorySpace);
  Shape s32_in_default_mem = ShapeUtil::MakeShapeWithDenseLayout(
      xla::S32, {},
      /*minor_to_major=*/{}, /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      kDefaultMemorySpace);
  Shape f32v1_in_default_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {1},
      /*minor_to_major=*/{0}, /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      kDefaultMemorySpace);
  Shape t_s32_f32v1_in_default_mem =
      ShapeUtil::MakeTupleShape({s32_in_default_mem, f32v1_in_default_mem});
  EXPECT_THAT(param, op::ShapeWithLayout(t_s32_f32v1_in_default_mem));
  EXPECT_THAT(while0, op::ShapeWithLayout(t_s32_f32v1_in_default_mem));
}

TEST_F(MemorySpaceAssignmentTest, EvictionsShouldntBeDelayed) {
  // This test reproduces an eviction scheduling bug where evictions to default
  // memory can happen later than intended, causing memory corruption. This test
  // is a variant of MemoryBoundednessBufferIntervalCompare but uses f32[4,3]
  // tensors instead, so at most two tensors should fit in the alternate memory
  // space at a given time. We have a number of redundant operations
  // (tanh_redundant ops) that do not have users. The bug was due to
  // SimplifyGraph removing dead instructions, and removing them from the
  // schedule. However, the CopyStart/CopyDone insertion relies on the schedule
  // indexes, so they could be inserted too late.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* tanh0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* tanh_redundant0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* tanh_redundant1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* tanh_redundant2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* tanh_redundant3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* tanh_redundant4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* tanh_redundant5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* tanh_redundant6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, tanh0));
  HloInstruction* tanh1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, negate0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* tanh2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, tanh1));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* tanh3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kTanh, tanh2));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({tanh3, negate3, tanh0}));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation,
      {p0, tanh0, tanh_redundant0, tanh_redundant1, tanh_redundant2,
       tanh_redundant3, tanh_redundant4, tanh_redundant5, tanh_redundant6,
       negate0, tanh1, negate1, tanh2, negate2, tanh3, negate3, tuple});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpaceUsingCostAnalysis(module.get());

  TF_ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                          HloAliasAnalysis::Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_live_range,
                          HloLiveRange::Run(module->schedule(), *alias_analysis,
                                            module->entry_computation()));

  std::vector<int> num_live_buffers_in_alternate_mem(
      hlo_live_range->flattened_instruction_sequence().size() + 1, 0);

  // Go through each value and for those that are allocated in the alternate
  // memory space, increment (inclusive) num_live_buffers_in_alternate_mem for
  // every time step that they are live.
  for (const HloValue* value : alias_analysis->dataflow_analysis().values()) {
    const Shape& shape = value->shape();
    if (!shape.has_layout() ||
        shape.layout().memory_space() == kDefaultMemorySpace) {
      continue;
    }

    HloLiveRange::TimeBound time_bound =
        hlo_live_range->buffer_live_ranges().at(value);
    for (int i = time_bound.start; i <= time_bound.end; ++i) {
      ++num_live_buffers_in_alternate_mem[i];
    }
  }

  // The test memory can at most hold two f32[4,3] buffers at a time. If there
  // is more than that, it means we have memory corruption.
  for (int i = 0; i < num_live_buffers_in_alternate_mem.size(); ++i) {
    EXPECT_LE(num_live_buffers_in_alternate_mem[i], 2);
  }
}

TEST_F(MemorySpaceAssignmentTest,
       InputOutputsInAlternateMemShouldntBeAssigned) {
  // When input/outputs are marked to be in the alternate memory (e.g.
  // go/tpu-fast-mem-inference), do not allocate those and assume they will live
  // in the alternate memory for the entire computation. The BufferAssignment
  // pass, which is run after this, will allocate those buffers.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
  Shape shape_in_default_mem = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      kDefaultMemorySpace);
  // p0 is in the default memory space.
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  // p1 is in the alternate memory space.
  HloInstruction* p1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape_in_alternate_mem, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, negate5));
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      shape_in_alternate_mem, HloOpcode::kAdd, negate6, p1));
  // Index {0} of the root instruction is in the alternate memory space, index
  // {1} is in the default memory space.
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({add, negate5}));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation,
                        {p0, p1, negate0, negate1, negate2, negate3, negate4,
                         negate5, negate6, add, tuple});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options = DefaultMemorySpaceOptions();
  options.is_allowed_in_alternate_mem_fn = [](const HloValue& value) {
    return true;
  };
  XLA_VLOG_LINES(3, module->ToString());
  std::unique_ptr<PresetAssignments> preset_assignments =
      AssignMemorySpace(module.get(), options);
  XLA_VLOG_LINES(3, module->ToString());
  // Ensure that p1 is in the alternate memory and add, which has p1 as an
  // operand, has a direct dependency to p1 (no CopyStart/CopyDone).
  EXPECT_THAT(p1, op::ShapeWithLayout(shape_in_default_mem));
  EXPECT_THAT(add, op::Add(op::Negate(), op::CopyDone()));
  // Make sure add is still in the alternate memory space.
  EXPECT_THAT(add, op::ShapeWithLayout(shape_in_alternate_mem));

  // Check the preset assignments and ensure the inputs/outputs in the alternate
  // memory space aren't in the preset assignments. Inputs/outputs in the
  // alternate memory space are left to BufferAssignment to be allocated.
  for (const auto& position_and_chunk : preset_assignments->chunks()) {
    const HloPosition& position = position_and_chunk.first;
    XLA_VLOG_LINES(3, position.instruction->ToString());
    EXPECT_NE(position.instruction, p1);
    EXPECT_NE(position.instruction, add);
  }
}

TEST_F(MemorySpaceAssignmentTest, PendingChunkMemoryCorruptionBug) {
  // Tests a memory corruption bug where the allocated chunk overlaps with a
  // pending chunk. To test this, we provide a new buffer interval compare where
  // we prioritize the allocation of sine, cosine, and tanh to create the
  // situation:
  //
  //    Max memory
  //  -------------------------------------------
  //      +------------+
  //      |     b      |
  //      +------------+
  //  +-------+
  //  |       |
  //  |       |
  //  |   a   |
  //  |       |                 +------------+
  //  |       |                 |     n      |
  //  +-------+                 +------------+
  //  -------------------------------------------
  //    Min memory          time ->
  //
  //
  // Then allocating for buffer d, we have these two prefetch buffers
  // overlapping:
  //
  //    Max memory
  //  -------------------------------------------
  //      +------------+ +----------+
  //      |     b      | | prefetch |
  //      +------------+ | for o    |
  //  +-------+     +---------+     |
  //  |       |     |    |    |     |
  //  |       |     |    |    |     |
  //  |   a   |     |    +----|-----+
  //  |       |     | prefetch| +------------+
  //  |       |     | for m   | |     n      |
  //  +-------+     +---------+ +------------+
  //  -------------------------------------------
  //    Min memory          time ->
  //
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  ENTRY %Entry {
    %param0 = f32[8,3] parameter(0)
    %param1 = f32[2,4] parameter(1)
    %a = f32[8,3] sine(%param0)
    %b = f32[2,4] cosine(%param1)
    %d = f32[8,3] tanh(%a)
    %c = f32[8,3] negate(%a)
    %e = f32[2,4] negate(%b)
    %f = f32[2,4] negate(%e)
    %g = f32[2,4] negate(%f)
    %h = f32[2,4] negate(%g)
    %i = f32[2,4] negate(%h)
    %j = f32[2,4] negate(%i)
    %k = f32[2,4] negate(%j)
    %l = f32[2,4] negate(%k)
    %m = f32[8,3] negate(%d)
    %n = f32[2,4] sine(%l)
    %o = f32[8,3] negate(%d)
    %p = f32[2,4] negate(%n)
    %q = f32[8,3] negate(%m)
    ROOT %tuple = (f32[2,4], f32[8,3], f32[8,3]) tuple(%p, %q, %o)
  }
  )";

  MsaBufferIntervalCompare buffer_interval_compare =
      [](const MsaBufferInterval& a, const MsaBufferInterval& b) {
        auto get_opcode_priority = [](const HloOpcode& opcode) {
          switch (opcode) {
            case HloOpcode::kSin:
              return 0;
            case HloOpcode::kCos:
              return 1;
            case HloOpcode::kTanh:
              return 2;
            default:
              return 3;
          }
        };

        return get_opcode_priority(a.buffer->defining_instruction()->opcode()) <
               get_opcode_priority(b.buffer->defining_instruction()->opcode());
      };
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 10);
  AssignMemorySpace(module.get(), DefaultMemorySpaceOptions(),
                    buffer_interval_compare, &prefetch_interval_picker);
}

TEST_F(MemorySpaceAssignmentTest, WhileAliasedArgumentRequiredAssignmentBug) {
  // Tests an overly pessimistic assertion when the same HloValue is passed
  // multiple times to a while HLO. We already handle this case that the two
  // arguments must alias and get the same allocation in AllocateSegment so the
  // assertion isn't necessary.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  while_condition {
    param1 = (f32[2,4], f32[2,4], f32[2,4]) parameter(0)
    ROOT cond = pred[] constant(true)
  }

  while_body {
    param2 = (f32[2,4], f32[2,4], f32[2,4]) parameter(0)
    gte2 = f32[2,4] get-tuple-element(param2), index=0
    gte3 = f32[2,4] get-tuple-element(param2), index=1
    gte4 = f32[2,4] get-tuple-element(param2), index=2
    add = f32[2,4] add(gte2, gte3)
    ROOT tuple2 = (f32[2,4], f32[2,4], f32[2,4]) tuple(add, gte3, gte4)
  }

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)
    a = f32[2,4] negate(param0)
    b = f32[2,4] negate(param0)
    tuple = (f32[2,4], f32[2,4], f32[2,4]) tuple(a, b, b)
    while = (f32[2,4], f32[2,4], f32[2,4]) while(tuple), condition=while_condition, body=while_body
    gte1 = f32[2,4] get-tuple-element(while), index=0
    gte2 = f32[2,4] get-tuple-element(while), index=1
    ROOT root = f32[2,4] add(gte1, gte2)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, DisallowedUseBug) {
  // When we have a disallowed use (in this case tanh), we aren't allowed to
  // allocate this use in alternate memory. However, if we have another use
  // after this on the same buffer (o), this use may refer to "a" instead of the
  // evicted value, which is illegal because "a" will be allocated in the
  // alternate memory space.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  ENTRY Entry {
    param0 = f32[8,3] parameter(0)
    param1 = f32[2,4] parameter(1)
    a = f32[8,3] cosine(param0)
    b = f32[2,4] negate(param1)
    d = f32[8,3] negate(a)
    c = f32[2,4] negate(b)
    e = f32[2,4] negate(c)
    f = f32[8,3] tanh(a)
    g = f32[2,4] negate(e)
    h = f32[2,4] negate(g)
    i = f32[2,4] negate(h)
    j = f32[2,4] negate(i)
    k = f32[2,4] negate(j)
    l = f32[2,4] negate(k)
    m = f32[2,4] negate(l)
    n = f32[2,4] sine(m)
    o = f32[8,3] negate(a)
    p = f32[2,4] negate(n)
    q = f32[8,3] add(o, f)
    r = f32[8,3] add(q, d)
    ROOT tuple = (f32[2,4], f32[8,3]) tuple(p, r)
  }
  )";

  MsaBufferIntervalCompare buffer_interval_compare =
      [](const MsaBufferInterval& a, const MsaBufferInterval& b) {
        auto get_opcode_priority = [](const HloOpcode& opcode) {
          switch (opcode) {
            case HloOpcode::kSin:
              return 0;
            case HloOpcode::kCos:
              return 1;
            case HloOpcode::kTanh:
              return 2;
            default:
              return 3;
          }
        };

        return get_opcode_priority(a.buffer->defining_instruction()->opcode()) <
               get_opcode_priority(b.buffer->defining_instruction()->opcode());
      };
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 10);
  Options options = DefaultMemorySpaceOptions();
  options.is_use_allowed_in_alternate_mem_fn = [](const HloUse& use) {
    return use.instruction->opcode() != HloOpcode::kTanh;
  };
  AssignMemorySpace(module.get(), options, buffer_interval_compare,
                    &prefetch_interval_picker);
}

TEST_F(MemorySpaceAssignmentTest, DisallowedUseBugInWhile) {
  // Test for situations where we disallow a use (tanh in this case) in the
  // alternate memory space and there is a subsequent use that also requires the
  // buffer to be in the default memory space. In this case, the allocation in
  // the default memory space might not be the very last one, so we need to
  // search the allocation sequence and find the one in the default memory
  // space.
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=3
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = f32[3]{0} get-tuple-element(p0), index=2
    gte3 = pred[] get-tuple-element(p0), index=3
    add = f32[3]{0} add(gte0, gte0)
    negate0 = f32[3]{0} negate(add)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    negate15 = f32[3]{0} negate(gte2)
    tanh = f32[3]{0} tanh(gte2)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) tuple(negate14, tanh, gte2, gte3)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy0 = f32[3]{0} copy(p0)
    copy1 = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) tuple(copy0, copy0, copy1, p1)
    while = (f32[3]{0}, f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    ROOT gte = f32[3]{0} get-tuple-element(while), index=2
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.is_use_allowed_in_alternate_mem_fn = [](const HloUse& use) {
    return use.instruction->opcode() != HloOpcode::kTanh;
  };
  AssignMemorySpace(module.get(), options);
}

TEST_F(MemorySpaceAssignmentTest, TwoLiveAllocationValuesBase) {
  // In this example, we have enough space to give negate.0 alternate memory,
  // and we put put negate.0 at the top of MSA's sort order. So, we expect that
  // it will get alternate memory.
  //
  // We are testing a fix for dual live AllocationsValues, with the following
  // setup:
  // - HloValue H containing the following positions: negate.0, cp-start.0{0}
  //   - AllocationValue A0 defined at negate.0
  //     - Segment A0.S0 define during [negate.0, cp-start.0]
  //.    - Segment A0.S1 defined during [cp-start.0, add.0]
  //   - AllocationValue A1 defined at cp-start.0{0}
  //     - Segment A1.S0 defined during [cp-start.0, cp-done.0]
  //
  // A0 and A1 are both live for more than 1 instruction.
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    /*00:*/ p.0 = f32[10,10,10,10] parameter(0)
    /*01:*/ p.1 = f32[10,10,10,10] parameter(1)
    /*02:*/ v.0 = f32[10,10,10,10] add(p.1, p.1)
    /*03:*/ negate.0 = f32[10,10,10,10] negate(p.0)
    /*04:*/ cp-start.0 = (f32[10,10,10,10], f32[10,10,10,10], u32[], u32[]) collective-permute-start(negate.0), source_target_pairs={{0,1},{2,3}}
    /*05:*/ v.1 = f32[10,10,10,10] add(v.0, v.0)
    /*06:*/ add.0 = f32[10,10,10,10] add(negate.0, negate.0)
    /*07:*/ v.2 = f32[10,10,10,10] add(v.1, v.1)
    /*08:*/ cp-done.0 = f32[10,10,10,10] collective-permute-done(cp-start.0)
    /*09:*/ v.3 = f32[10,10,10,10] add(v.2, v.2)
    /*10:*/ ROOT tuple.0 = (f32[10,10,10,10], f32[10,10,10,10], f32[10,10,10,10]) tuple(add.0, cp-done.0, v.3)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 4 * 10 * 10 * 10 * 10;
  MsaBufferIntervalCompare buffer_interval_compare =
      CreateBufferIntervalCompareFnFromInstructionNames({"negate.0"});
  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(1, 10);
  std::unique_ptr<PresetAssignments> preset_assignments =
      AssignMemorySpace(module.get(), options, buffer_interval_compare,
                        &prefetch_interval_picker);
  VLOG(1) << "Module after MSA:\n" << module->ToString();

  HloInstruction* copy0 = FindInstruction(module.get(), "negate.0");
  ASSERT_NE(copy0, nullptr);
  EXPECT_EQ(copy0->shape().layout().memory_space(), kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest,
       TwoLiveAllocationValuesTwoInstructionOverlap) {
  // In this example, we have enough space to give negate.0 alternate memory,
  // and we put put negate.0 at the top of MSA's sort order. So, we expect that
  // it will get alternate memory.
  //
  // We are testing a fix for dual live AllocationValues, with the following
  // setup:
  // - HloValue H containing the following positions: negate.0, cp-start.0{0}
  //   - AllocationValue A0 defined at negate.0
  //     - Segment A0.S0 define during [negate.0, cp-start.0]
  //.    - Segment A0.S1 defined during [cp-start.0, add.0]
  //   - AllocationValue A1 defined at cp-start.0{0}
  //     - Segment A1.S0 defined during [cp-start.0, cp-done.0]
  //
  // A0 and A1 are both live for 2 instructions
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    /*00:*/ p.0 = f32[10,10,10,10] parameter(0)
    /*01:*/ p.1 = f32[10,10,10,10] parameter(1)
    /*02:*/ v.0 = f32[10,10,10,10] add(p.1, p.1)
    /*03:*/ negate.0 = f32[10,10,10,10] negate(p.0)
    /*04:*/ cp-start.0 = (f32[10,10,10,10], f32[10,10,10,10], u32[], u32[]) collective-permute-start(negate.0), source_target_pairs={{0,1},{2,3}}
    /*05:*/ add.0 = f32[10,10,10,10] add(negate.0, negate.0)
    /*06:*/ v.1 = f32[10,10,10,10] add(v.0, v.0)
    /*07:*/ v.2 = f32[10,10,10,10] add(v.1, v.1)
    /*08:*/ cp-done.0 = f32[10,10,10,10] collective-permute-done(cp-start.0)
    /*09:*/ v.3 = f32[10,10,10,10] add(v.2, v.2)
    /*10:*/ ROOT tuple.0 = (f32[10,10,10,10], f32[10,10,10,10], f32[10,10,10,10]) tuple(add.0, cp-done.0, v.3)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 4 * 10 * 10 * 10 * 10;
  MsaBufferIntervalCompare buffer_interval_compare =
      CreateBufferIntervalCompareFnFromInstructionNames({"negate.0"});
  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(1, 10);
  std::unique_ptr<PresetAssignments> preset_assignments =
      AssignMemorySpace(module.get(), options, buffer_interval_compare,
                        &prefetch_interval_picker);
  VLOG(1) << "Module after MSA:\n" << module->ToString();

  HloInstruction* copy0 = FindInstruction(module.get(), "negate.0");
  ASSERT_NE(copy0, nullptr);
  EXPECT_EQ(copy0->shape().layout().memory_space(), kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest,
       TwoLiveAllocationValuesFirstLiveAllocationValueOutlastsSecond) {
  // In this example, we have enough space to give negate.0 alternate memory,
  // and we put put negate.0 at the top of MSA's sort order. So, we expect that
  // it will get alternate memory.
  //
  // We are testing a fix for dual live AllocationValues, with the following
  // setup:
  // - HloValue H containing the following positions: negate.0, cp-start.0{0}
  //   - AllocationValue A0 defined at negate.0
  //     - Segment A0.S0 define during [negate.0, cp-start.0]
  //.    - Segment A0.S1 defined during [cp-start.0, add.0]
  //     - Segment A0.S2 defined during [add.0, add.1]
  //   - AllocationValue A1 defined at cp-start.0{0}
  //     - Segment A1.S0 defined during [cp-start.0, cp-done.0]
  //
  // A0 and A1 are both live for more than 1 instruction. A0 is live beyond the
  // end of A1.
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    /*00:*/ p.0 = f32[10,10,10,10] parameter(0)
    /*01:*/ p.1 = f32[10,10,10,10] parameter(1)
    /*02:*/ v.0 = f32[10,10,10,10] add(p.1, p.1)
    /*03:*/ negate.0 = f32[10,10,10,10] negate(p.0)
    /*04:*/ cp-start.0 = (f32[10,10,10,10], f32[10,10,10,10], u32[], u32[]) collective-permute-start(negate.0), source_target_pairs={{0,1},{2,3}}
    /*05:*/ v.1 = f32[10,10,10,10] add(v.0, v.0)
    /*06:*/ add.0 = f32[10,10,10,10] add(negate.0, negate.0)
    /*07:*/ v.2 = f32[10,10,10,10] add(v.1, v.1)
    /*08:*/ cp-done.0 = f32[10,10,10,10] collective-permute-done(cp-start.0)
    /*09:*/ v.3 = f32[10,10,10,10] add(v.2, v.2)
    /*10:*/ add.1 = f32[10,10,10,10] add(add.0, negate.0)
    /*11:*/ ROOT tuple.0 = (f32[10,10,10,10], f32[10,10,10,10], f32[10,10,10,10]) tuple(add.1, cp-done.0, v.3)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 4 * 10 * 10 * 10 * 10;
  MsaBufferIntervalCompare buffer_interval_compare =
      CreateBufferIntervalCompareFnFromInstructionNames({"negate.0"});
  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(1, 10);
  std::unique_ptr<PresetAssignments> preset_assignments =
      AssignMemorySpace(module.get(), options, buffer_interval_compare,
                        &prefetch_interval_picker);
  VLOG(1) << "Module after MSA:\n" << module->ToString();

  HloInstruction* copy0 = FindInstruction(module.get(), "negate.0");
  ASSERT_NE(copy0, nullptr);
  EXPECT_EQ(copy0->shape().layout().memory_space(), kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest,
       TwoLiveAllocationValuesUnableToAllocateContiguousAltMem) {
  // In this example, we have enough space to give v.2 alternate memory,
  // and we put v.2 at the top of MSA's sort order. So, we expect that
  // it will get alternate memory. Second, we try to give negate.0 alternate
  // memory, but we can't. In order to give negate.0 alternate memory, we need
  // to give it contiguous alternate memory during cp-start.0 to cp-done.0.
  // (negate.0 and cp-start.0 {0} alias.) However, v.2 is taking too much
  // alternate memory to accommodate that request.
  //
  // We are testing a fix for dual live AllocationValues, with the following
  // setup:
  // - HloValue H containing the following positions: negate.0, cp-start.0{0}
  //   - AllocationValue A0 defined at negate.0
  //     - Segment A0.S0 define during [negate.0, cp-start.0]
  //.    - Segment A0.S1 defined during [cp-start.0, add.0]
  //   - AllocationValue A1 defined at cp-start.0{0}
  //     - Segment A1.S0 defined during [cp-start.0, cp-done.0]
  //
  // A0 and A1 are both live for more than 1 instruction.
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    /*00:*/ p.0 = f32[10,10,10,10] parameter(0)
    /*01:*/ p.1 = f32[10,10,10,10] parameter(1)
    /*02:*/ v.0 = f32[10,10,10,10] add(p.1, p.1)
    /*03:*/ negate.0 = f32[10,10,10,10] negate(p.0)
    /*04:*/ cp-start.0 = (f32[10,10,10,10], f32[10,10,10,10], u32[], u32[]) collective-permute-start(negate.0), source_target_pairs={{0,1},{2,3}}
    /*05:*/ v.1 = f32[10,10,10,10] add(v.0, v.0)
    /*06:*/ add.0 = f32[10,10,10,10] add(negate.0, negate.0)
    /*07:*/ v.2 = f32[10,10,10,10] add(v.1, v.1)
    /*08:*/ cp-done.0 = f32[10,10,10,10] collective-permute-done(cp-start.0)
    /*09:*/ v.3 = f32[10,10,10,10] add(v.2, v.2)
    /*10:*/ ROOT tuple.0 = (f32[10,10,10,10], f32[10,10,10,10], f32[10,10,10,10]) tuple(add.0, cp-done.0, v.3)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 4 * 10 * 10 * 10 * 10;
  MsaBufferIntervalCompare buffer_interval_compare =
      CreateBufferIntervalCompareFnFromInstructionNames({"v.2", "negate.0"});
  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(1, 10);
  std::unique_ptr<PresetAssignments> preset_assignments =
      AssignMemorySpace(module.get(), options, buffer_interval_compare,
                        &prefetch_interval_picker);
  VLOG(1) << "Module after MSA:\n" << module->ToString();

  HloInstruction* v2 = FindInstruction(module.get(), "v.2");
  ASSERT_NE(v2, nullptr);
  EXPECT_EQ(v2->shape().layout().memory_space(), kAlternateMemorySpace);
  HloInstruction* copy0 = FindInstruction(module.get(), "negate.0");
  ASSERT_NE(copy0, nullptr);
  EXPECT_NE(copy0->shape().layout().memory_space(), kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, AvoidRedundantEvictionInWhile) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    tanh = f32[3]{0} tanh(gte1)
    gte2 = pred[] get-tuple-element(p0), index=2
    negate0 = f32[3]{0} negate(gte0)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    add = f32[3]{0} add(negate14, tanh)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(add, gte1, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, p0, p1)
    while = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    gte = f32[3]{0} get-tuple-element(while), index=1
    ROOT negate = f32[3]{0} negate(gte)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect that while{1} is allocated to alternate memory space. Also expect
  // that this buffer is prefetched at the end of the while loop body but is
  // never evicted (since it has a copy in the default memory space).
  const HloInstruction* while_instr = FindInstruction(module.get(), "while");
  EXPECT_EQ(while_instr->shape().tuple_shapes(1).layout().memory_space(),
            kAlternateMemorySpace);
  const HloInstruction* gte1 = FindInstruction(module.get(), "gte1");
  EXPECT_EQ(gte1->user_count(), 1);
  EXPECT_EQ(gte1->users()[0]->opcode(), HloOpcode::kTanh);
  const HloInstruction* while_root =
      while_instr->while_body()->root_instruction();
  EXPECT_THAT(while_root->operand(1),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::GetTupleElement(op::Parameter(0))));
}

TEST_F(MemorySpaceAssignmentTest,
       RedundantEvictionEliminationShouldntAddRedundantParam) {
  // Check that if there wasn't an eviction in the while loop, we don't add the
  // buffer in default memory as an additional parameter to the while loop.
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    tanh = f32[3]{0} tanh(gte1)
    gte2 = pred[] get-tuple-element(p0), index=2
    negate0 = f32[3]{0} negate(gte0)
    negate1 = f32[3]{0} negate(negate0)
    add = f32[3]{0} add(negate1, tanh)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(add, gte1, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, p0, p1)
    while = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    gte = f32[3]{0} get-tuple-element(while), index=1
    ROOT negate = f32[3]{0} negate(gte)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect that while tuple shape contains 3 elements like the original.
  const HloInstruction* while_instr = FindInstruction(module.get(), "while");
  EXPECT_EQ(while_instr->shape().tuple_shapes_size(), 3);
}

TEST_F(MemorySpaceAssignmentTest, AvoidRedundantEvictionInNestedWhile) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond2 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body2 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    tanh = f32[3]{0} tanh(gte1)
    gte2 = pred[] get-tuple-element(p0), index=2
    negate0 = f32[3]{0} negate(gte0)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    add = f32[3]{0} add(negate14, tanh)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(add, gte1, gte2)
  }

  while_cond1 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body1 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT while2 = (f32[3]{0}, f32[3]{0}, pred[]) while(p0), condition=while_cond2, body=while_body2
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, p0, p1)
    while1 = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond1, body=while_body1
    gte = f32[3]{0} get-tuple-element(while1), index=1
    ROOT negate = f32[3]{0} negate(gte)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect that while1{1} and while2{1} are allocated to alternate memory
  // space. Also expect that this buffer is prefetched at the end of the while
  // loop body but is never evicted (since it has a copy in the default memory
  // space).
  const HloInstruction* while1_instr = FindInstruction(module.get(), "while1");
  EXPECT_EQ(while1_instr->shape().tuple_shapes(1).layout().memory_space(),
            kAlternateMemorySpace);
  const HloInstruction* while2_instr = FindInstruction(module.get(), "while2");
  EXPECT_EQ(while2_instr->shape().tuple_shapes(1).layout().memory_space(),
            kAlternateMemorySpace);
  const HloInstruction* gte1 = FindInstruction(module.get(), "gte1");
  EXPECT_EQ(gte1->user_count(), 1);
  EXPECT_EQ(gte1->users()[0]->opcode(), HloOpcode::kTanh);
  const HloInstruction* while_root =
      while2_instr->while_body()->root_instruction();
  EXPECT_THAT(while_root->operand(1),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::GetTupleElement(op::Parameter(0))));
}

TEST_F(MemorySpaceAssignmentTest, RedundantEvictionEliminationBug) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    tanh = f32[3]{0} tanh(gte1)
    gte2 = pred[] get-tuple-element(p0), index=2
    negate0 = f32[3]{0} negate(gte0)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    add0 = f32[3]{0} add(negate14, tanh)
    add1 = f32[3]{0} add(add0, gte1)
    negate = f32[3]{0} negate(add1)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(add1, negate, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, p0, p1)
    while = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    gte = f32[3]{0} get-tuple-element(while), index=1
    ROOT negate = f32[3]{0} negate(gte)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect that redundant eviction elimination doesn't kick in because
  // while{1} is updated within the body.
  const HloInstruction* while_instr = FindInstruction(module.get(), "while");
  EXPECT_EQ(while_instr->shape().tuple_shapes_size(), 3);
  EXPECT_EQ(while_instr->shape().tuple_shapes(1).layout().memory_space(),
            kAlternateMemorySpace);
  const HloInstruction* gte1 = FindInstruction(module.get(), "gte1");
  EXPECT_EQ(gte1->user_count(), 2);
  EXPECT_NE(
      absl::c_find_if(gte1->users(), HloPredicateIsOp<HloOpcode::kCopyStart>),
      gte1->users().end());
}

TEST_F(MemorySpaceAssignmentTest, RedundantEvictionEliminationInChainedWhile) {
  // Check against a bug where a while HLO feeding to another while HLO can
  // cause a crash if we performa redundant eviction elimination to the first
  // while but not the other (while operand/parameter shapes would mismatch).
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond1 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body1 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    tanh = f32[3]{0} tanh(gte1)
    gte2 = pred[] get-tuple-element(p0), index=2
    negate0 = f32[3]{0} negate(gte0)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    add = f32[3]{0} add(negate14, tanh)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(add, gte1, gte2)
  }

  while_cond2 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body2 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    tanh = f32[3]{0} tanh(gte1)
    gte2 = pred[] get-tuple-element(p0), index=2
    negate0 = f32[3]{0} negate(gte0)
    add = f32[3]{0} add(negate0, tanh)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(add, gte1, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, p0, p1)
    while1 = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond1, body=while_body1
    while2 = (f32[3]{0}, f32[3]{0}, pred[]) while(while1), condition=while_cond2, body=while_body2
    gte = f32[3]{0} get-tuple-element(while2), index=1
    ROOT negate = f32[3]{0} negate(gte)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect that while1 has one more value than while2 in its shape.
  EXPECT_EQ(
      FindInstruction(module.get(), "while1")->shape().tuple_shapes_size(),
      FindInstruction(module.get(), "while2")->shape().tuple_shapes_size() + 1);
}

TEST_F(MemorySpaceAssignmentTest, AvoidRedundantEvictionAfterWhile) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = pred[] get-tuple-element(p0), index=2
    add = f32[3]{0} add(gte0, gte1)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(gte0, add, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    negate0 = f32[3]{0} negate(p0)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, negate14, p1)
    while = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    gte0 = f32[3]{0} get-tuple-element(while), index=0
    gte1 = f32[3]{0} get-tuple-element(while), index=1
    negate20 = f32[3]{0} negate(gte1)
    negate21 = f32[3]{0} negate(negate20)
    negate22 = f32[3]{0} negate(negate21)
    negate23 = f32[3]{0} negate(negate22)
    negate24 = f32[3]{0} negate(negate23)
    negate25 = f32[3]{0} negate(negate24)
    negate26 = f32[3]{0} negate(negate25)
    negate27 = f32[3]{0} negate(negate26)
    negate28 = f32[3]{0} negate(negate27)
    negate29 = f32[3]{0} negate(negate28)
    negate30 = f32[3]{0} negate(negate29)
    negate31 = f32[3]{0} negate(negate30)
    negate32 = f32[3]{0} negate(negate31)
    negate33 = f32[3]{0} negate(negate32)
    negate34 = f32[3]{0} negate(negate33)
    ROOT add = f32[3]{0} add(negate34, gte0)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  EXPECT_THAT(
      module->entry_computation()->root_instruction()->operand(1),
      op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace, op::Copy()));
}

TEST_F(MemorySpaceAssignmentTest, AvoidRedundantEvictionAfterWhile2) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond1 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body1 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = pred[] get-tuple-element(p0), index=2
    add = f32[3]{0} add(gte0, gte1)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(gte0, add, gte2)
  }

  while_cond2 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body2 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = pred[] get-tuple-element(p0), index=2
    add = f32[3]{0} add(gte0, gte1)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(gte0, add, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    tuple1 = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, p0, p1)
    while1 = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple1), condition=while_cond1, body=while_body1
    gte0 = f32[3]{0} get-tuple-element(while1), index=0
    gte1 = f32[3]{0} get-tuple-element(while1), index=1
    negate0 = f32[3]{0} negate(gte1)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    tuple2 = (f32[3]{0}, f32[3]{0}, pred[]) tuple(gte0, negate14, p1)
    while2 = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple2), condition=while_cond2, body=while_body2
    gte2 = f32[3]{0} get-tuple-element(while2), index=0
    gte3 = f32[3]{0} get-tuple-element(while2), index=1
    negate20 = f32[3]{0} negate(gte3)
    negate21 = f32[3]{0} negate(negate20)
    negate22 = f32[3]{0} negate(negate21)
    negate23 = f32[3]{0} negate(negate22)
    negate24 = f32[3]{0} negate(negate23)
    negate25 = f32[3]{0} negate(negate24)
    negate26 = f32[3]{0} negate(negate25)
    negate27 = f32[3]{0} negate(negate26)
    negate28 = f32[3]{0} negate(negate27)
    negate29 = f32[3]{0} negate(negate28)
    negate30 = f32[3]{0} negate(negate29)
    negate31 = f32[3]{0} negate(negate30)
    negate32 = f32[3]{0} negate(negate31)
    negate33 = f32[3]{0} negate(negate32)
    negate34 = f32[3]{0} negate(negate33)
    ROOT add = f32[3]{0} add(negate34, gte2)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  EXPECT_THAT(
      module->entry_computation()->root_instruction()->operand(1),
      op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                    op::AsyncCopy(kDefaultMemorySpace, kAlternateMemorySpace,
                                  op::GetTupleElement(op::While()))));
}

TEST_F(MemorySpaceAssignmentTest,
       AfterWhileRedundantEarlierEvictionModifiedBuffer) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    gte2 = pred[] get-tuple-element(p0), index=2
    add = f32[3]{0} add(gte0, gte1)
    negate = f32[3]{0} negate(gte0)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(negate, add, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3]{0} copy(p0)
    negate0 = f32[3]{0} negate(p0)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, negate14, p1)
    while = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple), condition=while_cond, body=while_body
    gte0 = f32[3]{0} get-tuple-element(while), index=0
    gte1 = f32[3]{0} get-tuple-element(while), index=1
    negate20 = f32[3]{0} negate(gte1)
    negate21 = f32[3]{0} negate(negate20)
    negate22 = f32[3]{0} negate(negate21)
    negate23 = f32[3]{0} negate(negate22)
    negate24 = f32[3]{0} negate(negate23)
    negate25 = f32[3]{0} negate(negate24)
    negate26 = f32[3]{0} negate(negate25)
    negate27 = f32[3]{0} negate(negate26)
    negate28 = f32[3]{0} negate(negate27)
    negate29 = f32[3]{0} negate(negate28)
    negate30 = f32[3]{0} negate(negate29)
    negate31 = f32[3]{0} negate(negate30)
    negate32 = f32[3]{0} negate(negate31)
    negate33 = f32[3]{0} negate(negate32)
    negate34 = f32[3]{0} negate(negate33)
    ROOT add = f32[3]{0} add(negate34, gte0)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  EXPECT_THAT(
      module->entry_computation()->root_instruction()->operand(1),
      op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                    op::AsyncCopy(kDefaultMemorySpace, kAlternateMemorySpace,
                                  op::GetTupleElement(op::While()))));
}

TEST_F(MemorySpaceAssignmentTest,
       WhileRedundantEvictionWithInefficientAllocationBug) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  while_cond {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    tanh = f32[3]{0} tanh(gte1)
    gte2 = pred[] get-tuple-element(p0), index=2
    negate0 = f32[3]{0} negate(gte0)
    negate1 = f32[3]{0} negate(negate0)
    add = f32[3]{0} add(negate1, tanh)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(add, gte1, gte2)
  }

  while_cond1 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    ROOT gte = pred[] get-tuple-element(p0), index=2
  }

  while_body1 {
    p0 = (f32[3]{0}, f32[3]{0}, pred[]) parameter(0)
    gte0 = f32[3]{0} get-tuple-element(p0), index=0
    gte2 = pred[] get-tuple-element(p0), index=2
    negate0 = f32[3]{0} negate(gte0)
    negate1 = f32[3]{0} negate(negate0)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    gte1 = f32[3]{0} get-tuple-element(p0), index=1
    tanh = f32[3]{0} tanh(gte1)
    add = f32[3]{0} add(negate14, tanh)
    ROOT tuple = (f32[3]{0}, f32[3]{0}, pred[]) tuple(add, gte1, gte2)
  }

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = pred[] parameter(1)
    p2 = f32[3]{0} parameter(2)
    copy = f32[3]{0} copy(p0)
    tuple1 = (f32[3]{0}, f32[3]{0}, pred[]) tuple(copy, p0, p1)
    while1 = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple1), condition=while_cond, body=while_body
    gte0 = f32[3]{0} get-tuple-element(while1), index=0
    gte1 = f32[3]{0} get-tuple-element(while1), index=1
    negate0_entry = f32[3]{0} negate(gte1)
    gte2 = pred[] get-tuple-element(while1), index=2
    tuple2 = (f32[3]{0}, f32[3]{0}, pred[]) tuple(gte0, gte1, gte2)
    while2 = (f32[3]{0}, f32[3]{0}, pred[]) while(tuple2), condition=while_cond1, body=while_body1
    negate1 = f32[3]{0} negate(negate0_entry)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    negate10 = f32[3]{0} negate(negate9)
    negate11 = f32[3]{0} negate(negate10)
    negate12 = f32[3]{0} negate(negate11)
    negate13 = f32[3]{0} negate(negate12)
    negate14 = f32[3]{0} negate(negate13)
    gte = f32[3]{0} get-tuple-element(while2), index=1
    ROOT add = f32[3]{0} add(gte, negate14)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  // Inject GetInefficientAllocationSites to mark negate0_entry use as
  // inefficient. This triggers a corner case bug where allocating for while2{1}
  // in the retry allocation fails to find the previous required allocation in
  // default memory, and creates a new one which is wrong.
  bool marked_inefficient = false;
  options.get_inefficient_allocation_sites_fn =
      [&](absl::Span<HloPosition> defining_positions)
      -> std::vector<std::variant<HloPosition, HloUse>> {
    if (absl::c_find(defining_positions,
                     HloPosition{FindInstruction(module.get(), "while1"),
                                 {1}}) != defining_positions.end() &&
        !marked_inefficient) {
      LOG(INFO) << "Marking the use inefficient.";
      marked_inefficient = true;
      return {HloUse{FindInstruction(module.get(), "negate0_entry"), 0}};
    }
    return {};
  };
  AssignMemorySpace(module.get(), options);
}

TEST_F(MemorySpaceAssignmentTest, DisablePrefetch) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    p0 = f32[3]{0} parameter(0)
    p1 = f32[3]{0} parameter(1)
    negate1 = f32[3]{0} negate(p1)
    negate2 = f32[3]{0} negate(negate1)
    negate3 = f32[3]{0} negate(negate2)
    negate4 = f32[3]{0} negate(negate3)
    negate5 = f32[3]{0} negate(negate4)
    negate6 = f32[3]{0} negate(negate5)
    negate7 = f32[3]{0} negate(negate6)
    negate8 = f32[3]{0} negate(negate7)
    negate9 = f32[3]{0} negate(negate8)
    ROOT add = f32[3]{0} add(negate9, p0)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  Options options = DefaultMemorySpaceOptions();
  options.max_outstanding_prefetches = 0;
  AssignMemorySpace(module.get(), options);

  EXPECT_THAT(module->entry_computation()->root_instruction()->operand(1),
              op::Parameter());
}

TEST_F(MemorySpaceAssignmentTest, BitcastRoot) {
  // Tests against a bug where the root of entry computation is a bitcast
  // instruction and it ends up getting an allocation in the alternate memory.
  absl::string_view hlo_string = R"(
HloModule primitive_computation_gather.4, is_scheduled=true

%while_body {
  %param.1 = (s32[], f32[3,3,3]) parameter(0)
  %get-tuple-element.32 = s32[] get-tuple-element(%param.1), index=0
  %copy.6 = s32[] copy(s32[] %get-tuple-element.32)
  %constant.8 = s32[] constant(1)
  %add = s32[] add(s32[] %copy.6, s32[] %constant.8)
  %get-tuple-element.35 = f32[3,3,3] get-tuple-element(%param.1), index=1
  negate = f32[3,3,3] negate(get-tuple-element.35)
  ROOT %tuple.10 = (s32[], f32[3,3,3]) tuple(s32[] %add, f32[3,3,3] negate)
}

%while_cond {
  %param.0 = (s32[], f32[3,3,3]) parameter(0)
  %get-tuple-element = s32[] get-tuple-element(%param.0), index=0
  %constant.3 = s32[] constant(3)
  ROOT %compare = pred[] compare(s32[] %get-tuple-element, s32[] %constant.3), direction=LT
}

ENTRY %primitive_computation_gather.4 (parameter.1: f32[3,10,5], parameter.2: s32[3,1]) -> f32[3,3,3] {
  %constant.1 = s32[] constant(0)
  %copy.11 = s32[] copy(s32[] %constant.1)
  %constant = f32[] constant(0)
  %broadcast = f32[3,3,3] broadcast(f32[] %constant), dimensions={}
  %tuple.8 = (s32[], f32[3,3,3]) tuple(s32[] %copy.11, f32[3,3,3] %broadcast)
  %while = (s32[], f32[3,3,3]) while(%tuple.8), condition=%while_cond, body=%while_body
  %get-tuple-element.7 = f32[3,3,3] get-tuple-element(%while), index=1
  ROOT %bitcast.1 = f32[3,3,3] bitcast(f32[3,3,3] %get-tuple-element.7)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(!root->shape().has_layout() ||
              root->shape().layout().memory_space() == kDefaultMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, PrecoloredBuffer) {
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  ENTRY Entry {
    param0 = f32[8,3] parameter(0)
    param1 = f32[2,4] parameter(1)
    a = f32[8,3]{1,0:S(1)} cosine(param0)
    b = f32[2,4] negate(param1)
    d = f32[8,3] negate(a)
    c = f32[2,4] negate(b)
    e = f32[2,4] negate(c)
    f = f32[8,3] negate(d)
    g = f32[2,4] negate(e)
    h = f32[2,4] negate(g)
    i = f32[2,4] negate(h)
    j = f32[2,4] negate(i)
    k = f32[2,4] negate(j)
    l = f32[2,4] negate(k)
    m = f32[2,4] negate(l)
    n = f32[2,4] negate(m)
    o = f32[8,3] negate(f)
    p = f32[2,4] negate(n)
    q = f32[8,3] add(f, o)
    r = f32[8,3] add(q, a)
    ROOT tuple = (f32[2,4], f32[8,3]) tuple(p, r)
  }
  )";

  MsaBufferIntervalCompare buffer_interval_compare =
      [](const MsaBufferInterval& a, const MsaBufferInterval& b) {
        auto get_opcode_priority = [](const HloOpcode& opcode) {
          switch (opcode) {
            case HloOpcode::kNegate:
              return 0;
            case HloOpcode::kAdd:
              return 1;
            case HloOpcode::kCos:
              return 2;
            default:
              return 3;
          }
        };

        return get_opcode_priority(a.buffer->defining_instruction()->opcode()) <
               get_opcode_priority(b.buffer->defining_instruction()->opcode());
      };
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 10);
  Options options = DefaultMemorySpaceOptions();
  std::unique_ptr<PresetAssignments> preset_assignments =
      AssignMemorySpace(module.get(), options, buffer_interval_compare,
                        &prefetch_interval_picker);

  const HloInstruction* r = FindInstruction(module.get(), "r");
  const HloInstruction* d = FindInstruction(module.get(), "d");
  const HloInstruction* a = FindInstruction(module.get(), "a");
  // Make sure the r and d operands aren't prefetched.
  EXPECT_EQ(r->operand(1), a);
  EXPECT_EQ(d->operand(0), a);
  // Make sure they are allocated in the alternate memory.
  EXPECT_EQ(a->shape().layout().memory_space(), kAlternateMemorySpace);
  // Make sure the a buffer has an entry in the preset assignments.
  auto a_entry = std::find_if(
      preset_assignments->chunks().begin(), preset_assignments->chunks().end(),
      [&](std::pair<HloPosition, HeapSimulator::Chunk> position_and_chunk) {
        return position_and_chunk.first.instruction == a;
      });
  EXPECT_NE(a_entry, preset_assignments->chunks().end());
}

TEST_F(MemorySpaceAssignmentTest, PrecoloredBufferOOM) {
  // Same as above but there are two 96-byte values that are pinned to the
  // alternate memory (the size of the alternate memory is 128 bytes), which is
  // unsatisfiable.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  ENTRY Entry {
    param0 = f32[8,3] parameter(0)
    param1 = f32[2,4] parameter(1)
    a = f32[8,3]{1,0:S(1)} cosine(param0)
    b = f32[2,4] negate(param1)
    d = f32[8,3] negate(a)
    c = f32[2,4] negate(b)
    e = f32[2,4] negate(c)
    f = f32[8,3] negate(d)
    g = f32[2,4] negate(e)
    h = f32[2,4] negate(g)
    i = f32[2,4] negate(h)
    j = f32[2,4] negate(i)
    k = f32[2,4] negate(j)
    l = f32[2,4] negate(k)
    m = f32[2,4] negate(l)
    n = f32[2,4] negate(m)
    o = f32[8,3]{1,0:S(1)} negate(f)
    p = f32[2,4] negate(n)
    q = f32[8,3] add(f, o)
    r = f32[8,3] add(q, a)
    ROOT tuple = (f32[2,4], f32[8,3]) tuple(p, r)
  }
  )";

  MsaBufferIntervalCompare buffer_interval_compare =
      [](const MsaBufferInterval& a, const MsaBufferInterval& b) {
        auto get_opcode_priority = [](const HloOpcode& opcode) {
          switch (opcode) {
            case HloOpcode::kNegate:
              return 0;
            case HloOpcode::kAdd:
              return 1;
            case HloOpcode::kCos:
              return 2;
            default:
              return 3;
          }
        };

        return get_opcode_priority(a.buffer->defining_instruction()->opcode()) <
               get_opcode_priority(b.buffer->defining_instruction()->opcode());
      };
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 10);
  Options options = DefaultMemorySpaceOptions();
  auto status_or = AssignMemorySpaceAndReturnStatus(module.get(), options,
                                                    buffer_interval_compare,
                                                    &prefetch_interval_picker);
  EXPECT_THAT(
      status_or.status(),
      tsl::testing::StatusIs(
          tsl::error::FAILED_PRECONDITION,
          ::testing::HasSubstr("requires allocation in the alternate memory, "
                               "which could not be satisfied")));
}

TEST_F(MemorySpaceAssignmentTest, AsyncOpShortLiveRange) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  negate0 = bf16[4]{0} negate(param)
  collective-permute-start = (bf16[4]{0}, bf16[4]{0}, u32[], u32[]) collective-permute-start(negate0), source_target_pairs={{0,1},{1,2},{2,3}}
  negate1 = bf16[4]{0} negate(param)
  negate2 = bf16[4]{0} negate(negate1)
  negate3 = bf16[4]{0} negate(negate2)
  collective-permute-done = bf16[4]{0} collective-permute-done(collective-permute-start)
  ROOT add = add(collective-permute-done, negate3)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect both the source and destination buffers to get alternate memory
  // allocations.
  HloInstruction* collective_permute_start =
      module->entry_computation()->GetInstructionWithName(
          "collective-permute-start");
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(0)
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(1)
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, AsyncOpLongLiveRange) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  negate0 = bf16[4]{0} negate(param)
  collective-permute-start = (bf16[4]{0}, bf16[4]{0}, u32[], u32[]) collective-permute-start(negate0), source_target_pairs={{0,1},{1,2},{2,3}}
  negate1 = bf16[4]{0} negate(param)
  negate2 = bf16[4]{0} negate(negate1)
  negate3 = bf16[4]{0} negate(negate2)
  negate4 = bf16[4]{0} negate(negate3)
  negate5 = bf16[4]{0} negate(negate4)
  negate6 = bf16[4]{0} negate(negate5)
  negate7 = bf16[4]{0} negate(negate6)
  negate8 = bf16[4]{0} negate(negate7)
  negate9 = bf16[4]{0} negate(negate8)
  negate10 = bf16[4]{0} negate(negate9)
  negate11 = bf16[4]{0} negate(negate10)
  negate12 = bf16[4]{0} negate(negate11)
  negate13 = bf16[4]{0} negate(negate12)
  collective-permute-done = bf16[4]{0} collective-permute-done(collective-permute-start)
  ROOT add = add(collective-permute-done, negate13)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect none of the buffers to get alternate memory allocations because of
  // the long live range.
  HloInstruction* collective_permute_start =
      module->entry_computation()->GetInstructionWithName(
          "collective-permute-start");
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(0)
                  .layout()
                  .memory_space() == kDefaultMemorySpace);
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(1)
                  .layout()
                  .memory_space() == kDefaultMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, AsyncOpLongLiveRangeInputBufferConsumer) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  negate0 = bf16[4]{0} negate(param)
  collective-permute-start = (bf16[4]{0}, bf16[4]{0}, u32[], u32[]) collective-permute-start(negate0), source_target_pairs={{0,1},{1,2},{2,3}}
  negate1 = bf16[4]{0} negate(negate0)
  negate2 = bf16[4]{0} negate(negate1)
  negate3 = bf16[4]{0} negate(negate2)
  negate4 = bf16[4]{0} negate(negate3)
  negate5 = bf16[4]{0} negate(negate4)
  negate6 = bf16[4]{0} negate(negate5)
  negate7 = bf16[4]{0} negate(negate6)
  negate8 = bf16[4]{0} negate(negate7)
  negate9 = bf16[4]{0} negate(negate8)
  negate10 = bf16[4]{0} negate(negate9)
  negate11 = bf16[4]{0} negate(negate10)
  negate12 = bf16[4]{0} negate(negate11)
  negate13 = bf16[4]{0} negate(negate12)
  collective-permute-done = bf16[4]{0} collective-permute-done(collective-permute-start)
  ROOT add = add(collective-permute-done, negate13)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect none of the buffers to get alternate memory allocations because of
  // the long live range and because negate0 is also used by negate1.
  HloInstruction* collective_permute_start =
      module->entry_computation()->GetInstructionWithName(
          "collective-permute-start");
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(0)
                  .layout()
                  .memory_space() == kDefaultMemorySpace);
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(1)
                  .layout()
                  .memory_space() == kDefaultMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, InPlaceAsyncCollectivePermute) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  negate0 = bf16[4]{0} negate(param)
  negate1 = bf16[4]{0} negate(param)
  const0 = s32[] constant(0)
  const1 = s32[] constant(1)
  tuple0 = (s32[]) tuple(const0)
  tuple1 = (s32[]) tuple(const1)
  collective-permute-start = (bf16[4]{0}, bf16[4]{0}, u32[], u32[]) collective-permute-start(negate0, negate1, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  negate2 = bf16[4]{0} negate(param)
  negate3 = bf16[4]{0} negate(negate2)
  negate4 = bf16[4]{0} negate(negate3)
  collective-permute-done = bf16[4]{0} collective-permute-done(collective-permute-start)
  ROOT add = add(collective-permute-done, negate4)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect both the source and destination buffers to get alternate memory
  // allocations.
  HloInstruction* collective_permute_start =
      module->entry_computation()->GetInstructionWithName(
          "collective-permute-start");
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(0)
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(1)
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, InPlaceAsyncCollectivePermuteSameBuffer) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  negate0 = bf16[4]{0} negate(param)
  const0 = s32[] constant(0)
  const1 = s32[] constant(1)
  tuple0 = (s32[]) tuple(const0)
  tuple1 = (s32[]) tuple(const1)
  collective-permute-start = (bf16[4]{0}, bf16[4]{0}, u32[], u32[]) collective-permute-start(negate0, negate0, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  negate2 = bf16[4]{0} negate(param)
  negate3 = bf16[4]{0} negate(negate2)
  negate4 = bf16[4]{0} negate(negate3)
  collective-permute-done = bf16[4]{0} collective-permute-done(collective-permute-start)
  ROOT add = add(collective-permute-done, negate4)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect both the source and destination buffers to get alternate memory
  // allocations.
  HloInstruction* collective_permute_start =
      module->entry_computation()->GetInstructionWithName(
          "collective-permute-start");
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(0)
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
  EXPECT_TRUE(collective_permute_start->shape()
                  .tuple_shapes(1)
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest,
       InPlaceAsyncCollectivePermuteSameBufferChained) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  negate0 = bf16[4]{0} negate(param)
  const0 = s32[] constant(0)
  const1 = s32[] constant(1)
  tuple0 = (s32[]) tuple(const0)
  tuple1 = (s32[]) tuple(const1)
  collective-permute-start.1 = (bf16[4]{0}, bf16[4]{0}, u32[], u32[]) collective-permute-start(negate0, negate0, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  negate2 = bf16[4]{0} negate(param)
  negate3 = bf16[4]{0} negate(negate2)
  negate4 = bf16[4]{0} negate(negate3)
  collective-permute-done.1 = bf16[4]{0} collective-permute-done(collective-permute-start.1)
  collective-permute-start.2 = (bf16[4]{0}, bf16[4]{0}, u32[], u32[]) collective-permute-start(collective-permute-done.1, collective-permute-done.1, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  negate5 = bf16[4]{0} negate(negate4)
  negate6 = bf16[4]{0} negate(negate5)
  negate7 = bf16[4]{0} negate(negate6)
  collective-permute-done.2 = bf16[4]{0} collective-permute-done(collective-permute-start.2)
  ROOT add = add(collective-permute-done.2, negate7)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  // Expect both the source and destination buffers to get alternate memory
  // allocations.
  HloInstruction* collective_permute_start_1 =
      module->entry_computation()->GetInstructionWithName(
          "collective-permute-start.1");
  EXPECT_TRUE(collective_permute_start_1->shape()
                  .tuple_shapes(0)
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
  EXPECT_TRUE(collective_permute_start_1->shape()
                  .tuple_shapes(1)
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
  HloInstruction* collective_permute_start_2 =
      module->entry_computation()->GetInstructionWithName(
          "collective-permute-start.2");
  EXPECT_TRUE(collective_permute_start_2->shape()
                  .tuple_shapes(0)
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
  EXPECT_TRUE(collective_permute_start_2->shape()
                  .tuple_shapes(1)
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest,
       TupleInPlaceAsyncCollectivePermuteSameBufferChained) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  param2 = bf16[48]{0} parameter(1)
  negate0.1 = bf16[48]{0} negate(param2)
  negate0.2 = bf16[48]{0} negate(param2)
  const0 = s32[] constant(0)
  const1 = s32[] constant(1)
  tuple0.0 = (s32[]) tuple(const0)
  tuple0 = ((s32[]), (s32[])) tuple(tuple0.0, tuple0.0)
  tuple1.0 = (s32[]) tuple(const1)
  tuple1 = ((s32[]), (s32[])) tuple(tuple1.0, tuple1.0)
  tuple2 = (bf16[48]{0}, bf16[48]{0}) tuple(negate0.1, negate0.2)
  collective-permute-start.1 = ((bf16[48]{0}, bf16[48]{0}), (bf16[48]{0}, bf16[48]{0}), u32[], u32[]) collective-permute-start(tuple2, tuple2, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  negate2 = bf16[4]{0} negate(param)
  negate3 = bf16[4]{0} negate(negate2)
  negate4 = bf16[4]{0} negate(negate3)
  collective-permute-done.1 = (bf16[48]{0}, bf16[48]{0}) collective-permute-done(collective-permute-start.1)
  collective-permute-start.2 = ((bf16[48]{0}, bf16[48]{0}), (bf16[48]{0}, bf16[48]{0}), u32[], u32[]) collective-permute-start(collective-permute-done.1, collective-permute-done.1, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  negate5 = bf16[4]{0} negate(negate4)
  negate6 = bf16[4]{0} negate(negate5)
  negate7 = bf16[4]{0} negate(negate6)
  collective-permute-done.2 = (bf16[48]{0}, bf16[48]{0}) collective-permute-done(collective-permute-start.2)
  gte = bf16[48]{0} get-tuple-element(collective-permute-done.2), index=0
  ROOT root = (bf16[48]{0}, bf16[4]{0}) tuple(gte, negate7)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  const HloInstruction* cp_done1 =
      FindInstruction(module.get(), "collective-permute-done.1");
  EXPECT_EQ(cp_done1->operand(0)->opcode(), HloOpcode::kCollectivePermuteStart);
  const HloInstruction* cp_done2 =
      FindInstruction(module.get(), "collective-permute-done.2");
  EXPECT_EQ(cp_done2->operand(0)->opcode(), HloOpcode::kCollectivePermuteStart);
}

TEST_F(MemorySpaceAssignmentTest,
       TupleInPlaceAsyncCollectivePermuteSameBuffer) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  param2 = bf16[48]{0} parameter(1)
  negate0.1 = bf16[48]{0} negate(param2)
  negate0.2 = bf16[48]{0} negate(param2)
  const0 = s32[] constant(0)
  const1 = s32[] constant(1)
  tuple0.0 = (s32[]) tuple(const0)
  tuple0 = ((s32[]), (s32[])) tuple(tuple0.0, tuple0.0)
  tuple1.0 = (s32[]) tuple(const1)
  tuple1 = ((s32[]), (s32[])) tuple(tuple1.0, tuple1.0)
  tuple2 = (bf16[48]{0}, bf16[48]{0}) tuple(negate0.1, negate0.1)
  tuple3 = (bf16[48]{0}, bf16[48]{0}) tuple(negate0.2, negate0.2)
  collective-permute-start.1 = ((bf16[48]{0}, bf16[48]{0}), (bf16[48]{0}, bf16[48]{0}), u32[], u32[]) collective-permute-start(tuple2, tuple3, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  negate2 = bf16[4]{0} negate(param)
  negate3 = bf16[4]{0} negate(negate2)
  negate4 = bf16[4]{0} negate(negate3)
  collective-permute-done.1 = (bf16[48]{0}, bf16[48]{0}) collective-permute-done(collective-permute-start.1)
  gte = bf16[48]{0} get-tuple-element(collective-permute-done.1), index=0
  ROOT root = (bf16[48]{0}, bf16[4]{0}) tuple(gte, negate4)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  const HloInstruction* cp_done1 =
      FindInstruction(module.get(), "collective-permute-done.1");
  EXPECT_EQ(cp_done1->operand(0)->opcode(), HloOpcode::kCollectivePermuteStart);
}

TEST_F(MemorySpaceAssignmentTest,
       TupleInPlaceAsyncCollectivePermuteSameBufferRoot) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param = bf16[4]{0} parameter(0)
  param2 = bf16[48]{0} parameter(1)
  negate0.1 = bf16[48]{0} negate(param2)
  negate0.2 = bf16[48]{0} negate(param2)
  const0 = s32[] constant(0)
  const1 = s32[] constant(1)
  tuple0.0 = (s32[]) tuple(const0)
  tuple0 = ((s32[]), (s32[])) tuple(tuple0.0, tuple0.0)
  tuple1.0 = (s32[]) tuple(const1)
  tuple1 = ((s32[]), (s32[])) tuple(tuple1.0, tuple1.0)
  tuple2 = (bf16[48]{0}, bf16[48]{0}) tuple(negate0.1, negate0.1)
  tuple3 = (bf16[48]{0}, bf16[48]{0}) tuple(negate0.2, negate0.2)
  collective-permute-start.1 = ((bf16[48]{0}, bf16[48]{0}), (bf16[48]{0}, bf16[48]{0}), u32[], u32[]) collective-permute-start(tuple2, tuple3, tuple0, tuple1), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{1}}
  ROOT collective-permute-done.1 = (bf16[48]{0}, bf16[48]{0}) collective-permute-done(collective-permute-start.1)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  const HloInstruction* cp_done1 =
      FindInstruction(module.get(), "collective-permute-done.1");
  EXPECT_EQ(cp_done1->operand(0)->opcode(), HloOpcode::kCollectivePermuteStart);
  ShapeUtil::ForEachSubshape(
      cp_done1->shape(),
      [&](const Shape& subshape, const ShapeIndex& /*index*/) {
        if (subshape.IsArray() && subshape.has_layout()) {
          EXPECT_EQ(subshape.layout().memory_space(), kDefaultMemorySpace);
        }
      });
}

TEST_F(MemorySpaceAssignmentTest, TupleInPlaceAsyncCollectivePermuteRoot) {
  absl::string_view hlo_string = R"(
 HloModule inplace_collective_permute, is_scheduled=true

 ENTRY %inplace_collective_permute {
   %param.0 = u32[8,1,1] parameter(0)
   %constant.1000 = u32[] constant(1000)
   %broadcast.1 = u32[8,1,1] broadcast(u32[] %constant.1000), dimensions={}
   %broadcast.2 = u32[8,1,1] broadcast(u32[] %constant.1000), dimensions={}
   %tuple.input = (u32[8,1,1], u32[8,1,1]) tuple(u32[8,1,1] %param.0, u32[8,1,1] %param.0)
   %tuple.output = (u32[8,1,1], u32[8,1,1]) tuple(u32[8,1,1] %broadcast.1, u32[8,1,1] %broadcast.2)
   %constant.0 = s32[] constant(0)
   %constant.1 = s32[] constant(1)
   %constant.2 = s32[] constant(2)
   %indices.0.0.0 = (s32[], s32[], s32[]) tuple(s32[] %constant.0, s32[] %constant.0, s32[] %constant.0)
   %indices.1.0.0 = (s32[], s32[], s32[]) tuple(s32[] %constant.1, s32[] %constant.0, s32[] %constant.0)
   %indices.2.0.0 = (s32[], s32[], s32[]) tuple(s32[] %constant.2, s32[] %constant.0, s32[] %constant.0)
   %indices.000.100 = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple((s32[], s32[], s32[]) %indices.0.0.0, (s32[], s32[], s32[]) %indices.1.0.0)
   %indices.000.200 = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple((s32[], s32[], s32[]) %indices.0.0.0, (s32[], s32[], s32[]) %indices.2.0.0)
   %indices.000.0 = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple((s32[], s32[], s32[]) %indices.0.0.0, (s32[], s32[], s32[]) %indices.0.0.0)
   %input.indices = (((s32[], s32[], s32[]), (s32[], s32[], s32[])), ((s32[], s32[], s32[]), (s32[], s32[], s32[]))) tuple(((s32[], s32[], s32[]), (s32[], s32[], s32[])) %indices.000.100, ((s32[], s32[], s32[]), (s32[], s32[], s32[])) %indices.000.0)
   %output.indices = (((s32[], s32[], s32[]), (s32[], s32[], s32[])), ((s32[], s32[], s32[]), (s32[], s32[], s32[]))) tuple(((s32[], s32[], s32[]), (s32[], s32[], s32[])) %indices.000.100, ((s32[], s32[], s32[]), (s32[], s32[], s32[])) %indices.000.200)
   %collective-permute-start = ((u32[8,1,1], u32[8,1,1]), (u32[8,1,1], u32[8,1,1]), u32[], u32[]) collective-permute-start((u32[8,1,1], u32[8,1,1]) %tuple.input, (u32[8,1,1], u32[8,1,1]) %tuple.output, (((s32[], s32[], s32[]), (s32[], s32[], s32[])), ((s32[], s32[], s32[]), (s32[], s32[], s32[]))) %input.indices, (((s32[], s32[], s32[]), (s32[], s32[], s32[])), ((s32[], s32[], s32[]), (s32[], s32[], s32[]))) %output.indices), channel_id=42, source_target_pairs={{0,1},{1,0},{1,0},{0,1}}, slice_sizes={{4},{4},{4},{4}}
   ROOT %collective-permute-done = (u32[8,1,1], u32[8,1,1]) collective-permute-done(((u32[8,1,1], u32[8,1,1]), (u32[8,1,1], u32[8,1,1]), u32[], u32[]) %collective-permute-start)
 }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  const HloInstruction* cp_done =
      FindInstruction(module.get(), "collective-permute-done");
  EXPECT_EQ(cp_done->operand(0)->opcode(), HloOpcode::kCollectivePermuteStart);
  ShapeUtil::ForEachSubshape(
      cp_done->shape(),
      [&](const Shape& subshape, const ShapeIndex& /*index*/) {
        if (subshape.IsArray() && subshape.has_layout()) {
          EXPECT_EQ(subshape.layout().memory_space(), kDefaultMemorySpace);
        }
      });
}

TEST_F(MemorySpaceAssignmentTest, ReservedScopedMemory) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param0 = f32[2,4] parameter(0)
  a = f32[2,4] negate(param0)
  b = f32[2,4] negate(a)
  c = f32[2,4] negate(b)
  d = f32[2,4] negate(c)
  e = f32[2,4] negate(d)
  ROOT f = f32[2,4] add(e, b)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  // Make instruction c reserve 64 bytes in the alternate memory. This should
  // prevent both b and c to put their outputs in the alternate memory.
  options.reserved_scoped_memory_fn =
      [&](const HloInstruction* instruction,
          const absl::flat_hash_set<std::pair<int, ShapeIndex>>
              operands_in_alternate_memory,
          const absl::flat_hash_set<ShapeIndex> outputs_in_alternate_memory) {
        if (instruction->name() == "c") {
          return 100;
        }
        return 0;
      };
  AssignMemorySpace(module.get(), options);
  auto get_memory_space = [&](absl::string_view instruction_name) {
    return module->entry_computation()
        ->GetInstructionWithName(instruction_name)
        ->shape()
        .layout()
        .memory_space();
  };
  EXPECT_TRUE(get_memory_space("a") == kAlternateMemorySpace);
  EXPECT_TRUE(get_memory_space("b") == kDefaultMemorySpace);
  EXPECT_TRUE(get_memory_space("c") == kDefaultMemorySpace);
  EXPECT_TRUE(get_memory_space("d") == kAlternateMemorySpace);
  EXPECT_TRUE(get_memory_space("e") == kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, ConstantAllocationFar) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param0 = f32[2,4] parameter(0)
  const = f32[2,4] constant({...})
  a = f32[2,4] negate(param0)
  b = f32[2,4] negate(a)
  c = f32[2,4] negate(b)
  d = f32[2,4] negate(c)
  e = f32[2,4] negate(d)
  ROOT negate = f32[2,4] add(const, e)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
  EXPECT_TRUE(module->entry_computation()
                  ->GetInstructionWithName("const")
                  ->shape()
                  .layout()
                  .memory_space() == kDefaultMemorySpace);
  EXPECT_TRUE(module->entry_computation()
                  ->GetInstructionWithName("negate")
                  ->operand(0)
                  ->shape()
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, ConstantAllocationNear) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param0 = f32[2,4] parameter(0)
  a = f32[2,4] negate(param0)
  b = f32[2,4] negate(a)
  c = f32[2,4] negate(b)
  d = f32[2,4] negate(c)
  e = f32[2,4] negate(d)
  const = f32[2,4] constant({...})
  ROOT negate = f32[2,4] add(const, e)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
  EXPECT_TRUE(module->entry_computation()
                  ->GetInstructionWithName("const")
                  ->shape()
                  .layout()
                  .memory_space() == kDefaultMemorySpace);
  EXPECT_TRUE(module->entry_computation()
                  ->GetInstructionWithName("negate")
                  ->operand(0)
                  ->shape()
                  .layout()
                  .memory_space() == kAlternateMemorySpace);
}

// A mock MemorySpaceAssignmentRepacker class that accepts a map of
// (start_time,offset) -> new_offset values. Using this map, the repacker
// repacks the allocations to the new_offset.
class FakeMemorySpaceAssignmentRepacker : public MemorySpaceAssignmentRepacker {
 public:
  explicit FakeMemorySpaceAssignmentRepacker(
      absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t>& repack_map,
      std::function<void(absl::Span<AllocationBlock*>)> check_fun = nullptr,
      bool always_return_modified = false)
      : MemorySpaceAssignmentRepacker(/*max_size=*/128, /*alignment=*/8),
        repack_map_(repack_map),
        check_fun_(check_fun),
        always_return_modified_(always_return_modified) {}

  absl::StatusOr<bool> Repack(
      absl::Span<AllocationBlock*> allocations) override {
    bool modified = false;
    for (AllocationBlock* block : allocations) {
      absl::flat_hash_set<int64_t> colocations;
      std::string colocations_str;
      for (const AllocationBlock* colocation : block->GetColocations()) {
        absl::StrAppend(&colocations_str, colocation->id, ", ");
        colocations.insert(colocation->id);
      }
      VLOG(1) << "Alloc id: " << block->id << " time: ["
              << block->inclusive_start_time << ", " << block->end_time
              << "] size: " << block->size
              << " init offset: " << block->initial_offset << " colocations: {"
              << colocations_str << "}";
      auto it = repack_map_.find(
          {block->inclusive_start_time, block->initial_offset});
      if (it != repack_map_.end()) {
        modified = true;
        block->offset = it->second;
      } else {
        block->offset = block->initial_offset;
      }
      for (AllocationBlock* colocation : block->GetColocations()) {
        if (it != repack_map_.end()) {
          colocation->offset = it->second;
        } else {
          colocation->offset = colocation->initial_offset;
        }
      }
    }
    if (check_fun_) {
      check_fun_(allocations);
    }

    return always_return_modified_ || modified;
  }

 private:
  // A map from (start_time, offset) to new_offset.
  absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t> repack_map_;
  std::function<void(absl::Span<AllocationBlock*>)> check_fun_;
  bool always_return_modified_;
};

TEST_F(MemorySpaceAssignmentTest, Repack) {
  // We initially perform the following allocations at these offsets.
  //
  //    Max memory
  //  -------------------------------------------
  //
  //
  //
  //
  //      +------------+
  //      |     b      |
  //      +------------+
  //  +-------+                 +------------+
  //  |   a   |                 |     n      |
  //  +-------+                 +------------+
  //  -------------------------------------------
  //    Min memory          time ->
  //
  // Next up, we try to allocate the prefetch for m. However due to
  // fragmentation, this won't be possible:
  //
  //    Max memory
  //  -------------------------------------------
  //
  //
  //
  //                +---------+
  //      +------------+      |
  //      |     b   |  |      |
  //      +------------+      |
  //  +-------+     |         | +------------+
  //  |   a   |     |    d    | |     n      |
  //  +-------+     +---------+ +------------+
  //  -------------------------------------------
  //    Min memory          time ->
  //
  // We then call repack to repack the existing allocations which allows us to
  // allocate the prefetch for m:
  //
  //    Max memory
  //  -------------------------------------------
  //                +---------+
  //                |         |
  //                |         |
  //                |         |
  //  +-------+     |         |
  //  |   a   |     |    d    |
  //  +-------+     +---------+
  //      +------------+        +------------+
  //      |      b     |        |     n      |
  //      +------------+        +------------+
  //  -------------------------------------------
  //    Min memory          time ->
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  ENTRY Entry {
    param0 = f32[8,3] parameter(0)
    param1 = f32[2,4] parameter(1)
    a = f32[2,4] sine(param1)
    b = f32[2,4] cosine(param1)
    c = f32[8,3] negate(param0)
    j = f32[2,4] negate(a)
    d = f32[8,3] tanh(param0)
    k = f32[2,4] negate(j)
    l = f32[2,4] add(b, k)
    m = f32[8,3] negate(d)
    n = f32[2,4] sine(l)
    o = f32[8,3] negate(m)
    p = f32[2,4] negate(n)
    q = f32[8,3] negate(m)
    ROOT tuple = (f32[2,4], f32[8,3], f32[8,3]) tuple(p, q, o)
  }
  )";

  MsaBufferIntervalCompare buffer_interval_compare =
      [](const MsaBufferInterval& a, const MsaBufferInterval& b) {
        auto get_opcode_priority = [](const HloOpcode& opcode) {
          switch (opcode) {
            case HloOpcode::kSin:
              return 0;
            case HloOpcode::kCos:
              return 1;
            case HloOpcode::kTanh:
              return 2;
            default:
              return 3;
          }
        };

        return get_opcode_priority(a.buffer->defining_instruction()->opcode()) <
               get_opcode_priority(b.buffer->defining_instruction()->opcode());
      };
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 10);
  absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t> repack_map;
  // Move "a" from offset 0 to 32.
  repack_map[{2, 0}] = 32;
  // Move "b" from offset 32 to 0.
  repack_map[{3, 32}] = 0;
  FakeMemorySpaceAssignmentRepacker repacker =
      FakeMemorySpaceAssignmentRepacker(repack_map);
  Options options = DefaultMemorySpaceOptions();
  options.max_repacks = 1;
  options.repacker = &repacker;
  AssignMemorySpace(module.get(), options, buffer_interval_compare,
                    &prefetch_interval_picker);

  // If repacking succeeds, we should find the buffer for d in alternate memory.
  const HloInstruction* d =
      module->entry_computation()->GetInstructionWithName("d");
  EXPECT_EQ(d->shape().layout().memory_space(), kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, RepackExportsAliasedOffsets) {
  // This test is that we are correctly exporting aliased offsets for repacking.
  // In this example, the buffer produced at HLO "a" will be allocated first,
  // and will consist of four allocations:
  //    1) a produced in the alternate memory (and then evicted to the default
  //    memory). 2) a prefetched to the alternate memory to be used by q and
  //    while HLOs. 3) a used within the while loop body. 4) the output of while
  //    HLO, used by u.
  //
  // Since a will be allocated first (the test is crafted to prioritize sine
  // HLO), all four allocations should get the same (zero) offsets. However,
  // while allocations 2, 3, and 4 need to be colocated with each other,
  // allocation 1 doesn't need to be colocated with the other three.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  while_condition {
    param1 = (f32[2,4], f32[2,4]) parameter(0)
    ROOT cond = pred[] constant(true)
  }

  while_body {
    param2 = (f32[2,4], f32[2,4]) parameter(0)
    gte2 = f32[2,4] get-tuple-element(param2), index=0
    gte3 = f32[2,4] get-tuple-element(param2), index=1
    add = f32[2,4] add(gte2, gte3)
    ROOT tuple2 = (f32[2,4], f32[2,4]) tuple(add, gte3)
  }

  ENTRY Entry {
    param0 = f32[2,4] parameter(0)
    a = f32[2,4] sine(param0)
    b = f32[2,4] negate(a)
    c = f32[2,4] negate(b)
    d = f32[2,4] negate(c)
    e = f32[2,4] negate(d)
    f = f32[2,4] negate(e)
    g = f32[2,4] negate(f)
    h = f32[2,4] negate(g)
    i = f32[2,4] negate(h)
    j = f32[2,4] negate(i)
    k = f32[2,4] negate(j)
    l = f32[2,4] negate(k)
    m = f32[2,4] negate(l)
    n = f32[2,4] negate(m)
    o = f32[2,4] negate(n)
    p = f32[2,4] negate(o)
    q = f32[2,4] add(p, a)
    tuple = (f32[2,4], f32[2,4]) tuple(q, a)
    while = (f32[2,4], f32[2,4]) while(tuple), condition=while_condition, body=while_body
    gte0 = f32[2,4] get-tuple-element(while), index=0
    gte1 = f32[2,4] get-tuple-element(while), index=1
    r = f32[2,4] negate(gte0)
    s = f32[2,4] negate(r)
    t = f32[2,4] negate(s)
    constant = f32[] constant(0)
    broadcast = f32[8,4] broadcast(constant), dimensions={}
    cos = f32[8,4] cosine(broadcast)
    u = f32[2,4] add(t, gte1)
    v = f32[2,4] add(u, param0)
    w = f32[8,4] negate(cos)
    ROOT tuple3 = (f32[2,4], f32[8,4]) tuple(v, w)
  }
  )";

  MsaBufferIntervalCompare buffer_interval_compare =
      [](const MsaBufferInterval& a, const MsaBufferInterval& b) {
        auto get_opcode_priority = [](const HloOpcode& opcode) {
          switch (opcode) {
            case HloOpcode::kSin:
              return 0;
            case HloOpcode::kCos:
              return 1;
            case HloOpcode::kTanh:
              return 2;
            default:
              return 3;
          }
        };

        return get_opcode_priority(a.buffer->defining_instruction()->opcode()) <
               get_opcode_priority(b.buffer->defining_instruction()->opcode());
      };
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 10);
  absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t> repack_map;

  // Expect that of the four separate allocations for the "a" buffer, the first
  // and the next three are in separate colocations.
  auto check_fun = [](absl::Span<AllocationBlock*> allocations) {
    EXPECT_TRUE(allocations.at(0)->GetColocationsCount() == 1 ||
                allocations.at(0)->GetColocationsCount() == 3);
    EXPECT_EQ(allocations.at(1)->GetColocationsCount(), 3);
    EXPECT_EQ(allocations.at(2)->GetColocationsCount(), 3);
    EXPECT_TRUE(allocations.at(3)->GetColocationsCount() == 1 ||
                allocations.at(3)->GetColocationsCount() == 3);
  };
  FakeMemorySpaceAssignmentRepacker repacker =
      FakeMemorySpaceAssignmentRepacker(repack_map, check_fun);
  Options options = DefaultMemorySpaceOptions();
  options.max_repacks = 1;
  options.repacker = &repacker;
  AssignMemorySpace(module.get(), options, buffer_interval_compare,
                    &prefetch_interval_picker);
}

TEST_F(MemorySpaceAssignmentTest,
       RepackExportsAliasedOffsetsForReservedScopedMemory) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param0 = f32[2,4] parameter(0)
  a = f32[2,4] negate(param0)
  b = f32[2,4] negate(a)
  c = f32[2,4] negate(b)
  d = f32[2,4] negate(c)
  e = f32[2,4] negate(d)
  ROOT f = f32[2,4] add(e, b)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.max_repacks = 1;
  // Make two instructions reserve scoped memory.
  options.reserved_scoped_memory_fn =
      [&](const HloInstruction* instruction,
          const absl::flat_hash_set<std::pair<int, ShapeIndex>>
              operands_in_alternate_memory,
          const absl::flat_hash_set<ShapeIndex> outputs_in_alternate_memory) {
        if (instruction->name() == "c" || instruction->name() == "d") {
          return 100;
        }
        return 0;
      };

  absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t> repack_map;
  bool repacker_ran = false;

  // Expect that the first two value to repack has a colocations size of 2,
  // corresponding to the scoped allocations.
  auto check_fun = [&](absl::Span<AllocationBlock*> allocations) {
    EXPECT_EQ(allocations.at(0)->GetColocationsCount(), 2);
    EXPECT_EQ(allocations.at(1)->GetColocationsCount(), 2);
    repacker_ran = true;
  };
  FakeMemorySpaceAssignmentRepacker repacker =
      FakeMemorySpaceAssignmentRepacker(repack_map, check_fun);
  options.repacker = &repacker;
  AssignMemorySpace(module.get(), options);
  EXPECT_TRUE(repacker_ran);
}

TEST_F(MemorySpaceAssignmentTest,
       ReduceReservedScopedAllocationIfOperandInAlternateMemory) {
  // This test is designed to test UpdateReservedScopedAllocationSize() in MSA,
  // which will invoke reserved_scoped_memory_fn to update scoped allocation
  // size. UpdateReservedScopedAllocationSize() should iterate through all
  // scheduled instruction and check if either their operands or outputs has
  // been assigned in alternate memory. If so, corresponding operand/output will
  // be passed to reserved_scoped_memory_fn. We isolate
  // UpdateReservedScopedAllocationSize() by constructing a dummy
  // reserved_scoped_memory_fn that return +1 when operand set is empty, and
  // return +2 when output set is empty, because if either set of an instruction
  // is empty, it is gureented that some scoped allocation is required. We use
  // +1/+2 to distinguish the correctness of each set. Therefore, after MSA
  // pass, for each instruction, there are a few possible outcomes:
  // 1. If both operand set and output set are not empty, scoped allocation
  //    size should be 0, since reserved_scoped_memory_fn will return 0.
  // 2. If only operand set is empty, scoped allocation size should be 2, since
  //    reserved_scoped_memory_fn will return 2.
  // 3. If only output set is empty, scoped allocation size should be 1, since
  //    reserved_scoped_memory_fn will return 1.
  // 4. If both sets are empty, scoped allocation size should be 3.
  // Initially, UpdateReservedScopedAllocationSize() will only be invoked after
  // each MSA repacking, we use a similar test HLO module as used in "Repack"
  // test. This test is capable of testing if
  // UpdateReservedScopedAllocationSize() can correctly pass operand/output set
  // of all instructions to reserved_scoped_memory_fn.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  ENTRY Entry {
    param0 = f32[8,3] parameter(0)
    param1 = f32[2,4] parameter(1)
    a = f32[2,4] sine(param1)
    b = f32[2,4] cosine(param1)
    c = f32[8,3] negate(param0)
    j = f32[2,4] negate(a)
    d = f32[8,3] tanh(param0)
    k = f32[2,4] negate(j)
    l = f32[2,4] add(b, k)
    m = f32[8,3] negate(d)
    n = f32[2,4] sine(l)
    o = f32[8,3] negate(m)
    p = f32[2,4] negate(n)
    q = f32[8,3] negate(m)
    ROOT tuple = (f32[2,4], f32[8,3], f32[8,3], f32[8,3]) tuple(p, q, o, c)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t> repack_map;
  Options options = DefaultMemorySpaceOptions();
  options.max_repacks = 10;
  options.repack_after_every_allocation = true;
  options.reduce_scoped_memory_limit = true;
  options.reserved_scoped_memory_fn =
      [&](const HloInstruction* instruction,
          const absl::flat_hash_set<std::pair<int, ShapeIndex>>
              operands_in_alternate_memory,
          const absl::flat_hash_set<ShapeIndex> outputs_in_alternate_memory) {
        int64_t scoped_memory_size = 0;
        if (operands_in_alternate_memory.empty()) {
          scoped_memory_size += 1;
          LOG(INFO) << instruction->name()
                    << " has no operand in alternate memory";
        }
        if (outputs_in_alternate_memory.empty()) {
          scoped_memory_size += 2;
          LOG(INFO) << instruction->name()
                    << " has no output in alternate memory";
        }
        return scoped_memory_size;
      };
  FakeMemorySpaceAssignmentRepacker repacker =
      FakeMemorySpaceAssignmentRepacker(repack_map, nullptr);
  options.repacker = &repacker;
  std::unique_ptr<PresetAssignments> assignments =
      AssignMemorySpace(module.get(), options);
  // This lambda checks if an instruction's operand has been assigned in
  // alternate memory.
  auto instruction_consumes_assignment_fn =
      [&](absl::string_view instruction_name) -> bool {
    HloInstruction* instruction =
        module->entry_computation()->GetInstructionWithName(instruction_name);
    for (auto& pair : assignments->chunks()) {
      HloInstruction* consumer = pair.first.instruction;
      if (absl::c_any_of(instruction->operands(),
                         [&](const HloInstruction* operand) {
                           return operand == consumer;
                         })) {
        return true;
      }
    }
    return false;
  };
  // This lambda checks if an instruction's output has been assigned in
  // alternate memory.
  auto instruction_produces_assignment_fn =
      [&](absl::string_view instruction_name) -> bool {
    HloInstruction* instruction =
        module->entry_computation()->GetInstructionWithName(instruction_name);
    for (auto& pair : assignments->chunks()) {
      HloInstruction* producer = pair.first.instruction;
      if (producer == instruction) {
        return true;
      }
    }
    return false;
  };
  auto check_reserved_scoped_memory_fn =
      [&](absl::string_view instruction_name) -> bool {
    int64_t scoped_memory_size = -1;
    for (auto& pair : assignments->scoped_allocation_chunks()) {
      HloInstruction* instruction = pair.first;
      if (instruction->name() == instruction_name) {
        scoped_memory_size = pair.second.size;
      }
    }
    if (!instruction_consumes_assignment_fn(instruction_name)) {
      scoped_memory_size -= 1;
    }
    if (!instruction_produces_assignment_fn(instruction_name)) {
      scoped_memory_size -= 2;
    }
    return scoped_memory_size == 0;
  };
  for (auto& pair : assignments->assignment_informations()) {
    LOG(INFO) << "  space: " << pair.first << ", size: " << pair.second.size;
  }
  for (auto& pair : assignments->scoped_allocation_chunks()) {
    HloInstruction* instruction = pair.first;
    LOG(INFO) << instruction->name() << ": " << pair.second.size;
  }
  EXPECT_TRUE(check_reserved_scoped_memory_fn("a"));
  EXPECT_TRUE(check_reserved_scoped_memory_fn("b"));
  EXPECT_TRUE(check_reserved_scoped_memory_fn("c"));
  EXPECT_TRUE(check_reserved_scoped_memory_fn("j"));
  EXPECT_TRUE(check_reserved_scoped_memory_fn("d"));
  EXPECT_TRUE(check_reserved_scoped_memory_fn("k"));
  EXPECT_TRUE(check_reserved_scoped_memory_fn("l"));
  EXPECT_TRUE(check_reserved_scoped_memory_fn("m"));
  EXPECT_TRUE(check_reserved_scoped_memory_fn("n"));
  EXPECT_TRUE(check_reserved_scoped_memory_fn("o"));
  EXPECT_TRUE(check_reserved_scoped_memory_fn("p"));
  EXPECT_TRUE(check_reserved_scoped_memory_fn("q"));
}

TEST_F(MemorySpaceAssignmentTest, ScopedAllocationWithDifferentOffset) {
  // This is test is designed against a bug when
  // allocate_reserved_scoped_memory_at_same_offset to false, repack block of
  // scoped allocation has empty colocations. This is resolved by adding each
  // block itself as its own collocation. We test this by checking colocation
  // sizes during repacking.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true

  ENTRY Entry {
    param0 = f32[8,3] parameter(0)
    param1 = f32[2,4] parameter(1)
    a = f32[2,4] sine(param1)
    b = f32[2,4] cosine(param1)
    c = f32[8,3] negate(param0)
    j = f32[2,4] negate(a)
    d = f32[8,3] tanh(param0)
    k = f32[2,4] negate(j)
    l = f32[2,4] add(b, k)
    m = f32[8,3] negate(d)
    n = f32[2,4] sine(l)
    o = f32[8,3] negate(m)
    p = f32[2,4] negate(n)
    q = f32[8,3] negate(m)
    ROOT tuple = (f32[2,4], f32[8,3], f32[8,3]) tuple(p, q, o)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto check_fun = [](absl::Span<AllocationBlock*> allocations) {
    for (AllocationBlock* block : allocations) {
      if (block->inclusive_start_time == block->end_time) {
        EXPECT_GT(block->GetColocationsCount(), 0);
      }
    }
  };
  absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t> repack_map;
  FakeMemorySpaceAssignmentRepacker repacker =
      FakeMemorySpaceAssignmentRepacker(repack_map, check_fun);
  Options options = DefaultMemorySpaceOptions();
  // Scoped allocation needs to have non zero limit.
  options.reserved_scoped_memory_fn =
      [&](const HloInstruction* instruction,
          const absl::flat_hash_set<std::pair<int, ShapeIndex>>
              operands_in_alternate_memory,
          const absl::flat_hash_set<ShapeIndex> outputs_in_alternate_memory) {
        return 1;
      };
  options.max_repacks = 1;
  options.repacker = &repacker;
  options.allocate_reserved_scoped_memory_at_same_offset = false;
  AssignMemorySpace(module.get(), options);
}

TEST_F(MemorySpaceAssignmentTest,
       ReduceReservedScopedAllocationUpdatesPeakMemoryUsage) {
  // This test is designed to test that the peak_memory_usage_ is updated
  // correctly after scoped memory allocation is updated. The test HLO module
  // has two HLO values: a and b. The size of a is 64, and the size of b is 128.
  // We expect both of them to be allocated in alternate memory.
  //
  // The reduce_scoped_memory_fn will initially report the scoped
  // memory allocation of c as 64. So if b is allocated in the alternate memory,
  // the peak memory usage at the time of c will be 196 (64 as scoped memory +
  // 128 for its operand). This setup uses up all the memory and prevents a from
  // being allocated in alternate memory. However, we also set
  // reduce_scoped_memory_fn to report c's scoped memory allocation as 0 if its
  // operand is prefetched. So, after repacking, the peak memory usage at the
  // time of c should be reduced to 128 (0 as scoped memory + 128 for its
  // operand). This allows a to be allocated in the alternate memory.
  //
  // If the peak_memory_usage_ is not updated correctly, we will hit a check
  // failure and a will not be able to be allocated in alternate memory.
  absl::string_view hlo_string = R"(
  HloModule bug, is_scheduled=true
  ENTRY Entry {
    param0 = f32[] parameter(0)
    param1 = f32[16] parameter(1)

    a = f32[16] negate(param1)
    b = f32[32] broadcast(param0), dimensions={}
    c = f32[32] sine(b)
    d = f32[16] negate(a)
    ROOT tuple = (f32[16], f32[32]) tuple(d, c)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t> repack_map;
  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 128 + 64;
  options.max_repacks = 10;
  options.repack_after_every_allocation = true;
  options.reduce_scoped_memory_limit = true;
  options.reserved_scoped_memory_fn =
      [&](const HloInstruction* instruction,
          const absl::flat_hash_set<std::pair<int, ShapeIndex>>
              operands_in_alternate_memory,
          const absl::flat_hash_set<ShapeIndex> outputs_in_alternate_memory)
      -> int64_t {
    if (instruction->opcode() != HloOpcode::kSin) {
      return 0;
    }
    // Reserve 64 bytes as scoped memory, when we initially allocate scoped
    // memory for c. After b is prefetched, reduce the scoped memory to 0.
    if (operands_in_alternate_memory.empty()) {
      return 64;
    }
    return 0;
  };
  FakeMemorySpaceAssignmentRepacker repacker =
      FakeMemorySpaceAssignmentRepacker(repack_map, nullptr);
  options.repacker = &repacker;
  AssignMemorySpace(module.get(), options);
}

TEST_F(MemorySpaceAssignmentTest,
       RepackShouldntEraseRequiredAssignmentForConditionalOutput) {
  // This is a test case for b/171040271. Repacks erase the required assignments
  // (since some required assignments are inserted conditionally based on
  // allocation decisions), including the fact that conditional outputs are
  // always required to get assignments in the default memory. After repacking,
  // this required assignment was never added back, causing conditionals to get
  // alternate-memory allocations.
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation {
    p0 = (f32[3]) parameter(0)
    gte = f32[3] get-tuple-element(p0), index=0
    neg1 = f32[3] negate(gte)
    ROOT tuple1 = (f32[3]) tuple(neg1)
  }

  false_computation {
    p0 = (f32[3]) parameter(0)
    gte = f32[3] get-tuple-element(p0), index=0
    neg2 = f32[3] negate(gte)
    ROOT tuple2 = (f32[3]) tuple(neg2)
  }

  ENTRY entry {
    p0 = f32[3] parameter(0)
    p1 = pred[] parameter(1)
    copy = f32[3] copy(p0)
    tuple = (f32[3]) tuple(copy)
    conditional = (f32[3]) conditional(p1, tuple, tuple), true_computation=true_computation, false_computation=false_computation
    ROOT gte = f32[3] get-tuple-element(conditional), index=0
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t> repack_map;
  FakeMemorySpaceAssignmentRepacker repacker =
      FakeMemorySpaceAssignmentRepacker(repack_map, nullptr,
                                        /*always_return_modified=*/true);
  Options options = DefaultMemorySpaceOptions();
  options.max_repacks = 10;
  options.repacker = &repacker;
  options.repack_after_every_allocation = true;
  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 10);
  AssignMemorySpace(module.get(), options,
                    /*buffer_interval_compare=*/{}, &prefetch_interval_picker);
}

TEST_F(MemorySpaceAssignmentTest, Determinism) {
  // Run memory space assignment a few times to make sure every time it compiles
  // to the same thing.
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get());
  std::string module_str = module->ToString();

  for (int i = 0; i < 10; ++i) {
    std::unique_ptr<HloModule> other_module = CreateEvictAndPrefetchModule();
    AssignMemorySpace(other_module.get());
    EXPECT_EQ(module_str, other_module->ToString());
  }
}

TEST_F(MemorySpaceAssignmentTest, InPlaceOp) {
  // Tests that in-place ops like DynamicUpdateSlice get the same allocation as
  // its input.
  absl::string_view hlo_string = R"(
HloModule Module, is_scheduled=true

fused_computation {
  param0 = f32[2,3] parameter(0)
  constant.1 = f32[] constant(0)
  broadcast = f32[2,1] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  ROOT dynamic-update-slice.5 = f32[2,3] dynamic-update-slice(param0, broadcast, constant.3, constant.3)
}

ENTRY main {
  param = f32[2,3] parameter(0)
  negate = f32[2,3] negate(param)
  fusion = f32[2,3] fusion(negate), kind=kLoop, calls=fused_computation
  ROOT add = f32[2,3] add(fusion, fusion)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(module.get());
  HloInstruction* negate_instruction =
      module->entry_computation()->GetInstructionWithName("negate");
  int64_t negate_offset =
      GetAlternateMemoryOffset(*preset_assignments, negate_instruction);
  HloInstruction* fusion_instruction =
      module->entry_computation()->GetInstructionWithName("fusion");
  int64_t fusion_offset =
      GetAlternateMemoryOffset(*preset_assignments, fusion_instruction);
  // We expect negate and fusion to get the same offsets.
  EXPECT_EQ(negate_offset, fusion_offset);
  EXPECT_NE(negate_offset, -1);
}

TEST_F(MemorySpaceAssignmentTest, ConditionalInPlaceOp) {
  absl::string_view hlo_string = R"(
HloModule Module, is_scheduled=true

fused_computation {
  param0 = f32[2,3] parameter(0)
  constant.1 = f32[] constant(0)
  broadcast = f32[2,1] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  ROOT dynamic-update-slice.5 = f32[2,3] dynamic-update-slice(param0, broadcast, constant.3, constant.3)
}

true_computation {
  p0 = (f32[2,3]) parameter(0)
  gte = f32[2,3] get-tuple-element(p0), index=0
  ROOT neg1 = f32[2,3] negate(gte)
}

false_computation {
  p0 = (f32[2,3]) parameter(0)
  gte = f32[2,3] get-tuple-element(p0), index=0
  neg2 = f32[2,3] negate(gte)
  ROOT fusion = f32[2,3] fusion(neg2), kind=kLoop, calls=fused_computation
}

ENTRY entry {
  p0 = f32[2,3] parameter(0)
  p1 = pred[] parameter(1)
  copy = f32[2,3] copy(p0)
  tuple = (f32[2,3]) tuple(copy)
  ROOT conditional = f32[2,3] conditional(p1, tuple, tuple), true_computation=true_computation, false_computation=false_computation
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
}

TEST_F(MemorySpaceAssignmentTest, AsyncCallDisableAlternateMem) {
  absl::string_view hlo_string = R"(
HloModule Module, is_scheduled=true

called_comp {
  p0 = f32[2,3] parameter(0)
  negate10 = f32[2,3] negate(p0)
  negate11 = f32[2,3] negate(negate10)
  negate12 = f32[2,3] negate(negate11)
  negate13 = f32[2,3] negate(negate12)
  negate14 = f32[2,3] negate(negate13)
  ROOT negate15 = f32[2,3] negate(negate14)
}, execution_thread="foobar"

async_comp {
  p0 = f32[2,3] parameter(0)
  ROOT call = f32[2,3] call(p0), to_apply=called_comp
}, execution_thread="foobar"

ENTRY entry {
  p0 = f32[2,3] parameter(0)
  negate0 = f32[2,3] negate(p0)
  negate1 = f32[2,3] negate(negate0)
  negate2 = f32[2,3] negate(negate1)
  negate3 = f32[2,3] negate(negate2)
  negate4 = f32[2,3] negate(negate3)
  async-start = ((f32[2,3]), f32[2,3], f32[2]) async-start(negate1), async_execution_thread="foobar", calls=async_comp
  async-done = f32[2,3] async-done(async-start), async_execution_thread="foobar", calls=async_comp
  add0 = f32[2,3] add(negate0, async-done)
  negate5 = f32[2,3] negate(add0)
  negate6 = f32[2,3] negate(negate5)
  negate7 = f32[2,3] negate(negate6)
  negate8 = f32[2,3] negate(negate7)
  negate9 = f32[2,3] negate(negate8)
  ROOT add1 = f32[2,3] add(negate9, async-done)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.is_use_allowed_in_alternate_mem_fn = [](const HloUse& use) {
    return use.instruction->opcode() != HloOpcode::kAsyncStart &&
           use.instruction->opcode() != HloOpcode::kAsyncDone &&
           use.instruction->parent()->IsMainThread();
  };
  options.is_position_allowed_in_alternate_mem_fn = [](const HloPosition& pos) {
    return pos.instruction->opcode() != HloOpcode::kAsyncStart &&
           pos.instruction->opcode() != HloOpcode::kAsyncDone &&
           pos.instruction->parent()->IsMainThread();
  };
  AssignMemorySpace(module.get(), options);
  auto has_alternate_memory_allocation =
      [&](const HloInstruction* instruction) {
        bool result = false;
        auto shape_has_alternate_memory_allocation =
            [&](const Shape& subshape, const ShapeIndex& /*index*/) {
              if (subshape.IsArray() &&
                  subshape.layout().memory_space() == kAlternateMemorySpace) {
                result = true;
              }
            };
        ShapeUtil::ForEachSubshape(instruction->shape(),
                                   shape_has_alternate_memory_allocation);
        for (const HloInstruction* operand : instruction->operands()) {
          ShapeUtil::ForEachSubshape(operand->shape(),
                                     shape_has_alternate_memory_allocation);
        }
        return result;
      };

  // Check that the async ops themselves and the instructions inside async
  // computations do not have any alternate memory allocations.
  const HloInstruction* async_start =
      FindInstruction(module.get(), "async-start");
  const HloInstruction* async_done =
      FindInstruction(module.get(), "async-done");
  EXPECT_FALSE(has_alternate_memory_allocation(async_start));
  EXPECT_FALSE(has_alternate_memory_allocation(async_done));
  for (const HloInstruction* instruction :
       async_start->async_wrapped_instruction()
           ->called_computations()[0]
           ->instructions()) {
    EXPECT_FALSE(has_alternate_memory_allocation(instruction));
  }
  // Check that we still allow the tensor used/produced by the async computation
  // to be placed in the alternate memory before/after the async computation.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Add(op::Negate(),
                      op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                    op::AsyncDone())));
  EXPECT_THAT(async_start,
              op::AsyncStart(op::AsyncCopy(
                  kDefaultMemorySpace, kAlternateMemorySpace, op::Negate())));
}

TEST_F(MemorySpaceAssignmentTest, InefficientAllocation) {
  // The DUS in the fusion only accesses 1/3 of its input/output. The fusion
  // input/output buffer is a program input/output buffer, so it creates an
  // prefetch and an eviction. When we turn on detecting inefficient
  // allocations, we should catch this case and allocate the fusion input/output
  // in the default memory space.
  absl::string_view hlo_string = R"(
HloModule Module, is_scheduled=true

fused_computation {
  param0 = f32[2,3] parameter(0)
  constant.1 = f32[] constant(0)
  broadcast = f32[2,1] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  ROOT dynamic-update-slice.5 = f32[2,3] dynamic-update-slice(param0, broadcast, constant.3, constant.3)
}

ENTRY entry {
  p0 = f32[2,3] parameter(0)
  p1 = pred[] parameter(1)
  p2 = f32[2,3] parameter(2)
  neg0 = f32[2,3] negate(p2)
  neg1 = f32[2,3] negate(neg0)
  neg2 = f32[2,3] negate(neg1)
  neg3 = f32[2,3] negate(neg2)
  neg4 = f32[2,3] negate(neg3)
  neg5 = f32[2,3] negate(neg4)
  neg6 = f32[2,3] negate(neg5)
  neg7 = f32[2,3] negate(neg6)
  fusion = f32[2,3] fusion(p0), kind=kLoop, calls=fused_computation
  neg8 = f32[2,3] negate(neg7)
  neg9 = f32[2,3] negate(neg8)
  neg10 = f32[2,3] negate(neg9)
  neg11 = f32[2,3] negate(neg10)
  neg12 = f32[2,3] negate(neg11)
  neg13 = f32[2,3] negate(neg12)
  neg14 = f32[2,3] negate(neg13)
  neg15 = f32[2,3] negate(neg14)
  ROOT tuple = (f32[2,3], f32[2,3]) tuple(fusion, neg15)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  Options options = DefaultMemorySpaceOptions();
  options.enable_cross_program_prefetch = false;
  // Disable inefficiency check. Expect that the fusion output and operand are
  // in the alternate memory.
  options.inefficient_use_to_copy_ratio = 0.0;
  AssignMemorySpaceUsingCostAnalysis(module.get(),
                                     /*memory_space_options_override=*/options);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::AsyncCopy(kDefaultMemorySpace, kAlternateMemorySpace,
                              op::Fusion(op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter()))),
                op::Negate()));

  // Re-run MSA with inefficient use-to-copy ratio of 0.5. The fusion only uses
  // 8B of data (f32[2,1]) but copies 48B of data (prefetch and eviction of
  // f32[2,3]), so this should be considered inefficient (8/48 < 0.5).
  TF_ASSERT_OK_AND_ASSIGN(module, ParseAndReturnVerifiedModule(hlo_string));
  options.inefficient_use_to_copy_ratio = 0.5;
  AssignMemorySpaceUsingCostAnalysis(module.get(),
                                     /*memory_space_options_override=*/options);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Fusion(op::Parameter()), op::Negate()));
}

TEST_F(MemorySpaceAssignmentTest, InefficientAllocationLivelockBug) {
  // This is a carefully crafted test where two in-place operations on the same
  // buffer (fusion.1 and fusion.2) have a single very long executing operation
  // between them. This test deliberately sets a very low transcendentals per
  // second value to ensure the tanh op takes longer than what is allowed for a
  // no-copy allocation. A quirk of the prefetch interval picker allows a
  // prefetch to be scheduled during this tanh operation even though a no-copy
  // allocation isn't allowed. Because of this, the first time this buffer is
  // allocated, fusion.1 will be put in the alternate memory, but not fusion.2
  // because tanh executes for too long to allow a no-copy allocation. Then, we
  // check for inefficient allocations, and consider fusion.1 to be inefficient,
  // and add a required assignment in default memory for fusion.1 and
  // reallocate. When we reallocate, we aren't allowed to prefetch into
  // fusion.1, but fusion.2 succeeds. We then find fusion.2 to be inefficient,
  // so we throw away the required assignment on fusion.1 and reallocate.
  // Without the appropriate fix, this will go on forever, causing a livelock.
  absl::string_view hlo_string = R"(
HloModule Module, is_scheduled=true

fused_computation_1 {
  param0 = f32[5,4] parameter(0)
  constant.1 = f32[] constant(0)
  broadcast = f32[5,1] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  ROOT dynamic-update-slice.5 = f32[5,4] dynamic-update-slice(param0, broadcast, constant.3, constant.3)
}

fused_computation_2 {
  param0 = f32[5,4] parameter(0)
  constant.1 = f32[] constant(0)
  broadcast = f32[5,1] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  ROOT dynamic-update-slice.5 = f32[5,4] dynamic-update-slice(param0, broadcast, constant.3, constant.3)
}

ENTRY entry {
  p0 = f32[5,4] parameter(0)
  p1 = pred[] parameter(1)
  p2 = f32[2,3] parameter(2)
  neg0 = f32[2,3] negate(p2)
  neg1 = f32[2,3] negate(neg0)
  neg2 = f32[2,3] negate(neg1)
  neg3 = f32[2,3] negate(neg2)
  neg4 = f32[2,3] negate(neg3)
  neg5 = f32[2,3] negate(neg4)
  neg6 = f32[2,3] negate(neg5)
  neg7 = f32[2,3] negate(neg6)
  fusion.1 = f32[5,4] fusion(p0), kind=kLoop, calls=fused_computation_1
  tanh = f32[2,3] tanh(neg7)
  fusion.2 = f32[5,4] fusion(fusion.1), kind=kLoop, calls=fused_computation_2
  neg8 = f32[2,3] negate(tanh)
  neg9 = f32[2,3] negate(neg8)
  neg10 = f32[2,3] negate(neg0)
  neg11 = f32[2,3] negate(neg10)
  neg12 = f32[2,3] negate(neg11)
  ROOT tuple = (f32[5,4], f32[2,3]) tuple(fusion.2, neg12)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  Options options = DefaultMemorySpaceOptions();
  options.enable_cross_program_prefetch = false;
  options.inefficient_use_to_copy_ratio = 0.5;
  HloCostAnalysis::Options hlo_cost_options = DefaultHloCostAnalysisOptions();
  hlo_cost_options.set_transcendentals_per_second(0.4);

  AssignMemorySpaceUsingCostAnalysis(
      module.get(), /*memory_space_options_override=*/options,
      /*cost_analysis_options_override=*/std::nullopt,
      /*hlo_cost_options_override=*/hlo_cost_options);
}

TEST_F(MemorySpaceAssignmentTest,
       CalledComputationInefficientAllocationLiveLockBug) {
  // Parameter 2 is passed into the conditional but it is not actually used. A
  // bug in inefficient allocation pinned this buffer to the default memory at
  // the use time, but we should really use the "corrected" use time which is
  // the earliest-scheduled called computation parameter.
  absl::string_view hlo_string = R"(
  HloModule CondAllocation, is_scheduled=true

  true_computation {
    p0 = (f32[3], f32[3]) parameter(0)
    gte = f32[3] get-tuple-element(p0), index=0
    neg1 = f32[3] negate(gte)
    ROOT tuple1 = (f32[3]) tuple(neg1)
  }

  false_computation {
    p0 = (f32[3], f32[3]) parameter(0)
    gte = f32[3] get-tuple-element(p0), index=0
    neg2 = f32[3] negate(gte)
    ROOT tuple2 = (f32[3]) tuple(neg2)
  }

  ENTRY entry {
    p0 = f32[3] parameter(0)
    p1 = pred[] parameter(1)
    p2 = f32[3] parameter(2)
    copy0 = f32[3] copy(p0)
    negate0 = f32[3] negate(p0)
    negate1 = f32[3] negate(negate0)
    negate2 = f32[3] negate(negate1)
    negate3 = f32[3] negate(negate2)
    negate4 = f32[3] negate(negate3)
    negate5 = f32[3] negate(negate4)
    negate6 = f32[3] negate(negate5)
    negate7 = f32[3] negate(negate6)
    negate8 = f32[3] negate(negate7)
    tuple = (f32[3], f32[3]) tuple(copy0, p2)
    conditional = (f32[3]) conditional(p1, tuple, tuple), true_computation=true_computation, false_computation=false_computation
    gte = f32[3] get-tuple-element(conditional), index=0
    ROOT add = f32[3] add(gte, negate8)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  Options options = DefaultMemorySpaceOptions();
  options.enable_cross_program_prefetch = false;
  options.inefficient_use_to_copy_ratio = 0.5;
  HloCostAnalysis::Options hlo_cost_options = DefaultHloCostAnalysisOptions();
  hlo_cost_options.set_transcendentals_per_second(0.4);

  AssignMemorySpaceUsingCostAnalysis(
      module.get(), /*memory_space_options_override=*/options,
      /*cost_analysis_options_override=*/std::nullopt,
      /*hlo_cost_options_override=*/hlo_cost_options);
}

TEST_F(MemorySpaceAssignmentTest, AsyncOpElapsedTime) {
  // Test that async ops are treated to take no time. We assume async operations
  // are efficiently scheduled. So, in this example, collective-permute-start
  // should take zero time, which should be insufficient time to overlap a
  // prefetch for negate1's operand.
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param0 = bf16[16]{0} parameter(0)
  param1 = bf16[4]{0} parameter(1)
  collective-permute-start = (bf16[16]{0}, bf16[16]{0}, u32[], u32[]) collective-permute-start(param0), source_target_pairs={{0,1},{1,2},{2,3}}
  negate1 = bf16[4]{0} negate(param1)
  collective-permute-done = bf16[16]{0} collective-permute-done(collective-permute-start)
  ROOT negate2 = bf16[4]{0} negate(negate1)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AssignMemorySpaceUsingCostAnalysis(module.get());
  EXPECT_THAT(FindInstruction(module.get(), "negate1")->operand(0),
              op::Parameter(1));
}

TEST_F(MemorySpaceAssignmentTest, AliasedOperandBug) {
  // Test for a case where two aliased operands into the same instruction
  // (param0 and custom_call2) cause a violation of the required assignment.
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  param0 = f32[4,4]{0,1} parameter(0)
  param1 = f32[4]{0} parameter(1)
  param2 = f32[4,4]{0,1} parameter(2)
  negate0 = f32[4]{0} negate(param1)
  negate1 = f32[4]{0} negate(negate0)
  negate2 = f32[4]{0} negate(negate1)
  negate3 = f32[4]{0} negate(negate2)
  negate4 = f32[4]{0} negate(negate3)
  negate5 = f32[4]{0} negate(negate4)
  custom_call1 = f32[4,4]{0,1} custom-call(param0), custom_call_target="FooBar", output_to_operand_aliasing={{}: (0, {})}
  tanh = f32[4,4]{0,1} tanh(param2)
  negate6 = f32[4]{0} negate(negate5)
  negate7 = f32[4]{0} negate(negate6)
  negate8 = f32[4]{0} negate(negate7)
  negate9 = f32[4]{0} negate(negate8)
  negate10 = f32[4]{0} negate(negate9)
  negate11 = f32[4]{0} negate(negate10)
  negate12 = f32[4]{0} negate(negate11)
  negate13 = f32[4]{0} negate(negate12)
  negate14 = f32[4]{0} negate(negate13)
  negate15 = f32[4]{0} negate(negate14)
  negate16 = f32[4]{0} negate(negate15)
  custom_call2 = f32[4,4]{0,1} custom-call(custom_call1), custom_call_target="FooBar", output_to_operand_aliasing={{}: (0, {})}
  custom_call3 = f32[4,4]{0,1} custom-call(param0, custom_call2), custom_call_target="FooBar", output_to_operand_aliasing={{}: (0, {})}
  ROOT root = f32[4,4]{0,1} add(tanh, custom_call2)
}
  )";

  MsaBufferIntervalCompare buffer_interval_compare =
      [](const MsaBufferInterval& a, const MsaBufferInterval& b) {
        auto get_inst_priority = [](const HloInstruction* instruction) {
          if (instruction->name() == "param2") {
            return 0;
          }
          if (instruction->name() == "param0") {
            return 1;
          }
          return 2;
        };

        return get_inst_priority(a.buffer->defining_instruction()) <
               get_inst_priority(b.buffer->defining_instruction());
      };
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 10);
  Options options = DefaultMemorySpaceOptions();
  AssignMemorySpace(module.get(), options, buffer_interval_compare,
                    &prefetch_interval_picker);
}

TEST_F(MemorySpaceAssignmentTest, AsyncOpCustomFusionShortLiveRange) {
  absl::string_view hlo_string = R"(
HloModule Module, is_scheduled=true

fused_computation_start {
  param0 = f32[2,1] parameter(0)
  negate = f32[2,1] negate(param0)
  ROOT custom-call = (f32[2,1], f32[2,1], u32[], u32[]) custom-call(negate), custom_call_target="AsyncOpStart"
}

fused_computation_update {
  param0 = f32[2,1] parameter(0)
  param1 = f32[2,1] parameter(1)
  param2 = f32[2,1] parameter(2)
  param3 = f32[2,1] parameter(3)
  param4 = u32[] parameter(4)
  param5 = u32[] parameter(5)
  add = f32[2,1] add(param0, param1)
  negate = f32[2,1] negate(param2)
  ROOT tuple = (f32[2,1], f32[2,1], f32[2,1], f32[2,1], u32[], u32[]) tuple(add, param2, param3, negate, param4, param5)
}

fused_computation_done {
  param0 = f32[2,1] parameter(0)
  param1 = f32[2,1] parameter(1)
  param2 = u32[] parameter(2)
  param3 = u32[] parameter(3)
  negate = f32[2,1] negate(param0)
  ROOT custom-call = f32[2,1] custom-call(param0, param1, negate, param2, param3), custom_call_target="AsyncOpDone"
}

ENTRY main {
  param = f32[2,1] parameter(0)
  negate1 = f32[2,1] negate(param)
  negate2 = f32[2,1] negate(negate1)
  fusion1 = (f32[2,1], f32[2,1], u32[], u32[]) fusion(negate1), kind=kCustom, output_to_operand_aliasing={{0}: (0, {})}, calls=fused_computation_start
  negate3 = f32[2,1] negate(negate2)
  negate4 = f32[2,1] negate(negate3)
  gte0 = f32[2,1] get-tuple-element(fusion1), index=0
  gte1 = f32[2,1] get-tuple-element(fusion1), index=1
  gte2 = u32[] get-tuple-element(fusion1), index=2
  gte3 = u32[] get-tuple-element(fusion1), index=3
  fusion2 = (f32[2,1], f32[2,1], f32[2,1], f32[2,1], u32[], u32[]) fusion(negate4, negate2, gte0, gte1, gte2, gte3), kind=kLoop, output_to_operand_aliasing={{1}: (2, {}), {2}: (3, {}), {3}: (3, {}), {4}: (4, {}), {5}: (5, {})}, calls=fused_computation_update
  gte4 = f32[2,1] get-tuple-element(fusion2), index=0
  negate5 = f32[2,1] negate(gte4)
  gte5 = f32[2,1] get-tuple-element(fusion2), index=1
  gte6 = f32[2,1] get-tuple-element(fusion2), index=2
  gte7 = u32[] get-tuple-element(fusion2), index=4
  gte8 = u32[] get-tuple-element(fusion2), index=5
  fusion3 = f32[2,1] fusion(gte5, gte6, gte7, gte8), kind=kCustom, output_to_operand_aliasing={{}: (1, {})}, calls=fused_computation_done
  ROOT add = f32[2,1] add(negate5, fusion3)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.position_requires_contiguous_allocation_fn =
      [](const HloPosition& position) {
        absl::string_view inst_name = position.instruction->name();
        if (inst_name == "fusion1" ||
            (inst_name == "fusion2" && position.index != ShapeIndex({0}))) {
          return true;
        }
        return false;
      };
  AssignMemorySpace(module.get(), options);

  HloInstruction* fusion1 =
      module->entry_computation()->GetInstructionWithName("fusion1");
  HloInstruction* fusion2 =
      module->entry_computation()->GetInstructionWithName("fusion2");
  HloInstruction* fusion3 =
      module->entry_computation()->GetInstructionWithName("fusion3");

  EXPECT_THAT(fusion2->operand(2), op::GetTupleElement(fusion1, 0));
  EXPECT_THAT(fusion2->operand(3), op::GetTupleElement(fusion1, 1));
  EXPECT_THAT(fusion3->operand(0), op::GetTupleElement(fusion2, 1));
  EXPECT_THAT(fusion3->operand(1), op::GetTupleElement(fusion2, 2));
  EXPECT_THAT(fusion2->operand(2)->shape().layout().memory_space(),
              kAlternateMemorySpace);
  EXPECT_THAT(fusion2->operand(3)->shape().layout().memory_space(),
              kAlternateMemorySpace);
  EXPECT_THAT(fusion3->operand(0)->shape().layout().memory_space(),
              kAlternateMemorySpace);
  EXPECT_THAT(fusion3->operand(1)->shape().layout().memory_space(),
              kAlternateMemorySpace);
  // Operand 0 and 1 should get alternate memory allocations and so is the
  // output {0}.
  EXPECT_THAT(fusion2->operand(0)->shape().layout().memory_space(),
              kAlternateMemorySpace);
  EXPECT_THAT(fusion2->operand(1)->shape().layout().memory_space(),
              kAlternateMemorySpace);
  EXPECT_THAT(
      ShapeUtil::GetSubshape(fusion2->shape(), {0}).layout().memory_space(),
      kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, AsyncOpCustomFusionLongLiveRange) {
  absl::string_view hlo_string = R"(
HloModule Module, is_scheduled=true

fused_computation_start {
  param0 = f32[2,1] parameter(0)
  negate = f32[2,1] negate(param0)
  ROOT custom-call = (f32[2,1], f32[2,1], u32[], u32[]) custom-call(negate), custom_call_target="AsyncOpStart"
}

fused_computation_update {
  param0 = f32[2,1] parameter(0)
  param1 = f32[2,1] parameter(1)
  param2 = f32[2,1] parameter(2)
  param3 = f32[2,1] parameter(3)
  param4 = u32[] parameter(4)
  param5 = u32[] parameter(5)
  add = f32[2,1] add(param0, param1)
  negate = f32[2,1] negate(param2)
  ROOT tuple = (f32[2,1], f32[2,1], f32[2,1], f32[2,1], u32[], u32[]) tuple(add, param2, param3, negate, param4, param5)
}

fused_computation_done {
  param0 = f32[2,1] parameter(0)
  param1 = f32[2,1] parameter(1)
  param2 = u32[] parameter(2)
  param3 = u32[] parameter(3)
  negate = f32[2,1] negate(param0)
  ROOT custom-call = f32[2,1] custom-call(param0, param1, negate, param2, param3), custom_call_target="AsyncOpDone"
}

ENTRY main {
  param = f32[2,1] parameter(0)
  negate1 = f32[2,1] negate(param)
  negate2 = f32[2,1] negate(negate1)
  fusion1 = (f32[2,1], f32[2,1], u32[], u32[]) fusion(negate1), kind=kCustom, output_to_operand_aliasing={{0}: (0, {})}, calls=fused_computation_start
  negate3 = f32[2,1] negate(negate2)
  negate4 = f32[2,1] negate(negate3)
  negate5 = f32[2,1] negate(negate4)
  negate6 = f32[2,1] negate(negate5)
  negate7 = f32[2,1] negate(negate6)
  negate8 = f32[2,1] negate(negate7)
  negate9 = f32[2,1] negate(negate8)
  negate10 = f32[2,1] negate(negate9)
  negate11 = f32[2,1] negate(negate10)
  negate12 = f32[2,1] negate(negate11)
  gte0 = f32[2,1] get-tuple-element(fusion1), index=0
  gte1 = f32[2,1] get-tuple-element(fusion1), index=1
  gte2 = u32[] get-tuple-element(fusion1), index=2
  gte3 = u32[] get-tuple-element(fusion1), index=3
  fusion2 = (f32[2,1], f32[2,1], f32[2,1], f32[2,1], u32[], u32[]) fusion(negate12, negate2, gte0, gte1, gte2, gte3), kind=kLoop, output_to_operand_aliasing={{1}: (2, {}), {2}: (3, {}), {3}: (3, {}), {4}: (4, {}), {5}: (5, {})}, calls=fused_computation_update
  gte4 = f32[2,1] get-tuple-element(fusion2), index=0
  negate13 = f32[2,1] negate(gte4)
  gte5 = f32[2,1] get-tuple-element(fusion2), index=1
  gte6 = f32[2,1] get-tuple-element(fusion2), index=2
  gte7 = u32[] get-tuple-element(fusion2), index=4
  gte8 = u32[] get-tuple-element(fusion2), index=5
  fusion3 = f32[2,1] fusion(gte5, gte6, gte7, gte8), kind=kCustom, output_to_operand_aliasing={{}: (1, {})}, calls=fused_computation_done
  ROOT add = f32[2,1] add(negate13, fusion3)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.position_requires_contiguous_allocation_fn =
      [](const HloPosition& position) {
        absl::string_view inst_name = position.instruction->name();
        if (inst_name == "fusion1" ||
            (inst_name == "fusion2" && position.index != ShapeIndex({0}))) {
          return true;
        }
        return false;
      };
  AssignMemorySpace(module.get(), options);

  HloInstruction* fusion1 =
      module->entry_computation()->GetInstructionWithName("fusion1");
  HloInstruction* fusion2 =
      module->entry_computation()->GetInstructionWithName("fusion2");
  HloInstruction* fusion3 =
      module->entry_computation()->GetInstructionWithName("fusion3");
  EXPECT_THAT(fusion2->operand(2), op::GetTupleElement(fusion1, 0));
  EXPECT_THAT(fusion2->operand(2)->shape().layout().memory_space(),
              kDefaultMemorySpace);
  EXPECT_THAT(fusion2->operand(3), op::GetTupleElement(fusion1, 1));
  EXPECT_THAT(fusion2->operand(3)->shape().layout().memory_space(),
              kDefaultMemorySpace);
  EXPECT_THAT(fusion3->operand(0), op::GetTupleElement(fusion2, 1));
  EXPECT_THAT(fusion3->operand(0)->shape().layout().memory_space(),
              kDefaultMemorySpace);
  EXPECT_THAT(fusion3->operand(1), op::GetTupleElement(fusion2, 2));
  EXPECT_THAT(fusion3->operand(1)->shape().layout().memory_space(),
              kDefaultMemorySpace);
  // Operand 0 and 1 should get alternate memory allocations and so is the
  // output {0}.
  EXPECT_THAT(fusion2->operand(0)->shape().layout().memory_space(),
              kAlternateMemorySpace);
  EXPECT_THAT(fusion2->operand(1)->shape().layout().memory_space(),
              kAlternateMemorySpace);
  EXPECT_THAT(
      ShapeUtil::GetSubshape(fusion2->shape(), {0}).layout().memory_space(),
      kAlternateMemorySpace);
}

TEST_F(MemorySpaceAssignmentTest, AsyncOpCustomFusionMultipleUsers) {
  absl::string_view hlo_string = R"(
HloModule Module, is_scheduled=true

fused_computation_start {
  param0 = f32[2,1] parameter(0)
  negate = f32[2,1] negate(param0)
  ROOT custom-call = (f32[2,1], f32[2,1], u32[], u32[]) custom-call(negate), custom_call_target="AsyncOpStart"
}

fused_computation_update1 {
  param0 = f32[2,1] parameter(0)
  param1 = f32[2,1] parameter(1)
  param2 = f32[2,1] parameter(2)
  param3 = f32[2,1] parameter(3)
  param4 = u32[] parameter(4)
  param5 = u32[] parameter(5)
  add = f32[2,1] add(param0, param1)
  negate = f32[2,1] negate(param2)
  ROOT tuple = (f32[2,1], f32[2,1], f32[2,1], f32[2,1], u32[], u32[]) tuple(add, param2, param3, negate, param4, param5)
}

fused_computation_update2 {
  param0 = f32[2,1] parameter(0)
  param1 = f32[2,1] parameter(1)
  param2 = f32[2,1] parameter(2)
  param3 = f32[2,1] parameter(3)
  param4 = u32[] parameter(4)
  param5 = u32[] parameter(5)
  add = f32[2,1] add(param0, param1)
  negate = f32[2,1] negate(param2)
  ROOT tuple = (f32[2,1], f32[2,1], f32[2,1], f32[2,1], u32[], u32[]) tuple(add, param2, param3, negate, param4, param5)
}

fused_computation_done {
  param0 = f32[2,1] parameter(0)
  param1 = f32[2,1] parameter(1)
  param2 = u32[] parameter(2)
  param3 = u32[] parameter(3)
  negate = f32[2,1] negate(param0)
  ROOT custom-call = f32[2,1] custom-call(param0, param1, negate, param2, param3), custom_call_target="AsyncOpDone"
}

ENTRY main {
  param = f32[2,1] parameter(0)
  negate1 = f32[2,1] negate(param)
  negate2 = f32[2,1] negate(negate1)
  fusion1 = (f32[2,1], f32[2,1], u32[], u32[]) fusion(negate1), kind=kCustom, output_to_operand_aliasing={{0}: (0, {})}, calls=fused_computation_start
  negate3 = f32[2,1] negate(negate2)
  negate4 = f32[2,1] negate(negate3)
  gte0 = f32[2,1] get-tuple-element(fusion1), index=0
  gte1 = f32[2,1] get-tuple-element(fusion1), index=1
  gte2 = u32[] get-tuple-element(fusion1), index=2
  gte3 = u32[] get-tuple-element(fusion1), index=3
  fusion2 = (f32[2,1], f32[2,1], f32[2,1], f32[2,1], u32[], u32[]) fusion(negate4, negate2, gte0, gte1, gte2, gte3), kind=kLoop, output_to_operand_aliasing={{1}: (2, {}), {2}: (3, {}), {3}: (3, {}), {4}: (4, {}), {5}: (5, {})}, calls=fused_computation_update1
  gte4 = f32[2,1] get-tuple-element(fusion2), index=0
  negate5 = f32[2,1] negate(gte4)
  negate10 = f32[2,1] negate(negate5)
  negate11 = f32[2,1] negate(negate10)
  negate12 = f32[2,1] negate(negate11)
  negate13 = f32[2,1] negate(negate12)
  negate14 = f32[2,1] negate(negate13)
  negate15 = f32[2,1] negate(negate14)
  negate16 = f32[2,1] negate(negate15)
  negate17 = f32[2,1] negate(negate16)
  negate18 = f32[2,1] negate(negate17)
  negate19 = f32[2,1] negate(negate18)
  fusion3 = (f32[2,1], f32[2,1], f32[2,1], f32[2,1], u32[], u32[]) fusion(negate19, negate2, gte0, gte1, gte2, gte3), kind=kLoop, output_to_operand_aliasing={{1}: (2, {}), {2}: (3, {}), {3}: (3, {}), {4}: (4, {}), {5}: (5, {})}, calls=fused_computation_update2
  gte9 = f32[2,1] get-tuple-element(fusion3), index=0
  negate6 = f32[2,1] negate(gte9)
  gte5 = f32[2,1] get-tuple-element(fusion3), index=1
  gte6 = f32[2,1] get-tuple-element(fusion3), index=2
  gte7 = u32[] get-tuple-element(fusion3), index=4
  gte8 = u32[] get-tuple-element(fusion3), index=5
  fusion4 = f32[2,1] fusion(gte5, gte6, gte7, gte8), kind=kCustom, output_to_operand_aliasing={{}: (1, {})}, calls=fused_computation_done
  ROOT add = f32[2,1] add(negate6, fusion4)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.position_requires_contiguous_allocation_fn =
      [](const HloPosition& position) {
        absl::string_view inst_name = position.instruction->name();
        if (inst_name == "fusion1" ||
            (inst_name == "fusion2" && position.index != ShapeIndex({0})) ||
            (inst_name == "fusion3" && position.index != ShapeIndex({0}))) {
          return true;
        }
        return false;
      };
  AssignMemorySpace(module.get(), options);
}

// This test seeks to test that MSA will schedule async copy operations with
// schedule_after=-1 at the very beginning of the program.
//
// The machinery for this is a little opaque from the public API, so we attempt
// to get MSA to self-assign an async copies with schedule_after=-1 by
// exploiting how the hidden algorithm works. This is brittle and subject to
// inadvertent breakage in the future.
TEST_F(MemorySpaceAssignmentTest, HoistCopyStart) {
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true

  ENTRY cross_program_prefetch {
    p0 = (f32[8,8]{1,0}, f32[8,2]{1,0}) parameter(0)
    get-tuple-element.0 = f32[8,8]{1,0} get-tuple-element(p0), index=0
    add.0 = f32[8,8]{1,0} add(get-tuple-element.0, get-tuple-element.0)
    get-tuple-element.1 = f32[8,2]{1,0} get-tuple-element(p0), index=1
    dot.0 = f32[8,2]{1,0} dot(add.0, get-tuple-element.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    negate.1 = f32[8,2]{1,0} negate(dot.0)
    negate.2 = f32[8,2]{1,0} negate(negate.1)
    negate.3 = f32[8,2]{1,0} negate(negate.2)
    negate.4 = f32[8,2]{1,0} negate(negate.3)
    negate.5 = f32[8,2]{1,0} negate(negate.4)
    negate.6 = f32[8,2]{1,0} negate(negate.5)
    negate.7 = f32[8,2]{1,0} negate(negate.6)
    negate.8 = f32[8,2]{1,0} negate(negate.7)
    ROOT dot.1 = f32[2,2]{1,0} dot(negate.8, get-tuple-element.1), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.enable_cross_program_prefetch = true;
  AssignMemorySpace(module.get(), options);

  // Ensure that get-tuple-element.1 is chosen for cross-program prefetch.
  auto cross_program_prefetches = module->CrossProgramPrefetches();
  ASSERT_EQ(cross_program_prefetches.size(), 1);
  ASSERT_EQ(cross_program_prefetches[0].parameter, 0);
  ASSERT_EQ(cross_program_prefetches[0].index, ShapeIndex({1}));

  // Check that the async copy-start for get-tuple-element.1 is hoisted
  // after MSA (get-tuple-element.1 was initially the third operation of the
  // original schedule).
  //
  // We expect the only instructions before it are declaring parameter(0) and
  // get-tuple-element.1.
  for (auto* instruction : module->schedule()
                               .sequence(module->entry_computation())
                               .instructions()) {
    auto p0 = op::Parameter(0);
    auto get_tuple_element_1 = op::GetTupleElement(p0, 1);
    auto copy_start = op::CopyStart(get_tuple_element_1);
    EXPECT_THAT(instruction, AnyOf(p0, get_tuple_element_1, copy_start));
    if (::testing::Matches(copy_start)(instruction)) {
      EXPECT_TRUE(instruction->cross_program_prefetch_index().has_value());
      break;
    }
  }
}

// This test verifies that MSA's internal map of instructions and their operands
// in alternate memory is correct in the presenece of a cross-program prefetch.
// Internally, at the end of MSA, it validates that it's internal map is
// correct. So, as long as this test allocates a cross-program-prefetch in
// alternate memory and doesn't crash, it has passed.
TEST_F(MemorySpaceAssignmentTest,
       OperandsInAlternateMemoryWithCrossProgramPrefetch) {
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true

  ENTRY cross_program_prefetch {
    p0 = (f32[8,8]{1,0}, f32[8,2]{1,0}) parameter(0)
    get-tuple-element.0 = f32[8,8]{1,0} get-tuple-element(p0), index=0
    add.0 = f32[8,8]{1,0} add(get-tuple-element.0, get-tuple-element.0)
    get-tuple-element.1 = f32[8,2]{1,0} get-tuple-element(p0), index=1
    dot.0 = f32[8,2]{1,0} dot(add.0, get-tuple-element.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    negate.1 = f32[8,2]{1,0} negate(dot.0)
    negate.2 = f32[8,2]{1,0} negate(negate.1)
    ROOT dot.1 = f32[2,2]{1,0} dot(negate.2, get-tuple-element.1), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.enable_cross_program_prefetch = true;
  AssignMemorySpace(module.get(), options);

  // Ensure that get-tuple-element.1 is chosen for cross-program prefetch.
  auto cross_program_prefetches = module->CrossProgramPrefetches();
  ASSERT_EQ(cross_program_prefetches.size(), 1);
}

TEST_F(MemorySpaceAssignmentTest, WindowPrefetch) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

%fused_computation {
  %p0 = bf16[64,8]{1,0:T(8,128)(2,1)} parameter(0)
  %p1 = bf16[64,8]{1,0:T(8,128)(2,1)} parameter(1)
  %p2 = bf16[64,8]{1,0:T(8,128)(2,1)} parameter(2)
  %add0 = bf16[64,8]{1,0:T(8,128)(2,1)} add(%p0, %p1)
  ROOT %add1 = bf16[64,8]{1,0:T(8,128)(2,1)} add(%add0, %p2)
}

entry {
  %p0 = bf16[64,8]{1,0:T(8,128)(2,1)} parameter(0)
  %p1 = bf16[64,8]{1,0:T(8,128)(2,1)} parameter(1)
  %p2 = bf16[64,8]{1,0:T(8,128)(2,1)} parameter(2)
  ROOT fusion = bf16[64,8]{1,0:T(8,128)(2,1)} fusion(bf16[64,8]{1,0:T(8,128)(2,1)} %p0, bf16[64,8]{1,0:T(8,128)(2,1)} %p1, bf16[64,8]{1,0:T(8,128)(2,1)} %p2), kind=kLoop, calls=%fused_computation
}

)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Get info about window prefetch buffers, such as which operands they
  // correspond to and their sizes.
  auto window_prefetch_detail_fn = [&](const HloInstruction* instruction) {
    WindowPrefetchDetail window_prefetch_detail;
    const HloInstruction* fusion = FindInstruction(module.get(), "fusion");
    if (instruction == fusion) {
      for (int i = 0; i < 3; ++i) {
        auto* operand = window_prefetch_detail.add_windows();
        operand->set_operand(i);
        operand->set_size(32);
      }
    }
    return window_prefetch_detail;
  };

  Options options = DefaultMemorySpaceOptions();
  options.enable_window_prefetch = true;
  options.window_prefetch_detail_fn = window_prefetch_detail_fn;
  AssignMemorySpace(module.get(), options, /*max_prefetch_interval=*/10,
                    /*min_prefetch_interval=*/0);
  const HloInstruction* fusion = FindInstruction(module.get(), "fusion");
  // The fusion instruction should have 9 operands: the 3 original operands
  // plus 3 window prefetch buffers, plus 3 sync flags.
  EXPECT_EQ(fusion->operand_count(), 9);

  // The added operands are GetTupleElements of WindowPrefetch custom calls.
  for (int i = 3; i < 9; i++) {
    EXPECT_EQ(fusion->operand(i)->opcode(), HloOpcode::kGetTupleElement);
    const HloInstruction* window_prefetch = fusion->operand(i)->operand(0);
    EXPECT_TRUE(window_prefetch->IsCustomCall("WindowPrefetch"));
  }

  VLOG(2) << "module: " << module->ToString();
}

// The test verifies the schedule of prefetching window buffers. We use
// reserved_scoped_memory_fn to reserve a large amount of memory for an
// instruction that's not too far from the fusion and verify that a valid
// schedule can still be found, which prefetches the window buffers right
// after that instruction.
TEST_F(MemorySpaceAssignmentTest, WindowPrefetchSchedule) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

%fused_computation {
  %p0 = bf16[64,8]{1,0:T(8,128)(2,1)} parameter(0)
  ROOT %neg = bf16[64,8]{1,0:T(8,128)(2,1)} negate(%p0)
}

entry {
  t0 = bf16[64,8]{1,0:T(8,128)(2,1)} parameter(0)
  t1 = bf16[64,8]{1,0:T(8,128)(2,1)} negate(t0)
  t2 = bf16[64,8]{1,0:T(8,128)(2,1)} negate(t1)
  t3 = bf16[64,8]{1,0:T(8,128)(2,1)} fusion(bf16[64,8]{1,0:T(8,128)(2,1)} t0), kind=kLoop, calls=%fused_computation
  ROOT t4 = bf16[64,8]{1,0:T(8,128)(2,1)} add(t2, t3)
}

)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Get info about window prefetch buffers, such as which operands they
  // correspond to and their sizes.
  auto window_prefetch_detail_fn = [&](const HloInstruction* instruction) {
    WindowPrefetchDetail window_prefetch_detail;
    const HloInstruction* fusion = FindInstruction(module.get(), "t3");
    if (instruction == fusion) {
      auto* window_buffer = window_prefetch_detail.add_windows();
      window_buffer->set_operand(0);
      window_buffer->set_size(32);
    }
    return window_prefetch_detail;
  };

  // Set the reserved scoped memory of the negate instruction to be 128MB. This
  // will prevent the prefetching of window buffers from starting too early.
  auto reserved_scoped_memory_fn =
      [&](const HloInstruction* instruction,
          const absl::flat_hash_set<std::pair<int, ShapeIndex>>
              operands_in_alternate_memory,
          const absl::flat_hash_set<ShapeIndex> outputs_in_alternate_memory) {
        if (instruction->name() == "t1") {
          return 128 * 1024 * 1024;
        }
        return 0;
      };

  Options options = DefaultMemorySpaceOptions();
  options.enable_window_prefetch = true;
  options.window_prefetch_detail_fn = window_prefetch_detail_fn;
  options.reserved_scoped_memory_fn = reserved_scoped_memory_fn;
  AssignMemorySpace(module.get(), options, /*max_prefetch_interval=*/10,
                    /*min_prefetch_interval=*/0);
  const HloInstruction* fusion = FindInstruction(module.get(), "t3");
  // Verify that the fusion instruction is window prefetched. If the fusion
  // instruction's operand is window prefetched, the fusion instruction should
  // have more than one operand.
  EXPECT_GT(fusion->operand_count(), 1);
}

// This test verifies that window prefetched operands are seen by the
// reserved_scoped_memory_fn. Because window prefetched operands allocates space
// in the alternate memory, which will be identified as prefetched_operands.
// Therefore they will be seen by reserved_scoped_memory_fn.
TEST_F(MemorySpaceAssignmentTest,
       WindowPrefetchedOperandsAreSeenByReservedScopedMemoryFn) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  fused_computation {
    param0 = f32[1024] parameter(0)
    param1 = f32[1024] parameter(1)
    ROOT root = f32[1024] add(param0, param1)
  }

  ENTRY Entry {
    param0 = f32[1024] parameter(0)
    param1 = f32[1024] parameter(1)
    ROOT fusion = f32[1024] fusion(param0, param1), kind=kLoop, calls=fused_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* fusion = FindInstruction(module.get(), "fusion");
  bool seen_window_prefetched_operand = false;

  Options options = DefaultMemorySpaceOptions();
  options.max_repacks = 10;
  options.repack_after_every_allocation = true;
  options.reduce_scoped_memory_limit = true;
  options.reserved_scoped_memory_fn =
      [&](const HloInstruction* instruction,
          const absl::flat_hash_set<std::pair<int, ShapeIndex>>
              operands_in_alternate_memory,
          const absl::flat_hash_set<ShapeIndex> outputs_in_alternate_memory) {
        if (instruction == fusion && !operands_in_alternate_memory.empty()) {
          seen_window_prefetched_operand = true;
        }
        return 1;
      };

  // Make sure that the alternate memory is larger than the fusion operand's
  // full size, but smaller than its span buffer size, so that it will be window
  // prefetched.
  options.enable_window_prefetch = true;
  ASSERT_LT(options.max_size_in_bytes, 1024);
  ASSERT_GT(options.max_size_in_bytes, 32);
  // This lambda instructs MSA to allocate 32 bytes in the alternate memory as
  // span buffer of the fusion instruction.
  options.window_prefetch_detail_fn =
      [&](const HloInstruction* instruction) -> WindowPrefetchDetail {
    WindowPrefetchDetail detail;
    if (instruction == fusion) {
      WindowPrefetchDetail::WindowDetail* window = detail.add_windows();
      window->set_operand(0);
      window->set_size(32);
    }
    return detail;
  };

  // Run memory space assignment and verify that window prefetched operands are
  // seen by the reserved_scoped_memory_fn.
  absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t> repack_map;
  FakeMemorySpaceAssignmentRepacker repacker =
      FakeMemorySpaceAssignmentRepacker(repack_map, nullptr);
  options.repacker = &repacker;
  AssignMemorySpace(module.get(), options, /*max_prefetch_interval=*/10,
                    /*min_prefetch_interval=*/0);
  EXPECT_TRUE(seen_window_prefetched_operand);
}

using AsynchronousCopyOrderingTest = ::testing::Test;

TEST_F(AsynchronousCopyOrderingTest, Simple) {
  // Given asynchronous copies like the following, ensure the pipelining order
  // is maintained (earlier start time must have earlier end time).
  // 3,11       +-------+         OK
  // 1,8      +------+            OK
  // 5,14         +--------+      OK
  // 7,14           +------+      OK
  // 2,16      +-------------+    Violate
  // 9,12             +--+        Violate
  // 6,17          +----------+   Violate
  // 5,13         +-------+       OK (same start as 5,14)
  // 5,14         +--------+      OK (same as 5,14)
  auto alternate_mem_space = MemorySpace::kAlternate;
  AsynchronousCopyOrdering ordering;
  EXPECT_FALSE(ordering.ViolatesOrdering(3, 11));
  ordering.AddCopy({3, 11, 1, alternate_mem_space, 0});
  EXPECT_FALSE(ordering.ViolatesOrdering(1, 8));
  ordering.AddCopy({1, 8, 1, alternate_mem_space, 1});
  EXPECT_FALSE(ordering.ViolatesOrdering(5, 14));
  ordering.AddCopy({5, 14, 1, alternate_mem_space, 2});
  EXPECT_FALSE(ordering.ViolatesOrdering(7, 14));
  ordering.AddCopy({7, 14, 1, alternate_mem_space, 3});
  EXPECT_TRUE(ordering.ViolatesOrdering(2, 16));
  EXPECT_TRUE(ordering.ViolatesOrdering(9, 12));
  EXPECT_TRUE(ordering.ViolatesOrdering(6, 17));
  EXPECT_FALSE(ordering.ViolatesOrdering(5, 13));
  ordering.AddCopy({5, 13, 1, alternate_mem_space, 4});
  EXPECT_FALSE(ordering.ViolatesOrdering(5, 14));
  ordering.AddCopy({5, 14, 1, alternate_mem_space, 5});
}

TEST_F(AsynchronousCopyOrderingTest, SameInterval) {
  auto alternate_mem_space = MemorySpace::kAlternate;
  AsynchronousCopyOrdering ordering;
  EXPECT_FALSE(ordering.ViolatesOrdering(1, 5));
  EXPECT_FALSE(ordering.ViolatesOrdering(2, 4));
  ordering.AddCopy({1, 5, 1, alternate_mem_space, 0});
  EXPECT_TRUE(ordering.ViolatesOrdering(2, 4));
  ordering.AddCopy({1, 5, 1, alternate_mem_space, 1});
  EXPECT_TRUE(ordering.ViolatesOrdering(2, 4));
  ordering.AddCopy({1, 5, 1, alternate_mem_space, 2});
  EXPECT_TRUE(ordering.ViolatesOrdering(2, 4));
  ordering.RemoveCopy({1, 5, 1, alternate_mem_space, 1});
  EXPECT_TRUE(ordering.ViolatesOrdering(2, 4));
  ordering.RemoveCopy({1, 5, 1, alternate_mem_space, 2});
  EXPECT_TRUE(ordering.ViolatesOrdering(2, 4));
  ordering.RemoveCopy({1, 5, 1, alternate_mem_space, 0});
  EXPECT_FALSE(ordering.ViolatesOrdering(2, 4));
}

using AsynchronousCopyResourceTest = ::testing::Test;

TEST_F(AsynchronousCopyResourceTest, Simple) {
  // time:      0 1 2 3 4 5 6 7 8 9
  // resource:  2 3 1 6 7 1 7 2 2 4
  // -1,3,5    +-----+                OK
  // resource:  0 0 1 6 7 1 7 2 2 4
  //  1,4,4        +---+              OK
  // resource:  0 0 0 3 7 1 7 2 2 4
  //  5,9,10               +-----+
  // resource:  0 0 0 3 7 1 0 0 1 4
  //  4,9,3              +-------+    Violate
  //  4,8,2              +-----+      OK; The 5,9 copy shifts resource to right.
  // resource:  0 0 0 3 7 0 0 0 0 4
  auto alternate_mem_space = MemorySpace::kAlternate;
  AsynchronousCopyResource resource(
      {2.0, 3.0, 1.0, 6.0, 7.0, 1.0, 7.0, 2.0, 2.0, 4.0});
  EXPECT_TRUE(resource.HasEnoughResource(-1, 3, 5.0));
  resource.AddCopy({-1, 3, 5.0, alternate_mem_space, 0});
  EXPECT_TRUE(resource.HasEnoughResource(1, 4, 4.0));
  resource.AddCopy({1, 4, 4.0, alternate_mem_space, 1});
  EXPECT_TRUE(resource.HasEnoughResource(5, 9, 10.0));
  resource.AddCopy({5, 9, 10.0, alternate_mem_space, 2});
  EXPECT_FALSE(resource.HasEnoughResource(4, 9, 3.0));
  EXPECT_TRUE(resource.HasEnoughResource(4, 8, 2.0));
  resource.AddCopy({4, 8, 2.0, alternate_mem_space, 3});
}

TEST_F(AsynchronousCopyResourceTest, Propagate) {
  // time:      0 1 2 3 4 5 6 7 8 9
  // resource:  2 2 2 2 2 2 2 2 2 2
  // 6,10,2                  +-----+   OK
  // resource:  2 2 2 2 2 2 2 0 2 2
  // 5,9,2                 +-----+     OK
  // resource:  2 2 2 2 2 2 0 0 2 2
  // 4,8,2               +-----+       OK
  // resource:  2 2 2 2 2 0 0 0 2 2
  // 3,7,2             +-----+         OK
  // resource:  2 2 2 2 0 0 0 0 2 2
  // 2,6,2           +-----+           OK
  // resource:  2 2 2 0 0 0 0 0 2 2
  // 1,5,2         +-----+             OK
  // resource:  2 2 0 0 0 0 0 0 2 2
  // 0,4,3       +-----+               OK
  // resource:  2 0 0 0 0 0 0 0 1 2
  // 0,4,3       +-----+               OK
  // resource:  2 0 0 0 0 0 0 0 0 0
  // 0,4,1       +-----+               Violate
  auto alternate_mem_space = MemorySpace::kAlternate;
  AsynchronousCopyResource resource(
      {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0});
  EXPECT_TRUE(resource.HasEnoughResource(6, 10, 2.0));
  resource.AddCopy({6, 10, 2.0, alternate_mem_space, 0});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(5, 9, 2.0));
  resource.AddCopy({5, 9, 2.0, alternate_mem_space, 1});
  EXPECT_TRUE(resource.HasEnoughResource(4, 8, 2.0));
  resource.AddCopy({4, 8, 2.0, alternate_mem_space, 2});
  EXPECT_TRUE(resource.HasEnoughResource(3, 7, 2.0));
  resource.AddCopy({3, 7, 2.0, alternate_mem_space, 3});
  EXPECT_TRUE(resource.HasEnoughResource(2, 6, 2.0));
  resource.AddCopy({2, 6, 2.0, alternate_mem_space, 4});
  EXPECT_TRUE(resource.HasEnoughResource(1, 5, 2.0));
  resource.AddCopy({1, 5, 2.0, alternate_mem_space, 5});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(0, 4, 3.0));
  resource.AddCopy({0, 4, 3.0, alternate_mem_space, 6});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(0, 4, 3.0));
  resource.AddCopy({0, 4, 3.0, alternate_mem_space, 7});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
  EXPECT_FALSE(resource.HasEnoughResource(0, 4, 1.0));
}

TEST_F(AsynchronousCopyResourceTest, CantPropagate) {
  // time:      0 1 2 3 4 5 6 7 8 9
  // resource:  2 2 2 2 2 2 2 2 2 2
  // 5,10,2                +-------+   OK
  // resource:  2 2 2 2 2 2 0 2 2 2
  // 4,7,2               +---+         OK
  // resource:  2 2 2 2 2 0 0 2 2 2
  // 4,8,4               +-----+       OK
  // resource:  2 2 2 2 2 0 0 0 0 2
  // 3,6,4             +---+           Violate
  auto alternate_mem_space = MemorySpace::kAlternate;
  AsynchronousCopyResource resource(
      {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0});
  EXPECT_TRUE(resource.HasEnoughResource(5, 10, 2.0));
  resource.AddCopy({5, 10, 2.0, alternate_mem_space, 0});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(4, 7, 2.0));
  resource.AddCopy({4, 7, 2.0, alternate_mem_space, 1});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(4, 8, 4.0));
  resource.AddCopy({4, 8, 4.0, alternate_mem_space, 2});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0}));
  EXPECT_FALSE(resource.HasEnoughResource(3, 6, 4.0));
}

TEST_F(AsynchronousCopyResourceTest, Nested) {
  // time:      0 1 2 3 4
  // resource:  2 2 2 2 2
  // 1,3,2         +-+       OK
  // resource:  2 2 0 2 2
  // 0,4,4       +-----+     Violate
  auto alternate_mem_space = MemorySpace::kAlternate;
  AsynchronousCopyResource resource({2.0, 2.0, 2.0, 2.0, 2.0});
  EXPECT_TRUE(resource.HasEnoughResource(1, 3, 2.0));
  resource.AddCopy({1, 3, 2.0, alternate_mem_space, 0});
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 2.0, 2.0}));
  EXPECT_FALSE(resource.HasEnoughResource(0, 4, 4.0));
}

TEST_F(AsynchronousCopyResourceTest, Remove) {
  // time:      0 1 2 3 4
  // resource:  2 2 2 2 2
  // add:2,5,2       +---+   OK
  // resource:  2 2 2 0 2
  // add:-1,2,3+---+         OK
  // resource:  0 1 2 0 2
  // add:0,4,4   +-----+     OK
  // resource:  0 0 0 0 1
  // rem:0,4,4   +-----+
  // resource:  0 1 2 0 2
  // rem:2,5,2       +---+
  // resource:  0 1 2 2 2
  // rem:-1,2,3+---+
  // resource:  2 2 2 2 2
  auto alternate_mem_space = MemorySpace::kAlternate;
  AsynchronousCopyResource resource({2.0, 2.0, 2.0, 2.0, 2.0});
  AsynchronousCopy copy1{2, 5, 2.0, alternate_mem_space, 0};
  AsynchronousCopy copy2{-1, 2, 3.0, alternate_mem_space, 1};
  AsynchronousCopy copy3{0, 4, 4.0, alternate_mem_space, 2};
  EXPECT_TRUE(resource.HasEnoughResource(2, 5, 2.0));
  resource.AddCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 2.0, 0.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(-1, 2, 3.0));
  resource.AddCopy(copy2);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({0.0, 1.0, 2.0, 0.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(0, 4, 4.0));
  resource.AddCopy(copy3);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({0.0, 0.0, 0.0, 0.0, 1.0}));
  resource.RemoveCopy(copy3);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({0.0, 1.0, 2.0, 0.0, 2.0}));
  resource.RemoveCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({0.0, 1.0, 2.0, 2.0, 2.0}));
  resource.RemoveCopy(copy2);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0}));
}

TEST_F(AsynchronousCopyResourceTest, NestedRemove) {
  // time:      0 1 2 3 4
  // resource:  2 2 2 2 2
  // add:1,3,2     +-+       OK
  // resource:  2 2 0 2 2
  // add:0,4,4   +-----+     Violate
  // rem:1,3,2     +-+
  // resource:  2 2 2 2 2
  // add:0,4,4   +-----+     OK
  // resource:  2 0 0 2 2
  // add:1,3,2     +-+       Violate
  // rem:0,4,4   +-----+
  // resource:  2 2 2 2 2
  // add:1,3,2     +-+       OK
  // resource:  2 2 0 2 2
  auto alternate_mem_space = MemorySpace::kAlternate;
  AsynchronousCopyResource resource({2.0, 2.0, 2.0, 2.0, 2.0});
  AsynchronousCopy copy1{1, 3, 2.0, alternate_mem_space, 0};
  AsynchronousCopy copy2{0, 4, 4.0, alternate_mem_space, 1};
  EXPECT_TRUE(resource.HasEnoughResource(1, 3, 2.0));
  resource.AddCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 2.0, 2.0}));
  EXPECT_FALSE(resource.HasEnoughResource(0, 4, 4.0));
  resource.RemoveCopy(copy1);
  auto current_resources = resource.GetCurrentResources();
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(0, 4, 4.0));
  resource.AddCopy(copy2);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 0.0, 0.0, 2.0, 2.0}));
  EXPECT_FALSE(resource.HasEnoughResource(1, 3, 2.0));
  resource.RemoveCopy(copy2);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(1, 3, 2.0));
}

TEST_F(AsynchronousCopyResourceTest, PropagateRemove) {
  // time:      0 1 2 3 4 5 6 7 8 9
  // resource:  2 2 2 2 2 2 2 2 2 2
  // add:6,10,2              +-----+   OK
  // resource:  2 2 2 2 2 2 2 0 2 2
  // add:5,9,2             +-----+     OK
  // resource:  2 2 2 2 2 2 0 0 2 2
  // add:4,8,2           +-----+       OK
  // resource:  2 2 2 2 2 0 0 0 2 2
  // add:3,7,2         +-----+         OK
  // resource:  2 2 2 2 0 0 0 0 2 2
  // add:2,6,2       +-----+           OK
  // resource:  2 2 2 0 0 0 0 0 2 2
  // add:1,5,2     +-----+             OK
  // resource:  2 2 0 0 0 0 0 0 2 2
  // add:0,4,3   +-----+               OK
  // resource:  2 0 0 0 0 0 0 0 1 2
  // add:0,5,3   +-------+             OK
  // resource:  2 0 0 0 0 0 0 0 0 0
  // rem:0,5,3   +-------+
  // resource:  2 0 0 0 0 0 0 0 1 2
  // rem:0,4,3   +-----+
  // resource:  2 2 0 0 0 0 0 0 2 2
  auto alternate_mem_space = MemorySpace::kAlternate;
  AsynchronousCopyResource resource(
      {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0});
  EXPECT_TRUE(resource.HasEnoughResource(6, 10, 2.0));
  resource.AddCopy({6, 10, 2.0, alternate_mem_space, 0});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(5, 9, 2.0));
  resource.AddCopy({5, 9, 2.0, alternate_mem_space, 1});
  EXPECT_TRUE(resource.HasEnoughResource(4, 8, 2.0));
  resource.AddCopy({4, 8, 2.0, alternate_mem_space, 2});
  EXPECT_TRUE(resource.HasEnoughResource(3, 7, 2.0));
  resource.AddCopy({3, 7, 2.0, alternate_mem_space, 3});
  EXPECT_TRUE(resource.HasEnoughResource(2, 6, 2.0));
  resource.AddCopy({2, 6, 2.0, alternate_mem_space, 4});
  EXPECT_TRUE(resource.HasEnoughResource(1, 5, 2.0));
  resource.AddCopy({1, 5, 2.0, alternate_mem_space, 5});
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0}));
  AsynchronousCopy copy1{0, 4, 3.0, alternate_mem_space, 6};
  EXPECT_TRUE(resource.HasEnoughResource(0, 4, 3.0));
  resource.AddCopy(copy1);
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(0, 5, 3.0));
  AsynchronousCopy copy2{0, 5, 3.0, alternate_mem_space, 7};
  resource.AddCopy(copy2);
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
  resource.RemoveCopy(copy2);
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0}));
  resource.RemoveCopy(copy1);
  EXPECT_EQ(
      resource.GetCurrentResources(),
      std::vector<float>({2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0}));
}

TEST_F(AsynchronousCopyResourceTest, StartAtZeroAndRemove) {
  // time:      0 1 2 3 4
  // resource:  0 0 1 1 2
  // add:0,4,2   +-----+     OK
  // resource:  0 0 0 0 2
  // rem:0,4,2   +-----+
  // resource:  0 0 1 1 2
  // add:0,4,2   +-----+     OK
  // resource:  0 0 0 0 2
  auto alternate_mem_space = MemorySpace::kAlternate;
  AsynchronousCopyResource resource({0.0, 0.0, 1.0, 1.0, 2.0});
  AsynchronousCopy copy1{0, 4, 2.0, alternate_mem_space, 0};
  EXPECT_TRUE(resource.HasEnoughResource(0, 4, 2.0));
  resource.AddCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({0.0, 0.0, 0.0, 0.0, 2.0}));
  resource.RemoveCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({0.0, 0.0, 1.0, 1.0, 2.0}));
  resource.AddCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({0.0, 0.0, 0.0, 0.0, 2.0}));
}

// Below test only works when the resource values are scaled to int64 to avoid
// floating point precision issues.
TEST_F(AsynchronousCopyResourceTest, ConsumeResourceScaledIntegerResource) {
  auto alternate_mem_space = MemorySpace::kAlternate;
  AsynchronousCopyResource resource(
      {5.71429e-10, 8.71333e-09, 8.71333e-09, 1.74267e-08, 1.74267e-08});
  AsynchronousCopy copy1{0, 2, 8.71333e-09, alternate_mem_space, 0};
  EXPECT_TRUE(resource.HasEnoughResource(0, 2, 8.71333e-09));
  resource.AddCopy(copy1);

  AsynchronousCopy copy2{0, 3, 4.35667e-09, alternate_mem_space, 1};
  EXPECT_TRUE(resource.HasEnoughResource(0, 3, 4.35667e-09));
  resource.AddCopy(copy2);

  AsynchronousCopy copy3{2, 4, 4.35667e-09, alternate_mem_space, 2};
  EXPECT_TRUE(resource.HasEnoughResource(2, 4, 4.35667e-09));
  resource.AddCopy(copy3);

  // This call to RemoveCopy should not cause a crash.
  resource.RemoveCopy(copy1);
}

TEST_F(AsynchronousCopyResourceTest, OutOfOrderRemovalSameStartTime) {
  // time:      0 1 2 3 4
  // resource:  2 2 2 2 2
  // add:1,3,1     +-+       OK
  // resource:  2 2 1 2 2
  // add:1,4,2     +---+     OK
  // resource:  2 2 0 1 2
  // rem:1,3,1     +-+
  // resource:  2 2 0 2 2
  // add:1,5,1     +-----+   OK
  // resource:  2 2 0 1 2
  // add:1,5,1     +-----+   OK
  // resource:  2 2 0 0 2
  // add:1,5,1     +-----+   OK
  // resource:  2 2 0 0 1
  // add:1,5,1     +-----+   OK
  // resource:  2 2 0 0 0
  // add:1,5,1     +-----+   Violate
  // rem:1,4,2     +---+
  // resource:  2 2 0 0 2
  // rem:1,5,1     +-----+
  // resource:  2 2 0 1 2
  // rem:1,5,1     +-----+
  // resource:  2 2 0 2 2
  // rem:1,5,1     +-----+
  // resource:  2 2 1 2 2
  // rem:1,5,1     +-----+
  // resource:  2 2 2 2 2
  auto alternate_mem_space = MemorySpace::kAlternate;
  AsynchronousCopyResource resource({2.0, 2.0, 2.0, 2.0, 2.0});
  AsynchronousCopy copy1{1, 3, 1.0, alternate_mem_space, 0};
  AsynchronousCopy copy2{1, 4, 2.0, alternate_mem_space, 1};
  EXPECT_TRUE(resource.HasEnoughResource(1, 3, 1.0));
  resource.AddCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 1.0, 2.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(1, 4, 2.0));
  resource.AddCopy(copy2);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 1.0, 2.0}));
  resource.RemoveCopy(copy1);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 2.0, 2.0}));

  AsynchronousCopy copy3{1, 5, 1.0, alternate_mem_space, 2};
  AsynchronousCopy copy4{1, 5, 1.0, alternate_mem_space, 3};
  AsynchronousCopy copy5{1, 5, 1.0, alternate_mem_space, 4};
  AsynchronousCopy copy6{1, 5, 1.0, alternate_mem_space, 5};
  EXPECT_TRUE(resource.HasEnoughResource(1, 5, 1.0));
  resource.AddCopy(copy3);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 1.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(1, 5, 1.0));
  resource.AddCopy(copy4);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 0.0, 2.0}));
  EXPECT_TRUE(resource.HasEnoughResource(1, 5, 1.0));
  resource.AddCopy(copy5);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 0.0, 1.0}));
  EXPECT_TRUE(resource.HasEnoughResource(1, 5, 1.0));
  resource.AddCopy(copy6);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 0.0, 0.0}));
  EXPECT_FALSE(resource.HasEnoughResource(1, 5, 1.0));

  resource.RemoveCopy(copy2);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 0.0, 2.0}));
  resource.RemoveCopy(copy3);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 1.0, 2.0}));
  resource.RemoveCopy(copy4);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 0.0, 2.0, 2.0}));
  resource.RemoveCopy(copy5);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 1.0, 2.0, 2.0}));
  resource.RemoveCopy(copy6);
  EXPECT_EQ(resource.GetCurrentResources(),
            std::vector<float>({2.0, 2.0, 2.0, 2.0, 2.0}));
}

TEST_F(AsynchronousCopyResourceTest, HasEnoughResourceMultiCheckSuccess) {
  // time:      0 1 2 3 4 5 6 7 8 9
  // resource:  2 1 3 6 7 3 7 2 2 4
  // -1,3,5    +-----+                OK
  // resource:  0 0 1 6 7 3 7 2 2 4
  //  1,10,4       +---------------+  OK
  // resource:  0 0 0 3 7 3 7 2 2 4
  //  0,6,4    +-----------+
  //  4,6,3              +-+          2 copies OK; The 1,10 copy shifts.
  // resource:  0 0 0 0 6 0 7 2 2 4
  auto alternate_mem_space = MemorySpace::kAlternate;
  AsynchronousCopyResource resource(
      {2.0, 1.0, 3.0, 6.0, 7.0, 3.0, 7.0, 2.0, 2.0, 4.0});
  EXPECT_TRUE(resource.HasEnoughResource(-1, 3, 5.0));
  resource.AddCopy({-1, 3, 5.0, alternate_mem_space, 0});
  EXPECT_TRUE(resource.HasEnoughResource(1, 10, 4.0));
  resource.AddCopy({1, 10, 4.0, alternate_mem_space, 1});

  LOG(INFO) << "AsynchronousCopyResource after setup:\n"
            << resource.Dump(0, 10, alternate_mem_space);

  // We run the check in a loop to demonstrate that it is not modifying the
  // underlying data structures.
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(
        resource.HasEnoughResourceMultiCheck({{0, 6, 4.0}, {4, 6, 3.0}}));
  }
}

TEST_F(AsynchronousCopyResourceTest, HasEnoughResourceMultiCheckFailure) {
  // time:      0 1 2 3 4 5 6 7 8 9
  // resource:  2 1 3 6 7 3 7 2 2 4
  // -1,3,5    +-----+                OK
  // resource:  0 0 1 6 7 3 7 2 2 4
  //  1,10,4       +---------------+  OK
  // resource:  0 0 0 3 7 3 7 2 2 4
  //  0,6,4    +-----------+
  //  4,6,4              +-+          Not-OK
  auto alternate_mem_space = MemorySpace::kAlternate;
  AsynchronousCopyResource resource(
      {2.0, 1.0, 3.0, 6.0, 7.0, 3.0, 7.0, 2.0, 2.0, 4.0});
  EXPECT_TRUE(resource.HasEnoughResource(-1, 3, 5.0));
  resource.AddCopy({-1, 3, 5.0, alternate_mem_space, 0});
  EXPECT_TRUE(resource.HasEnoughResource(1, 10, 4.0));
  resource.AddCopy({1, 10, 4.0, alternate_mem_space, 1});

  LOG(INFO) << "AsynchronousCopyResource after setup:\n"
            << resource.Dump(0, 10, alternate_mem_space);

  EXPECT_FALSE(
      resource.HasEnoughResourceMultiCheck({{0, 6, 4.0}, {4, 6, 4.0}}));
}

TEST_F(AsynchronousCopyResourceTest,
       HasEnoughResourceMultiCheckRegressionTest) {
  auto alternate_mem_space = MemorySpace::kAlternate;
  AsynchronousCopyResource resource({/*0:*/ 24.0f,
                                     /*1:*/ 0.0f,
                                     /*2:*/ 6.0f,
                                     /*3:*/ 411.0f,
                                     /*4:*/ 3479.0f,
                                     /*5:*/ 0.0f,
                                     /*6:*/ 0.0f,
                                     /*7:*/ 1537.0f,
                                     /*8:*/ 3095.0f,
                                     /*9:*/ 0.0f,
                                     /*10:*/ 26.7f});
  AsynchronousCopy copy1({1, 8, 170.8f, alternate_mem_space, 1});
  AsynchronousCopy copy2({2, 8, 170.8f, alternate_mem_space, 2});
  resource.AddCopy(copy1);
  resource.AddCopy(copy2);

  LOG(INFO) << "AsynchronousCopyResource after setup:\n"
            << resource.Dump(0, 11, alternate_mem_space);
  // Under the  current AsynchronousCopyResource implementation, this
  // HasEnoughResource check fails. Although, other designs could rearrange
  // resources in a manner that fits the check.
  EXPECT_FALSE(
      resource.HasEnoughResourceMultiCheck({{0, 4, 170.8}, {1, 4, 170.8}}));
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "lhs"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, rhs_shape, "rhs"));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {lhs, rhs, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  EXPECT_EQ(cross_program_prefetches[0].parameter, 1);
  EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({}));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Parameter(0),
                      op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                    op::Parameter(1))));
}

TEST_F(MemorySpaceAssignmentTest, MultiCrossProgramPrefetchTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kFirstOutput = 4;
  constexpr int kSecondOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto first_weight_shape = ShapeUtil::MakeShape(F32, {kFeature, kFirstOutput});
  auto second_weight_shape =
      ShapeUtil::MakeShape(F32, {kFirstOutput, kSecondOutput});
  auto intermediate_shape = ShapeUtil::MakeShape(F32, {kBatch, kFirstOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kSecondOutput});
  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "lhs"));
  HloInstruction* first_weight = builder.AddInstruction(
      HloInstruction::CreateParameter(1, first_weight_shape, "first_weight"));
  HloInstruction* second_weight = builder.AddInstruction(
      HloInstruction::CreateParameter(2, second_weight_shape, "second_weight"));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto first_dot = builder.AddInstruction(
      HloInstruction::CreateDot(intermediate_shape, lhs, first_weight,
                                dot_dnums, DefaultPrecisionConfig(2)));

  auto second_dot = builder.AddInstruction(
      HloInstruction::CreateDot(result_shape, first_dot, second_weight,
                                dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(
      computation, {lhs, first_weight, second_weight, first_dot, second_dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options = DefaultMemorySpaceOptions();
  options.max_cross_program_prefetches = -1;
  options.max_size_in_bytes = 256;
  options.alignment_in_bytes = 8;
  options.verify = true;
  AssignMemorySpace(module.get(), options);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 2);
  EXPECT_EQ(cross_program_prefetches[0].parameter, 1);
  EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({}));
  EXPECT_EQ(cross_program_prefetches[1].parameter, 2);
  EXPECT_EQ(cross_program_prefetches[1].index, ShapeIndex({}));

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Dot(op::Dot(op::Parameter(0),
                      op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                    op::Parameter(1))),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::Parameter(2))));
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchTupleTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  auto tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));

  auto lhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(lhs_shape, param, 0));
  auto rhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(rhs_shape, param, 1));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {param, lhs, rhs, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  EXPECT_EQ(cross_program_prefetches[0].parameter, 0);
  EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({1}));
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchBitcastTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kOutput, kFeature});
  auto bitcast_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "lhs"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, rhs_shape, "rhs"));

  auto bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(bitcast_shape, rhs));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, bitcast, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {lhs, rhs, bitcast, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  EXPECT_EQ(cross_program_prefetches[0].parameter, 1);
  EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({}));
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchBitcastTupleTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kOutput, kFeature});
  auto bitcast_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  auto tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));

  auto lhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(lhs_shape, param, 0));
  auto rhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(rhs_shape, param, 1));

  auto bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(bitcast_shape, rhs));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, bitcast, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {param, lhs, rhs, bitcast, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  EXPECT_EQ(cross_program_prefetches[0].parameter, 0);
  EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({1}));
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchNestedTupleTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  auto tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});
  auto tuple_tuple_shape = ShapeUtil::MakeTupleShape({tuple_shape});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_tuple_shape, "p0"));

  auto gte = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(tuple_shape, param, 0));

  auto lhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(lhs_shape, gte, 0));
  auto rhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(rhs_shape, gte, 1));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {param, gte, lhs, rhs, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchUnusedParamTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, rhs_shape, "p0"));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {param});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchTooBigTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 8;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "lhs"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, rhs_shape, "rhs"));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {lhs, rhs, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchTooBigTupleTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 8;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  auto tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));

  auto lhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(lhs_shape, param, 0));
  auto rhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(rhs_shape, param, 1));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {param, lhs, rhs, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchFusionTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 2;
  constexpr int kFeature = 2;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});

  auto module = CreateNewVerifiedModule();
  HloComputation::Builder fusion_builder("fusion");
  {
    HloInstruction* lhs = fusion_builder.AddInstruction(
        HloInstruction::CreateParameter(0, lhs_shape, "lhs"));
    HloInstruction* rhs = fusion_builder.AddInstruction(
        HloInstruction::CreateParameter(1, rhs_shape, "rhs"));
    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(1);
    dot_dnums.add_rhs_contracting_dimensions(0);
    auto dot = fusion_builder.AddInstruction(HloInstruction::CreateDot(
        result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));
    (void)dot;
  }
  HloComputation* fusion_computation =
      module->AddEmbeddedComputation(fusion_builder.Build());

  auto activations = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{0.0, 1.0}, {2.0, 3.0}})));
  auto weights = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{0.0, 1.0}, {2.0, 3.0}})));
  HloInstruction* fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      result_shape, HloInstruction::FusionKind::kCustom, {activations, weights},
      fusion_computation));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {activations, weights, fusion});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchFusionTupleTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 2;
  constexpr int kFeature = 2;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShape(F32, {kFeature, kOutput});
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  auto tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});

  auto module = CreateNewVerifiedModule();
  HloComputation::Builder fusion_builder("fusion");
  {
    HloInstruction* param = fusion_builder.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "p0"));
    auto lhs = fusion_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(lhs_shape, param, 0));
    auto rhs = fusion_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(rhs_shape, param, 1));
    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(1);
    dot_dnums.add_rhs_contracting_dimensions(0);
    auto dot = fusion_builder.AddInstruction(HloInstruction::CreateDot(
        result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));
    (void)dot;
  }
  HloComputation* fusion_computation =
      module->AddEmbeddedComputation(fusion_builder.Build());

  auto activations = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{0.0, 1.0}, {2.0, 3.0}})));
  auto weights = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{0.0, 1.0}, {2.0, 3.0}})));
  HloInstruction* tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({activations, weights}));
  HloInstruction* fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      result_shape, HloInstruction::FusionKind::kCustom, {tuple},
      fusion_computation));

  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {activations, weights, tuple, fusion});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchPinnedTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {kFeature, kOutput},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "lhs"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, rhs_shape, "rhs"));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {lhs, rhs, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options = DefaultMemorySpaceOptions();
  options.is_allowed_in_alternate_mem_fn = [](const HloValue& value) {
    return true;
  };
  std::unique_ptr<PresetAssignments> preset_assignments =
      AssignMemorySpace(module.get(), options);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_GT(cross_program_prefetches.size(), 0);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchPinnedTupleTest) {
  HloComputation::Builder builder(TestName());

  constexpr int kBatch = 8;
  constexpr int kFeature = 8;
  constexpr int kOutput = 2;

  auto lhs_shape = ShapeUtil::MakeShape(F32, {kBatch, kFeature});
  auto rhs_shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {kFeature, kOutput},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{},
      /*tail_padding_alignment_in_elements=*/1, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
  auto result_shape = ShapeUtil::MakeShape(F32, {kBatch, kOutput});
  auto tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape, rhs_shape});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));

  auto lhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(lhs_shape, param, 0));
  auto rhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(rhs_shape, param, 1));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      result_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {param, lhs, rhs, dot});
  TF_CHECK_OK(module->set_schedule(schedule));

  Options options = DefaultMemorySpaceOptions();
  options.is_allowed_in_alternate_mem_fn = [](const HloValue& value) {
    return true;
  };
  std::unique_ptr<PresetAssignments> preset_assignments =
      AssignMemorySpace(module.get(), options);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_GT(cross_program_prefetches.size(), 0);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramRootDupMayAlias) {
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true, input_output_alias={ {}: (0, {}, may-alias) }
    ENTRY CrossProgramPrefetch {
      c0 = s32[1,2] constant({{77, 77}})
      c1 = s32[] constant(0)
      p0 = s32[2,2] parameter(0)
      ROOT dup = s32[2,2] dynamic-update-slice(s32[2,2] p0, s32[1,2] c0, s32[] c1, s32[] c1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(
      module.get(), DefaultMemorySpaceOptions(),
      /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
  EXPECT_THAT(FindInstruction(module.get(), "dup")->operand(0),
              op::Parameter(0));
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramRootDusFusionMayAlias) {
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true, input_output_alias={ {}: (0, {}, may-alias) }
    fused_computation {
      fused_p0 = s32[2,2] parameter(0)
      fused_p1 = s32[1,2] parameter(1)
      fused_p2 = s32[] parameter(2)
      fused_p3 = s32[] parameter(3)
      ROOT dus = s32[2,2] dynamic-update-slice(fused_p0, fused_p1, fused_p2, fused_p3)
    }

    ENTRY CrossProgramPrefetch {
      p0 = s32[2,2] parameter(0)
      c0 = s32[1,2] constant({{77, 77}})
      c1 = s32[] constant(0)
      bitcast1 = s32[2,2] bitcast(p0)
      ROOT fusion = s32[2,2] fusion(bitcast1, c0, c1, c1), kind=kLoop, calls=fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(
      module.get(), DefaultMemorySpaceOptions(),
      /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramRootDup) {
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true
    ENTRY CrossProgramPrefetch {
      c0 = s32[1,2] constant({{77, 77}})
      c1 = s32[] constant(0)
      p0 = s32[2,2] parameter(0)
      ROOT dup = s32[2,2] dynamic-update-slice(s32[2,2] p0, s32[1,2] c0, s32[] c1, s32[] c1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(
      module.get(), DefaultMemorySpaceOptions(),
      /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
  EXPECT_THAT(FindInstruction(module.get(), "dup")->operand(0),
              op::Parameter(0));
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramRootDupDot) {
  // Cross program prefetch since the parameter and the root don't alias.
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true
    ENTRY CrossProgramPrefetch {
      c0 = s32[1,2] constant({{77, 77}})
      c1 = s32[] constant(0)
      p0 = s32[2,2] parameter(0)
      p1 = s32[2,2] parameter(1)
      dup = s32[2,2] dynamic-update-slice(s32[2,2] p0, s32[1,2] c0, s32[] c1, s32[] c1)
      ROOT dot = s32[2,2] dot(p1, dup), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(
      module.get(), DefaultMemorySpaceOptions(),
      /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  EXPECT_THAT(FindInstruction(module.get(), "dup")->operand(0),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::Parameter(0)));
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramRootDotMayAlias) {
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true, input_output_alias={ {}: (0, {}, may-alias) }
    ENTRY CrossProgramPrefetch {
      p0 = s32[2,2] parameter(0)
      p1 = s32[2,2] parameter(1)
      ROOT dot = s32[2,2] dot(p1, p0), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(
      module.get(), DefaultMemorySpaceOptions(),
      /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
  EXPECT_THAT(FindInstruction(module.get(), "dot")->operand(1),
              op::Parameter(0));
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramRootLiveOutBug) {
  // Input-output aliased buffers should not be cross-program prefetched since
  // the update on the buffer will not be reflected on the next program
  // execution (the data in the alternate memory would be stale).
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true, input_output_alias={ {0}: (0, {}, may-alias) }
    fused_computation {
      p0 = s32[2,2] parameter(0)
      p1 = s32[2,2] parameter(1)
      slice = s32[1,2] slice(p1), slice={[0:1], [0:2]}
      c1 = s32[] constant(0)
      ROOT dus = s32[2,2] dynamic-update-slice(s32[2,2] p0, s32[1,2] slice, s32[] c1, s32[] c1)
    }

    ENTRY CrossProgramPrefetch {
      p0 = s32[2,2] parameter(0)
      p1 = s32[2,2] parameter(1)
      dot = s32[2,2] dot(p1, p0), lhs_contracting_dims={0}, rhs_contracting_dims={0}
      fusion = s32[2,2] fusion(p0, dot), kind=kLoop, calls=fused_computation
      ROOT root = (s32[2,2], s32[2,2]) tuple(fusion, dot)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(
      module.get(), DefaultMemorySpaceOptions(),
      /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramRootParameter) {
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true
    ENTRY CrossProgramPrefetch {
      p0 = s32[2,2] parameter(0)
      ROOT bitcast = u32[2,2] bitcast(p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(
      module.get(), DefaultMemorySpaceOptions(),
      /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 0);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchNoReuse) {
  // This test is for checking if the cross-program-prefetched buffer is freed
  // after its last use and there is an end-of-program prefetch.
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true

  ENTRY CrossProgramPrefetch {
    p0 = f32[8,8]{1,0} parameter(0)
    p1 = f32[8,2]{1,0} parameter(1)
    dot = f32[8,2]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    negate.1 = f32[8,2]{1,0} negate(dot)
    negate.2 = f32[8,2]{1,0} negate(negate.1)
    negate.3 = f32[8,2]{1,0} negate(negate.2)
    negate.4 = f32[8,2]{1,0} negate(negate.3)
    negate.5 = f32[8,2]{1,0} negate(negate.4)
    negate.6 = f32[8,2]{1,0} negate(negate.5)
    negate.7 = f32[8,2]{1,0} negate(negate.6)
    negate.8 = f32[8,2]{1,0} negate(negate.7)
    ROOT negate.9 = f32[8,2]{1,0} negate(negate.8)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto options = DefaultMemorySpaceOptions();
  // Enough space to fit the cross-program prefetch for both p0 and p1.
  options.max_size_in_bytes = 512;
  auto preset_assignments = AssignMemorySpace(module.get(), options,
                                              /*max_prefetch_interval=*/5,
                                              /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  EXPECT_EQ(cross_program_prefetches[0].parameter, 1);
  EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis,
      HloDataflowAnalysis::Run(*module));
  LOG(ERROR) << "module: " << module->ToString();
  const HloValue& cross_program_prefetched_value =
      dataflow_analysis->GetValueDefinedAt(
          module->entry_computation()->parameter_instruction(1), {});
  // Expect that there are two prefetches that use this value, one is the
  // cross-program prefetch, the other is the end-of-program prefetch.
  auto is_cross_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_cross_program_prefetch),
            1);
  auto is_end_of_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           !use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_end_of_program_prefetch),
            1);
  // Also verify that the copy-done for the end-of-program prefetch is the last
  // instruction in schedule.
  const HloInstruction* last_instruction =
      module->schedule()
          .sequence(module->entry_computation())
          .instructions()[module->entry_computation()->instruction_count() - 1];
  EXPECT_THAT(last_instruction, op::CopyDone());
  EXPECT_NE(last_instruction, module->entry_computation()->root_instruction());
  // Cross program prefetch would use offset 0 because that's the first
  // assignment. Since we are freeing the cross-program prefetch buffer, we
  // would also expect to see some of the intermediate computations (one of the
  // negate ops) to also get 0 offset allocations.
  bool has_zero_offset_allocations = false;
  for (auto pos_and_chunk : preset_assignments->chunks()) {
    if (pos_and_chunk.first.instruction->opcode() == HloOpcode::kNegate &&
        pos_and_chunk.second.offset == 0) {
      has_zero_offset_allocations = true;
    }
  }
  EXPECT_TRUE(has_zero_offset_allocations);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchWithOverrideNoReuse) {
  // This test is same as above, but with an override to cross-program prefetch
  // parameter0 as opposed to p0 and limiting the max alternate memory
  // size to 256 bytes so that both p0 and p1 cannot be assigned to alternate
  // memory and priority is given to p0.
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true

  ENTRY CrossProgramPrefetch {
    p0 = f32[8,8]{1,0} parameter(0)
    p1 = f32[8,2]{1,0} parameter(1)
    dot = f32[8,2]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    negate.1 = f32[8,2]{1,0} negate(dot)
    negate.2 = f32[8,2]{1,0} negate(negate.1)
    negate.3 = f32[8,2]{1,0} negate(negate.2)
    negate.4 = f32[8,2]{1,0} negate(negate.3)
    negate.5 = f32[8,2]{1,0} negate(negate.4)
    negate.6 = f32[8,2]{1,0} negate(negate.5)
    negate.7 = f32[8,2]{1,0} negate(negate.6)
    negate.8 = f32[8,2]{1,0} negate(negate.7)
    ROOT negate.9 = f32[8,2]{1,0} negate(negate.8)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto options = DefaultMemorySpaceOptions();
  const std::string text_proto = R"pb(
    overrides {
      hlo_position_matcher {
        instruction_name_regex: "p(.*)"
        instruction_regex: ".*parameter\\(0\\).*"
      }
      override_options { assign_first: true }
      apply_to_cross_program_prefetches: true
    })pb";
  TF_ASSERT_OK_AND_ASSIGN(options.msa_sort_order_overrides,
                          ParseTextProto<MsaSortOrderOverrides>(text_proto));
  options.max_size_in_bytes = 256;
  auto preset_assignments = AssignMemorySpace(module.get(), options,
                                              /*max_prefetch_interval=*/5,
                                              /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  EXPECT_EQ(cross_program_prefetches[0].parameter, 0);
  EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis,
      HloDataflowAnalysis::Run(*module));
  LOG(ERROR) << "module: " << module->ToString();
  const HloValue& cross_program_prefetched_value =
      dataflow_analysis->GetValueDefinedAt(
          module->entry_computation()->parameter_instruction(0), {});
  // Expect that there are two prefetches that use this value, one is the
  // cross-program prefetch, the other is the end-of-program prefetch.
  auto is_cross_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_cross_program_prefetch),
            1);
  auto is_end_of_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           !use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_end_of_program_prefetch),
            1);
  // Also verify that the copy-done for the end-of-program prefetch is the last
  // instruction in schedule.
  const HloInstruction* last_instruction =
      module->schedule()
          .sequence(module->entry_computation())
          .instructions()[module->entry_computation()->instruction_count() - 1];
  EXPECT_THAT(last_instruction, op::CopyDone());
  EXPECT_NE(last_instruction, module->entry_computation()->root_instruction());
  // Cross program prefetch would use offset 0 because that's the first
  // assignment. Since we are freeing the cross-program prefetch buffer, we
  // would also expect to see some of the intermediate computations (one of the
  // negate ops) to also get 0 offset allocations.
  bool has_zero_offset_allocations = false;
  for (auto pos_and_chunk : preset_assignments->chunks()) {
    if (pos_and_chunk.first.instruction->opcode() == HloOpcode::kNegate &&
        pos_and_chunk.second.offset == 0) {
      has_zero_offset_allocations = true;
    }
  }
  EXPECT_TRUE(has_zero_offset_allocations);
}

TEST_F(MemorySpaceAssignmentTest, UserAnnotatedCrossProgramPrefetchNoReuse) {
  // This test is same as above, but with user directive to cross-program
  // prefetch parameter0 as opposed to p0 and limiting the max alternate memory
  // size to 256 bytes so that both p0 and p1 cannot be assigned to alternate
  // memory and priority is given to p0.
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true, entry_computation_layout={(f32[8,8]{1,0:S(1)}, f32[8,2]{1,0})->f32[8,2]{1,0}}

  ENTRY CrossProgramPrefetch {
    p0 = f32[8,8]{1,0:S(1)} parameter(0)
    p1 = f32[8,2]{1,0} parameter(1)
    dot = f32[8,2]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    negate.1 = f32[8,2]{1,0} negate(dot)
    negate.2 = f32[8,2]{1,0} negate(negate.1)
    negate.3 = f32[8,2]{1,0} negate(negate.2)
    negate.4 = f32[8,2]{1,0} negate(negate.3)
    negate.5 = f32[8,2]{1,0} negate(negate.4)
    negate.6 = f32[8,2]{1,0} negate(negate.5)
    negate.7 = f32[8,2]{1,0} negate(negate.6)
    negate.8 = f32[8,2]{1,0} negate(negate.7)
    ROOT negate.9 = f32[8,2]{1,0} negate(negate.8)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 256;
  auto preset_assignments = AssignMemorySpace(module.get(), options,
                                              /*max_prefetch_interval=*/5,
                                              /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  EXPECT_EQ(cross_program_prefetches[0].parameter, 0);
  EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis,
      HloDataflowAnalysis::Run(*module));
  LOG(ERROR) << "module: " << module->ToString();
  const HloValue& cross_program_prefetched_value =
      dataflow_analysis->GetValueDefinedAt(
          module->entry_computation()->parameter_instruction(0), {});
  // Expect that there are two prefetches that use this value, one is the
  // cross-program prefetch, the other is the end-of-program prefetch.
  auto is_cross_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_cross_program_prefetch),
            1);
  auto is_end_of_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           !use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_end_of_program_prefetch),
            1);
  // Also verify that the copy-done for the end-of-program prefetch is the last
  // instruction in schedule.
  const HloInstruction* last_instruction =
      module->schedule()
          .sequence(module->entry_computation())
          .instructions()[module->entry_computation()->instruction_count() - 1];
  EXPECT_THAT(last_instruction, op::CopyDone());
  EXPECT_NE(last_instruction, module->entry_computation()->root_instruction());
  // Cross program prefetch would use offset 0 because that's the first
  // assignment. Since we are freeing the cross-program prefetch buffer, we
  // would also expect to see some of the intermediate computations (one of the
  // negate ops) to also get 0 offset allocations.
  bool has_zero_offset_allocations = false;
  for (auto pos_and_chunk : preset_assignments->chunks()) {
    if (pos_and_chunk.first.instruction->opcode() == HloOpcode::kNegate &&
        pos_and_chunk.second.offset == 0) {
      has_zero_offset_allocations = true;
    }
  }
  EXPECT_TRUE(has_zero_offset_allocations);
  XLA_VLOG_LINES(3, module->ToString());
  bool found = false;
  for (auto* c : module->computations()) {
    for (auto* instr : c->instructions()) {
      if (instr->name() == "p0") {
        found = true;
        EXPECT_EQ(instr->shape().layout().memory_space(), 0);
        EXPECT_EQ(module->entry_computation_layout()
                      .parameter_layout(0)
                      .shape()
                      .layout()
                      .memory_space(),
                  0);
      }
    }
  }
  EXPECT_TRUE(found);
}

TEST_F(MemorySpaceAssignmentTest,
       UserAnnotatedCrossProgramPrefetchWithoutPropagationToParameterNoReuse) {
  // This test is same as above, but the S(1) memory space specified in the
  // layout to cross-program prefetch p0 is only present in the entry
  // computation layout and has not been propagated to the parameter
  // instruction. This still works as the previous test.
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true, entry_computation_layout={(f32[8,8]{1,0:S(1)}, f32[8,2]{1,0})->f32[8,2]{1,0}}

  ENTRY CrossProgramPrefetch {
    p0 = f32[8,8]{1,0} parameter(0)
    p1 = f32[8,2]{1,0} parameter(1)
    dot = f32[8,2]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    negate.1 = f32[8,2]{1,0} negate(dot)
    negate.2 = f32[8,2]{1,0} negate(negate.1)
    negate.3 = f32[8,2]{1,0} negate(negate.2)
    negate.4 = f32[8,2]{1,0} negate(negate.3)
    negate.5 = f32[8,2]{1,0} negate(negate.4)
    negate.6 = f32[8,2]{1,0} negate(negate.5)
    negate.7 = f32[8,2]{1,0} negate(negate.6)
    negate.8 = f32[8,2]{1,0} negate(negate.7)
    ROOT negate.9 = f32[8,2]{1,0} negate(negate.8)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 256;
  auto preset_assignments = AssignMemorySpace(module.get(), options,
                                              /*max_prefetch_interval=*/5,
                                              /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  EXPECT_EQ(cross_program_prefetches[0].parameter, 0);
  EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis,
      HloDataflowAnalysis::Run(*module));
  LOG(ERROR) << "module: " << module->ToString();
  const HloValue& cross_program_prefetched_value =
      dataflow_analysis->GetValueDefinedAt(
          module->entry_computation()->parameter_instruction(0), {});
  // Expect that there are two prefetches that use this value, one is the
  // cross-program prefetch, the other is the end-of-program prefetch.
  auto is_cross_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_cross_program_prefetch),
            1);
  auto is_end_of_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           !use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_end_of_program_prefetch),
            1);
  // Also verify that the copy-done for the end-of-program prefetch is the last
  // instruction in schedule.
  const HloInstruction* last_instruction =
      module->schedule()
          .sequence(module->entry_computation())
          .instructions()[module->entry_computation()->instruction_count() - 1];
  EXPECT_THAT(last_instruction, op::CopyDone());
  EXPECT_NE(last_instruction, module->entry_computation()->root_instruction());
  // Cross program prefetch would use offset 0 because that's the first
  // assignment. Since we are freeing the cross-program prefetch buffer, we
  // would also expect to see some of the intermediate computations (one of the
  // negate ops) to also get 0 offset allocations.
  bool has_zero_offset_allocations = false;
  for (auto pos_and_chunk : preset_assignments->chunks()) {
    if (pos_and_chunk.first.instruction->opcode() == HloOpcode::kNegate &&
        pos_and_chunk.second.offset == 0) {
      has_zero_offset_allocations = true;
    }
  }
  EXPECT_TRUE(has_zero_offset_allocations);
  XLA_VLOG_LINES(3, module->ToString());
  bool found = false;
  for (auto* c : module->computations()) {
    for (auto* instr : c->instructions()) {
      if (instr->name() == "p0") {
        found = true;
        EXPECT_EQ(instr->shape().layout().memory_space(), 0);
        EXPECT_EQ(module->entry_computation_layout()
                      .parameter_layout(0)
                      .shape()
                      .layout()
                      .memory_space(),
                  0);
      }
    }
  }
  EXPECT_TRUE(found);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchTupleNoReuse) {
  // This test is for checking if the cross-program-prefetched buffer is freed
  // after its last use and there is an end-of-program prefetch.
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true

  ENTRY CrossProgramPrefetch {
    p0 = (f32[8,8]{1,0}, f32[8,2]{1,0}) parameter(0)
    get-tuple-element = f32[8,8]{1,0} get-tuple-element(p0), index=0
    get-tuple-element.1 = f32[8,2]{1,0} get-tuple-element(p0), index=1
    dot = f32[8,2]{1,0} dot(get-tuple-element, get-tuple-element.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    negate.1 = f32[8,2]{1,0} negate(dot)
    negate.2 = f32[8,2]{1,0} negate(negate.1)
    negate.3 = f32[8,2]{1,0} negate(negate.2)
    negate.4 = f32[8,2]{1,0} negate(negate.3)
    negate.5 = f32[8,2]{1,0} negate(negate.4)
    negate.6 = f32[8,2]{1,0} negate(negate.5)
    negate.7 = f32[8,2]{1,0} negate(negate.6)
    negate.8 = f32[8,2]{1,0} negate(negate.7)
    ROOT negate.9 = f32[8,2]{1,0} negate(negate.8)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpace(
      module.get(), DefaultMemorySpaceOptions(),
      /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  EXPECT_EQ(cross_program_prefetches[0].parameter, 0);
  EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({1}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis,
      HloDataflowAnalysis::Run(*module));
  const HloValue& cross_program_prefetched_value =
      dataflow_analysis->GetValueDefinedAt(
          module->entry_computation()->parameter_instruction(0), {1});
  // Expect that there are two prefetches that use this value, one is the
  // cross-program prefetch, the other is the end-of-program prefetch.
  auto is_cross_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_cross_program_prefetch),
            1);
  auto is_end_of_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           !use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_end_of_program_prefetch),
            1);
  // Also verify that the copy-done for the end-of-program prefetch is the last
  // instruction in schedule.
  const HloInstruction* last_instruction =
      module->schedule()
          .sequence(module->entry_computation())
          .instructions()[module->entry_computation()->instruction_count() - 1];
  EXPECT_THAT(last_instruction, op::CopyDone());
  EXPECT_NE(last_instruction, module->entry_computation()->root_instruction());
  // Cross program prefetch would use offset 0 because that's the first
  // assignment. Since we are freeing the cross-program prefetch buffer, we
  // would also expect to see some of the intermediate computations (one of the
  // negate ops) to also get 0 offset allocations.
  bool has_zero_offset_allocations = false;
  for (auto pos_and_chunk : preset_assignments->chunks()) {
    if (pos_and_chunk.first.instruction->opcode() == HloOpcode::kNegate &&
        pos_and_chunk.second.offset == 0) {
      has_zero_offset_allocations = true;
    }
  }
  EXPECT_TRUE(has_zero_offset_allocations);
}

TEST_F(MemorySpaceAssignmentTest,
       CrossProgramPrefetchEndOfProgramPrefetchAndWhile) {
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true

  while_condition {
    param1 = (f32[8,2]{1,0}, f32[8,2]{1,0}) parameter(0)
    ROOT cond = pred[] constant(true)
  }

  while_body {
    param2 = (f32[8,2]{1,0}, f32[8,2]{1,0}) parameter(0)
    gte2 = f32[8,2]{1,0} get-tuple-element(param2), index=0
    gte3 = f32[8,2]{1,0} get-tuple-element(param2), index=1
    add = f32[8,2]{1,0} add(gte2, gte3)
    negate.2 = f32[8,2]{1,0} negate(add)
    negate.3 = f32[8,2]{1,0} negate(negate.2)
    negate.4 = f32[8,2]{1,0} negate(negate.3)
    negate.5 = f32[8,2]{1,0} negate(negate.4)
    negate.6 = f32[8,2]{1,0} negate(negate.5)
    negate.7 = f32[8,2]{1,0} negate(negate.6)
    negate.8 = f32[8,2]{1,0} negate(negate.7)
    ROOT tuple2 = (f32[8,2]{1,0}, f32[8,2]{1,0}) tuple(negate.8, gte3)
  }

  ENTRY CrossProgramPrefetch {
    p0 = f32[8,8]{1,0} parameter(0)
    p1 = f32[8,2]{1,0} parameter(1)
    dot = f32[8,2]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    negate.1 = f32[8,2]{1,0} negate(dot)
    tuple = (f32[8,2]{1,0}, f32[8,2]{1,0}) tuple(negate.1, dot)
    while = (f32[8,2]{1,0}, f32[8,2]{1,0}) while(tuple), condition=while_condition, body=while_body
    ROOT gte0 = f32[8,2]{1,0} get-tuple-element(while), index=0
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto preset_assignments = AssignMemorySpaceUsingCostAnalysis(module.get());

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  EXPECT_EQ(cross_program_prefetches[0].parameter, 1);
  EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis,
      HloDataflowAnalysis::Run(*module));
  LOG(ERROR) << "module: " << module->ToString();
  const HloValue& cross_program_prefetched_value =
      dataflow_analysis->GetValueDefinedAt(
          module->entry_computation()->parameter_instruction(1), {});
  // Expect that there are two prefetches that use this value, one is the
  // cross-program prefetch, the other is the end-of-program prefetch.
  auto is_cross_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_cross_program_prefetch),
            1);
  auto is_end_of_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           !use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_end_of_program_prefetch),
            1);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchReuse) {
  // This tests the scenario that the cross-program-prefetched buffer is used
  // again close to the end of the computation. In this case, it is better not
  // to free the buffer.
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true

  ENTRY CrossProgramPrefetch {
    p0 = f32[8,8]{1,0} parameter(0)
    p1 = f32[8,2]{1,0} parameter(1)
    dot = f32[8,2]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    negate.1 = f32[8,2]{1,0} negate(dot)
    negate.2 = f32[8,2]{1,0} negate(negate.1)
    negate.3 = f32[8,2]{1,0} negate(negate.2)
    negate.4 = f32[8,2]{1,0} negate(negate.3)
    negate.5 = f32[8,2]{1,0} negate(negate.4)
    negate.6 = f32[8,2]{1,0} negate(negate.5)
    negate.7 = f32[8,2]{1,0} negate(negate.6)
    negate.8 = f32[8,2]{1,0} negate(negate.7)
    ROOT dot.2 = f32[2,2]{1,0} dot(negate.8, p1), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AssignMemorySpace(module.get(), DefaultMemorySpaceOptions(),
                    /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  EXPECT_EQ(cross_program_prefetches[0].parameter, 1);
  EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis,
      HloDataflowAnalysis::Run(*module));
  const HloValue& cross_program_prefetched_value =
      dataflow_analysis->GetValueDefinedAt(
          module->entry_computation()->parameter_instruction(1), {});
  // Expect that there is one prefetch that use this value, the cross-program
  // prefetch. There shouldn't be an end-of-program prefetch.
  auto is_cross_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_cross_program_prefetch),
            1);
  auto is_end_of_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           !use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_end_of_program_prefetch),
            0);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchTupleReuse) {
  // This tests the scenario that the cross-program-prefetched buffer is used
  // again close to the end of the computation. In this case, it is better not
  // to free the buffer.
  absl::string_view hlo_string = R"(
  HloModule cross_program_prefetch, is_scheduled=true

  ENTRY CrossProgramPrefetch {
    p0 = (f32[8,8]{1,0}, f32[8,2]{1,0}) parameter(0)
    get-tuple-element = f32[8,8]{1,0} get-tuple-element(p0), index=0
    get-tuple-element.1 = f32[8,2]{1,0} get-tuple-element(p0), index=1
    dot = f32[8,2]{1,0} dot(get-tuple-element, get-tuple-element.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    negate.1 = f32[8,2]{1,0} negate(dot)
    negate.2 = f32[8,2]{1,0} negate(negate.1)
    negate.3 = f32[8,2]{1,0} negate(negate.2)
    negate.4 = f32[8,2]{1,0} negate(negate.3)
    negate.5 = f32[8,2]{1,0} negate(negate.4)
    negate.6 = f32[8,2]{1,0} negate(negate.5)
    negate.7 = f32[8,2]{1,0} negate(negate.6)
    negate.8 = f32[8,2]{1,0} negate(negate.7)
    ROOT dot.2 = f32[2,2]{1,0} dot(negate.8, get-tuple-element.1), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AssignMemorySpace(module.get(), DefaultMemorySpaceOptions(),
                    /*max_prefetch_interval=*/5, /*min_prefetch_interval=*/2);

  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
  EXPECT_EQ(cross_program_prefetches[0].parameter, 0);
  EXPECT_EQ(cross_program_prefetches[0].index, ShapeIndex({1}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis,
      HloDataflowAnalysis::Run(*module));
  const HloValue& cross_program_prefetched_value =
      dataflow_analysis->GetValueDefinedAt(
          module->entry_computation()->parameter_instruction(0), {1});
  // Expect that there is one prefetch that use this value, the cross-program
  // prefetch. There shouldn't be an end-of-program prefetch.
  auto is_cross_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_cross_program_prefetch),
            1);
  auto is_end_of_program_prefetch = [](const HloUse& use) {
    return use.instruction->opcode() == HloOpcode::kCopyStart &&
           !use.instruction->cross_program_prefetch_index().has_value();
  };
  EXPECT_EQ(absl::c_count_if(cross_program_prefetched_value.GetUses(),
                             is_end_of_program_prefetch),
            0);
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchBufferUnused) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

%fused_computation {
  %param_0.2 = f32[32]{0} parameter(0)
  %param_1.4 = s32[100]{0} parameter(1)
  %custom-call.1 = s32[100]{0} custom-call(s32[100]{0} %param_1.4), custom_call_target="AssumeGatherIndicesInBound", operand_layout_constraints={s32[100]{0}}
  %slice.1 = s32[32]{0} slice(s32[100]{0} %custom-call.1), slice={[0:32]}
  %reshape.7 = s32[32]{0} reshape(s32[32]{0} %slice.1)
  %transpose.5 = s32[32]{0} transpose(s32[32]{0} %reshape.7), dimensions={0}
  %gather.1 = f32[32]{0} gather(f32[32]{0} %param_0.2, s32[32]{0} %transpose.5), offset_dims={}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1}
  %transpose.4 = f32[32]{0} transpose(f32[32]{0} %gather.1), dimensions={0}
  ROOT %reshape.6 = f32[32]{0} reshape(f32[32]{0} %transpose.4)
}

%i.reduce_sub_computation {
  %rhs = s32[] parameter(1)
  %lhs = s32[] parameter(0)
  ROOT %add = s32[] add(s32[] %lhs, s32[] %rhs)
}

%fused_computation.1 {
  %constant.4 = s32[] constant(0)
  %broadcast.4 = s32[100]{0} broadcast(s32[] %constant.4), dimensions={}
  %param_0.4 = s32[32]{0} parameter(0)
  %pad.1 = s32[100]{0} pad(s32[32]{0} %param_0.4, s32[] %constant.4), padding=0_68
  %constant.3 = s32[] constant(76031)
  %broadcast.3 = s32[100]{0} broadcast(s32[] %constant.3), dimensions={}
  ROOT %clamp.1 = s32[100]{0} clamp(s32[100]{0} %broadcast.4, s32[100]{0} %pad.1, s32[100]{0} %broadcast.3)
}

ENTRY %main {
  %constant = s32[] constant(0)
  %i = s32[32,1]{0,1} parameter(1)
  %o = f32[32]{0} parameter(0)
  %reduce = s32[32]{0} reduce(s32[32,1]{0,1} %i, s32[] %constant), dimensions={1}, to_apply=%i.reduce_sub_computation
  %fusion.1 = s32[100]{0} fusion(s32[32]{0} %reduce), kind=kLoop, calls=%fused_computation.1
  ROOT %fusion = f32[32]{0} fusion(f32[32]{0} %o, s32[100]{0} %fusion.1), kind=kCustom, calls=%fused_computation
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Fusion(op::AsyncCopy(kAlternateMemorySpace,
                                       kDefaultMemorySpace, op::Parameter(0)),
                         op::Fusion()));
}

TEST_F(MemorySpaceAssignmentTest, CrossProgramPrefetchPermissiveMode) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

fused_computation {
  param_0 = f32[2] parameter(0)
  param_1 = f32[4,2] parameter(1)
  broadcast = f32[4,2] broadcast(param_0), dimensions={1}
  ROOT multiply = f32[4,2] multiply(broadcast, param_1)
}

ENTRY entry {
  p0 = f32[2] parameter(0)
  p1 = f32[4,2] parameter(1)
  fusion = f32[4,2] fusion(p0, p1), kind=kLoop, calls=fused_computation
  ROOT negate = f32[4,2] negate(fusion)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  Options options = DefaultMemorySpaceOptions();
  options.cross_program_prefetch_permissive_mode = true;
  AssignMemorySpace(module.get(), options);
  auto cross_program_prefetches = module->CrossProgramPrefetches();
  EXPECT_EQ(cross_program_prefetches.size(), 1);
}

// Test description:
// - Setup: Make sure p1 can not be prefetched to alternate memory until after
//   instruction c. We do this by causing p0 to be prefetched to alternate
//   memory for use in c. Since p0 is larger than 1/2 of alternate memory, we
//   will not be able to prefetch p1 until after p0 is unallocated.
// - Test: prefetch p1, after p0 is unallocated from alternate memory (after
//   instruction c).
TEST_F(MemorySpaceAssignmentTest, CopyResourceIntegration) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY main {
  p0 = s32[8,8] parameter(0)
  p1 = s32[8,8] parameter(1)
  p2 = s32[] parameter(2)
  a = negate(p2)
  b = negate(a)
  c = add(p0, p0)
  d = negate(b)
  e = negate(d)
  f = add(p1, p1)

  ROOT result = tuple(e,c,f)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 300;

  // Setup cost analysis so it takes 2 instructions to prefetch anything.
  HloCostAnalysis::Properties properties;
  properties[HloCostAnalysis::kBytesAccessedKey] = kBytesPerSecond;
  HloCostAnalysis hlo_cost_analysis(HloCostAnalysis::DefaultShapeSize,
                                    properties);
  HloCostAnalysisWithAcceptState hlo_cost_analysis_wrapper(hlo_cost_analysis);
  CostAnalysisOptions cost_analysis_options;
  cost_analysis_options.default_mem_bandwidth_bytes_per_second =
      kBytesPerSecond;
  OpCostManager op_cost_manager(
      OpCostManager::Options{
          /*enable_cache=*/false,
          /*enable_analysis_logging=*/false,
      },
      OpCostManager::CalculationNode::CreateLeaf(
          "HloCostAnalysis",
          CreateHloCostAnalysisCalculator(hlo_cost_analysis_wrapper),
          /*enable_cache=*/false));
  TF_ASSERT_OK_AND_ASSIGN(auto cost_analysis,
                          FakeCostAnalysis::Create(op_cost_manager, *module,
                                                   cost_analysis_options));
  cost_analysis->SetOverrideForGetInstructionElapsed(
      [](const HloInstruction& instruction) -> float { return 10.0; });
  cost_analysis->SetOverrideForGetAsyncCopyElapsed(
      [](const Shape& shape) -> float { return 20.0; });
  options.cost_analysis = cost_analysis.get();
  CostAnalysisPrefetchIntervalPicker prefetch_interval_picker(
      CostAnalysisPrefetchIntervalPicker(
          *cost_analysis, /*min_overlap_to_async_copy_ratio=*/0.8,
          /*preferred_overlap_to_async_copy_ratio=*/1.5,
          /*max_overlap_to_mem_size_async_copy_ratio=*/10.0,
          /*mem_size_bytes=*/options.max_size_in_bytes));

  // p0 has the highest priority, followed by p1, followed by everything else.
  MsaBufferIntervalCompare compare = [](const MsaBufferInterval& lhs,
                                        const MsaBufferInterval& rhs) -> bool {
    auto lookup = [](const MsaBufferInterval& x) {
      // An arbitrary value that is greater than that for p0 and p1.
      int priority = 100;
      if (x.buffer->instruction()->name() == "p0") {
        priority = 0;
      } else if (x.buffer->instruction()->name() == "p1") {
        priority = 1;
      }
      return std::make_tuple(priority, x.buffer->instruction()->name());
    };

    return lookup(lhs) < lookup(rhs);
  };

  // Run test.
  AssignMemorySpace(module.get(), options, compare, &prefetch_interval_picker);

  // - Make sure the setup occurred, i.e., that p0 is prefetched to alternate
  //   memory for use by c.
  // - Make sure p1 is prefetched.
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(_,
                // p0 is prefetched to alternate memory for use by c.
                op::Add(op::AsyncCopy(kAlternateMemorySpace,
                                      kDefaultMemorySpace, op::Parameter(0)),
                        op::AsyncCopy(kAlternateMemorySpace,
                                      kDefaultMemorySpace, op::Parameter(0))),
                // p1 is prefetched to alternate memory for use by f.
                op::Add(op::AsyncCopy(kAlternateMemorySpace,
                                      kDefaultMemorySpace, op::Parameter(1)),
                        op::AsyncCopy(kAlternateMemorySpace,
                                      kDefaultMemorySpace, op::Parameter(1)))));

  // Check the schedule
  const std::vector<HloInstruction*>& schedule =
      module->schedule().sequence(module->entry_computation()).instructions();
  auto find_schedule_index = [&schedule](absl::string_view name) -> int {
    for (int i = 0; i < schedule.size(); ++i) {
      if (schedule[i]->name() == name) {
        return i;
      }
    }
    LOG(FATAL) << "Unable to find index of instruction with name " << name;
  };
  int c_index = find_schedule_index("c");
  int p1_copy_start = find_schedule_index(module->entry_computation()
                                              ->root_instruction()  // result
                                              ->operand(2)          // f
                                              ->operand(0)          // copy done
                                              ->operand(0)  // copy start
                                              ->name());
  int d_index = find_schedule_index("d");
  int e_index = find_schedule_index("e");
  int p1_copy_end = find_schedule_index(module->entry_computation()
                                            ->root_instruction()  // result
                                            ->operand(2)          // f
                                            ->operand(0)          // copy done
                                            ->name());
  int f_index = find_schedule_index("f");
  // We expect to start copying p1 after c.
  EXPECT_EQ(p1_copy_start, c_index + 1);
  // d and e should follow come between p1's copy start and end.
  EXPECT_EQ(d_index, p1_copy_start + 1);
  EXPECT_EQ(e_index, d_index + 1);
  EXPECT_EQ(p1_copy_end, e_index + 1);
  // f should immediately follow the end of p1's copy.
  EXPECT_EQ(f_index, p1_copy_end + 1);
}

TEST_F(MemorySpaceAssignmentTest, ExpandScopedAlternateMemory) {
  absl::string_view hlo_string = R"(
  HloModule TestModule, is_scheduled=true
    ENTRY Main {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      p2 = f32[8,8] parameter(2)
      p3 = f32[8,8] parameter(3)

      v0 = add(p0, p1)
      v1 = add(v0, p1)
      v2 = add(v1, p1)

      v3 = multiply(v2, p2)
      v4 = multiply(v3, p3)

      ROOT t = tuple(v3, v4)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  MsaBufferIntervalCompare buffer_interval_compare =
      [](const MsaBufferInterval& lhs, const MsaBufferInterval& rhs) {
        auto lookup = [](const MsaBufferInterval& x) {
          // An arbitrary value that is greater than that used for 'prefetch'.
          int priority = 100;
          if (x.buffer->instruction()->name() == "p2") {
            priority = 1;
          } else if (x.buffer->instruction()->name() == "p3") {
            priority = 2;
          }
          return std::make_tuple(priority, x.buffer->instruction()->name());
        };

        return lookup(lhs) < lookup(rhs);
      };
  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 1000);

  Options options = DefaultMemorySpaceOptions();
  options.max_size_in_bytes = 600;
  options.reserved_scoped_memory_fn =
      [](const HloInstruction* instruction,
         const absl::flat_hash_set<
             std::pair<int, ShapeIndex>>& /*operands_in_alternate_memory*/,
         const absl::flat_hash_set<
             ShapeIndex>& /*outputs_in_alternate_memory*/) { return 10; };
  options.expanded_scoped_alternate_memory_mode =
      ExpandedScopedAlternateMemoryMode::ENABLED;
  options.alignment_in_bytes = 10;
  std::unique_ptr<PresetAssignments> preset_assignments =
      AssignMemorySpace(module.get(), options, buffer_interval_compare,
                        &prefetch_interval_picker);

  VLOG(1) << "Post-MSA module:\n" << module->ToString();

  // We expect MSA to do the following:
  // A. Initially allocate [0, 10) for scoped alternate memory, for each
  //    instruction.
  // B. Since, p2 comes first in the buffer sorting, we expect it to be
  //    allocated [10, 266) for a prefetch
  // C. Since, p3 comes next in the buffer sorting, we expect it to be allocated
  //    [270, 526) for a prefetch
  // D. Finally, MSA will try to expand the scoped alternate memory allocations
  //    to the largest available buffers, keeping in mind the prefetches.

  // Check B and C.
  for (const auto& [position, chunk] : preset_assignments->chunks()) {
    if (position.instruction->opcode() == HloOpcode::kCopyDone) {
      ASSERT_EQ(position.instruction->operand_count(), 1);
      const HloInstruction* copy_start = position.instruction->operand(0);
      ASSERT_EQ(copy_start->operand_count(), 1);
      const HloInstruction* copy_operand = copy_start->operand(0);
      if (copy_operand->name() == "p2") {
        EXPECT_EQ(chunk.offset, 10);
        EXPECT_EQ(chunk.size, 256);
      } else if (copy_operand->name() == "p3") {
        EXPECT_EQ(chunk.offset, 270);
        EXPECT_EQ(chunk.size, 256);
      }
    }
  }

  // Check D.
  for (const auto& [instruction, chunk] :
       preset_assignments->scoped_allocation_chunks()) {
    if (instruction->name() == "p0") {
      // Extended scoped allocation.
      EXPECT_EQ(chunk.offset, 0);
      EXPECT_EQ(chunk.size, 600);
    } else if (instruction->name() == "p1") {
      // Extended scoped allocation.
      EXPECT_EQ(chunk.offset, 0);
      EXPECT_EQ(chunk.size, 600);
    } else if (instruction->name() == "p2") {
      // Extended scoped allocation.
      EXPECT_EQ(chunk.offset, 0);
      EXPECT_EQ(chunk.size, 600);
    } else if (instruction->name() == "p3") {
      // Moved scoped allocation.
      EXPECT_EQ(chunk.offset, 270);
      EXPECT_EQ(chunk.size, 330);
    } else if (instruction->name() == "v0") {
      // Moved scoped allocation.
      EXPECT_EQ(chunk.offset, 530);
      EXPECT_EQ(chunk.size, 70);
    } else if (instruction->name() == "v1") {
      // Moved scoped allocation.
      EXPECT_EQ(chunk.offset, 530);
      EXPECT_EQ(chunk.size, 70);
    } else if (instruction->name() == "v2") {
      // Moved scoped allocation.
      EXPECT_EQ(chunk.offset, 530);
      EXPECT_EQ(chunk.size, 70);
    } else if (instruction->name() == "v3") {
      // Moved scoped allocation.
      EXPECT_EQ(chunk.offset, 530);
      EXPECT_EQ(chunk.size, 70);
    } else if (instruction->name() == "v4") {
      // Extended scoped allocation.
      EXPECT_EQ(chunk.offset, 0);
      EXPECT_EQ(chunk.size, 270);
    } else if (instruction->name() == "t") {
      // Extended scoped allocation.
      EXPECT_EQ(chunk.offset, 0);
      EXPECT_EQ(chunk.size, 600);
    }
  }
}

class SlicedPrefetchTest : public MemorySpaceAssignmentTestBase {
 protected:
  // Used by CheckSchedule() to classify instructions in the schedule.
  enum class InstructionClass {
    kUnknown,
    // A slice start that we care about, as determined by the test.
    kRelatedSliceStart,
    // A slice done that we care about, as determined by the test.
    kRelatedSliceDone,
    // A concat-bitcast that we care about, as determined by the test.
    kRelatedConcatBitcast,
    // A non-copy-like instruction that slices should start after.
    kStartAfterNonCopy,
    // A non-copy-like instruction that slices should finish before.
    kDoneBeforeNonCopy,
    // A copy, slice, or concat-bitcast that we don't care about.
    kUnrelatedCopyLike,
    // All other instructions.
    kUnrelatedNonCopy,
  };

  static std::string InstructionClassToString(
      InstructionClass instruction_class) {
    switch (instruction_class) {
      case InstructionClass::kUnknown:
        return "unknown";
      case InstructionClass::kRelatedSliceStart:
        return "slice start";
      case InstructionClass::kRelatedSliceDone:
        return "slice done";
      case InstructionClass::kRelatedConcatBitcast:
        return "concat-bitcast";
      case InstructionClass::kStartAfterNonCopy:
        return "start after non-copy";
      case InstructionClass::kDoneBeforeNonCopy:
        return "done before non-copy";
      case InstructionClass::kUnrelatedCopyLike:
        return "unrelated copy-like";
      case InstructionClass::kUnrelatedNonCopy:
        return "unrelated non-copy";
    }
  }

  // A class that can be mocked to set expectations on slice proposals. To do
  // that, we set Options::propose_slice_fn to a lambda that calls our mocks
  // ProposeSlices() method.
  class SliceProposer {
   public:
    SliceProposer() = default;
    virtual ~SliceProposer() = default;

    virtual absl::StatusOr<SliceProposalCollection> ProposeSlices(
        const Shape& shape, const SlicedPrefetchOptions& options) = 0;
  };

  class MockSliceProposer : public SliceProposer {
   public:
    MOCK_METHOD(absl::StatusOr<SliceProposalCollection>, ProposeSlices,
                (const Shape& shape, const SlicedPrefetchOptions& options),
                (override));
  };

  // An HloInstruction* matcher for matching the asynchronous sliced copies
  // produced by MSA. In particular, the matcher performs the following
  // checks:
  // - The copy is concluded with a concat-bitcast custom call, or a
  //   bitcast of a concat-bitcast custom call if expect_bitcasted_io is true
  // - The operands to the concat-bitcast are asynchronous slices of the
  //   expected operand, or asynchronous slices of a bitcast of the expected
  //   operand if expect_bitcasted_io is true
  // - The number of slices is as expected (i.e.,
  //   expected_slice_params_per_slice_in_spatial_order_.size())
  // - The copy is from and to the correct memory spaces
  // - The shape before and after the copy is the same
  // - When the slices are sorted in expected spatial order, their slice
  //   starts and limits are as expected
  // - The slices are to the correct memory space
  // - All slices have slice strides of 1
  class AsyncSlicedCopy
      : public ::testing::MatcherInterface<const HloInstruction*> {
   public:
    // The parameters in expected_slice_params_per_slice_in_spatial_order should
    // be sorted in the order in which we expect the corresponding slices to be
    // laid out in memory.
    AsyncSlicedCopy(int64_t to_space, int64_t from_space,
                    std::vector<std::vector<SliceParam>>
                        expected_slice_params_per_slice_in_spatial_order,
                    ::testing::Matcher<const HloInstruction*> operand,
                    bool expect_bitcasted_io)
        : to_space_(to_space),
          from_space_(from_space),
          expected_slice_params_per_slice_in_spatial_order_(
              std::move(expected_slice_params_per_slice_in_spatial_order)),
          base_hlo_matcher_(CreateBaseHloMatcher(
              operand, expected_slice_params_per_slice_in_spatial_order_.size(),
              expect_bitcasted_io)),
          expect_bitcasted_io_(expect_bitcasted_io) {}

    bool MatchAndExplain(
        const HloInstruction* instruction,
        ::testing::MatchResultListener* listener) const override {
      // Match opcodes and number of operands.
      if (!base_hlo_matcher_.MatchAndExplain(instruction, listener)) {
        return false;
      }

      // Check if the copied result has the proper memory space.
      if (!MatchMemorySpace(instruction, to_space_, "copy result", listener)) {
        return false;
      }

      // Find some instructions in the async copy.
      const HloInstruction* concat_bitcast =
          (expect_bitcasted_io_ ? instruction->operand(0) : instruction);
      VLOG(2) << "AsyncSlicedCopy identified the concat-bitcast as "
              << concat_bitcast->name();
      const HloInstruction* copy_operand =
          concat_bitcast->operand(0)->operand(0)->operand(0);
      const HloInstruction* original_copy_operand =
          (expect_bitcasted_io_ ? copy_operand->operand(0) : copy_operand);
      VLOG(2) << "AsyncSlicedCopy identified the copy operand as "
              << copy_operand->name() << ", and the original copy operand as "
              << original_copy_operand->name();

      // Check if the copied tensor has the proper memory space.
      if (!MatchMemorySpace(original_copy_operand, from_space_, "copy operand",
                            listener)) {
        return false;
      }

      // Check if the copied tensor retains its shape.
      if (!Shape::Equal().IgnoreMemorySpaceInLayout()(
              instruction->shape(), original_copy_operand->shape())) {
        *listener << " has a shape of "
                  << original_copy_operand->shape().ToString(
                         /*print_layout=*/true)
                  << " before copying but a shape of "
                  << instruction->shape().ToString(/*print_layout=*/true)
                  << " after copying (ignoring memory space)";

        return false;
      }

      // This should already be checked in the custom call matcher.
      CHECK_EQ(concat_bitcast->operand_count(),
               expected_slice_params_per_slice_in_spatial_order_.size());

      // Check if the slicing parameters are correct and if the slices are to
      // the correct memory space.
      std::vector<const HloInstruction*> sorted_slices =
          SortSlicesInExpectedSpatialOrder(concat_bitcast);
      for (int i = 0; i < sorted_slices.size(); ++i) {
        const HloInstruction* slice =
            sorted_slices[i]->async_wrapped_instruction();

        if (!MatchMemorySpace(slice, to_space_, "slice", listener)) {
          return false;
        }

        const std::vector<SliceParam>& expected_slice_params_per_dim =
            expected_slice_params_per_slice_in_spatial_order_[i];
        if (slice->slice_starts().empty()) {
          *listener << " has slice (" << slice->name()
                    << "), with no slicing parameters";
          return false;
        }
        if (slice->slice_limits().size() != slice->slice_starts().size() ||
            slice->slice_strides().size() != slice->slice_limits().size()) {
          *listener
              << " has slice (" << slice->name()
              << "), with an inconsistent number slice starts/limits/strides";
          return false;
        }
        if (slice->slice_starts().size() !=
            copy_operand->shape().dimensions().size()) {
          *listener
              << " has slice (" << slice->name() << "), with "
              << slice->slice_starts().size()
              << " slice parameters (i.e., starts/limits/strides), expected "
              << expected_slice_params_per_slice_in_spatial_order_.size();
          return false;
        }
        for (int dim = 0; dim < slice->slice_starts().size(); ++dim) {
          const SliceParam& expected_slice_params =
              expected_slice_params_per_dim[dim];
          if (slice->slice_starts()[dim] !=
              expected_slice_params.start_inclusive) {
            *listener << " has slice (" << slice->name()
                      << "), with slice start of " << slice->slice_starts()[dim]
                      << " at dim " << dim << ", expected "
                      << expected_slice_params.start_inclusive;
            return false;
          }
          if (slice->slice_limits()[dim] !=
              expected_slice_params.end_exclusive) {
            *listener << " has slice (" << slice->name()
                      << "), with slice limit of " << slice->slice_limits()[dim]
                      << " at dim " << dim << ", expected "
                      << expected_slice_params.end_exclusive;
            return false;
          }
          if (slice->slice_strides()[dim] != 1) {
            *listener << " has slice (" << slice->name()
                      << "), slice stride of " << slice->slice_strides()[dim]
                      << " at dim " << dim << ", expected 1";
            return false;
          }
        }
      }

      return true;
    }

    void DescribeTo(std::ostream* os) const override {
      base_hlo_matcher_.DescribeTo(os);
      std::vector<std::string> slice_parameters_per_operand;
      for (int op_idx = 0;
           op_idx < expected_slice_params_per_slice_in_spatial_order_.size();
           ++op_idx) {
        std::vector<std::string> slice_params_per_dim;
        for (int dim = 0;
             dim <
             expected_slice_params_per_slice_in_spatial_order_[op_idx].size();
             ++dim) {
          const SliceParam& slice_params =
              expected_slice_params_per_slice_in_spatial_order_[op_idx][dim];
          slice_params_per_dim.push_back(absl::StrCat(
              "dim ", dim, ": {start: ", slice_params.start_inclusive,
              ", limit: ", slice_params.end_exclusive, "}"));
        }
        slice_parameters_per_operand.push_back(
            absl::StrCat("operand ", op_idx, ": { ",
                         absl::StrJoin(slice_params_per_dim, ", "), " }"));
      }
      *os << " (copying from memory space " << from_space_ << " to "
          << to_space_
          << ", with asynchronous slice operands using the following slice "
             "parameters: { "
          << absl::StrJoin(slice_parameters_per_operand, ", ") << " })";
    }

   private:
    static ::testing::Matcher<const HloInstruction*> CreateBaseHloMatcher(
        ::testing::Matcher<const HloInstruction*> operand, int64_t num_slices,
        bool expect_bitcasted_io) {
      if (expect_bitcasted_io) {
        return op::Bitcast(op::CustomCall(
            kConcatBitcastCustomCall,
            std::vector<::testing::Matcher<const HloInstruction*>>(
                num_slices,
                op::AsyncDone(op::AsyncStart(op::Bitcast(operand))))));
      }
      return op::CustomCall(
          kConcatBitcastCustomCall,
          std::vector<::testing::Matcher<const HloInstruction*>>(
              num_slices, op::AsyncDone(op::AsyncStart(operand))));
    }

    static bool MatchMemorySpace(const HloInstruction* instruction,
                                 int64_t expected_memory_space,
                                 absl::string_view error_message_identifier,
                                 ::testing::MatchResultListener* listener) {
      if (!instruction->shape().has_layout()) {
        *listener << " contains " << error_message_identifier << " named "
                  << instruction->name()
                  << " without a layout, expected a layout with memory space "
                  << expected_memory_space;
        return false;
      }
      if (instruction->shape().layout().memory_space() !=
          expected_memory_space) {
        *listener << " contains " << error_message_identifier << " named "
                  << instruction->name() << " in memory space "
                  << expected_memory_space << ", expected  "
                  << expected_memory_space;
        return false;
      }

      return true;
    }

    int64_t to_space_;
    int64_t from_space_;
    std::vector<std::vector<SliceParam>>
        expected_slice_params_per_slice_in_spatial_order_;
    ::testing::Matcher<const HloInstruction*> base_hlo_matcher_;
    bool expect_bitcasted_io_;
  };

  // Returns an AsyncSlicedCopy matcher.
  static inline ::testing::Matcher<const HloInstruction*> IsAsyncSlicedCopy(
      int64_t to_space, int64_t from_space,
      std::vector<std::vector<SliceParam>>
          expected_slice_params_per_slice_in_spatial_order,
      ::testing::Matcher<const HloInstruction*> operand_matcher,
      bool expect_bitcasted_io = false) {
    return ::testing::MakeMatcher(new AsyncSlicedCopy(
        to_space, from_space, expected_slice_params_per_slice_in_spatial_order,
        operand_matcher, expect_bitcasted_io));
  }

  // We make our own matcher for SlicedPrefetchOptions to work around the fact
  // third_party/tensorflow does not have any generic way to match proto
  // buffers.
  class SlicedPrefetchOptionsMatcher
      : public ::testing::MatcherInterface<const SlicedPrefetchOptions&> {
   public:
    explicit SlicedPrefetchOptionsMatcher(
        SlicedPrefetchOptions expected_options)
        : expected_options_(std::move(expected_options)) {}

    bool MatchAndExplain(
        const SlicedPrefetchOptions& options,
        ::testing::MatchResultListener* listener) const override {
      if (options.max_slices() != expected_options_.max_slices()) {
        *listener << " has " << options.max_slices() << " max slices, expected "
                  << expected_options_.max_slices();
        return false;
      }

      if (options.min_bytes() != expected_options_.min_bytes()) {
        *listener << " has " << options.min_bytes() << " min bytes, expected "
                  << expected_options_.min_bytes();
        return false;
      }

      if (options.fail_on_non_alignment_boundary_slice_proposal() !=
          expected_options_.fail_on_non_alignment_boundary_slice_proposal()) {
        *listener
            << " has fail_on_non_alignment_boundary_slice_proposal set to "
            << options.fail_on_non_alignment_boundary_slice_proposal()
            << ", expected "
            << expected_options_
                   .fail_on_non_alignment_boundary_slice_proposal();
        return false;
      }

      return true;
    }

    void DescribeTo(std::ostream* os) const override {
      *os << " has the following options: max_slices("
          << expected_options_.max_slices() << "), min_bytes("
          << expected_options_.min_bytes()
          << ") fail_on_non_alignment_boundary_slice_proposal("
          << expected_options_.fail_on_non_alignment_boundary_slice_proposal()
          << ")";
    }

   private:
    SlicedPrefetchOptions expected_options_;
  };

  // Returns an SlicedPrefetchOptions matcher.
  static inline ::testing::Matcher<const SlicedPrefetchOptions&>
  EqualsSlicedPrefetchOptions(SlicedPrefetchOptions expected_options) {
    return ::testing::MakeMatcher(
        new SlicedPrefetchOptionsMatcher(std::move(expected_options)));
  }

  // Slices can be passed to the concat-bitcast in any order. This function
  // sorts a the slices in the order they should spatially (in memory). Note,
  // this function is specific to the way we are constructing slices for the
  // test. E.g., it relies on the first dimension of the tensor to be the
  // slice dimension.
  //
  // REQUIRES:
  // - All operands of concat_bitcast must be asynchronous slices.
  static std::vector<const HloInstruction*> SortSlicesInExpectedSpatialOrder(
      const HloInstruction* concat_bitcast) {
    std::vector<const HloInstruction*> sorted_slices(
        concat_bitcast->operands().begin(), concat_bitcast->operands().end());

    absl::c_sort(sorted_slices, [](const HloInstruction* lhs,
                                   const HloInstruction* rhs) {
      CHECK(IsAsyncSliceDone(lhs));
      CHECK(IsAsyncSliceDone(rhs));
      CHECK(!lhs->async_wrapped_instruction()->slice_starts().empty());
      CHECK(!rhs->async_wrapped_instruction()->slice_starts().empty());
      return lhs->async_wrapped_instruction()->slice_starts().front() <
             rhs->async_wrapped_instruction()->slice_starts().front();
    });

    return sorted_slices;
  }

  // Returns true if instruction is an async copy start.
  static bool IsAsyncCopyStart(const HloInstruction* instruction) {
    return instruction->opcode() == HloOpcode::kCopyStart;
  }

  // Returns true if instruction is an async copy done.
  static bool IsAsyncCopyDone(const HloInstruction* instruction) {
    return instruction->opcode() == HloOpcode::kCopyDone;
  }

  // Returns true if instruction is an async slice start.
  static bool IsAsyncSliceStart(const HloInstruction* instruction) {
    return instruction->opcode() == HloOpcode::kAsyncStart &&
           instruction->async_wrapped_instruction()->opcode() ==
               HloOpcode::kSlice;
  }

  // Returns true if instruction is an async slice done.
  static bool IsAsyncSliceDone(const HloInstruction* instruction) {
    return instruction->opcode() == HloOpcode::kAsyncDone &&
           instruction->async_wrapped_instruction()->opcode() ==
               HloOpcode::kSlice;
  }

  // Returns true if instruction is a concat-bitcast.
  static bool IsConcatBitcast(const HloInstruction* instruction) {
    return instruction->IsCustomCall(kConcatBitcastCustomCall);
  }

  // Returns the index of the first instruction with the given name.
  static absl::StatusOr<int> FindScheduleIndexOfInstruction(
      const std::vector<HloInstruction*>& schedule, absl::string_view name,
      InstructionClass c) {
    for (int i = 0; i < schedule.size(); ++i) {
      if (schedule[i]->name() == name) {
        return i;
      }
    }

    return NotFound(
        "%s",
        absl::StrCat("Could not find ", InstructionClassToString(c),
                     " instruction ", name, " in the instruction schedule."));
  }

  // Returns a scheduled instruction with the specified name or null.
  static const HloInstruction* FindNamedScheduledInstruction(
      const HloModule& module, absl::string_view name) {
    for (const HloInstruction* i : module.entry_computation()->instructions()) {
      if (i->name() == name) {
        return i;
      }
    }
    return nullptr;
  }

  static absl::StatusOr<std::vector<int>> GetSliceStartIndicies(
      const std::vector<HloInstruction*>& schedule,
      const HloInstruction* concat_bitcast) {
    std::vector<int> indicies;

    if (!IsConcatBitcast(concat_bitcast)) {
      return InvalidArgumentStrCat(concat_bitcast->name(),
                                   " is not a concat-bitcast.");
    }
    for (int i = 0; i < concat_bitcast->operand_count(); ++i) {
      const HloInstruction* async_slice_done = concat_bitcast->operand(i);
      if (!IsAsyncSliceDone(async_slice_done)) {
        return InvalidArgumentStrCat("Operand ", i, " of ",
                                     concat_bitcast->name(),
                                     " is not an async-slice-done.");
      }
      const HloInstruction* async_slice_start = async_slice_done->operand(0);
      if (!IsAsyncSliceStart(async_slice_start)) {
        return InvalidArgumentStrCat("Operand 0, of operand ", i, " of ",
                                     concat_bitcast->name(),
                                     " is not an async-slice-start.");
      }
      TF_ASSIGN_OR_RETURN(
          int schedule_index,
          FindScheduleIndexOfInstruction(schedule, async_slice_start->name(),
                                         InstructionClass::kRelatedSliceStart));
      indicies.push_back(schedule_index);
    }

    return indicies;
  }

  // REQUIRES:
  // - Concat-bitcast and all slices were found in the schedule used to
  //   construct schedule_to_class.
  static absl::Status ConcatBitcastAndSlicesAfterInstruction(
      const std::vector<HloInstruction*>& schedule,
      const std::vector<InstructionClass>& schedule_to_class,
      int slices_start_after_index) {
    for (int i = 0; i < slices_start_after_index; ++i) {
      InstructionClass c = schedule_to_class[i];
      const HloInstruction* instruction = schedule[i];

      if (c == InstructionClass::kRelatedSliceStart ||
          c == InstructionClass::kRelatedSliceDone ||
          c == InstructionClass::kRelatedConcatBitcast) {
        return FailedPrecondition(
            "%s", absl::StrCat(InstructionClassToString(c), " ",
                               instruction->name(), " is scheduled at ", i,
                               ", but is expected to be after ",
                               schedule[slices_start_after_index]->name(),
                               " at ", slices_start_after_index, "."));
      }
    }

    return absl::OkStatus();
  }

  // REQUIRES:
  // - Concat-bitcast and all slices were found in the schedule used to
  //   construct schedule_to_class.
  static absl::Status AtLeastOneNonCopyLikeInstructionBetweenSliceStarts(
      const std::vector<HloInstruction*>& schedule,
      const std::vector<InstructionClass>& schedule_to_class) {
    bool found_non_copy_since_last_slice_start = true;
    for (int i = 0; i < schedule_to_class.size(); ++i) {
      InstructionClass c = schedule_to_class[i];

      if (c == InstructionClass::kRelatedSliceStart &&
          !found_non_copy_since_last_slice_start) {
        return FailedPrecondition(
            "%s",
            absl::StrCat(
                "Did not find a non-copy-like instruction between slice start ",
                schedule[i]->name(), " at ", i,
                " and the previous slice start."));
      }

      if (c == InstructionClass::kRelatedSliceStart) {
        found_non_copy_since_last_slice_start = false;
      } else if (c == InstructionClass::kUnrelatedNonCopy) {
        found_non_copy_since_last_slice_start = true;
      }
    }

    return absl::OkStatus();
  }

  // REQUIRES:
  // - Concat-bitcast and all slices were found in the schedule used to
  //   construct schedule_to_class.
  static absl::Status OneSliceStartAfterInstructionWithNoCopyLikeBetween(
      const std::vector<HloInstruction*>& schedule,
      const std::vector<InstructionClass>& schedule_to_class,
      int slices_start_after_index) {
    int first_slice_start_after_schedule_after = -1;
    int first_non_copy_after_schedule_after = -1;
    for (int i = slices_start_after_index + 1;
         i < schedule_to_class.size() &&
         (first_slice_start_after_schedule_after == -1 ||
          first_non_copy_after_schedule_after == -1);
         ++i) {
      if (first_slice_start_after_schedule_after == -1 &&
          schedule_to_class[i] == InstructionClass::kRelatedSliceStart) {
        first_slice_start_after_schedule_after = i;
        continue;
      }
      if (first_non_copy_after_schedule_after == -1 &&
          schedule_to_class[i] == InstructionClass::kUnrelatedNonCopy) {
        first_non_copy_after_schedule_after = i;
        continue;
      }
    }
    if (first_slice_start_after_schedule_after == -1) {
      return NotFound(
          "%s", absl::StrCat("Could not find a slice start instruction "
                             "after start after instruction ",
                             schedule[slices_start_after_index]->name(), " at ",
                             slices_start_after_index, "."));
    }
    if (first_non_copy_after_schedule_after == -1) {
      return NotFound(
          "%s", absl::StrCat("Could not a find non-copy-like instruction "
                             "after start after instruction ",
                             schedule[slices_start_after_index]->name(), " at ",
                             slices_start_after_index, "."));
    }
    if (first_slice_start_after_schedule_after >
        first_non_copy_after_schedule_after) {
      return FailedPrecondition(
          "%s", absl::StrCat(
                    "Unexpectedly found a non-copy-like instruction at ",
                    first_non_copy_after_schedule_after, ", between ",
                    schedule[slices_start_after_index]->name(), " at ",
                    slices_start_after_index, ", and the first slice start at ",
                    first_slice_start_after_schedule_after, "."));
    }

    return absl::OkStatus();
  }

  // REQUIRES:
  // - Concat-bitcast and all slices were found in the schedule used to
  //   construct schedule_to_class.
  static absl::Status ConcatBitcastAndSlicesBeforeInstruction(
      const std::vector<HloInstruction*>& schedule,
      const std::vector<InstructionClass>& schedule_to_class,
      int slices_done_before_index) {
    for (int i = slices_done_before_index + 1; i < schedule_to_class.size();
         ++i) {
      InstructionClass c = schedule_to_class[i];
      const HloInstruction* instruction = schedule[i];

      if (c == InstructionClass::kRelatedSliceStart ||
          c == InstructionClass::kRelatedSliceDone ||
          c == InstructionClass::kRelatedConcatBitcast) {
        return FailedPrecondition(
            "%s", absl::StrCat(InstructionClassToString(c), " ",
                               instruction->name(), " is scheduled at ", i,
                               ", but is expected to be before ",
                               schedule[slices_done_before_index]->name(),
                               " at ", slices_done_before_index, "."));
      }
    }

    return absl::OkStatus();
  }

  // REQUIRES:
  // - Concat-bitcast and all slices were found in the schedule used to
  //   construct schedule_to_class.
  static absl::Status
  ConcatBitcastAndSliceDonesBeforeInstructionWithNoCopyLikeBetween(
      const std::vector<HloInstruction*>& schedule,
      const std::vector<InstructionClass>& schedule_to_class,
      int slices_done_before_index) {
    bool found_non_copy = false;
    for (int i = slices_done_before_index - 1; i >= 0; --i) {
      InstructionClass c = schedule_to_class[i];
      const HloInstruction* instruction = schedule[i];

      if (c == InstructionClass::kUnrelatedNonCopy) {
        found_non_copy = true;
        continue;
      }

      if (found_non_copy && (c == InstructionClass::kRelatedSliceDone ||
                             c == InstructionClass::kRelatedConcatBitcast)) {
        return FailedPrecondition(
            "%s",
            absl::StrCat("Found non-copy instruction between ",
                         InstructionClassToString(c), " ", instruction->name(),
                         " at ", i, ", and slice done before instruction ",
                         schedule[slices_done_before_index]->name(), " at ",
                         slices_done_before_index, "."));
      }
    }

    return absl::OkStatus();
  }

  // REQUIRES:
  // - Concat-bitcast and all slices were found in the schedule used to
  //   construct schedule_to_class.
  static absl::Status ConcatBitcastAfterSliceDones(
      const std::vector<HloInstruction*>& schedule,
      const std::vector<InstructionClass>& schedule_to_class) {
    int concat_bitcast_index = -1;
    for (int i = 0; i < schedule_to_class.size(); ++i) {
      InstructionClass c = schedule_to_class[i];
      const HloInstruction* instruction = schedule[i];

      if (concat_bitcast_index == -1 &&
          c == InstructionClass::kRelatedConcatBitcast) {
        concat_bitcast_index = i;
        continue;
      }
      if (concat_bitcast_index != -1 &&
          c == InstructionClass::kRelatedSliceDone) {
        return FailedPrecondition(
            "%s", absl::StrCat("Unexpectedly, found concat-bitcast ",
                               schedule[concat_bitcast_index]->name(), " at ",
                               concat_bitcast_index,
                               ", which is before the slice done ",
                               instruction->name(), " at ", i, "."));
      }
    }

    return absl::OkStatus();
  }

  // Return an OK status iff:
  // - concat_bitcast and all of its slices come after
  //   slices_start_after_instruction_name in the schedule AND
  // - at least one slice start comes after slices_start_after_instruction_name
  //   in the schedule, with no non-copy-like instruction between AND
  // - if expect_slices_started_at_different_times is true, at least one
  //   non-copy-like instruction comes between each slice start AND
  // - concat_bitcast and all of its slices come before
  //   slices_done_before_instruction_name in the schedule AND
  // - concat_bitcast and all of its slice dones come before
  //   slices_done_before_instruction_name in the schedule, with no
  //   non-copy-like instruction between AND
  // - concat_bitcast comes after all slice dones AND
  static absl::Status CheckSchedule(
      const HloModule& module, const HloInstruction* concat_bitcast,
      absl::string_view slices_start_after_instruction_name,
      absl::string_view slices_done_before_instruction_name,
      bool expect_slices_started_at_different_times) {
    CHECK(concat_bitcast->IsCustomCall(kConcatBitcastCustomCall));

    // Get the schedule.
    auto entry_schedule =
        module.schedule().sequence(module.entry_computation()).instructions();

    // Initialize schedule_to_class to classify instructions in the schedule.
    std::vector<InstructionClass> schedule_to_class(
        entry_schedule.size(), InstructionClass::kUnrelatedNonCopy);
    for (int i = 0; i < entry_schedule.size(); ++i) {
      const HloInstruction* instruction = entry_schedule[i];
      if (IsAsyncCopyStart(instruction) || IsAsyncCopyDone(instruction) ||
          IsAsyncSliceStart(instruction) || IsAsyncSliceDone(instruction) ||
          IsConcatBitcast(instruction)) {
        schedule_to_class[i] = InstructionClass::kUnrelatedCopyLike;
      }
    }

    // Update schedule_to_class with the instructions we care about.
    int slices_start_after_index;
    TF_ASSIGN_OR_RETURN(slices_start_after_index,
                        FindScheduleIndexOfInstruction(
                            entry_schedule, slices_start_after_instruction_name,
                            InstructionClass::kStartAfterNonCopy));
    schedule_to_class[slices_start_after_index] =
        InstructionClass::kStartAfterNonCopy;
    int slices_done_before_index;
    TF_ASSIGN_OR_RETURN(slices_done_before_index,
                        FindScheduleIndexOfInstruction(
                            entry_schedule, slices_done_before_instruction_name,
                            InstructionClass::kDoneBeforeNonCopy));
    schedule_to_class[slices_done_before_index] =
        InstructionClass::kDoneBeforeNonCopy;
    int concat_bitcast_index;
    TF_ASSIGN_OR_RETURN(concat_bitcast_index,
                        FindScheduleIndexOfInstruction(
                            entry_schedule, concat_bitcast->name(),
                            InstructionClass::kRelatedConcatBitcast));
    schedule_to_class[concat_bitcast_index] =
        InstructionClass::kRelatedConcatBitcast;
    for (const HloInstruction* slice : concat_bitcast->operands()) {
      int done_index;
      TF_ASSIGN_OR_RETURN(done_index, FindScheduleIndexOfInstruction(
                                          entry_schedule, slice->name(),
                                          InstructionClass::kRelatedSliceDone));
      schedule_to_class[done_index] = InstructionClass::kRelatedSliceDone;
      int start_index;
      TF_ASSIGN_OR_RETURN(start_index,
                          FindScheduleIndexOfInstruction(
                              entry_schedule, slice->operand(0)->name(),
                              InstructionClass::kRelatedSliceStart));
      schedule_to_class[start_index] = InstructionClass::kRelatedSliceStart;
    }

    // Perform scheduling checks.
    TF_RETURN_IF_ERROR(ConcatBitcastAndSlicesAfterInstruction(
        entry_schedule, schedule_to_class, slices_start_after_index));
    TF_RETURN_IF_ERROR(OneSliceStartAfterInstructionWithNoCopyLikeBetween(
        entry_schedule, schedule_to_class, slices_start_after_index));
    if (expect_slices_started_at_different_times) {
      TF_RETURN_IF_ERROR(AtLeastOneNonCopyLikeInstructionBetweenSliceStarts(
          entry_schedule, schedule_to_class));
    }
    TF_RETURN_IF_ERROR(ConcatBitcastAndSlicesBeforeInstruction(
        entry_schedule, schedule_to_class, slices_done_before_index));
    TF_RETURN_IF_ERROR(
        ConcatBitcastAndSliceDonesBeforeInstructionWithNoCopyLikeBetween(
            entry_schedule, schedule_to_class, slices_done_before_index));
    TF_RETURN_IF_ERROR(
        ConcatBitcastAfterSliceDones(entry_schedule, schedule_to_class));

    return absl::OkStatus();
  }

  // Returns absl::OkStatus iff:
  // - Each slice is assigned a chunk that is the same size as the slice
  //   instruction's shape.
  // - When the slices of sliced_copy_result are sorted in expected spatial
  //   order, they are assigned chunks that spatially fall in the same order AND
  // - The slices of sliced_copy_result are assigned contiguous memory chunks
  //   AND
  // - The sliced_copy_result is assigned a chunk that is the concatenation of
  //   the slice chunks AND
  // - The size of the chunk assigned to the sliced_copy_result has the same
  //   size as the instruction's shape
  static absl::Status CheckSliceChunks(const PresetAssignments& assignments,
                                       const HloInstruction* sliced_copy_result,
                                       bool expect_bitcasted_io = false) {
    const HloInstruction* concat_bitcast =
        (expect_bitcasted_io ? sliced_copy_result->operand(0)
                             : sliced_copy_result);
    CHECK(concat_bitcast->IsCustomCall(kConcatBitcastCustomCall));

    absl::flat_hash_map<const HloInstruction*, Chunk> slices_to_chunks;
    std::optional<Chunk> result_chunk = std::nullopt;

    for (const std::pair<HloPosition, Chunk>& position_chunk_pair :
         assignments.chunks()) {
      if (position_chunk_pair.first.instruction == sliced_copy_result) {
        if (result_chunk.has_value()) {
          return FailedPrecondition(
              "%s", absl::StrCat("Sliced copy ", sliced_copy_result->name(),
                                 " is assigned more than one chunk: ",
                                 result_chunk->ToString(), " and ",
                                 position_chunk_pair.second.ToString()));
        }
        result_chunk = position_chunk_pair.second;
      }
      for (const HloInstruction* slice : concat_bitcast->operands()) {
        if (position_chunk_pair.first.instruction == slice) {
          auto it = slices_to_chunks.find(slice);
          if (it != slices_to_chunks.end()) {
            return FailedPrecondition(
                "%s", absl::StrCat("Slice ", slice->name(),
                                   " is assigned more than one chunk: ",
                                   it->second.ToString(), " and ",
                                   position_chunk_pair.second.ToString()));
          }
          slices_to_chunks[slice] = position_chunk_pair.second;
        }
      }
    }

    std::vector<const HloInstruction*> sorted_slices =
        SortSlicesInExpectedSpatialOrder(concat_bitcast);
    VLOG(1) << "Chunk assignments for " << sliced_copy_result->name() << ":\n"
            << absl::StrJoin(
                   sorted_slices, "\n",
                   [&](std::string* out, const HloInstruction* slice) {
                     auto it = slices_to_chunks.find(slice);
                     std::string chunk = "no chunk assigned";
                     if (it != slices_to_chunks.end()) {
                       chunk = it->second.ToString();
                     }
                     absl::StrAppend(out, "  slice ", slice->name(), ": ",
                                     chunk);
                   })
            << "\n  sliced copy result " << sliced_copy_result->name() << ": "
            << (result_chunk.has_value() ? result_chunk->ToString()
                                         : "no chunk assigned");
    if (sorted_slices.empty()) {
      return absl::OkStatus();
    }

    // Check that slices are assigned contiguous chunks that are spatially
    // ordered according to sorted_slices. Also make sure that slices are
    // assigned chunks with sizes that match their shape.
    int64_t previous_end = -1;
    int64_t min_offset = std::numeric_limits<int64_t>::max();
    int64_t max_limit = std::numeric_limits<int64_t>::min();
    for (const HloInstruction* slice : sorted_slices) {
      auto it = slices_to_chunks.find(slice);
      if (it == slices_to_chunks.end()) {
        return FailedPrecondition(
            "%s",
            absl::StrCat("Slice ", slice->name(), " is not assigned a chunk"));
      }
      const Chunk& chunk = it->second;

      if (chunk.size != ShapeSize(slice->shape())) {
        return FailedPrecondition(
            "%s",
            absl::StrCat("Slice ", slice->name(), " is assigned chunk ",
                         chunk.ToString(), " with size ", chunk.size,
                         ". Expected a size of ", ShapeSize(slice->shape()),
                         ", to match its shape."));
      }

      if (previous_end != -1 && chunk.offset != previous_end) {
        return FailedPrecondition(
            "%s", absl::StrCat(
                      "Slice ", slice->name(), " starts at offset ",
                      chunk.offset, ". Expected it to start at ", previous_end,
                      " because that's where the previous slice ended."));
      }
      previous_end = chunk.chunk_end();
      min_offset = std::min(min_offset, chunk.offset);
      max_limit = std::max(max_limit, chunk.chunk_end());
    }

    // Check that the sliced copy result is assigned a chunk that is the
    // concatenation of the slice chunks.
    if (!result_chunk.has_value()) {
      return FailedPrecondition(
          "%s", absl::StrCat("Sliced copy result ", sliced_copy_result->name(),
                             " is not assigned a chunk."));
    }
    Chunk expected_result_chunk = Chunk::FromOffsetEnd(min_offset, max_limit);
    if (!(*result_chunk == expected_result_chunk)) {
      return FailedPrecondition(
          "%s", absl::StrCat("Sliced copy result ", sliced_copy_result->name(),
                             " is assigned chunk ", result_chunk->ToString(),
                             ", but it's expected to be assigned chunk ",
                             expected_result_chunk.ToString()));
    }
    if (result_chunk->size != ShapeSize(sliced_copy_result->shape())) {
      return FailedPrecondition(
          "%s", absl::StrCat("Sliced copy result ", sliced_copy_result->name(),
                             " is assigned chunk ", result_chunk->ToString(),
                             " with size ", result_chunk->size,
                             ". Expected a size of ",
                             ShapeSize(sliced_copy_result->shape()),
                             ", to match its shape."));
    }

    return absl::OkStatus();
  }

  SlicedPrefetchTest() {
    // Force tests to fail if ProposeSlices is unexpectedly called.
    EXPECT_CALL(slice_proposer_, ProposeSlices(_, _)).Times(0);

    options_.max_size_in_bytes = 1024;
    options_.sliced_prefetch_options.set_max_slices(2);
    options_.sliced_prefetch_options.set_min_bytes(8);
    options_.propose_slice_fn = [&](const Shape& shape,
                                    const SlicedPrefetchOptions& options) {
      return slice_proposer_.ProposeSlices(shape, options);
    };
    options_.get_equivalent_s8_shape_fn = [](const Shape& original_shape) {
      return ShapeUtil::MakeShape(S8, {ShapeSize(original_shape)});
    };
  }

  // Optional method to setup common ProposeSlices expectations for several
  // tests.
  void SetupProposeSlicesToExpect2SlicesOfF32x8x8() {
    EXPECT_CALL(slice_proposer_,
                ProposeSlices(f32_8_8_, EqualsSlicedPrefetchOptions(
                                            options_.sliced_prefetch_options)))
        .WillRepeatedly(Return(SliceProposalCollection({
            SliceProposal({f32_4_8_, std::vector<SliceParam>({{0, 4}, {0, 8}}),
                           ShapeSize(f32_4_8_)}),
            SliceProposal({f32_4_8_, std::vector<SliceParam>({{4, 8}, {0, 8}}),
                           ShapeSize(f32_4_8_)}),
        })));
  }

  const Shape f32_8_8_ = ShapeUtil::MakeShape(F32, {8, 8});
  const Shape f32_4_8_ = ShapeUtil::MakeShape(F32, {4, 8});
  MockSliceProposer slice_proposer_;
  Options options_ = DefaultMemorySpaceOptions();
};

TEST_F(SlicedPrefetchTest, TwoSlices) {
  std::string hlo_text = R"zz(
HloModule Slice, is_scheduled=true

ENTRY main {
  p0 = f32[8,8] parameter(0)
  p1 = f32[8,8] parameter(1)

  a = f32[8,8] tanh(p0)
  b = f32[8,8] tanh(a)
  c = f32[8,8] tanh(b)

  ROOT r = f32[8,8] add(c, p1)
})zz";

  SetupProposeSlicesToExpect2SlicesOfF32x8x8();

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  VLOG(1) << "Original module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  std::unique_ptr<PresetAssignments> assignments = AssignMemorySpace(
      module.get(), options_,
      /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/1);

  VLOG(1) << "Post-MSA module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  auto root = module->entry_computation()->root_instruction();

  // Expect p1 to be copied via a sliced prefetch for use in r.
  EXPECT_THAT(root, op::Add(_, IsAsyncSlicedCopy(
                                   kAlternateMemorySpace, kDefaultMemorySpace,
                                   {{{0, 4}, {0, 8}}, {{4, 8}, {0, 8}}},
                                   op::Parameter(1))));

  // Check the instruction schedule.
  TF_EXPECT_OK(
      CheckSchedule(*module, root->operand(1),
                    /*slices_start_after_instruction_name=*/"p1",
                    /*slices_done_before_instruction_name=*/"r",
                    /*expect_slices_started_at_different_times=*/true));

  // Check expectations on the chunks assigned to the asynchronous sliced copy.
  TF_EXPECT_OK(CheckSliceChunks(*assignments, root->operand(1)));
}

TEST_F(SlicedPrefetchTest, TwoSlicesWithCopyReplacement) {
  std::string hlo_text = R"zz(
HloModule Slice, is_scheduled=true

ENTRY main {
  p0 = f32[8,8] parameter(0)
  p1 = f32[8,8] parameter(1)

  a = f32[8,8] tanh(p0)
  b = f32[8,8] tanh(a)
  c = f32[8,8] tanh(b)

  p1_copy1 = f32[8,8] copy(p1)
  p1_copy2 = f32[8,8] copy(p1)

  r1 = f32[8,8] add(c, p1_copy1)
  r2 = f32[8,8] add(c, p1_copy2)

  ROOT r = f32[8,8] add(r1, r2)
})zz";
  Options options = options_;
  options.enable_sync_copy_replacement = true;
  SetupProposeSlicesToExpect2SlicesOfF32x8x8();

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  VLOG(1) << "Original module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  std::unique_ptr<PresetAssignments> assignments = AssignMemorySpace(
      module.get(), options,
      /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/1);

  VLOG(1) << "Post-MSA module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());
}

TEST_F(SlicedPrefetchTest, ThreeSlices) {
  std::string hlo_text = R"zz(
HloModule Slice, is_scheduled=true

ENTRY main {
  p0 = f32[8,8] parameter(0)
  p1 = f32[8,8] parameter(1)

  a = f32[8,8] tanh(p0)
  b = f32[8,8] tanh(a)
  c = f32[8,8] tanh(b)

  ROOT r = f32[8,8] add(c, p1)
})zz";
  const Shape f32_3_8 = ShapeUtil::MakeShape(F32, {3, 8});
  const Shape f32_2_8 = ShapeUtil::MakeShape(F32, {2, 8});

  options_.sliced_prefetch_options.set_max_slices(3);

  EXPECT_CALL(slice_proposer_,
              ProposeSlices(f32_8_8_, EqualsSlicedPrefetchOptions(
                                          options_.sliced_prefetch_options)))
      .WillRepeatedly(Return(SliceProposalCollection({
          SliceProposal({f32_3_8, std::vector<SliceParam>({{0, 3}, {0, 8}}),
                         ShapeSize(f32_3_8)}),
          SliceProposal({f32_3_8, std::vector<SliceParam>({{3, 6}, {0, 8}}),
                         ShapeSize(f32_3_8)}),
          SliceProposal({f32_2_8, std::vector<SliceParam>({{6, 8}, {0, 8}}),
                         ShapeSize(f32_2_8)}),
      })));

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  VLOG(1) << "Original module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  std::unique_ptr<PresetAssignments> assignments = AssignMemorySpace(
      module.get(), options_,
      /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/1);

  VLOG(1) << "Post-MSA module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  auto root = module->entry_computation()->root_instruction();

  // Expect p1 to be copied via a sliced prefetch for use in r.
  EXPECT_THAT(
      root,
      op::Add(_, IsAsyncSlicedCopy(
                     kAlternateMemorySpace, kDefaultMemorySpace,
                     {{{0, 3}, {0, 8}}, {{3, 6}, {0, 8}}, {{6, 8}, {0, 8}}},
                     op::Parameter(1))));

  // Check the instruction schedule.
  TF_EXPECT_OK(
      CheckSchedule(*module, root->operand(1),
                    /*slices_start_after_instruction_name=*/"p1",
                    /*slices_done_before_instruction_name=*/"r",
                    /*expect_slices_started_at_different_times=*/true));

  // Check expectations on the chunks assigned to the asynchronous sliced copy.
  TF_EXPECT_OK(CheckSliceChunks(*assignments, root->operand(1)));
}

TEST_F(SlicedPrefetchTest, SlicingDisabled) {
  std::string hlo_text = R"zz(
HloModule Slice, is_scheduled=true

ENTRY main {
  p0 = f32[8,8] parameter(0)
  p1 = f32[8,8] parameter(1)

  a = f32[8,8] tanh(p0)
  b = f32[8,8] tanh(a)
  c = f32[8,8] tanh(b)

  ROOT r = f32[8,8] add(c, p1)
})zz";

  options_.sliced_prefetch_options.set_max_slices(0);

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  VLOG(1) << "Original module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  std::unique_ptr<PresetAssignments> assignments = AssignMemorySpace(
      module.get(), options_,
      /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/1);

  VLOG(1) << "Post-MSA module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  // Check that there are not any sliced prefetches in the schedule.
  auto entry_schedule =
      module->schedule().sequence(module->entry_computation()).instructions();
  for (const HloInstruction* instruction : entry_schedule) {
    EXPECT_FALSE(IsAsyncSliceStart(instruction));
    EXPECT_FALSE(IsAsyncSliceDone(instruction));
    EXPECT_FALSE(IsConcatBitcast(instruction));
  }
}

TEST_F(SlicedPrefetchTest, TooSmallToSlice) {
  std::string hlo_text = R"zz(
HloModule Slice, is_scheduled=true

ENTRY main {
  p0 = f32[8,8] parameter(0)
  p1 = f32[8,8] parameter(1)

  a = f32[8,8] tanh(p0)
  b = f32[8,8] tanh(a)
  c = f32[8,8] tanh(b)

  ROOT r = f32[8,8] add(c, p1)
})zz";

  options_.sliced_prefetch_options.set_min_bytes(1000000000);

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  VLOG(1) << "Original module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  std::unique_ptr<PresetAssignments> assignments = AssignMemorySpace(
      module.get(), options_,
      /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/1);

  VLOG(1) << "Post-MSA module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  // No tensor is big enough to be sliced, so check that there are not any
  // sliced prefetches.
  auto entry_schedule =
      module->schedule().sequence(module->entry_computation()).instructions();
  for (const HloInstruction* instruction : entry_schedule) {
    EXPECT_FALSE(IsAsyncSliceStart(instruction));
    EXPECT_FALSE(IsAsyncSliceDone(instruction));
    EXPECT_FALSE(IsConcatBitcast(instruction));
  }
}

TEST_F(SlicedPrefetchTest, FallbackToUnsliced) {
  std::string hlo_text = R"zz(
HloModule Slice, is_scheduled=true

ENTRY main {
  p0 = f32[8,8] parameter(0)
  p1 = f32[8,8] parameter(1)

  a = f32[8,8] tanh(p0)
  b = f32[8,8] tanh(a)
  c = f32[8,8] tanh(b)

  ROOT r = f32[8,8] add(c, p1)
})zz";

  EXPECT_CALL(slice_proposer_,
              ProposeSlices(f32_8_8_, EqualsSlicedPrefetchOptions(
                                          options_.sliced_prefetch_options)))
      .WillRepeatedly(Return(absl::StatusOr<SliceProposalCollection>(
          FailedPrecondition("%s", "Cannot slice."))));

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  VLOG(1) << "Original module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  std::unique_ptr<PresetAssignments> assignments = AssignMemorySpace(
      module.get(), options_,
      /*max_prefetch_interval=*/10, /*min_prefetch_interval=*/1);

  VLOG(1) << "Post-MSA module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  // No tensor is big enough to be sliced, so check that there are not any
  // sliced prefetches.
  auto entry_schedule =
      module->schedule().sequence(module->entry_computation()).instructions();
  for (const HloInstruction* instruction : entry_schedule) {
    EXPECT_FALSE(IsAsyncSliceStart(instruction));
    EXPECT_FALSE(IsAsyncSliceDone(instruction));
    EXPECT_FALSE(IsConcatBitcast(instruction));
  }
}

TEST_F(SlicedPrefetchTest, UsingCostAnalysisIntervalPicker) {
  std::string hlo_text = R"zz(
HloModule Slice, is_scheduled=true

ENTRY main {
  p0 = f32[8,8] parameter(0)
  p1 = f32[8,8] parameter(1)

  a = f32[8,8] tanh(p0)
  b = f32[8,8] tanh(a)
  c = f32[8,8] tanh(b)

  ROOT r = f32[8,8] add(c, p1)
})zz";

  SetupProposeSlicesToExpect2SlicesOfF32x8x8();

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  VLOG(1) << "Original module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  std::unique_ptr<PresetAssignments> assignments =
      AssignMemorySpaceUsingCostAnalysis(
          module.get(), /*memory_space_options_override=*/options_);

  VLOG(1) << "Post-MSA module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  auto root = module->entry_computation()->root_instruction();

  // Expect p1 to be copied via a sliced prefetch for use in r.
  EXPECT_THAT(root, op::Add(_, IsAsyncSlicedCopy(
                                   kAlternateMemorySpace, kDefaultMemorySpace,
                                   {{{0, 4}, {0, 8}}, {{4, 8}, {0, 8}}},
                                   op::Parameter(1))));

  // Check the instruction schedule.
  TF_EXPECT_OK(CheckSchedule(
      *module, root->operand(1),
      // The CostAnalysisPrefetchIntervalPicker does not necessarily pick the
      // earliest possible time to start the prefetch.
      /*slices_start_after_instruction_name=*/"a",
      /*slices_done_before_instruction_name=*/"r",
      /*expect_slices_started_at_different_times=*/true));

  // Check expectations on the chunks assigned to the asynchronous sliced copy.
  TF_EXPECT_OK(CheckSliceChunks(*assignments, root->operand(1)));
}

TEST_F(SlicedPrefetchTest, LoopAliasing) {
  std::string hlo_text = R"zz(
HloModule Slice, is_scheduled=true

WhileBody {
  body_param = (f32[8,8], f32[8,8], f32[], f32[]) parameter(0)
  v0 = f32[8,8] get-tuple-element(body_param), index=0
  v1 = f32[8,8] get-tuple-element(body_param), index=1
  i = f32[] get-tuple-element(body_param), index=2
  limit = f32[] get-tuple-element(body_param), index=3
  one = f32[] constant(1)

  new_i = f32[] add(i, one)
  new_v1 = f32[8,8] add(v0, v1)

  ROOT while_result = (f32[8,8], f32[8,8], f32[], f32[]) tuple(v0, new_v1, new_i, limit)
}

WhileCond {
  cond_param = (f32[8,8], f32[8,8], f32[], f32[]) parameter(0)
  i = f32[] get-tuple-element(cond_param), index=2
  limit = f32[] get-tuple-element(cond_param), index=3

  ROOT cond_result = pred[] compare(i, limit), direction=LT
}

ENTRY main {
  p0 = f32[8,8] parameter(0)
  p1 = f32[8,8] parameter(1)
  iterations = f32[] parameter(2)
  initial = f32[] constant(0)

  a = f32[8,8] tanh(p0)
  b = f32[8,8] tanh(a)
  c = f32[8,8] tanh(b)

  t = (f32[8,8], f32[8,8], f32[], f32[]) tuple(p0, p1, initial, iterations)
  w = (f32[8,8], f32[8,8], f32[], f32[]) while(t), condition=WhileCond, body=WhileBody
  d = f32[8,8] get-tuple-element(w), index=1

  ROOT r = f32[8,8] add(c, d)
})zz";

  SetupProposeSlicesToExpect2SlicesOfF32x8x8();

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  VLOG(1) << "Original module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  std::unique_ptr<PresetAssignments> assignments =
      AssignMemorySpaceUsingCostAnalysis(
          module.get(), /*memory_space_options_override=*/options_);

  VLOG(1) << "Post-MSA module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  auto root = module->entry_computation()->root_instruction();

  // Expect p1 to be copied with a slice.
  ASSERT_THAT(
      root,  //
      // r from module
      op::Add(_,
              // d from module
              op::GetTupleElement(
                  // w from module
                  op::While(
                      // t from module
                      op::Tuple(_,
                                IsAsyncSlicedCopy(
                                    kAlternateMemorySpace, kDefaultMemorySpace,
                                    {{{0, 4}, {0, 8}}, {{4, 8}, {0, 8}}},
                                    op::Parameter(1)),
                                _, _)),
                  1)));

  // In the resultant code, ensure that the prefetch of p1 is aliased throughout
  // the while loop.
  HloInstruction* w = root->mutable_operand(1)->mutable_operand(0);
  HloInstruction* t = w->mutable_operand(0);
  HloInstruction* concat_bitcast = t->mutable_operand(1);
  HloComputation* while_body = w->while_body();
  HloInstruction* body_param = while_body->parameter_instruction(0);
  HloComputation* while_cond = w->while_condition();
  HloInstruction* cond_param = while_cond->parameter_instruction(0);

  // Things we expect to alias with the concat_bitcast.
  absl::flat_hash_set<HloPosition> expected_aliases({
      HloPosition{concat_bitcast, {}},
      HloPosition{w, {1}},
      HloPosition{t, {1}},
      HloPosition{body_param, {1}},
      HloPosition{cond_param, {1}},
  });

  // Check the aliasing.
  auto alias_analysis = HloAliasAnalysis::Run(module.get()).value();
  VLOG(2) << alias_analysis->ToString();
  const HloBuffer& concat_bitcast_buffer =
      alias_analysis->GetUniqueBufferAt(concat_bitcast);
  EXPECT_THAT(concat_bitcast_buffer.ComputePositions(),
              ::testing::IsSupersetOf(expected_aliases));

  // Since expected_aliases are aliased, we expect only 1 to be assigned a
  // chunk.
  int num_chunks_for_expected_aliases = 0;
  for (const auto& position_chunk_pair : assignments->chunks()) {
    if (expected_aliases.contains(position_chunk_pair.first)) {
      num_chunks_for_expected_aliases++;
    }
  }
  EXPECT_EQ(num_chunks_for_expected_aliases, 1);
}

class MockRepacker : public MemorySpaceAssignmentRepacker {
 public:
  MockRepacker()
      : MemorySpaceAssignmentRepacker(std::numeric_limits<int64_t>::max(), 1) {}

  MOCK_METHOD(absl::StatusOr<bool>, Repack, (absl::Span<AllocationBlock*>),
              (override));
};

// Here, we test the following things:
// - Without repacking, we are unable to prefetch p4.
// - With repacking, we are able to prefetch p4.
// - When repacking occurs, we expect p2 and p3 to have been allocated chunks.
//   We are only proposing slices for f32[32, 16] and not f32[16,16]; thus, we
//   expect slicing metadata to be attached to the repacking block for p2 but
//   not p3.
// - We make the repacker assign the first slice (in time) of p2 the larger
//   offset. After MSA, we check to make sure the fist slice is using the
//   larger slicing parameters
TEST_F(SlicedPrefetchTest, Repack) {
  absl::string_view hlo_string = R"(
HloModule Slice, is_scheduled=true

ENTRY main {
  /* parameters */
  p0 = f32[] parameter(0)
  p1 = f32[16,16] parameter(1)
  p2 = f32[32,16] parameter(2)
  p3 = f32[16,16] parameter(3)
  p4 = f32[32,16] parameter(4)

  /* filler that we can prefetch over */
  x1 = f32[] add(p0,p0)
  x2 = f32[] add(x1, x1)

  /* uses of p1 and p3 */
  a = f32[16,16] sine(p1)
  c = f32[16,16] sine(p3)

  /* more filler, giving us time to prefetch p4, when repacking */
  x3 = f32[] add(x2, x2)
  x4 = f32[] add(x3, x3)

  /* uses of p2 and p4 */
  b = f32[32,16] sine(p2)
  d = f32[32,16] sine(p4)

  /* make sure that x4, a, b, c, d are not dead code */
  z1 = f32[16,16] broadcast(x4), dimensions={}
  z2 = f32[16,16] add(z1, a)
  z3 = f32[32,16] concatenate(z2, c), dimensions={0}
  z4 = f32[32,16] add(z3, b)
  ROOT z5 = f32[32,16] add(z4, d)
})";

  // Create 2 copies of the module, one to run without repacking and one to run
  // with repacking.
  TF_ASSERT_OK_AND_ASSIGN(auto module_no_repacking,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(auto module_with_repacking,
                          ParseAndReturnVerifiedModule(hlo_string));
  VLOG(1) << "Original module:\n"
          << module_no_repacking->ToString(HloPrintOptions::ShortParsable());

  // Setup slicing expectations so that we slice f32[32, 16], but not
  // f32[16,16].
  Shape f32_16_16 = ShapeUtil::MakeShape(F32, {16, 16});
  Shape f32_32_16 = ShapeUtil::MakeShape(F32, {32, 16});
  EXPECT_CALL(slice_proposer_,
              ProposeSlices(f32_16_16, EqualsSlicedPrefetchOptions(
                                           options_.sliced_prefetch_options)))
      .WillRepeatedly(Return(SliceProposalCollection({})));
  EXPECT_CALL(slice_proposer_,
              ProposeSlices(f32_32_16, EqualsSlicedPrefetchOptions(
                                           options_.sliced_prefetch_options)))
      .WillRepeatedly(Return(SliceProposalCollection({
          SliceProposal({f32_16_16, std::vector<SliceParam>({{0, 16}, {0, 16}}),
                         ShapeSize(f32_16_16)}),
          SliceProposal({f32_16_16,
                         std::vector<SliceParam>({{16, 32}, {0, 16}}),
                         ShapeSize(f32_16_16)}),
      })));

  // Force MSA to prefer prefetching (in order) p1, p2, p3, p4, and then
  // anything else.
  MsaBufferIntervalCompare buffer_interval_compare =
      [](const MsaBufferInterval& lhs, const MsaBufferInterval& rhs) {
        auto lookup = [](const MsaBufferInterval& x) {
          // An arbitrary value that is greater than that for p1, p2, p3, and
          // p4.
          int priority = 100;
          if (x.buffer->instruction()->name() == "p1") {
            priority = 1;
          } else if (x.buffer->instruction()->name() == "p2") {
            priority = 2;
          } else if (x.buffer->instruction()->name() == "p3") {
            priority = 3;
          } else if (x.buffer->instruction()->name() == "p4") {
            priority = 4;
          }
          return std::make_tuple(priority, x.buffer->instruction()->name());
        };

        return lookup(lhs) < lookup(rhs);
      };

  // Configure MSA.
  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(2, 50);
  options_.max_size_in_bytes = 4 * 1024;
  options_.max_repacks = 0;

  // Run MSA without repacking
  std::unique_ptr<PresetAssignments> assignments =
      AssignMemorySpace(module_no_repacking.get(), options_,
                        buffer_interval_compare, &prefetch_interval_picker);
  VLOG(1) << "Post-MSA module (no repacking):\n"
          << module_no_repacking->ToString(HloPrintOptions::ShortParsable());

  // If repacking is disabled, p4 (the source of d) should not be prefetched.
  const HloInstruction* d =
      FindNamedScheduledInstruction(*module_no_repacking, "d");
  ASSERT_NE(d, nullptr);
  EXPECT_FALSE(IsConcatBitcast(d->operand(0)));

  // Configure MSA to repack.
  MockRepacker repacker;
  absl::flat_hash_map<std::pair<int64_t, int64_t>, int64_t> repack_map;
  EXPECT_CALL(repacker, Repack(_))
      .WillRepeatedly([](absl::Span<AllocationBlock*> allocations)
                          -> absl::StatusOr<bool> {
        bool found_p2 = false;
        bool found_p3 = false;
        for (AllocationBlock* block : allocations) {
          VLOG(1) << "Allocation block: " << block->ToString();

          if (block->inclusive_start_time == 3 &&
              block->initial_offset == 1024 && block->size == 2048) {
            // Move "p2" from offset 1024 -> 2048.
            found_p2 = true;
            block->offset = 2048;
            // We expect p2 to be sliced. Check that it has slicing information
            // in its AllocationBlock.
            EXPECT_TRUE(block->original_slice_data.has_value());
            if (block->original_slice_data.has_value()) {
              SlicedAllocationData expected(
                  {{AllocatedSlice{1024, 1024, /*inclusive_start_time=*/3},
                    AllocatedSlice{1024, 2048, /*inclusive_start_time=*/7}}});
              EXPECT_EQ(*block->original_slice_data, expected)
                  << "\nExpected: " << expected.ToString()
                  << "\nGot: " << block->original_slice_data->ToString();
              // Set the first slice for p2 to be place at the larger offset.
              block->repacked_slice_data = SlicedAllocationData(
                  {{AllocatedSlice{1024, 2048, /*inclusive_start_time=*/7},
                    AllocatedSlice{1024, 3072, /*inclusive_start_time=*/3}}});
            }
          } else if (block->inclusive_start_time == 4 &&
                     block->initial_offset == 3072 && block->size == 1024) {
            // Move "p3" from offset 3072 -> 1024.
            found_p3 = true;
            block->offset = 1024;
            // We do not expect p3 to be sliced. Thus, it should not have
            // slicing information in its AllocationBlock.
            EXPECT_FALSE(block->original_slice_data.has_value());
          } else {
            block->offset = block->initial_offset;
          }
        }

        EXPECT_TRUE(found_p2);
        EXPECT_TRUE(found_p3);

        return true;
      });
  options_.max_repacks = 1;
  options_.repacker = &repacker;
  assignments =
      AssignMemorySpace(module_with_repacking.get(), options_,
                        buffer_interval_compare, &prefetch_interval_picker);
  VLOG(1) << "Post-MSA module (with repacking):\n"
          << module_with_repacking->ToString(HloPrintOptions::ShortParsable());

  // If repacking is enabled, p4 (the source of d) should be prefetched.
  d = FindNamedScheduledInstruction(*module_with_repacking, "d");
  ASSERT_NE(d, nullptr);
  EXPECT_TRUE(IsConcatBitcast(d->operand(0)));

  // Check expectations on the chunks assigned to the asynchronous sliced copy.
  // In particular, we want to make sure the slices are still contiguous.
  TF_EXPECT_OK(CheckSliceChunks(*assignments, d->operand(0)));

  // Find the slices and offsets for p2, in the order they start in the
  // schedule.
  std::vector<const HloInstruction*> p2_slice_dones;
  for (const HloInstruction* i :
       module_with_repacking->entry_computation()->instructions()) {
    if (IsAsyncSliceStart(i) && i->operand_count() == 1 &&
        i->operand(0)->name() == "p2") {
      ASSERT_EQ(i->user_count(), 1);
      p2_slice_dones.push_back(i->users()[0]);
    }
  }
  ASSERT_EQ(p2_slice_dones.size(), 2);
  std::vector<int64_t> p2_slice_offsets;
  for (const HloInstruction* i : p2_slice_dones) {
    for (const std::pair<HloPosition, Chunk>& position_chunk_pair :
         assignments->chunks()) {
      if (position_chunk_pair.first.instruction == i) {
        p2_slice_offsets.push_back(position_chunk_pair.second.offset);
      }
    }
  }
  ASSERT_EQ(p2_slice_offsets.size(), 2);

  // We expect the first slice of p2 to have the larger offsets.
  EXPECT_THAT(p2_slice_dones[0]->async_wrapped_instruction()->slice_starts(),
              ::testing::ElementsAreArray({16, 0}));
  EXPECT_THAT(p2_slice_dones[0]->async_wrapped_instruction()->slice_limits(),
              ::testing::ElementsAreArray({32, 16}));
  EXPECT_EQ(p2_slice_offsets[0], 3072);
  EXPECT_THAT(p2_slice_dones[1]->async_wrapped_instruction()->slice_starts(),
              ::testing::ElementsAreArray({0, 0}));
  EXPECT_THAT(p2_slice_dones[1]->async_wrapped_instruction()->slice_limits(),
              ::testing::ElementsAreArray({16, 16}));
  EXPECT_EQ(p2_slice_offsets[1], 2048);
}

struct ModuleAndAssignments {
  std::unique_ptr<VerifiedHloModule> module;
  std::unique_ptr<PresetAssignments> assignments;
};

// In this test, we ensure that sliced prefetching does not attempt to start a
// slice during a different computation than the one where the slice finishes.
// We do this by forcing a sliced prefetch to start just before back-to-back
// while loops and to immediately finish after them. We use while loops with
// different expected elapse times, so that the ideal place to start the second
// slice is during one of the while loops.
TEST_F(SlicedPrefetchTest, BackToBackWhileLoops) {
  // Define constants for building our test HLO.
  const std::string while_cond = R"zz(
WhileCond$ID {
  cond_param = (f32[8,8], f32[8,8], f32[], f32[]) parameter(0)
  i = f32[] get-tuple-element(cond_param), index=2
  limit = f32[] get-tuple-element(cond_param), index=3

  ROOT cond_result = pred[] compare(i, limit), direction=LT
})zz";

  const std::string while_body = R"zz(
WhileBody$ID {
  body_param = (f32[8,8], f32[8,8], f32[], f32[]) parameter(0)
  v0 = f32[8,8] get-tuple-element(body_param), index=0
  v1 = f32[8,8] get-tuple-element(body_param), index=1
  i = f32[] get-tuple-element(body_param), index=2
  limit = f32[] get-tuple-element(body_param), index=3
  one = f32[] constant(1)

  new_i = f32[] add(i, one)
  $COMPUTATION

  ROOT while_result = (f32[8,8], f32[8,8], f32[], f32[]) tuple(v0, new_v1, new_i, limit)
})zz";

  const std::string while_computation_cheap = R"zz(
  new_v1 = f32[8,8] add(v0, v1))zz";

  std::string while_computation_expensive = R"zz(
  new_v1_0 = f32[8,8] add(v0, v1)
  new_v1_1 = f32[8,8] tanh(new_v1_0)
  new_v1_2 = f32[8,8] tanh(new_v1_1)
  new_v1_3 = f32[8,8] tanh(new_v1_2)
  new_v1 = f32[8,8] tanh(new_v1_3))zz";

  std::string module_text = R"zz(
HloModule Slice, is_scheduled=true

$WHILEBODY1
$WHILECOND1
$WHILEBODY2
$WHILECOND2

ENTRY main {
  loop1_input1 = f32[8,8] parameter(0)
  loop1_input2 = f32[8,8] parameter(1)
  loop1_iterations = f32[] parameter(2)
  loop1_begin = f32[] constant(0)
  loop1_tuple = (f32[8,8], f32[8,8], f32[], f32[]) tuple(loop1_input1, loop1_input2, loop1_iterations, loop1_begin)
  loop2_input1 = f32[8,8] parameter(3)
  loop2_input2 = f32[8,8] parameter(4)
  loop2_iterations = f32[] parameter(5)
  loop2_begin = f32[] constant(0)
  loop2_tuple = (f32[8,8], f32[8,8], f32[], f32[]) tuple(loop2_input1, loop2_input2, loop2_iterations, loop2_begin)

  prefetch = f32[8,8] parameter(6)
  loop1_output = (f32[8,8], f32[8,8], f32[], f32[]) while(loop1_tuple), condition=WhileCond1, body=WhileBody1
  loop2_output = (f32[8,8], f32[8,8], f32[], f32[]) while(loop2_tuple), condition=WhileCond2, body=WhileBody2
  prefetch_use = f32[8,8] tanh(prefetch)

  loop1_result = f32[8,8] get-tuple-element(loop1_output), index=1
  loop2_result = f32[8,8] get-tuple-element(loop2_output), index=1

  tmp1 = f32[8,8] add(loop1_result, loop2_result)
  ROOT r = f32[8,8] add(tmp1, prefetch_use)
})zz";

  // A lambda for generating HLO with 2 while loops called back to back. The
  // first while loop will execute while_computation1 and the second while loop
  // will execute while_computation2.
  auto gen_hlo = [&](absl::string_view while_computation1,
                     absl::string_view while_computation2) {
    return absl::StrReplaceAll(
        module_text,
        {
            {"$WHILEBODY1",
             absl::StrReplaceAll(
                 while_body,
                 {{"$ID", "1"}, {"$COMPUTATION", while_computation1}})},
            {"$WHILECOND1", absl::StrReplaceAll(while_cond, {{"$ID", "1"}})},
            {"$WHILEBODY2",
             absl::StrReplaceAll(
                 while_body,
                 {{"$ID", "2"}, {"$COMPUTATION", while_computation2}})},
            {"$WHILECOND2", absl::StrReplaceAll(while_cond, {{"$ID", "2"}})},
        });
  };

  // Configure MSA.
  SetupProposeSlicesToExpect2SlicesOfF32x8x8();
  // Force MSA to prefer prefetching 'prefetch'.
  MsaBufferIntervalCompare buffer_interval_compare =
      [](const MsaBufferInterval& lhs, const MsaBufferInterval& rhs) {
        auto lookup = [](const MsaBufferInterval& x) {
          // An arbitrary value that is greater than that used for 'prefetch'.
          int priority = 100;
          if (x.buffer->instruction()->name() == "prefetch") {
            priority = 1;
          }
          return std::make_tuple(priority, x.buffer->instruction()->name());
        };

        return lookup(lhs) < lookup(rhs);
      };
  // We set the minimum prefetch interval to a large enough value (32) to force
  // us to prefetch around both while loops, and not just 1.
  InstructionCountPrefetchIntervalPicker prefetch_interval_picker(32, 100);
  options_.max_size_in_bytes = 4 * 64;

  // Define a lambda for running MSA on the specified HLO, with the
  // configuration above.
  auto run_msa =
      [&](absl::string_view hlo_text) -> absl::StatusOr<ModuleAndAssignments> {
    ModuleAndAssignments module_and_assignments;
    TF_ASSIGN_OR_RETURN(module_and_assignments.module,
                        ParseAndReturnVerifiedModule(hlo_text));
    VLOG(1) << "Original module:\n"
            << module_and_assignments.module->ToString(
                   HloPrintOptions::ShortParsable());
    module_and_assignments.assignments =
        AssignMemorySpace(module_and_assignments.module.get(), options_,
                          buffer_interval_compare, &prefetch_interval_picker);
    VLOG(1) << "Post-MSA module:\n"
            << module_and_assignments.module->ToString(
                   HloPrintOptions::ShortParsable());
    return module_and_assignments;
  };

  // In this case, less time elapses during the first while loop than the
  // second. Make sure we start the second slice between the two while loops,
  // rather than during the second while loop.
  TF_ASSERT_OK_AND_ASSIGN(
      ModuleAndAssignments module_and_assignments1,
      run_msa(gen_hlo(while_computation_cheap, while_computation_expensive)));
  auto root1 =
      module_and_assignments1.module->entry_computation()->root_instruction();
  EXPECT_THAT(root1, op::Add(_, op::Tanh(IsAsyncSlicedCopy(
                                    kAlternateMemorySpace, kDefaultMemorySpace,
                                    {{{0, 4}, {0, 8}}, {{4, 8}, {0, 8}}},
                                    op::Parameter(6)))));
  TF_EXPECT_OK(CheckSchedule(
      *module_and_assignments1.module, root1->operand(1)->operand(0),
      /*slices_start_after_instruction_name=*/"prefetch",
      /*slices_done_before_instruction_name=*/"prefetch_use",
      /*expect_slices_started_at_different_times=*/true));
  auto entry_schedule1 =
      module_and_assignments1.module->schedule()
          .sequence(module_and_assignments1.module->entry_computation())
          .instructions();
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<int> start_indicies,
      GetSliceStartIndicies(entry_schedule1, root1->operand(1)->operand(0)));
  ASSERT_EQ(start_indicies.size(), 2);
  TF_ASSERT_OK_AND_ASSIGN(
      int first_while,
      FindScheduleIndexOfInstruction(
          entry_schedule1, "loop1_output",
          SlicedPrefetchTest::InstructionClass::kUnrelatedNonCopy));
  TF_ASSERT_OK_AND_ASSIGN(
      int second_while,
      FindScheduleIndexOfInstruction(
          entry_schedule1, "loop2_output",
          SlicedPrefetchTest::InstructionClass::kUnrelatedNonCopy));
  EXPECT_TRUE(
      absl::c_is_sorted<std::vector<int>>(
          {start_indicies[0], first_while, start_indicies[1], second_while}) ||
      absl::c_is_sorted<std::vector<int>>(
          {start_indicies[1], first_while, start_indicies[0], second_while}));

  // In this case, more time elapses during the first while loop than the
  // second. This should push us to use a normal prefetch, rather than slicing,
  // since the ideal time to start the second slice will get pushed before
  // both while loops.
  TF_ASSERT_OK_AND_ASSIGN(
      ModuleAndAssignments module_and_assignments2,
      run_msa(gen_hlo(while_computation_expensive, while_computation_cheap)));
  auto root2 =
      module_and_assignments2.module->entry_computation()->root_instruction();
  EXPECT_THAT(root2, op::Add(_, op::Tanh(op::AsyncCopy(kAlternateMemorySpace,
                                                       kDefaultMemorySpace,
                                                       op::Parameter(6)))));
  auto entry_schedule2 =
      module_and_assignments2.module->schedule()
          .sequence(module_and_assignments2.module->entry_computation())
          .instructions();
  TF_ASSERT_OK_AND_ASSIGN(
      int copy_done,
      FindScheduleIndexOfInstruction(
          entry_schedule2, root2->operand(1)->operand(0)->name(),
          SlicedPrefetchTest::InstructionClass::kUnrelatedNonCopy));
  TF_ASSERT_OK_AND_ASSIGN(
      int copy_start,
      FindScheduleIndexOfInstruction(
          entry_schedule2, root2->operand(1)->operand(0)->operand(0)->name(),
          SlicedPrefetchTest::InstructionClass::kUnrelatedNonCopy));
  TF_ASSERT_OK_AND_ASSIGN(
      first_while,
      FindScheduleIndexOfInstruction(
          entry_schedule2, "loop1_output",
          SlicedPrefetchTest::InstructionClass::kUnrelatedNonCopy));
  TF_ASSERT_OK_AND_ASSIGN(
      second_while,
      FindScheduleIndexOfInstruction(
          entry_schedule2, "loop2_output",
          SlicedPrefetchTest::InstructionClass::kUnrelatedNonCopy));
  EXPECT_TRUE(absl::c_is_sorted<std::vector<int>>(
      {copy_start, first_while, second_while, copy_done}));
}

using RepackingTest = ::testing::Test;

TEST_F(RepackingTest, Colocations) {
  AllocationBlock a{10, 20, 100, 0, 1000, 0};
  AllocationBlock b{15, 25, 150, 0, 2000, 1};
  AllocationBlock c{18, 22, 50, 0, 500, 2};
  AllocationBlock d{5, 9, 20, 0, 3000, 3};
  AllocationBlock e{17, 22, 100, 0, 1500, 4};
  AllocationBlock f{25, 27, 150, 0, 2500, 5};

  // a doesn't have other colocations.
  a.next_colocated = &a;
  // b and c are colocated.
  b.next_colocated = &c;
  c.next_colocated = &b;
  // d, e, and f are colocated.
  d.next_colocated = &f;
  e.next_colocated = &d;
  f.next_colocated = &e;

  EXPECT_EQ(a.GetColocationsCount(), 1);
  EXPECT_THAT(a.GetColocations(), UnorderedElementsAre(&a));
  EXPECT_EQ(b.GetColocationsCount(), 2);
  EXPECT_THAT(b.GetColocations(), UnorderedElementsAre(&b, &c));
  EXPECT_EQ(c.GetColocationsCount(), 2);
  EXPECT_THAT(c.GetColocations(), UnorderedElementsAre(&b, &c));
  EXPECT_EQ(d.GetColocationsCount(), 3);
  EXPECT_THAT(d.GetColocations(), UnorderedElementsAre(&d, &e, &f));
  EXPECT_EQ(e.GetColocationsCount(), 3);
  EXPECT_THAT(e.GetColocations(), UnorderedElementsAre(&d, &e, &f));
  EXPECT_EQ(f.GetColocationsCount(), 3);
  EXPECT_THAT(f.GetColocations(), UnorderedElementsAre(&d, &e, &f));
}

TEST_F(SlicedPrefetchTest, UniformSizedSlicing) {
  std::string hlo_text = R"zz(
HloModule Slice, is_scheduled=true

ENTRY main {
  p0 = f32[8,8] parameter(0)
  p1 = f32[8,8] parameter(1)
  p2 = f32[8,16] parameter(2)
  constant1 = f32[] constant(1.1)

  a = f32[8,8] tanh(p0)
  b = f32[8,8] tanh(a)
  c = f32[8,8] tanh(b)
  d = f32[8,8] tanh(c)
  e = f32[8,8] tanh(d)
  f = f32[8,8] tanh(e)
  g = f32[8,8] tanh(f)
  h = f32[8,8] tanh(g)

  x = f32[8,8] add(p1, h)
  padded_x = f32[8,16] pad(x, constant1), padding=0_0x0_8
  ROOT r = f32[8,16] add(padded_x, p2)
})zz";
  const Shape f32_8_16 = ShapeUtil::MakeShape(F32, {8, 16});
  const Shape s8_128 = ShapeUtil::MakeShape(S8, {128});

  options_.sliced_prefetch_options.set_max_slices(100000);
  options_.sliced_prefetch_options.set_preferred_slice_size(4 * 8 * 4);

  EXPECT_CALL(slice_proposer_,
              ProposeSlices(f32_8_8_, EqualsSlicedPrefetchOptions(
                                          options_.sliced_prefetch_options)))
      .WillRepeatedly(Return(SliceProposalCollection({
          SliceProposal(
              {s8_128, std::vector<SliceParam>({{0, 128}}), ShapeSize(s8_128)}),
          SliceProposal({s8_128, std::vector<SliceParam>({{128, 256}}),
                         ShapeSize(s8_128)}),
      })));

  EXPECT_CALL(slice_proposer_,
              ProposeSlices(f32_8_16, EqualsSlicedPrefetchOptions(
                                          options_.sliced_prefetch_options)))
      .WillRepeatedly(Return(SliceProposalCollection({
          SliceProposal(
              {s8_128, std::vector<SliceParam>({{0, 128}}), ShapeSize(s8_128)}),
          SliceProposal({s8_128, std::vector<SliceParam>({{128, 256}}),
                         ShapeSize(s8_128)}),
          SliceProposal({s8_128, std::vector<SliceParam>({{256, 384}}),
                         ShapeSize(s8_128)}),
          SliceProposal({s8_128, std::vector<SliceParam>({{384, 512}}),
                         ShapeSize(s8_128)}),
      })));

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  VLOG(1) << "Original module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  std::unique_ptr<PresetAssignments> assignments = AssignMemorySpace(
      module.get(), options_,
      /*max_prefetch_interval=*/100, /*min_prefetch_interval=*/1);

  VLOG(1) << "Post-MSA module:\n"
          << module->ToString(HloPrintOptions::ShortParsable());

  auto root = module->entry_computation()->root_instruction();

  // Expect p1 to be asynchronously copied via 2 slices, and p2 to be
  // asynchronously copied via 4 slices. We expect p1 and p2 to be bitcast
  // before slicing and after slicing.
  EXPECT_THAT(
      root,
      op::Add(op::Pad(op::Add(IsAsyncSlicedCopy(
                                  kAlternateMemorySpace, kDefaultMemorySpace,
                                  {{{0, 128}}, {{128, 256}}}, op::Parameter(1),
                                  /*expect_bitcasted_io=*/true),
                              /*don't care*/ _),
                      /*padding constant*/ _),
              IsAsyncSlicedCopy(
                  kAlternateMemorySpace, kDefaultMemorySpace,
                  {{{0, 128}}, {{128, 256}}, {{256, 384}}, {{384, 512}}},
                  op::Parameter(2), /*expect_bitcasted_io=*/true)));

  // Check expectations on the chunks assigned to the asynchronous sliced copy.
  TF_EXPECT_OK(CheckSliceChunks(*assignments, root->operand(1),
                                /*expect_bitcasted_io=*/true));
  TF_EXPECT_OK(CheckSliceChunks(*assignments,
                                root->operand(0)->operand(0)->operand(0),
                                /*expect_bitcasted_io=*/true));
}

}  // namespace
}  // namespace memory_space_assignment
}  // namespace xla

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/memory_space_assignment.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

constexpr int64 kPointerSize = 8;
constexpr float kAsyncCopyBandwidth = 100;
constexpr float kAlternateMemBandwidth = 1000;
constexpr float kBytesPerSecond = 100;
constexpr float kFlopsPerSecond = 1000;
constexpr float kTranscendentalsPerSecond = 10;

int64 ShapeSize(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, kPointerSize);
}

class MemorySpaceAssignmentTest : public HloTestBase,
                                  public ::testing::WithParamInterface<bool> {
 protected:
  // We use the following two memory space values to describe the default (slow
  // and large) and alternate (fast and small) memory spaces.
  const int64 kDefaultMemorySpace = 0;
  const int64 kAlternateMemorySpace = 1;

  std::unique_ptr<PresetAssignments> AssignMemorySpaceUsingCostAnalysis(
      HloModule* module) {
    HloCostAnalysis hlo_cost_analysis(ShapeSize);
    hlo_cost_analysis.set_flops_per_second(kFlopsPerSecond);
    hlo_cost_analysis.set_bytes_per_second(kBytesPerSecond);
    hlo_cost_analysis.set_transcendentals_per_second(kTranscendentalsPerSecond);
    for (HloComputation* computation : module->MakeNonfusionComputations()) {
      TF_CHECK_OK(computation->Accept(&hlo_cost_analysis));
    }
    MemorySpaceAssignmentCostAnalysis cost_analysis(
        hlo_cost_analysis, kAsyncCopyBandwidth, kAlternateMemBandwidth);
    CostAnalysisPrefetchIntervalPicker prefetch_interval_picker(
        CostAnalysisPrefetchIntervalPicker(
            cost_analysis, /*min_async_copy_to_overlap_ratio=*/0.8,
            /*max_async_copy_to_overlap_ratio=*/10.0));
    return AssignMemorySpace(
        module, /*max_outstanding_async_copies=*/-1,
        MemorySpaceAssignment::GetMemoryBoundednessBufferIntervalCompare(
            cost_analysis),
        &prefetch_interval_picker);
  }

  std::unique_ptr<PresetAssignments> AssignMemorySpace(
      HloModule* module, int64 max_outstanding_async_copies = -1,
      int64 max_prefetch_interval = 10, int64 min_prefetch_interval = 2) {
    InstructionCountPrefetchIntervalPicker prefetch_interval_picker(
        min_prefetch_interval, max_prefetch_interval);
    return AssignMemorySpace(module, max_outstanding_async_copies,
                             /*buffer_interval_compare=*/{},
                             &prefetch_interval_picker);
  }

  std::unique_ptr<PresetAssignments> AssignMemorySpace(
      HloModule* module, int64 max_outstanding_async_copies,
      absl::optional<MemorySpaceAssignment::BufferIntervalCompare>
          buffer_interval_compare,
      PrefetchIntervalPicker* prefetch_interval_picker) {
    auto size_fn = [](const BufferValue& buffer) {
      return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
    };

    auto is_allowed_in_alternate_mem = [](const HloValue& value) {
      // Check if the value belongs to the entry computation.
      HloInstruction* instruction = value.instruction();
      HloComputation* computation = instruction->parent();
      bool in_entry_computation =
          (computation == computation->parent()->entry_computation());
      if (in_entry_computation &&
          instruction->opcode() == HloOpcode::kParameter) {
        return false;
      }
      return true;
    };

    MemorySpaceAssignment::Options options;
    options.alternate_memory_space = kAlternateMemorySpace;
    options.max_size_in_bytes = 128;
    options.alignment_in_bytes = 8;
    options.buffer_interval_compare = buffer_interval_compare;
    options.prefetch_interval_picker = prefetch_interval_picker;
    options.size_fn = size_fn;
    options.is_allowed_in_alternate_mem_fn = is_allowed_in_alternate_mem;
    options.max_outstanding_async_copies = max_outstanding_async_copies;
    options.allocate_across_sequential_calls = GetParam();
    options.verify = true;
    std::unique_ptr<PresetAssignments> preset_assignments =
        MemorySpaceAssignment::Run(module, options).ValueOrDie();
    CheckPresetAssignments(preset_assignments.get());
    return preset_assignments;
  }

  void CheckPresetAssignments(const PresetAssignments* preset_assignments) {
    // Ensure that the exported preset assignments point to layouts in the
    // alternate memory.  Also ensure that the positions are unique. Note that
    // we're using a std::set instead of absl::flat_hash_set because we can make
    // use of HloPosition's comparator logic instead of providing a hasher.
    std::set<HloPosition> positions_in_preset_assignments;
    for (auto& position_and_chunk : preset_assignments->chunks()) {
      HloPosition position = position_and_chunk.first;
      EXPECT_EQ(positions_in_preset_assignments.find(position),
                positions_in_preset_assignments.end());
      positions_in_preset_assignments.insert(position);
      const Shape& subshape =
          ShapeUtil::GetSubshape(position.instruction->shape(), position.index);
      EXPECT_EQ(subshape.layout().memory_space(), kAlternateMemorySpace)
          << "Exported position is not in alternate mem: "
          << position.ToString();
    }
  }

  std::unique_ptr<HloModule> CreateEvictAndPrefetchModule() {
    HloComputation::Builder builder(TestName());
    Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
    HloInstruction* p0 =
        builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
    HloInstruction* p1 =
        builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
    HloInstruction* tanh = builder.AddInstruction(
        HloInstruction::CreateUnary(shape, HloOpcode::kTanh, p0));
    // tanh should be placed in the alternate memory since there isn't much
    // contention in the beginning. However, tanh has another consumer at the
    // end. So it should be kicked out to default memory and prefetched back in.
    // The graph below is meant to increase the contention to force
    // eviction/prefetch behavior.
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
    return module;
  }
};

TEST_P(MemorySpaceAssignmentTest, ParameterOnly) {
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

TEST_P(MemorySpaceAssignmentTest, Simple) {
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
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
  EXPECT_THAT(p0, op::ShapeWithLayout(shape));
  EXPECT_THAT(p1, op::ShapeWithLayout(shape));
  EXPECT_THAT(mul, op::ShapeWithLayout(shape));
  EXPECT_THAT(add, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(sub, op::ShapeWithLayout(shape_in_alternate_mem));

  // Make sure the preset assignments is sane.
  EXPECT_EQ(preset_assignments->chunks().size(), 2);
  EXPECT_EQ(preset_assignments->sizes().size(), 1);
  // Ensure the offset assigned to add and sub are different.
  EXPECT_NE(preset_assignments->chunks()[0].second.offset,
            preset_assignments->chunks()[1].second.offset);
}

TEST_P(MemorySpaceAssignmentTest, NegateChain) {
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
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
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

TEST_P(MemorySpaceAssignmentTest, EvictAndPrefetch) {
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get());

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Add(op::Add(),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::AsyncCopy(kDefaultMemorySpace,
                                          kAlternateMemorySpace, op::Tanh()))));
}

TEST_P(MemorySpaceAssignmentTest, EvictAndPrefetchLimitAsyncCopies0) {
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/0);

  EXPECT_EQ(MemorySpaceAssignment::CountMaximumOutstandingAsyncCopies(*module),
            0);
}

TEST_P(MemorySpaceAssignmentTest, EvictAndPrefetchLimitAsyncCopies1) {
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/1);

  EXPECT_EQ(MemorySpaceAssignment::CountMaximumOutstandingAsyncCopies(*module),
            1);
}

TEST_P(MemorySpaceAssignmentTest, EvictAndPrefetchLimitAsyncCopies2) {
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/2);

  EXPECT_EQ(MemorySpaceAssignment::CountMaximumOutstandingAsyncCopies(*module),
            2);
}

TEST_P(MemorySpaceAssignmentTest, DontEvictWhenThereIsDefaultMemAllocation) {
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

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/1);

  // We expect the second argument to multiply is prefetched c.
  EXPECT_THAT(f, op::Multiply(op::Add(), op::CopyDone()));
  // We make sure that the second argument to this multiply is not evicted
  // CopyDone but is the original c.
  EXPECT_THAT(h, op::Multiply(op::Subtract(), op::Multiply()));
}

TEST_P(MemorySpaceAssignmentTest, EvictAndPrefetchAndPrefetch) {
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

TEST_P(MemorySpaceAssignmentTest, While) {
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

  AssignMemorySpace(module.get());

  // Ensure the tuple value and buffers used in the while instruction are
  // exempted from using the alternate memory when allocating across sequential
  // calls is disabled. However, body_data_mul is independent and can be safely
  // be placed in the alternate memory.
  const bool allocate_across_sequential_calls = GetParam();
  if (!allocate_across_sequential_calls) {
    EXPECT_THAT(tuple, op::ShapeWithLayout(tuple_shape));
    EXPECT_THAT(data, op::ShapeWithLayout(shape));
    EXPECT_THAT(iter, op::ShapeWithLayout(scalar_shape));
    EXPECT_THAT(body_data, op::ShapeWithLayout(shape));
    EXPECT_THAT(body_iter, op::ShapeWithLayout(scalar_shape));
    EXPECT_THAT(cond_iter, op::ShapeWithLayout(scalar_shape));
  }
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
  EXPECT_THAT(body_data_mul, op::ShapeWithLayout(shape_in_alternate_mem));
}

TEST_P(MemorySpaceAssignmentTest, Tuple) {
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

TEST_P(MemorySpaceAssignmentTest, Bitcast) {
  // Bitcasts can cause the position in the alternate memory to appear multiple
  // times in the preset assignments. This test ensure the preset assignments
  // refer to unique positions.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));
  HloInstruction* negate = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0));
  HloInstruction* bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(shape, negate));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, bitcast, p1));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0, p1, negate, bitcast, add});
  TF_CHECK_OK(module->set_schedule(schedule));

  AssignMemorySpace(module.get());

  EXPECT_EQ(bitcast->shape().layout().memory_space(), kAlternateMemorySpace);
}

TEST_P(MemorySpaceAssignmentTest, Bitcast2) {
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

TEST_P(MemorySpaceAssignmentTest, Bitcast3) {
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

TEST_P(MemorySpaceAssignmentTest, BitcastTuple) {
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

TEST_P(MemorySpaceAssignmentTest, BitcastGetTupleElementTuple) {
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

TEST_P(MemorySpaceAssignmentTest, GetSimplifiedOperandBug) {
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

TEST_P(MemorySpaceAssignmentTest, BitcastMultiUse) {
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
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
  EXPECT_THAT(negate0->operand(0), op::ShapeWithLayout(shape));
  EXPECT_THAT(add->operand(0), op::ShapeWithLayout(shape_in_alternate_mem));
}

TEST_P(MemorySpaceAssignmentTest, BitcastMultiUseTuple) {
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
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
  EXPECT_THAT(negate0->operand(0), op::ShapeWithLayout(shape));
  EXPECT_THAT(fusion->operand(0)->operand(0),
              op::ShapeWithLayout(shape_in_alternate_mem));
}

TEST_P(MemorySpaceAssignmentTest, BitcastScheduleBug) {
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

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
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

TEST_P(MemorySpaceAssignmentTest, TupleSelect) {
  // Make sure tuple-select is not optimized away.
  absl::string_view hlo_string = R"(
  HloModule tuple, is_scheduled=true

  ENTRY %main (a: f32[2], b: f32[2], c: f32[2], d: f32[2], cond: pred[]) -> f32[2] {
    %cond = pred[]{:T(128)E(32)} parameter(4)
    %token0 = token[] after-all()
    %d = f32[2]{0:T(128)} parameter(3)
    %c = f32[2]{0:T(128)} parameter(2)
    %b = f32[2]{0:T(128)} parameter(1)
    %a = f32[2]{0:T(128)} parameter(0)
    %tup0 = (f32[2]{0:T(128)}, f32[2]{0:T(128)}) tuple(f32[2]{0:T(128)} %a, f32[2]{0:T(128)} %b)
    %tup1 = (f32[2]{0:T(128)}, f32[2]{0:T(128)}) tuple(f32[2]{0:T(128)} %c, f32[2]{0:T(128)} %d)
    %s = (f32[2]{0:T(128)}, f32[2]{0:T(128)}) tuple-select(pred[]{:T(128)E(32)} %cond, (f32[2]{0:T(128)}, f32[2]{0:T(128)}) %tup0, (f32[2]{0:T(128)}, f32[2]{0:T(128)}) %tup1)
    %gte = f32[2]{0:T(128)} get-tuple-element((f32[2]{0:T(128)}, f32[2]{0:T(128)}) %s), index=0
    ROOT %negate = f32[2]{0:T(128)} negate(f32[2]{0:T(128)} %gte)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AssignMemorySpace(module.get());

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Negate(op::GetTupleElement(op::TupleSelect())));
}

TEST_P(MemorySpaceAssignmentTest, AddDependency) {
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

TEST_P(MemorySpaceAssignmentTest, WhileAllocationBug) {
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

  MemorySpaceAssignment::BufferIntervalCompare buffer_interval_compare =
      [](const MemorySpaceAssignment::BufferInterval& a,
         const MemorySpaceAssignment::BufferInterval& b) {
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
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    buffer_interval_compare, &prefetch_interval_picker);

  for (const HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kWhile) {
      const Shape& while_subshape =
          ShapeUtil::GetSubshape(instruction->shape(), {0});
      EXPECT_NE(while_subshape.layout().memory_space(), kAlternateMemorySpace);
    }
  }
}

TEST_P(MemorySpaceAssignmentTest, ControlPredecessorsBug) {
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

TEST_P(MemorySpaceAssignmentTest,
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

TEST_P(MemorySpaceAssignmentTest, LastUseOpt) {
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
      op::Multiply(op::Add(op::Parameter(0), op::Parameter(0)),
                   op::Subtract(op::Parameter(0),
                                op::Add(op::Parameter(0), op::Parameter(0)))));
}

TEST_P(MemorySpaceAssignmentTest, CopyOrdering) {
  // Test to make sure the CopyStarts follow the same CopyDone order. The shapes
  // are picked in increasing order to exploit the fact that heap simulator
  // processes larger tensors first. This checks the ability of the compiler to
  // reschedule:
  //
  //  CS1            CD1
  //   +--------------+
  //    +-----------+
  //   CS2         CD2
  //
  // into:
  //
  //    CS1          CD1
  //     +------------+
  //    +-----------+
  //   CS2         CD2
  HloComputation::Builder builder(TestName());
  Shape shape1 = ShapeUtil::MakeShape(F32, {2, 1});
  Shape shape2 = ShapeUtil::MakeShape(F32, {2, 2});
  Shape shape3 = ShapeUtil::MakeShape(F32, {2, 3});
  Shape shape4 = ShapeUtil::MakeShape(F32, {2, 4});
  PaddingConfig padding_config = MakeEdgePaddingConfig({{0, 0}, {0, 1}});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape3, shape4});
  HloInstruction* p0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p"));
  HloInstruction* p4 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape4, p0, 1));
  HloInstruction* p3 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape3, p0, 0));
  HloInstruction* p2 =
      builder.AddInstruction(HloInstruction::CreateParameter(2, shape2, "p2"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape1, "p1"));
  HloInstruction* negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, p1));
  HloInstruction* negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate0));
  HloInstruction* negate2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate1));
  HloInstruction* negate3 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate2));
  HloInstruction* negate4 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate3));
  HloInstruction* negate5 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate4));
  HloInstruction* negate6 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape1, HloOpcode::kNegate, negate5));
  HloInstruction* padding_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(F32)));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape1, HloOpcode::kAdd, negate6, p1));
  HloInstruction* padded_add1 = builder.AddInstruction(
      HloInstruction::CreatePad(shape2, add1, padding_value, padding_config));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape2, HloOpcode::kAdd, padded_add1, p2));
  HloInstruction* padded_add2 = builder.AddInstruction(
      HloInstruction::CreatePad(shape3, add2, padding_value, padding_config));
  HloInstruction* negate7 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape4, HloOpcode::kNegate, p4));
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape3, HloOpcode::kAdd, padded_add2, p3));
  HloInstruction* padded_add3 = builder.AddInstruction(
      HloInstruction::CreatePad(shape4, add3, padding_value, padding_config));
  HloInstruction* add4 = builder.AddInstruction(HloInstruction::CreateBinary(
      shape4, HloOpcode::kAdd, padded_add3, negate7));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(computation, {p0,
                                      p4,
                                      p3,
                                      p2,
                                      p1,
                                      negate0,
                                      negate1,
                                      negate2,
                                      negate3,
                                      negate4,
                                      negate5,
                                      negate6,
                                      padding_value,
                                      add1,
                                      padded_add1,
                                      add2,
                                      padded_add2,
                                      negate7,
                                      add3,
                                      padded_add3,
                                      add4});
  TF_CHECK_OK(module->set_schedule(schedule));

  // Use a large max prefetch interval to force CopyStart/CopyDone right after
  // the parameters.
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/50);

  // Iterate over the schedule to make sure CopyStart order and the
  // corresponding CopyDone order match.
  std::list<const HloInstruction*> copy_starts;
  for (HloInstruction* instruction : module->schedule()
                                         .sequence(module->entry_computation())
                                         .instructions()) {
    if (instruction->opcode() == HloOpcode::kCopyStart) {
      copy_starts.push_back(instruction);
    }
    if (instruction->opcode() == HloOpcode::kCopyDone) {
      EXPECT_EQ(copy_starts.front(), instruction->operand(0));
      copy_starts.pop_front();
    }
  }
}

TEST_P(MemorySpaceAssignmentTest, NonEntryComputationSchedule1) {
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

  AssignMemorySpace(module.get(), -1, 50);
}

TEST_P(MemorySpaceAssignmentTest, NonEntryComputationSchedule2) {
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

  AssignMemorySpace(module.get(), -1, 5);
}

TEST_P(MemorySpaceAssignmentTest, NonEntryComputationSchedule3) {
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

  AssignMemorySpace(module.get(), -1, 5);
}

TEST_P(MemorySpaceAssignmentTest, NonEntryComputationSchedule4) {
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

  AssignMemorySpace(module.get(), -1, 5);
}

TEST_P(MemorySpaceAssignmentTest, NonEntryComputationSchedule5) {
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
  AssignMemorySpace(module.get(), -1, 20);
}

TEST_P(MemorySpaceAssignmentTest, NonEntryComputationSchedule6) {
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
  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/25);

  int64 memory_space_across_while = kDefaultMemorySpace;
  bool allocate_across_sequential_calls = GetParam();
  if (allocate_across_sequential_calls) {
    memory_space_across_while = kAlternateMemorySpace;
  }

  // Index {0} of the while loop argument is not written inside the while loop,
  // so it can be trivially placed in the alternate memory space.
  *ShapeUtil::GetMutableSubshape(&tuple_shape, {0})->mutable_layout() =
      LayoutUtil::MakeLayout(
          /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
          kAlternateMemorySpace);
  // Indexes {1} and {2} of the while loop argument are only placed in the
  // alternate memory if we enable the allocate_across_sequential_calls option.
  *ShapeUtil::GetMutableSubshape(&tuple_shape, {1})->mutable_layout() =
      LayoutUtil::MakeLayout(
          /*minor_to_major=*/{}, /*tiles=*/{}, /*element_size_in_bits=*/0,
          memory_space_across_while);
  *ShapeUtil::GetMutableSubshape(&tuple_shape, {2})->mutable_layout() =
      LayoutUtil::MakeLayout(
          /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
          memory_space_across_while);

  // Expect the layout for the while loop and its aliased buffers.
  EXPECT_THAT(while_op, op::ShapeWithLayout(tuple_shape));
  EXPECT_THAT(while_op->operand(0), op::ShapeWithLayout(tuple_shape));
  EXPECT_THAT(cond_param, op::ShapeWithLayout(tuple_shape));
  EXPECT_THAT(body_param, op::ShapeWithLayout(tuple_shape));
  EXPECT_THAT(body_out, op::ShapeWithLayout(tuple_shape));
}

TEST_P(MemorySpaceAssignmentTest, DanglingCopy) {
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

TEST_P(MemorySpaceAssignmentTest, MultiOutputFusion) {
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

TEST_P(MemorySpaceAssignmentTest, TupleInput) {
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

TEST_P(MemorySpaceAssignmentTest, TupleToTuple1) {
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

  AssignMemorySpace(module.get(), -1, 5);
  EXPECT_THAT(fusion1,
              op::Fusion(op::Tuple(
                  op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                op::GetTupleElement(op::Fusion(), 0)),
                  op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                                op::GetTupleElement(op::Fusion(), 1)))));
}

TEST_P(MemorySpaceAssignmentTest, TupleToTuple2) {
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

  AssignMemorySpace(module.get(), -1, 5);

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

TEST_P(MemorySpaceAssignmentTest, TupleToTuple3) {
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

TEST_P(MemorySpaceAssignmentTest, InputOutputAlias) {
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
  TF_CHECK_OK(module->input_output_alias_config().SetUpAlias(
      {0}, 0, {0}, HloInputOutputAliasConfig::AliasKind::kSystemAlias));
  TF_CHECK_OK(module->input_output_alias_config().SetUpAlias(
      {1}, 0, {1}, HloInputOutputAliasConfig::AliasKind::kSystemAlias));

  AssignMemorySpace(module.get());

  // Make sure the input is in the default memory space.
  EXPECT_EQ(p->shape().tuple_shapes(0).layout().memory_space(),
            kDefaultMemorySpace);
  EXPECT_EQ(p->shape().tuple_shapes(1).layout().memory_space(),
            kDefaultMemorySpace);
}

TEST_P(MemorySpaceAssignmentTest, CostAnalysis) {
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
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
  EXPECT_THAT(negate0, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate2, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate3, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate4, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate5, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(negate6, op::ShapeWithLayout(shape_in_alternate_mem));
}

TEST_P(MemorySpaceAssignmentTest, MemoryBoundednessBufferIntervalCompare) {
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
  Shape shape_in_default_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {4, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kDefaultMemorySpace);
  // Expect only negates to be in alternate memory space. Not all might fit but
  // make sure at least one does.
  std::vector<HloInstruction*> negate_instructions = {negate0, negate1, negate2,
                                                      negate3, negate4};
  int64 num_negates_in_alternate_mem = absl::c_count_if(
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

TEST_P(MemorySpaceAssignmentTest, SimpleWhileTupleTest) {
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

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/-1,
                    /*max_prefetch_interval=*/50);

  // Ensure all parameters and while are placed in default memory.
  Shape shape_in_default_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {4, 6},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kDefaultMemorySpace);
  Shape s32_in_default_mem = ShapeUtil::MakeShapeWithLayout(
      xla::S32, {},
      /*minor_to_major=*/{}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kDefaultMemorySpace);
  Shape f32v1_in_default_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {1},
      /*minor_to_major=*/{0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kDefaultMemorySpace);
  Shape t_s32_f32v1_in_default_mem =
      ShapeUtil::MakeTupleShape({s32_in_default_mem, f32v1_in_default_mem});
  EXPECT_THAT(param, op::ShapeWithLayout(t_s32_f32v1_in_default_mem));
  EXPECT_THAT(while0, op::ShapeWithLayout(t_s32_f32v1_in_default_mem));
}

TEST_P(MemorySpaceAssignmentTest, EvictionsShouldntBeDelayed) {
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

TEST_P(MemorySpaceAssignmentTest,
       InputOutputsInAlternateMemShouldntBeAssigned) {
  // When input/outputs are marked to be in the alternate memory (e.g.
  // go/tpu-fast-mem-inference), do not allocate those and assume they will live
  // in the alternate memory for the entire computation. The BufferAssignment
  // pass, which is run after this, will allocate those buffers.
  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
      kAlternateMemorySpace);
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

  std::unique_ptr<PresetAssignments> preset_assignments =
      AssignMemorySpace(module.get());

  // Ensure that p1 is in the alternate memory and add, which has p1 as an
  // operand, has a direct dependency to p1 (no CopyStart/CopyDone).
  EXPECT_THAT(p1, op::ShapeWithLayout(shape_in_alternate_mem));
  EXPECT_THAT(add, op::Add(op::Negate(), op::Parameter(1)));
  // Make sure add is still in the alternate memory space.
  EXPECT_THAT(add, op::ShapeWithLayout(shape_in_alternate_mem));

  // Check the preset assignments and ensure the inputs/outputs in the alternate
  // memory space aren't in the preset assignments. Inputs/outputs in the
  // alternate memory space are left to BufferAssignment to be allocated.
  for (const auto& position_and_chunk : preset_assignments->chunks()) {
    const HloPosition& position = position_and_chunk.first;
    EXPECT_NE(position.instruction, p1);
    EXPECT_NE(position.instruction, add);
  }
}

INSTANTIATE_TEST_SUITE_P(MemorySpaceAssignmentInstantiation,
                         MemorySpaceAssignmentTest,
                         ::testing::Values(false, true));

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
  auto alternate_mem_space = MemorySpaceAssignment::MemorySpace::kAlternate;
  AsynchronousCopyOrdering ordering;
  EXPECT_FALSE(ordering.ViolatesOrdering(3, 11));
  ordering.AddCopy({3, 11, alternate_mem_space});
  EXPECT_FALSE(ordering.ViolatesOrdering(1, 8));
  ordering.AddCopy({1, 8, alternate_mem_space});
  EXPECT_FALSE(ordering.ViolatesOrdering(5, 14));
  ordering.AddCopy({5, 14, alternate_mem_space});
  EXPECT_FALSE(ordering.ViolatesOrdering(7, 14));
  ordering.AddCopy({7, 14, alternate_mem_space});
  EXPECT_TRUE(ordering.ViolatesOrdering(2, 16));
  EXPECT_TRUE(ordering.ViolatesOrdering(9, 12));
  EXPECT_TRUE(ordering.ViolatesOrdering(6, 17));
  EXPECT_FALSE(ordering.ViolatesOrdering(5, 13));
  ordering.AddCopy({5, 13, alternate_mem_space});
  EXPECT_FALSE(ordering.ViolatesOrdering(5, 14));
  ordering.AddCopy({5, 14, alternate_mem_space});
}

}  // namespace
}  // namespace xla

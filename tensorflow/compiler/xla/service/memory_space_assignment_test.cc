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

class MemorySpaceAssignmentTest : public HloTestBase {
 protected:
  // We use the following two memory space values to describe the default (slow
  // and large) and alternate (fast and small) memory spaces.
  const int64 kDefaultMemorySpace = 0;
  const int64 kAlternateMemorySpace = 1;

  std::unique_ptr<PresetAssignments> AssignMemorySpace(
      HloModule* module, int64 max_outstanding_async_copies = -1,
      int64 max_prefetch_interval = 10) {
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

    std::unique_ptr<PresetAssignments> preset_assignments =
        MemorySpaceAssignment::Run(
            module, kAlternateMemorySpace,
            /*max_size_in_bytes=*/128,
            /*min_prefetch_interval=*/2, max_prefetch_interval,
            /*alternate_memory_space_alignment_in_bytes=*/8, size_fn,
            is_allowed_in_alternate_mem, max_outstanding_async_copies)
            .ValueOrDie();
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

TEST_F(MemorySpaceAssignmentTest, EvictAndPrefetch) {
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get());

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Add(op::Add(),
              op::AsyncCopy(kAlternateMemorySpace, kDefaultMemorySpace,
                            op::AsyncCopy(kDefaultMemorySpace,
                                          kAlternateMemorySpace, op::Tanh()))));

  EXPECT_EQ(MemorySpaceAssignment::CountMaximumOutstandingAsyncCopies(*module),
            2);
}

TEST_F(MemorySpaceAssignmentTest, EvictAndPrefetchLimitAsyncCopies0) {
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/0);

  EXPECT_EQ(MemorySpaceAssignment::CountMaximumOutstandingAsyncCopies(*module),
            0);
}

TEST_F(MemorySpaceAssignmentTest, EvictAndPrefetchLimitAsyncCopies1) {
  std::unique_ptr<HloModule> module = CreateEvictAndPrefetchModule();

  AssignMemorySpace(module.get(), /*max_outstanding_async_copies=*/1);

  EXPECT_EQ(MemorySpaceAssignment::CountMaximumOutstandingAsyncCopies(*module),
            1);
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

  AssignMemorySpace(module.get());

  // Ensure the tuple value and buffers used in the while instruction are
  // exempted from using the alternate memory. However, body_data_mul is
  // independent and can be safely be placed in the alternate memory.
  EXPECT_THAT(tuple, op::ShapeWithLayout(tuple_shape));
  EXPECT_THAT(data, op::ShapeWithLayout(shape));
  EXPECT_THAT(iter, op::ShapeWithLayout(scalar_shape));
  EXPECT_THAT(body_data, op::ShapeWithLayout(shape));
  EXPECT_THAT(body_iter, op::ShapeWithLayout(scalar_shape));
  EXPECT_THAT(cond_iter, op::ShapeWithLayout(scalar_shape));
  Shape shape_in_alternate_mem = ShapeUtil::MakeShapeWithLayout(
      F32, {2, 3},
      /*minor_to_major=*/{1, 0}, /*tiles=*/{}, /*element_size_in_bits=*/0,
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

  EXPECT_EQ(bitcast->shape().layout().memory_space(), kAlternateMemorySpace);
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
  EXPECT_EQ(bitcast1->shape().layout().memory_space(), kAlternateMemorySpace);
  EXPECT_EQ(add->shape().layout().memory_space(), kAlternateMemorySpace);
  // bitcast2 will no longer have a consumer and should get DCE'd, so we don't
  // care about its memory space.
  EXPECT_EQ(bitcast3->shape().layout().memory_space(), kAlternateMemorySpace);
  EXPECT_EQ(bitcast4->shape().layout().memory_space(), kAlternateMemorySpace);
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
      op::Multiply(op::Add(op::Parameter(0), op::Parameter(0)),
                   op::Subtract(op::Parameter(0),
                                op::Add(op::Parameter(0), op::Parameter(0)))));
}

TEST_F(MemorySpaceAssignmentTest, CopyOrdering) {
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

  AssignMemorySpace(module.get(), -1, 50);
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

  AssignMemorySpace(module.get(), -1, 5);
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

  AssignMemorySpace(module.get(), -1, 5);
}

TEST_F(MemorySpaceAssignmentTest, NonEntryComputationSchedule4) {
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

  AssignMemorySpace(module.get(), -1, 5);
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

}  // namespace
}  // namespace xla

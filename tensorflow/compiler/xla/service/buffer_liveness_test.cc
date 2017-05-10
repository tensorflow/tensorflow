/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/buffer_liveness.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

class BufferLivenessTest : public HloTestBase {
 protected:
  // Returns the LogicalBuffer defined at the given instruction and
  // index. CHECKs if no buffer is defined at that point.
  const LogicalBuffer& GetBuffer(const BufferLiveness& liveness,
                                 const HloInstruction* instruction,
                                 const ShapeIndex& index) {
    const std::vector<const LogicalBuffer*>& pointed_to =
        liveness.points_to_analysis()
            .GetPointsToSet(instruction)
            .element(index);
    CHECK_EQ(1, pointed_to.size());
    CHECK_EQ(instruction, pointed_to[0]->instruction());
    CHECK(index == pointed_to[0]->index());
    return *pointed_to[0];
  }

  // Returns true if the top-level buffers for instructions 'a' and 'b' may
  // interfere. Precondition: 'a' and 'b' are array-shaped.
  bool InstructionsMayInterfere(const BufferLiveness& liveness,
                                HloInstruction* a, HloInstruction* b) {
    EXPECT_FALSE(ShapeUtil::IsTuple(a->shape()));
    EXPECT_FALSE(ShapeUtil::IsTuple(b->shape()));
    return liveness.MayInterfere(
        GetBuffer(liveness, /*instruction=*/a, /*index=*/{}),
        GetBuffer(liveness, /*instruction=*/b, /*index=*/{}));
  }

  // Returns true if the tuple elements at 'index' for instructions 'a' and 'b'
  // may interfere. Precondition: 'a' and 'b' are tuple-shaped, with equal
  // tuple element sub-shapes.
  bool TupleElementsMayInterfere(const BufferLiveness& liveness,
                                 HloInstruction* a, HloInstruction* b,
                                 const ShapeIndex& index) {
    // Check that top-level shapes are tuple and tuple element shapes are equal.
    EXPECT_TRUE(ShapeUtil::IsTuple(a->shape()));
    EXPECT_TRUE(ShapeUtil::IsTuple(b->shape()));
    EXPECT_TRUE(
        ShapeUtil::Compatible(ShapeUtil::GetSubshape(a->shape(), index),
                              ShapeUtil::GetSubshape(b->shape(), index)));
    // Lookup PointsTo set for instructions 'a' and 'b'.
    auto& points_to_analysis = liveness.points_to_analysis();
    const std::vector<const LogicalBuffer*>& points_to_a =
        points_to_analysis.GetPointsToSet(a).element(index);
    const std::vector<const LogicalBuffer*>& points_to_b =
        points_to_analysis.GetPointsToSet(b).element(index);
    // Make sure PointsTo sets for 'a' and 'b' are unambiguous.
    EXPECT_EQ(1, points_to_a.size());
    EXPECT_EQ(points_to_a.size(), points_to_b.size());
    // Check interference.
    return liveness.MayInterfere(*points_to_a[0], *points_to_b[0]);
  }

  // Returns true if the top-level buffers for the given instruction maybe
  // liveout of the entry computation.
  // Precondition: instruction is array-shaped.
  bool InstructionMaybeLiveOut(const BufferLiveness& liveness,
                               HloInstruction* instruction) {
    return liveness.MaybeLiveOut(
        GetBuffer(liveness, instruction, /*index=*/{}));
  }

  std::unique_ptr<HloComputation> BuildDummyComputation() {
    auto builder = HloComputation::Builder(TestName() + "_dummy");
    builder.AddInstruction(HloInstruction::CreateParameter(0, vec_, "param"));
    return builder.Build();
  }

  const Shape vec_ = ShapeUtil::MakeShape(xla::F32, {42});
};

TEST_F(BufferLivenessTest, ElementwiseChain) {
  // A simple chain of elementwise operations. No buffers should interfere.
  //
  // param --> negate -> exp -> log
  //
  auto builder = HloComputation::Builder(TestName());
  auto param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, vec_, "param"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vec_, HloOpcode::kNegate, param));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vec_, HloOpcode::kExp, negate));
  auto log = builder.AddInstruction(
      HloInstruction::CreateUnary(vec_, HloOpcode::kLog, exp));

  auto module = MakeUnique<HloModule>(TestName());
  module->AddEntryComputation(builder.Build());

  auto liveness =
      BufferLiveness::Run(module.get(),
                          MakeUnique<DependencyHloOrdering>(module.get()))
          .ConsumeValueOrDie();

  // Entry params always interfere.
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param, negate));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param, exp));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param, log));

  // No buffers should interfere.
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, negate, exp));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, negate, log));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, exp, negate));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, exp, log));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, log, negate));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, log, exp));

  // Buffers should interfere with itself.
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, exp, exp));

  // Only log is live out.
  EXPECT_FALSE(InstructionMaybeLiveOut(*liveness, param));
  EXPECT_FALSE(InstructionMaybeLiveOut(*liveness, negate));
  EXPECT_FALSE(InstructionMaybeLiveOut(*liveness, exp));
  EXPECT_TRUE(InstructionMaybeLiveOut(*liveness, log));
}

TEST_F(BufferLivenessTest, MultipleEntryParameters_Sequential) {
  // Two entry params, which interfere with each other.
  //
  // param0 --> negate ---------------\
  //                   param1 --> exp --> add
  auto builder = HloComputation::Builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, vec_, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, vec_, "param1"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vec_, HloOpcode::kNegate, param0));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vec_, HloOpcode::kExp, param1));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(vec_, HloOpcode::kAdd, negate, exp));

  auto module = MakeUnique<HloModule>(TestName());
  HloComputation* entry = module->AddEntryComputation(builder.Build());

  SequentialHloOrdering::HloModuleSequence sequence;
  sequence.insert({entry, {param0, negate, param1, exp, add}});
  auto liveness = BufferLiveness::Run(
                      module.get(),
                      MakeUnique<SequentialHloOrdering>(module.get(), sequence))
                      .ConsumeValueOrDie();

  // Entry params always interfere.
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param0, param1));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param0, negate));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param0, exp));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param0, add));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param1, param0));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param1, negate));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param1, exp));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param1, add));

  // Negate and exp still interfere.
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, negate, exp));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, exp, negate));

  // But {negate, add} and {exp, add} don't interfere.
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, negate, add));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, add, negate));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, exp, add));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, add, exp));
}

TEST_F(BufferLivenessTest, NonElementwiseOperand) {
  // A chain of operations with two elementwise and one non-elementwise. The
  // elementwise op should not interfere with its operand, while the
  // non-elementwise op should interfere. Entry params always interfere.
  //
  // param --> exp -> negate -> reverse
  //
  auto builder = HloComputation::Builder(TestName());
  auto param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, vec_, "param"));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vec_, HloOpcode::kExp, param));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vec_, HloOpcode::kNegate, exp));
  auto reverse =
      builder.AddInstruction(HloInstruction::CreateReverse(vec_, negate, {0}));

  auto module = MakeUnique<HloModule>(TestName());
  module->AddEntryComputation(builder.Build());

  auto liveness =
      BufferLiveness::Run(module.get(),
                          MakeUnique<DependencyHloOrdering>(module.get()))
          .ConsumeValueOrDie();

  // Entry params always interfere.
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param, exp));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param, negate));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param, reverse));

  // Negate is elementwise, so doesn't interfere with its operand.
  // Reverse is non-elementwise, so does interfere with its operand.
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, exp, negate));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, negate, reverse));
}

TEST_F(BufferLivenessTest, OverlappedBuffers) {
  // Verify simultaneously live buffers interfere (exp and negate).
  //
  // param --> negate -> add
  //     \---> exp -----/
  //
  auto builder = HloComputation::Builder(TestName());
  auto param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, vec_, "param"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vec_, HloOpcode::kNegate, param));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vec_, HloOpcode::kExp, param));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(vec_, HloOpcode::kAdd, negate, exp));

  auto module = MakeUnique<HloModule>(TestName());
  module->AddEntryComputation(builder.Build());

  auto liveness =
      BufferLiveness::Run(module.get(),
                          MakeUnique<DependencyHloOrdering>(module.get()))
          .ConsumeValueOrDie();

  // Entry params always interfere.
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param, negate));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param, exp));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param, add));

  // Negate and exp interfere with each other, but not with add.
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, negate, exp));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, exp, negate));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, negate, add));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, add, negate));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, exp, add));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, add, exp));
}

TEST_F(BufferLivenessTest, OverlappedBuffersSequentialOrder) {
  // Identical to the test OverlappedBuffer but using a sequential ordering of
  // HLO instructions.
  //
  // param --> negate -> add
  //     \---> exp -----/
  //
  // Sequential order:
  //  param, negate, exp, add
  //
  // Liveness is identical to the DependencyHloOrdering.
  auto builder = HloComputation::Builder(TestName());
  auto param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, vec_, "param"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vec_, HloOpcode::kNegate, param));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vec_, HloOpcode::kExp, param));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(vec_, HloOpcode::kAdd, negate, exp));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  SequentialHloOrdering::HloModuleSequence module_sequence;
  std::vector<const HloInstruction*> order = {param, negate, exp, add};
  module_sequence.emplace(computation, order);
  auto liveness =
      BufferLiveness::Run(module.get(), MakeUnique<SequentialHloOrdering>(
                                            module.get(), module_sequence))
          .ConsumeValueOrDie();

  // Entry params always interfere.
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param, negate));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param, exp));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param, add));

  // Negate and exp interfere with each other, but not with add.
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, negate, exp));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, exp, negate));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, negate, add));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, add, negate));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, exp, add));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, add, exp));
}

TEST_F(BufferLivenessTest, TupleLiveOut) {
  // Verify MaybeLiveOut with nested tuples. Result of computation looks like:
  //
  //   Tuple({Tuple({Negate(Param)}, Exp(Negate(Param)))})
  //
  // All values should be live out except Param.
  auto builder = HloComputation::Builder(TestName());
  auto param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, vec_, "param"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vec_, HloOpcode::kNegate, param));
  auto inner_tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({negate}));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vec_, HloOpcode::kExp, negate));
  auto outer_tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({inner_tuple, exp}));

  auto module = MakeUnique<HloModule>(TestName());
  module->AddEntryComputation(builder.Build());

  auto liveness =
      BufferLiveness::Run(module.get(),
                          MakeUnique<DependencyHloOrdering>(module.get()))
          .ConsumeValueOrDie();

  // All buffers should be live out except the param
  EXPECT_FALSE(InstructionMaybeLiveOut(*liveness, param));
  EXPECT_TRUE(InstructionMaybeLiveOut(*liveness, negate));
  EXPECT_TRUE(InstructionMaybeLiveOut(*liveness, inner_tuple));
  EXPECT_TRUE(InstructionMaybeLiveOut(*liveness, exp));
  EXPECT_TRUE(InstructionMaybeLiveOut(*liveness, outer_tuple));
}

// bitcast liveout.

TEST_F(BufferLivenessTest, EmbeddedComputation) {
  // Test MaybeLiveOut and MayInterfere for embedded computation.
  auto module = MakeUnique<HloModule>(TestName());

  auto embedded_builder = HloComputation::Builder(TestName() + "_embedded");
  auto embedded_param = embedded_builder.AddInstruction(
      HloInstruction::CreateParameter(0, vec_, "embedded_param"));
  auto embedded_log = embedded_builder.AddInstruction(
      HloInstruction::CreateUnary(vec_, HloOpcode::kLog, embedded_param));

  auto embedded_computation =
      module->AddEmbeddedComputation(embedded_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, vec_, "param"));
  auto call = builder.AddInstruction(
      HloInstruction::CreateCall(vec_, {param}, embedded_computation));

  module->AddEntryComputation(builder.Build());

  auto liveness =
      BufferLiveness::Run(module.get(),
                          MakeUnique<DependencyHloOrdering>(module.get()))
          .ConsumeValueOrDie();

  // Buffers in different computations should always interfere.
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, embedded_log, call));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, embedded_param, param));
  EXPECT_FALSE(
      InstructionsMayInterfere(*liveness, embedded_param, embedded_log));

  // The only buffers for which MaybeLiveOut == true are those live out
  // of the entry computation. Buffers live out of embedded computations should
  // return false for this method.
  EXPECT_FALSE(InstructionMaybeLiveOut(*liveness, embedded_log));
  EXPECT_TRUE(InstructionMaybeLiveOut(*liveness, call));
}

TEST_F(BufferLivenessTest, TupleConstantLiveOut) {
  // Verify non top-level elements of a nested tuple constant are properly
  // marked as liveout. Computation:
  //
  //   GetTupleElement(0, TupleConstant({{0, 1}, {3}})
  //
  // Only the array buffers containing 0 and 1 are liveout of the
  // computation. The buffer containing {0, 1} is copied by GetTupleElement, and
  // the buffers containing {3} and 3 are dead.
  auto builder = HloComputation::Builder(TestName());
  auto inner_tuple0 =
      LiteralUtil::MakeTuple({LiteralUtil::CreateR0<int64>(0).get(),
                              LiteralUtil::CreateR0<int64>(1).get()});
  auto inner_tuple1 =
      LiteralUtil::MakeTuple({LiteralUtil::CreateR0<int64>(3).get()});
  auto tuple_constant = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::MakeTuple({inner_tuple0.get(), inner_tuple1.get()})));
  builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      inner_tuple0->shape(), tuple_constant, 0));

  auto module = MakeUnique<HloModule>(TestName());
  module->AddEntryComputation(builder.Build());

  auto liveness =
      BufferLiveness::Run(module.get(),
                          MakeUnique<DependencyHloOrdering>(module.get()))
          .ConsumeValueOrDie();

  // Only the element buffers of the tuple constant which are pointed to by
  // the GetTupleElement instruction should be liveout.
  EXPECT_FALSE(liveness->MaybeLiveOut(
      GetBuffer(*liveness, tuple_constant, /*index=*/{})));
  EXPECT_TRUE(liveness->MaybeLiveOut(
      GetBuffer(*liveness, tuple_constant, /*index=*/{0})));
  EXPECT_TRUE(liveness->MaybeLiveOut(
      GetBuffer(*liveness, tuple_constant, /*index=*/{0, 0})));
  EXPECT_TRUE(liveness->MaybeLiveOut(
      GetBuffer(*liveness, tuple_constant, /*index=*/{0, 1})));
  EXPECT_FALSE(liveness->MaybeLiveOut(
      GetBuffer(*liveness, tuple_constant, /*index=*/{1})));
  EXPECT_FALSE(liveness->MaybeLiveOut(
      GetBuffer(*liveness, tuple_constant, /*index=*/{1, 0})));
  EXPECT_FALSE(liveness->MaybeLiveOut(
      GetBuffer(*liveness, tuple_constant, /*index=*/{1, 0})));
}

TEST_F(BufferLivenessTest, IndependentTupleElements) {
  auto builder = HloComputation::Builder(TestName());
  // Create param0 Tuple.
  auto tuple_param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape(
             {ShapeUtil::MakeShape(F32, {8}), ShapeUtil::MakeShape(S32, {4})}),
      "param0"));
  // Create independent computations for each tuple elememt.

  // Tuple element0 computation:
  //   Add(GetTupleElement(tuple_param0, 0), const0)
  auto tuple_element0_shape =
      ShapeUtil::GetSubshape(tuple_param0->shape(), {0});
  auto tuple_element0 =
      builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          tuple_element0_shape, tuple_param0, 0));
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f})));
  auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
      tuple_element0_shape, HloOpcode::kAdd, tuple_element0, const0));

  // Tuple element1 computation:
  //   Add(GetTupleElement(tuple_param0, 1), const1)
  auto tuple_element1_shape =
      ShapeUtil::GetSubshape(tuple_param0->shape(), {1});
  auto tuple_element1 =
      builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          tuple_element1_shape, tuple_param0, 1));
  auto const1 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f})));
  auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
      tuple_element1_shape, HloOpcode::kAdd, tuple_element1, const1));

  // Create output tuple.
  auto tuple_root =
      builder.AddInstruction(HloInstruction::CreateTuple({add0, add1}));

  auto module = MakeUnique<HloModule>(TestName());
  module->AddEntryComputation(BuildDummyComputation());
  module->AddEmbeddedComputation(builder.Build());

  auto liveness =
      BufferLiveness::Run(module.get(),
                          MakeUnique<DependencyHloOrdering>(module.get()))
          .ConsumeValueOrDie();

  // We compare tuple element pairs that are input/output to the computation:
  // 1) (input_tuple_element, output_tuple_element) = ('tuple_element0', 'add0')
  // 2) (input_tuple_element, output_tuple_element) = ('tuple_element1', 'add1')

  // Tuple output element 'add0' does not depend on input 'tuple_element1'.
  // Tuple output element 'add1' does not depend on input 'tuple_element0'.

  // Both element pair does not interfere, because there is no other dependency
  // on the pairs tuple input element, and so liveness can compute that all
  // users of the input tuple element execute before the associated output
  // tuple element.
  EXPECT_FALSE(
      TupleElementsMayInterfere(*liveness, tuple_param0, tuple_root, {0}));
  EXPECT_FALSE(
      TupleElementsMayInterfere(*liveness, tuple_param0, tuple_root, {1}));
}

TEST_F(BufferLivenessTest, DependentTupleElements) {
  auto builder = HloComputation::Builder(TestName());
  // Create param0 Tuple.
  auto tuple_param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape(
             {ShapeUtil::MakeShape(F32, {8}), ShapeUtil::MakeShape(F32, {8})}),
      "param0"));
  // Create dependent computations for each tuple elememt.

  // Tuple element0 computation:
  //   Add(GetTupleElement(tuple_param0, 0), const0)
  auto tuple_element0_shape =
      ShapeUtil::GetSubshape(tuple_param0->shape(), {0});
  auto tuple_element0 =
      builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          tuple_element0_shape, tuple_param0, 0));
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f})));
  auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
      tuple_element0_shape, HloOpcode::kAdd, tuple_element0, const0));

  // Tuple element1 computation:
  //   Add(GetTupleElement(tuple_param0, 0), GetTupleElement(tuple_param0, 1))
  auto tuple_element1_shape =
      ShapeUtil::GetSubshape(tuple_param0->shape(), {1});
  auto tuple_element1 =
      builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          tuple_element1_shape, tuple_param0, 1));
  auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
      tuple_element1_shape, HloOpcode::kAdd, tuple_element0, tuple_element1));

  // Create output tuple.
  auto tuple_root =
      builder.AddInstruction(HloInstruction::CreateTuple({add0, add1}));

  auto module = MakeUnique<HloModule>(TestName());
  module->AddEntryComputation(BuildDummyComputation());
  module->AddEmbeddedComputation(builder.Build());

  auto liveness =
      BufferLiveness::Run(module.get(),
                          MakeUnique<DependencyHloOrdering>(module.get()))
          .ConsumeValueOrDie();

  // We compare tuple element pairs that are input/output to the computation:
  // 1) (input_tuple_element, output_tuple_element) = ('tuple_element0', 'add0')
  // 2) (input_tuple_element, output_tuple_element) = ('tuple_element1', 'add1')

  // The first tuple element pair output 'add0', has no dependency on second
  // tuple element pairs input 'tuple_element1'.

  // The second tuple element pair output 'add1', has a dependency on first
  // tuple element pairs input 'tuple_element0'.

  // The first tuple element pair does interfere, because liveness cannot
  // compute that all references to 'tuple_element0' are executed before 'add0'
  // (because of the depenency of 'add1' on 'tuple_element0').
  EXPECT_TRUE(
      TupleElementsMayInterfere(*liveness, tuple_param0, tuple_root, {0}));

  // The second tuple element pair does not interfere, because there is no
  // other dependency on 'tuple_element1', and so liveness can compute that
  // all users execute before 'add1'.
  EXPECT_FALSE(
      TupleElementsMayInterfere(*liveness, tuple_param0, tuple_root, {1}));
}

class FusedDynamicUpdateSliceLivenessTest : public BufferLivenessTest {
 protected:
  // Builds and runs a computation (see test case computation graphs below).
  // Runs BufferLiveness on this computation.
  // Returns whether buffer interference is detected between tuple-shaped
  // parameter and root instructions at tuple element 1.
  bool Run(const bool update_uses_tuple_element1,
           const bool fuse_gte0 = false) {
    auto builder = HloComputation::Builder(TestName());
    // Create param0 Tuple.
    Shape data_shape = ShapeUtil::MakeShape(F32, {8});
    Shape update_shape = ShapeUtil::MakeShape(F32, {3});
    auto tuple_param0 = builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeTupleShape({data_shape, data_shape}), "param0"));

    auto gte0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape, tuple_param0, 0));

    auto gte1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape, tuple_param0, 1));

    auto update = builder.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR1<float>({2.f, 2.f, 2.f})));
    HloInstruction* slice = nullptr;
    if (update_uses_tuple_element1) {
      // Create a slice instruction as an additional user of 'gte1'.
      slice = builder.AddInstruction(
          HloInstruction::CreateSlice(update_shape, gte1, {0}, {3}));
      update = builder.AddInstruction(HloInstruction::CreateBinary(
          update_shape, HloOpcode::kAdd, update, slice));
    }
    // Create a DynamicUpdateSlice instruction of tuple element 1 with 'update'.
    auto starts = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32>({2})));
    auto dynamic_update_slice =
        builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            data_shape, gte1, update, starts));
    // Create output tuple.
    auto tuple_root = builder.AddInstruction(
        HloInstruction::CreateTuple({gte0, dynamic_update_slice}));
    // Build module and get reference to entry computation.
    auto module = MakeUnique<HloModule>(TestName());
    module->AddEntryComputation(BuildDummyComputation());
    auto* computation = module->AddEmbeddedComputation(builder.Build());
    // Create fusion instruction based on number of tuple element 1 users.
    if (update_uses_tuple_element1) {
      computation->CreateFusionInstruction(
          {dynamic_update_slice, starts, update, CHECK_NOTNULL(slice), gte1},
          HloInstruction::FusionKind::kLoop);
    } else {
      computation->CreateFusionInstruction(
          {dynamic_update_slice, starts, update, gte1},
          HloInstruction::FusionKind::kLoop);
    }
    // Create fusion instruction for tuple element 0 (if requested).
    if (fuse_gte0) {
      computation->CreateFusionInstruction({gte0},
                                           HloInstruction::FusionKind::kLoop);
    }

    // Run BufferLiveness on 'module'.
    auto liveness =
        BufferLiveness::Run(module.get(),
                            MakeUnique<DependencyHloOrdering>(module.get()))
            .ConsumeValueOrDie();
    // Return whether or not buffers interfernce is detected between
    // 'tuple_param0' and 'tuple_root' at shape index '{1}'.
    return TupleElementsMayInterfere(*liveness, tuple_param0, tuple_root, {1});
  }
};

// Tests that live ranges of buffers Param0[1] and Tuple[1] (which alias fusion)
// do not overlap with the following computation:
//
//         Param0
//        /     \
//     GTE(0)  Fusion ----------->  FusionParam
//        |      |                      |
//        |      |                    GTE(1) Const Const
//        |      |                      \      |    /
//        |      |                    DynamicUpdateSlice  // fused root
//         \    /
//          Tuple  // computation root
//
TEST_F(FusedDynamicUpdateSliceLivenessTest, NoInterference) {
  EXPECT_FALSE(Run(/*update_uses_tuple_element1=*/false));
}

// Tests that live ranges of buffers Param0[1] and Tuple[1] (which aliases
// 'fusion1') do not overlap in the presence of another fusion instruction
// (which is a user of 'param0' at a different tuple index).
// BufferLiveness should detect no uses of Param0 at index {1} in Fusion0
// (because Fusion0 only uses Param0 at index {0}).
//
//                               Param0
//                               /    \
//      FusionParam  <----- Fusion0  Fusion1 ------>  FusionParam
//         |                    |      |                 |
//        GTE(0)                |      |               GTE(1) Const Const
//                              |      |                  \      |    /
//                               \    /                DynamicUpdateSlice
//                               Tuple
//
TEST_F(FusedDynamicUpdateSliceLivenessTest, NoInterferenceWithUnrelatedFusion) {
  EXPECT_FALSE(Run(/*update_uses_tuple_element1=*/false, /*fuse_gte0=*/true));
}

// Tests that live ranges of buffers Param0[1] and Tuple[1] (which alias fusion)
// do overlap because GTE(1) has two users:
// 1) DynamicUpdateSlice at operand 0.
// 2) Slice at operand 0.
//
//         Param0
//        /     \   Const
//       /       \  /
//     GTE(0)  Fusion ----------->  FusionParam FusionParam
//        |      |                      |         |
//        |      |                    GTE(1)      /
//        |      |                      | \      /
//        |      |                      | Slice /
//        |      |                      |   \  /
//        |      |                      |   Add   Const
//        |      |                      |    |      |
//        |      |                    DynamicUpdateSlice  // fused root
//         \    /
//          Tuple  // computation root
//
TEST_F(FusedDynamicUpdateSliceLivenessTest, WithInterference) {
  EXPECT_TRUE(Run(/*update_uses_tuple_element1=*/true));
}

class DynamicUpdateSliceLivenessTest : public BufferLivenessTest {
 protected:
  // Builds and runs a computation (see test case computation graphs below).
  // Runs BufferLiveness on this computation.
  // Returns whether buffer interference is detected between tuple-shaped
  // parameter and root instructions at tuple element 1.
  bool Run(const bool tuple_element1_has_two_uses) {
    auto builder = HloComputation::Builder(TestName());
    // Create param0 Tuple.
    Shape data_shape = ShapeUtil::MakeShape(F32, {8});
    Shape update_shape = ShapeUtil::MakeShape(F32, {3});
    auto tuple_param0 = builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeTupleShape({data_shape, data_shape}), "param0"));

    auto gte0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape, tuple_param0, 0));

    auto gte1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape, tuple_param0, 1));

    auto update = builder.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR1<float>({2.f, 2.f, 2.f})));

    if (tuple_element1_has_two_uses) {
      // Add 'gte0' and 'gte1' to create another user of 'gte1'.
      gte0 = builder.AddInstruction(HloInstruction::CreateBinary(
          data_shape, HloOpcode::kAdd, gte0, gte1));
    }
    // Create a DynamicUpdateSlice instruction of tuple element 1 with 'update'.
    auto starts = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32>({2})));
    auto dynamic_update_slice =
        builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            data_shape, gte1, update, starts));
    // Create output tuple.
    auto tuple_root = builder.AddInstruction(
        HloInstruction::CreateTuple({gte0, dynamic_update_slice}));
    // Build module and get reference to entry computation.
    auto module = MakeUnique<HloModule>(TestName());
    module->AddEntryComputation(BuildDummyComputation());
    module->AddEmbeddedComputation(builder.Build());
    // Run BufferLiveness on 'module'.
    auto liveness =
        BufferLiveness::Run(module.get(),
                            MakeUnique<DependencyHloOrdering>(module.get()))
            .ConsumeValueOrDie();
    // Return whether or not buffers interfernce is detected between
    // 'tuple_param0' and 'tuple_root' at shape index '{1}'.
    return TupleElementsMayInterfere(*liveness, tuple_param0, tuple_root, {1});
  }
};

// Tests that live ranges of buffers Param0[1] and Tuple[1] do not overlap in
// the following computation (because DynamicUpdateSlice (at operand 0) is the
// unique user):
//
//     Parameter0
//      |      |
//    GTE(0) GTE(1) Const Const
//      |      \      |    /
//      |    DynamicUpdateSlice
//       \    /
//        Tuple
//
TEST_F(DynamicUpdateSliceLivenessTest, NoInterference) {
  EXPECT_FALSE(Run(/*tuple_element1_has_two_uses=*/false));
}

// Tests that live ranges of buffers Param0[1] and Tuple[1] do overlap because
// GTE(1) has two users:
// 1) DynamicUpdateSlice at operand 0.
// 2) Add at operand 1.
//
//     Parameter0
//      |      |
//    GTE(0) GTE(1)
//      |   /  |
//      |  /   |
//      Add    |     Const Const
//      |      |      |      |
//      |    DynamicUpdateSlice
//       \    /
//        Tuple
//
TEST_F(DynamicUpdateSliceLivenessTest, WithInterference) {
  EXPECT_TRUE(Run(/*tuple_element1_has_two_uses=*/true));
}

}  // namespace

}  // namespace xla

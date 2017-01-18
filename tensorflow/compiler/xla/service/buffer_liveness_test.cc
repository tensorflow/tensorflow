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

  // No buffers should interfere.
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, param, negate));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, negate, exp));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, exp, negate));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, exp, log));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, param, log));

  // Buffers should interfere with itself.
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, exp, exp));

  // Only log is live out.
  EXPECT_FALSE(InstructionMaybeLiveOut(*liveness, param));
  EXPECT_FALSE(InstructionMaybeLiveOut(*liveness, negate));
  EXPECT_FALSE(InstructionMaybeLiveOut(*liveness, exp));
  EXPECT_TRUE(InstructionMaybeLiveOut(*liveness, log));
}

TEST_F(BufferLivenessTest, NonElementwiseOperand) {
  // A chain of operations with one elementwise and one non-elementwise. The
  // elementwise op should not interfere with its operand, while the
  // non-elementwise op should interfere.
  //
  // param --> negate -> reverse
  //
  auto builder = HloComputation::Builder(TestName());
  auto param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, vec_, "param"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vec_, HloOpcode::kNegate, param));
  auto reverse =
      builder.AddInstruction(HloInstruction::CreateReverse(vec_, negate, {0}));

  auto module = MakeUnique<HloModule>(TestName());
  module->AddEntryComputation(builder.Build());

  auto liveness =
      BufferLiveness::Run(module.get(),
                          MakeUnique<DependencyHloOrdering>(module.get()))
          .ConsumeValueOrDie();

  // No buffers should interfere.
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, param, negate));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, negate, reverse));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, param, negate));
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

  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param, negate));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param, exp));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, negate, exp));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, param, add));
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
  // Liveness is identical to the DependencyHloOrdering except that 'param' and
  // exp no longer interfere.
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

  EXPECT_TRUE(InstructionsMayInterfere(*liveness, param, negate));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, param, exp));
  EXPECT_TRUE(InstructionsMayInterfere(*liveness, negate, exp));
  EXPECT_FALSE(InstructionsMayInterfere(*liveness, param, add));
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
  module->AddEntryComputation(builder.Build());

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
  module->AddEntryComputation(builder.Build());

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

}  // namespace

}  // namespace xla

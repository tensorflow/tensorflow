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

#include "tensorflow/compiler/xla/service/copy_insertion.h"

#include <set>

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

using ::testing::UnorderedElementsAre;

int64 CountCopies(const HloComputation& computation) {
  int64 count = 0;
  for (const auto& instruction : computation.instructions()) {
    if (instruction->opcode() == HloOpcode::kCopy) {
      count++;
    }
  }
  return count;
}

int64 CountCopies(const HloModule& module) {
  int64 count = 0;
  for (const auto& computation : module.computations()) {
    count += CountCopies(*computation);
  }
  return count;
}

int64 CountControlEdges(const HloComputation& computation) {
  int64 count = 0;
  for (const auto& instruction : computation.instructions()) {
    count += instruction->control_successors().size();
  }
  return count;
}

int64 CountControlEdges(const HloModule& module) {
  int64 count = 0;
  for (const auto& computation : module.computations()) {
    count += CountControlEdges(*computation);
  }
  return count;
}

class CopyInsertionTest : public HloTestBase {
 protected:
  void InsertCopies(HloModule* module) {
    CopyInsertion copy_insertion;
    ASSERT_IS_OK(copy_insertion.Run(module).status());
  }

  const Shape scalar_shape_ = ShapeUtil::MakeShape(F32, {});
};

TEST_F(CopyInsertionTest, SingleParameter) {
  // Computation is a single parameter passed into a tuple. The parameter should
  // be copied before entering the tuple.
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "x"));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({x}));

  EXPECT_THAT(x->users(), UnorderedElementsAre(tuple));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Copy(x)));
}

TEST_F(CopyInsertionTest, SingleConstant) {
  // Computation is a single constant passed into a tuple. The parameter should
  // be copied before entering the tuple.
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({constant}));

  EXPECT_THAT(constant->users(), UnorderedElementsAre(tuple));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Copy(constant)));
}

TEST_F(CopyInsertionTest, ExistingCopiesNotRemoved) {
  // Verify that kCopy instructions which change layout and exist before
  // copy-insertion remain in the graph after copy-insertion.
  auto module = CreateNewVerifiedModule();

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* constant =
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<float>({{0.f, 2.f}, {2.f, 4.f}})));
  auto minor_to_major = LayoutUtil::MinorToMajor(constant->shape());
  Layout reversed_layout =
      LayoutUtil::MakeLayoutFromMajorToMinor(minor_to_major);
  Shape copy_shape = constant->shape();
  *copy_shape.mutable_layout() = reversed_layout;
  HloInstruction* copy_1 = builder.AddInstruction(
      HloInstruction::CreateUnary(copy_shape, HloOpcode::kCopy, constant));
  HloInstruction* copy_2 = builder.AddInstruction(
      HloInstruction::CreateUnary(copy_shape, HloOpcode::kCopy, constant));
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      constant->shape(), HloOpcode::kAdd, copy_1, copy_2));
  builder.AddInstruction(
      HloInstruction::CreateUnary(add->shape(), HloOpcode::kCopy, add));

  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(CountCopies(*module), 3);

  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 2);

  EXPECT_EQ(module->entry_computation()->root_instruction(), add);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Add(op::Copy(op::Constant()), op::Copy(op::Constant())));
}

TEST_F(CopyInsertionTest, MultipleConstantsAndParameters) {
  // Create a computation with more than one constant and parameter. Only one of
  // each constant/parameter is pointed to by the output tuple. Only these
  // instructions should be copied.
  auto builder = HloComputation::Builder(TestName());

  HloInstruction* constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  HloInstruction* constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));

  HloInstruction* x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "x"));
  HloInstruction* y = builder.AddInstruction(
      HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(F32, {}), "y"));

  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, constant1, y));

  builder.AddInstruction(HloInstruction::CreateTuple({constant2, x, add}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 2);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::Copy(constant2), op::Copy(x), op::Add(constant1, y)));
}

TEST_F(CopyInsertionTest, AmbiguousPointsToSet) {
  // Create a computation using select which has an ambiguous points-to set for
  // the computation result. Verify that copies are added properly.
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  HloInstruction* constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  HloInstruction* constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(3.0)));

  HloInstruction* tuple1 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  HloInstruction* tuple2 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant3, constant2}));

  HloInstruction* pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateTernary(
      tuple1->shape(), HloOpcode::kTupleSelect, pred, tuple1, tuple2));

  EXPECT_THAT(constant1->users(), UnorderedElementsAre(tuple1));
  EXPECT_THAT(constant2->users(), UnorderedElementsAre(tuple1, tuple2));
  EXPECT_THAT(constant3->users(), UnorderedElementsAre(tuple2));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  HloInstruction* old_root = module->entry_computation()->root_instruction();
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 2);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Copy(op::GetTupleElement(old_root)),
                        op::Copy(op::GetTupleElement(old_root))));
}

TEST_F(CopyInsertionTest, BitcastParameter) {
  // The output of a bitcast is its operand (same buffer), so a bitcast
  // parameter feeding the result must have a copy added.
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {4}), "x"));
  HloInstruction* bitcast = builder.AddInstruction(
      HloInstruction::CreateBitcast(ShapeUtil::MakeShape(F32, {2, 2}), x));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_THAT(x->users(), UnorderedElementsAre(bitcast));

  HloInstruction* old_root = module->entry_computation()->root_instruction();
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(old_root));
}

TEST_F(CopyInsertionTest, BitcastConstant) {
  // The output of a bitcast is its operand (same buffer), so a bitcast
  // constant feeding the result must have a copy added.
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* constant =
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<float>({1.0, 42.0})));
  HloInstruction* bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(
          ShapeUtil::MakeShape(F32, {2, 2}), constant));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_THAT(constant->users(), UnorderedElementsAre(bitcast));

  HloInstruction* old_root = module->entry_computation()->root_instruction();
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(old_root));
}

TEST_F(CopyInsertionTest, BitcastTupleElementParameter) {
  // Same as BitcastParameter, but the bitcast is wrapped in a tuple.
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {4}), "x"));
  HloInstruction* bitcast = builder.AddInstruction(
      HloInstruction::CreateBitcast(ShapeUtil::MakeShape(F32, {2, 2}), x));
  builder.AddInstruction(HloInstruction::CreateTuple({bitcast}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_THAT(x->users(), UnorderedElementsAre(bitcast));

  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Copy(bitcast)));
}

TEST_F(CopyInsertionTest, NestedTupleParameter) {
  // Construct a trivial computation where the root of the computation is a
  // nested tuple-shaped parameter. The parameter should be deep copied and the
  // copy should be the root of the computation.
  auto builder = HloComputation::Builder(TestName());

  // Param shape is: ((F32[], S32[1,2,3]), F32[42])
  builder.AddInstruction(HloInstruction::CreateParameter(
      0,
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {}),
                                      ShapeUtil::MakeShape(S32, {1, 2, 3})}),
           ShapeUtil::MakeShape(F32, {42})}),
      "param0"));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(HloOpcode::kParameter,
            module->entry_computation()->root_instruction()->opcode());

  HloInstruction* old_root = module->entry_computation()->root_instruction();
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 3);

  HloInstruction* new_root = module->entry_computation()->root_instruction();
  EXPECT_NE(old_root, new_root);

  EXPECT_THAT(
      new_root,
      op::Tuple(
          op::Tuple(
              op::Copy(op::GetTupleElement(op::GetTupleElement(old_root))),
              op::Copy(op::GetTupleElement(op::GetTupleElement(old_root)))),
          op::Copy(op::GetTupleElement(old_root))));
}

TEST_F(CopyInsertionTest, ElementOfNestedTupleParameter) {
  // Construct a computation where the root of the computation is a tuple
  // element of a nested tuple-shaped parameter.
  auto builder = HloComputation::Builder(TestName());

  // Param shape is: ((F32[], S32[1,2,3]), F32[42])
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0,
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {}),
                                      ShapeUtil::MakeShape(S32, {1, 2, 3})}),
           ShapeUtil::MakeShape(F32, {42})}),
      "param0"));

  // The return value of the computation is the zero-th element of the nested
  // tuple. This element is itself a tuple.
  auto gte = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetSubshape(param->shape(), {0}), param, 0));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(gte, module->entry_computation()->root_instruction());

  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 2);

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(op::Copy(op::GetTupleElement(op::GetTupleElement(param))),
                op::Copy(op::GetTupleElement(op::GetTupleElement(param)))));
}

TEST_F(CopyInsertionTest, AmbiguousTopLevelRoot) {
  // Create a computation using select which has an ambiguous points-to set for
  // the top-level buffer of the root of the computation. Verify that a shallow
  // copy is added.
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  HloInstruction* constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));

  HloInstruction* tuple1 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  HloInstruction* tuple2 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant2, constant1}));

  HloInstruction* pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloInstruction* select = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple1->shape(), HloOpcode::kTupleSelect, pred, tuple1, tuple2));
  HloInstruction* gte =
      builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          ShapeUtil::GetSubshape(select->shape(), {0}), select, 0));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(gte, module->entry_computation()->root_instruction());

  HloInstruction* old_root = module->entry_computation()->root_instruction();
  InsertCopies(module.get());
  EXPECT_EQ(CountCopies(*module), 1);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(old_root));
}

class WhileCopyInsertionTest : public CopyInsertionTest {
 protected:
  WhileCopyInsertionTest() : module_(CreateNewVerifiedModule()) {}

  // Builds a While condition computation which reads the induction variable
  // from the tuple parameter, and returns a predicate indicating whether this
  // value is less than the constant '10'.
  // The parameter 'nested' specifies the loop state shape from which to
  // read the induction variable.
  std::unique_ptr<HloComputation> BuildConditionComputation(
      const Shape& loop_state_shape) {
    auto builder = HloComputation::Builder(TestName() + ".Condition");
    auto limit_const = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(10)));
    auto loop_state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape, "loop_state"));
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            limit_const->shape(), loop_state, 0));
    builder.AddInstruction(HloInstruction::CreateCompare(
        condition_result_shape_, induction_variable, limit_const,
        ComparisonDirection::kLt));
    return builder.Build();
  }

  // Builds a While body computation with one output tuple element dependent on
  // both input tuple elements.
  // EX:
  // Body({in0, in1})
  //   out0 = Add(in0, 1)
  //   out1 = Add(BCast(in0), in1)
  //   Tuple(out0, out1)
  std::unique_ptr<HloComputation> BuildDependentBodyComputation() {
    auto builder = HloComputation::Builder(TestName() + ".Body");
    // Create param instruction to access loop state.
    auto loop_state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape_, "loop_state"));
    // Update the induction variable GTE(0).
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            induction_variable_shape_, loop_state, 0));
    auto inc = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
    auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
        induction_variable->shape(), HloOpcode::kAdd, induction_variable, inc));
    // Update data GTE(1).
    auto data = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, loop_state, 1));
    // Use 'induction_variable' in computation with no path to output tuple.
    Shape f32_scalar_shape = ShapeUtil::MakeShape(F32, {});
    auto convert = builder.AddInstruction(
        HloInstruction::CreateConvert(f32_scalar_shape, induction_variable));
    auto update = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, convert, {}));
    auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, data, update));
    // Create output Tuple.
    builder.AddInstruction(HloInstruction::CreateTuple({add0, add1}));
    return builder.Build();
  }

  // Builds a While body computation with two output tuple elements dependent on
  // both input tuple elements.
  //
  // EX: Body({in0, in1, in2})
  //   out0 = Add(in0, 1)
  //   out1 = in1
  //   out2 = in2
  //   Tuple(out0, out1, out2)
  std::unique_ptr<HloComputation> BuildDependentBodyComputation2() {
    auto builder = HloComputation::Builder(TestName() + ".Body");

    const Shape& loop_state_shape = ShapeUtil::MakeTupleShape(
        {induction_variable_shape_, data_shape_, data_shape_});

    auto loop_state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape, "loop_state"));

    // Update the induction variable GTE(0).
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            induction_variable_shape_, loop_state, 0));
    auto inc = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));

    // add0 = Add(in0, 1)
    auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
        induction_variable->shape(), HloOpcode::kAdd, induction_variable, inc));
    // data1 = GTE(1).
    HloInstruction* data1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, loop_state, 1));

    // data2 = GTE(2).
    HloInstruction* data2 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, loop_state, 2));

    // Create output Tuple.
    builder.AddInstruction(HloInstruction::CreateTuple({add0, data1, data2}));

    return builder.Build();
  }

  // Builds a While body computation with read-only tuple element 0.
  // EX:
  // Body({in0, in1})
  //   out0 = in0
  //   out1 = Add(BCast(in0), in1)
  //   Tuple(out0, out1)
  std::unique_ptr<HloComputation> BuildDependentBodyOneReadOnlyComputation() {
    auto builder = HloComputation::Builder(TestName() + ".Body");
    // Create param instruction to access loop state.
    auto loop_state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape_, "loop_state"));
    // Update the induction variable GTE(0).
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            induction_variable_shape_, loop_state, 0));
    // Update data GTE(1).
    auto data = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, loop_state, 1));

    // Use 'induction_variable' in computation with no path to output tuple.
    Shape f32_scalar_shape = ShapeUtil::MakeShape(F32, {});
    auto convert = builder.AddInstruction(
        HloInstruction::CreateConvert(f32_scalar_shape, induction_variable));
    auto update = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, convert, {}));
    auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, data, update));
    // Create output Tuple.
    builder.AddInstruction(
        HloInstruction::CreateTuple({induction_variable, add1}));
    return builder.Build();
  }

  // Builds a While body computation with independent outputs.
  // EX:
  // Body({in0, in1})
  //   out0 = Add(in0, 1)
  //   out1 = Add(in1, {1, 1, 1, 1, 1, 1, 1, 1})
  //   Tuple(out0, out1)
  std::unique_ptr<HloComputation> BuildIndependentBodyComputation(
      bool nested = false) {
    auto builder = HloComputation::Builder(TestName() + ".Body");
    // Create param instruction to access loop state.
    const Shape& loop_state_shape =
        nested ? nested_loop_state_shape_ : loop_state_shape_;

    auto loop_state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape, "loop_state"));
    // Update the induction variable GTE(0).
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            induction_variable_shape_, loop_state, 0));
    auto inc = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
    // add0 = Add(in0, 1)
    auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
        induction_variable->shape(), HloOpcode::kAdd, induction_variable, inc));
    // Update data GTE(1).
    HloInstruction* data = nullptr;
    if (nested) {
      data = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          nested_tuple_shape_, loop_state, 1));
      data = builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(data_shape_, data, 0));
    } else {
      data = builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(data_shape_, loop_state, 1));
    }
    auto update = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(
            {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f})));
    // add1 = Add(in1, {1, 1, 1, 1, 1, 1, 1, 1})
    auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, data, update));
    // Create output Tuple.
    if (nested) {
      auto nested_tuple =
          builder.AddInstruction(HloInstruction::CreateTuple({add1, add1}));
      builder.AddInstruction(HloInstruction::CreateTuple({add0, nested_tuple}));
    } else {
      builder.AddInstruction(HloInstruction::CreateTuple({add0, add1}));
    }
    return builder.Build();
  }

  // Builds a While body computation with the following nested tuple
  // sub-computation:
  //                            |
  //                    GTE(loop_state, 1)
  //                       /           \
  // GTE(GTE(loop_state, 1), 0)     GTE(GTE(loop_state, 1), 1)
  //           |                              |
  //          Add                           Reverse
  //           |                              |
  std::unique_ptr<HloComputation> BuildNestedBodyComputation() {
    auto builder = HloComputation::Builder(TestName() + ".Body");
    // Create param instruction to access loop state.
    auto loop_state = builder.AddInstruction(HloInstruction::CreateParameter(
        0, nested_loop_state_shape_, "loop_state"));
    // Update GTE(0).
    auto gte0 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
        induction_variable_shape_, loop_state, 0));
    auto inc = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
    auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
        gte0->shape(), HloOpcode::kAdd, gte0, inc));

    // GTE(loop_state, 1)
    auto gte1 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
        nested_tuple_shape_, loop_state, 1));
    // GTE(GTE(loop_state, 1), 0) -> Add
    auto gte10 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, gte1, 0));
    auto update10 = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(
            {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f})));
    auto add10 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, gte10, update10));

    // GTE(GTE(loop_state, 1), 1) -> Reverse
    auto gte11 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, gte1, 1));
    auto rev11 = builder.AddInstruction(
        HloInstruction::CreateReverse(data_shape_, gte11, {0}));

    // Create output Tuple.
    auto inner_tuple =
        builder.AddInstruction(HloInstruction::CreateTuple({add10, rev11}));
    builder.AddInstruction(HloInstruction::CreateTuple({add0, inner_tuple}));
    return builder.Build();
  }

  // Builds a While instruction using 'condition' and 'body' sub-computations.
  // Init operand is initialized to zeros of appropriate shape.
  HloInstruction* BuildWhileInstruction(HloComputation* condition,
                                        HloComputation* body,
                                        bool nested = false) {
    auto builder = HloComputation::Builder(TestName() + ".While");
    auto induction_var_init = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));

    auto data_init = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(
            {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f})));

    if (nested) {
      auto inner_init = builder.AddInstruction(
          HloInstruction::CreateTuple({data_init, data_init}));
      auto loop_state_init = builder.AddInstruction(
          HloInstruction::CreateTuple({induction_var_init, inner_init}));
      auto while_hlo = builder.AddInstruction(HloInstruction::CreateWhile(
          loop_state_init->shape(), condition, body, loop_state_init));
      module_->AddEntryComputation(builder.Build());
      return while_hlo;
    }

    auto loop_state_init = builder.AddInstruction(
        HloInstruction::CreateTuple({induction_var_init, data_init}));
    auto while_hlo = builder.AddInstruction(HloInstruction::CreateWhile(
        loop_state_shape_, condition, body, loop_state_init));
    module_->AddEntryComputation(builder.Build());
    return while_hlo;
  }

  HloInstruction* BuildWhileInstruction_InitPointsToConstant() {
    auto builder = HloComputation::Builder(TestName() + ".While");
    auto data_init = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(
            {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f})));
    return BuildWhileInstructionWithCustomInit(loop_state_shape_, data_init,
                                               &builder);
  }

  HloInstruction* BuildWhileInstruction_InitPointsToParameter() {
    auto builder = HloComputation::Builder(TestName() + ".While");
    auto data_init = builder.AddInstruction(
        HloInstruction::CreateParameter(0, data_shape_, "data_init"));
    return BuildWhileInstructionWithCustomInit(loop_state_shape_, data_init,
                                               &builder);
  }

  HloInstruction* BuildWhileInstruction_InitPointsToAmbiguous() {
    auto builder = HloComputation::Builder(TestName() + ".While");

    auto one = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
    auto v1 = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, one, {}));
    auto zero = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
    auto v2 = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, zero, {}));

    auto tuple1 = builder.AddInstruction(HloInstruction::CreateTuple({v1, v2}));
    auto tuple2 = builder.AddInstruction(HloInstruction::CreateTuple({v2, v1}));

    auto pred = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
    auto data_init = builder.AddInstruction(HloInstruction::CreateTernary(
        nested_tuple_shape_, HloOpcode::kTupleSelect, pred, tuple1, tuple2));

    return BuildWhileInstructionWithCustomInit(nested_loop_state_shape_,
                                               data_init, &builder);
  }

  HloInstruction* BuildWhileInstruction_InitPointsToNonDistinct() {
    auto builder = HloComputation::Builder(TestName() + ".While");

    auto one = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
    auto one_vec = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, one, {}));
    auto data_init =
        builder.AddInstruction(HloInstruction::CreateTuple({one_vec, one_vec}));

    return BuildWhileInstructionWithCustomInit(nested_loop_state_shape_,
                                               data_init, &builder);
  }

  HloInstruction* BuildWhileInstruction_InitPointsToInterfering() {
    auto builder = HloComputation::Builder(TestName() + ".While");
    auto one = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
    auto data_init = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, one, {}));
    auto one_vec = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(
            {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f})));
    // Take a reference to 'data_init' to make it interfere with while result.
    auto add = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, data_init, one_vec));

    auto xla_while = BuildWhileInstructionWithCustomInit(loop_state_shape_,
                                                         data_init, &builder);

    // Add an additional binary operation operating on the while and the
    // interfering add so that neither operation is dead.
    auto gte = xla_while->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(
            ShapeUtil::GetSubshape(xla_while->shape(), {1}), xla_while, 1));
    auto sub = xla_while->parent()->AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kSubtract, add, gte));
    auto gte0 = xla_while->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(
            ShapeUtil::GetSubshape(xla_while->shape(), {0}), xla_while, 0));
    auto tuple = xla_while->parent()->AddInstruction(
        HloInstruction::CreateTuple({gte0, sub}));

    xla_while->parent()->set_root_instruction(tuple);

    return xla_while;
  }

  HloInstruction* BuildWhileInstructionWithCustomInit(
      const Shape& loop_state_shape, HloInstruction* data_init,
      HloComputation::Builder* builder) {
    const bool nested =
        ShapeUtil::Equal(loop_state_shape, nested_loop_state_shape_);
    auto induction_var_init = builder->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
    auto condition = module_->AddEmbeddedComputation(
        BuildConditionComputation(loop_state_shape));
    auto body = module_->AddEmbeddedComputation(
        BuildIndependentBodyComputation(nested));
    auto loop_state_init = builder->AddInstruction(
        HloInstruction::CreateTuple({induction_var_init, data_init}));
    auto while_hlo = builder->AddInstruction(HloInstruction::CreateWhile(
        loop_state_shape, condition, body, loop_state_init));
    module_->AddEntryComputation(builder->Build());
    return while_hlo;
  }

  std::unique_ptr<HloModule> module_;
  Shape induction_variable_shape_ = ShapeUtil::MakeShape(S32, {});
  Shape data_shape_ = ShapeUtil::MakeShape(F32, {8});
  Shape loop_state_shape_ =
      ShapeUtil::MakeTupleShape({induction_variable_shape_, data_shape_});
  Shape nested_tuple_shape_ =
      ShapeUtil::MakeTupleShape({data_shape_, data_shape_});
  Shape nested_loop_state_shape_ = ShapeUtil::MakeTupleShape(
      {induction_variable_shape_, nested_tuple_shape_});
  Shape condition_result_shape_ = ShapeUtil::MakeShape(PRED, {});
};

// Tests while body computation with independent tuple elements:
//
//   While.Body({in0, in1})
//     out0 = Add(in0, 1)
//     out1 = Add(in1, {1, 1, 1, 1, 1, 1, 1, 1})
//     Tuple(out0, out1)
//
// CopyInsertion pass should not generate any copies.
//
TEST_F(WhileCopyInsertionTest, IndependentTupleElements) {
  auto condition = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape_));
  auto body =
      module_->AddEmbeddedComputation(BuildIndependentBodyComputation());
  auto while_hlo = BuildWhileInstruction(condition, body);

  InsertCopies(module_.get());

  // Body should have no copies as the adds can be done inplace.
  EXPECT_EQ(CountCopies(*body), 0);
  EXPECT_EQ(CountControlEdges(*module_), 0);

  // Both init indices need copies as they are constants.
  EXPECT_THAT(while_hlo->operand(0),
              op::Tuple(op::Copy(op::Constant()), op::Copy(op::Constant())));
}

// Tests while body computation with dependent tuple elements:
//
//   While.Body({in0, in1})
//     out0 = Add(in0, 1)
//     out1 = Add(BCast(in0), in1)
//     Tuple(out0, out1)
//
// CopyInsertion pass should convert the root instruction to:
//
//     Tuple(Copy(out0), out1)
//
TEST_F(WhileCopyInsertionTest, DependentTupleElements) {
  auto condition = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape_));
  auto body = module_->AddEmbeddedComputation(BuildDependentBodyComputation());
  auto while_hlo = BuildWhileInstruction(condition, body);

  InsertCopies(module_.get());

  EXPECT_EQ(CountCopies(*body), 1);
  EXPECT_EQ(CountControlEdges(*body), 0);

  EXPECT_THAT(
      body->root_instruction(),
      op::Tuple(op::Add(), op::Add(op::GetTupleElement(), op::Broadcast())));

  auto add = body->root_instruction()->operand(0);
  auto bcast = body->root_instruction()->operand(1)->operand(1);
  ASSERT_EQ(add->opcode(), HloOpcode::kAdd);
  ASSERT_EQ(bcast->opcode(), HloOpcode::kBroadcast);

  EXPECT_THAT(while_hlo->while_body()->root_instruction(),
              op::Tuple(op::Add(op::Copy(), op::Constant()),
                        op::Add(op::GetTupleElement(),
                                op::Broadcast(op::Convert(op::Copy())))));

  // Both init indices need copies as they are constants.
  EXPECT_THAT(while_hlo->operand(0),
              op::Tuple(op::Copy(op::Constant()), op::Copy(op::Constant())));
}

// Tests while body computation with read-only tuple element 0:
//
//                         PARAMETER
//                         /       \
//                      GTE(0)     GTE(1)
//                        |  \      |
//                        |   BCAST |
//                        |      \  |
//                        |       ADD
//                        |        |
//                         \      /
//                           TUPLE (root)
//
// CopyInsertion pass should not generate any copies for the while body.
TEST_F(WhileCopyInsertionTest, DependentTupleElements_OneReadOnly) {
  auto condition = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape_));
  auto body = module_->AddEmbeddedComputation(
      BuildDependentBodyOneReadOnlyComputation());
  BuildWhileInstruction(condition, body);

  InsertCopies(module_.get());

  // No copies or control edges should be inserted. The body is legal as is.
  EXPECT_EQ(CountCopies(*body), 0);
  EXPECT_EQ(CountControlEdges(*body), 0);
}

// Same as above, but with two while loops, sharing entry parameters.
TEST_F(WhileCopyInsertionTest,
       DependentTupleElements_OneReadOnly_TwoLoops_EntryParams) {
  auto condition1 = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape_));
  auto condition2 = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape_));
  auto body1 = module_->AddEmbeddedComputation(
      BuildDependentBodyOneReadOnlyComputation());
  auto body2 = module_->AddEmbeddedComputation(
      BuildDependentBodyOneReadOnlyComputation());

  auto builder = HloComputation::Builder(TestName() + ".While");
  auto iter_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, induction_variable_shape_, "iter"));
  auto data_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, data_shape_, "data"));
  auto loop_init = builder.AddInstruction(
      HloInstruction::CreateTuple({iter_param, data_param}));

  auto while_hlo1 = builder.AddInstruction(HloInstruction::CreateWhile(
      loop_state_shape_, condition1, body1, loop_init));
  auto while_hlo2 = builder.AddInstruction(HloInstruction::CreateWhile(
      loop_state_shape_, condition2, body2, loop_init));

  // Add a couple elements from each of the while so both whiles are live.
  auto gte1 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetSubshape(while_hlo1->shape(), {0}), while_hlo1, 0));
  auto gte2 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetSubshape(while_hlo2->shape(), {0}), while_hlo2, 0));
  builder.AddInstruction(
      HloInstruction::CreateBinary(gte1->shape(), HloOpcode::kAdd, gte1, gte2));

  auto entry = module_->AddEntryComputation(builder.Build());

  InsertCopies(module_.get());

  // Neither body should have any copies or control edges in them.
  EXPECT_EQ(CountCopies(*body1), 0);
  EXPECT_EQ(CountCopies(*body2), 0);
  EXPECT_EQ(CountControlEdges(*body1), 0);
  EXPECT_EQ(CountControlEdges(*body2), 0);

  // Only two copies should be necessary. Each of the whiles should have
  // a copy of tuple element 1 (init value is a parameter, and the element is
  // not non-read-only) so each of the while bodies gets its own buffer to write
  // element 1 into.
  EXPECT_EQ(CountCopies(*entry), 2);

  EXPECT_EQ(while_hlo1->operand(0)->operand(1)->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(while_hlo2->operand(0)->operand(1)->opcode(), HloOpcode::kCopy);

  // The two copies of element 1 should be different.
  EXPECT_NE(while_hlo1->operand(0)->operand(1),
            while_hlo2->operand(0)->operand(1));
}

// Same as above, but with two while loops, sharing non-parameters.
TEST_F(WhileCopyInsertionTest,
       DependentTupleElements_OneReadOnly_TwoLoops_NonParams) {
  auto condition1 = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape_));
  auto condition2 = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape_));
  auto body1 = module_->AddEmbeddedComputation(
      BuildDependentBodyOneReadOnlyComputation());
  auto body2 = module_->AddEmbeddedComputation(
      BuildDependentBodyOneReadOnlyComputation());

  auto builder = HloComputation::Builder(TestName() + ".While");
  auto iter_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, induction_variable_shape_, "iter"));
  auto data_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, data_shape_, "data"));
  // Add dummy ops to ensure loop_init elements aren't entry parameters.
  Shape f32_scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto convert = builder.AddInstruction(
      HloInstruction::CreateConvert(f32_scalar_shape, iter_param));
  auto iter_value = builder.AddInstruction(
      HloInstruction::CreateUnary(convert->shape(), HloOpcode::kExp, convert));
  auto convert2 = builder.AddInstruction(
      HloInstruction::CreateConvert(induction_variable_shape_, iter_value));
  auto data_value = builder.AddInstruction(HloInstruction::CreateUnary(
      data_param->shape(), HloOpcode::kExp, data_param));
  auto loop_init = builder.AddInstruction(
      HloInstruction::CreateTuple({convert2, data_value}));

  auto while_hlo1 = builder.AddInstruction(HloInstruction::CreateWhile(
      loop_state_shape_, condition1, body1, loop_init));
  auto while_hlo2 = builder.AddInstruction(HloInstruction::CreateWhile(
      loop_state_shape_, condition2, body2, loop_init));

  // Add a couple elements from each of the while so both whiles are not dead.
  auto gte1 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetSubshape(while_hlo1->shape(), {0}), while_hlo1, 0));
  auto gte2 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetSubshape(while_hlo2->shape(), {0}), while_hlo2, 0));
  builder.AddInstruction(
      HloInstruction::CreateBinary(gte1->shape(), HloOpcode::kAdd, gte1, gte2));
  auto entry = module_->AddEntryComputation(builder.Build());

  InsertCopies(module_.get());

  // Ideally only one copy should be necessary. One of the whiles should
  // have a copy of tuple element 1 (the non-read-only element) so each of the
  // while bodies gets its own buffer to write element 1 into. However, the
  // analysis isn't perfect and adds an additional copy of element 0.
  EXPECT_EQ(CountCopies(*entry), 2);

  EXPECT_THAT(while_hlo1->operand(0),
              op::Tuple(op::Convert(op::Exp()), op::Copy(op::Exp())));
  EXPECT_THAT(while_hlo2->operand(0),
              op::Tuple(op::Convert(op::Exp()), op::Copy(op::Exp())));
}

// Tests while body computation with nested tuple elements:
//
//                            |
//                    GTE(loop_state, 1)
//                       /          \
// GTE(GTE(loop_state, 1), 0)     GTE(GTE(loop_state, 1), 1)
//           |                              |
//          Add                           Reverse
//           |                              |
//
// CopyInsertion pass will conceptually generate the following, but with the
// actual GTE and Tuple instructions optimized away:
//
//                    Tuple  // old root
//                   /     \
//                  /       \
//                GTE(0)   GTE(1)
//                  |       /  \
//                  |      /    \
//                  |    GTE(0) GTE(1)
//                  |       |    |
//                  |       |   Copy
//                  |       |    |
//                   \      |   /
//                    \    Tuple  // "inner" tuple.
//                     \    /
//                      \  /
//                     Tuple  // new root
//
TEST_F(WhileCopyInsertionTest, NestedTupleElements) {
  auto condition = module_->AddEmbeddedComputation(
      BuildConditionComputation(nested_loop_state_shape_));
  auto body = module_->AddEmbeddedComputation(BuildNestedBodyComputation());
  BuildWhileInstruction(condition, body, true);

  //  HloInstruction* old_root = body->root_instruction();
  InsertCopies(module_.get());

  // The only copy necessary is for the kReverse as it cannot be done
  // in-place (instruction can share buffer with operand). The other elements of
  // the loop state are kAdd instructions which can be done in-place.
  EXPECT_EQ(CountCopies(*body), 1);

  // Each element of the init needs a copy as all are constants.
  EXPECT_EQ(CountCopies(*module_), 4);

  // Either the kReverse itself must be copied or the operand of the kReverse
  // must be copied.
  if (body->root_instruction()->operand(1)->operand(1)->opcode() ==
      HloOpcode::kCopy) {
    EXPECT_THAT(
        body->root_instruction(),
        op::Tuple(op::Add(), op::Tuple(op::Add(), op::Copy(op::Reverse()))));
  } else {
    EXPECT_THAT(
        body->root_instruction(),
        op::Tuple(op::Add(), op::Tuple(op::Add(), op::Reverse(op::Copy()))));
  }
}

// Tests while init instruction which points-to a constant.
//
//     init = Tuple(Constant(S32, {}), Constant(F32, {8}))
//
// CopyInsertion pass should add copies for both constants.
//
TEST_F(WhileCopyInsertionTest, InitPointsToConstant) {
  auto while_hlo = BuildWhileInstruction_InitPointsToConstant();

  InsertCopies(module_.get());
  EXPECT_EQ(CountCopies(*while_hlo->while_body()), 0);
  EXPECT_EQ(CountCopies(*module_), 2);

  EXPECT_THAT(while_hlo->operand(0),
              op::Tuple(op::Copy(op::Constant()), op::Copy(op::Constant())));
}

// Tests while init instruction which points-to a parameter.
//
//     init = Tuple(Constant(S32, {}), Parameter(F32, {8}))
//
// CopyInsertion pass should add copies for both the constant and parameter.
//
TEST_F(WhileCopyInsertionTest, InitPointsToParameter) {
  auto while_hlo = BuildWhileInstruction_InitPointsToParameter();

  InsertCopies(module_.get());
  EXPECT_EQ(CountCopies(*while_hlo->while_body()), 0);
  EXPECT_EQ(CountCopies(*module_), 2);

  EXPECT_THAT(while_hlo->operand(0),
              op::Tuple(op::Copy(op::Constant()), op::Copy(op::Parameter())));
}

// Tests while init instruction which has an ambiguous points-to set.
//
//     select = Select(pred, tuple1, tuple2)
//     init = Tuple(Constant(S32, {}), Parameter(F32, {8}))
//
// CopyInsertion pass will conceptually generate the following, but with some of
// the actual GTE and Tuple instructions optimized away:
//
//                    Tuple  // old init
//                   /     \
//                  /       \
//                GTE(0)   GTE(1)
//                  |       /  \
//                  |      /    \
//                  |    GTE(0) GTE(1)
//                  |       |    |
//                Copy   Copy   Copy
//                  |       |    |
//                   \      |   /
//                    \    Tuple
//                     \    /
//                      \  /
//                     Tuple  // new init
//
TEST_F(WhileCopyInsertionTest, InitPointsToAmbiguous) {
  auto while_hlo = BuildWhileInstruction_InitPointsToAmbiguous();

  InsertCopies(module_.get());
  EXPECT_EQ(CountCopies(*module_), 4);
  // The entry computation requires three copies to resolve the ambiguity of two
  // init elements and the constant passed in as one of the init elements.
  EXPECT_EQ(CountCopies(*module_->entry_computation()), 3);
  EXPECT_THAT(while_hlo->operand(0),
              op::Tuple(op::Copy(op::Constant()),
                        op::Tuple(op::Copy(op::GetTupleElement()),
                                  op::Copy(op::GetTupleElement()))));

  // The body requires one copy because the buffer set is not distinct: the
  // result of one of the adds is written into two elements of the output of the
  // loop body. Either element might be copied.
  EXPECT_EQ(CountCopies(*while_hlo->while_body()), 1);
  if (while_hlo->while_body()
          ->root_instruction()
          ->operand(1)
          ->operand(0)
          ->opcode() == HloOpcode::kCopy) {
    EXPECT_THAT(
        while_hlo->while_body()->root_instruction(),
        op::Tuple(op::Add(), op::Tuple(op::Copy(op::Add()), op::Add())));
  } else {
    EXPECT_THAT(
        while_hlo->while_body()->root_instruction(),
        op::Tuple(op::Add(), op::Tuple(op::Add(), op::Copy(op::Add()))));
  }
}

// Tests while init instruction which has a non-distinct points-to set.
//
//     init = Tuple(Constant(S32, {}), Tuple({vec_one, vec_one}))
//
// CopyInsertion pass will conceptually generate the following, but with some of
// the actual GTE and Tuple instructions optimized away:
//
//                    Tuple  // old init
//                   /     \
//                  /       \
//                GTE(0)   GTE(1)
//                  |       /  \
//                  |      /    \
//                  |    GTE(0) GTE(1)
//                  |       |    |
//                Copy   Copy   Copy
//                  |       |    |
//                   \      |   /
//                    \    Tuple
//                     \    /
//                      \  /
//                     Tuple  // new init
//
TEST_F(WhileCopyInsertionTest, InitPointsToNonDistinct) {
  auto while_hlo = BuildWhileInstruction_InitPointsToNonDistinct();

  InsertCopies(module_.get());

  // The entry computation requires two copies to resolve the non-distinctness of
  // two init elements and the constant passed in as one of the init
  // elements. Either element can be copied for the distinctness issue.
  EXPECT_EQ(CountCopies(*module_->entry_computation()), 2);
  if (while_hlo->operand(0)->operand(1)->operand(0)->opcode() ==
      HloOpcode::kCopy) {
    EXPECT_THAT(
        while_hlo->operand(0),
        op::Tuple(op::Copy(op::Constant()),
                  op::Tuple(op::Copy(op::Broadcast()), op::Broadcast())));
  } else {
    EXPECT_THAT(
        while_hlo->operand(0),
        op::Tuple(op::Copy(op::Constant()),
                  op::Tuple(op::Broadcast(), op::Copy(op::Broadcast()))));
  }

  // The body requires one copy because the buffer set is not distinct: the
  // result of one of the adds is written into two elements of the output of the
  // loop body. Either element might be copied.
  EXPECT_EQ(CountCopies(*while_hlo->while_body()), 1);
  if (while_hlo->while_body()
          ->root_instruction()
          ->operand(1)
          ->operand(0)
          ->opcode() == HloOpcode::kCopy) {
    EXPECT_THAT(
        while_hlo->while_body()->root_instruction(),
        op::Tuple(op::Add(), op::Tuple(op::Copy(op::Add()), op::Add())));
  } else {
    EXPECT_THAT(
        while_hlo->while_body()->root_instruction(),
        op::Tuple(op::Add(), op::Tuple(op::Add(), op::Copy(op::Add()))));
  }
}

// Tests while init instruction buffer which interferes with while result
// buffer.
//
//     init_data = Broadcast(...)
//     add_unrelated = Add(init_data) // takes a reference to cause interference
//     init = Tuple(Constant(S32, {}), init_data))
//
// CopyInsertion pass should copy both operands.
//
TEST_F(WhileCopyInsertionTest, InitPointsToInterfering) {
  auto while_hlo = BuildWhileInstruction_InitPointsToInterfering();

  InsertCopies(module_.get());
  EXPECT_EQ(CountCopies(*module_), 2);
  EXPECT_EQ(CountCopies(*while_hlo->while_body()), 0);

  EXPECT_THAT(while_hlo->operand(0),
              op::Tuple(op::Copy(op::Constant()), op::Copy(op::Broadcast())));
}

// Tests while init instruction buffer which has a non-distinct points-to set:
//
//     init = Tuple(Parameter(S32, {}), Parameter(F32, {8},
//                  Parameter(F32, {8})))
//
// where the second and third parameters are identical *and* the tuple shared
// by another while instruction.
//
// Verifies that the resulting point-to set is distinct in the resulting Tuple
// (non-identical Copys). In other words, verifies that copy sharing does not
// insert identical copies to the resulting tuple.
TEST_F(WhileCopyInsertionTest, InitPointsToNonDistinctUsedByTwoWhileLoops) {
  // Loop body that outputs tuple comprises two elements dependent on the init
  // tuple.
  const Shape& loop_state_shape = ShapeUtil::MakeTupleShape(
      {induction_variable_shape_, data_shape_, data_shape_});

  auto condition1 = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape));
  auto condition2 = module_->AddEmbeddedComputation(
      BuildConditionComputation(loop_state_shape));
  auto body1 =
      module_->AddEmbeddedComputation(BuildDependentBodyComputation2());
  auto body2 =
      module_->AddEmbeddedComputation(BuildDependentBodyComputation2());

  auto builder = HloComputation::Builder(TestName() + ".While");

  auto iter_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, induction_variable_shape_, "iter"));
  auto data_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, data_shape_, "data"));

  // Loop init tuple contains two identical parameter buffers.
  auto loop_init = builder.AddInstruction(
      HloInstruction::CreateTuple({iter_param, data_param, data_param}));

  // Two while loops shares the same loop init tuple.
  auto while_hlo1 = builder.AddInstruction(HloInstruction::CreateWhile(
      loop_state_shape, condition1, body1, loop_init));
  auto while_hlo2 = builder.AddInstruction(HloInstruction::CreateWhile(
      loop_state_shape, condition2, body2, loop_init));

  // Add add instruction so neither while is dead.
  auto gte1 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetSubshape(while_hlo1->shape(), {0}), while_hlo1, 0));
  auto gte2 = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetSubshape(while_hlo1->shape(), {0}), while_hlo2, 0));
  builder.AddInstruction(
      HloInstruction::CreateBinary(gte1->shape(), HloOpcode::kAdd, gte1, gte2));

  module_->AddEntryComputation(builder.Build());

  InsertCopies(module_.get());

  // None of the bodies should have copies or control flow edges.
  EXPECT_EQ(CountCopies(*body1), 0);
  EXPECT_EQ(CountCopies(*body2), 0);

  // The loop bodies pass through elements 1 and 2 in the init tuple, so ideally
  // these should not need to be copied before either while. However, copy
  // insertion is not able to reason about the transparency of elements through
  // while bodies in all circumstances so extra copies are added (b/xxx).
  EXPECT_EQ(CountCopies(*module_->entry_computation()), 2);

  EXPECT_THAT(while_hlo1->operand(0),
              op::Tuple(op::Copy(), op::Parameter(), op::Parameter()));
  EXPECT_THAT(while_hlo2->operand(0),
              op::Tuple(op::Copy(), op::Parameter(), op::Parameter()));
}

TEST_F(CopyInsertionTest, SwizzlingWhile) {
  // Test a while instruction with a body which permutes its tuple parameter
  // elements.
  auto module = CreateNewVerifiedModule();
  const Shape loop_state_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  // Body simply interchanges the two tuple elements in the loop state.
  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "param"));
  auto body_element_0 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 0));
  auto body_element_1 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 1));
  body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_element_1, body_element_0}));
  HloComputation* body = module->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "param"));
  auto cond_constant = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  cond_builder.AddInstruction(HloInstruction::CreateUnary(
      cond_constant->shape(), HloOpcode::kNot, cond_constant));
  HloComputation* condition =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape, condition, body, tuple));
  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 6);

  // The loop state elements should be copied at the parameter and at the root
  // with a control edge in between (see DeepCopyAndAddControlEdges). This is
  // technically one more copy than is strictly necessary, but in order to have
  // only three copies the copies of different loop state elements must be
  // ordered with a control edge.
  EXPECT_EQ(CountCopies(*body), 4);
  EXPECT_EQ(CountControlEdges(*body), 2);

  EXPECT_THAT(body->root_instruction(),
              op::Tuple(op::Copy(op::Copy()), op::Copy(op::Copy())));

  EXPECT_EQ(CountCopies(*module->entry_computation()), 2);
  EXPECT_THAT(xla_while->operand(0), op::Tuple(op::Copy(), op::Copy()));
}

TEST_F(CopyInsertionTest, CrossingParameters) {
  // Test a case where two parameters' dataflow cross with each other while
  // input and output are aliased with same index:
  //
  //  (p0 ,  p1)
  //   | \   /|
  //   |  \ / |
  // alias X  alias
  //   |  / \ |
  //   | /   \|
  //  (p1  ,  p0)
  auto module = CreateNewVerifiedModule();
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "0"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 1));
  builder.AddInstruction(HloInstruction::CreateTuple({gte1, gte0}));
  module->AddEntryComputation(builder.Build());
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{0},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kUserAlias));
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{1}, /*param_number=*/0, /*param_index=*/{1},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kUserAlias));
  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 4);
}

TEST_F(CopyInsertionTest, ParametersAliasing) {
  // Test a case where two parameters' dataflow don't interfere with each other
  // while aliased.
  //
  //  (p0 ,  p1)
  //   |      |
  //   |      |
  // alias   alias
  //   |      |
  //   |      |
  //  (p0 ,  p1)
  auto module = CreateNewVerifiedModule();
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 1));
  builder.AddInstruction(HloInstruction::CreateTuple({gte0, gte1}));
  module->AddEntryComputation(builder.Build());
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{0},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kUserAlias));
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{1}, /*param_number=*/0, /*param_index=*/{1},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kUserAlias));
  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 0);
}

TEST_F(CopyInsertionTest, ParameterWithNoAliasing) {
  // Test a case where no parameter is aliased with result. In this case, copy
  // should be added
  //
  //  (p0 ,  p1)
  //   |      |
  //   |      |
  //   |      |
  //   |      |
  //   |      |
  //  (p0 ,  p1)
  auto module = CreateNewVerifiedModule();
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 1));
  builder.AddInstruction(HloInstruction::CreateTuple({gte0, gte1}));
  module->AddEntryComputation(builder.Build());
  InsertCopies(module.get());

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Copy(op::GetTupleElement(param, 0)),
                        op::Copy(op::GetTupleElement(param, 1))));

  EXPECT_EQ(CountCopies(*module), 2);
}

TEST_F(CopyInsertionTest, ParameterWithPartialAliasing) {
  // Test a case where one parameter is aliased with result while another one
  // isn't.
  //
  //  (p0 ,  p1)
  //   |      |
  //   |      |
  // alias    |
  //   |      |
  //   |      |
  //  (p0 ,  p1)
  auto module = CreateNewVerifiedModule();
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 1));
  builder.AddInstruction(HloInstruction::CreateTuple({gte0, gte1}));
  module->AddEntryComputation(builder.Build());
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{0},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kUserAlias));
  InsertCopies(module.get());

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::GetTupleElement(param, 0),
                        op::Copy(op::GetTupleElement(param, 1))));

  EXPECT_EQ(CountCopies(*module), 1);
}

TEST_F(CopyInsertionTest, ParameterAndParallelOpsWithPartialAliasing) {
  // Test a case where one parameter is aliased with result while another one
  // isn't.
  //
  //   +-- (p0 ,  p1)
  //   |    |      |
  //   |    |      |
  // alias Negate  Negate
  //   |    |      |
  //   |    |      |
  //   +-- (p0 ,  p1)
  auto module = CreateNewVerifiedModule();
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 1));

  auto negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape_, HloOpcode::kNegate, gte0));

  auto negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape_, HloOpcode::kNegate, gte1));
  builder.AddInstruction(HloInstruction::CreateTuple({negate0, negate1}));
  module->AddEntryComputation(builder.Build());
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{0},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kUserAlias));
  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 0);
}

TEST_F(CopyInsertionTest, ParameterAndOpsWithPartialAliasing) {
  // Test a case where one parameter is aliased with result while another one
  // isn't.
  //
  //   +-- (p0 ,  p1)
  //   |    |      |
  //   |    |      |
  // alias Negate  Negate
  //   |    |      |
  //   |    Add----+
  //   |    |      |
  //   +-- (p0 ,  p1)
  auto module = CreateNewVerifiedModule();
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "p0"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, param, 1));

  auto negate0 = builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape_, HloOpcode::kNegate, gte0));

  auto negate1 = builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape_, HloOpcode::kNegate, gte1));

  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, negate0, negate1));
  builder.AddInstruction(HloInstruction::CreateTuple({add, negate1}));
  module->AddEntryComputation(builder.Build());
  ASSERT_IS_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{0},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kUserAlias));
  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 0);
}

TEST_F(CopyInsertionTest, SwizzlingWhileWithOneOp) {
  // Test a while instruction with a body which permutes its tuple parameter
  // elements and applies one operation to one of the elements. The addition of
  // the operation (instruction) on the element makes the live range of the
  // respective input and output elements different than if the instruction were
  // not there (as in the SwizzlingWhile test above).
  auto module = CreateNewVerifiedModule();
  const Shape loop_state_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  // Body interchanges the two tuple elements in the loop state and negates one
  // of them.
  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "param"));
  auto body_element_0 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 0));
  auto body_element_1 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 1));
  auto negate = body_builder.AddInstruction(HloInstruction::CreateUnary(
      scalar_shape_, HloOpcode::kNegate, body_element_1));
  body_builder.AddInstruction(
      HloInstruction::CreateTuple({negate, body_element_0}));
  HloComputation* body = module->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "param"));
  auto cond_constant = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  cond_builder.AddInstruction(HloInstruction::CreateUnary(
      cond_constant->shape(), HloOpcode::kNot, cond_constant));
  HloComputation* condition =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant1, constant2}));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape, condition, body, tuple));
  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 6);

  // The loop state elements should be copied at the parameter and at the root
  // with a control edge in between (see DeepCopyAndAddControlEdges).
  EXPECT_EQ(CountCopies(*body), 4);
  EXPECT_EQ(CountControlEdges(*body), 2);

  EXPECT_THAT(
      body->root_instruction(),
      op::Tuple(op::Copy(op::Negate(op::Copy())), op::Copy(op::Copy())));

  EXPECT_EQ(CountCopies(*module->entry_computation()), 2);
  EXPECT_THAT(xla_while->operand(0), op::Tuple(op::Copy(), op::Copy()));
}

TEST_F(CopyInsertionTest, SwizzlingWhileSharedInput) {
  // Test a while instruction with a body which permutes it's tuple parameter
  // elements similar to SwizzlinWhile above. However, in this test the input to
  // the while body is a single constant (both loop state elements are the same
  // constant). This means no copies are necessary because both loop state
  // elements are the same so interchanging them is a no-op.
  auto module = CreateNewVerifiedModule();
  const Shape loop_state_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_});

  // Body simply interchanges the two tuple elements in the loop state.
  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "param"));
  auto body_element_0 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 0));
  auto body_element_1 = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape_, body_param, 1));
  body_builder.AddInstruction(
      HloInstruction::CreateTuple({body_element_1, body_element_0}));
  HloComputation* body = module->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "param"));
  auto cond_constant = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  cond_builder.AddInstruction(HloInstruction::CreateUnary(
      cond_constant->shape(), HloOpcode::kNot, cond_constant));
  HloComputation* condition =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({constant, constant}));
  builder.AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape, condition, body, tuple));
  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 2);
  EXPECT_EQ(CountCopies(*body), 0);

  EXPECT_EQ(CountCopies(*module->entry_computation()), 2);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Copy(), op::Copy()));
}

TEST_F(CopyInsertionTest, SequentialWhiles) {
  // Construct a computation with a series of sequential while instructions
  // containing four loop state elements:
  //
  //   element 0 is passed to each while directly from an entry parameter.
  //
  //   element 1 is passed transparently in series through all the while bodies.
  //
  //   element 2 is negated in each while body. (in-place possible)
  //
  //   element 3 is reversed in each while body. (in-place not possible)
  //
  const Shape element_shape = ShapeUtil::MakeShape(F32, {42});
  const Shape loop_state_shape = ShapeUtil::MakeTupleShape(
      {element_shape, element_shape, element_shape, element_shape});

  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param_0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, element_shape, "param_0"));
  auto param_1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, element_shape, "param_1"));
  auto param_2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, element_shape, "param_2"));
  auto param_3 = builder.AddInstruction(
      HloInstruction::CreateParameter(3, element_shape, "param_3"));

  // The number of sequential kWhile instructions.
  const int kNumWhiles = 3;

  HloInstruction* prev_element_1 = param_1;
  HloInstruction* prev_element_2 = param_2;
  HloInstruction* prev_element_3 = param_3;

  // Vector containing all of the while instructions.
  std::vector<const HloInstruction*> whiles;
  for (int i = 0; i < kNumWhiles; ++i) {
    auto body_builder = HloComputation::Builder("body");
    auto body_param = body_builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape, "param"));
    auto body_element_0 = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(element_shape, body_param, 0));
    auto body_element_1 = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(element_shape, body_param, 1));
    auto body_element_2 = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(element_shape, body_param, 2));
    auto body_element_3 = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(element_shape, body_param, 3));
    auto negate = body_builder.AddInstruction(HloInstruction::CreateUnary(
        element_shape, HloOpcode::kNegate, body_element_2));
    auto reverse = body_builder.AddInstruction(
        HloInstruction::CreateReverse(element_shape, body_element_3, {0}));
    body_builder.AddInstruction(HloInstruction::CreateTuple(
        {body_element_0, body_element_1, negate, reverse}));
    HloComputation* body = module->AddEmbeddedComputation(body_builder.Build());

    auto cond_builder = HloComputation::Builder("condition");
    cond_builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape, "param"));
    auto cond_constant = cond_builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
    cond_builder.AddInstruction(HloInstruction::CreateUnary(
        cond_constant->shape(), HloOpcode::kNot, cond_constant));
    HloComputation* condition =
        module->AddEmbeddedComputation(cond_builder.Build());

    auto while_init = builder.AddInstruction(HloInstruction::CreateTuple(
        {param_0, prev_element_1, prev_element_2, prev_element_3}));

    auto xla_while = builder.AddInstruction(HloInstruction::CreateWhile(
        loop_state_shape, condition, body, while_init));
    whiles.push_back(xla_while);
    if (i != kNumWhiles - 1) {
      prev_element_1 = builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(element_shape, xla_while, 1));
      prev_element_2 = builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(element_shape, xla_while, 2));
      prev_element_3 = builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(element_shape, xla_while, 3));
    }
  }

  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());

  // Each while body has one copy. And each loop state element is copied once in
  // the entry computation.
  EXPECT_EQ(CountCopies(*module), 4 + kNumWhiles);

  // Each while body should have exactly one copy for element three which is an
  // op (kReverse) which cannot be done in place.
  for (const HloInstruction* xla_while : whiles) {
    EXPECT_EQ(CountCopies(*xla_while->while_body()), 1);
  }

  EXPECT_THAT(whiles[0]->operand(0), op::Tuple(op::Parameter(), op::Parameter(),
                                               op::Copy(), op::Copy()));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Copy(), op::Copy(), op::GetTupleElement(),
                        op::GetTupleElement()));
}

TEST_F(CopyInsertionTest, WhileBodyWithConstantRoot) {
  // Test a while body and condition which are each simply a constant (root of
  // computation is a constant). The body constant should be copied.
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param_0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param_0"));

  auto body_builder = HloComputation::Builder("body");
  body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(123.0)));
  HloComputation* body = module->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* condition =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(scalar_shape_, condition, body, param_0));

  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 2);

  EXPECT_THAT(xla_while->operand(0), op::Copy(op::Parameter()));
  EXPECT_THAT(body->root_instruction(), op::Copy(op::Constant()));
  EXPECT_THAT(condition->root_instruction(), op::Constant());
}

TEST_F(CopyInsertionTest, TokensShouldNotBeCopied) {
  string module_string = R"(
HloModule TokensShouldNotBeCopied

%Body (param.1: (s32[], token[])) -> (s32[], token[]) {
  %param.1 = (s32[], token[]) parameter(0)
  %get-tuple-element.1 = s32[] get-tuple-element((s32[], token[]) %param.1), index=0
  %constant.1 = s32[] constant(1)
  %add = s32[] add(s32[] %get-tuple-element.1, s32[] %constant.1)
  %get-tuple-element.2 = token[] get-tuple-element((s32[], token[]) %param.1), index=1
  %after-all = token[] after-all(token[] %get-tuple-element.2)
  ROOT %tuple = (s32[], token[]) tuple(s32[] %add, token[] %after-all)
}

%Cond (param: (s32[], token[])) -> pred[] {
  %param = (s32[], token[]) parameter(0)
  %get-tuple-element = s32[] get-tuple-element((s32[], token[]) %param), index=0
  %constant = s32[] constant(42)
  ROOT %less-than = pred[] compare(s32[] %get-tuple-element, s32[] %constant), direction=LT
}

ENTRY %TokensShouldNotBeCopied () -> s32[] {
  %one = s32[] constant(1)
  %negative_one = s32[] negate(%one)
  %init_token = token[] after-all()
  %init_tuple = (s32[], token[]) tuple(s32[] %negative_one, token[] %init_token)
  %while = (s32[], token[]) while((s32[], token[]) %init_tuple), condition=%Cond, body=%Body
  ROOT %root = s32[] get-tuple-element((s32[], token[]) %while), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  InsertCopies(module.get());

  // There should be no copies added because tokens should not be copied.
  EXPECT_EQ(CountCopies(*module), 0);
}

std::unique_ptr<HloComputation> MakeTrivialCondition(const Shape& shape) {
  auto builder = HloComputation::Builder("trivial_condition");
  builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "loop_state"));
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kNot, constant));
  return builder.Build();
}

std::unique_ptr<HloComputation> MakeBenchmarkWhileBody() {
  auto builder = HloComputation::Builder("benchmark_loop_body");
  const Shape element_shape = ShapeUtil::MakeShape(F32, {42});
  const Shape loop_state_shape =
      ShapeUtil::MakeTupleShape({element_shape, element_shape, element_shape});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "loop_state"));
  HloInstruction* element_0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(element_shape, param, 0));
  HloInstruction* element_1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(element_shape, param, 1));
  HloInstruction* element_2 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(element_shape, param, 2));

  HloInstruction* rev_1 = builder.AddInstruction(
      HloInstruction::CreateReverse(element_shape, element_1, {0}));
  HloInstruction* add_1_2 = builder.AddInstruction(HloInstruction::CreateBinary(
      element_shape, HloOpcode::kAdd, element_1, element_2));

  builder.AddInstruction(
      HloInstruction::CreateTuple({element_0, rev_1, add_1_2}));
  return builder.Build();
}

void BM_SequentialWhiles(int num_iters, int num_whiles) {
  // This benchmark constructs a chain of sequential while instructions.
  tensorflow::testing::StopTiming();
  for (int i = 0; i < num_iters; ++i) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsFromFlags());
    HloModule module("BM_SequentialWhiles", config);

    auto builder = HloComputation::Builder("BM_SequentialWhiles");
    HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(F32, {42}), "x"));
    HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
        1, ShapeUtil::MakeShape(F32, {42}), "y"));
    HloInstruction* z = builder.AddInstruction(HloInstruction::CreateParameter(
        2, ShapeUtil::MakeShape(F32, {42}), "z"));
    HloInstruction* init =
        builder.AddInstruction(HloInstruction::CreateTuple({x, y, z}));

    HloInstruction* prev_loop_state = init;
    for (int w = 0; w < num_whiles; ++w) {
      HloComputation* condition =
          module.AddEmbeddedComputation(MakeTrivialCondition(init->shape()));
      HloComputation* body =
          module.AddEmbeddedComputation(MakeBenchmarkWhileBody());
      prev_loop_state = builder.AddInstruction(HloInstruction::CreateWhile(
          init->shape(), condition, body, prev_loop_state));
    }
    module.AddEntryComputation(builder.Build());

    CopyInsertion copy_insertion;

    tensorflow::testing::StartTiming();
    ASSERT_IS_OK(copy_insertion.Run(&module).status());
    tensorflow::testing::StopTiming();

    // The entry computation should have three copies, and each body has one.
    ASSERT_EQ(CountCopies(module), 3 + num_whiles);
  }
}

void BM_ParallelWhiles(int num_iters, int num_whiles) {
  // This benchmark constructs a fan-out of parallel while instructions.
  tensorflow::testing::StopTiming();
  for (int i = 0; i < num_iters; ++i) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsFromFlags());
    HloModule module("BM_SequentialWhiles", config);

    auto builder = HloComputation::Builder("BM_ParallelWhiles");
    HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(F32, {42}), "x"));
    HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
        1, ShapeUtil::MakeShape(F32, {42}), "y"));
    HloInstruction* z = builder.AddInstruction(HloInstruction::CreateParameter(
        2, ShapeUtil::MakeShape(F32, {42}), "z"));
    HloInstruction* init =
        builder.AddInstruction(HloInstruction::CreateTuple({x, y, z}));

    HloInstruction* sum = nullptr;
    for (int w = 0; w < num_whiles; ++w) {
      HloComputation* condition =
          module.AddEmbeddedComputation(MakeTrivialCondition(init->shape()));
      HloComputation* body =
          module.AddEmbeddedComputation(MakeBenchmarkWhileBody());

      HloInstruction* xla_while = builder.AddInstruction(
          HloInstruction::CreateWhile(init->shape(), condition, body, init));

      if (sum == nullptr) {
        sum = builder.AddInstruction(
            HloInstruction::CreateGetTupleElement(x->shape(), xla_while, 0));
      } else {
        HloInstruction* element_0 = builder.AddInstruction(
            HloInstruction::CreateGetTupleElement(x->shape(), xla_while, 0));
        sum = builder.AddInstruction(HloInstruction::CreateBinary(
            x->shape(), HloOpcode::kAdd, sum, element_0));
      }
    }
    module.AddEntryComputation(builder.Build());

    CopyInsertion copy_insertion;

    tensorflow::testing::StartTiming();
    ASSERT_IS_OK(copy_insertion.Run(&module).status());
    tensorflow::testing::StopTiming();

    // Each body receives of copy of two of the parameters (the corresponding
    // elements in the body are modified), and there is one copy in each body.
    ASSERT_EQ(CountCopies(module), 3 * num_whiles);
  }
}

std::unique_ptr<HloComputation> MakeBenchmarkWhileBody(
    const int num_tuple_inputs) {
  auto builder = HloComputation::Builder("benchmark_loop_body");
  const Shape element_shape = ShapeUtil::MakeShape(F32, {});
  std::vector<Shape> input_shape(num_tuple_inputs, element_shape);
  const Shape loop_state_shape = ShapeUtil::MakeTupleShape(input_shape);
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "loop_state"));
  std::vector<HloInstruction*> gte_nodes(num_tuple_inputs);
  for (int i = 0; i < num_tuple_inputs; ++i) {
    gte_nodes[i] = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(element_shape, param, i));
  }
  builder.AddInstruction(HloInstruction::CreateTuple(gte_nodes));
  return builder.Build();
}

void BM_ManyElementTuple(int num_iters, const int num_tuple_inputs) {
  tensorflow::testing::StopTiming();
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsFromFlags());
  CopyInsertion copy_insertion;
  const Shape element_shape = ShapeUtil::MakeShape(F32, {});
  std::vector<HloInstruction*> tuple_params(num_tuple_inputs);
  for (int i = 0; i < num_iters; ++i) {
    auto builder = HloComputation::Builder("BM_ParallelWhiles");
    HloModule module("BM_ManyElementTuple", config);
    for (int j = 0; j < num_tuple_inputs; ++j) {
      tuple_params[j] = builder.AddInstruction(
          HloInstruction::CreateParameter(j, element_shape, ""));
    }
    HloInstruction* init =
        builder.AddInstruction(HloInstruction::CreateTuple(tuple_params));
    HloComputation* condition =
        module.AddEmbeddedComputation(MakeTrivialCondition(init->shape()));
    HloComputation* body =
        module.AddEmbeddedComputation(MakeBenchmarkWhileBody(num_tuple_inputs));
    HloInstruction* xla_while = builder.AddInstruction(
        HloInstruction::CreateWhile(init->shape(), condition, body, init));
    builder.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(F32, {}), xla_while, 0));
    module.AddEntryComputation(builder.Build());
    tensorflow::testing::StartTiming();
    ASSERT_IS_OK(copy_insertion.Run(&module).status());
    tensorflow::testing::StopTiming();
  }
}

BENCHMARK(BM_SequentialWhiles)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096);
BENCHMARK(BM_ParallelWhiles)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096);
BENCHMARK(BM_ManyElementTuple)->Arg(1024)->Arg(12288);

TEST_F(CopyInsertionTest, SimpleControlFlowTest) {
  const string& hlo_string = R"(
HloModule TestModule

if-body.v5 {
  constant.3 = s32[] constant(-1)
  p.1 = (s32[], (s32[], s32[], s32[]), (s32[])) parameter(0)
  get-tuple-element.18 = (s32[], s32[], s32[]) get-tuple-element(p.1), index=1
  get-tuple-element.65 = s32[] get-tuple-element(get-tuple-element.18), index=0
  get-tuple-element.66 = s32[] get-tuple-element(get-tuple-element.18), index=1
  add.3 = s32[] add(get-tuple-element.65, get-tuple-element.66)
  tuple.33 = (s32[]) tuple(add.3)
  ROOT tuple.34 = (s32[], (s32[], s32[], s32[]), (s32[])) tuple(constant.3, get-tuple-element.18, tuple.33)
}

if-condition.v4 {
  p.2 = (s32[], (s32[], s32[], s32[]), (s32[])) parameter(0)
  get-tuple-element.67 = s32[] get-tuple-element(p.2), index=0
  constant.4 = s32[] constant(0)
  ROOT equal-to = pred[] compare(get-tuple-element.67, constant.4), direction=EQ
}

_functionalize_body_1__.v28 {
  arg_tuple.4 = (s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.68 = s32[] get-tuple-element(arg_tuple.4), index=0
  constant.7 = s32[] constant(1)
  add.4 = s32[] add(get-tuple-element.68, constant.7)
  get-tuple-element.69 = s32[] get-tuple-element(arg_tuple.4), index=1
  get-tuple-element.70 = s32[] get-tuple-element(arg_tuple.4), index=2
  less-than-or-equal-to = pred[] compare(get-tuple-element.69, get-tuple-element.70), direction=LE
  constant.8 = s32[] constant(0)
  select = s32[] select(less-than-or-equal-to, constant.8, constant.7)
  get-tuple-element.71 = s32[] get-tuple-element(arg_tuple.4), index=3
  tuple.35 = (s32[], s32[], s32[]) tuple(get-tuple-element.69, get-tuple-element.71, get-tuple-element.70)
  tuple.36 = (s32[]) tuple(constant.8)
  tuple.37 = (s32[], (s32[], s32[], s32[]), (s32[])) tuple(select, tuple.35, tuple.36)
  while = (s32[], (s32[], s32[], s32[]), (s32[])) while(tuple.37), condition=if-condition.v4, body=if-body.v5
  get-tuple-element.72 = (s32[]) get-tuple-element(while), index=2
  get-tuple-element.73 = s32[] get-tuple-element(get-tuple-element.72), index=0
  ROOT tuple.38 = (s32[], s32[], s32[], s32[]) tuple(add.4, get-tuple-element.69, get-tuple-element.70, get-tuple-element.73)
}

cond_wrapper.v3.1 {
  inputs.1 = (s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.75 = s32[] get-tuple-element(inputs.1), index=0
  constant.11 = s32[] constant(7)
  ROOT less-than.2 = pred[] compare(get-tuple-element.75, constant.11), direction=LT
}

_functionalize_body_2__.v25 {
  arg_tuple.5 = (s32[], s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.76 = s32[] get-tuple-element(arg_tuple.5), index=0
  get-tuple-element.77 = s32[] get-tuple-element(arg_tuple.5), index=2
  get-tuple-element.78 = s32[] get-tuple-element(arg_tuple.5), index=3
  get-tuple-element.79 = s32[] get-tuple-element(arg_tuple.5), index=4
  tuple.39 = (s32[], s32[], s32[], s32[]) tuple(get-tuple-element.76, get-tuple-element.77, get-tuple-element.78, get-tuple-element.79)
  while.2 = (s32[], s32[], s32[], s32[]) while(tuple.39), condition=cond_wrapper.v3.1, body=_functionalize_body_1__.v28
  get-tuple-element.80 = s32[] get-tuple-element(while.2), index=0
  get-tuple-element.81 = s32[] get-tuple-element(arg_tuple.5), index=1
  constant.12 = s32[] constant(1)
  add.5 = s32[] add(get-tuple-element.81, constant.12)
  get-tuple-element.82 = s32[] get-tuple-element(while.2), index=3
  ROOT tuple.40 = (s32[], s32[], s32[], s32[], s32[]) tuple(get-tuple-element.80, add.5, get-tuple-element.77, get-tuple-element.78, get-tuple-element.82)
}

cond_wrapper.v3.2 {
  inputs.2 = (s32[], s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.83 = s32[] get-tuple-element(inputs.2), index=1
  constant.13 = s32[] constant(5)
  ROOT less-than.3 = pred[] compare(get-tuple-element.83, constant.13), direction=LT
}

ENTRY TestComputation {
  arg_tuple.6 = (s32[], s32[], s32[], s32[], s32[]) parameter(0)
  ROOT while.3 = (s32[], s32[], s32[], s32[], s32[]) while(arg_tuple.6), condition=cond_wrapper.v3.2, body=_functionalize_body_2__.v25
}
)";
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string);
  auto module = module_or_status.ConsumeValueOrDie();
  InsertCopies(module.get());
}

TEST_F(CopyInsertionTest, ControlFlowTest) {
  const string& hlo_string = R"(
HloModule TestModule

if-body.v5 {
  constant.3 = s32[] constant(-1)
  p.1 = (s32[], (s32[], s32[], s32[]), (s32[])) parameter(0)
  get-tuple-element.18 = (s32[], s32[], s32[]) get-tuple-element(p.1), index=1
  get-tuple-element.65 = s32[] get-tuple-element(get-tuple-element.18), index=0
  get-tuple-element.66 = s32[] get-tuple-element(get-tuple-element.18), index=1
  add.3 = s32[] add(get-tuple-element.65, get-tuple-element.66)
  tuple.33 = (s32[]) tuple(add.3)
  ROOT tuple.34 = (s32[], (s32[], s32[], s32[]), (s32[])) tuple(constant.3, get-tuple-element.18, tuple.33)
}

if-condition.v4 {
  p.2 = (s32[], (s32[], s32[], s32[]), (s32[])) parameter(0)
  get-tuple-element.67 = s32[] get-tuple-element(p.2), index=0
  constant.4 = s32[] constant(0)
  ROOT equal-to = pred[] compare(get-tuple-element.67, constant.4), direction=EQ
}

if-body.v5.1 {
  constant.5 = s32[] constant(-1)
  p.3 = (s32[], (s32[], s32[], s32[]), (s32[])) parameter(0)
  get-tuple-element.68 = (s32[], s32[], s32[]) get-tuple-element(p.3), index=1
  get-tuple-element.70 = s32[] get-tuple-element(get-tuple-element.68), index=2
  multiply.1 = s32[] multiply(get-tuple-element.70, get-tuple-element.70)
  tuple.35 = (s32[]) tuple(multiply.1)
  ROOT tuple.36 = (s32[], (s32[], s32[], s32[]), (s32[])) tuple(constant.5, get-tuple-element.68, tuple.35)
}

if-condition.v4.1 {
  p.4 = (s32[], (s32[], s32[], s32[]), (s32[])) parameter(0)
  get-tuple-element.71 = s32[] get-tuple-element(p.4), index=0
  constant.6 = s32[] constant(1)
  ROOT equal-to.1 = pred[] compare(get-tuple-element.71, constant.6), direction=EQ
}

_functionalize_body_1__.v28 {
  arg_tuple.4 = (s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.72 = s32[] get-tuple-element(arg_tuple.4), index=0
  constant.7 = s32[] constant(1)
  add.4 = s32[] add(get-tuple-element.72, constant.7)
  get-tuple-element.73 = s32[] get-tuple-element(arg_tuple.4), index=1
  get-tuple-element.74 = s32[] get-tuple-element(arg_tuple.4), index=2
  less-than-or-equal-to = pred[] compare(get-tuple-element.73, get-tuple-element.74), direction=LE
  constant.8 = s32[] constant(0)
  select = s32[] select(less-than-or-equal-to, constant.8, constant.7)
  get-tuple-element.75 = s32[] get-tuple-element(arg_tuple.4), index=3
  tuple.37 = (s32[], s32[], s32[]) tuple(get-tuple-element.73, get-tuple-element.75, get-tuple-element.74)
  tuple.38 = (s32[]) tuple(constant.8)
  tuple.39 = (s32[], (s32[], s32[], s32[]), (s32[])) tuple(select, tuple.37, tuple.38)
  while = (s32[], (s32[], s32[], s32[]), (s32[])) while(tuple.39), condition=if-condition.v4, body=if-body.v5
  while.1 = (s32[], (s32[], s32[], s32[]), (s32[])) while(while), condition=if-condition.v4.1, body=if-body.v5.1
  get-tuple-element.76 = (s32[]) get-tuple-element(while.1), index=2
  get-tuple-element.77 = s32[] get-tuple-element(get-tuple-element.76), index=0
  ROOT tuple.40 = (s32[], s32[], s32[], s32[]) tuple(add.4, get-tuple-element.73, get-tuple-element.74, get-tuple-element.77)
}

cond_wrapper.v3.1 {
  inputs.1 = (s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.78 = s32[] get-tuple-element(inputs.1), index=0
  constant.11 = s32[] constant(7)
  ROOT less-than.2 = pred[] compare(get-tuple-element.78, constant.11), direction=LT
}

_functionalize_body_2__.v25 {
  arg_tuple.5 = (s32[], s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.79 = s32[] get-tuple-element(arg_tuple.5), index=0
  get-tuple-element.80 = s32[] get-tuple-element(arg_tuple.5), index=2
  get-tuple-element.81 = s32[] get-tuple-element(arg_tuple.5), index=3
  get-tuple-element.82 = s32[] get-tuple-element(arg_tuple.5), index=4
  tuple.41 = (s32[], s32[], s32[], s32[]) tuple(get-tuple-element.79, get-tuple-element.80, get-tuple-element.81, get-tuple-element.82)
  while.2 = (s32[], s32[], s32[], s32[]) while(tuple.41), condition=cond_wrapper.v3.1, body=_functionalize_body_1__.v28
  get-tuple-element.83 = s32[] get-tuple-element(while.2), index=0
  get-tuple-element.84 = s32[] get-tuple-element(arg_tuple.5), index=1
  constant.12 = s32[] constant(1)
  add.5 = s32[] add(get-tuple-element.84, constant.12)
  get-tuple-element.85 = s32[] get-tuple-element(while.2), index=3
  ROOT tuple.42 = (s32[], s32[], s32[], s32[], s32[]) tuple(get-tuple-element.83, add.5, get-tuple-element.80, get-tuple-element.81, get-tuple-element.85)
}

cond_wrapper.v3.2 {
  inputs.2 = (s32[], s32[], s32[], s32[], s32[]) parameter(0)
  get-tuple-element.86 = s32[] get-tuple-element(inputs.2), index=1
  constant.13 = s32[] constant(5)
  ROOT less-than.3 = pred[] compare(get-tuple-element.86, constant.13), direction=LT
}

ENTRY TestComputation {
  arg_tuple.6 = (s32[], s32[], s32[], s32[], s32[]) parameter(0)
  ROOT while.3 = (s32[], s32[], s32[], s32[], s32[]) while(arg_tuple.6), condition=cond_wrapper.v3.2, body=_functionalize_body_2__.v25
}
)";
  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string);
  auto module = module_or_status.ConsumeValueOrDie();
  InsertCopies(module.get());
}

TEST_F(CopyInsertionTest, NestedWhiles) {
  // Verify that only no unnecessary copies remain after copy insertion for
  // trivial nested whiles (b/112472605).
  const string& hlo_string = R"(
HloModule TestModule

cond.inner {
  ROOT param.cond.inner = pred[] parameter(0)
}

body.inner {
  param.body.inner = pred[] parameter(0)
  ROOT not = pred[] not(param.body.inner)
}

cond.outer {
  ROOT param.cond.outer = pred[] parameter(0)
}

body.outer {
  param.cond.outer = pred[] parameter(0)
  ROOT while = pred[] while(param.cond.outer), condition=cond.inner, body=body.inner
}

ENTRY TestComputation {
  entry_param = pred[] parameter(0)
  ROOT while = pred[] while(entry_param), condition=cond.outer, body=body.outer
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InsertCopies(module.get());

  // There should only be a single copy inserted, and it's in the entry
  // computation.
  EXPECT_EQ(CountCopies(*module), 1);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::While(op::Copy(op::Parameter())));
}

}  // namespace
}  // namespace xla

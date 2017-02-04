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

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include "tensorflow/compiler/xla/test_helpers.h"

namespace xla {
namespace {

class CopyInsertionTest : public HloTestBase {
 protected:
  void InsertCopies(HloModule* module) {
    CopyInsertion copy_insertion;
    EXPECT_IS_OK(copy_insertion.Run(module).status());

    // Verify the points to set of the root of the computation after copy
    // insertion contains no constants or parameters.
    auto points_to_analysis =
        TuplePointsToAnalysis::Run(module).ConsumeValueOrDie();
    const std::set<const LogicalBuffer*> maybe_live_out_buffers =
        points_to_analysis
            ->GetPointsToSet(module->entry_computation()->root_instruction())
            .CreateFlattenedSet();
    for (const LogicalBuffer* buffer : maybe_live_out_buffers) {
      EXPECT_NE(buffer->instruction()->opcode(), HloOpcode::kConstant);
      EXPECT_NE(buffer->instruction()->opcode(), HloOpcode::kParameter);
    }
  }

  // OperandTree is a test helper class that simplifies the expression of
  // an expected tree of operands (starting at some root instruction) in a
  // unit test.
  // Each HLO instruction is represented as a node in the OperandTree.
  struct OperandTree {
    // The expected opcode for this OperandTree node.
    HloOpcode opcode;
    // The set of operands expected for this OperandTree node.
    std::vector<OperandTree> operands;
    // If non-null, a pointer to the expected HloInstruction at this node.
    const HloInstruction* instruction = nullptr;

    // Returns a mutable reference to operand 'i' of this node.
    OperandTree& op(int i) {
      if (i >= operands.size()) {
        operands.resize(i + 1);
      }
      return operands[i];
    }

    // Check that 'instruction' and its operands match expected values recorded
    // in OperandTree.
    void Check(const HloInstruction* instruction) {
      EXPECT_EQ(opcode, instruction->opcode());
      if (instruction != nullptr) {
        EXPECT_EQ(instruction, instruction);
      }
      if (operands.empty()) {
        return;
      }
      EXPECT_EQ(operands.size(), instruction->operand_count());
      for (int i = 0; i < instruction->operand_count(); ++i) {
        operands[i].Check(instruction->operand(i));
      }
    }
  };
};

#define EXPECT_INST(A, E...) EXPECT_EQ(A, (std::set<HloInstruction*>{E}))

TEST_F(CopyInsertionTest, SingleParameter) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "x"));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({x}));

  EXPECT_INST(x->users(), tuple);

  HloModule module(TestName());
  module.AddEntryComputation(builder.Build());

  HloInstruction* old_root = module.entry_computation()->root_instruction();
  InsertCopies(&module);
  HloInstruction* new_root = module.entry_computation()->root_instruction();

  // Check path from 'new_root' to 'old_root'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kTuple;

  op_tree.op(0).opcode = HloOpcode::kCopy;
  op_tree.op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(0).op(0).op(0).instruction = old_root;

  op_tree.Check(new_root);
}

TEST_F(CopyInsertionTest, SingleConstant) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({constant}));

  EXPECT_INST(constant->users(), tuple);

  HloModule module(TestName());
  module.AddEntryComputation(builder.Build());

  HloInstruction* old_root = module.entry_computation()->root_instruction();
  InsertCopies(&module);
  HloInstruction* new_root = module.entry_computation()->root_instruction();

  // Check path from 'new_root' to 'old_root'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kTuple;

  op_tree.op(0).opcode = HloOpcode::kCopy;
  op_tree.op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(0).op(0).op(0).instruction = old_root;

  op_tree.Check(new_root);
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

  HloModule module(TestName());
  module.AddEntryComputation(builder.Build());

  HloInstruction* old_root = module.entry_computation()->root_instruction();
  InsertCopies(&module);
  HloInstruction* new_root = module.entry_computation()->root_instruction();

  // "constant2" and parameter "x" are pointed to by the tuple and should be
  // copied.

  // Check all paths from 'new_root' to 'old_root'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kTuple;

  op_tree.op(0).opcode = HloOpcode::kCopy;
  op_tree.op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(0).op(0).op(0).instruction = old_root;

  op_tree.op(1).opcode = HloOpcode::kCopy;
  op_tree.op(1).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(1).op(0).op(0).instruction = old_root;

  op_tree.op(2).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(2).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(2).op(0).instruction = old_root;

  op_tree.Check(new_root);
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
      tuple1->shape(), HloOpcode::kSelect, pred, tuple1, tuple2));

  EXPECT_INST(constant1->users(), tuple1);
  EXPECT_INST(constant2->users(), tuple1, tuple2);
  EXPECT_INST(constant3->users(), tuple2);

  HloModule module(TestName());
  module.AddEntryComputation(builder.Build());

  HloInstruction* old_root = module.entry_computation()->root_instruction();
  InsertCopies(&module);
  HloInstruction* new_root = module.entry_computation()->root_instruction();

  // Check all paths from 'new_root' to 'old_root'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kTuple;

  op_tree.op(0).opcode = HloOpcode::kCopy;
  op_tree.op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).op(0).opcode = HloOpcode::kSelect;
  op_tree.op(0).op(0).op(0).instruction = old_root;

  op_tree.op(1).opcode = HloOpcode::kCopy;
  op_tree.op(1).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(0).op(0).opcode = HloOpcode::kSelect;
  op_tree.op(1).op(0).op(0).instruction = old_root;

  op_tree.Check(new_root);
}

TEST_F(CopyInsertionTest, BitcastParameter) {
  // The output of a bitcast is its operand (same buffer), so a bitcast
  // parameter feeding the result must have a copy added.
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {4}), "x"));
  HloInstruction* bitcast = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {2, 2}), HloOpcode::kBitcast, x));

  HloModule module(TestName());
  module.AddEntryComputation(builder.Build());

  EXPECT_INST(x->users(), bitcast);

  HloInstruction* old_root = module.entry_computation()->root_instruction();
  InsertCopies(&module);
  HloInstruction* new_root = module.entry_computation()->root_instruction();

  // Check path from 'new_root' to 'old_root'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kCopy;
  op_tree.op(0).opcode = HloOpcode::kBitcast;
  op_tree.op(0).instruction = old_root;

  op_tree.Check(new_root);
}

TEST_F(CopyInsertionTest, BitcastConstant) {
  // The output of a bitcast is its operand (same buffer), so a bitcast
  // constant feeding the result must have a copy added.
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* constant =
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<float>({1.0, 42.0})));
  HloInstruction* bitcast = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {2, 2}), HloOpcode::kBitcast, constant));

  HloModule module(TestName());
  module.AddEntryComputation(builder.Build());

  EXPECT_INST(constant->users(), bitcast);

  HloInstruction* old_root = module.entry_computation()->root_instruction();
  InsertCopies(&module);
  HloInstruction* new_root = module.entry_computation()->root_instruction();

  // Check path from 'new_root' to 'old_root'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kCopy;
  op_tree.op(0).opcode = HloOpcode::kBitcast;
  op_tree.op(0).instruction = old_root;

  op_tree.Check(new_root);
}

TEST_F(CopyInsertionTest, BitcastTupleElementParameter) {
  // Same as BitcastParameter, but the bitcast is wrapped in a tuple.
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {4}), "x"));
  HloInstruction* bitcast = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {2, 2}), HloOpcode::kBitcast, x));
  builder.AddInstruction(HloInstruction::CreateTuple({bitcast}));

  HloModule module(TestName());
  module.AddEntryComputation(builder.Build());

  EXPECT_EQ(1, x->user_count());
  EXPECT_EQ(*x->users().begin(), bitcast);

  HloInstruction* old_root = module.entry_computation()->root_instruction();
  InsertCopies(&module);
  HloInstruction* new_root = module.entry_computation()->root_instruction();

  // Check path from 'new_root' to 'old_root'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kTuple;
  op_tree.op(0).opcode = HloOpcode::kCopy;
  op_tree.op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(0).op(0).op(0).instruction = old_root;

  op_tree.Check(new_root);
}

TEST_F(CopyInsertionTest, NestedTupleParameter) {
  // Construct a trivial computation where the root of the computation is a
  // nested tuple-shaped parameter. The parameter should be deep copied and the
  // copy should be the root of the computation.
  auto builder = HloComputation::Builder(TestName());

  // Param shape is: ((F32[], S32[1,2,3]), F32[42])
  builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape(
             {ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {}),
                                         ShapeUtil::MakeShape(S32, {1, 2, 3})}),
              ShapeUtil::MakeShape(F32, {42})}),
      "param0"));

  HloModule module(TestName());
  module.AddEntryComputation(builder.Build());

  EXPECT_EQ(HloOpcode::kParameter,
            module.entry_computation()->root_instruction()->opcode());

  HloInstruction* old_root = module.entry_computation()->root_instruction();
  InsertCopies(&module);
  HloInstruction* new_root = module.entry_computation()->root_instruction();
  EXPECT_NE(old_root, new_root);

  // Check all paths from 'new_root' to 'old_root'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kTuple;

  op_tree.op(0).opcode = HloOpcode::kTuple;
  op_tree.op(0).op(0).opcode = HloOpcode::kCopy;
  op_tree.op(0).op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).op(0).op(0).op(0).opcode = HloOpcode::kParameter;
  op_tree.op(0).op(0).op(0).op(0).op(0).instruction = old_root;

  op_tree.op(0).opcode = HloOpcode::kTuple;
  op_tree.op(0).op(1).opcode = HloOpcode::kCopy;
  op_tree.op(0).op(1).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(1).op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(1).op(0).op(0).op(0).opcode = HloOpcode::kParameter;
  op_tree.op(0).op(1).op(0).op(0).op(0).instruction = old_root;

  op_tree.op(1).opcode = HloOpcode::kCopy;
  op_tree.op(1).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(0).op(0).opcode = HloOpcode::kParameter;
  op_tree.op(1).op(0).op(0).instruction = old_root;

  op_tree.Check(new_root);
}

TEST_F(CopyInsertionTest, ElementOfNestedTupleParameter) {
  // Construct a computation where the root of the computation is a tuple
  // element of a nested tuple-shaped parameter.
  auto builder = HloComputation::Builder(TestName());

  // Param shape is: ((F32[], S32[1,2,3]), F32[42])
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape(
             {ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {}),
                                         ShapeUtil::MakeShape(S32, {1, 2, 3})}),
              ShapeUtil::MakeShape(F32, {42})}),
      "param0"));

  // The return value of the computation is the zero-th elemnt of the nested
  // tuple. This element is itself a tuple.
  auto gte = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeUtil::GetSubshape(param->shape(), {0}), param, 0));

  HloModule module(TestName());
  module.AddEntryComputation(builder.Build());

  EXPECT_EQ(gte, module.entry_computation()->root_instruction());

  HloInstruction* old_root = module.entry_computation()->root_instruction();
  InsertCopies(&module);
  HloInstruction* new_root = module.entry_computation()->root_instruction();

  // Check all paths from 'new_root' to 'old_root'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kTuple;

  op_tree.op(0).opcode = HloOpcode::kCopy;
  op_tree.op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).op(0).instruction = old_root;

  op_tree.op(1).opcode = HloOpcode::kCopy;
  op_tree.op(1).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(0).op(0).instruction = old_root;

  op_tree.Check(new_root);
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
      tuple1->shape(), HloOpcode::kSelect, pred, tuple1, tuple2));
  HloInstruction* gte =
      builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          ShapeUtil::GetSubshape(select->shape(), {0}), select, 0));

  HloModule module(TestName());
  module.AddEntryComputation(builder.Build());

  EXPECT_EQ(gte, module.entry_computation()->root_instruction());

  HloInstruction* old_root = module.entry_computation()->root_instruction();
  InsertCopies(&module);
  HloInstruction* new_root = module.entry_computation()->root_instruction();

  // Check path from 'new_root' to 'old_root'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kCopy;
  op_tree.op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).instruction = old_root;

  op_tree.Check(new_root);
}

class WhileCopyInsertionTest : public CopyInsertionTest {
 protected:
  WhileCopyInsertionTest() : module_(TestName()) {}

  // Builds a While condition computation which reads the induction variable
  // from the tuple parameter, and returns a predicate indicating whether this
  // value is less than the constant '10'.
  // The parameter 'nested' specifies the loop state shape from which to
  // read the induction variable.
  std::unique_ptr<HloComputation> BuildConditionComputation(
      bool nested = false) {
    auto builder = HloComputation::Builder(TestName() + ".Condition");
    auto limit_const = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(10)));
    const Shape& loop_state_shape =
        nested ? nested_loop_state_shape_ : loop_state_shape_;
    auto loop_state = builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_state_shape, "loop_state"));
    auto induction_variable =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            limit_const->shape(), loop_state, 0));
    builder.AddInstruction(
        HloInstruction::CreateBinary(condition_result_shape_, HloOpcode::kLt,
                                     induction_variable, limit_const));
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
    auto update = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, induction_variable, {8}));
    auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, data, update));
    // Create output Tuple.
    builder.AddInstruction(HloInstruction::CreateTuple({add0, add1}));
    return builder.Build();
  }

  // Builds a While body computation with read-only tuple element 0.
  // both input tuple elements.
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
    auto update = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, induction_variable, {8}));
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
  std::unique_ptr<HloComputation> BuildIndependentBodyComputation() {
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
    // add0 = Add(in0, 1)
    auto add0 = builder.AddInstruction(HloInstruction::CreateBinary(
        induction_variable->shape(), HloOpcode::kAdd, induction_variable, inc));
    // Update data GTE(1).
    auto data = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(data_shape_, loop_state, 1));
    auto update = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(
            {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f})));
    // add0 = Add(in1, {1, 1, 1, 1, 1, 1, 1, 1})
    auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, data, update));
    // Create output Tuple.
    builder.AddInstruction(HloInstruction::CreateTuple({add0, add1}));
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
  void BuildWhileInstruction(HloComputation* condition, HloComputation* body,
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
      builder.AddInstruction(HloInstruction::CreateWhile(
          loop_state_shape_, condition, body, loop_state_init));
      module_.AddEntryComputation(builder.Build());
      return;
    }

    auto loop_state_init = builder.AddInstruction(
        HloInstruction::CreateTuple({induction_var_init, data_init}));
    builder.AddInstruction(HloInstruction::CreateWhile(
        loop_state_shape_, condition, body, loop_state_init));
    module_.AddEntryComputation(builder.Build());
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
        HloInstruction::CreateBroadcast(data_shape_, one, {1}));
    auto zero = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
    auto v2 = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, zero, {1}));

    auto tuple1 = builder.AddInstruction(HloInstruction::CreateTuple({v1, v2}));
    auto tuple2 = builder.AddInstruction(HloInstruction::CreateTuple({v2, v1}));

    auto pred = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
    auto data_init = builder.AddInstruction(HloInstruction::CreateTernary(
        nested_tuple_shape_, HloOpcode::kSelect, pred, tuple1, tuple2));

    return BuildWhileInstructionWithCustomInit(nested_loop_state_shape_,
                                               data_init, &builder);
  }

  HloInstruction* BuildWhileInstruction_InitPointsToNonDistinct() {
    auto builder = HloComputation::Builder(TestName() + ".While");

    auto one = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
    auto one_vec = builder.AddInstruction(
        HloInstruction::CreateBroadcast(data_shape_, one, {1}));
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
        HloInstruction::CreateBroadcast(data_shape_, one, {1}));
    auto one_vec = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(
            {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f})));
    // Take a reference to 'data_init' to make it interfere with while result.
    builder.AddInstruction(HloInstruction::CreateBinary(
        data_shape_, HloOpcode::kAdd, data_init, one_vec));

    return BuildWhileInstructionWithCustomInit(loop_state_shape_, data_init,
                                               &builder);
  }

  HloInstruction* BuildWhileInstructionWithCustomInit(
      const Shape& loop_state_shape, HloInstruction* data_init,
      HloComputation::Builder* builder) {
    auto induction_var_init = builder->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
    auto condition =
        module_.AddEmbeddedComputation(BuildConditionComputation());
    auto body =
        module_.AddEmbeddedComputation(BuildIndependentBodyComputation());
    auto loop_state_init = builder->AddInstruction(
        HloInstruction::CreateTuple({induction_var_init, data_init}));
    auto while_hlo = builder->AddInstruction(HloInstruction::CreateWhile(
        loop_state_shape, condition, body, loop_state_init));
    module_.AddEntryComputation(builder->Build());
    return while_hlo;
  }

  HloModule module_;
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
  auto condition = module_.AddEmbeddedComputation(BuildConditionComputation());
  auto body = module_.AddEmbeddedComputation(BuildIndependentBodyComputation());
  BuildWhileInstruction(condition, body);

  HloInstruction* old_root = body->root_instruction();
  InsertCopies(&module_);
  HloInstruction* new_root = body->root_instruction();

  // No copies should be inserted so root should not be updated.
  CHECK_EQ(old_root, new_root);
}

// Tests while body computation with dependent tuple elements:
//
//   While.Body({in0, in1})
//     out0 = Add(in0, 1)
//     out1 = Add(BCast(in0), in1)
//     Tuple(out0, out1)
//
// CopyInsertion pass should generate:
//
//                    Tuple  // old root
//                   /    \
//                GTE(0)  GTE(1)
//                  |       |
//                 Copy     |
//                   \     /
//                    Tuple  // new root
//
TEST_F(WhileCopyInsertionTest, DependentTupleElements) {
  auto condition = module_.AddEmbeddedComputation(BuildConditionComputation());
  auto body = module_.AddEmbeddedComputation(BuildDependentBodyComputation());
  BuildWhileInstruction(condition, body);

  HloInstruction* old_root = body->root_instruction();
  InsertCopies(&module_);
  HloInstruction* new_root = body->root_instruction();

  // Check all paths from 'new_root' to 'old_root'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kTuple;

  op_tree.op(0).opcode = HloOpcode::kCopy;
  op_tree.op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(0).op(0).op(0).instruction = old_root;

  op_tree.op(1).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(1).op(0).instruction = old_root;

  op_tree.Check(new_root);
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
// CopyInsertion pass should not generate any copies.
//
TEST_F(WhileCopyInsertionTest, DependentTupleElements_OneReadOnly) {
  auto condition = module_.AddEmbeddedComputation(BuildConditionComputation());
  auto body = module_.AddEmbeddedComputation(
      BuildDependentBodyOneReadOnlyComputation());
  BuildWhileInstruction(condition, body);

  HloInstruction* old_root = body->root_instruction();
  InsertCopies(&module_);
  HloInstruction* new_root = body->root_instruction();

  // No copies should be inserted so root should not be updated.
  CHECK_EQ(old_root, new_root);
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
// CopyInsertion pass should generate:
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
  auto condition =
      module_.AddEmbeddedComputation(BuildConditionComputation(true));
  auto body = module_.AddEmbeddedComputation(BuildNestedBodyComputation());
  BuildWhileInstruction(condition, body, true);

  HloInstruction* old_root = body->root_instruction();
  InsertCopies(&module_);
  HloInstruction* new_root = body->root_instruction();

  // Check all paths from 'new_root' to 'old_root'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kTuple;

  op_tree.op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(0).op(0).instruction = old_root;

  op_tree.op(1).opcode = HloOpcode::kTuple;

  op_tree.op(1).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(1).op(0).op(0).op(0).instruction = old_root;

  op_tree.op(1).op(1).opcode = HloOpcode::kCopy;
  op_tree.op(1).op(1).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(1).op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(1).op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(1).op(1).op(0).op(0).op(0).instruction = old_root;

  op_tree.Check(new_root);
}

// Tests while init instruction which points-to a constant.
//
//     init = Tuple(Constant(S32, {}), Constant(F32, {8}))
//
// CopyInsertion pass should generate:
//
//                    Tuple  // old init
//                   /    \
//                GTE(0)  GTE(1)
//                  |       |
//                 Copy    Copy
//                   \     /
//                    Tuple  // new init
//
TEST_F(WhileCopyInsertionTest, InitPointsToConstant) {
  auto while_hlo = BuildWhileInstruction_InitPointsToConstant();
  auto old_init = while_hlo->operand(0);
  InsertCopies(&module_);
  auto new_init = while_hlo->operand(0);

  // Check all paths from 'new_init' to 'old_init'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kTuple;

  op_tree.op(0).opcode = HloOpcode::kCopy;
  op_tree.op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(0).op(0).op(0).instruction = old_init;

  op_tree.op(1).opcode = HloOpcode::kCopy;
  op_tree.op(1).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(1).op(0).op(0).instruction = old_init;

  op_tree.Check(new_init);
}

// Tests while init instruction which points-to a parameter.
//
//     init = Tuple(Constant(S32, {}), Parameter(F32, {8}))
//
// CopyInsertion pass should generate:
//
//                    Tuple  // old init
//                   /    \
//                GTE(0)  GTE(1)
//                  |       |
//                 Copy    Copy
//                   \     /
//                    Tuple  // new init
//
TEST_F(WhileCopyInsertionTest, InitPointsToParameter) {
  auto while_hlo = BuildWhileInstruction_InitPointsToParameter();
  auto old_init = while_hlo->operand(0);
  InsertCopies(&module_);
  auto new_init = while_hlo->operand(0);

  // Check all paths from 'new_init' to 'old_init'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kTuple;

  op_tree.op(0).opcode = HloOpcode::kCopy;
  op_tree.op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(0).op(0).op(0).instruction = old_init;

  op_tree.op(1).opcode = HloOpcode::kCopy;
  op_tree.op(1).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(1).op(0).op(0).instruction = old_init;

  op_tree.Check(new_init);
}

// Tests while init instruction which has an ambiguous points-to set.
//
//     select = Select(pred, tuple1, tuple2)
//     init = Tuple(Constant(S32, {}), Parameter(F32, {8}))
//
// CopyInsertion pass should generate:
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
  auto old_init = while_hlo->operand(0);
  InsertCopies(&module_);
  auto new_init = while_hlo->operand(0);

  // Check all paths from 'new_init' to 'old_init'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kTuple;

  op_tree.op(0).opcode = HloOpcode::kCopy;
  op_tree.op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(0).op(0).op(0).instruction = old_init;

  op_tree.op(1).opcode = HloOpcode::kTuple;

  op_tree.op(1).op(0).opcode = HloOpcode::kCopy;
  op_tree.op(1).op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(0).op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(0).op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(1).op(0).op(0).op(0).op(0).instruction = old_init;

  op_tree.op(1).op(1).opcode = HloOpcode::kCopy;
  op_tree.op(1).op(1).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(1).op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(1).op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(1).op(1).op(0).op(0).op(0).instruction = old_init;

  op_tree.Check(new_init);
}

// Tests while init instruction which has a non-distinct points-to set.
//
//     init = Tuple(Constant(S32, {}), Tuple({vec_one, vec_one}))
//
// CopyInsertion pass should generate:
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
  auto old_init = while_hlo->operand(0);
  InsertCopies(&module_);
  auto new_init = while_hlo->operand(0);

  // Check all paths from 'new_init' to 'old_init'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kTuple;

  op_tree.op(0).opcode = HloOpcode::kCopy;
  op_tree.op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(0).op(0).op(0).instruction = old_init;

  op_tree.op(1).opcode = HloOpcode::kTuple;

  op_tree.op(1).op(0).opcode = HloOpcode::kCopy;
  op_tree.op(1).op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(0).op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(0).op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(1).op(0).op(0).op(0).op(0).instruction = old_init;

  op_tree.op(1).op(1).opcode = HloOpcode::kCopy;
  op_tree.op(1).op(1).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(1).op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(1).op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(1).op(1).op(0).op(0).op(0).instruction = old_init;

  op_tree.Check(new_init);
}

// Tests while init instruction buffer which interfers with while result buffer.
//
//     init_data = Broadcast(...)
//     add_unrelated = Add(init_data) // takes a reference to cause interference
//     init = Tuple(Constant(S32, {}), init_data))
//
// CopyInsertion pass should generate:
//
//                    Tuple  // old init
//                   /    \
//                GTE(0)  GTE(1)
//                  |       |
//                 Copy    Copy
//                   \     /
//                    Tuple  // new init
//
TEST_F(WhileCopyInsertionTest, InitPointsToInterfering) {
  auto while_hlo = BuildWhileInstruction_InitPointsToInterfering();
  auto old_init = while_hlo->operand(0);
  InsertCopies(&module_);
  auto new_init = while_hlo->operand(0);

  // Check all paths from 'new_init' to 'old_init'.
  OperandTree op_tree;
  op_tree.opcode = HloOpcode::kTuple;

  op_tree.op(0).opcode = HloOpcode::kCopy;
  op_tree.op(0).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(0).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(0).op(0).op(0).instruction = old_init;

  op_tree.op(1).opcode = HloOpcode::kCopy;
  op_tree.op(1).op(0).opcode = HloOpcode::kGetTupleElement;
  op_tree.op(1).op(0).op(0).opcode = HloOpcode::kTuple;
  op_tree.op(1).op(0).op(0).instruction = old_init;

  op_tree.Check(new_init);
}

}  // namespace
}  // namespace xla

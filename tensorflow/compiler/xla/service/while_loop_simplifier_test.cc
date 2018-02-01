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

#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class WhileLoopSimplifierTest : public HloVerifiedTestBase {
 public:
  // Makes a computation that contains a loop that runs num_iters times.
  HloComputation* MakeSimpleLoop(int num_iters, HloModule* module);

  // Makes a computation which has one parameter, of the given shape, and always
  // returns PRED[]{true}.  This is useful as a dummy loop condition.
  HloComputation* MakeAlwaysTrueComputation(const Shape& param_shape,
                                            HloModule* module);
};

HloComputation* WhileLoopSimplifierTest::MakeSimpleLoop(int num_iters,
                                                        HloModule* module) {
  HloComputation::Builder builder(TestName());

  auto loop_iter_init = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<int32>(42)));
  auto loop_data_init = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<int32>({0, 1, 2})));
  auto loop_init = builder.AddInstruction(
      HloInstruction::CreateTuple({loop_iter_init, loop_data_init}));

  HloComputation* condition;
  {
    HloComputation::Builder cond_builder(TestName() + ".condition");
    auto loop_var = cond_builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_init->shape(), "loop_var"));
    auto loop_induction_var =
        cond_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            ShapeUtil::MakeShape(S32, {}), loop_var, 0));
    auto limit = cond_builder.AddInstruction(HloInstruction::CreateConstant(
        Literal::CreateR0<int32>(42 + num_iters)));
    cond_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, loop_induction_var,
        limit));
    condition = module->AddEmbeddedComputation(cond_builder.Build());
  }

  HloComputation* body;
  {
    HloComputation::Builder body_builder(TestName() + ".body");
    auto loop_var = body_builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_init->shape(), "loop_var"));
    auto loop_induction_var =
        body_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            ShapeUtil::MakeShape(S32, {}), loop_var, 0));
    auto new_loop_induction_var =
        body_builder.AddInstruction(HloInstruction::CreateBinary(
            loop_induction_var->shape(), HloOpcode::kAdd, loop_induction_var,
            body_builder.AddInstruction(
                HloInstruction::CreateConstant(Literal::CreateR0<int32>(1)))));
    auto loop_data =
        body_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            loop_data_init->shape(), loop_var, 1));
    auto new_loop_data =
        body_builder.AddInstruction(HloInstruction::CreateBinary(
            loop_data_init->shape(), HloOpcode::kMultiply, loop_data,
            loop_data));
    body_builder.AddInstruction(
        HloInstruction::CreateTuple({new_loop_induction_var, new_loop_data}));
    body = module->AddEmbeddedComputation(body_builder.Build());
  }

  builder.AddInstruction(HloInstruction::CreateWhile(
      loop_init->shape(), condition, body, loop_init));

  return module->AddEntryComputation(builder.Build());
}

HloComputation* WhileLoopSimplifierTest::MakeAlwaysTrueComputation(
    const Shape& param_shape, HloModule* module) {
  HloComputation::Builder builder(TestName() + ".always_true");
  builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(true)));
  return module->AddEmbeddedComputation(builder.Build());
}

TEST_F(WhileLoopSimplifierTest, WhileLoopWithZeroIterations) {
  HloComputation* computation = MakeSimpleLoop(/*num_iters=*/0, &module());
  ASSERT_TRUE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              op::Tuple(op::Constant(), op::Constant()));
}

TEST_F(WhileLoopSimplifierTest, WhileLoopWithOneIteration) {
  HloComputation* computation = MakeSimpleLoop(/*num_iters=*/1, &module());
  ASSERT_TRUE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              op::Tuple(op::Add(), op::Multiply()));
}

TEST_F(WhileLoopSimplifierTest, WhileLoopWithTwoIterations) {
  MakeSimpleLoop(/*num_iters=*/2, &module());
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

TEST_F(WhileLoopSimplifierTest, WhileLoopWithControlDependency) {
  HloComputation* computation = MakeSimpleLoop(/*num_iters=*/1, &module());
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* true_op = while_op->while_body()->AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(true)));
  TF_ASSERT_OK(true_op->AddControlDependencyTo(
      while_op->while_body()->root_instruction()));
  ASSERT_TRUE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction()->control_predecessors(),
              ElementsAre(op::Constant()))
      << computation->ToString();
}

// Loops that contain send/recv nodes can't be simplified; the loop structure
// around send/recv nodes must be preserved.
TEST_F(WhileLoopSimplifierTest, NotRemovedIfContainsSend) {
  HloComputation* computation = MakeSimpleLoop(/*num_iters=*/1, &module());
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* while_body = while_op->while_body();
  auto* send = while_body->AddInstruction(HloInstruction::CreateSend(
      while_body->AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<bool>(true))),
      /*channel_id=*/0));
  while_body->AddInstruction(HloInstruction::CreateSendDone(send));
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

TEST_F(WhileLoopSimplifierTest, NotRemovedIfContainsRecv) {
  HloComputation* computation = MakeSimpleLoop(/*num_iters=*/1, &module());
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* while_body = while_op->while_body();
  auto* recv = while_body->AddInstruction(
      HloInstruction::CreateRecv(ShapeUtil::MakeShape(F32, {1}),
                                 /*channel_id=*/0));
  while_body->AddInstruction(HloInstruction::CreateRecvDone(recv));
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

// The limitation on not being able to simplify loops that contain infeeds (and
// other non-removable instructions) isn't fundamental -- it just stems from the
// fact that our infrastructure sees simplifying such a loop as tantamount to
// removing the non-removable instruction.
TEST_F(WhileLoopSimplifierTest, NotRemovedIfContainsNonRemovableInstruction) {
  HloComputation* computation = MakeSimpleLoop(/*num_iters=*/1, &module());
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* while_body = while_op->while_body();
  while_body->AddInstruction(
      HloInstruction::CreateInfeed(ShapeUtil::MakeShape(F32, {1}), "config"));
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

// Check that we don't crash when given a loop whose shape is not a tuple.
TEST_F(WhileLoopSimplifierTest, IgnoreNonTupleShapedLoop) {
  HloComputation::Builder builder(TestName());
  auto loop_init = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<int32>(42)));

  HloComputation* condition;
  {
    HloComputation::Builder cond_builder(TestName() + ".condition");
    auto param = cond_builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_init->shape(), "loop_var"));
    cond_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, param,
        cond_builder.AddInstruction(
            HloInstruction::CreateConstant(Literal::CreateR0<int32>(100)))));
    condition = module().AddEmbeddedComputation(cond_builder.Build());
  }

  HloComputation* body;
  {
    HloComputation::Builder body_builder(TestName() + ".body");
    auto param = body_builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_init->shape(), "loop_var"));
    body_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(S32, {}), HloOpcode::kAdd, param,
        body_builder.AddInstruction(
            HloInstruction::CreateConstant(Literal::CreateR0<int32>(-1)))));
    body = module().AddEmbeddedComputation(body_builder.Build());
  }

  builder.AddInstruction(HloInstruction::CreateWhile(
      loop_init->shape(), condition, body, loop_init));

  module().AddEntryComputation(builder.Build());
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

// Construct a loop where we swap the tuple elements in each iteration.
// Although the tuple elements aren't used in the loop, we don't eliminate them,
// because the swapping side-effect is visible to users of the loop.
TEST_F(WhileLoopSimplifierTest, SwapTupleIndices) {
  HloComputation::Builder builder(TestName());
  auto loop_init = builder.AddInstruction(HloInstruction::CreateTuple({
      builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<int32>(0))),
      builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<int32>(1))),
  }));

  HloComputation* condition =
      MakeAlwaysTrueComputation(loop_init->shape(), &module());
  HloComputation* body;
  {
    HloComputation::Builder body_builder(TestName() + ".body");
    auto param = body_builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_init->shape(), "loop_var"));
    auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
    body_builder.AddInstruction(HloInstruction::CreateTuple({
        body_builder.AddInstruction(
            HloInstruction::CreateGetTupleElement(scalar_s32, param, 1)),
        body_builder.AddInstruction(
            HloInstruction::CreateGetTupleElement(scalar_s32, param, 0)),
    }));
    body = module().AddEmbeddedComputation(body_builder.Build());
  }

  builder.AddInstruction(HloInstruction::CreateWhile(
      loop_init->shape(), condition, body, loop_init));

  module().AddEntryComputation(builder.Build());
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

// Construct a loop where we assign a constant to tuple element 0 in each
// iteration.  We can't eliminate tuple element 0, even though we never use its
// value.
TEST_F(WhileLoopSimplifierTest, UnusedButModifiedTupleElement) {
  HloComputation::Builder builder(TestName());
  auto loop_init = builder.AddInstruction(
      HloInstruction::CreateTuple({builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<int32>(0)))}));

  HloComputation* condition =
      MakeAlwaysTrueComputation(loop_init->shape(), &module());
  HloComputation* body;
  {
    HloComputation::Builder body_builder(TestName() + ".body");
    body_builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_init->shape(), "loop_var"));
    body_builder.AddInstruction(HloInstruction::CreateTuple({
        body_builder.AddInstruction(
            HloInstruction::CreateConstant(Literal::CreateR0<int32>(1))),
    }));
    body = module().AddEmbeddedComputation(body_builder.Build());
  }

  builder.AddInstruction(HloInstruction::CreateWhile(
      loop_init->shape(), condition, body, loop_init));

  module().AddEntryComputation(builder.Build());
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

// Nothing to simplify in a while loop whose tuple has 0 elements.
TEST_F(WhileLoopSimplifierTest, EmptyTuple) {
  HloComputation::Builder builder(TestName());
  auto loop_init = builder.AddInstruction(HloInstruction::CreateTuple({}));

  HloComputation* condition =
      MakeAlwaysTrueComputation(loop_init->shape(), &module());
  HloComputation* body;
  {
    HloComputation::Builder body_builder(TestName() + ".body");
    body_builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_init->shape(), "loop_var"));
    body_builder.AddInstruction(HloInstruction::CreateTuple({}));
    body = module().AddEmbeddedComputation(body_builder.Build());
  }

  builder.AddInstruction(HloInstruction::CreateWhile(
      loop_init->shape(), condition, body, loop_init));
  module().AddEntryComputation(builder.Build());
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

// While loop where one tuple element is used twice in the body, and thus can't
// be simplified away.
TEST_F(WhileLoopSimplifierTest, ElemUsedTwice) {
  HloComputation::Builder builder(TestName());
  auto loop_init = builder.AddInstruction(HloInstruction::CreateTuple({
      builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<int32>(0))),
      builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<int32>(1))),
  }));

  HloComputation* condition =
      MakeAlwaysTrueComputation(loop_init->shape(), &module());

  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  HloComputation* body;
  {
    HloComputation::Builder body_builder(TestName() + ".body");
    auto* param = body_builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_init->shape(), "param0"));
    auto* gte0 = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, /*index=*/0));
    // get0 is used twice in the loop body's tuple.
    body_builder.AddInstruction(HloInstruction::CreateTuple({gte0, gte0}));
    body = module().AddEmbeddedComputation(body_builder.Build());
  }

  builder.AddInstruction(HloInstruction::CreateWhile(
      loop_init->shape(), condition, body, loop_init));
  module().AddEntryComputation(builder.Build());
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

// This while loop has three tuple elements.  Element 0 is unused and should be
// removed. Element 1 is used by the loop body, and element 2 is used by the
// loop condition; these two should stay.
TEST_F(WhileLoopSimplifierTest, RemoveUnusedOperand) {
  HloComputation::Builder builder(TestName());
  auto loop_init = builder.AddInstruction(HloInstruction::CreateTuple({
      builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<int32>(0))),
      builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<int32>(0))),
      builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<int32>(0))),
  }));
  auto loop_shape = loop_init->shape();
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});

  HloComputation* condition;
  {
    HloComputation::Builder cond_builder(TestName() + ".loop_condition");
    auto param = cond_builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_shape, "param0"));
    cond_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kEq,
        cond_builder.AddInstruction(
            HloInstruction::CreateConstant(Literal::CreateR0<int32>(0))),
        cond_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            scalar_s32, param, /*index=*/2))));
    condition = module().AddEmbeddedComputation(cond_builder.Build());
  }

  HloComputation* body;
  {
    HloComputation::Builder body_builder(TestName() + ".body");
    auto* param = body_builder.AddInstruction(
        HloInstruction::CreateParameter(0, loop_shape, "loop_var"));

    auto* tuple0 = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, /*index=*/0));
    auto* tuple1 = body_builder.AddInstruction(HloInstruction::CreateBinary(
        scalar_s32, HloOpcode::kAdd,
        body_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            scalar_s32, param, /*index=*/1)),
        body_builder.AddInstruction(
            HloInstruction::CreateConstant(Literal::CreateR0<int32>(1)))));
    auto* tuple2 = body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, /*index=*/2));
    body_builder.AddInstruction(
        HloInstruction::CreateTuple({tuple0, tuple1, tuple2}));

    body = module().AddEmbeddedComputation(body_builder.Build());
  }

  auto* while_op = builder.AddInstruction(HloInstruction::CreateWhile(
      loop_init->shape(), condition, body, loop_init));

  module().AddEntryComputation(builder.Build());
  EXPECT_TRUE(WhileLoopSimplifier().Run(&module()).ValueOrDie());

  // We leave most of the checking to HloVerifiedTestBase, which runs the
  // verifier on module() at the end of this test.
  HloInstruction* new_while_op = *std::find_if(
      module().entry_computation()->instructions().begin(),
      module().entry_computation()->instructions().end(),
      [&](const HloInstruction* instr) {
        return instr != while_op && instr->opcode() == HloOpcode::kWhile;
      });
  EXPECT_TRUE(
      ShapeUtil::Equal(new_while_op->shape(),
                       ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32})))
      << ShapeUtil::HumanString(new_while_op->shape());
  EXPECT_THAT(
      new_while_op->while_body()->root_instruction(),
      op::Tuple(
          op::Add(op::GetTupleElement(op::Parameter(0), /*tuple_index=*/0),
                  op::Constant()),
          op::GetTupleElement(op::Parameter(0), /*tuple_index=*/1)));

  EXPECT_THAT(new_while_op->while_condition()->root_instruction(),
              op::Eq(op::Constant(),
                     op::GetTupleElement(op::Parameter(0), /*tuple_index=*/1)));
}

TEST_F(WhileLoopSimplifierTest, BodyHasNonTupleRoot) {
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  Shape while_shape = ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".passthrough");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloComputation* result = module().AddEmbeddedComputation(builder.Build());

    result->AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 1));
    return result;
  }();

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, &module()),
      while_body, init_value));
  module().AddEntryComputation(builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopSimplifier{}.Run(&module()));
  EXPECT_FALSE(simplified_loop);
}

}  // namespace
}  // namespace xla

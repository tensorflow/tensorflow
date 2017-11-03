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
  HloComputation* MakeSimpleLoop(HloModule* module, int num_iters);
};

HloComputation* WhileLoopSimplifierTest::MakeSimpleLoop(HloModule* module,
                                                        int num_iters) {
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

TEST_F(WhileLoopSimplifierTest, WhileLoopWithZeroIterations) {
  HloModule module(TestName());
  HloComputation* computation = MakeSimpleLoop(&module, /*num_iters=*/0);
  ASSERT_TRUE(WhileLoopSimplifier().Run(&module).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              op::Tuple(op::Constant(), op::Constant()));
}

TEST_F(WhileLoopSimplifierTest, WhileLoopWithOneIteration) {
  HloModule module(TestName());
  HloComputation* computation = MakeSimpleLoop(&module, /*num_iters=*/1);
  ASSERT_TRUE(WhileLoopSimplifier().Run(&module).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              op::Tuple(op::Add(), op::Multiply()));
}

TEST_F(WhileLoopSimplifierTest, WhileLoopWithTwoIterations) {
  HloModule module(TestName());
  MakeSimpleLoop(&module, /*num_iters=*/2);
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module).ValueOrDie());
}

TEST_F(WhileLoopSimplifierTest, WhileLoopWithControlDependency) {
  HloModule module(TestName());
  HloComputation* computation = MakeSimpleLoop(&module, /*num_iters=*/1);
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* true_op = while_op->while_body()->AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(true)));
  TF_ASSERT_OK(true_op->AddControlDependencyTo(
      while_op->while_body()->root_instruction()));
  ASSERT_TRUE(WhileLoopSimplifier().Run(&module).ValueOrDie());
  EXPECT_THAT(computation->root_instruction()->control_predecessors(),
              ElementsAre(op::Constant()))
      << computation->ToString();
}

// Loops that contain send/recv nodes can't be simplified; the loop structure
// around send/recv nodes must be preserved.
TEST_F(WhileLoopSimplifierTest, NotRemovedIfContainsSend) {
  HloModule module(TestName());
  HloComputation* computation = MakeSimpleLoop(&module, /*num_iters=*/1);
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* while_body = while_op->while_body();
  while_body->AddInstruction(HloInstruction::CreateSend(
      while_body->AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<bool>(true))),
      /*channel_id=*/0));
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module).ValueOrDie());
}

TEST_F(WhileLoopSimplifierTest, NotRemovedIfContainsRecv) {
  HloModule module(TestName());
  HloComputation* computation = MakeSimpleLoop(&module, /*num_iters=*/1);
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* while_body = while_op->while_body();
  while_body->AddInstruction(
      HloInstruction::CreateRecv(ShapeUtil::MakeShape(F32, {1}),
                                 /*channel_id=*/0));
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module).ValueOrDie());
}

// The limitation on not being able to simplify loops that contain infeeds (and
// other non-removable instructions) isn't fundamental -- it just stems from the
// fact that our infrastructure sees simplifying such a loop as tantamount to
// removing the non-removable instruction.
TEST_F(WhileLoopSimplifierTest, NotRemovedIfContainsNonRemovableInstruction) {
  HloModule module(TestName());
  HloComputation* computation = MakeSimpleLoop(&module, /*num_iters=*/1);
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* while_body = while_op->while_body();
  while_body->AddInstruction(
      HloInstruction::CreateInfeed(ShapeUtil::MakeShape(F32, {1}), "config"));
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module).ValueOrDie());
}

}  // namespace
}  // namespace xla

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

#include "tensorflow/compiler/xla/service/while_loop_invariant_code_motion.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class WhileLoopInvariantCodeMotionTest : public HloVerifiedTestBase {
 public:
  // Makes a computation which has one parameter, of the given shape, and always
  // returns PRED[]{true}.  This is useful as a dummy loop condition.
  HloComputation* MakeAlwaysTrueComputation(const Shape& param_shape,
                                            HloModule* module);
};

static void FindOnlyWhileInstruction(HloComputation* computation,
                                     HloInstruction** while_instruction) {
  *while_instruction = nullptr;
  for (auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kWhile) {
      ASSERT_EQ(*while_instruction, nullptr);
      *while_instruction = instr;
    }
  }

  ASSERT_NE(*while_instruction, nullptr);
}

HloComputation* WhileLoopInvariantCodeMotionTest::MakeAlwaysTrueComputation(
    const Shape& param_shape, HloModule* module) {
  HloComputation::Builder builder(TestName() + ".always_true");
  builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(true)));
  return module->AddEmbeddedComputation(builder.Build());
}

TEST_F(WhileLoopInvariantCodeMotionTest, HoistOneInvariantOperation) {
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  Shape while_shape =
      ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32, scalar_s32});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 1));
    HloInstruction* add_result =
        builder.AddInstruction(HloInstruction::CreateBinary(
            scalar_s32, HloOpcode::kAdd, gte_0, gte_1));
    builder.AddInstruction(
        HloInstruction::CreateTuple({gte_0, gte_1, add_result}));

    return module().AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, &module()),
      while_body, init_value));
  HloComputation* entry_computation =
      module().AddEntryComputation(builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopInvariantCodeMotion{}.Run(&module()));
  EXPECT_TRUE(simplified_loop);

  HloInstruction* transformed_while;
  FindOnlyWhileInstruction(entry_computation, &transformed_while);

  EXPECT_THAT(entry_computation->instructions(), Contains(op::Add()));
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::Add())));
}

TEST_F(WhileLoopInvariantCodeMotionTest, HoistInvariantOperationTree) {
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  Shape while_shape =
      ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32, scalar_s32});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 1));
    HloInstruction* gte_2_loop_variant = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 2));

    HloInstruction* add_result =
        builder.AddInstruction(HloInstruction::CreateBinary(
            scalar_s32, HloOpcode::kAdd, gte_0, gte_1));
    HloInstruction* mul_result =
        builder.AddInstruction(HloInstruction::CreateBinary(
            scalar_s32, HloOpcode::kMultiply, add_result, gte_1));
    HloInstruction* negate_result =
        builder.AddInstruction(HloInstruction::CreateUnary(
            scalar_s32, HloOpcode::kNegate, mul_result));
    HloInstruction* constant = builder.AddInstruction(
        HloInstruction::CreateConstant(Literal::CreateR0<int32>(4)));
    HloInstruction* sub_result =
        builder.AddInstruction(HloInstruction::CreateBinary(
            scalar_s32, HloOpcode::kSubtract, negate_result, constant));
    HloInstruction* divide_result =
        builder.AddInstruction(HloInstruction::CreateBinary(
            scalar_s32, HloOpcode::kDivide, sub_result, gte_2_loop_variant));
    builder.AddInstruction(
        HloInstruction::CreateTuple({gte_0, gte_1, divide_result}));

    return module().AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, &module()),
      while_body, init_value));
  HloComputation* entry_computation =
      module().AddEntryComputation(builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopInvariantCodeMotion{}.Run(&module()));
  EXPECT_TRUE(simplified_loop);

  HloInstruction* transformed_while;
  FindOnlyWhileInstruction(entry_computation, &transformed_while);

  EXPECT_THAT(entry_computation->instructions(),
              AllOf(Contains(op::Add()), Contains(op::Multiply()),
                    Contains(op::Negate()), Contains(op::Subtract()),
                    Contains(op::Constant()),

                    // The division had a loop varying operand so that better
                    // not be hoisted.
                    Not(Contains(op::Divide()))));

  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(AnyOf(op::Add(), op::Multiply(), op::Negate(),
                             op::Subtract(), op::Constant()))));

  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Contains(op::Divide()));
}

TEST_F(WhileLoopInvariantCodeMotionTest,
       DontHoistTriviallyLoopVaryingComputation) {
  // Basic negative test: the add expression is not loop invariant.
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  Shape while_shape = ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 1));
    HloInstruction* add_result =
        builder.AddInstruction(HloInstruction::CreateBinary(
            scalar_s32, HloOpcode::kAdd, gte_0, gte_1));
    builder.AddInstruction(HloInstruction::CreateTuple({gte_0, add_result}));

    return module().AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  auto* while_inst = builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, &module()),
      while_body, init_value));

  module().AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopInvariantCodeMotion{}.Run(&module()));
  EXPECT_FALSE(simplified_loop);

  EXPECT_THAT(while_inst->while_body()->instructions(), Contains(op::Add()));
}

TEST_F(WhileLoopInvariantCodeMotionTest,
       DontHoistLoopVaryingComputationWithAlternatingTuples) {
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  Shape while_shape =
      ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32, scalar_s32});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 1));
    HloInstruction* add_result =
        builder.AddInstruction(HloInstruction::CreateBinary(
            scalar_s32, HloOpcode::kAdd, gte_0, gte_1));
    builder.AddInstruction(
        HloInstruction::CreateTuple({gte_1, gte_0, add_result}));

    return module().AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  auto* while_inst = builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, &module()),
      while_body, init_value));

  module().AddEntryComputation(builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopInvariantCodeMotion{}.Run(&module()));
  EXPECT_FALSE(simplified_loop);

  EXPECT_THAT(while_inst->while_body()->instructions(), Contains(op::Add()));
}

TEST_F(WhileLoopInvariantCodeMotionTest, DontHoistInstructionWithSideEffects) {
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  Shape while_shape = ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 1));
    builder.AddInstruction(
        HloInstruction::CreateOutfeed(scalar_s32, gte_0, ""));
    builder.AddInstruction(HloInstruction::CreateTuple({gte_0, gte_1}));

    return module().AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  auto* while_inst = builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, &module()),
      while_body, init_value));

  module().AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopInvariantCodeMotion{}.Run(&module()));
  EXPECT_FALSE(simplified_loop);

  EXPECT_THAT(while_inst->while_body()->instructions(),
              Contains(op::Outfeed()));
}

TEST_F(WhileLoopInvariantCodeMotionTest, DontHoistBitcastAlone) {
  // The bitcast's user, an outfeed, can't be hoisted, so don't hoist the
  // bitcast either.
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  auto scalar_f32 = ShapeUtil::MakeShape(F32, {});
  Shape while_shape = ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 1));
    HloInstruction* bitcast_inst = builder.AddInstruction(
        HloInstruction::CreateUnary(scalar_f32, HloOpcode::kBitcast, gte_0));
    builder.AddInstruction(
        HloInstruction::CreateOutfeed(scalar_f32, bitcast_inst, ""));
    builder.AddInstruction(HloInstruction::CreateTuple({gte_0, gte_1}));

    return module().AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  auto* while_inst = builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, &module()),
      while_body, init_value));

  module().AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopInvariantCodeMotion{}.Run(&module()));
  EXPECT_FALSE(simplified_loop);

  EXPECT_THAT(while_inst->while_body()->instructions(),
              Contains(op::Outfeed()));
  EXPECT_THAT(while_inst->while_body()->instructions(),
              Contains(op::Bitcast()));
}

TEST_F(WhileLoopInvariantCodeMotionTest, HoistBitcastIfNeeded) {
  // The bitcast's user can be hoisted, so hoist the bitcast too.
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  auto scalar_f32 = ShapeUtil::MakeShape(F32, {});
  Shape while_shape =
      ShapeUtil::MakeTupleShape({scalar_s32, scalar_f32, scalar_f32});

  HloComputation* while_body = [&]() {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_f32, param, 1));
    HloInstruction* bitcast_inst = builder.AddInstruction(
        HloInstruction::CreateUnary(scalar_f32, HloOpcode::kBitcast, gte_0));
    HloInstruction* add_inst =
        builder.AddInstruction(HloInstruction::CreateBinary(
            scalar_f32, HloOpcode::kAdd, bitcast_inst, gte_1));
    builder.AddInstruction(
        HloInstruction::CreateTuple({gte_0, gte_1, add_inst}));

    return module().AddEmbeddedComputation(builder.Build());
  }();

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, &module()),
      while_body, init_value));

  HloComputation* entry_computation =
      module().AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopInvariantCodeMotion{}.Run(&module()));
  EXPECT_TRUE(simplified_loop);

  HloInstruction* transformed_while;
  FindOnlyWhileInstruction(entry_computation, &transformed_while);

  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::Add())));
  EXPECT_THAT(transformed_while->while_body()->instructions(),
              Each(Not(op::Bitcast())));
  EXPECT_THAT(entry_computation->instructions(), Contains(op::Add()));
  EXPECT_THAT(entry_computation->instructions(), Contains(op::Bitcast()));
}

TEST_F(WhileLoopInvariantCodeMotionTest, DontHoistControlDependencies) {
  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
  Shape while_shape =
      ShapeUtil::MakeTupleShape({scalar_s32, scalar_s32, scalar_s32});

  HloComputation* while_body;
  {
    HloComputation::Builder builder(TestName() + ".while_body");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, while_shape, "param"));
    HloInstruction* gte_0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 0));
    HloInstruction* gte_1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_s32, param, 1));
    HloInstruction* add_result =
        builder.AddInstruction(HloInstruction::CreateBinary(
            scalar_s32, HloOpcode::kAdd, gte_0, gte_1));
    TF_ASSERT_OK(param->AddControlDependencyTo(add_result));
    builder.AddInstruction(
        HloInstruction::CreateTuple({gte_0, gte_1, add_result}));

    while_body = module().AddEmbeddedComputation(builder.Build());
  }

  HloComputation::Builder builder(TestName());
  auto* init_value = builder.AddInstruction(
      HloInstruction::CreateParameter(0, while_shape, "init_value"));
  builder.AddInstruction(HloInstruction::CreateWhile(
      while_shape, MakeAlwaysTrueComputation(while_shape, &module()),
      while_body, init_value));
  module().AddEntryComputation(builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(bool simplified_loop,
                          WhileLoopInvariantCodeMotion{}.Run(&module()));
  EXPECT_FALSE(simplified_loop);
}

TEST_F(WhileLoopInvariantCodeMotionTest, BodyHasNonTupleRoot) {
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
                          WhileLoopInvariantCodeMotion{}.Run(&module()));
  EXPECT_FALSE(simplified_loop);
}

}  // namespace
}  // namespace xla

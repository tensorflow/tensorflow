/* Copyright 2018 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/while_loop_condition_simplify.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using WhileLoopConditionSimplifyTest = HloTestBase;

TEST_F(WhileLoopConditionSimplifyTest, SimplifyDoubleConditionalTie) {
  auto hlo_module = CreateNewModule();

  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});
  int32 loop_bound = 10;
  /* Create while condition */
  auto builder_cond = HloComputation::Builder(TestName());
  auto tuple_cond = builder_cond.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
  auto limit0_cond = builder_cond.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(loop_bound)));
  auto limit1_cond = builder_cond.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(loop_bound)));
  auto c0_cond =
      builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
          ShapeUtil::MakeShape(S32, {}), tuple_cond, 0));
  auto c1_cond =
      builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
          ShapeUtil::MakeShape(S32, {}), tuple_cond, 1));
  auto lt0_cond = builder_cond.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c0_cond, limit0_cond));
  auto lt1_cond = builder_cond.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c1_cond, limit1_cond));
  auto and_cond = builder_cond.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(PRED, {}), HloOpcode::kAnd, lt0_cond, lt1_cond));
  HloComputation* comp_cond =
      hlo_module->AddEmbeddedComputation(builder_cond.Build());

  /* Create while body */
  auto builder_body = HloComputation::Builder(TestName());
  auto tuple_body = builder_body.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
  auto c0_body = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, tuple_body, 0));
  auto c1_body = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, tuple_body, 1));
  auto one = builder_body.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
  auto new_c0_body = builder_body.AddInstruction(HloInstruction::CreateBinary(
      c0_body->shape(), HloOpcode::kAdd, c0_body, one));
  auto new_c1_body = builder_body.AddInstruction(HloInstruction::CreateBinary(
      c1_body->shape(), HloOpcode::kAdd, c1_body, one));

  auto new_tuple_body = builder_body.AddInstruction(
      HloInstruction::CreateTuple({new_c0_body, new_c1_body}));

  HloComputation* comp_body =
      hlo_module->AddEmbeddedComputation(builder_body.Build());

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c0_init = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto c1_init = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));

  auto init = builder_main.AddInstruction(
      HloInstruction::CreateTuple({c0_init, c1_init}));

  auto main = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  builder_main.AddInstruction(HloInstruction::CreateTuple({main}));

  hlo_module->AddEntryComputation(builder_main.Build());
  WhileLoopConditionSimplify wlcs;
  EXPECT_TRUE(wlcs.Run(hlo_module.get()).ValueOrDie());
  EXPECT_TRUE(limit0_cond->parent() == comp_cond ||
              limit1_cond->parent() == comp_cond);
  if (limit0_cond->parent() == comp_cond) {
    EXPECT_TRUE(limit0_cond->parent() == comp_cond);
    EXPECT_TRUE(limit1_cond->parent() != comp_cond);
    EXPECT_TRUE(c0_cond->parent() == comp_cond);
    EXPECT_TRUE(c1_cond->parent() != comp_cond);
    EXPECT_TRUE(lt0_cond->parent() == comp_cond);
    EXPECT_TRUE(lt1_cond->parent() != comp_cond);
    EXPECT_TRUE(and_cond->parent() != comp_cond);
    EXPECT_TRUE(c0_body->parent() == comp_body);
    EXPECT_TRUE(c1_body->parent() != comp_body);
    EXPECT_TRUE(new_c0_body->parent() == comp_body);
    EXPECT_TRUE(new_c1_body->parent() != comp_body);
    EXPECT_EQ(new_tuple_body->operand(0), new_c0_body);
    EXPECT_EQ(new_tuple_body->operand(1), new_c0_body);
  } else {
    EXPECT_TRUE(limit0_cond->parent() != comp_cond);
    EXPECT_TRUE(limit1_cond->parent() == comp_cond);
    EXPECT_TRUE(c0_cond->parent() != comp_cond);
    EXPECT_TRUE(c1_cond->parent() == comp_cond);
    EXPECT_TRUE(lt0_cond->parent() != comp_cond);
    EXPECT_TRUE(lt1_cond->parent() == comp_cond);
    EXPECT_TRUE(and_cond->parent() != comp_cond);
    EXPECT_TRUE(c0_body->parent() != comp_body);
    EXPECT_TRUE(c1_body->parent() == comp_body);
    EXPECT_TRUE(new_c0_body->parent() != comp_body);
    EXPECT_TRUE(new_c1_body->parent() == comp_body);
    EXPECT_EQ(new_tuple_body->operand(0), new_c1_body);
    EXPECT_EQ(new_tuple_body->operand(1), new_c1_body);
  }
}

TEST_F(WhileLoopConditionSimplifyTest, SimplifyDoubleConditionalUneven) {
  auto hlo_module = CreateNewModule();

  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});
  int32 l_bound0 = 10;
  int32 l_bound1 = 15;
  /* Create while condition */
  auto builder_cond = HloComputation::Builder(TestName());
  auto tuple_cond = builder_cond.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
  auto limit0_cond = builder_cond.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(l_bound0)));
  auto limit1_cond = builder_cond.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(l_bound1)));
  auto c0_cond =
      builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
          ShapeUtil::MakeShape(S32, {}), tuple_cond, 0));
  auto c1_cond =
      builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
          ShapeUtil::MakeShape(S32, {}), tuple_cond, 1));
  auto lt0_cond = builder_cond.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c0_cond, limit0_cond));
  auto lt1_cond = builder_cond.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c1_cond, limit1_cond));
  auto and_cond = builder_cond.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(PRED, {}), HloOpcode::kAnd, lt0_cond, lt1_cond));
  HloComputation* comp_cond =
      hlo_module->AddEmbeddedComputation(builder_cond.Build());

  /* Create while body */
  auto builder_body = HloComputation::Builder(TestName());
  auto tuple_body = builder_body.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
  auto c0_body = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, tuple_body, 0));
  auto c1_body = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, tuple_body, 1));
  auto one = builder_body.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
  auto new_c0_body = builder_body.AddInstruction(HloInstruction::CreateBinary(
      c0_body->shape(), HloOpcode::kAdd, c0_body, one));
  auto new_c1_body = builder_body.AddInstruction(HloInstruction::CreateBinary(
      c1_body->shape(), HloOpcode::kAdd, c1_body, one));

  auto new_tuple_body = builder_body.AddInstruction(
      HloInstruction::CreateTuple({new_c0_body, new_c1_body}));

  HloComputation* comp_body =
      hlo_module->AddEmbeddedComputation(builder_body.Build());

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c0_init = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto c1_init = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));

  auto init = builder_main.AddInstruction(
      HloInstruction::CreateTuple({c0_init, c1_init}));

  auto main = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  builder_main.AddInstruction(HloInstruction::CreateTuple({main}));

  hlo_module->AddEntryComputation(builder_main.Build());
  WhileLoopConditionSimplify wlcs;
  EXPECT_TRUE(wlcs.Run(hlo_module.get()).ValueOrDie());
  EXPECT_TRUE(limit0_cond->parent() == comp_cond);
  EXPECT_TRUE(limit1_cond->parent() != comp_cond);
  EXPECT_TRUE(c0_cond->parent() == comp_cond);
  EXPECT_TRUE(c1_cond->parent() != comp_cond);
  EXPECT_TRUE(lt0_cond->parent() == comp_cond);
  EXPECT_TRUE(lt1_cond->parent() != comp_cond);
  EXPECT_TRUE(and_cond->parent() != comp_cond);
  EXPECT_TRUE(c0_body->parent() == comp_body);
  EXPECT_TRUE(c1_body->parent() != comp_body);
  EXPECT_TRUE(new_c0_body->parent() == comp_body);
  EXPECT_TRUE(new_c1_body->parent() != comp_body);
  EXPECT_EQ(new_tuple_body->operand(0), new_c0_body);
  EXPECT_EQ(new_tuple_body->operand(1), new_c0_body);
}

TEST_F(WhileLoopConditionSimplifyTest, DontSimplifyNonIntegral) {
  auto hlo_module = CreateNewModule();

  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  /* Create while condition */
  HloComputation* comp_cond;
  {
    auto builder_cond = HloComputation::Builder(TestName());
    auto tuple = builder_cond.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
    auto limit0 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(10)));
    auto limit1 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(12)));
    auto c0 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(F32, {}), tuple, 0));
    auto c1 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(F32, {}), tuple, 1));
    auto lt0 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c0, limit0));
    auto lt1 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kGt, c1, limit1));
    builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kAnd, lt0, lt1));

    comp_cond = hlo_module->AddEmbeddedComputation(builder_cond.Build());
  }

  /* Create while body */
  HloComputation* comp_body;
  {
    auto builder_body = HloComputation::Builder(TestName());
    auto tuple = builder_body.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
    auto c0 = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 0));
    auto c1 = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 1));
    auto one = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1)));
    auto new_c0 = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c0->shape(), HloOpcode::kAdd, c0, one));
    auto new_c1 = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c1->shape(), HloOpcode::kAdd, c1, one));

    builder_body.AddInstruction(HloInstruction::CreateTuple({new_c0, new_c1}));

    comp_body = hlo_module->AddEmbeddedComputation(builder_body.Build());
  }

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c0 = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
  auto c1 = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));

  auto init =
      builder_main.AddInstruction(HloInstruction::CreateTuple({c0, c1}));

  auto main = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  builder_main.AddInstruction(HloInstruction::CreateTuple({main}));

  hlo_module->AddEntryComputation(builder_main.Build());
  WhileLoopConditionSimplify wlcs;
  EXPECT_FALSE(wlcs.Run(hlo_module.get()).ValueOrDie());
}

TEST_F(WhileLoopConditionSimplifyTest, DontSimplifyNonUniqueCond) {
  auto hlo_module = CreateNewModule();

  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape});

  /* Create while condition */
  HloComputation* comp_cond;
  {
    auto builder_cond = HloComputation::Builder(TestName());
    auto tuple = builder_cond.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
    auto limit = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(10)));
    auto c = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 0));
    auto lt = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c, limit));
    builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kAnd, lt, lt));

    comp_cond = hlo_module->AddEmbeddedComputation(builder_cond.Build());
  }

  /* Create while body */
  HloComputation* comp_body;
  {
    auto builder_body = HloComputation::Builder(TestName());
    auto tuple = builder_body.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
    auto c = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 0));
    auto one = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
    auto new_c = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c->shape(), HloOpcode::kAdd, c, one));

    builder_body.AddInstruction(HloInstruction::CreateTuple({new_c}));

    comp_body = hlo_module->AddEmbeddedComputation(builder_body.Build());
  }

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));

  auto init = builder_main.AddInstruction(HloInstruction::CreateTuple({c}));

  auto main = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  builder_main.AddInstruction(HloInstruction::CreateTuple({main}));

  hlo_module->AddEntryComputation(builder_main.Build());
  WhileLoopConditionSimplify wlcs;
  EXPECT_FALSE(wlcs.Run(hlo_module.get()).ValueOrDie());
}

TEST_F(WhileLoopConditionSimplifyTest, DontSimplifyIncrementNotOne) {
  auto hlo_module = CreateNewModule();

  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  /* Create while condition */
  HloComputation* comp_cond;
  {
    auto builder_cond = HloComputation::Builder(TestName());
    auto tuple = builder_cond.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
    auto limit0 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(10)));
    auto limit1 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(12)));
    auto c0 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 0));
    auto c1 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 1));
    auto lt0 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c0, limit0));
    auto lt1 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kGt, c1, limit1));
    builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kAnd, lt0, lt1));

    comp_cond = hlo_module->AddEmbeddedComputation(builder_cond.Build());
  }

  /* Create while body */
  HloComputation* comp_body;
  {
    auto builder_body = HloComputation::Builder(TestName());
    auto tuple = builder_body.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
    auto c0 = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 0));
    auto c1 = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 1));
    auto one = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
    auto two = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(2)));
    auto new_c0 = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c0->shape(), HloOpcode::kAdd, c0, one));
    auto new_c1 = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c1->shape(), HloOpcode::kAdd, c1, two));

    builder_body.AddInstruction(HloInstruction::CreateTuple({new_c0, new_c1}));

    comp_body = hlo_module->AddEmbeddedComputation(builder_body.Build());
  }

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c0 = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto c1 = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));

  auto init =
      builder_main.AddInstruction(HloInstruction::CreateTuple({c0, c1}));

  auto main = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  builder_main.AddInstruction(HloInstruction::CreateTuple({main}));

  hlo_module->AddEntryComputation(builder_main.Build());
  WhileLoopConditionSimplify wlcs;
  EXPECT_FALSE(wlcs.Run(hlo_module.get()).ValueOrDie());
}

TEST_F(WhileLoopConditionSimplifyTest, DontSimplifyNonConst) {
  auto hlo_module = CreateNewModule();

  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  /* Create while condition */
  HloComputation* comp_cond;
  {
    auto builder_cond = HloComputation::Builder(TestName());
    auto tuple = builder_cond.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
    auto limit0 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(10)));
    auto limit1 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(12)));
    auto c0 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 0));
    auto c1 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 1));
    auto lt0 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c0, limit0));
    auto lt1 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kGt, c1, limit1));
    builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kAnd, lt0, lt1));

    comp_cond = hlo_module->AddEmbeddedComputation(builder_cond.Build());
  }

  /* Create while body */
  HloComputation* comp_body;
  {
    auto builder_body = HloComputation::Builder(TestName());
    auto tuple = builder_body.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
    auto c0 = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 0));
    auto c1 = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 1));
    auto one = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
    auto new_c0 = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c0->shape(), HloOpcode::kAdd, c0, one));
    auto new_c1 = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c1->shape(), HloOpcode::kAdd, c1, one));

    builder_body.AddInstruction(HloInstruction::CreateTuple({new_c0, new_c1}));

    comp_body = hlo_module->AddEmbeddedComputation(builder_body.Build());
  }

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c0 = builder_main.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "c0"));
  auto c1 = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));

  auto init =
      builder_main.AddInstruction(HloInstruction::CreateTuple({c0, c1}));

  auto main = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  builder_main.AddInstruction(HloInstruction::CreateTuple({main}));

  hlo_module->AddEntryComputation(builder_main.Build());
  WhileLoopConditionSimplify wlcs;
  EXPECT_FALSE(wlcs.Run(hlo_module.get()).ValueOrDie());
}

TEST_F(WhileLoopConditionSimplifyTest,
       SimplifyDoubleConditionalCheckUsesReplaced) {
  auto hlo_module = CreateNewModule();

  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape, scalar_shape});
  int32 loop_bound0 = 10;
  int32 loop_bound1 = 9;
  /* Create while condition */
  auto builder_cond = HloComputation::Builder(TestName());
  auto tuple_cond = builder_cond.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
  auto limit0_cond = builder_cond.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(loop_bound0)));
  auto limit1_cond = builder_cond.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(loop_bound1)));
  auto c0_cond =
      builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
          ShapeUtil::MakeShape(S32, {}), tuple_cond, 0));
  auto c1_cond =
      builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
          ShapeUtil::MakeShape(S32, {}), tuple_cond, 1));
  auto lt0_cond = builder_cond.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c0_cond, limit0_cond));
  auto lt1_cond = builder_cond.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c1_cond, limit1_cond));
  auto and_cond = builder_cond.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(PRED, {}), HloOpcode::kAnd, lt0_cond, lt1_cond));
  HloComputation* comp_cond =
      hlo_module->AddEmbeddedComputation(builder_cond.Build());

  /* Create while body */
  auto builder_body = HloComputation::Builder(TestName());
  auto tuple_body = builder_body.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
  auto c0_body = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, tuple_body, 0));
  auto c1_body = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, tuple_body, 1));
  auto val = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, tuple_body, 2));
  auto one = builder_body.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
  auto new_c0_body = builder_body.AddInstruction(HloInstruction::CreateBinary(
      c0_body->shape(), HloOpcode::kAdd, c0_body, one));
  auto new_c1_body = builder_body.AddInstruction(HloInstruction::CreateBinary(
      c1_body->shape(), HloOpcode::kAdd, c1_body, one));
  auto new_val = builder_body.AddInstruction(HloInstruction::CreateBinary(
      val->shape(), HloOpcode::kAdd, c0_body, val));

  auto new_tuple_body = builder_body.AddInstruction(
      HloInstruction::CreateTuple({new_c0_body, new_c1_body, new_val}));

  HloComputation* comp_body =
      hlo_module->AddEmbeddedComputation(builder_body.Build());

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c0_init = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto c1_init = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto val_init = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));

  auto init = builder_main.AddInstruction(
      HloInstruction::CreateTuple({c0_init, c1_init, val_init}));

  auto main = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  builder_main.AddInstruction(HloInstruction::CreateTuple({main}));

  hlo_module->AddEntryComputation(builder_main.Build());
  WhileLoopConditionSimplify wlcs;
  EXPECT_TRUE(wlcs.Run(hlo_module.get()).ValueOrDie());
  EXPECT_TRUE(limit0_cond->parent() != comp_cond);
  EXPECT_TRUE(limit1_cond->parent() == comp_cond);
  EXPECT_TRUE(c0_cond->parent() != comp_cond);
  EXPECT_TRUE(c1_cond->parent() == comp_cond);
  EXPECT_TRUE(lt0_cond->parent() != comp_cond);
  EXPECT_TRUE(lt1_cond->parent() == comp_cond);
  EXPECT_TRUE(and_cond->parent() != comp_cond);
  EXPECT_TRUE(c0_body->parent() != comp_body);
  EXPECT_TRUE(c1_body->parent() == comp_body);
  EXPECT_TRUE(new_c0_body->parent() != comp_body);
  EXPECT_TRUE(new_c1_body->parent() == comp_body);
  EXPECT_EQ(new_val->operand(0), c1_body);
  EXPECT_EQ(new_tuple_body->operand(0), new_c1_body);
  EXPECT_EQ(new_tuple_body->operand(1), new_c1_body);
}

TEST_F(WhileLoopConditionSimplifyTest, DontSimplifySingleConditional) {
  auto hlo_module = CreateNewModule();

  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape});

  /* Create while condition */
  HloComputation* comp_cond;
  {
    auto builder_cond = HloComputation::Builder(TestName());
    auto tuple = builder_cond.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
    auto limit = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(10)));
    auto c = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 0));
    builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c, limit));

    comp_cond = hlo_module->AddEmbeddedComputation(builder_cond.Build());
  }

  /* Create while body */
  HloComputation* comp_body;
  {
    auto builder_body = HloComputation::Builder(TestName());
    auto tuple = builder_body.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
    auto c = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 0));
    auto one = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
    auto new_c = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c->shape(), HloOpcode::kAdd, c, one));

    builder_body.AddInstruction(HloInstruction::CreateTuple({new_c}));

    comp_body = hlo_module->AddEmbeddedComputation(builder_body.Build());
  }

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));

  auto init = builder_main.AddInstruction(HloInstruction::CreateTuple({c}));

  auto main = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  builder_main.AddInstruction(HloInstruction::CreateTuple({main}));

  hlo_module->AddEntryComputation(builder_main.Build());
  WhileLoopConditionSimplify wlcs;
  EXPECT_FALSE(wlcs.Run(hlo_module.get()).ValueOrDie());
}

TEST_F(WhileLoopConditionSimplifyTest, DontSimplifyTripleConditional) {
  auto hlo_module = CreateNewModule();

  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape, scalar_shape});

  /* Create while condition */
  HloComputation* comp_cond;
  {
    auto builder_cond = HloComputation::Builder(TestName());
    auto tuple = builder_cond.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
    auto limit0 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(10)));
    auto limit1 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(12)));
    auto limit2 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(2)));
    auto c0 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 0));
    auto c1 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 1));
    auto c2 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 2));
    auto lt0 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c0, limit0));
    auto lt1 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c1, limit1));
    auto lt2 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c2, limit2));
    auto and_lt0_lt1 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kAnd, lt0, lt1));
    builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kAnd, and_lt0_lt1, lt2));

    comp_cond = hlo_module->AddEmbeddedComputation(builder_cond.Build());
  }

  /* Create while body */
  HloComputation* comp_body;
  {
    auto builder_body = HloComputation::Builder(TestName());
    auto tuple = builder_body.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
    auto c0 = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 0));
    auto c1 = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 1));
    auto c2 = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 2));
    auto one = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
    auto new_c0 = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c0->shape(), HloOpcode::kAdd, c0, one));
    auto new_c1 = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c1->shape(), HloOpcode::kAdd, c1, one));
    auto new_c2 = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c2->shape(), HloOpcode::kAdd, c2, one));

    builder_body.AddInstruction(
        HloInstruction::CreateTuple({new_c0, new_c1, new_c2}));

    comp_body = hlo_module->AddEmbeddedComputation(builder_body.Build());
  }

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c0 = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto c1 = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto c2 = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));

  auto init =
      builder_main.AddInstruction(HloInstruction::CreateTuple({c0, c1, c2}));

  auto main = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  builder_main.AddInstruction(HloInstruction::CreateTuple({main}));

  hlo_module->AddEntryComputation(builder_main.Build());
  WhileLoopConditionSimplify wlcs;
  EXPECT_FALSE(wlcs.Run(hlo_module.get()).ValueOrDie());
}

TEST_F(WhileLoopConditionSimplifyTest, DontSimplifyAnythingButLTs) {
  auto hlo_module = CreateNewModule();

  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  /* Create while condition */
  HloComputation* comp_cond;
  {
    auto builder_cond = HloComputation::Builder(TestName());
    auto tuple = builder_cond.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
    auto limit0 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(10)));
    auto limit1 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(12)));
    auto c0 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 0));
    auto c1 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 1));
    auto lt0 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c0, limit0));
    auto lt1 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kGt, c1, limit1));
    builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kAnd, lt0, lt1));

    comp_cond = hlo_module->AddEmbeddedComputation(builder_cond.Build());
  }

  /* Create while body */
  HloComputation* comp_body;
  {
    auto builder_body = HloComputation::Builder(TestName());
    auto tuple = builder_body.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
    auto c0 = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 0));
    auto c1 = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 1));
    auto one = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
    auto new_c0 = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c0->shape(), HloOpcode::kAdd, c0, one));
    auto new_c1 = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c1->shape(), HloOpcode::kAdd, c1, one));

    builder_body.AddInstruction(HloInstruction::CreateTuple({new_c0, new_c1}));

    comp_body = hlo_module->AddEmbeddedComputation(builder_body.Build());
  }

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c0 = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto c1 = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));

  auto init =
      builder_main.AddInstruction(HloInstruction::CreateTuple({c0, c1}));

  auto main = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  builder_main.AddInstruction(HloInstruction::CreateTuple({main}));

  hlo_module->AddEntryComputation(builder_main.Build());
  WhileLoopConditionSimplify wlcs;
  EXPECT_FALSE(wlcs.Run(hlo_module.get()).ValueOrDie());
}

TEST_F(WhileLoopConditionSimplifyTest, DontSimplifyAnythingButANDs) {
  auto hlo_module = CreateNewModule();

  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  /* Create while condition */
  HloComputation* comp_cond;
  {
    auto builder_cond = HloComputation::Builder(TestName());
    auto tuple = builder_cond.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
    auto limit0 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(10)));
    auto limit1 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(12)));
    auto c0 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 0));
    auto c1 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(S32, {}), tuple, 1));
    auto lt0 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c0, limit0));
    auto lt1 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c1, limit1));
    builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kOr, lt0, lt1));

    comp_cond = hlo_module->AddEmbeddedComputation(builder_cond.Build());
  }

  /* Create while body */
  HloComputation* comp_body;
  {
    auto builder_body = HloComputation::Builder(TestName());
    auto tuple = builder_body.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
    auto c0 = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 0));
    auto c1 = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 1));
    auto one = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
    auto new_c0 = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c0->shape(), HloOpcode::kAdd, c0, one));
    auto new_c1 = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c1->shape(), HloOpcode::kAdd, c1, one));

    builder_body.AddInstruction(HloInstruction::CreateTuple({new_c0, new_c1}));

    comp_body = hlo_module->AddEmbeddedComputation(builder_body.Build());
  }

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto c0 = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto c1 = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));

  auto init =
      builder_main.AddInstruction(HloInstruction::CreateTuple({c0, c1}));

  auto main = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  builder_main.AddInstruction(HloInstruction::CreateTuple({main}));

  hlo_module->AddEntryComputation(builder_main.Build());
  WhileLoopConditionSimplify wlcs;
  EXPECT_FALSE(wlcs.Run(hlo_module.get()).ValueOrDie());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla

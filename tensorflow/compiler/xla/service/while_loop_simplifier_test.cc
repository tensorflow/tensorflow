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
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class WhileLoopSimplifierTest : public HloVerifiedTestBase {
 protected:
  // Makes an HloModule that contains a loop with `num_iters` iteration.
  void MakeModuleWithSimpleLoop(int num_iters);

  // Similar to MakeModuleWithSimpleLoop except that the loop bound is passed to
  // the loop-condition through an element of a tuple which is the
  // loop-condition parameter.
  void MakeModuleWithSimpleLoopTupleElementLoopBound(int num_iters);
};

void WhileLoopSimplifierTest::MakeModuleWithSimpleLoop(int num_iters) {
  string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] less-than(get-tuple-element.3, constant.2)
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";

  string hlo_string = tensorflow::str_util::StringReplace(
      hlo_string_template, "{{LOOP_BOUND}}",
      tensorflow::strings::StrCat(42 + num_iters),
      /*replace_all=*/true);
  ParseAndVerifyModule(hlo_string);
}

void WhileLoopSimplifierTest::MakeModuleWithSimpleLoopTupleElementLoopBound(
    int num_iters) {
  string hlo_string_template = R"(
  HloModule SimpleLoopWithIndirectLoopBound
  SimpleLoopWithIndirectLoopBound.body {
    loop_var.1 = (s32[], s32[3]{0}, s32[]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    limit = s32[] get-tuple-element(loop_var.1), index=2
    ROOT tuple = (s32[], s32[3]{0}, s32[]) tuple(add, multiply, limit)
  }
  SimpleLoopWithIndirectLoopBound.condition {
    loop_var.2 = (s32[], s32[3]{0}, s32[]) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    get-tuple-element.4 = s32[] get-tuple-element(loop_var.2), index=2
    ROOT less-than = pred[] less-than(get-tuple-element.3, get-tuple-element.4)
  }
  ENTRY SimpleLoopWithIndirectLoopBound {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    constant.2 = s32[] constant({{LOOP_BOUND}})
    tuple.1 = (s32[], s32[3]{0}, s32[]) tuple(constant.3, constant.4,
      constant.2)
    ROOT while = (s32[], s32[3]{0}, s32[]) while(tuple.1),
      condition=SimpleLoopWithIndirectLoopBound.condition,
      body=SimpleLoopWithIndirectLoopBound.body
  }
  )";

  string hlo_string = tensorflow::str_util::StringReplace(
      hlo_string_template, "{{LOOP_BOUND}}",
      tensorflow::strings::StrCat(42 + num_iters),
      /*replace_all=*/true);
  ParseAndVerifyModule(hlo_string);
}

TEST_F(WhileLoopSimplifierTest, LoopWithZeroIterationSimiplified) {
  MakeModuleWithSimpleLoop(/*num_iters=*/0);
  HloModule* the_module = &module();
  ASSERT_TRUE(WhileLoopSimplifier().Run(the_module).ValueOrDie());
  EXPECT_THAT(the_module->entry_computation()->root_instruction(),
              op::Tuple(op::Constant(), op::Constant()));
}

TEST_F(WhileLoopSimplifierTest,
       LoopWithZeroIterationTupleElementLoopBoundSimplified) {
  MakeModuleWithSimpleLoopTupleElementLoopBound(/*num_iters=*/0);
  HloModule* the_module = &module();
  ASSERT_TRUE(WhileLoopSimplifier().Run(the_module).ValueOrDie());
  EXPECT_THAT(the_module->entry_computation()->root_instruction(),
              op::Tuple(op::Constant(), op::Constant(), op::Constant()));
}

TEST_F(WhileLoopSimplifierTest, LoopWithOneIterationSimplified) {
  MakeModuleWithSimpleLoop(/*num_iters=*/1);
  HloModule* the_module = &module();
  ASSERT_TRUE(WhileLoopSimplifier().Run(the_module).ValueOrDie());
  EXPECT_THAT(the_module->entry_computation()->root_instruction(),
              op::Tuple(op::Add(), op::Multiply()));
}

TEST_F(WhileLoopSimplifierTest,
       LoopWithOneIterationTupleELementLoopBoundSimplified) {
  MakeModuleWithSimpleLoopTupleElementLoopBound(/*num_iters=*/1);
  HloModule* the_module = &module();
  ASSERT_TRUE(WhileLoopSimplifier().Run(the_module).ValueOrDie());
  EXPECT_THAT(the_module->entry_computation()->root_instruction(),
              op::Tuple(op::Add(), op::Multiply(), op::Constant()));
}

TEST_F(WhileLoopSimplifierTest, LoopWithTwoIterationsNotSimplified) {
  MakeModuleWithSimpleLoop(/*num_iters=*/2);
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

TEST_F(WhileLoopSimplifierTest,
       LoopWithControlDependencySimplifiedDependencyPreserved) {
  MakeModuleWithSimpleLoop(/*num_iters=*/1);
  HloModule* the_module = &module();
  HloComputation* computation = the_module->entry_computation();
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* true_op = while_op->while_body()->AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(true)));
  TF_ASSERT_OK(true_op->AddControlDependencyTo(
      while_op->while_body()->root_instruction()));
  ASSERT_TRUE(WhileLoopSimplifier().Run(the_module).ValueOrDie());
  EXPECT_THAT(computation->root_instruction()->control_predecessors(),
              ElementsAre(op::Constant()))
      << computation->ToString();
}

// Loops that contain send/recv nodes can't be simplified; the loop structure
// around send/recv nodes must be preserved.
TEST_F(WhileLoopSimplifierTest, LoopWithSendNotSimplified) {
  MakeModuleWithSimpleLoop(/*num_iters=*/1);
  HloModule* the_module = &module();
  HloComputation* computation = the_module->entry_computation();
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* while_body = while_op->while_body();
  auto* token = while_body->AddInstruction(HloInstruction::CreateAfterAll({}));
  auto* send = while_body->AddInstruction(HloInstruction::CreateSend(
      while_body->AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<bool>(true))),
      token,
      /*channel_id=*/0));
  while_body->AddInstruction(HloInstruction::CreateSendDone(send));
  EXPECT_FALSE(WhileLoopSimplifier().Run(the_module).ValueOrDie());
}

TEST_F(WhileLoopSimplifierTest, LoopWithRecvNotSimplified) {
  MakeModuleWithSimpleLoop(/*num_iters=*/1);
  HloModule* the_module = &module();
  HloComputation* computation = the_module->entry_computation();
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* while_body = while_op->while_body();
  auto* token = while_body->AddInstruction(HloInstruction::CreateAfterAll({}));
  auto* recv = while_body->AddInstruction(
      HloInstruction::CreateRecv(ShapeUtil::MakeShape(F32, {1}), token,
                                 /*channel_id=*/0));
  while_body->AddInstruction(HloInstruction::CreateRecvDone(recv));
  EXPECT_FALSE(WhileLoopSimplifier().Run(the_module).ValueOrDie());
}

// The limitation on not being able to simplify loops that contain infeeds (and
// other non-removable instructions) isn't fundamental -- it just stems from the
// fact that our infrastructure sees simplifying such a loop as tantamount to
// removing the non-removable instruction.
TEST_F(WhileLoopSimplifierTest, LoopWithInfeedNotSimplified) {
  MakeModuleWithSimpleLoop(/*num_iters=*/1);
  HloModule* the_module = &module();
  HloComputation* computation = the_module->entry_computation();
  auto* while_op = computation->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  auto* while_body = while_op->while_body();
  auto token = while_body->AddInstruction(HloInstruction::CreateAfterAll({}));
  while_body->AddInstruction(HloInstruction::CreateInfeed(
      ShapeUtil::MakeShape(F32, {1}), token, "config"));
  EXPECT_FALSE(WhileLoopSimplifier().Run(the_module).ValueOrDie());
}

// A non-tuple shaped loop shouldn't be simplified or crash the compiler.
TEST_F(WhileLoopSimplifierTest, NonTupleShapedLoopNotSimplified) {
  const string hlo_string = R"(
 HloModule NonTupleShapedLoop
 NonTupleShapedLoop.body {
   loop_var.1 = s32[] parameter(0)
   constant.1 = s32[] constant(-1)
   ROOT add = s32[] add(s32[] loop_var.1, s32[] constant.1)
 }
 NonTupleShapedLoop.condition {
   loop_var = s32[] parameter(0)
   constant = s32[] constant(100)
   ROOT less-than = pred[] less-than(s32[] loop_var, s32[] constant)
 }
 ENTRY INonTupleShapedLoop {
   constant.2 = s32[] constant(42)
   ROOT while = s32[] while(s32[] constant.2),
     condition=NonTupleShapedLoop.condition,
     body=NonTupleShapedLoop.body
  }
  )";

  ParseAndVerifyModule(hlo_string);
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

// A while loop that does nothing else besides swapping tuple elements
// can't be simplified as the result of the swapping is visible to users of the
// loop.
TEST_F(WhileLoopSimplifierTest, LoopSwappingTupleElementsNotSimplified) {
  const string hlo_string = R"(
  HloModule SwappingTupleElements
  SwappingTupleElements.body {
    loop_var = (s32[], s32[]) parameter(0)
    get-tuple-element = s32[] get-tuple-element((s32[], s32[]) loop_var),index=1
    get-tuple-element.1 = s32[] get-tuple-element((s32[], s32[]) loop_var),
      index=0
    ROOT tuple = (s32[], s32[]) tuple(s32[] get-tuple-element,
      s32[] get-tuple-element.1)
  }
  SwappingTupleElements.always_true {
   param = (s32[], s32[]) parameter(0)
   ROOT constant = pred[] constant(true)
  }
  ENTRY SwappingTupleElements {
   x = s32[] parameter(0)
   y = s32[] parameter(1)
   tuple.1 = (s32[], s32[]) tuple(s32[] x, s32[] y)
   ROOT while = (s32[], s32[]) while((s32[], s32[]) tuple.1),
     condition=SwappingTupleElements.always_true,
     body=SwappingTupleElements.body
  }
  )";

  ParseAndVerifyModule(hlo_string);
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

// Construct a loop where we assign a constant to tuple element 0 in each
// iteration.  We can't eliminate tuple element 0, even though we never use its
// value.
TEST_F(WhileLoopSimplifierTest,
       LoopWithUnusedButModifiedTupleElementNotSimplified) {
  const string hlo_string = R"(
  HloModule UnusedButModifiedTupleElement
  UnusedButModifiedTupleElement.body {
    loop_var = (s32[]) parameter(0)
    constant.1 = s32[] constant(1)
    ROOT tuple = (s32[]) tuple(s32[] constant.1)
  }
  UnusedButModifiedTupleElement.always_true {
    param = (s32[]) parameter(0)
   ROOT  constant = pred[] constant(true)
  }
  ENTRY  UnusedButModifiedTupleElement {
    constant.2 = s32[] constant(0)
    tuple.1 = (s32[]) tuple(s32[]  constant.2)
    ROOT while = (s32[]) while((s32[]) tuple.1),
      condition=UnusedButModifiedTupleElement.always_true,
      body=UnusedButModifiedTupleElement.body
  }
  )";

  ParseAndVerifyModule(hlo_string);
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

// Nothing to simplify in a while loop whose tuple has 0 elements.
TEST_F(WhileLoopSimplifierTest, LoopWithEmptyTupleNotSimplified) {
  const string hlo_string = R"(
  HloModule EmptyTuple
  EmptyTuple.body {
    loop_var = () parameter(0)
    ROOT  tuple = () tuple()
  }
  EmptyTuple.always_true {
   param = () parameter(0)
   ROOT constant = pred[] constant(true)
  }
  ENTRY EmptyTuple {
   tuple.1 = () tuple()
   ROOT while = () while(() tuple.1), condition=EmptyTuple.always_true,
     body=EmptyTuple.body
  }
  )";

  ParseAndVerifyModule(hlo_string);
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

// While loop where one tuple element is used twice in the body, and thus can't
// be simplified away.
TEST_F(WhileLoopSimplifierTest, LoopWithElemUsedTwiceNotSimplified) {
  const string hlo_string = R"(
  HloModule ElemUsedTwice
  ElemUsedTwice.body {
    param0 = (s32[], s32[]) parameter(0)
    get-tuple-element = s32[] get-tuple-element((s32[], s32[]) param0), index=0
    ROOT tuple = (s32[], s32[]) tuple(s32[] get-tuple-element,
      s32[] get-tuple-element)
  }
  ElemUsedTwice.always_true {
    param = (s32[], s32[]) parameter(0)
    ROOT constant = pred[] constant(true)
  }
  ENTRY ElemUsedTwice {
   x = s32[] parameter(0)
   y = s32[] parameter(1)
   tuple.1 = (s32[], s32[]) tuple(s32[] x, s32[] y)
   ROOT while = (s32[], s32[]) while((s32[], s32[]) tuple.1),
     condition=ElemUsedTwice.always_true, body=ElemUsedTwice.body
  }
  )";

  ParseAndVerifyModule(hlo_string);
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

// This while loop has three tuple elements.  Element 0 is unused and should be
// removed. Element 1 is used by the loop body, and element 2 is used by the
// loop condition; these two should stay.
TEST_F(WhileLoopSimplifierTest, RemoveUnusedLoopOperands) {
  const string hlo_string = R"(
  HloModule RemoveUnusedOperands
  RemoveUnusedOperands.body {
    loop_var = (s32[], s32[], s32[]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element((s32[], s32[],
      s32[]) loop_var), index=0
    get-tuple-element.2 = s32[] get-tuple-element((s32[], s32[],
      s32[]) loop_var), index=1
    constant.1 = s32[] constant(1)
    add = s32[] add(s32[] get-tuple-element.2, s32[] constant.1)
    get-tuple-element.3 = s32[] get-tuple-element((s32[], s32[], s32[])
      loop_var), index=2
    ROOT tuple = (s32[], s32[], s32[]) tuple(s32[] get-tuple-element.1,
      s32[] add, s32[] get-tuple-element.3)
  }
  RemoveUnusedOperands.loop_condition {
    constant.2 = s32[] constant(0)
    param0 = (s32[], s32[], s32[]) parameter(0)
    get-tuple-element = s32[] get-tuple-element((s32[], s32[], s32[]) param0),
      index=2
    ROOT equal-to = pred[] equal-to(s32[] constant.2, s32[] get-tuple-element)
  }
  ENTRY RemoveUnusedOperands {
    x = s32[] parameter(0)
    constant.3 = s32[] constant(0)
    y = s32[] parameter(1)
    tuple.1 = (s32[], s32[], s32[]) tuple(s32[] x, s32[] constant.3,
      s32[] y)
    ROOT while = (s32[], s32[], s32[]) while((s32[], s32[], s32[]) tuple.1),
      condition=RemoveUnusedOperands.loop_condition,
      body=RemoveUnusedOperands.body
  }
  )";

  ParseAndVerifyModule(hlo_string);
  HloModule* the_module = &module();
  EXPECT_TRUE(WhileLoopSimplifier().Run(the_module).ValueOrDie());

  // The original while instruction is still left in the module as a dead
  // instruction, find a while instruction with a different name as the new
  // while instruction.
  HloInstruction* new_while_op =
      *std::find_if(the_module->entry_computation()->instructions().begin(),
                    the_module->entry_computation()->instructions().end(),
                    [&](const HloInstruction* instr) {
                      return (instr->opcode() == HloOpcode::kWhile &&
                              instr->name() != "while");
                    });

  auto scalar_s32 = ShapeUtil::MakeShape(S32, {});
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

TEST_F(WhileLoopSimplifierTest, LoopWithNonTupleBodyShapeNotSimplified) {
  const string hlo_string = R"(
  HloModule BodyHasNonTupleRoot
  BodyHasNonTupleRoot.passthrough {
    ROOT param = (s32[], s32[]) parameter(0)
  }
  BodyHasNonTupleRoot.always_true {
    param.1 = (s32[], s32[]) parameter(0)
    ROOT constant = pred[] constant(true)
  }
  ENTRY BodyHasNonTupleRoot {
    init_value = (s32[], s32[]) parameter(0)
    ROOT while = (s32[], s32[]) while((s32[], s32[]) init_value),
      condition=BodyHasNonTupleRoot.always_true,
      body=BodyHasNonTupleRoot.passthrough
  }
  )";

  ParseAndVerifyModule(hlo_string);
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

TEST_F(WhileLoopSimplifierTest,
       LoopWithNonTupleBodyRootInstructionNotSimplified) {
  const string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT custom-call = (s32[], s32[3]{0}) custom-call(add, multiply),
      custom_call_target="x"
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(44)
    ROOT less-than = pred[] less-than(get-tuple-element.3, constant.2)
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";

  ParseAndVerifyModule(hlo_string);
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

TEST_F(WhileLoopSimplifierTest, LoopWithArrayConstantNotSimplified) {
  const string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[3]{0}, s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[3]{0} get-tuple-element(loop_var.1), index=2
    add.2 = s32[3]{0} add(get-tuple-element.2, get-tuple-element.3)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, add.2, get-tuple-element.3)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}, s32[3]{0}) parameter(0)
    get-tuple-element.4 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(47)
    ROOT less-than = pred[] less-than(get-tuple-element.4, constant.2)
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(42)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4, constant.4)
    ROOT while = (s32[], s32[3]{0}, s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";

  ParseAndVerifyModule(hlo_string);
  EXPECT_FALSE(WhileLoopSimplifier().Run(&module()).ValueOrDie());
}

}  // namespace
}  // namespace xla

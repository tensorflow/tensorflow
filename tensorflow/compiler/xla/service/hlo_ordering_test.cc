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

#include "tensorflow/compiler/xla/service/hlo_ordering.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_scheduling.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

class HloOrderingTest : public HloTestBase {};

TEST_F(HloOrderingTest, LastUseScheduledFirst) {
  // Tests scheduling of the following HLO code:
  //
  //   %ab = abs(%param)
  //   %exp = exp(%param)
  //   %add = add(%ab, %exp)
  //   %negate = negate(%exp)
  //   %sub = subtract(%add, %negate)
  //
  // %add should be scheduled before %negate because %add is the last (and only)
  // use of %ab. Scheduling %add first then frees up %ab's buffer.
  const Shape vec = ShapeUtil::MakeShape(xla::F32, {42});
  auto builder = HloComputation::Builder(TestName());
  auto param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, vec, "param"));
  auto ab = builder.AddInstruction(
      HloInstruction::CreateUnary(vec, HloOpcode::kAbs, param));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vec, HloOpcode::kExp, param));

  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(vec, HloOpcode::kAdd, ab, exp));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vec, HloOpcode::kNegate, exp));
  auto sub = builder.AddInstruction(
      HloInstruction::CreateBinary(vec, HloOpcode::kSubtract, add, negate));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(
      SequentialHloOrdering::HloModuleSequence sequence,
      CreateMemoryMinimizingSequence(*module, [](const LogicalBuffer& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape());
      }));
  // Verify that all instructions are in the sequence.
  EXPECT_EQ(module->entry_computation()->instruction_count(),
            sequence.at(module->entry_computation()).size());

  // The first instruction should be the parameter and the last the root "sub".
  EXPECT_EQ(param, sequence.at(module->entry_computation()).front());
  EXPECT_EQ(sub, sequence.at(module->entry_computation()).back());

  SequentialHloOrdering ordering(module.get(), sequence);
  EXPECT_TRUE(ordering.ExecutesBefore(add, negate));
}

TEST_F(HloOrderingTest, InstructionsInDifferentComputations) {
  // Tests the ordering of instructions in different computations using the
  // following HLO code:
  //
  // Entry computation:
  //   %x = Call(A, {})
  //   %y = Call(B, {%x})
  //
  // Computation A:
  //   %a = Call(C, {})
  //
  // Computation B:
  //   %b = Call(C, {})
  //
  // Computation C:
  //   %c = Constant(42.0f)
  //
  // This results in a diamond-shaped callgraph.
  auto module = CreateNewModule();
  const Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});

  auto builder_c = HloComputation::Builder("C");
  HloInstruction* c = builder_c.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  HloComputation* computation_c =
      module->AddEmbeddedComputation(builder_c.Build());

  auto builder_b = HloComputation::Builder("B");
  builder_b.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param"));
  HloInstruction* b = builder_b.AddInstruction(
      HloInstruction::CreateCall(scalar_shape, {}, computation_c));
  HloComputation* computation_b =
      module->AddEmbeddedComputation(builder_b.Build());

  auto builder_a = HloComputation::Builder("A");
  HloInstruction* a = builder_a.AddInstruction(
      HloInstruction::CreateCall(scalar_shape, {}, computation_c));
  HloComputation* computation_a =
      module->AddEmbeddedComputation(builder_a.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* x = builder.AddInstruction(
      HloInstruction::CreateCall(scalar_shape, {}, computation_a));
  HloInstruction* y = builder.AddInstruction(
      HloInstruction::CreateCall(scalar_shape, {x}, computation_b));
  module->AddEntryComputation(builder.Build());

  DependencyHloOrdering ordering(module.get());
  EXPECT_TRUE(ordering.ExecutesBefore(x, y));
  EXPECT_FALSE(ordering.ExecutesBefore(y, x));

  EXPECT_TRUE(ordering.ExecutesBefore(a, b));
  EXPECT_FALSE(ordering.ExecutesBefore(b, a));

  EXPECT_FALSE(ordering.ExecutesBefore(a, x));
  EXPECT_TRUE(ordering.ExecutesBefore(a, y));
  EXPECT_FALSE(ordering.ExecutesBefore(x, a));
  EXPECT_FALSE(ordering.ExecutesBefore(y, a));

  EXPECT_FALSE(ordering.ExecutesBefore(b, x));
  EXPECT_FALSE(ordering.ExecutesBefore(b, y));
  EXPECT_TRUE(ordering.ExecutesBefore(x, b));
  EXPECT_FALSE(ordering.ExecutesBefore(y, b));

  // Instruction 'c' is called from multiple callsites and should be unordered
  // relative to all other instructions in the module.
  EXPECT_FALSE(ordering.ExecutesBefore(c, a));
  EXPECT_FALSE(ordering.ExecutesBefore(c, b));
  EXPECT_FALSE(ordering.ExecutesBefore(c, x));
  EXPECT_FALSE(ordering.ExecutesBefore(c, y));
  EXPECT_FALSE(ordering.ExecutesBefore(a, c));
  EXPECT_FALSE(ordering.ExecutesBefore(b, c));
  EXPECT_FALSE(ordering.ExecutesBefore(x, c));
  EXPECT_FALSE(ordering.ExecutesBefore(y, c));
}

TEST_F(HloOrderingTest, InstructionsInWhileComputations) {
  // Tests the ordering of instructions in the body and condition of a while
  // instruction. HLO code:
  //
  // body(F32[]) %param):
  //   %negate = Negate(%param)
  //
  // condition(F32[] %param):
  //   %convert = Convert<PRED>(%param)
  //
  // entry:
  //   %constant = Constant(1.0)
  //   return While(%constant, body, condition)
  //
  auto module = CreateNewModule();
  const Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});

  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "body_param"));
  auto negate = body_builder.AddInstruction(HloInstruction::CreateUnary(
      scalar_shape, HloOpcode::kNegate, body_param));
  HloComputation* body = module->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  auto cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "cond_param"));
  auto convert = cond_builder.AddInstruction(HloInstruction::CreateConvert(
      ShapeUtil::MakeShape(xla::PRED, {}), cond_param));
  HloComputation* condition =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(scalar_shape, condition, body, constant));
  module->AddEntryComputation(builder.Build());

  DependencyHloOrdering ordering(module.get());
  EXPECT_TRUE(ordering.ExecutesBefore(constant, xla_while));
  EXPECT_TRUE(ordering.ExecutesBefore(constant, cond_param));
  EXPECT_TRUE(ordering.ExecutesBefore(constant, convert));
  EXPECT_TRUE(ordering.ExecutesBefore(constant, body_param));
  EXPECT_TRUE(ordering.ExecutesBefore(constant, negate));

  // The while should be unordered relative to the body and condition
  // instructions.
  EXPECT_FALSE(ordering.ExecutesBefore(xla_while, body_param));
  EXPECT_FALSE(ordering.ExecutesBefore(xla_while, cond_param));
  EXPECT_FALSE(ordering.ExecutesBefore(body_param, xla_while));
  EXPECT_FALSE(ordering.ExecutesBefore(cond_param, xla_while));

  // Condition instructions should be ordered before body instructions.
  EXPECT_TRUE(ordering.ExecutesBefore(cond_param, body_param));
  EXPECT_TRUE(ordering.ExecutesBefore(convert, body_param));
  EXPECT_TRUE(ordering.ExecutesBefore(cond_param, negate));
  EXPECT_TRUE(ordering.ExecutesBefore(convert, negate));

  EXPECT_FALSE(ordering.ExecutesBefore(body_param, cond_param));
}

TEST_F(HloOrderingTest, ValuesInWhileComputations) {
  // Tests the ordering of values (defined by dataflow analysis) in the body and
  // condition of a while instruction. HLO code:
  //
  // body(F32[]) %param):
  //   %negate = Negate(%param)
  //
  // condition(F32[] %param):
  //   %convert = Convert<PRED>(%param)
  //
  // entry:
  //   %constant = Constant(1.0)
  //   %while = While(%constant, body, condition)
  //   %add = Add(%constant, %while)
  //
  auto module = CreateNewModule();
  const Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});

  auto body_builder = HloComputation::Builder("body");
  auto body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "body_param"));
  auto negate = body_builder.AddInstruction(HloInstruction::CreateUnary(
      scalar_shape, HloOpcode::kNegate, body_param));
  HloComputation* body = module->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  auto cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "cond_param"));
  auto convert = cond_builder.AddInstruction(HloInstruction::CreateConvert(
      ShapeUtil::MakeShape(xla::PRED, {}), cond_param));
  HloComputation* condition =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0)));
  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(scalar_shape, condition, body, constant));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, constant, xla_while));
  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto dataflow,
                          HloDataflowAnalysis::Run(*module, /*ssa_form=*/true));
  DependencyHloOrdering ordering(module.get());

  // Init value is defined before the while, but live range is not before the
  // while because of the use of the init value in the add.
  EXPECT_TRUE(ordering.IsDefinedBefore(dataflow->GetValueDefinedAt(constant),
                                       dataflow->GetValueDefinedAt(xla_while)));
  EXPECT_FALSE(ordering.LiveRangeStrictlyBefore(
      dataflow->GetValueDefinedAt(constant),
      dataflow->GetValueDefinedAt(xla_while), *dataflow));
  EXPECT_TRUE(ordering.MayInterfere(dataflow->GetValueDefinedAt(constant),
                                    dataflow->GetValueDefinedAt(xla_while),
                                    *dataflow));

  // Any value defined in the body or condition is defined before the while, and
  // has a live range strictly before the while.
  EXPECT_TRUE(ordering.IsDefinedBefore(dataflow->GetValueDefinedAt(negate),
                                       dataflow->GetValueDefinedAt(xla_while)));
  EXPECT_TRUE(ordering.LiveRangeStrictlyBefore(
      dataflow->GetValueDefinedAt(negate),
      dataflow->GetValueDefinedAt(xla_while), *dataflow));
  EXPECT_FALSE(ordering.MayInterfere(dataflow->GetValueDefinedAt(negate),
                                     dataflow->GetValueDefinedAt(xla_while),
                                     *dataflow));

  EXPECT_TRUE(ordering.IsDefinedBefore(dataflow->GetValueDefinedAt(convert),
                                       dataflow->GetValueDefinedAt(xla_while)));
  EXPECT_TRUE(ordering.LiveRangeStrictlyBefore(
      dataflow->GetValueDefinedAt(convert),
      dataflow->GetValueDefinedAt(xla_while), *dataflow));
  EXPECT_FALSE(ordering.MayInterfere(dataflow->GetValueDefinedAt(convert),
                                     dataflow->GetValueDefinedAt(xla_while),
                                     *dataflow));

  // The live range of the while should be before the add.
  EXPECT_TRUE(ordering.IsDefinedBefore(dataflow->GetValueDefinedAt(xla_while),
                                       dataflow->GetValueDefinedAt(add)));
  ASSERT_EQ(dataflow->GetValueDefinedAt(xla_while).uses().size(), 1);

  const HloUse& while_use = dataflow->GetValueDefinedAt(xla_while).uses()[0];
  EXPECT_EQ(while_use.instruction, add);
  EXPECT_TRUE(ordering.UseIsBeforeValueDefinition(
      while_use, dataflow->GetValueDefinedAt(add), *dataflow));
  EXPECT_TRUE(ordering.LiveRangeStrictlyBefore(
      dataflow->GetValueDefinedAt(xla_while), dataflow->GetValueDefinedAt(add),
      *dataflow));
}

// Regression test for HloOrdering::ToString() crashing when fed a computation
// containing a fusion node.
TEST_F(HloOrderingTest, ToStringDoesNotCrash) {
  const char* module_str = R"(
HloModule test_module

body.v8 {
  prev.1 = (s32[], f32[3]{0}, f32[3]{0}, f32[3]{0}) parameter(0)
  get-tuple-element.4 = s32[] get-tuple-element(prev.1), index=0
  constant.1 = s32[] constant(1)
  add = s32[] add(get-tuple-element.4, constant.1)
  get-tuple-element.5 = f32[3]{0} get-tuple-element(prev.1), index=3
  get-tuple-element.6 = f32[3]{0} get-tuple-element(prev.1), index=1
  get-tuple-element.7 = f32[3]{0} get-tuple-element(prev.1), index=2
  ROOT tuple = (s32[], f32[3]{0}, f32[3]{0}, f32[3]{0}) tuple(add, get-tuple-element.5, get-tuple-element.6, get-tuple-element.7)
}

condition.v4 {
  constant.2 = s32[] constant(2)
  prev.2 = (s32[], f32[3]{0}, f32[3]{0}, f32[3]{0}) parameter(0)
  get-tuple-element.8 = s32[] get-tuple-element(prev.2), index=0
  ROOT greater-than = pred[] greater-than(constant.2, get-tuple-element.8)
}

fused_computation {
  get-tuple-element.5.param_1 = f32[3]{0} parameter(1)
  get-tuple-element.6.param_2 = f32[3]{0} parameter(2)
  add.4 = f32[3]{0} add(get-tuple-element.5.param_1, get-tuple-element.6.param_2)
  get-tuple-element.7.param_1.1 = f32[3]{0} parameter(0)
  ROOT add.5 = f32[3]{0} add(add.4, get-tuple-element.7.param_1.1)
}

ENTRY while.v11 {
  constant.5 = s32[] constant(0)
  constant.6 = f32[3]{0} constant({1, 1, 1})
  constant.7 = f32[3]{0} constant({2, 2, 2})
  constant.8 = f32[3]{0} constant({3, 3, 3})
  tuple.1 = (s32[], f32[3]{0}, f32[3]{0}, f32[3]{0}) tuple(constant.5, constant.6, constant.7, constant.8)
  while = (s32[], f32[3]{0}, f32[3]{0}, f32[3]{0}) while(tuple.1), condition=condition.v4, body=body.v8
  get-tuple-element.9 = f32[3]{0} get-tuple-element(while), index=3
  get-tuple-element.10 = f32[3]{0} get-tuple-element(while), index=1
  get-tuple-element.11 = f32[3]{0} get-tuple-element(while), index=2
  ROOT fusion = f32[3]{0} fusion(get-tuple-element.9, get-tuple-element.10, get-tuple-element.11), kind=kLoop, calls=fused_computation
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          tools::Parse(module_str));
  DependencyHloOrdering ordering(module.get());
  ordering.ToString();  // Shouldn't crash.
}

}  // namespace
}  // namespace xla

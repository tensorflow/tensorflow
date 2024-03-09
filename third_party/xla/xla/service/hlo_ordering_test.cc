/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/hlo_ordering.h"

#include <memory>
#include <string>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

class HloOrderingTest : public HloTestBase {};

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
  auto module = CreateNewVerifiedModule();
  const Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});

  auto builder_c = HloComputation::Builder("C");
  HloInstruction* c = builder_c.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
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
  auto module = CreateNewVerifiedModule();
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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
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

TEST_F(HloOrderingTest, ParametersDefinedBeforeOthers) {
  // Entry parameter should always be defined before other instruction.
  auto module = CreateNewVerifiedModule();
  const Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param"));
  module->AddEntryComputation(builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(auto dataflow,
                          HloDataflowAnalysis::Run(*module, /*ssa_form=*/true));

  DependencyHloOrdering ordering(module.get());
  EXPECT_TRUE(ordering.IsDefinedBefore(dataflow->GetValueDefinedAt(param),
                                       dataflow->GetValueDefinedAt(constant)));
  EXPECT_TRUE(!ordering.IsDefinedBefore(dataflow->GetValueDefinedAt(constant),
                                        dataflow->GetValueDefinedAt(param)));
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
  auto module = CreateNewVerifiedModule();
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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
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
  // Value defined as init of while interferes with instructions in the
  // condition other than the parameter.
  EXPECT_FALSE(ordering.LiveRangeStrictlyBefore(
      dataflow->GetValueDefinedAt(constant),
      dataflow->GetValueDefinedAt(convert), *dataflow));
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
  EXPECT_TRUE(ordering.MayInterfere(dataflow->GetValueDefinedAt(constant),
                                    dataflow->GetValueDefinedAt(xla_while),
                                    *dataflow));
  EXPECT_TRUE(ordering.IsDefinedBefore(dataflow->GetValueDefinedAt(constant),
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
  ASSERT_EQ(dataflow->GetValueDefinedAt(xla_while).GetUses().size(), 1);

  const HloUse* while_use =
      dataflow->GetValueDefinedAt(xla_while).GetUses().data();
  EXPECT_EQ(while_use->instruction, add);
  EXPECT_TRUE(ordering.UsesBeforeValueDefinition(
      {&while_use, 1}, dataflow->GetValueDefinedAt(add), *dataflow));
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
  ROOT greater-than = pred[] compare(constant.2, get-tuple-element.8), direction=GT
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
                          ParseAndReturnVerifiedModule(module_str));
  DependencyHloOrdering ordering(module.get());
  ordering.ToString();  // Shouldn't crash.
}

TEST_F(HloOrderingTest, ConditionalInstructionOrdering) {
  const char* module_str = R"(
HloModule test_conditional_module

true_branch {
  param.1 = (s32[], s32[]) parameter(0)
  get-tuple-element.1 = s32[] get-tuple-element(param.1), index=0
  get-tuple-element.2 = s32[] get-tuple-element(param.1), index=1
  add.1 = s32[] add(get-tuple-element.1, get-tuple-element.2)
  ROOT tuple.1 = (s32[], s32[]) tuple(add.1, get-tuple-element.1)
}

false_branch {
  param.2 = (s32[], s32[]) parameter(0)
  get-tuple-element.3 = s32[] get-tuple-element(param.2), index=0
  get-tuple-element.4 = s32[] get-tuple-element(param.2), index=1
  add.2 = s32[] add(get-tuple-element.3, get-tuple-element.4)
  ROOT tuple.2 = (s32[], s32[]) tuple(add.2, get-tuple-element.4)
}

ENTRY root {
  param.3 = (pred[], (s32[], s32[])) parameter(0)
  pred.1 = pred[] get-tuple-element(param.3), index=0
  cond_arg.1 = (s32[], s32[]) get-tuple-element(param.3), index=1
  conditional = (s32[], s32[]) conditional(pred.1, cond_arg.1, cond_arg.1), true_computation=true_branch, false_computation=false_branch
  cond_res.1 = s32[] get-tuple-element(conditional), index=0
  cond_res.2 = s32[] get-tuple-element(conditional), index=1
  add.3 = s32[] add(cond_res.1, cond_res.2)
  ROOT result = (s32[], s32[], s32[]) tuple(add.3, cond_res.1, cond_res.2)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(auto dataflow,
                          HloDataflowAnalysis::Run(*module, /*ssa_form=*/true));
  DependencyHloOrdering ordering(module.get());

  // Even though the true and false branches has no ordering, since they do not
  // interfere (as they are mutually exclusive), we define the true computation
  // to be before the false one.
  // Similarly, any instruction in the true or false branches are considered
  // before the conditional instruction. The roots are effectively "at the same
  // time" WRT the conditional, but they are Phi-ed anyway.
  HloInstruction* add_1 = FindInstruction(module.get(), "add.1");
  HloInstruction* add_2 = FindInstruction(module.get(), "add.2");
  HloInstruction* add_3 = FindInstruction(module.get(), "add.3");
  HloInstruction* conditional = FindInstruction(module.get(), "conditional");
  EXPECT_TRUE(ordering.IsDefinedBefore(dataflow->GetValueDefinedAt(add_1),
                                       dataflow->GetValueDefinedAt(add_2)));
  EXPECT_TRUE(
      ordering.IsDefinedBefore(dataflow->GetValueDefinedAt(add_2),
                               dataflow->GetValueDefinedAt(conditional)));
  EXPECT_TRUE(
      ordering.IsDefinedBefore(dataflow->GetValueDefinedAt(add_1),
                               dataflow->GetValueDefinedAt(conditional)));
  EXPECT_TRUE(ordering.IsDefinedBefore(dataflow->GetValueDefinedAt(add_1),
                                       dataflow->GetValueDefinedAt(add_3)));
  EXPECT_TRUE(ordering.IsDefinedBefore(dataflow->GetValueDefinedAt(add_2),
                                       dataflow->GetValueDefinedAt(add_3)));
}

TEST_F(HloOrderingTest,
       ValuesLiveOutOfModuleInterfereWithInstructionsAfterRoot) {
  // Tests that values live out of the module should interfere with values
  // defined after the root instruction. That is:
  //
  //   %param = param(0)
  //   ROOT %root = negate(%param)
  //   %dead = Constant(123.0)
  //
  // %root should interfere with %dead.
  auto module = CreateNewVerifiedModule();
  const Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param"));
  HloInstruction* root = builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape, HloOpcode::kNegate, param));
  HloInstruction* dead = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(123.0f)));
  HloComputation* entry =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/root));

  HloSchedule schedule(module.get());
  schedule.set_sequence(entry, {param, root, dead});
  TF_ASSERT_OK(schedule.Verify());
  SequentialHloOrdering ordering(schedule);

  TF_ASSERT_OK_AND_ASSIGN(auto dataflow,
                          HloDataflowAnalysis::Run(*module, /*ssa_form=*/true));

  EXPECT_FALSE(ordering.ExecutesBefore(root, dead));
  EXPECT_FALSE(ordering.ExecutesBefore(dead, root));

  EXPECT_FALSE(ordering.LiveRangeStrictlyBefore(
      dataflow->GetValueDefinedAt(root), dataflow->GetValueDefinedAt(dead),
      *dataflow));

  EXPECT_TRUE(ordering.MayInterfere(dataflow->GetValueDefinedAt(root),
                                    dataflow->GetValueDefinedAt(dead),
                                    *dataflow));
}

TEST_F(HloOrderingTest,
       ValuesLiveOutOfComputationInterfereWithInstructionsAfterRoot) {
  // Tests that values live out of a computation should interfere with values
  // defined after the root instruction of the computation. That is:
  //
  // subcomputation:
  //   %param = param(0)
  //   ROOT %root = negate(%param)
  //   %dead = Constant(123.0)
  //
  // entry computation:
  //   %c = constant(42.0)
  //   ROOT %call = call({%c}), subcomputation
  //
  // %root should interfere with %dead.
  auto module = CreateNewVerifiedModule();
  const Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});

  auto subbuilder = HloComputation::Builder(TestName() + ".sub");
  HloInstruction* param = subbuilder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param"));
  HloInstruction* root = subbuilder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape, HloOpcode::kNegate, param));
  HloInstruction* dead = subbuilder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(123.0f)));
  HloComputation* subcomputation = module->AddEmbeddedComputation(
      subbuilder.Build(/*root_instruction=*/root));

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  HloInstruction* call = builder.AddInstruction(
      HloInstruction::CreateCall(scalar_shape, {c}, subcomputation));
  HloComputation* entry = module->AddEntryComputation(builder.Build());

  HloSchedule schedule(module.get());
  schedule.set_sequence(subcomputation, {param, root, dead});
  schedule.set_sequence(entry, {c, call});
  TF_ASSERT_OK(schedule.Verify());
  SequentialHloOrdering ordering(schedule);

  TF_ASSERT_OK_AND_ASSIGN(auto dataflow,
                          HloDataflowAnalysis::Run(*module, /*ssa_form=*/true));

  EXPECT_FALSE(ordering.ExecutesBefore(root, dead));
  EXPECT_FALSE(ordering.ExecutesBefore(dead, root));

  EXPECT_FALSE(ordering.LiveRangeStrictlyBefore(
      dataflow->GetValueDefinedAt(root), dataflow->GetValueDefinedAt(dead),
      *dataflow));

  EXPECT_TRUE(ordering.MayInterfere(dataflow->GetValueDefinedAt(root),
                                    dataflow->GetValueDefinedAt(dead),
                                    *dataflow));
}

TEST_F(HloOrderingTest, InterferenceWithOuterRoot) {
  absl::string_view hlo_string = R"(
HloModule InterferenceWithOuterRoot, is_scheduled=true

Embedded (embedded_param: f32[4096,4096]) -> f32[4096,4096] {
  embedded_param = f32[4096,4096]{1,0} parameter(0)
  multiply = f32[4096,4096]{1,0} multiply(embedded_param, embedded_param)
  ROOT log = f32[4096,4096]{1,0} log(multiply)
}

ENTRY InterferenceWithOuterRoot {
  param = f32[4096,4096]{1,0} parameter(0)
  ROOT add = f32[4096,4096]{1,0} add(param, param)
  call = f32[4096,4096]{1,0} call(param), to_apply=Embedded
}

)";
  HloModuleConfig hlo_config;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, hlo_config));
  TF_ASSERT_OK_AND_ASSIGN(auto dataflow,
                          HloDataflowAnalysis::Run(*module, /*ssa_form=*/true));
  DependencyHloOrdering ordering(module.get());
  auto multiply = FindInstruction(module.get(), "multiply");
  auto add = FindInstruction(module.get(), "add");

  EXPECT_TRUE(ordering.MayInterfere(dataflow->GetValueDefinedAt(multiply),
                                    dataflow->GetValueDefinedAt(add),
                                    *dataflow));
}

TEST_F(HloOrderingTest, RootNotLastInstruction) {
  // This is a test for b/189219227. When the root instruction is scheduled not
  // as the last instruction, it still lives out. If the only use of a value is
  // this early root, we want HloOrdering to tell us that it actually doesn't
  // execute before the operations that come after the root.
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

body2 {
  p_body2 = (f32[2]{0}) parameter(0)
  p_body2.1 = f32[2]{0} get-tuple-element(p_body2), index=0
  add.3 = f32[2]{0} add(p_body2.1, p_body2.1)
  ROOT root2 = (f32[2]{0}) tuple(add.3)
}

condition2 {
  p_cond2 = (f32[2]{0}) parameter(0)
  ROOT result = pred[] constant(true)
}

body {
  p_body = (f32[2]{0}) parameter(0)
  p_body.1 = f32[2]{0} get-tuple-element(p_body), index=0
  ROOT root = (f32[2]{0}) tuple(p_body.1)
  copy = f32[2]{0} copy(p_body.1)
  tuple = (f32[2]{0}) tuple(copy)
  while.1 = (f32[2]{0}) while(tuple), condition=condition2, body=body2
}

condition {
  p_cond = (f32[2]{0}) parameter(0)
  ROOT result = pred[] constant(true)
}

ENTRY entry {
  const0 = f32[2]{0} constant({1, 2})
  while_init = (f32[2]{0}) tuple(const0)
  ROOT while.0 = (f32[2]{0}) while(while_init), condition=condition, body=body
}
)";
  HloModuleConfig hlo_config;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, hlo_config));
  TF_ASSERT_OK_AND_ASSIGN(auto dataflow,
                          HloDataflowAnalysis::Run(*module, /*ssa_form=*/true));
  SequentialHloOrdering ordering(module->schedule());
  auto root = FindInstruction(module.get(), "root");
  auto p_body_2 = FindInstruction(module.get(), "p_body2");

  auto tuple_use = HloUse{root, 0};
  const HloValue& value = dataflow->GetUniqueValueAt(p_body_2, {0});
  EXPECT_FALSE(
      ordering.UsesBeforeValueDefinition({&tuple_use}, value, *dataflow));
}

TEST_F(HloOrderingTest, AsyncCallUses) {
  absl::string_view hlo_string = R"(
HloModule single_sc_async_call

%called_computation {
  %out_param = s32[1024]{0} parameter(1)
  %input = s32[1024]{0} parameter(0)
  %size = s32[] constant(256)
  %index = s32[] custom-call(), custom_call_target="Baz"
  %start = s32[] multiply(s32[] %size, s32[] %index)
  %input2 = s32[256]{0} dynamic-slice(s32[1024]{0} %input, s32[] %start), dynamic_slice_sizes={256}
  %output = s32[256]{0} add(s32[256]{0} %input2, s32[256]{0} %input2)
  ROOT %output2 = s32[1024]{0} dynamic-update-slice(s32[1024]{0} %out_param, s32[256]{0} %output, s32[] %start)
}, execution_thread="foobar"

%async_wrapped {
  %async_param = s32[1024]{0} parameter(0)
  %async_param.1 = s32[1024]{0} parameter(1)
  ROOT %call = s32[1024]{0} call(s32[1024]{0} %async_param, s32[1024]{0} %async_param.1), to_apply=%called_computation
}, execution_thread="foobar"

ENTRY %main {
  %input.1 = s32[1024]{0} parameter(0)
  %buf = s32[1024]{0} custom-call(), custom_call_target="AllocateBuffer"
  %async-start = ((s32[1024]{0}, s32[1024]{0}), s32[1024]{0}, u32[]) async-start(s32[1024]{0} %input.1, s32[1024]{0} %buf), async_execution_thread="foobar", calls=%async_wrapped
  ROOT %async-done = s32[1024]{0} async-done(((s32[1024]{0}, s32[1024]{0}), s32[1024]{0}, u32[]) %async-start), async_execution_thread="foobar", calls=%async_wrapped
}
)";
  HloModuleConfig hlo_config;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, hlo_config));
  TF_ASSERT_OK_AND_ASSIGN(auto dataflow,
                          HloDataflowAnalysis::Run(*module, /*ssa_form=*/true));
  DependencyHloOrdering ordering(module.get());
  auto async_start = FindInstruction(module.get(), "async-start");
  auto async_done = FindInstruction(module.get(), "async-done");
  auto call = FindInstruction(module.get(), "call");
  auto output2 = FindInstruction(module.get(), "output2");

  auto async_start_use = HloUse{async_start, 1};
  auto async_done_use = HloUse{async_done, 0, {0, 1}};
  auto call_use = HloUse{call, 1};
  const HloValue& value = dataflow->GetUniqueValueAt(output2, {});
  EXPECT_TRUE(ordering.UsesBeforeValueDefinition(
      {&async_start_use, &call_use, &async_done_use}, value, *dataflow));
}

TEST_F(HloOrderingTest, OrderingBetweenAsyncOpAndItsWrapped) {
  constexpr absl::string_view hlo = R"(
HloModule test

%async_computation {
  %param_0 = f32[10,32,512]{2,1,0:T(8,128)S(5)} parameter(0)
  %param_1 = f32[1,32,512]{2,1,0:T(8,128)} parameter(1)
  %param_2 = s32[]{:T(128)} parameter(2)
  %param_3 = s32[]{:T(128)} parameter(3)
  %param_4 = s32[]{:T(128)} parameter(4)
  ROOT %dynamic-update-slice.1 = f32[10,32,512]{2,1,0:T(8,128)S(5)}
    dynamic-update-slice(%param_0, %param_1, %param_2, %param_3, %param_4)
}

ENTRY %main {
  %param.1 = (s32[]{:T(128)}, f32[32,512]{1,0:T(8,128)},
              f32[10,32,512]{2,1,0:T(8,128)S(5)}) parameter(0)
  %get-tuple-element.132 = f32[10,32,512]{2,1,0:T(8,128)S(5)} get-tuple-element(
    %param.1), index=2
  %get-tuple-element.131 = f32[32,512]{1,0:T(8,128)} get-tuple-element(
    %param.1), index=1
  %cosine.0 = f32[32,512]{1,0:T(8,128)} cosine(%get-tuple-element.131)
  %reshape.6 = f32[1,32,512]{2,1,0:T(8,128)} reshape(%cosine.0)
  %get-tuple-element.130 = s32[]{:T(128)} get-tuple-element(%param.1), index=0
  %constant.49 = s32[]{:T(128)} constant(0)
  %compare.13 = pred[]{:T(512)} compare(
      %get-tuple-element.130, %constant.49), direction=LT
  %constant.50 = s32[]{:T(128)} constant(10)
  %add.22 = s32[]{:T(128)} add(%get-tuple-element.130, %constant.50)
  %select.6 = s32[]{:T(128)} select(
      %compare.13, %add.22, %get-tuple-element.130)
  %dynamic-update-slice-start = (
    (f32[10,32,512]{2,1,0:T(8,128)S(5)}, f32[1,32,512]{2,1,0:T(8,128)},
     s32[]{:T(128)}, s32[]{:T(128)}, s32[]{:T(128)}),
     f32[10,32,512]{2,1,0:T(8,128)S(5)}, u32[]) async-start(
      %get-tuple-element.132, %reshape.6, %select.6,
      %constant.49, %constant.49), calls=%async_computation
  ROOT %dynamic-update-slice-done = f32[10,32,512]{2,1,0:T(8,128)S(5)}
    async-done(%dynamic-update-slice-start), calls=%async_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  TF_ASSERT_OK_AND_ASSIGN(auto dataflow,
                          HloDataflowAnalysis::Run(*module, /*ssa_form=*/true));
  DependencyHloOrdering ordering(module.get());
  auto* async_start =
      FindInstruction(module.get(), "dynamic-update-slice-start");
  auto* async_done = FindInstruction(module.get(), "dynamic-update-slice-done");
  auto* dus = FindInstruction(module.get(), "dynamic-update-slice.1");
  EXPECT_EQ(ordering.GetExecutionConstraint(async_start, dus),
            HloOrdering::ExecutionConstraint::kIsSame);
  EXPECT_EQ(ordering.GetExecutionConstraint(async_done, dus),
            HloOrdering::ExecutionConstraint::kIsSame);
}
}  // namespace
}  // namespace xla

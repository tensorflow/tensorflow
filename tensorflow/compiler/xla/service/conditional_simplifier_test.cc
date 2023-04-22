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

#include "tensorflow/compiler/xla/service/conditional_simplifier.h"

#include <string>
#include <utility>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class ConditionalSimplifierTest : public HloTestBase {
 public:
  // Makes a computation that contains a conditional with constant predicate.
  HloComputation* MakeConditional(HloModule* module, bool is_constant = true);
};

HloComputation* ConditionalSimplifierTest::MakeConditional(HloModule* module,
                                                           bool is_constant) {
  HloComputation::Builder builder(TestName());

  // true_computation returns param+1.
  HloComputation* true_computation;
  {
    HloComputation::Builder true_computation_builder(TestName() +
                                                     ".true_computation");
    auto param =
        true_computation_builder.AddInstruction(HloInstruction::CreateParameter(
            0, ShapeUtil::MakeShape(S32, {}), "param"));
    auto one = true_computation_builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));

    true_computation_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(S32, {}), HloOpcode::kAdd, param, one));

    true_computation =
        module->AddEmbeddedComputation(true_computation_builder.Build());
  }

  // false_computation returns param+42.
  HloComputation* false_computation;
  {
    HloComputation::Builder false_computation_builder(TestName() +
                                                      ".false_computation");
    auto param = false_computation_builder.AddInstruction(
        HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(S32, {}),
                                        "param"));
    auto forty_two = false_computation_builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(42)));

    false_computation_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(S32, {}), HloOpcode::kAdd, param, forty_two));
    false_computation =
        module->AddEmbeddedComputation(false_computation_builder.Build());
  }

  auto false_instrn = builder.AddInstruction(
      is_constant
          ? HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false))
          : HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(PRED, {}),
                                            "cond"));
  auto false_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(S32, {}), "false_param"));
  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));

  builder.AddInstruction(HloInstruction::CreateConditional(
      ShapeUtil::MakeShape(S32, {}), false_instrn, one, true_computation,
      false_param, false_computation));

  return module->AddEntryComputation(builder.Build());
}

TEST_F(ConditionalSimplifierTest, ConditionalGetsInlined) {
  auto m = CreateNewVerifiedModule();
  HloComputation* computation = MakeConditional(m.get());
  ASSERT_TRUE(ConditionalSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Parameter(), op::Constant()));
}

TEST_F(ConditionalSimplifierTest, BranchGetsInlined) {
  auto m = CreateNewVerifiedModule();
  HloComputation* computation = MakeConditional(m.get(), /*is_constant=*/false);
  ASSERT_TRUE(ConditionalSimplifier().Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      computation->root_instruction(),
      op::Select(op::Parameter(1), op::Add(op::Constant(), op::Constant()),
                 op::Add(op::Parameter(0), op::Constant())));
}

TEST_F(ConditionalSimplifierTest, ConditionalWithControlDependency) {
  auto m = CreateNewVerifiedModule();
  HloComputation* computation = MakeConditional(m.get());

  auto* true_op = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  TF_ASSERT_OK(
      true_op->AddControlDependencyTo(computation->root_instruction()));

  EXPECT_FALSE(ConditionalSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(ConditionalSimplifierTest, NotRemovedIfContainsSend) {
  auto m = CreateNewVerifiedModule();
  HloComputation* computation = MakeConditional(m.get());
  auto* conditional = computation->root_instruction();
  ASSERT_EQ(conditional->opcode(), HloOpcode::kConditional);

  auto* true_computation = conditional->true_computation();
  auto* token = true_computation->AddInstruction(HloInstruction::CreateToken());
  auto* send = true_computation->AddInstruction(HloInstruction::CreateSend(
      true_computation->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true))),
      token, /*channel_id=*/0));
  true_computation->AddInstruction(HloInstruction::CreateSendDone(send));
  EXPECT_FALSE(ConditionalSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(ConditionalSimplifierTest, NotRemovedIfContainsRecv) {
  auto m = CreateNewVerifiedModule();
  HloComputation* computation = MakeConditional(m.get());
  auto* conditional = computation->root_instruction();
  ASSERT_EQ(conditional->opcode(), HloOpcode::kConditional);

  auto* true_computation = conditional->true_computation();
  auto* token = true_computation->AddInstruction(HloInstruction::CreateToken());
  auto* recv = true_computation->AddInstruction(HloInstruction::CreateRecv(
      ShapeUtil::MakeShape(F32, {1}), token, /*channel_id=*/0));
  true_computation->AddInstruction(HloInstruction::CreateRecvDone(recv));
  EXPECT_FALSE(ConditionalSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(ConditionalSimplifierTest, NotRemovedIfContainsNonRemovableInstruction) {
  auto m = CreateNewVerifiedModule();
  HloComputation* computation = MakeConditional(m.get());
  auto* conditional = computation->root_instruction();
  ASSERT_EQ(conditional->opcode(), HloOpcode::kConditional);
  auto* false_computation = conditional->false_computation();
  auto token = false_computation->AddInstruction(HloInstruction::CreateToken());
  false_computation->AddInstruction(HloInstruction::CreateInfeed(
      ShapeUtil::MakeShape(F32, {1}), token, "config"));
  EXPECT_FALSE(ConditionalSimplifier().Run(m.get()).ValueOrDie());
}

TEST_F(ConditionalSimplifierTest, TrivalOperandsRemoved) {
  absl::string_view hlo_string =
      R"(
HloModule UnusedTupleOperands
on_false {
  t = (f32[20,40], f32[40,40], f32[20,40], f32[40,40]) parameter(0)
  lhs = f32[20,40] get-tuple-element(t), index=0
  rhs = f32[40,40] get-tuple-element(t), index=1
  dot = f32[20,40] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT result = (f32[20,40]) tuple(dot)
}

on_true {
  t = (f32[20,40], f32[40,40], f32[20,40], f32[40,40]) parameter(0)
  lhs = f32[20,40] get-tuple-element(t), index=2
  rhs = f32[40,40] get-tuple-element(t), index=3
  dot = f32[20,40] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT result = (f32[20,40]) tuple(dot)
}

ENTRY main {
  c0_0 = f32[20,40] parameter(0)
  c0_1 = f32[40,40] parameter(1)
  c1_0 = f32[20,40] parameter(2)
  c1_1 = f32[40,40] parameter(3)
  p = pred[] parameter(4)
  t = (f32[20,40], f32[40,40], f32[20,40], f32[40,40]) tuple(c0_0, c0_1, c1_0, c1_1)
  call = (f32[20,40]) call(t), to_apply=on_true
  ROOT result = (f32[20,40]) conditional(p,t,t), false_computation=on_false, true_computation=on_true
}
)";
  auto status = ParseAndReturnVerifiedModule(hlo_string);
  TF_ASSERT_OK(status.status());
  std::unique_ptr<HloModule> module = status.ConsumeValueOrDie();
  HloVerifier v(/*layout_sensitive=*/false, /*allow_mixed_precision=*/false);
  TF_ASSERT_OK(v.Run(module.get()).status());
  EXPECT_TRUE(ConditionalSimplifier().Run(module.get()).ValueOrDie());
  TF_ASSERT_OK(v.Run(module.get()).status());
  HloInstruction* conditional = module->entry_computation()->root_instruction();
  EXPECT_TRUE(conditional != nullptr);
  EXPECT_EQ(conditional->operand(1)->shape().tuple_shapes().size(), 2);
  EXPECT_EQ(conditional->operand(2)->shape().tuple_shapes().size(), 2);
  // For the call operation, nothing should have changed.
  HloInstruction* call = FindInstruction(module.get(), "call");
  EXPECT_EQ(
      call->to_apply()->parameter_instruction(0)->shape().tuple_shapes().size(),
      4);
}

TEST_F(ConditionalSimplifierTest,
       TwoConditionalsCreatedInReversedLexicalOrder) {
  absl::string_view hlo_string = R"(
  HloModule DeadConditional
    computation.1 {
      param.1 = s64[] parameter(0)
      constant.1 = s64[] constant(1)
      ROOT add.1 = s64[] add(param.1, constant.1)
    }

    computation.2 {
      param.2 = s64[] parameter(0)
      constant.2 = s64[] constant(2)
      ROOT add.2 = s64[] add(param.2, constant.2)
   }

    computation.3 {
      param.3 = s64[] parameter(0)
      constant.3 = s64[] constant(3)
      ROOT add.3 = s64[] add(param.3, constant.3)
    }

    computation.4 {
      param.4 = s64[] parameter(0)
      constant.4 = s64[] constant(4)
      ROOT add.4 = s64[] add(param.4, constant.4)
    }

    ENTRY KernelEntry {
      param.1 = s64[] parameter(0)
      param.2 = s64[] parameter(1)
      param.3 = s64[] parameter(2)
      param.4 = pred[] parameter(3)

      conditional_1 = s64[] conditional(param.4, param.3, param.2),
        true_computation=computation.3, false_computation=computation.4
      constant.1 = pred[] constant(false)
      ROOT conditional_2 = s64[] conditional(constant.1, conditional_1,
        param.1), true_computation=computation.1,
        false_computation=computation.2
    })";
  auto status = ParseAndReturnVerifiedModule(hlo_string);
  TF_ASSERT_OK(status.status());
  std::unique_ptr<HloModule> module = status.ConsumeValueOrDie();
  HloVerifier v(/*layout_sensitive=*/false, /*allow_mixed_precision=*/false);
  TF_ASSERT_OK(v.Run(module.get()).status());

  // Replace conditional_1 with a clone that is created after conditional_2.
  HloInstruction* conditional_1 =
      FindInstruction(module.get(), "conditional_1");
  HloInstruction* conditional_1_clone =
      conditional_1->parent()->AddInstruction(conditional_1->Clone());
  TF_ASSERT_OK(conditional_1->ReplaceAllUsesWith(conditional_1_clone));
  TF_ASSERT_OK(conditional_1->parent()->RemoveInstruction(conditional_1));

  EXPECT_TRUE(ConditionalSimplifier().Run(module.get()).ValueOrDie());
}

TEST_F(ConditionalSimplifierTest, RemoveDeadRoots) {
  absl::string_view hlo_string =
      R"(
HloModule RemoveDeadRoots
on_false {
  t = (f32[20,40], f32[40,40]) parameter(0)
  lhs = f32[20,40] get-tuple-element(t), index=0
  rhs = f32[40,40] get-tuple-element(t), index=1
  dot = f32[20,40] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  after-all = token[] after-all()
  outfeed = token[] outfeed(dot, after-all)
  ROOT result = (f32[20,40]) tuple(dot)
}

on_true {
  t = (f32[20,40], f32[40,40]) parameter(0)
  lhs = f32[20,40] get-tuple-element(t), index=0
  add = f32[20,40] add(lhs, lhs)
  ROOT result = (f32[20,40]) tuple(add)
}

ENTRY main {
  c0_0 = f32[20,40] parameter(0)
  c0_1 = f32[40,40] parameter(1)
  p = pred[] parameter(2)
  t = (f32[20,40], f32[40,40]) tuple(c0_0, c0_1)
  conditional = (f32[20, 40]) conditional(p,t,t), false_computation=on_false, true_computation=on_true
  ROOT result = () tuple()
}
)";
  auto status = ParseAndReturnVerifiedModule(hlo_string);
  TF_ASSERT_OK(status.status());
  HloVerifier v(/*layout_sensitive=*/false, /*allow_mixed_precision=*/false);
  TF_ASSERT_OK(v.Run(status.ValueOrDie().get()).status());
  EXPECT_TRUE(
      ConditionalSimplifier().Run(status.ValueOrDie().get()).ValueOrDie());
  TF_ASSERT_OK(v.Run(status.ValueOrDie().get()).status());
  HloInstruction* conditional =
      FindInstruction(status.ValueOrDie().get(), "conditional");
  // The conditional root should be replaced with an empty tuple.
  EXPECT_EQ(ShapeUtil::TupleElementCount(conditional->shape()), 0);
}

TEST_F(ConditionalSimplifierTest, SecondTupleElementUnusedAndRemoved) {
  absl::string_view hlo_string =
      R"(
HloModule SecondTupleElementUnusedAndRemoved

on_true {
  arg_tuple.7 = (f32[10,10]{1,0}) parameter(0)
  get-tuple-element.9 = f32[10,10]{1,0} get-tuple-element(arg_tuple.7), index=0
  copy = f32[10,10]{1,0} copy(get-tuple-element.9)
  ROOT tuple.6 = (f32[10,10]{1,0}, f32[10,10]{1,0}) tuple(copy, get-tuple-element.9)
}

on_false {
  constant.17 = f32[] constant(0)
  constant.18 = f32[] constant(1)
  rng.19 = f32[10,10]{1,0} rng(constant.17, constant.18), distribution=rng_uniform
  arg_tuple.14 = (f32[10,10]{1,0}) parameter(0)
  get-tuple-element.16 = f32[10,10]{1,0} get-tuple-element(arg_tuple.14), index=0
  ROOT tuple.7 = (f32[10,10]{1,0}, f32[10,10]{1,0}) tuple(rng.19, get-tuple-element.16)
}

ENTRY main {
  constant.38 = pred[] constant(true)
  arg_tuple.30 = (s32[], f32[10,10]{1,0}) parameter(0)
  get-tuple-element.21 = f32[10,10]{1,0} get-tuple-element(arg_tuple.30), index=1
  tuple.1 = (f32[10,10]{1,0}) tuple(get-tuple-element.21)
  conditional = (f32[10,10]{1,0}, f32[10,10]{1,0}) conditional(constant.38, tuple.1, tuple.1), true_computation=on_true, false_computation=on_false
  get-first-index = f32[10,10]{1,0} get-tuple-element(conditional), index=0
  ROOT result = (f32[10,10]{1,0}) tuple(get-first-index)
}
)";
  auto status = ParseAndReturnVerifiedModule(hlo_string);
  TF_ASSERT_OK(status.status());
  HloVerifier v(/*layout_sensitive=*/false, /*allow_mixed_precision=*/false);
  TF_ASSERT_OK(v.Run(status.ValueOrDie().get()).status());
  EXPECT_TRUE(
      ConditionalSimplifier().Run(status.ValueOrDie().get()).ValueOrDie());
  TF_ASSERT_OK(v.Run(status.ValueOrDie().get()).status());
  const HloInstruction* conditional =
      FindInstruction(status.ValueOrDie().get(), "conditional");
  // The second element of "conditional" result tuple (f32[10,10], f32[10,10])
  // should be removed since it is not referenced by any GTE instructions
  // (see "get-first-index" instruction in hlo_string).
  EXPECT_EQ(ShapeUtil::TupleElementCount(conditional->shape()), 1);
}

TEST_F(ConditionalSimplifierTest, FirstTupleElementUnusedAndRemoved) {
  absl::string_view hlo_string =
      R"(
HloModule FirstTupleElementUnusedAndRemoved

on_true {
  arg_tuple.7 = (f32[10,10]{1,0}) parameter(0)
  get-tuple-element.9 = f32[10,10]{1,0} get-tuple-element(arg_tuple.7), index=0
  copy = f32[10,10]{1,0} copy(get-tuple-element.9)
  ROOT tuple.6 = (f32[10,10]{1,0}, f32[10,10]{1,0}) tuple(copy, get-tuple-element.9)
}

on_false {
  constant.17 = f32[] constant(0)
  constant.18 = f32[] constant(1)
  rng.19 = f32[10,10]{1,0} rng(constant.17, constant.18), distribution=rng_uniform
  arg_tuple.14 = (f32[10,10]{1,0}) parameter(0)
  get-tuple-element.16 = f32[10,10]{1,0} get-tuple-element(arg_tuple.14), index=0
  ROOT tuple.7 = (f32[10,10]{1,0}, f32[10,10]{1,0}) tuple(rng.19, get-tuple-element.16)
}

ENTRY main {
  constant.38 = pred[] constant(true)
  arg_tuple.30 = (s32[], f32[10,10]{1,0}) parameter(0)
  get-tuple-element.21 = f32[10,10]{1,0} get-tuple-element(arg_tuple.30), index=1
  tuple.1 = (f32[10,10]{1,0}) tuple(get-tuple-element.21)
  conditional = (f32[10,10]{1,0}, f32[10,10]{1,0}) conditional(constant.38, tuple.1, tuple.1), true_computation=on_true, false_computation=on_false
  get-second-index = f32[10,10]{1,0} get-tuple-element(conditional), index=1
  ROOT result = (f32[10,10]{1,0}) tuple(get-second-index)
}
)";
  auto status = ParseAndReturnVerifiedModule(hlo_string);
  TF_ASSERT_OK(status.status());
  HloVerifier v(/*layout_sensitive=*/false, /*allow_mixed_precision=*/false);
  TF_ASSERT_OK(v.Run(status.ValueOrDie().get()).status());
  EXPECT_TRUE(
      ConditionalSimplifier().Run(status.ValueOrDie().get()).ValueOrDie());
  TF_ASSERT_OK(v.Run(status.ValueOrDie().get()).status());
  const HloInstruction* conditional =
      FindInstruction(status.ValueOrDie().get(), "conditional");
  // The first element of "conditional" result tuple (f32[10,10], f32[10,10])
  // should be removed since it is not referenced by any GTE instructions (see
  // "get-second-index" instruction in hlo_string).
  EXPECT_EQ(ShapeUtil::TupleElementCount(conditional->shape()), 1);
}

// Before:
//       gte          rng
//      /   \        /   \
//      |   |        |   |
//     on_true      on_false
//    (f32, f32)   (f32, f32)
//         |           |
//          \         /
//          conditional
//          (f32, f32)
//
// After:
//       gte          rng
//        |            |
//     on_true      on_false
//      (f32)        (f32)
//         |           |
//          \         /
//          conditional
//             (f32)
//
// The 'rng' in on_false is to add side-effect so that conditional is not being
// simplified and replaced with 'select' instruction by TryRemoveConditional.
TEST_F(ConditionalSimplifierTest, MergeDuplicateTupleElements) {
  absl::string_view hlo_string =
      R"(
HloModule MergeDuplicateTupleElements

on_true {
  param-true = (f32[]) parameter(0)
  gte-true = f32[] get-tuple-element(param-true), index=0
  ROOT tuple-true = (f32[], f32[]) tuple(gte-true, gte-true)
}

on_false {
  param-false = (f32[]) parameter(0)
  constant.0 = f32[] constant(0)
  constant.1 = f32[] constant(1)
  rng = f32[] rng(constant.0, constant.1), distribution=rng_uniform
  ROOT tuple-false = (f32[], f32[]) tuple(rng, rng)
}

ENTRY main {
  comp = pred[] parameter(0)
  arg = (f32[]) parameter(1)
  conditional = (f32[], f32[]) conditional(comp, arg, arg), true_computation=on_true, false_computation=on_false
  gte.0 = f32[] get-tuple-element(conditional), index=0
  gte.1 = f32[] get-tuple-element(conditional), index=1
  ROOT add = f32[] add(gte.0, gte.1)
}
)";
  auto status = ParseAndReturnVerifiedModule(hlo_string);
  TF_ASSERT_OK(status.status());
  HloVerifier v(/*layout_sensitive=*/false, /*allow_mixed_precision=*/false);
  TF_ASSERT_OK(v.Run(status.ValueOrDie().get()).status());
  EXPECT_TRUE(
      ConditionalSimplifier().Run(status.ValueOrDie().get()).ValueOrDie());
  TF_ASSERT_OK(v.Run(status.ValueOrDie().get()).status());
  const HloInstruction* conditional =
      FindInstruction(status.ValueOrDie().get(), "conditional");
  EXPECT_EQ(ShapeUtil::TupleElementCount(conditional->shape()), 1);
  const HloInstruction* gte_0 =
      FindInstruction(status.ValueOrDie().get(), "gte.0");
  const HloInstruction* gte_1 =
      FindInstruction(status.ValueOrDie().get(), "gte.1");
  EXPECT_EQ(gte_0->tuple_index(), 0);
  EXPECT_EQ(gte_1->tuple_index(), 0);
}

// Since select can only be used on arrays, use after-all for token types.
TEST_F(ConditionalSimplifierTest, SimplifyConditionalWithTokens) {
  absl::string_view hlo_string =
      R"(
HloModule SimplifyConditionalWithTokens

true_comp {
  ROOT parameter.13 = (token[]) parameter(0)
}

false_comp {
  ROOT parameter.21 = (token[]) parameter(0)
}

ENTRY entry {
  parameter.29 = pred[] parameter(0)
  token.1 = token[] after-all()
  token.2 = token[] after-all()
  tuple.3 = (token[]) tuple(token.1)
  tuple.4 = (token[]) tuple(token.2)
  ROOT conditional.5 = (token[]) conditional(parameter.29, tuple.3, tuple.4), true_computation=true_comp, false_computation=false_comp
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloVerifier v(/*layout_sensitive=*/false, /*allow_mixed_precision=*/false);
  TF_ASSERT_OK(v.Run(module.get()).status());
  EXPECT_TRUE(ConditionalSimplifier().Run(module.get()).ValueOrDie());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::AfterAll(
                  op::GetTupleElement(op::Tuple(op::AfterAll()), 0),
                  op::GetTupleElement(op::Tuple(op::AfterAll()), 0))));
}

}  // namespace

}  // namespace xla

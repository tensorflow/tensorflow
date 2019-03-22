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
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class ConditionalSimplifierTest : public HloTestBase {
 public:
  // Makes a computation that contains a conditional with constant predicate.
  HloComputation* MakeConditional(HloModule* module);
};

HloComputation* ConditionalSimplifierTest::MakeConditional(HloModule* module) {
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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
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
  ROOT result = (f32[20, 40]) conditional(p,t,t), false_computation=on_false, true_computation=on_true
}
)";
  auto status = ParseHloString(hlo_string);
  TF_ASSERT_OK(status.status());
  HloVerifier v(false, false);
  TF_ASSERT_OK(v.Run(status.ValueOrDie().get()).status());
  EXPECT_TRUE(
      ConditionalSimplifier().Run(status.ValueOrDie().get()).ValueOrDie());
  TF_ASSERT_OK(v.Run(status.ValueOrDie().get()).status());
  EXPECT_EQ(status.ValueOrDie()
                ->entry_computation()
                ->root_instruction()
                ->operand(1)
                ->shape()
                .tuple_shapes()
                .size(),
            2);
  EXPECT_EQ(status.ValueOrDie()
                ->entry_computation()
                ->root_instruction()
                ->operand(2)
                ->shape()
                .tuple_shapes()
                .size(),
            2);
}
}  // namespace

}  // namespace xla

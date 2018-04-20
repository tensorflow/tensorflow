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
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class ConditionalSimplifierTest : public HloVerifiedTestBase {
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
        HloInstruction::CreateConstant(Literal::CreateR0<int32>(1)));

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
        HloInstruction::CreateConstant(Literal::CreateR0<int32>(42)));

    false_computation_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(S32, {}), HloOpcode::kAdd, param, forty_two));
    false_computation =
        module->AddEmbeddedComputation(false_computation_builder.Build());
  }

  auto false_instrn = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(false)));
  auto false_param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(S32, {}), "false_param"));
  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<int32>(1)));

  builder.AddInstruction(HloInstruction::CreateConditional(
      ShapeUtil::MakeShape(S32, {}), false_instrn, one, true_computation,
      false_param, false_computation));

  return module->AddEntryComputation(builder.Build());
}

TEST_F(ConditionalSimplifierTest, ConditionalGetsInlined) {
  HloComputation* computation = MakeConditional(&module());
  ASSERT_TRUE(ConditionalSimplifier().Run(&module()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Parameter(), op::Constant()));
}

TEST_F(ConditionalSimplifierTest, ConditionalWithControlDependency) {
  HloComputation* computation = MakeConditional(&module());

  auto* true_op = computation->AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<bool>(true)));
  TF_ASSERT_OK(
      true_op->AddControlDependencyTo(computation->root_instruction()));

  EXPECT_FALSE(ConditionalSimplifier().Run(&module()).ValueOrDie());
}

TEST_F(ConditionalSimplifierTest, NotRemovedIfContainsSend) {
  HloComputation* computation = MakeConditional(&module());
  auto* conditional = computation->root_instruction();
  ASSERT_EQ(conditional->opcode(), HloOpcode::kConditional);

  auto* true_computation = conditional->true_computation();
  auto* send = true_computation->AddInstruction(HloInstruction::CreateSend(
      true_computation->AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<bool>(true))),
      /*channel_id=*/0));
  true_computation->AddInstruction(HloInstruction::CreateSendDone(send));
  EXPECT_FALSE(ConditionalSimplifier().Run(&module()).ValueOrDie());
}

TEST_F(ConditionalSimplifierTest, NotRemovedIfContainsRecv) {
  HloComputation* computation = MakeConditional(&module());
  auto* conditional = computation->root_instruction();
  ASSERT_EQ(conditional->opcode(), HloOpcode::kConditional);

  auto* true_computation = conditional->true_computation();
  auto* recv = true_computation->AddInstruction(HloInstruction::CreateRecv(
      ShapeUtil::MakeShape(F32, {1}), /*channel_id=*/0));
  true_computation->AddInstruction(HloInstruction::CreateRecvDone(recv));
  EXPECT_FALSE(ConditionalSimplifier().Run(&module()).ValueOrDie());
}

TEST_F(ConditionalSimplifierTest, NotRemovedIfContainsNonRemovableInstruction) {
  HloComputation* computation = MakeConditional(&module());
  auto* conditional = computation->root_instruction();
  ASSERT_EQ(conditional->opcode(), HloOpcode::kConditional);
  auto* false_computation = conditional->false_computation();
  false_computation->AddInstruction(
      HloInstruction::CreateInfeed(ShapeUtil::MakeShape(F32, {1}), "config"));
  EXPECT_FALSE(ConditionalSimplifier().Run(&module()).ValueOrDie());
}

}  // namespace
}  // namespace xla

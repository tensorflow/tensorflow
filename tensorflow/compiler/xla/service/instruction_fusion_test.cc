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

#include "tensorflow/compiler/xla/service/instruction_fusion.h"

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {

using InstructionFusionTest = HloTestBase;

TEST_F(InstructionFusionTest,
       CostlyProducerAndOperandElementReusingConsumerNotFused) {
  HloComputation::Builder builder(TestName());
  HloInstruction* const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(5)));
  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {}), HloOpcode::kExp, const0));
  HloInstruction* broadcast2 =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::MakeShape(S32, {1}), exp1, {0}));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(broadcast2, computation->root_instruction());
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
  EXPECT_EQ(broadcast2, computation->root_instruction());
}

TEST_F(InstructionFusionTest,
       NonCostlyProducerAndOperandElementReusingConsumerFused) {
  HloComputation::Builder builder(TestName());
  HloInstruction* const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(5)));
  HloInstruction* negate1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {}), HloOpcode::kNegate, const0));
  HloInstruction* broadcast2 =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::MakeShape(S32, {1}), negate1, {0}));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(broadcast2, computation->root_instruction());
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
  EXPECT_EQ(HloOpcode::kFusion, computation->root_instruction()->opcode());
}

TEST_F(InstructionFusionTest,
       CostlyProducerAndNonOperandElementReusingConsumerFused_Reshape) {
  HloComputation::Builder builder(TestName());
  HloInstruction* const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(5)));
  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {}), HloOpcode::kExp, const0));
  HloInstruction* reshape2 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {}), exp1));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(reshape2, computation->root_instruction());
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
  EXPECT_EQ(HloOpcode::kFusion, computation->root_instruction()->opcode());
}

TEST_F(InstructionFusionTest,
       CostlyProducerAndNonOperandElementReusingConsumerFused_Transpose) {
  HloComputation::Builder builder(TestName());
  HloInstruction* const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(5)));
  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {}), HloOpcode::kExp, const0));
  HloInstruction* transpose2 = builder.AddInstruction(
      HloInstruction::CreateTranspose(ShapeUtil::MakeShape(S32, {}), exp1, {}));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(transpose2, computation->root_instruction());
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
  EXPECT_EQ(HloOpcode::kFusion, computation->root_instruction()->opcode());
}

TEST_F(InstructionFusionTest, PotentialBitcastReshapeOfParameterUnfused) {
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(S32, {}), "0"));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {1, 1}), param0));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(reshape1, computation->root_instruction());
  EXPECT_FALSE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
}

TEST_F(InstructionFusionTest, PotentialBitcastSimpleReshapeOfParameterUnfused) {
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(S32, {}), "0"));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {1, 1}), param0));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(reshape1, computation->root_instruction());
  EXPECT_FALSE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
}

TEST_F(InstructionFusionTest, PotentialBitcastTransposeOfParameterUnfused) {
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(S32, {}), "0"));
  auto transpose1 = builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(S32, {}), param0, {}));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(transpose1, computation->root_instruction());
  EXPECT_FALSE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
}

}  // namespace xla

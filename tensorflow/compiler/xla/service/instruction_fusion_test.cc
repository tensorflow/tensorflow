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

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {

using InstructionFusionTest = HloTestBase;

TEST_F(InstructionFusionTest, PotentialBitcastReshapeOfParameterUnfused) {
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(S32, {}), "0"));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {1, 1}), param0));

  auto module = CreateNewModule();
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

  auto module = CreateNewModule();
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(transpose1, computation->root_instruction());
  EXPECT_FALSE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
}

TEST_F(InstructionFusionTest, AvoidDuplicationIfNotAllFusable) {
  HloComputation::Builder builder(TestName());
  auto shape = ShapeUtil::MakeShape(F32, {16, 16});
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "1"));
  HloInstruction* binary1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));
  builder.AddInstruction(HloInstruction::CreateSend(binary1, 0));
  HloInstruction* unary = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kAbs, binary1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(unary, computation->root_instruction());
  EXPECT_FALSE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
}

TEST_F(InstructionFusionTest, AllowUnaryDuplication) {
  HloComputation::Builder builder(TestName());
  auto shape = ShapeUtil::MakeShape(F32, {16, 16});
  auto param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "0"));
  HloInstruction* unary1 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kFloor, param0));
  builder.AddInstruction(HloInstruction::CreateSend(unary1, 0));
  HloInstruction* unary2 = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kAbs, unary1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(unary2, computation->root_instruction());
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
}

TEST_F(InstructionFusionTest, AllowEffectiveUnaryDuplication) {
  auto shape = ShapeUtil::MakeShape(F32, {16, 16});
  auto small_shape = ShapeUtil::MakeShape(F32, {16});
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, small_shape, "0"));
  auto param1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "1"));
  HloInstruction* binary1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));
  builder.AddInstruction(HloInstruction::CreateSend(binary1, 0));
  HloInstruction* unary = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kAbs, binary1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(unary, computation->root_instruction());
  EXPECT_TRUE(
      InstructionFusion(InstructionFusion::IsExpensive, /*may_duplicate=*/true)
          .Run(module.get())
          .ValueOrDie());
}

}  // namespace xla

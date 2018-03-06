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

#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace gpu {

using InstructionFusionTest = HloTestBase;

TEST_F(InstructionFusionTest,
       CostlyProducerAndOperandElementReusingConsumerNotFused) {
  HloComputation::Builder builder(TestName());
  HloInstruction* const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0(5)));
  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {}), HloOpcode::kExp, const0));
  HloInstruction* broadcast2 =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::MakeShape(S32, {1}), exp1, {0}));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(broadcast2, computation->root_instruction());
  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie());
  EXPECT_EQ(broadcast2, computation->root_instruction());
}

TEST_F(InstructionFusionTest,
       NonCostlyProducerAndOperandElementReusingConsumerFused) {
  HloComputation::Builder builder(TestName());
  HloInstruction* const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0(5)));
  HloInstruction* negate1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {}), HloOpcode::kNegate, const0));
  HloInstruction* broadcast2 =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::MakeShape(S32, {1}), negate1, {0}));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(broadcast2, computation->root_instruction());
  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), op::Fusion());
}

TEST_F(InstructionFusionTest,
       CostlyProducerAndNonOperandElementReusingConsumerFused_Reshape) {
  HloComputation::Builder builder(TestName());
  HloInstruction* const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0(5)));
  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {}), HloOpcode::kExp, const0));
  HloInstruction* reshape2 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {}), exp1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(reshape2, computation->root_instruction());
  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), op::Fusion());
}

TEST_F(InstructionFusionTest,
       CostlyProducerAndNonOperandElementReusingConsumerFused_Transpose) {
  HloComputation::Builder builder(TestName());
  HloInstruction* const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0(5)));
  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {}), HloOpcode::kExp, const0));
  HloInstruction* transpose2 = builder.AddInstruction(
      HloInstruction::CreateTranspose(ShapeUtil::MakeShape(S32, {}), exp1, {}));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(transpose2, computation->root_instruction());
  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), op::Fusion());
}

TEST_F(InstructionFusionTest, PotentialBitcastReshapeOfDotUnfused) {
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(S32, {1, 1}), "0"));
  auto dot1 = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(S32, {1, 1}), HloOpcode::kDot, param0, param0));
  auto reshape2 = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(S32, {1, 1, 1}), dot1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(reshape2, computation->root_instruction());
  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie());
}

TEST_F(InstructionFusionTest, PotentialBitcastTransposeOfDotUnfused) {
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(S32, {1, 1}), "0"));
  auto dot1 = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(S32, {1, 1}), HloOpcode::kDot, param0, param0));
  auto transpose2 = builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(S32, {1, 1}), dot1, {0, 1}));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(transpose2, computation->root_instruction());
  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie());
}

TEST_F(InstructionFusionTest, GetTupleElementFused) {
  HloComputation::Builder builder(TestName());
  Shape data_shape = ShapeUtil::MakeShape(F32, {8});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({data_shape, data_shape});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  auto gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, param, 0));
  auto gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(data_shape, param, 1));
  builder.AddInstruction(
      HloInstruction::CreateBinary(data_shape, HloOpcode::kAdd, gte0, gte1));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(HloOpcode::kFusion, root->opcode());
  HloInstruction* fused_root = root->fused_expression_root();
  EXPECT_EQ(HloOpcode::kAdd, fused_root->opcode());
  // Check that operands of 'fused_root' are GTE.
  EXPECT_EQ(HloOpcode::kGetTupleElement, fused_root->operand(0)->opcode());
  EXPECT_EQ(HloOpcode::kGetTupleElement, fused_root->operand(1)->opcode());
}

TEST_F(InstructionFusionTest, BitcastIntoAdd) {
  auto module = tools::Parse(R"(
    HloModule test_module

    ENTRY BroadcastIntoAdd {
      p0 = f32[4,1,1]{2,1,0} parameter(0)
      p1 = f32[4,1]{1,0} parameter(1)
      bitcast = f32[4,1]{1,0} bitcast(p0)
      ROOT add = f32[4,1] add(bitcast, p1)
    })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_THAT(root->fused_expression_root(),
              op::Add(op::Bitcast(op::Parameter()), op::Parameter()));
}

TEST_F(InstructionFusionTest, AddIntoBitcast) {
  auto module = tools::Parse(R"(
    HloModule test_module

    ENTRY BroadcastIntoAdd {
      p0 = f32[4,1,1]{2,1,0} parameter(0)
      p1 = f32[4,1]{1,0} parameter(1)
      add = f32[4,1] add(p0, p1)
      ROOT bitcast = f32[4,1,1] bitcast(add)
    })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_THAT(root->fused_expression_root(),
              op::Bitcast(op::Add(op::Parameter(), op::Parameter())));
}

}  // namespace gpu
}  // namespace xla

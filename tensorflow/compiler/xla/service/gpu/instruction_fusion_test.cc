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

TEST_F(InstructionFusionTest, PotentialBitcastTransposeOfConvolutionUnfused) {
  HloComputation::Builder builder(TestName());
  auto input = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 1, 1, 3}), "input"));
  auto filter = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1, 1, 1, 2}), "filter"));

  Window conv_window;
  WindowDimension* conv_window_row = conv_window.add_dimensions();
  conv_window_row->set_size(1);
  WindowDimension* conv_window_col = conv_window.add_dimensions();
  conv_window_col->set_size(2);
  conv_window_col->set_padding_high(1);

  ConvolutionDimensionNumbers conv_dnums;
  conv_dnums.set_input_batch_dimension(0);
  conv_dnums.set_output_batch_dimension(0);
  conv_dnums.set_input_feature_dimension(1);
  conv_dnums.set_output_feature_dimension(1);
  conv_dnums.add_input_spatial_dimensions(2);
  conv_dnums.add_output_spatial_dimensions(2);
  conv_dnums.add_input_spatial_dimensions(3);
  conv_dnums.add_output_spatial_dimensions(3);
  conv_dnums.set_kernel_output_feature_dimension(0);
  conv_dnums.set_kernel_input_feature_dimension(1);
  conv_dnums.add_kernel_spatial_dimensions(2);
  conv_dnums.add_kernel_spatial_dimensions(3);

  auto conv = builder.AddInstruction(
      HloInstruction::CreateConvolve(ShapeUtil::MakeShape(F32, {1, 1, 1, 3}),
                                     input, filter, conv_window, conv_dnums));
  auto transpose = builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {3, 1, 1, 1}), conv, {3, 2, 1, 0}));
  builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {3}), transpose));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
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

}  // namespace gpu
}  // namespace xla

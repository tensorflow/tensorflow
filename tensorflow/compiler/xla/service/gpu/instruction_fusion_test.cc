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

// Tests that broadcasts fused into a fusion with a reduce root.
TEST_F(InstructionFusionTest, BroadcastIntoReduce) {
  auto module = tools::Parse(R"(
    HloModule test_module

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY BroadcastIntoReduce {
      constant = f32[] constant(1)
      broadcast = f32[16,16,16,16]{3,2,1,0} broadcast(constant), dimensions={}
      constant.1 = f32[] constant(0)
      ROOT reduce = f32[] reduce(broadcast, constant.1), dimensions={0,1,2,3},
                                                         to_apply=add
    })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_THAT(root->fused_expression_root(),
              op::Reduce(op::Broadcast(op::Parameter()), op::Parameter()));
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

TEST_F(InstructionFusionTest, DontFuseGTE) {
  auto module = tools::Parse(R"(
  HloModule test_module
  ENTRY DontFuseGTE {
    p0 = (f32[10], f32[10]) parameter(0)
    gte0 = f32[10] get-tuple-element(p0), index=0
    gte1 = f32[10] get-tuple-element(p0), index=1
    ROOT add = f32[10] add(gte0, gte1)
  })")
                    .ValueOrDie();

  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie());
}

TEST_F(InstructionFusionTest, DotOutputFusion) {
  auto module = tools::Parse(R"(
  HloModule test_module
  ENTRY OutputFusion {
    constant = f32[] constant(3)
    p0 = f32[4,3]{1,0} parameter(0)
    p1 = f32[4,3]{1,0} parameter(1)
    transpose = f32[3,4]{1,0} transpose(p1), dimensions={1, 0}
    dot = f32[4,4]{1,0} dot(p0, transpose)
    ROOT mul = f32[4,4] multiply(constant, dot)
  })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_THAT(
      root->fused_expression_root(),
      op::Multiply(op::Parameter(),
                   op::Dot(op::Parameter(), op::Transpose(op::Parameter()))));
}

// Compute sum(1/p0), where p0 has type f32, twice.  Check that the division is
// duplicated and fused into both reduces.
TEST_F(InstructionFusionTest, FloatingPointDivIsCheap) {
  auto module = tools::Parse(R"(
  HloModule test_module
  Add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }
  ENTRY TestComputation {
    zero = f32[] constant(0)
    one = f32[] constant(1)
    p0 = f32[100] parameter(0)
    recip = f32[100] divide(one, p0)
    sum1 = f32[] reduce(recip, zero), dimensions={0}, to_apply=Add
    sum2 = f32[] reduce(recip, zero), dimensions={0}, to_apply=Add
    ROOT root = (f32[], f32[]) tuple(sum1, sum2)
  })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::Fusion(), op::Fusion()));
}

// Compute sum(100/p0), where p0 has type s32, twice.  Check that the division
// is *not* duplicated and fused into both reduces, because we say that integer
// division is not cheap.
TEST_F(InstructionFusionTest, IntegerDivIsNotCheap) {
  auto module = tools::Parse(R"(
  HloModule test_module
  Add {
    lhs = s32[] parameter(0)
    rhs = s32[] parameter(1)
    ROOT add = s32[] add(lhs, rhs)
  }
  ENTRY TestComputation {
    zero = s32[] constant(0)
    one_hundred = s32[] constant(100)
    p0 = s32[100] parameter(0)
    recip = s32[100] divide(one_hundred, p0)
    sum1 = s32[] reduce(recip, zero), dimensions={0}, to_apply=Add
    sum2 = s32[] reduce(recip, zero), dimensions={0}, to_apply=Add
    ROOT mul = (s32[], s32[]) tuple(sum1, sum2)
  })")
                    .ValueOrDie();

  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie());
}

}  // namespace gpu
}  // namespace xla

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

#include "tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.h"

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace cpu {

using InstructionFusionTest = HloTestBase;

TEST_F(InstructionFusionTest, BroadcastFused) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {8});
  Shape result_shape = ShapeUtil::MakeShape(F32, {8, 8});
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  auto broadcast1 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(result_shape, param0, {1}));
  builder.AddInstruction(HloInstruction::CreateUnary(
      result_shape, HloOpcode::kNegate, broadcast1));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto computation = module->entry_computation();
  auto did_fusion = CpuInstructionFusion().Run(module.get());
  ASSERT_TRUE(did_fusion.ok());
  EXPECT_TRUE(did_fusion.ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  ASSERT_EQ(HloOpcode::kFusion, root->opcode());
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kLoop);
  HloInstruction* fused_root = root->fused_expression_root();
  EXPECT_EQ(HloOpcode::kNegate, fused_root->opcode());
  EXPECT_EQ(HloOpcode::kBroadcast, fused_root->operand(0)->opcode());
}

TEST_F(InstructionFusionTest, SliceBeforeReverseNotFused) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {8});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {4});
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  // The (slice, reverse) pair can't be fused into a loop because reverse
  // doesn't act elementwise on slice.
  auto slice1 = builder.AddInstruction(
      HloInstruction::CreateSlice(slice_shape, param0, {0}, {4}, {1}));
  auto reverse2 = builder.AddInstruction(
      HloInstruction::CreateReverse(slice_shape, slice1, {0}));
  builder.AddInstruction(
      HloInstruction::CreateUnary(slice_shape, HloOpcode::kNegate, reverse2));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  auto computation = module->entry_computation();
  auto did_fusion = CpuInstructionFusion().Run(module.get());
  ASSERT_TRUE(did_fusion.ok());
  EXPECT_TRUE(did_fusion.ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  ASSERT_EQ(HloOpcode::kFusion, root->opcode());
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kLoop);
  HloInstruction* fused_root = root->fused_expression_root();
  EXPECT_EQ(HloOpcode::kNegate, fused_root->opcode());
  EXPECT_EQ(HloOpcode::kReverse, fused_root->operand(0)->opcode());
  EXPECT_EQ(HloOpcode::kSlice, root->operand(0)->opcode());
}

}  // namespace cpu
}  // namespace xla

int main(int argc, char** argv) {
  return xla::ParseDebugOptionsFlagsAndRunTests(argc, argv);
}

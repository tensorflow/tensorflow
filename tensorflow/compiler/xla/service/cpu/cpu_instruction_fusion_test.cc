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

#include <algorithm>
#include <set>

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace cpu {
namespace {

using InstructionFusionTest = HloTestBase;

TEST_F(InstructionFusionTest, DotOperationFusion_Basic_0) {
  HloComputation::Builder builder(TestName());
  HloInstruction* arg0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1024, 256}), "arg0"));
  HloInstruction* arg1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {256, 1}), "arg1"));

  HloInstruction* exp0 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {1024, 256}), HloOpcode::kExp, arg0));
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {1024, 1}), HloOpcode::kDot, exp0, arg1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(dot, computation->root_instruction());
  EXPECT_TRUE(CpuInstructionFusion().Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), op::Fusion());
}

TEST_F(InstructionFusionTest, DotOperationFusion_Basic_1) {
  HloComputation::Builder builder(TestName());
  HloInstruction* arg0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 256}), "arg0"));
  HloInstruction* arg1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {256, 1024}), "arg1"));

  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {256, 1024}), HloOpcode::kExp, arg1));
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {1, 1024}), HloOpcode::kDot, arg0, exp1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(dot, computation->root_instruction());
  EXPECT_TRUE(CpuInstructionFusion().Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), op::Fusion());
}

TEST_F(InstructionFusionTest, DotOperationFusion_Bitcast) {
  HloComputation::Builder builder(TestName());
  HloInstruction* arg0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {2, 512, 2, 128}), "arg0"));
  HloInstruction* arg1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {256, 1}), "arg1"));

  HloInstruction* exp0 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {2, 512, 2, 128}), HloOpcode::kExp, arg0));
  HloInstruction* bitcast0 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {1024, 256}), HloOpcode::kBitcast, exp0));
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {1024, 1}), HloOpcode::kDot, bitcast0, arg1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(dot, computation->root_instruction());
  EXPECT_TRUE(CpuInstructionFusion().Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), op::Fusion());
}

TEST_F(InstructionFusionTest, DotOperationFusion_Reshape) {
  HloComputation::Builder builder(TestName());
  HloInstruction* arg0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {2, 512, 2, 128}), "arg0"));
  HloInstruction* arg1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {256, 1}), "arg1"));

  HloInstruction* exp0 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {2, 512, 2, 128}), HloOpcode::kExp, arg0));
  HloInstruction* reshape0 =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(S32, {1024, 256}), exp0));
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {1024, 1}), HloOpcode::kDot, reshape0, arg1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(dot, computation->root_instruction());
  EXPECT_TRUE(CpuInstructionFusion().Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), op::Fusion());
}

TEST_F(InstructionFusionTest, DotOperationFusion_TooLarge) {
  HloComputation::Builder builder(TestName());
  HloInstruction* arg0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 32 * 1024}), "arg0"));
  HloInstruction* arg1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {256, 32 * 1024}), "arg1"));

  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {256, 32 * 1024}), HloOpcode::kExp, arg1));
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {1, 32 * 1024}), HloOpcode::kDot, arg0, exp1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(dot, computation->root_instruction());
  EXPECT_FALSE(CpuInstructionFusion().Run(module.get()).ValueOrDie());
  EXPECT_EQ(dot, computation->root_instruction());
}

TEST_F(InstructionFusionTest, DotOperationFusion_ElementReuse) {
  HloComputation::Builder builder(TestName());
  HloInstruction* arg0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {2, 256}), "arg0"));
  HloInstruction* arg1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {256, 1024}), "arg1"));

  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {256, 1024}), HloOpcode::kExp, arg1));
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {2, 1024}), HloOpcode::kDot, arg0, exp1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(dot, computation->root_instruction());
  EXPECT_FALSE(CpuInstructionFusion().Run(module.get()).ValueOrDie());
  EXPECT_EQ(dot, computation->root_instruction());
}

TEST_F(InstructionFusionTest, DotOperationFusion_TransposeFusion) {
  HloComputation::Builder builder(TestName());
  HloInstruction* arg0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 256}), "arg0"));
  HloInstruction* arg1 = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {1024, 256}), "arg1"));

  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {1024, 256}), HloOpcode::kExp, arg1));
  HloInstruction* transpose1 =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(S32, {256, 1024}), exp1, {1, 0}));
  builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {1, 1024}), HloOpcode::kDot, arg0, transpose1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  TransposeFolding transpose_folding(
      [](const HloInstruction& dot,
         const TransposeFolding::OperandIndices& candidate_operands) {
        return candidate_operands;
      },
      TransposeFolding::NeverFoldTranspose);
  EXPECT_TRUE(transpose_folding.Run(module.get()).ValueOrDie());
  EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(computation->root_instruction()->fusion_kind(),
            HloInstruction::FusionKind::kTransposeDot);
  EXPECT_FALSE(CpuInstructionFusion().Run(module.get()).ValueOrDie());
  EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(computation->root_instruction()->fusion_kind(),
            HloInstruction::FusionKind::kTransposeDot);
}

class OpcodeFusionTest : public InstructionFusionTest {
 protected:
  // Runs CPU instruction fusion on the given module, and tests that the result
  // contains a fused op at the root with exactly the given multiset of opcodes.
  void RunFusionAndCheckOpcodesWereFused(
      HloModule* module, const std::multiset<HloOpcode>& expected_opcodes) {
    auto computation = module->entry_computation();
    auto did_fusion = CpuInstructionFusion().Run(module);
    ASSERT_TRUE(did_fusion.ok());
    EXPECT_TRUE(did_fusion.ValueOrDie());

    HloInstruction* root = computation->root_instruction();
    ASSERT_THAT(root, op::Fusion());
    EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kLoop);

    std::vector<HloOpcode> fused_opcodes(root->fused_instructions().size());
    std::transform(root->fused_instructions().begin(),
                   root->fused_instructions().end(), fused_opcodes.begin(),
                   [](const std::unique_ptr<HloInstruction>& hlo) {
                     return hlo->opcode();
                   });

    EXPECT_EQ(
        std::multiset<HloOpcode>(fused_opcodes.begin(), fused_opcodes.end()),
        expected_opcodes);
  }
};

TEST_F(OpcodeFusionTest, Exponential_Bitcast_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {1, 4});
  Shape result_shape = ShapeUtil::MakeShape(F32, {4});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  // InstructionFusion::ShouldFuse() precludes fusing a bitcast whose operand
  // is a parameter, so create an operand between the parameter and bitcast.
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(param_shape, HloOpcode::kExp, param0));
  HloInstruction* bitcast2 = builder.AddInstruction(
      HloInstruction::CreateUnary(result_shape, HloOpcode::kBitcast, exp1));
  builder.AddInstruction(
      HloInstruction::CreateUnary(result_shape, HloOpcode::kNegate, bitcast2));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(), {HloOpcode::kNegate, HloOpcode::kBitcast, HloOpcode::kExp,
                     HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, Broadcast_Bitcast_DynamicSlice_Tanh) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {8});
  Shape starts_shape = ShapeUtil::MakeShape(F32, {2});
  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {1, 8, 8});
  Shape bitcast_shape = ShapeUtil::MakeShape(F32, {8, 8});
  Shape dynamic_slice_shape = ShapeUtil::MakeShape(F32, {4, 4});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, starts_shape, "starts"));
  HloInstruction* broadcast2 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(broadcast_shape, param0, {1}));
  HloInstruction* bitcast3 = builder.AddInstruction(HloInstruction::CreateUnary(
      bitcast_shape, HloOpcode::kBitcast, broadcast2));
  HloInstruction* dynamic_slice4 =
      builder.AddInstruction(HloInstruction::CreateDynamicSlice(
          dynamic_slice_shape, bitcast3, param1, {4, 4}));
  builder.AddInstruction(HloInstruction::CreateUnary(
      dynamic_slice_shape, HloOpcode::kTanh, dynamic_slice4));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kTanh, HloOpcode::kDynamicSlice, HloOpcode::kBitcast,
       HloOpcode::kBroadcast, HloOpcode::kParameter, HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, Broadcast_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {8});
  Shape result_shape = ShapeUtil::MakeShape(F32, {8, 8});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  HloInstruction* broadcast1 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(result_shape, param0, {1}));
  builder.AddInstruction(HloInstruction::CreateUnary(
      result_shape, HloOpcode::kNegate, broadcast1));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kNegate, HloOpcode::kBroadcast, HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, DynamicSlice_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {4});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {1});
  Shape result_shape = ShapeUtil::MakeShape(F32, {2});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, slice_shape, "starts"));
  HloInstruction* dynamic_slice2 = builder.AddInstruction(
      HloInstruction::CreateDynamicSlice(result_shape, param0, param1, {2}));
  builder.AddInstruction(HloInstruction::CreateUnary(
      result_shape, HloOpcode::kNegate, dynamic_slice2));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(), {HloOpcode::kNegate, HloOpcode::kDynamicSlice,
                     HloOpcode::kParameter, HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, Exponential_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {4});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(param_shape, HloOpcode::kExp, param0));
  builder.AddInstruction(
      HloInstruction::CreateUnary(param_shape, HloOpcode::kNegate, exp1));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kNegate, HloOpcode::kExp, HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, Reshape_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {4, 4});
  Shape result_shape = ShapeUtil::MakeShape(F32, {16});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  HloInstruction* reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(result_shape, param0));
  builder.AddInstruction(
      HloInstruction::CreateUnary(result_shape, HloOpcode::kNegate, reshape1));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kNegate, HloOpcode::kReshape, HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, Reverse_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {8});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  HloInstruction* reverse1 = builder.AddInstruction(
      HloInstruction::CreateReverse(param_shape, param0, {0}));
  builder.AddInstruction(
      HloInstruction::CreateUnary(param_shape, HloOpcode::kNegate, reverse1));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kNegate, HloOpcode::kReverse, HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, Slice_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {4});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {2});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  HloInstruction* slice1 = builder.AddInstruction(
      HloInstruction::CreateSlice(slice_shape, param0, {0}, {4}, {2}));
  builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {2}), HloOpcode::kNegate, slice1));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(),
      {HloOpcode::kNegate, HloOpcode::kSlice, HloOpcode::kParameter});
}

TEST_F(OpcodeFusionTest, Exponential_Transpose_Negate) {
  HloComputation::Builder builder(TestName());
  Shape param_shape = ShapeUtil::MakeShape(F32, {3, 4});
  Shape result_shape = ShapeUtil::MakeShape(F32, {4, 3});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "param"));
  // InstructionFusion::ShouldFuse() precludes fusing a transpose whose operand
  // is a parameter, so create an operand between the parameter and transpose.
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(param_shape, HloOpcode::kExp, param0));
  HloInstruction* transpose2 = builder.AddInstruction(
      HloInstruction::CreateTranspose(result_shape, exp1, {1, 0}));
  builder.AddInstruction(HloInstruction::CreateUnary(
      result_shape, HloOpcode::kNegate, transpose2));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  RunFusionAndCheckOpcodesWereFused(
      module.get(), {HloOpcode::kNegate, HloOpcode::kTranspose, HloOpcode::kExp,
                     HloOpcode::kParameter});
}

}  // namespace
}  // namespace cpu
}  // namespace xla

int main(int argc, char** argv) {
  return xla::ParseDebugOptionsFlagsAndRunTests(argc, argv);
}

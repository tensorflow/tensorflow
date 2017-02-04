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

#include "tensorflow/compiler/xla/service/transpose_folding.h"

#include <memory>
#include <set>
#include <vector>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

class TransposeFoldingTest : public ::testing::Test {
 protected:
  void FoldTranspose(HloModule* module) {
    TransposeFolding transpose_folding(gpu::ImplementedAsGemm);
    EXPECT_IS_OK(transpose_folding.Run(module).status());
  }
};

TEST_F(TransposeFoldingTest, FoldTranspose) {
  auto builder = HloComputation::Builder("entry_computation");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {2, 3}),
      /*name=*/"x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {2, 3}),
      /*name=*/"y"));
  HloInstruction* transpose_y =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {3, 2}), y, {1, 0}));
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {2, 2}), /*opcode=*/HloOpcode::kDot,
      /*lhs=*/x, /*rhs=*/transpose_y));

  HloModule module("test_module");
  HloComputation* entry_computation =
      module.AddEntryComputation(builder.Build(dot));
  FoldTranspose(&module);

  // Instructions after folding: x, y, and the fusion.
  std::set<HloInstruction*> instruction_set;
  for (auto& instruction : entry_computation->instructions()) {
    instruction_set.insert(instruction.get());
  }
  CHECK_EQ(1, instruction_set.erase(x)) << "x is not in entry_computation.";
  CHECK_EQ(1, instruction_set.erase(y)) << "y is not in entry_computation.";
  CHECK_EQ(1, instruction_set.size())
      << "entry_computation should contain exactly 3 instructions.";
  HloInstruction* fusion = *instruction_set.begin();
  EXPECT_EQ(HloOpcode::kFusion, fusion->opcode());

  // The fusion instruction should contain two parameters, one transpose and
  // one dot.
  EXPECT_EQ(4, fusion->fused_instructions().size());
}

TEST_F(TransposeFoldingTest, FoldTransposeConstant) {
  auto builder = HloComputation::Builder("entry_computation");
  // 2x1
  HloInstruction* const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2<float>({{1}, {2}})));
  // 3x2
  HloInstruction* const1 =
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}, {5, 6}})));
  HloInstruction* transpose0 =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {1, 2}), const0, {1, 0}));
  HloInstruction* transpose1 =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {2, 3}), const1, {1, 0}));
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {1, 3}), /*opcode=*/HloOpcode::kDot,
      /*lhs=*/transpose0, /*rhs=*/transpose1));

  HloModule module("test_module");
  HloComputation* entry_computation =
      module.AddEntryComputation(builder.Build(dot));
  FoldTranspose(&module);

  for (auto& instruction : entry_computation->instructions()) {
    if (instruction->opcode() == HloOpcode::kFusion) {
      CHECK_EQ(2, instruction->operand_count());
      EXPECT_EQ(const0, instruction->operand(0));
      EXPECT_EQ(const1, instruction->operand(1));
    }
  }

  // The created fusion instruction should contain two parameters, two
  // transposes (one for each parameter) and one dot.
  EXPECT_EQ(5,
            entry_computation->root_instruction()->fused_instructions().size());
}

TEST_F(TransposeFoldingTest, FuseWithConstantOperands) {
  auto builder = HloComputation::Builder("entry");
  // (1.0 + 2.0) * (2.0 - 3.0)
  HloInstruction* const1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  HloInstruction* const2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  HloInstruction* const3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(3.0)));
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      const1->shape(), HloOpcode::kAdd, const1, const2));
  HloInstruction* sub = builder.AddInstruction(HloInstruction::CreateBinary(
      const2->shape(), HloOpcode::kSubtract, const2, const3));
  HloInstruction* mul = builder.AddInstruction(HloInstruction::CreateBinary(
      add->shape(), HloOpcode::kMultiply, add, sub));

  HloModule module("fuse_with_constant_operands");
  HloComputation* entry_computation =
      module.AddEntryComputation(builder.Build(mul));
  HloInstruction* call = module.OutlineExpressionFromComputation(
      {add, sub, mul}, "", entry_computation);
  EXPECT_EQ(call, entry_computation->root_instruction());
  HloComputation* callee_computation = call->to_apply();
  // The arguments to the call should be const1, const2, and const3.
  EXPECT_MATCH(call->operands(), testing::UnorderedMatcher<HloInstruction*>(
                                     const1, const2, const3));

  // The callee should contain 3 parameters and 3 binary operators.
  EXPECT_EQ(6, callee_computation->instructions().size());
}

}  // namespace xla

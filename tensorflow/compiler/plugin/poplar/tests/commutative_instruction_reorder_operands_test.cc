/* Copyright 2018 Graphcore. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/commutative_instruction_reorder_operands.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using CommutativeInstructionReorderOperandsTest = HloTestBase;

TEST_F(CommutativeInstructionReorderOperandsTest, ReorderUnary) {
  Shape s1 = ShapeUtil::MakeShape(F32, {});
  Shape s2 = ShapeUtil::MakeShape(F32, {2, 2});

  auto builder = HloComputation::Builder(TestName());
  auto i1 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "i1"));
  auto i2 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, s2, "i2"));
  auto b1 = builder.AddInstruction(HloInstruction::CreateBroadcast(s2, i1, {}));
  builder.AddInstruction(
      HloInstruction::CreateBinary(s2, HloOpcode::kAdd, b1, i2));

  auto computation = builder.Build();
  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  {
    const auto* root_inst = hlo_module->entry_computation()->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kBroadcast);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kParameter);
  }

  CommutativeInstructionReorderOperands ciro;
  EXPECT_TRUE(ciro.Run(hlo_module.get()).ValueOrDie());

  {
    const auto* root_inst = hlo_module->entry_computation()->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kParameter);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kBroadcast);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, DontReorderUnaryElementwise) {
  Shape s = ShapeUtil::MakeShape(F32, {2, 2});

  auto builder = HloComputation::Builder(TestName());
  auto i1 = builder.AddInstruction(HloInstruction::CreateParameter(0, s, "i1"));
  auto i2 = builder.AddInstruction(HloInstruction::CreateParameter(1, s, "i2"));
  auto e1 = builder.AddInstruction(
      HloInstruction::CreateUnary(s, HloOpcode::kExp, i1));
  builder.AddInstruction(
      HloInstruction::CreateBinary(s, HloOpcode::kAdd, e1, i2));

  auto computation = builder.Build();
  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  {
    const auto* root_inst = hlo_module->entry_computation()->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kExp);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kParameter);
  }

  CommutativeInstructionReorderOperands ciro;
  EXPECT_FALSE(ciro.Run(hlo_module.get()).ValueOrDie());

  {
    const auto* root_inst = hlo_module->entry_computation()->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kExp);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kParameter);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, ReorderBinary) {
  Shape s1 = ShapeUtil::MakeShape(F32, {2, 1});
  Shape s2 = ShapeUtil::MakeShape(F32, {2, 2});

  PaddingConfig padding;
  auto dimension = padding.add_dimensions();
  dimension->set_edge_padding_low(0);
  dimension->set_edge_padding_high(0);
  dimension->set_interior_padding(0);
  dimension = padding.add_dimensions();
  dimension->set_edge_padding_low(0);
  dimension->set_edge_padding_high(1);
  dimension->set_interior_padding(0);

  auto builder = HloComputation::Builder(TestName());
  auto zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto i1 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "i1"));
  auto i2 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, s2, "i2"));
  auto pad =
      builder.AddInstruction(HloInstruction::CreatePad(s2, i1, zero, padding));
  builder.AddInstruction(
      HloInstruction::CreateBinary(s2, HloOpcode::kMultiply, pad, i2));

  auto computation = builder.Build();
  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  {
    const auto* root_inst = hlo_module->entry_computation()->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kPad);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kParameter);
  }

  CommutativeInstructionReorderOperands ciro;
  EXPECT_TRUE(ciro.Run(hlo_module.get()).ValueOrDie());

  {
    const auto* root_inst = hlo_module->entry_computation()->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kParameter);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kPad);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, DontReorderBinary) {
  Shape s = ShapeUtil::MakeShape(F32, {2, 2});

  auto builder = HloComputation::Builder(TestName());
  auto i1 = builder.AddInstruction(HloInstruction::CreateParameter(0, s, "i1"));
  auto i2 = builder.AddInstruction(HloInstruction::CreateParameter(1, s, "i2"));
  auto i3 = builder.AddInstruction(HloInstruction::CreateParameter(2, s, "i3"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(s, HloOpcode::kAdd, i1, i2));
  builder.AddInstruction(
      HloInstruction::CreateBinary(s, HloOpcode::kMultiply, add, i2));

  auto computation = builder.Build();
  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  {
    const auto* root_inst = hlo_module->entry_computation()->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kAdd);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kParameter);
  }

  CommutativeInstructionReorderOperands ciro;
  EXPECT_FALSE(ciro.Run(hlo_module.get()).ValueOrDie());

  {
    const auto* root_inst = hlo_module->entry_computation()->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kAdd);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kParameter);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, DontReorderBothReshaping) {
  Shape s1 = ShapeUtil::MakeShape(F32, {});
  Shape s2 = ShapeUtil::MakeShape(F32, {2, 2});

  auto builder = HloComputation::Builder(TestName());
  auto i1 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "i1"));
  auto i2 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, s1, "i2"));
  auto b1 = builder.AddInstruction(HloInstruction::CreateBroadcast(s2, i1, {}));
  auto b2 = builder.AddInstruction(HloInstruction::CreateBroadcast(s2, i2, {}));
  builder.AddInstruction(
      HloInstruction::CreateBinary(s2, HloOpcode::kAdd, b1, b2));

  auto computation = builder.Build();
  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  {
    const auto* root_inst = hlo_module->entry_computation()->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kBroadcast);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kBroadcast);
    EXPECT_THAT(root_inst->operand(0), b1);
    EXPECT_THAT(root_inst->operand(1), b2);
  }

  CommutativeInstructionReorderOperands ciro;
  EXPECT_FALSE(ciro.Run(hlo_module.get()).ValueOrDie());

  {
    const auto* root_inst = hlo_module->entry_computation()->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kBroadcast);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kBroadcast);
    EXPECT_THAT(root_inst->operand(0), b1);
    EXPECT_THAT(root_inst->operand(1), b2);
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla

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

#include "tensorflow/compiler/plugin/poplar/driver/wide_const_finder.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using WideConstFinderTest = HloTestBase;

TEST_F(WideConstFinderTest, ReplaceWideConstants) {
  Shape s1 = ShapeUtil::MakeShape(S32, {2, 2});
  Shape s2 = ShapeUtil::MakeShape(F32, {2, 2});

  auto builder = HloComputation::Builder(TestName());
  auto i1 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "i1"));
  auto i2 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, s2, "i2"));
  auto c1 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int>({{0, 0}, {0, 0}})));
  auto c2 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{0, 0}, {0, 0}})));
  auto add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(s1, HloOpcode::kAdd, i1, c1));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(s1, HloOpcode::kAdd, i2, c2));

  builder.AddInstruction(HloInstruction::CreateTuple({add1, add2}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  EXPECT_THAT(hlo_module->computation_count(), 1);
  EXPECT_THAT(hlo_module->entry_computation()->instruction_count(), 7);

  WideConstFinder finder;
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());
  EXPECT_THAT(hlo_module->entry_computation()->instruction_count(), 11);
  // note: the original constant isn't removed

  HloInstruction* inst;

  inst = hlo_module->entry_computation()->root_instruction();
  inst = inst->mutable_operand(0)->mutable_operand(1);
  EXPECT_THAT(inst->opcode(), HloOpcode::kBroadcast);
  EXPECT_TRUE(ShapeUtil::Equal(inst->shape(), s1));
  inst = inst->mutable_operand(0);
  EXPECT_THAT(inst->opcode(), HloOpcode::kConstant);
  EXPECT_TRUE(ShapeUtil::Equal(inst->shape(), ShapeUtil::MakeShape(S32, {})));

  inst = hlo_module->entry_computation()->root_instruction();
  inst = inst->mutable_operand(1)->mutable_operand(1);
  EXPECT_THAT(inst->opcode(), HloOpcode::kBroadcast);
  EXPECT_TRUE(ShapeUtil::Equal(inst->shape(), s2));
  inst = inst->mutable_operand(0);
  EXPECT_THAT(inst->opcode(), HloOpcode::kConstant);
  EXPECT_TRUE(ShapeUtil::Equal(inst->shape(), ShapeUtil::MakeShape(F32, {})));
}

TEST_F(WideConstFinderTest, DontReplaceScalars) {
  Shape s1 = ShapeUtil::MakeShape(S32, {});

  auto builder = HloComputation::Builder(TestName());
  auto in =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "input"));
  auto c1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(0)));
  auto c2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(1)));
  auto add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(s1, HloOpcode::kAdd, in, c1));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(s1, HloOpcode::kAdd, in, c2));

  builder.AddInstruction(HloInstruction::CreateTuple({add1, add2}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  EXPECT_THAT(hlo_module->computation_count(), 1);
  EXPECT_THAT(hlo_module->entry_computation()->instruction_count(), 6);

  WideConstFinder finder;
  EXPECT_TRUE(finder.Run(hlo_module.get()).ValueOrDie());
  EXPECT_THAT(hlo_module->entry_computation()->instruction_count(), 6);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla

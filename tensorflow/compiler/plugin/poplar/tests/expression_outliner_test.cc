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

#include "tensorflow/compiler/plugin/poplar/driver/expression_outliner.h"

#include "tensorflow/compiler/xla/service/shape_inference.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ExpressionOutlinerTest = HloTestBase;

// General extraction
//
//  i i  i  i
//  \ /  \ /
//   a    b
//    \  /
//     c
//     |
TEST_F(ExpressionOutlinerTest, OutlineSimpleTree) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "input2"));
  auto in3 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, shape, "input3"));
  auto in4 = builder.AddInstruction(
      HloInstruction::CreateParameter(3, shape, "input4"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, in1, in2));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kSubtract, in3, in4));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kMultiply, add, sub));
  builder.AddInstruction(HloInstruction::CreateTuple({mul}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  std::set<const HloInstruction*> inplace;

  ExpressionOutliner eo(inplace);
  EXPECT_TRUE(eo.Run(hlo_module.get()).ValueOrDie());

  auto* comp = hlo_module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 6);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand_count(), 4);
}

// Shared inputs to outlined section (a+b+c)
//
//  i   i   i
//  \  / \ /
//   a    b
//    \  /
//     c
//     |
TEST_F(ExpressionOutlinerTest, OutlineTreeWithSharedInputs) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "input2"));
  auto in3 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, shape, "input3"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, in1, in2));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kSubtract, in1, in3));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kMultiply, add, sub));
  builder.AddInstruction(HloInstruction::CreateTuple({mul}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  std::set<const HloInstruction*> inplace;

  ExpressionOutliner eo(inplace);
  EXPECT_TRUE(eo.Run(hlo_module.get()).ValueOrDie());

  auto* comp = hlo_module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 5);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand_count(), 3);
}

// Don't outline a single operation
TEST_F(ExpressionOutlinerTest, DontOutlineSingleOps) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "input2"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, in1, in2));
  builder.AddInstruction(HloInstruction::CreateTuple({add}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  std::set<const HloInstruction*> inplace;

  ExpressionOutliner eo(inplace);
  EXPECT_TRUE(eo.Run(hlo_module.get()).ValueOrDie());

  auto* comp = hlo_module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 4);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kAdd);
}

// Outline a subgraph which contains two paths
//
//  i i
//  \ /
//   a
//  / \
// b  c
// \  /
//  d
//  |
TEST_F(ExpressionOutlinerTest, OutlineTwoPaths) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "input2"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, in1, in2));
  auto sin = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kSin, add));
  auto cos = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, add));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kSubtract, cos, sin));
  builder.AddInstruction(HloInstruction::CreateTuple({sub}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  std::set<const HloInstruction*> inplace;

  ExpressionOutliner eo(inplace);
  EXPECT_TRUE(eo.Run(hlo_module.get()).ValueOrDie());

  auto* comp = hlo_module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 4);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand_count(), 2);
}

// Don't outline op 'b' where 'X' is not part of the outline (only outline c+d)
//
//     a
//    / \
//   b  c
//  / \ /
// X   d
// |   |
TEST_F(ExpressionOutlinerTest, DontOutlineOpsWithOutputsOutsideOfTheSubgraph) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "input2"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, in1, in2));
  auto sin = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kSin, add));
  auto cos = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, add));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kSubtract, cos, sin));
  builder.AddInstruction(HloInstruction::CreateTuple({sub, cos}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  std::set<const HloInstruction*> inplace;

  ExpressionOutliner eo(inplace);
  EXPECT_TRUE(eo.Run(hlo_module.get()).ValueOrDie());

  auto* comp = hlo_module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 6);
  EXPECT_THAT(inst->operand_count(), 2);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand_count(), 2);
  EXPECT_THAT(inst->operand(1)->opcode(), HloOpcode::kCos);
}

// Do two independent outlines
TEST_F(ExpressionOutlinerTest, OutlineTwoSubgraphs) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "input2"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, in1, in2));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kSubtract, in1, in2));
  auto sin = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kSin, add));
  auto cos = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, sub));
  builder.AddInstruction(HloInstruction::CreateTuple({sin, cos}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  std::set<const HloInstruction*> inplace;

  ExpressionOutliner eo(inplace);
  EXPECT_TRUE(eo.Run(hlo_module.get()).ValueOrDie());

  auto* comp = hlo_module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 5);
  EXPECT_THAT(inst->operand_count(), 2);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand_count(), 2);
  EXPECT_THAT(inst->operand(1)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(1)->operand_count(), 2);
}

// two paths of uneven length
//
//  i i
//  \ /
//   a
//  / \
// b  c
// |  |
// |  d
// \  /
//  e
//  |
TEST_F(ExpressionOutlinerTest, OutlineTwoPathUnevenLength1) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "input2"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, in1, in2));
  auto sin = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kSin, add));
  auto cos = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, add));
  auto neg = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, cos));
  auto abs = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kAbs, neg));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kSubtract, sin, abs));
  builder.AddInstruction(HloInstruction::CreateTuple({sub}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  std::set<const HloInstruction*> inplace;

  ExpressionOutliner eo(inplace);
  EXPECT_TRUE(eo.Run(hlo_module.get()).ValueOrDie());

  auto* comp = hlo_module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 4);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand_count(), 2);
}

// two paths of uneven length
//
//  i i
//  \ /
//   a
//  / \
// b  c
// |  |
// d  |
// \  /
//  e
//  |
TEST_F(ExpressionOutlinerTest, OutlineTwoPathUnevenLength2) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "input2"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, in1, in2));
  auto sin = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kSin, add));
  auto cos = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, add));
  auto neg = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, cos));
  auto abs = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kAbs, neg));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kSubtract, abs, sin));
  builder.AddInstruction(HloInstruction::CreateTuple({sub}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  std::set<const HloInstruction*> inplace;

  ExpressionOutliner eo(inplace);
  EXPECT_TRUE(eo.Run(hlo_module.get()).ValueOrDie());

  auto* comp = hlo_module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 4);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand_count(), 2);
}

// three paths
//
//  i i
//  \ /
//   a
//  / \
// b  c
// \ / \
//  d  e
//  \ /
//   f
//   |
TEST_F(ExpressionOutlinerTest, OutlineThreePaths) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "input2"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, in1, in2));
  auto sin = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kSin, add));
  auto cos = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, add));
  auto neg = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, cos));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kMultiply, sin, cos));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kSubtract, mul, neg));
  builder.AddInstruction(HloInstruction::CreateTuple({sub}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  std::set<const HloInstruction*> inplace;

  ExpressionOutliner eo(inplace);
  EXPECT_TRUE(eo.Run(hlo_module.get()).ValueOrDie());

  auto* comp = hlo_module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 4);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand_count(), 2);
}

// two clusters (bdf, eg)
//
//    i i
//    \ /
//     a
//    / \
//   b  c
//  /\ / \
// |  d  e
// \ /   |
//  f    g
//  |    |
TEST_F(ExpressionOutlinerTest, OutlineThreePathsEarlyExit) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "input2"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, in1, in2));
  auto sin = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kSin, add));
  auto cos = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, add));
  auto neg = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, cos));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kExp, neg));
  auto mul = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kMultiply, sin, cos));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kSubtract, mul, sin));
  builder.AddInstruction(HloInstruction::CreateTuple({sub, exp}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  std::set<const HloInstruction*> inplace;

  ExpressionOutliner eo(inplace);
  EXPECT_TRUE(eo.Run(hlo_module.get()).ValueOrDie());

  auto* comp = hlo_module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 7);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(1)->opcode(), HloOpcode::kCall);

  auto* ext_1 = inst->operand(0)->to_apply();
  auto* ext_2 = inst->operand(1)->to_apply();
  int n_inst_1 = ext_1->instruction_count() - ext_1->num_parameters();
  int n_inst_2 = ext_2->instruction_count() - ext_2->num_parameters();
  EXPECT_THAT(n_inst_1 + n_inst_2, 5);
}

// Does the right thing with in-place ops
//
//  i i i
//  \ / |
//   a  |
//  / \ |
// b  c |
// \  / /
//  d  /
//  \ /
//   e  <- inplace-op
TEST_F(ExpressionOutlinerTest, OutlineWithInplace) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "input1"));
  auto in2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "input2"));
  auto in3 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, shape, "input3"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, in1, in2));
  auto sin = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kSin, add));
  auto cos = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, add));
  auto mul = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, cos, sin));
  auto sub = builder.AddInstruction(
    HloInstruction::CreateBinary(shape, HloOpcode::kSubtract, in3, mul));
  builder.AddInstruction(
      HloInstruction::CreateTuple({sub}));

  auto computation = builder.Build();

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(std::move(computation));

  std::set<const HloInstruction*> inplace;
  inplace.insert(sub);

  ExpressionOutliner eo(inplace);
  EXPECT_TRUE(eo.Run(hlo_module.get()).ValueOrDie());

  auto* comp = hlo_module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 6);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kSubtract);
  EXPECT_THAT(inst->operand(0)->operand(1)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand(1)->operand_count(), 2);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla

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

#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace {

AlgebraicSimplifier::ValidBitcastCallback bitcasting_callback() {
  return [](const Shape&, const Shape&) { return true; };
}
AlgebraicSimplifier::ValidBitcastCallback non_bitcasting_callback() {
  return [](const Shape&, const Shape&) { return false; };
}

using AlgebraicSimplifierTest = HloTestBase;

// Test that A + 0 is simplified to A
TEST_F(AlgebraicSimplifierTest, AddZero) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, param0, zero));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that A - 0 is simplified to A
TEST_F(AlgebraicSimplifierTest, SubZero) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kSubtract, param0, zero));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSubtract);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that A/1 is simplified to A for a scalar.
TEST_F(AlgebraicSimplifierTest, DivOneScalar) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, param0, one));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, div);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that A/1 is simplified to A for an array.
TEST_F(AlgebraicSimplifierTest, DivOneArray) {
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* one = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 1.0}, {1.0, 1.0}})));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kDivide, param0, one));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, div);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that get_element(make_tuple({A,B}),1) is simplified to B
TEST_F(AlgebraicSimplifierTest, SelectMakeTuple) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r0f32, "param2"));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({param0, param1}));
  HloInstruction* get = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(r0f32, tuple, 1));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, get, param2));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, add);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, add);
  EXPECT_EQ(root->operand(0), param1);
  EXPECT_EQ(root->operand(1), param2);
}

// Test that exp(A)/exp(B) is simplified to exp(A-B)
TEST_F(AlgebraicSimplifierTest, ExpDiv) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* exp0 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param0));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param1));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, exp0, exp1));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kDivide);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kExp);
  EXPECT_EQ(root->operand_count(), 1);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kSubtract);
  EXPECT_EQ(root->operand(0)->operand(0), param0);
  EXPECT_EQ(root->operand(0)->operand(1), param1);
}

// Test that ln(exp(A)) is simplified to A
TEST_F(AlgebraicSimplifierTest, LnExp) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* exp0 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param0));
  builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kLog, exp0));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kLog);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root, param0);
}

// Test that ln(exp(A)/exp(B)) is simplified to A-B
TEST_F(AlgebraicSimplifierTest, LnExpDiv) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* exp0 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param0));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param1));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, exp0, exp1));
  builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kLog, div));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kLog);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSubtract);
  EXPECT_EQ(root->operand(0), param0);
  EXPECT_EQ(root->operand(1), param1);
}

// Test that pow(A, 0) where A is a scalar is simplified to the scalar
// constant 1.
TEST_F(AlgebraicSimplifierTest, Pow0Scalar) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, zero));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(LiteralUtil::GetFirstElement<float>(root->literal()), 1);
}

// Test that pow(A, 0) where A is not a scalar is simplified to broadcast(1).
TEST_F(AlgebraicSimplifierTest, Pow0Vector) {
  Shape r1f32 = ShapeUtil::MakeShape(F32, {42});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kPower, param0, zero));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), r1f32))
      << ShapeUtil::HumanString(root->shape());
  EXPECT_EQ(root->dimensions().size(), 0);
  EXPECT_TRUE(ShapeUtil::IsScalar(root->operand(0)->shape()));
  EXPECT_EQ(LiteralUtil::GetFirstElement<float>(root->operand(0)->literal()),
            1);
}

// Test that pow(A, 1) is simplified to A.
TEST_F(AlgebraicSimplifierTest, Pow1) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, one));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root, param0);
}

// Test that pow(A, 2) is simplified to A*A.
TEST_F(AlgebraicSimplifierTest, Pow2) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* two = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, two));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kMultiply);
  EXPECT_EQ(root->operand(0), param0);
  EXPECT_EQ(root->operand(1), param0);
}

// Test that pow(A, -1) is simplified to 1/A.
TEST_F(AlgebraicSimplifierTest, PowNegative1) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* negative_one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(-1)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0f32, HloOpcode::kPower,
                                                      param0, negative_one));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kDivide);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(LiteralUtil::GetFirstElement<float>(root->operand(0)->literal()),
            1);
  EXPECT_EQ(root->operand(1), param0);
}

TEST_F(AlgebraicSimplifierTest, ReshapeBroadcast) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});

  auto builder = HloComputation::Builder(TestName());
  auto op = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {3, 2}), "op"));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {6}), op));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {1, 6}), reshape1, {1}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {3, 2}), broadcast));

  auto computation = builder.Build();
  auto module = MakeUnique<HloModule>(TestName());
  module->AddEntryComputation(std::move(computation));
  HloInstruction* root = module->entry_computation()->root_instruction();
  HloPassFix<AlgebraicSimplifier> simplifier(/*is_layout_sensitive=*/false,
                                             non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kParameter);
}

// Test that convert(A, $TYPE) is simplified to A if A is of type $TYPE.
TEST_F(AlgebraicSimplifierTest, ConvertBetweenSameType) {
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(F32, {}), input));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(HloOpcode::kConvert, computation->root_instruction()->opcode());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_EQ(HloOpcode::kConstant, computation->root_instruction()->opcode());
}

TEST_F(AlgebraicSimplifierTest, ConvertF32ToS64) {
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(S64, {}), input));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(HloOpcode::kConvert, computation->root_instruction()->opcode());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_EQ(HloOpcode::kConstant, computation->root_instruction()->opcode());
  EXPECT_EQ(LiteralUtil::GetFirstElement<int64>(
                computation->root_instruction()->literal()),
            42);
}

TEST_F(AlgebraicSimplifierTest, ConvertS64ToF32) {
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64>(42)));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(F32, {}), input));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(HloOpcode::kConvert, computation->root_instruction()->opcode());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_EQ(HloOpcode::kConstant, computation->root_instruction()->opcode());
  EXPECT_EQ(LiteralUtil::GetFirstElement<float>(
                computation->root_instruction()->literal()),
            42.0f);
}

TEST_F(AlgebraicSimplifierTest, ConvertF32ArrayToS64Array) {
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({42.0f, 19.0f})));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(S64, {2}), input));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(HloOpcode::kConvert, computation->root_instruction()->opcode());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_EQ(HloOpcode::kConstant, computation->root_instruction()->opcode());
  EXPECT_EQ(
      LiteralUtil::Get<int64>(computation->root_instruction()->literal(), {0}),
      42);
  EXPECT_EQ(
      LiteralUtil::Get<int64>(computation->root_instruction()->literal(), {1}),
      19);
}

// Test that copies are removed.
TEST_F(AlgebraicSimplifierTest, RemoveCopy) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* copy = builder.AddInstruction(
      HloInstruction::CreateUnary(param0->shape(), HloOpcode::kCopy, param0));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(copy, computation->root_instruction());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_EQ(param0, computation->root_instruction());
}

// Test that a simplification which changes layouts is not performed if layout
// sensitive is true.
TEST_F(AlgebraicSimplifierTest, CopyWithDifferentLayout) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 2}), "param0"));
  HloInstruction* copy = builder.AddInstruction(
      HloInstruction::CreateUnary(param0->shape(), HloOpcode::kCopy, param0));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  // Set to different layouts.
  *param0->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  *copy->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({1, 0});

  EXPECT_EQ(copy, computation->root_instruction());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/true,
                                 non_bitcasting_callback());
  EXPECT_FALSE(simplifier.Run(module.get()).ValueOrDie());

  // Copy has not been removed.
  EXPECT_EQ(copy, computation->root_instruction());
}

// Test that a simplification which preserves layouts is performed if layout
// sensitive is true.
TEST_F(AlgebraicSimplifierTest, CopyWithSameLayout) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 2}), "param0"));
  HloInstruction* copy = builder.AddInstruction(
      HloInstruction::CreateUnary(param0->shape(), HloOpcode::kCopy, param0));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  // Set to same layouts.
  *param0->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  *copy->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  EXPECT_EQ(copy, computation->root_instruction());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/true,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  // Copy has been removed.
  EXPECT_EQ(param0, computation->root_instruction());
}

// Test that a reshape which could be replaced with a bitcast is not if
// add_bitcasts is false.
TEST_F(AlgebraicSimplifierTest, NoBitcastAdded) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 2}), "param0"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {1, 2, 1, 1, 2, 1}), param0));

  *param0->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  *reshape->mutable_shape()->mutable_layout() =
      LayoutUtil::MakeLayout({0, 1, 2, 3, 4, 5});

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(reshape, computation->root_instruction());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/true,
                                 non_bitcasting_callback());
  EXPECT_FALSE(simplifier.Run(module.get()).ValueOrDie());

  // Reshape is not replaced with a bitcast.
  EXPECT_EQ(reshape, computation->root_instruction());
}

// Test transforming reshapes to bitcasts under various conditions.
TEST_F(AlgebraicSimplifierTest, ReshapeReplacedWithBitcast) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 2}), "param0"));
  *param0->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  // Reshape which can be transformed into a bitcast.
  HloInstruction* transformable_reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {1, 2, 1, 1, 2, 1}), param0));
  *transformable_reshape->mutable_shape()->mutable_layout() =
      LayoutUtil::MakeLayout({0, 1, 2, 3, 4, 5});

  // Reshape does not just add degenerate dimensions.
  HloInstruction* dimensions_wrong_reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {1, 4, 1, 1, 1, 1}), param0));
  *dimensions_wrong_reshape->mutable_shape()->mutable_layout() =
      LayoutUtil::MakeLayout({0, 1, 2, 3, 4, 5});

  // Reshape has wrong layout.
  HloInstruction* layout_wrong_reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {1, 2, 1, 1, 2, 1}), param0));
  *layout_wrong_reshape->mutable_shape()->mutable_layout() =
      LayoutUtil::MakeLayout({5, 4, 3, 2, 1, 0});

  // Collect all the reshapes into a tuple so they are not dead.
  builder.AddInstruction(HloInstruction::CreateTuple(
      {transformable_reshape, dimensions_wrong_reshape, layout_wrong_reshape}));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(transformable_reshape, computation->root_instruction()->operand(0));
  EXPECT_EQ(dimensions_wrong_reshape,
            computation->root_instruction()->operand(1));
  EXPECT_EQ(layout_wrong_reshape, computation->root_instruction()->operand(2));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/true,
                                 bitcasting_callback());
  simplifier.Run(module.get()).ValueOrDie();

  // Verify that only the first reshape is replaced.
  EXPECT_NE(transformable_reshape, computation->root_instruction()->operand(0));
  EXPECT_EQ(HloOpcode::kBitcast,
            computation->root_instruction()->operand(0)->opcode());
  EXPECT_EQ(dimensions_wrong_reshape,
            computation->root_instruction()->operand(1));
  EXPECT_EQ(layout_wrong_reshape, computation->root_instruction()->operand(2));
}

TEST_F(AlgebraicSimplifierTest, ReshapeAfterEffectiveUnary) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 3, 4, 5}), "param"));
  HloInstruction* movable_reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {1, 2, 3, 4, 5}), param));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(ShapeUtil::MakeShape(F32, {1, 2, 3, 4, 5}),
                                   HloOpcode::kMaximum, movable_reshape, zero));
  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kMaximum);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 bitcasting_callback());
  simplifier.Run(module.get()).ValueOrDie();
  EXPECT_EQ(HloOpcode::kReshape, computation->root_instruction()->opcode());
  EXPECT_EQ(HloOpcode::kMaximum,
            computation->root_instruction()->operand(0)->opcode());
}

TEST_F(AlgebraicSimplifierTest, TransposeEqualsBitcast1) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {50, 14, 14, 64}), "param"));
  *param->mutable_shape()->mutable_layout() =
      LayoutUtil::MakeLayout({1, 2, 0, 3});

  HloInstruction* transpose =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {14, 14, 50, 64}), param, {1, 2, 0, 3}));
  *transpose->mutable_shape()->mutable_layout() =
      LayoutUtil::MakeLayout({0, 1, 2, 3});

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/true,
                                 bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  // Verify that the reshape is replaced.
  EXPECT_EQ(2, computation->instruction_count());
  EXPECT_EQ(HloOpcode::kBitcast, computation->root_instruction()->opcode());
}

TEST_F(AlgebraicSimplifierTest, TransposeEqualsBitcast2) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {5, 2, 3, 4}), "param"));
  *param->mutable_shape()->mutable_layout() =
      LayoutUtil::MakeLayout({1, 2, 3, 0});

  HloInstruction* transpose =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {5, 3, 4, 2}), param, {0, 2, 3, 1}));
  *transpose->mutable_shape()->mutable_layout() =
      LayoutUtil::MakeLayout({3, 1, 2, 0});

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/true,
                                 bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  // Verify that the reshape is replaced.
  EXPECT_EQ(2, computation->instruction_count());
  EXPECT_EQ(HloOpcode::kBitcast, computation->root_instruction()->opcode());
}

TEST_F(AlgebraicSimplifierTest, ReshapesMerged) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 2}), "param0"));

  HloInstruction* reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {2, 1, 2}), param0));

  HloInstruction* reshape2 =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {1, 2, 1, 1, 2, 1}), reshape1));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(reshape2, computation->root_instruction());
  EXPECT_EQ(reshape1, computation->root_instruction()->operand(0));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_EQ(HloOpcode::kReshape, computation->root_instruction()->opcode());
  EXPECT_EQ(HloOpcode::kParameter,
            computation->root_instruction()->operand(0)->opcode());
}

TEST_F(AlgebraicSimplifierTest, TransposesMerged) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 3, 4}), "param0"));

  HloInstruction* transpose1 =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {3, 4, 2}), param0, {1, 2, 0}));

  HloInstruction* transpose2 =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {4, 3, 2}), transpose1, {1, 0, 2}));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(transpose2, computation->root_instruction());
  EXPECT_EQ(transpose1, computation->root_instruction()->operand(0));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_EQ(HloOpcode::kTranspose, computation->root_instruction()->opcode());
  EXPECT_EQ(std::vector<int64>({2, 1, 0}),
            computation->root_instruction()->dimensions());
  EXPECT_EQ(HloOpcode::kParameter,
            computation->root_instruction()->operand(0)->opcode());
}

// Test merging reshape and broadcast.
TEST_F(AlgebraicSimplifierTest, ReshapeAndBroadcastMerged) {
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {5}), "param0"));
  auto reshape1 = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {1, 5, 1}), param0));
  builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {1, 2, 3, 5, 1}), reshape1, {0, 2, 3}));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_EQ(HloOpcode::kBroadcast, computation->root_instruction()->opcode());
  EXPECT_EQ(HloOpcode::kParameter,
            computation->root_instruction()->operand(0)->opcode());
}

// Test merging broadcast and reshape.
TEST_F(AlgebraicSimplifierTest, BroadcastAndReshapeMerged) {
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {2, 3}), "param0"));
  auto broadcast1 = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {1, 2, 3, 7, 12, 1}), param0, {1, 2}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {2, 3, 7, 2, 1, 3, 2}), broadcast1));

  auto module = MakeUnique<HloModule>(TestName());
  auto computation = module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_EQ(HloOpcode::kBroadcast, computation->root_instruction()->opcode());
  EXPECT_EQ(HloOpcode::kParameter,
            computation->root_instruction()->operand(0)->opcode());
}

TEST_F(AlgebraicSimplifierTest, BroadcastAndReshape_1_3x1_3) {
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 1}), param, {1}));
  builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {3}), broadcast));

  auto module = MakeUnique<HloModule>(TestName());
  module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  EXPECT_FALSE(simplifier.Run(module.get()).ValueOrDie());
}

TEST_F(AlgebraicSimplifierTest, BroadcastAndReshape_4_3x2x4_6x1x1x4) {
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {4}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 2, 4}), param, {2}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 1, 1, 4}), broadcast));

  auto module = MakeUnique<HloModule>(TestName());
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_EQ(HloOpcode::kBroadcast, computation->root_instruction()->opcode());
  EXPECT_MATCH(computation->root_instruction()->dimensions(),
               testing::VectorMatcher<int64>({3}));
}

TEST_F(AlgebraicSimplifierTest, BroadcastAndReshape_1_3x2x1_6x1x1x1) {
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 2, 1}), param, {2}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 1, 1, 1}), broadcast));

  auto module = MakeUnique<HloModule>(TestName());
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_EQ(HloOpcode::kBroadcast, computation->root_instruction()->opcode());
  const std::vector<int64> broadcast_dims =
      computation->root_instruction()->dimensions();
  EXPECT_EQ(1, broadcast_dims.size());
  EXPECT_TRUE(broadcast_dims[0] == 1 || broadcast_dims[0] == 2 ||
              broadcast_dims[3] == 3);
}

TEST_F(AlgebraicSimplifierTest, BroadcastAndReshape_4_3x2x4x2_6x8) {
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {4}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 2, 4, 2}), param, {2}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 8}), broadcast));

  auto module = MakeUnique<HloModule>(TestName());
  module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  EXPECT_FALSE(simplifier.Run(module.get()).ValueOrDie());
}

TEST_F(AlgebraicSimplifierTest, RemoveNoopPad) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 2}), "param"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  PaddingConfig no_padding;
  for (int i = 0; i < 2; ++i) {
    auto dimension = no_padding.add_dimensions();
    dimension->set_edge_padding_low(0);
    dimension->set_edge_padding_high(0);
    dimension->set_interior_padding(0);
  }
  builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(F32, {2, 2}), param, zero, no_padding));

  HloModule module(TestName());
  HloComputation* computation = module.AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());
  EXPECT_EQ(1, computation->instruction_count());
}

TEST_F(AlgebraicSimplifierTest, NegativePadding) {
  // Verify that a pad instruction with negative padding is replaced with a
  // pad with non-negative padding followed by a slice.
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {10, 10}), "param"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  PaddingConfig padding;
  int64 low_padding[2] = {-1, -2};
  int64 high_padding[2] = {2, -3};
  for (int i = 0; i < 2; ++i) {
    auto dimension = padding.add_dimensions();
    dimension->set_edge_padding_low(low_padding[i]);
    dimension->set_edge_padding_high(high_padding[i]);
    dimension->set_interior_padding(0);
  }
  HloInstruction* pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(F32, {11, 5}), param, zero, padding));

  HloModule module(TestName());
  HloComputation* computation = module.AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());

  auto has_negative_padding = [](const HloInstruction* pad) {
    for (auto& padding_dimension : pad->padding_config().dimensions()) {
      if (padding_dimension.edge_padding_low() < 0 ||
          padding_dimension.edge_padding_high() < 0) {
        return true;
      }
    }
    return false;
  };

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_EQ(computation->root_instruction(), pad);
  EXPECT_TRUE(has_negative_padding(pad));

  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kSlice);
  const HloInstruction* root_operand =
      computation->root_instruction()->operand(0);
  EXPECT_EQ(root_operand->opcode(), HloOpcode::kPad);
  EXPECT_FALSE(has_negative_padding(root_operand));
}

TEST_F(AlgebraicSimplifierTest, RemoveNoopReshape) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 3}), "param"));
  builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {2, 3}), param));

  HloModule module(TestName());
  HloComputation* computation = module.AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());
  EXPECT_EQ(1, computation->instruction_count());
}

TEST_F(AlgebraicSimplifierTest, RemoveNoopSlice) {
  HloComputation::Builder builder(TestName());
  const int64 dim0 = 2;
  const int64 dim1 = 3;
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {dim0, dim1}), "param"));
  builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {dim0, dim1}), param, /*start_indices=*/{0, 0},
      /*limit_indices=*/{dim0, dim1}));

  HloModule module(TestName());
  HloComputation* computation = module.AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());
  EXPECT_EQ(1, computation->instruction_count());
}

TEST_F(AlgebraicSimplifierTest, ConvertConvToMatmul) {
  struct ConvTestOptions {
    int in_batch = 10;
    int in_height = 2;
    int in_width = 2;
    int in_channels = 3;
    int f_width = 1;
    int f_height = 1;
    int f_output_channels = 10;
    int row_stride = 1;
    int row_padding = 0;
    int col_stride = 1;
    int col_padding = 0;
    bool input_minor_to_major_layout = false;
    bool filter_minor_to_major_layout = false;
    bool output_minor_to_major_layout = false;

    const char* dim_order = "NHWC";         // can use chars NHWC in any order.
    const char* kernel_dim_order = "HWIO";  // can use chars HWIO in any order.

    ConvTestOptions& Reset() {
      *this = ConvTestOptions();
      return *this;
    }
  };

  ConvTestOptions options;

  // Builds a convolution from <options> and runs algebraic simplification on
  // the computation. Returns a string description of the result of
  // simplification.
  auto build_and_simplify = [&options, this]() -> string {
    HloComputation::Builder b(TestName());

    Window window;
    auto* f_dim_1 = window.add_dimensions();
    f_dim_1->set_size(options.f_height);
    f_dim_1->set_stride(options.row_stride);
    f_dim_1->set_padding_low(options.row_padding);
    f_dim_1->set_padding_high(options.row_padding);
    f_dim_1->set_window_dilation(1);
    f_dim_1->set_base_dilation(1);
    auto* f_dim_2 = window.add_dimensions();
    f_dim_2->set_size(options.f_width);
    f_dim_2->set_stride(options.col_stride);
    f_dim_2->set_padding_low(options.col_padding);
    f_dim_2->set_padding_high(options.col_padding);
    f_dim_2->set_window_dilation(1);
    f_dim_2->set_base_dilation(1);

    ConvolutionDimensionNumbers dnums;
    std::vector<int64> in_dims;
    int in_channel_idx = -1;
    dnums.add_spatial_dimensions(-1);  // filled in later
    dnums.add_spatial_dimensions(-1);  // filled in later
    for (int i = 0; i < strlen(options.dim_order); ++i) {
      char ch = options.dim_order[i];
      if (ch == 'N') {
        dnums.set_batch_dimension(i);
        in_dims.push_back(options.in_batch);
      } else if (ch == 'H') {
        dnums.set_spatial_dimensions(0, i);
        in_dims.push_back(options.in_height);
      } else if (ch == 'W') {
        dnums.set_spatial_dimensions(1, i);
        in_dims.push_back(options.in_width);
      } else if (ch == 'C') {
        dnums.set_feature_dimension(i);
        in_dims.push_back(options.in_channels);
        in_channel_idx = i;
      }
    }

    std::vector<int64> f_dims;
    dnums.add_kernel_spatial_dimensions(-1);  // filled in later
    dnums.add_kernel_spatial_dimensions(-1);  // filled in later
    for (int i = 0; i < strlen(options.kernel_dim_order); ++i) {
      char ch = options.kernel_dim_order[i];
      if (ch == 'H') {
        dnums.set_kernel_spatial_dimensions(0, i);
        f_dims.push_back(options.f_height);
      } else if (ch == 'W') {
        dnums.set_kernel_spatial_dimensions(1, i);
        f_dims.push_back(options.f_width);
      } else if (ch == 'I') {
        dnums.set_kernel_input_feature_dimension(i);
        f_dims.push_back(options.in_channels);
      } else if (ch == 'O') {
        dnums.set_kernel_output_feature_dimension(i);
        f_dims.push_back(options.f_output_channels);
      }
    }

    auto out_dims = in_dims;
    out_dims[in_channel_idx] = options.f_output_channels;

    auto make_shape = [](tensorflow::gtl::ArraySlice<int64> dims,
                         bool minor_to_major_layout) {
      if (minor_to_major_layout) {
        return ShapeUtil::MakeShapeWithLayout(F32, dims, {0, 1, 2, 3});
      } else {
        return ShapeUtil::MakeShape(F32, dims);
      }
    };
    auto in_shape = make_shape(in_dims, options.input_minor_to_major_layout);
    auto f_shape = make_shape(f_dims, options.filter_minor_to_major_layout);
    auto out_shape = make_shape(out_dims, options.output_minor_to_major_layout);

    HloInstruction* input =
        b.AddInstruction(HloInstruction::CreateParameter(0, in_shape, "input"));
    HloInstruction* filter =
        b.AddInstruction(HloInstruction::CreateParameter(1, f_shape, "filter"));

    b.AddInstruction(HloInstruction::CreateConvolve(out_shape, input, filter,
                                                    window, dnums));

    HloModule module(TestName());
    auto* computation = module.AddEntryComputation(b.Build());

    AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/true,
                                   bitcasting_callback());
    if (!simplifier.Run(&module).ValueOrDie()) {
      return "NO_CHANGE";
    }
    auto* root = computation->root_instruction();
    if (root->opcode() == HloOpcode::kBitcast &&
        root->operand(0)->opcode() == HloOpcode::kDot) {
      auto lhs_shape = root->operand(0)->operand(0)->shape();
      auto rhs_shape = root->operand(0)->operand(1)->shape();
      return tensorflow::strings::StrCat(
          tensorflow::str_util::Join(lhs_shape.dimensions(), "x"), " DOT ",
          tensorflow::str_util::Join(rhs_shape.dimensions(), "x"));
    }
    return "UNEXPECTED CHANGE";
  };

  // Default options are the simplest case and succeed.
  options.Reset();
  EXPECT_EQ("40x3 DOT 3x10", build_and_simplify());

  // Swapping dim spatial and batch order works.
  options.Reset().dim_order = "NWHC";
  EXPECT_EQ("40x3 DOT 3x10", build_and_simplify());
  options.Reset().dim_order = "WHNC";
  EXPECT_EQ("40x3 DOT 3x10", build_and_simplify());
  // Channel dimension earlier fails.
  options.Reset().dim_order = "HWCN";
  EXPECT_EQ("NO_CHANGE", build_and_simplify());
  options.Reset().dim_order = "CHWN";
  EXPECT_EQ("NO_CHANGE", build_and_simplify());

  // Filtering dims spatial dims can be anywhere, since they are 1x1.
  options.Reset().kernel_dim_order = "WHIO";
  EXPECT_EQ("40x3 DOT 3x10", build_and_simplify());
  options.Reset().kernel_dim_order = "IWOH";
  EXPECT_EQ("40x3 DOT 3x10", build_and_simplify());
  options.Reset().kernel_dim_order = "IWHO";
  EXPECT_EQ("40x3 DOT 3x10", build_and_simplify());
  // But moving output channel before input channel fails.
  options.Reset().kernel_dim_order = "HWOI";
  EXPECT_EQ("NO_CHANGE", build_and_simplify());
  options.Reset().kernel_dim_order = "WHOI";
  EXPECT_EQ("NO_CHANGE", build_and_simplify());
  options.Reset().kernel_dim_order = "OWIH";
  EXPECT_EQ("NO_CHANGE", build_and_simplify());
  options.Reset().kernel_dim_order = "OWHI";
  EXPECT_EQ("NO_CHANGE", build_and_simplify());

  // Combine different dim and kernel dim orders.
  options.Reset().kernel_dim_order = "IWHO";
  options.dim_order = "WHNC";
  EXPECT_EQ("40x3 DOT 3x10", build_and_simplify());

  // Test invalid cases from wrong filter size, strides, or padding.
  options.Reset().f_width = 2;
  EXPECT_EQ("NO_CHANGE", build_and_simplify());
  options.Reset().f_height = 2;
  EXPECT_EQ("NO_CHANGE", build_and_simplify());
  options.Reset().row_stride = 2;
  EXPECT_EQ("NO_CHANGE", build_and_simplify());
  options.Reset().col_stride = 2;
  EXPECT_EQ("NO_CHANGE", build_and_simplify());
  options.Reset().col_padding = 1;
  EXPECT_EQ("NO_CHANGE", build_and_simplify());
  options.Reset().row_padding = 1;
  EXPECT_EQ("NO_CHANGE", build_and_simplify());

  // The default dim_order is "NHWC". Col-major layout makes C the most major.
  options.Reset().input_minor_to_major_layout = true;
  options.output_minor_to_major_layout = true;
  EXPECT_EQ("NO_CHANGE", build_and_simplify());

  // The input and output have different layouts.
  options.Reset().input_minor_to_major_layout = true;
  EXPECT_EQ("NO_CHANGE", build_and_simplify());

  // C is most minor, and I is more major than O.
  options.Reset().input_minor_to_major_layout = true;
  options.filter_minor_to_major_layout = true;
  options.output_minor_to_major_layout = true;
  options.dim_order = "CHWN";
  options.kernel_dim_order = "OIHW";
  EXPECT_EQ("40x3 DOT 3x10", build_and_simplify());

  // C is not the most minor dimension.
  options.Reset().input_minor_to_major_layout = true;
  options.filter_minor_to_major_layout = true;
  options.output_minor_to_major_layout = true;
  options.dim_order = "HWNC";
  options.kernel_dim_order = "OIHW";
  EXPECT_EQ("NO_CHANGE", build_and_simplify());

  // I is more minor than O.
  options.Reset().input_minor_to_major_layout = true;
  options.filter_minor_to_major_layout = true;
  options.output_minor_to_major_layout = true;
  options.dim_order = "CHWN";
  options.kernel_dim_order = "IOHW";
  EXPECT_EQ("NO_CHANGE", build_and_simplify());
}

// Test that max(min(A, x), y) is transformed to clamp(y, A, x)
TEST_F(AlgebraicSimplifierTest, MaxMinToClamp) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* min_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  HloInstruction* max_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  HloInstruction* min = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kMinimum, param0, min_value));
  HloInstruction* max = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kMaximum, min, max_value));

  HloModule module(TestName());
  auto computation = module.AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, max);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());
  root = computation->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kClamp);
  EXPECT_EQ(root->operand(0), max_value);
  EXPECT_EQ(root->operand(1), param0);
  EXPECT_EQ(root->operand(2), min_value);
}

// Test that min(max(A, x), y) is transformed to clamp(x, A, y) for scalar
// values.
TEST_F(AlgebraicSimplifierTest, MinMaxToClamp) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* min_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  HloInstruction* max_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  HloInstruction* max = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kMaximum, param0, max_value));
  HloInstruction* min = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kMinimum, max, min_value));

  HloModule module(TestName());
  auto computation = module.AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, min);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kClamp);
  EXPECT_EQ(root->operand(0), max_value);
  EXPECT_EQ(root->operand(1), param0);
  EXPECT_EQ(root->operand(2), min_value);
}

// Test that min(max(A, x), y) is transformed to clamp(x, A, y) for
// broadcasted scalar values.
TEST_F(AlgebraicSimplifierTest, MinMaxWithBroadcastToClamp) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  Shape r1f32 = ShapeUtil::MakeShape(F32, {100});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* min_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  HloInstruction* max_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  HloInstruction* max = builder.AddInstruction(HloInstruction::CreateBinary(
      r1f32, HloOpcode::kMaximum, param0, max_value));
  HloInstruction* min = builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kMinimum, max, min_value));

  HloModule module(TestName());
  auto computation = module.AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, min);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kClamp);
  EXPECT_EQ(root->operand(0), max_value);
  EXPECT_EQ(root->operand(1), param0);
  EXPECT_EQ(root->operand(2), min_value);
}

// Test that min(max(A, non-constant1), non-constant2) is not canonicalized to
// clamp(non-constant1, A, non-constant2)
TEST_F(AlgebraicSimplifierTest, MinMaxNotToClamp) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* min_value = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* max_value = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r0f32, "param2"));
  HloInstruction* max = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kMaximum, param0, max_value));
  HloInstruction* min = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kMinimum, max, min_value));

  HloModule module(TestName());
  auto computation = module.AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  EXPECT_FALSE(simplifier.Run(&module).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, min);
}

// Test that min(f(max(A, constant1)), constant2) is not transformed to
// clamp(constant1, A, constant2)
TEST_F(AlgebraicSimplifierTest, MinEquationWithMaxNotToClamp) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* min_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  HloInstruction* max_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  HloInstruction* max = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kMaximum, param0, max_value));
  HloInstruction* fmax = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, max, max_value));
  HloInstruction* min = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kMinimum, fmax, min_value));

  HloModule module(TestName());
  auto computation = module.AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, min);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  EXPECT_FALSE(simplifier.Run(&module).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, min);
}

// Test that slice(broadcast(/*scalar value*/)) simplifies to a single
// broadcast.
TEST_F(AlgebraicSimplifierTest, ScalarBroadcastToSlice) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* scalar_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "scalar_param"));

  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {4, 5, 6, 7});
  HloInstruction* broadcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          broadcast_shape, scalar_param,
          AsInt64Slice(broadcast_shape.dimensions())));

  Shape slice_shape = ShapeUtil::MakeShape(F32, {2, 2, 3, 3});
  HloInstruction* slice = builder.AddInstruction(HloInstruction::CreateSlice(
      slice_shape, broadcast, {0, 1, 2, 3}, {2, 3, 5, 6}));

  HloModule module(TestName());
  auto computation = module.AddEntryComputation(builder.Build());

  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, slice);
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), slice_shape));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());

  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());

  // Running simplification again should not result in any further changes.
  ASSERT_FALSE(simplifier.Run(&module).ValueOrDie());

  root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(scalar_param, root->operand(0));
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), slice_shape));
}

// Test that reshape(transpose(broadcast(/*scalar value*/))) simplifies to a
// single broadcast.
TEST_F(AlgebraicSimplifierTest, ScalarBroadcastToTransposeReshape) {
  HloComputation::Builder builder(TestName());
  HloInstruction* forty_two = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));

  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {4, 5, 6});
  HloInstruction* broadcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          broadcast_shape, forty_two,
          AsInt64Slice(broadcast_shape.dimensions())));

  HloInstruction* transpose =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {6, 5, 4}), broadcast, {2, 1, 0}));

  Shape reshape_shape = ShapeUtil::MakeShape(F32, {30, 1, 4});
  HloInstruction* reshape = builder.AddInstruction(
      HloInstruction::CreateReshape(reshape_shape, transpose));

  HloModule module(TestName());
  auto computation = module.AddEntryComputation(builder.Build());

  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, reshape);
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), reshape_shape));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());

  root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(forty_two, root->operand(0));
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), reshape_shape));
}

TEST_F(AlgebraicSimplifierTest, ReversalOfTrivialDimensionsToBitcast) {
  HloComputation::Builder builder(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {448, 2048, 1, 1});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  builder.AddInstruction(
      HloInstruction::CreateReverse(shape, a, /*dimensions=*/{2, 3}));

  HloModule module(TestName());
  auto computation = module.AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(a, root);
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), shape));
}

}  // namespace
}  // namespace xla

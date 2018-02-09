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
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

AlgebraicSimplifier::ValidBitcastCallback bitcasting_callback() {
  return [](const Shape&, const Shape&) { return true; };
}

AlgebraicSimplifier::ValidBitcastCallback non_bitcasting_callback() {
  return [](const Shape&, const Shape&) { return false; };
}

class AlgebraicSimplifierTest : public HloVerifiedTestBase {};

// Test that A + 0 is simplified to A
TEST_F(AlgebraicSimplifierTest, AddZero) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0f)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, param0, zero));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that Const + A is canonicalized to A + Const.
TEST_F(AlgebraicSimplifierTest, AddConstOnLHS) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0(42.0f)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, constant, param0));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, op::Add(param0, op::Constant()));
}

// Test that [(A + C1) + C2] => [A + (C1 + C2)] for constants C1 and C2.
TEST_F(AlgebraicSimplifierTest, AddReassociateMergeConstants) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0(42.0f)));
  HloInstruction* constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0(3.14159f)));

  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, param0, constant1));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, add1, constant2));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, op::Add(param0, op::Add(constant1, constant2)));
}

TEST_F(AlgebraicSimplifierTest, AddBroadcastZeroR0Operand) {
  Shape r2f32 = ShapeUtil::MakeShape(F32, {3, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0f)));
  HloInstruction* bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(r2f32, zero, {0, 1}));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kAdd, bcast, param0));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

TEST_F(AlgebraicSimplifierTest, AddBroadcastZeroR1Operand) {
  Shape r2f32 = ShapeUtil::MakeShape(F32, {3, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<float>({0, 0, 0})));
  HloInstruction* bcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(r2f32, zero, {1}));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kAdd, bcast, param0));

  auto module = CreateNewModule();
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
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0f)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kSubtract, param0, zero));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSubtract);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that A - Const is canonicalized to A + (-Const).
TEST_F(AlgebraicSimplifierTest, SubConstCanonicalization) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kSubtract, param0, constant));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSubtract);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, op::Add(param0, op::Negate(constant)));
}

// Test that (A/B)/C is simplified to A/(B*C).
TEST_F(AlgebraicSimplifierTest, LhsDivOfDiv) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r0f32, "param2"));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, param0, param1));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, div, param2));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Divide(op::Divide(param0, param1), param2));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Divide(param0, op::Multiply(param1, param2)));
}

// Test that A/(B/C) is simplified to (A*C)/B.
TEST_F(AlgebraicSimplifierTest, RhsDivOfDiv) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r0f32, "param2"));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, param1, param2));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, param0, div));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Divide(param0, op::Divide(param1, param2)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Divide(op::Multiply(param0, param2), param1));
}

// Test that (A/B)/(C/D) is simplified to (A*D)/(B*C).
TEST_F(AlgebraicSimplifierTest, DivOfDivAndDiv) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  Shape r2f32 = ShapeUtil::MakeShape(F32, {42, 123});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r2f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r2f32, "param2"));
  HloInstruction* param3 = builder.AddInstruction(
      HloInstruction::CreateParameter(3, r0f32, "param3"));
  HloInstruction* div0 = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kDivide, param0, param1));
  HloInstruction* div1 = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kDivide, param2, param3));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kDivide, div0, div1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      op::Divide(op::Divide(param0, param1), op::Divide(param2, param3)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      op::Divide(op::Multiply(param0, param3), op::Multiply(param1, param2)));
  EXPECT_TRUE(
      ShapeUtil::Compatible(computation->root_instruction()->shape(), r2f32));
}

// Test that A/exp(B) is simplified to A*exp(-B).
TEST_F(AlgebraicSimplifierTest, DivOfExp) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param1));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, param0, exp));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Divide(param0, op::Exp(param1)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Multiply(param0, op::Exp(op::Negate(param1))));
}

// Test that A/pow(B,C) is simplified to A*pow(B,-C).
TEST_F(AlgebraicSimplifierTest, DivOfPower) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r0f32, "param2"));
  HloInstruction* power = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param1, param2));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, param0, power));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Divide(param0, op::Power(param1, param2)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Multiply(param0, op::Power(param1, op::Negate(param2))));
}

// Test that broadcasting is done on the right step when simplifying A/pow(B,C)
// to A*pow(B,-C).
TEST_F(AlgebraicSimplifierTest, DivOfBroadcastingPower) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  Shape r1f32 = ShapeUtil::MakeShape(F32, {7});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r1f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r0f32, "param2"));
  HloInstruction* power = builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kPower, param1, param2));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kDivide, param0, power));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Divide(param0, op::Power(param1, param2)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  ASSERT_THAT(computation->root_instruction(),
              op::Multiply(param0, op::Power(param1, op::Negate(param2))));

  const HloInstruction* negate =
      computation->root_instruction()->operand(1)->operand(1);
  const Shape& negate_shape = negate->shape();
  EXPECT_EQ(0, negate_shape.dimensions_size());
}

// A / Const => A * (1 / Const)
TEST_F(AlgebraicSimplifierTest, DivideByConstant) {
  Shape r1f32 = ShapeUtil::MakeShape(F32, {3});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* constant =
      builder.AddInstruction(HloInstruction::CreateConstant(
          Literal::CreateR1<float>({0.f, 1.f, 2.f})));
  builder.AddInstruction(HloInstruction::CreateBinary(r1f32, HloOpcode::kDivide,
                                                      param0, constant));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Multiply(param0, op::Divide(op::Constant(), constant)));
}

// pow(pow(A, X), Y) => pow(A, X*Y)
TEST_F(AlgebraicSimplifierTest, PowerOfPower) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  Shape r1f32 = ShapeUtil::MakeShape(F32, {7});
  HloComputation::Builder builder(TestName());
  HloInstruction* base = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* exp2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r0f32, "param2"));
  HloInstruction* inner_power = builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kPower, base, exp1));
  builder.AddInstruction(HloInstruction::CreateBinary(r1f32, HloOpcode::kPower,
                                                      inner_power, exp2));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              op::Power(base, op::Multiply(exp1, exp2)));
}

// Don't simplify pow(pow(A, X), Y) => pow(A, X*Y) if X and Y are complex
// numbers.
TEST_F(AlgebraicSimplifierTest, PowerOfPowerComplex) {
  Shape r0c64 = ShapeUtil::MakeShape(C64, {});
  Shape r1f32 = ShapeUtil::MakeShape(F32, {7});
  HloComputation::Builder builder(TestName());
  HloInstruction* base = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0c64, "param1"));
  HloInstruction* exp2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r0c64, "param2"));
  HloInstruction* inner_power = builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kPower, base, exp1));
  builder.AddInstruction(HloInstruction::CreateBinary(r1f32, HloOpcode::kPower,
                                                      inner_power, exp2));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_FALSE(simplifier.Run(module.get()).ValueOrDie());
}

// Test that A/1 is simplified to A for a scalar.
TEST_F(AlgebraicSimplifierTest, DivOneScalar) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0f)));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, param0, one));

  auto module = CreateNewModule();
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
      Literal::CreateR2<float>({{1.0, 1.0}, {1.0, 1.0}})));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kDivide, param0, one));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, div);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that complex(real(c), imag(c)) is simplified to c.
TEST_F(AlgebraicSimplifierTest, ComplexOfRealImagC) {
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 2});
  Shape r2c64 = ShapeUtil::MakeShape(C64, {2, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2c64, "param0"));
  HloInstruction* real = builder.AddInstruction(
      HloInstruction::CreateUnary(r2f32, HloOpcode::kReal, param0));
  HloInstruction* imag = builder.AddInstruction(
      HloInstruction::CreateUnary(r2f32, HloOpcode::kImag, param0));
  HloInstruction* cplx = builder.AddInstruction(
      HloInstruction::CreateBinary(r2c64, HloOpcode::kComplex, real, imag));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, cplx);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that real(complex(r,i)) is simplified to r.
TEST_F(AlgebraicSimplifierTest, RealOfComplex) {
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r2f32, "param1"));
  HloInstruction* cplx = builder.AddInstruction(
      HloInstruction::CreateBinary(ShapeUtil::ChangeElementType(r2f32, C64),
                                   HloOpcode::kComplex, param0, param1));
  HloInstruction* real = builder.AddInstruction(
      HloInstruction::CreateUnary(r2f32, HloOpcode::kReal, cplx));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, real);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that imag(complex(r,i)) is simplified to i.
TEST_F(AlgebraicSimplifierTest, ImagOfComplex) {
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r2f32, "param1"));
  HloInstruction* cplx = builder.AddInstruction(
      HloInstruction::CreateBinary(ShapeUtil::ChangeElementType(r2f32, C64),
                                   HloOpcode::kComplex, param0, param1));
  HloInstruction* imag = builder.AddInstruction(
      HloInstruction::CreateUnary(r2f32, HloOpcode::kImag, cplx));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, imag);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param1);
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, add);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, op::Add(param1, param2));
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Divide(op::Exp(param0), op::Exp(param1)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Exp(op::Subtract(param0, param1)));
}

// Test that exp(A)*exp(B) is simplified to exp(A+B)
TEST_F(AlgebraicSimplifierTest, ExpMul) {
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
      HloInstruction::CreateBinary(r0f32, HloOpcode::kMultiply, exp0, exp1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Multiply(op::Exp(param0), op::Exp(param1)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Exp(op::Add(param0, param1)));
}

// Test that pow(exp(A), B) is simplified to exp(A*B)
TEST_F(AlgebraicSimplifierTest, PowExp) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* exp0 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param0));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, exp0, param1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Power(op::Exp(param0), param1));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Exp(op::Multiply(param0, param1)));
}

// Test that ln(pow(A, B)) is simplified to ln(A)*B
TEST_F(AlgebraicSimplifierTest, LnPow) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* pow = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, param1));
  builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kLog, pow));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Log(op::Power(param0, param1)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Multiply(op::Log(param0), param1));
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Log(op::Exp(param0)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_EQ(computation->root_instruction(), param0);
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Log(op::Divide(op::Exp(param0), op::Exp(param1))));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), op::Subtract(param0, param1));
}

// Test that pow(A, 0) where A is a scalar is simplified to the scalar
// constant 1.
TEST_F(AlgebraicSimplifierTest, Pow0Scalar) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, zero));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Power(param0, zero));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Constant());
  EXPECT_EQ(root->literal().GetFirstElement<float>(), 1);
}

// Test that pow(A, 0) where A is not a scalar is simplified to broadcast(1).
TEST_F(AlgebraicSimplifierTest, Pow0Vector) {
  Shape r1f32 = ShapeUtil::MakeShape(F32, {42});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kPower, param0, zero));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Power(param0, zero));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Broadcast());
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), r1f32))
      << ShapeUtil::HumanString(root->shape());
  EXPECT_EQ(root->dimensions().size(), 0);
  EXPECT_TRUE(ShapeUtil::IsScalar(root->operand(0)->shape()));
  EXPECT_EQ(root->operand(0)->literal().GetFirstElement<float>(), 1);
}

// Test that pow(A, 1) is simplified to A.
TEST_F(AlgebraicSimplifierTest, Pow1) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, one));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Power(param0, one));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_EQ(computation->root_instruction(), param0);
}

// Test that pow(A, 2) is simplified to A*A.
TEST_F(AlgebraicSimplifierTest, Pow2) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* two = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, two));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Power(param0, two));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), op::Multiply(param0, param0));
}

// Test that pow(A, -1) is simplified to 1/A.
TEST_F(AlgebraicSimplifierTest, PowNegative1) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* negative_one = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(-1)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0f32, HloOpcode::kPower,
                                                      param0, negative_one));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Power(param0, negative_one));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, op::Divide(op::Broadcast(), param0));
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(root->operand(0)->operand(0)->literal().GetFirstElement<float>(),
            1);
}

TEST_F(AlgebraicSimplifierTest, ZeroSizedConvolution) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {3, 3, 0}), "lhs"));

  HloInstruction* rhs = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {3, 0, 3}), "rhs"));

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.set_input_feature_dimension(2);

  dnums.set_output_batch_dimension(0);
  dnums.add_output_spatial_dimensions(1);
  dnums.set_output_feature_dimension(2);

  dnums.add_kernel_spatial_dimensions(0);
  dnums.set_kernel_input_feature_dimension(1);
  dnums.set_kernel_output_feature_dimension(2);
  Window window;
  WindowDimension* dim = window.add_dimensions();
  dim->set_size(3);
  dim->set_padding_low(0);
  dim->set_padding_high(0);
  dim->set_stride(1);
  dim->set_window_dilation(1);
  dim->set_base_dilation(1);
  dim->set_window_reversal(false);
  // Create add computation.
  std::unique_ptr<HloModule> module = CreateNewModule();
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {3, 3, 3}), lhs, rhs, window, dnums));
  module->AddEntryComputation(builder.Build());
  HloPassFix<AlgebraicSimplifier> simplifier(/*is_layout_sensitive=*/false,
                                             non_bitcasting_callback());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Convolution(lhs, rhs));
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Broadcast(op::Constant()));
}

TEST_F(AlgebraicSimplifierTest, ZeroSizedReduceWindow) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {3, 0}), "op"));
  Window window;
  for (int64 i = 0; i < 2; ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_size(1);
    dim->set_padding_low(1);
    dim->set_padding_high(1);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  // Create add computation.
  std::unique_ptr<HloModule> module = CreateNewModule();
  HloComputation* add_computation = nullptr;
  {
    HloComputation::Builder builder(TestName() + ".add");
    const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
    HloInstruction* p0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* p1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "p1"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, p0, p1));
    add_computation = module->AddEmbeddedComputation(builder.Build());
  }
  builder.AddInstruction(HloInstruction::CreateReduceWindow(
      ShapeUtil::MakeShape(F32, {5, 2}), param,
      builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0f))),
      window, add_computation));
  module->AddEntryComputation(builder.Build());
  HloPassFix<AlgebraicSimplifier> simplifier(/*is_layout_sensitive=*/false,
                                             non_bitcasting_callback());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::ReduceWindow(param, op::Constant()));
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Broadcast(op::Constant()));
}

TEST_F(AlgebraicSimplifierTest, ZeroSizedPad) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {3, 0}), "op"));
  PaddingConfig padding;
  for (int i = 0; i < 2; ++i) {
    PaddingConfig::PaddingConfigDimension* dimension = padding.add_dimensions();
    dimension->set_edge_padding_low(1);
    dimension->set_edge_padding_high(1);
    dimension->set_interior_padding(0);
  }
  builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(F32, {5, 2}), param,
      builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0(0.0f))),
      padding));
  std::unique_ptr<HloModule> module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Pad(param, op::Constant()));
  HloPassFix<AlgebraicSimplifier> simplifier(/*is_layout_sensitive=*/false,
                                             non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Broadcast(op::Constant()));
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
  auto module = CreateNewModule();
  module->AddEntryComputation(std::move(computation));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Reshape(op::Broadcast(op::Reshape(op))));

  HloPassFix<AlgebraicSimplifier> simplifier(/*is_layout_sensitive=*/false,
                                             non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(module->entry_computation()->root_instruction(), op);
}

// Test that convert(A, $TYPE) is simplified to A if A is of type $TYPE.
TEST_F(AlgebraicSimplifierTest, ConvertBetweenSameType) {
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(F32, {}), input));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Convert(input));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), input);
}

// Test that copies are removed.
TEST_F(AlgebraicSimplifierTest, RemoveCopy) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  builder.AddInstruction(
      HloInstruction::CreateUnary(param0->shape(), HloOpcode::kCopy, param0));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Copy(param0));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), param0);
}

// Test that unary concatenates are removed.
TEST_F(AlgebraicSimplifierTest, RemoveUnaryConcatenate) {
  Shape r1f32 = ShapeUtil::MakeShape(F32, {100});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  builder.AddInstruction(
      HloInstruction::CreateConcatenate(param0->shape(), {param0}, 0));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Concatenate(param0));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), param0);
}

// Test that empty operands of concatenates are removed.
TEST_F(AlgebraicSimplifierTest, RemoveEmptyConcatenateOperands) {
  const int kParamLength = 100;
  Shape r1f32 = ShapeUtil::MakeShape(F32, {kParamLength});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r1f32, "param1"));
  HloInstruction* empty_literal = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<float>({})));
  HloInstruction* empty_slice =
      builder.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {0}), param1, {42}, {42}, {1}));
  Shape result_shape = ShapeUtil::MakeShape(F32, {3 * kParamLength});
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      result_shape, {empty_literal, param0, param0, empty_slice, param1}, 0));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      op::Concatenate(empty_literal, param0, param0, empty_slice, param1));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Concatenate(param0, param0, param1));
}

// Test a concatenate with only empty operands is removed.
TEST_F(AlgebraicSimplifierTest, OnlyEmptyConcatenateOperands) {
  const int kParamLength = 100;
  Shape r1f32 = ShapeUtil::MakeShape(F32, {kParamLength});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* empty_literal = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<float>({})));
  HloInstruction* empty_slice =
      builder.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {0}), param0, {42}, {42}, {1}));
  Shape result_shape = ShapeUtil::MakeShape(F32, {0});
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      result_shape, {empty_literal, empty_slice}, 0));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Concatenate(empty_literal, empty_slice));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_EQ(computation->root_instruction(), empty_literal);
}

// Test that concat with a scalar broadcast becomes a pad.
TEST_F(AlgebraicSimplifierTest, ConcatenateOfBroadcastBecomesPad) {
  Shape r1f32 = ShapeUtil::MakeShape(F32, {100});
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(r1f32, param1, {}));
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      param0->shape(), {broadcast, param0}, 0));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), op::Pad(param0, param1));
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Set to different layouts.
  *param0->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  *copy->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({1, 0});

  EXPECT_THAT(computation->root_instruction(), op::Copy(param0));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/true,
                                 non_bitcasting_callback());
  EXPECT_FALSE(simplifier.Run(module.get()).ValueOrDie());

  // Copy has not been removed.
  EXPECT_THAT(computation->root_instruction(), op::Copy(param0));
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Set to same layouts.
  *param0->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  *copy->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  EXPECT_THAT(computation->root_instruction(), op::Copy(param0));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/true,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  // Copy has been removed.
  EXPECT_THAT(computation->root_instruction(), param0);
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Reshape(param0));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/true,
                                 non_bitcasting_callback());
  EXPECT_FALSE(simplifier.Run(module.get()).ValueOrDie());

  // Reshape is not replaced with a bitcast.
  EXPECT_THAT(computation->root_instruction(), op::Reshape(param0));
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Tuple(transformable_reshape, dimensions_wrong_reshape,
                        layout_wrong_reshape));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/true,
                                 bitcasting_callback());
  simplifier.Run(module.get()).ValueOrDie();

  // Verify that only the first reshape is replaced.
  EXPECT_THAT(
      computation->root_instruction(),
      op::Tuple(op::Bitcast(), dimensions_wrong_reshape, layout_wrong_reshape));
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
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0f)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(ShapeUtil::MakeShape(F32, {1, 2, 3, 4, 5}),
                                   HloOpcode::kMaximum, movable_reshape, zero));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Maximum(op::Reshape(param), zero));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 bitcasting_callback());

  simplifier.Run(module.get()).ValueOrDie();
  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Maximum(param, zero)));
}

// Regression test for a bug in the reshape sinking transformation, where
// moving a reshape to a scalar led to a crash.
TEST_F(AlgebraicSimplifierTest, ReshapeToScalarNotHoistedAfterEffectiveUnary) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 1}), "param"));
  HloInstruction* reshape = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {}), param));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<float>({1., 2., 3.})));
  builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {3}), HloOpcode::kMaximum, reshape, zero));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Maximum(op::Reshape(param), zero));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 bitcasting_callback());

  simplifier.Run(module.get()).ValueOrDie();

  EXPECT_THAT(computation->root_instruction(),
              op::Maximum(op::Reshape(param), zero));
}

// Regression test for a bug where if we failed to sink a reshape, we'd set the
// 'changed' bit in AlgebraicSimplifier to false.
TEST_F(AlgebraicSimplifierTest, FailureToSinkReshapeDoesntAffectChangedBit) {
  HloComputation::Builder builder(TestName());

  // This add (param0 + 0) can be simplified.
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd,
      builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape, "param0")),
      builder.AddInstruction(HloInstruction::CreateConstant(
          Literal::CreateR2<float>({{0, 0}, {0, 0}})))));

  builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {4}), add));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 bitcasting_callback());
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
}

// Regression test for a bug where if we failed to sink a reshape, we'd set the
// 'changed' bit in AlgebraicSimplifier to false.
TEST_F(AlgebraicSimplifierTest, FailureToSinkBroadcastDoesntAffectChangedBit) {
  HloComputation::Builder builder(TestName());

  // This add (param0 + 0) can be simplified.
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd,
      builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape, "param0")),
      builder.AddInstruction(HloInstruction::CreateConstant(
          Literal::CreateR2<float>({{0, 0}, {0, 0}})))));

  builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {2, 2, 2}), add, /*broadcast_dimensions=*/{0}));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 bitcasting_callback());
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Transpose(param));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/true,
                                 bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  // Verify that the reshape is replaced.
  EXPECT_THAT(computation->root_instruction(), op::Bitcast(param));
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Transpose(param));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/true,
                                 bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  // Verify that the reshape is replaced.
  EXPECT_THAT(computation->root_instruction(), op::Bitcast(param));
}

TEST_F(AlgebraicSimplifierTest, ReshapesMerged) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 2}), "param0"));

  HloInstruction* reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {2, 1, 2}), param0));

  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {1, 2, 1, 1, 2, 1}), reshape1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Reshape(param0)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), op::Reshape(param0));
}

TEST_F(AlgebraicSimplifierTest, CopiesMerged) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShapeWithDescendingLayout(F32, {2, 2, 2}),
          "param0"));

  HloInstruction* copy1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShapeWithLayout(F32, {2, 2, 2}, {0, 1, 2}),
      HloOpcode::kCopy, param0));

  builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShapeWithLayout(F32, {2, 2, 2}, {0, 2, 1}),
      HloOpcode::kCopy, copy1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Copy(op::Copy(param0)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/true,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), op::Copy(param0));
}

TEST_F(AlgebraicSimplifierTest, TransposesMerged) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 3, 4}), "param0"));

  HloInstruction* transpose1 =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {3, 4, 2}), param0, {1, 2, 0}));

  builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 3, 2}), transpose1, {1, 0, 2}));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Transpose(transpose1));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), op::Transpose(param0));
  EXPECT_EQ(std::vector<int64>({2, 1, 0}),
            computation->root_instruction()->dimensions());
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Broadcast(op::Reshape(param0)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), op::Broadcast(param0));
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Broadcast(param0)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), op::Broadcast(param0));
}

TEST_F(AlgebraicSimplifierTest, BroadcastAndReshape_1_3x1_3) {
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 1}), param, {1}));
  builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {3}), broadcast));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Broadcast(param)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  EXPECT_FALSE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Broadcast(param)));
}

TEST_F(AlgebraicSimplifierTest, BroadcastAndReshape_4_3x2x4_6x1x1x4) {
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {4}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 2, 4}), param, {2}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 1, 1, 4}), broadcast));

  auto module = CreateNewModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Broadcast(param)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), op::Broadcast(param));
  EXPECT_THAT(computation->root_instruction()->dimensions(),
              ::testing::ElementsAre(3));
}

TEST_F(AlgebraicSimplifierTest, BroadcastAndReshape_1_3x2x1_6x1x1x1) {
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 2, 1}), param, {2}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 1, 1, 1}), broadcast));

  auto module = CreateNewModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Broadcast(param)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), op::Broadcast(param));
  const std::vector<int64> broadcast_dims =
      computation->root_instruction()->dimensions();
  EXPECT_EQ(1, broadcast_dims.size());
  EXPECT_THAT(broadcast_dims[0], ::testing::AnyOf(1, 2, 3));
}

TEST_F(AlgebraicSimplifierTest, BroadcastAndReshape_4_3x2x4x2_6x8) {
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {4}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 2, 4, 2}), param, {2}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 8}), broadcast));

  auto module = CreateNewModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Broadcast(param)));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  EXPECT_FALSE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Reshape(op::Broadcast(param)));
}

TEST_F(AlgebraicSimplifierTest, RemoveNoopPad) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 2}), "param"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0f)));
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

  EXPECT_THAT(computation->root_instruction(), op::Pad(param, zero));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), param);
}

TEST_F(AlgebraicSimplifierTest, NegativePadding) {
  // Verify that a pad instruction with negative padding is replaced with a
  // pad with non-negative padding followed by a slice.
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {10, 10}), "param"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0f)));
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

  EXPECT_THAT(computation->root_instruction(), op::Pad(param, zero));
  EXPECT_TRUE(has_negative_padding(pad));

  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), op::Slice(op::Pad(param, zero)));
  EXPECT_FALSE(
      has_negative_padding(computation->root_instruction()->operand(0)));
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

  EXPECT_THAT(computation->root_instruction(), op::Reshape(param));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), param);
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
      /*limit_indices=*/{dim0, dim1}, /*strides=*/{1, 1}));

  HloModule module(TestName());
  HloComputation* computation = module.AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), op::Slice(param));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), param);
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
    // filled in later
    dnums.add_input_spatial_dimensions(-1);
    dnums.add_output_spatial_dimensions(-1);
    dnums.add_input_spatial_dimensions(-1);
    dnums.add_output_spatial_dimensions(-1);
    for (int i = 0; i < strlen(options.dim_order); ++i) {
      char ch = options.dim_order[i];
      if (ch == 'N') {
        dnums.set_input_batch_dimension(i);
        dnums.set_output_batch_dimension(i);
        in_dims.push_back(options.in_batch);
      } else if (ch == 'H') {
        dnums.set_input_spatial_dimensions(0, i);
        dnums.set_output_spatial_dimensions(0, i);
        in_dims.push_back(options.in_height);
      } else if (ch == 'W') {
        dnums.set_input_spatial_dimensions(1, i);
        dnums.set_output_spatial_dimensions(1, i);
        in_dims.push_back(options.in_width);
      } else if (ch == 'C') {
        dnums.set_input_feature_dimension(i);
        dnums.set_output_feature_dimension(i);
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
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0f)));
  HloInstruction* max_value = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0f)));
  HloInstruction* min = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kMinimum, param0, min_value));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kMaximum, min, max_value));

  HloModule module(TestName());
  auto computation = module.AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Maximum(op::Minimum(param0, min_value), max_value));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Clamp(max_value, param0, min_value));
}

// Test that min(max(A, x), y) is transformed to clamp(x, A, y) for scalar
// values.
TEST_F(AlgebraicSimplifierTest, MinMaxToClamp) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* min_value = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0f)));
  HloInstruction* max_value = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0f)));
  HloInstruction* max = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kMaximum, param0, max_value));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kMinimum, max, min_value));

  HloModule module(TestName());
  auto computation = module.AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Minimum(op::Maximum(param0, max_value), min_value));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Clamp(max_value, param0, min_value));
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
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0f)));
  HloInstruction* max_value = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0f)));
  HloInstruction* max = builder.AddInstruction(HloInstruction::CreateBinary(
      r1f32, HloOpcode::kMaximum, param0, max_value));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kMinimum, max, min_value));

  HloModule module(TestName());
  auto computation = module.AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Minimum(op::Maximum(param0, max_value), min_value));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Clamp(max_value, param0, min_value));
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
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kMinimum, max, min_value));

  HloModule module(TestName());
  auto computation = module.AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Minimum(op::Maximum(param0, max_value), min_value));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  EXPECT_FALSE(simplifier.Run(&module).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Minimum(op::Maximum(param0, max_value), min_value));
}

// Test that min(f(max(A, constant1)), constant2) is not transformed to
// clamp(constant1, A, constant2)
TEST_F(AlgebraicSimplifierTest, MinEquationWithMaxNotToClamp) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* min_value = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0f)));
  HloInstruction* max_value = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(1.0f)));
  HloInstruction* max = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kMaximum, param0, max_value));
  HloInstruction* fmax = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, max, max_value));
  builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kMinimum, fmax, min_value));

  HloModule module(TestName());
  auto computation = module.AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              op::Minimum(op::Add(op::Maximum(param0, max_value), max_value),
                          min_value));

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  EXPECT_FALSE(simplifier.Run(&module).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              op::Minimum(op::Add(op::Maximum(param0, max_value), max_value),
                          min_value));
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
      slice_shape, broadcast, {0, 1, 2, 3}, {2, 3, 5, 6}, {1, 1, 1, 1}));

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
  EXPECT_THAT(root, op::Broadcast(scalar_param));
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), slice_shape));
}

// Test that reshape(transpose(broadcast(/*scalar value*/))) simplifies to a
// single broadcast.
TEST_F(AlgebraicSimplifierTest, ScalarBroadcastToTransposeReshape) {
  HloComputation::Builder builder(TestName());
  HloInstruction* forty_two = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0f)));

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
  EXPECT_THAT(root, op::Broadcast(forty_two));
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), reshape_shape));
}

// Test that ReduceWindow(Pad(op, x), y) can simplify to ReduceWindow(op, x).
TEST_F(AlgebraicSimplifierTest, FoldPadIntoReduceWindow) {
  HloModule module(TestName());
  HloComputation::Builder builder(TestName());

  // Create operand to the pad.
  HloInstruction* operand =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 2, 3, 4}), "p0"));

  // Create the pad.
  PaddingConfig padding = MakeNoPaddingConfig(4);
  padding.mutable_dimensions(1)->set_edge_padding_low(1);
  padding.mutable_dimensions(3)->set_edge_padding_high(2);

  HloInstruction* pad_value = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(5.0f)));
  HloInstruction* pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(F32, {1, 3, 3, 5}), operand, pad_value, padding));

  // Create add computation.
  HloComputation* add_computation = nullptr;
  {
    HloComputation::Builder builder(TestName() + ".add");
    const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
    HloInstruction* p0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* p1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "p1"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, p0, p1));
    add_computation = module.AddEmbeddedComputation(builder.Build());
  }

  // Create the reduce-window.
  Window window;
  for (int64 i = 0; i < ShapeUtil::Rank(pad->shape()); ++i) {
    auto* dim = window.add_dimensions();
    dim->set_size(1);
    dim->set_padding_low(10);
    dim->set_padding_high(100);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  const Shape reduce_window_shape =
      ShapeUtil::MakeShape(F32, {111, 113, 113, 115});
  HloInstruction* reduce_init_value = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(5.0f)));
  HloInstruction* reduce_window =
      builder.AddInstruction(HloInstruction::CreateReduceWindow(
          reduce_window_shape, pad, reduce_init_value, window,
          add_computation));

  // Build the computation and run the simplifier.
  auto computation = module.AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, reduce_window);
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(&module).ValueOrDie());

  // Running simplification again should not result in any further changes.
  ASSERT_FALSE(simplifier.Run(&module).ValueOrDie());

  // Verify the result
  root = computation->root_instruction();
  EXPECT_THAT(root, op::ReduceWindow(operand, op::Constant()));
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), reduce_window_shape))
      << ShapeUtil::HumanString(root->shape()) << " vs "
      << ShapeUtil::HumanString(reduce_window_shape);
  EXPECT_EQ(root->window().dimensions(0).padding_low(), 10);
  EXPECT_EQ(root->window().dimensions(1).padding_low(), 11);
  EXPECT_EQ(root->window().dimensions(2).padding_low(), 10);
  EXPECT_EQ(root->window().dimensions(3).padding_low(), 10);
  EXPECT_EQ(root->window().dimensions(0).padding_high(), 100);
  EXPECT_EQ(root->window().dimensions(1).padding_high(), 100);
  EXPECT_EQ(root->window().dimensions(2).padding_high(), 100);
  EXPECT_EQ(root->window().dimensions(3).padding_high(), 102);
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
  EXPECT_EQ(a, root);
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), shape));
}

TEST_F(AlgebraicSimplifierTest, IteratorInvalidation) {
  // Dots add computations to the parent module. Test that, when the HloModule's
  // computations are updated, then iterator invalidation doesn't occur
  // when running on subsequent computations.
  Shape r1f32 = ShapeUtil::MakeShape(F32, {1});
  HloComputation::Builder builder(TestName() + ".Dot");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r1f32, "x"));
  HloInstruction* y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r1f32, "y"));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  builder.AddInstruction(HloInstruction::CreateDot(r1f32, x, y, dot_dnums));
  std::unique_ptr<HloComputation> dot_computation(builder.Build());

  HloComputation::Builder call_builder(TestName() + ".Call");
  HloInstruction* zero = call_builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<float>({0.0f})));
  HloInstruction* one = call_builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<float>({1.0f})));
  call_builder.AddInstruction(
      HloInstruction::CreateCall(r1f32, {zero, one}, dot_computation.get()));

  auto module = CreateNewModule();
  module->AddEmbeddedComputation(std::move(dot_computation));
  module->AddEntryComputation(call_builder.Build());
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
}

// Test that a constant with tuple shape becomes a tuple of constants.
TEST_F(AlgebraicSimplifierTest, ConstantTupleBecomesTupleOfConstants) {
  HloComputation::Builder builder(TestName());
  const float constant_scalar = 7.3f;
  std::initializer_list<float> constant_vector = {1.1f, 2.0f, 3.3f};
  std::unique_ptr<Literal> value =
      Literal::MakeTuple({Literal::CreateR0<float>(constant_scalar).get(),
                          Literal::CreateR1<float>(constant_vector).get()});
  builder.AddInstruction(HloInstruction::CreateConstant(std::move(value)));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              op::Tuple(op::Constant(), op::Constant()));
}

// A dynamic-slice is trivial if its start indices are all zeroes and the size
// of its input equals the size of its output.  In this case, the dynamic slice
// is equal to its input.
TEST_F(AlgebraicSimplifierTest, TrivialDynamicSlice) {
  HloComputation::Builder builder(TestName());

  Shape shape = ShapeUtil::MakeShape(F32, {10, 100, 1000});
  builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      shape,
      builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape, "slice_from")),
      builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR1<int>({0, 0, 0}))),
      /*slice_sizes=*/{10, 100, 1000}));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), op::Parameter());
}

// A dynamic-update-slice is trivial if its start indices are all zeroes and the
// size of its "update" equals the size of its output.  In this case, the
// dynamic-update-slice is equal to its update.
TEST_F(AlgebraicSimplifierTest, TrivialDynamicUpdateSlice) {
  HloComputation::Builder builder(TestName());

  Shape full_shape = ShapeUtil::MakeShape(F32, {10, 100, 1000});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {10, 1, 1000});

  HloInstruction* slice =
      builder.AddInstruction(HloInstruction::CreateDynamicSlice(
          slice_shape,
          builder.AddInstruction(
              HloInstruction::CreateParameter(0, full_shape, "slice_from")),
          builder.AddInstruction(HloInstruction::CreateParameter(
              1, ShapeUtil::MakeShape(U32, {3}), "slice_indices")),
          /*slice_sizes=*/{10, 1, 1000}));

  builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      slice_shape,
      builder.AddInstruction(
          HloInstruction::CreateParameter(2, slice_shape, "to_update")),
      slice,
      builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR1<int>({0, 0, 0})))));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              op::DynamicSlice(op::Parameter(), op::Parameter()));
}

struct PadReduceWindowEffectiveBroadcastCase {
  std::vector<int64> input_spatials;
  std::vector<int64> symmetric_pad_spatials;
  std::vector<int64> reduce_window_spatials;
  // Whether to use `B F S0 S1` form vs `B S0 S1 F` form.
  //
  // This doesn't test any different functionality but is useful for making sure
  // kBroadcast nodes are well formed.
  bool prepend_a;
  bool should_become_broadcast;

  string ToTestCaseName() const {
    return tensorflow::strings::StrCat(
        tensorflow::str_util::Join(input_spatials, ","), ";",
        tensorflow::str_util::Join(symmetric_pad_spatials, ","), ";",
        tensorflow::str_util::Join(reduce_window_spatials, ","), ";", prepend_a,
        ";", should_become_broadcast);
  }
};

void PrintTo(const PadReduceWindowEffectiveBroadcastCase& c, std::ostream* os) {
  *os << c.ToTestCaseName();
}

class PadReduceWindowEffectiveBroadcastTest
    : public AlgebraicSimplifierTest,
      public ::testing::WithParamInterface<
          PadReduceWindowEffectiveBroadcastCase> {};

TEST_P(PadReduceWindowEffectiveBroadcastTest, DoIt) {
  const auto& param = GetParam();

  // a and b are parallel bounds we can either turn into a B F S0 S1 or
  // `B S0 S1 F` kind of pattern.
  auto decorate_spatials = [&param](tensorflow::gtl::ArraySlice<int64> spatials,
                                    int64 a, int64 b) {
    std::vector<int64> result;
    if (param.prepend_a) {
      result.push_back(a);
    }
    for (int64 s : spatials) {
      result.push_back(s);
    }
    if (!param.prepend_a) {
      result.push_back(a);
    }
    result.push_back(b);
    return result;
  };

  HloComputation::Builder builder(TestName());
  const Shape input_shape = ShapeUtil::MakeShape(
      F32, decorate_spatials(param.input_spatials, 128, 2048));
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));

  PaddingConfig padding = window_util::MakeSymmetricPadding(
      decorate_spatials(param.symmetric_pad_spatials, 0, 0));
  HloInstruction* pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(
          F32, decorate_spatials(param.reduce_window_spatials, 128, 2048)),
      input,
      builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0(0.0f))),
      padding));

  std::unique_ptr<HloModule> module = CreateNewModule();
  HloComputation* add_computation = nullptr;
  {
    HloComputation::Builder builder(TestName() + ".add");
    const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
    HloInstruction* p0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* p1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "p1"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, p0, p1));
    add_computation = module->AddEmbeddedComputation(builder.Build());
  }

  TF_ASSERT_OK_AND_ASSIGN(
      const Shape output_shape,
      ShapeInference::InferPadShape(input_shape, ShapeUtil::MakeShape(F32, {}),
                                    padding));
  Window window = window_util::MakeWindow(
      decorate_spatials(param.reduce_window_spatials, 1, 1));
  auto zero = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.0f)));
  builder.AddInstruction(HloInstruction::CreateReduceWindow(
      output_shape, pad, zero, window, add_computation));

  auto computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  TF_ASSERT_OK_AND_ASSIGN(bool run_successful, simplifier.Run(module.get()));
  ASSERT_TRUE(run_successful);

  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), output_shape));

  if (param.should_become_broadcast) {
    EXPECT_THAT(computation->root_instruction(), op::Broadcast(::testing::_));
  } else {
    EXPECT_THAT(computation->root_instruction(),
                op::ReduceWindow(::testing::_, zero));
  }
}

const std::vector<PadReduceWindowEffectiveBroadcastCase>&
PadReduceWindowEffectiveBroadcastCases() {
  static auto* cases = new std::vector<PadReduceWindowEffectiveBroadcastCase>{
      {/*input_spatials=*/{1, 1}, /*symmetric_pad_amount=*/{6, 6},
       /*reduce_window_spatials=*/{7, 7}, /*prepend_a=*/true,
       /*should_become_broadcast=*/true},  //
      {/*input_spatials=*/{1, 1}, /*symmetric_pad_amount=*/{6, 6},
       /*reduce_window_spatials=*/{7, 7}, /*prepend_a=*/false,
       /*should_become_broadcast=*/true},  //
      {/*input_spatials=*/{2, 2}, /*symmetric_pad_amount=*/{6, 6},
       /*reduce_window_spatials=*/{7, 7}, /*prepend_a=*/true,
       /*should_become_broadcast=*/false},  //
      {/*input_spatials=*/{1, 1}, /*symmetric_pad_amount=*/{2, 2},
       /*reduce_window_spatials=*/{5, 5}, /*prepend_a=*/true,
       /*should_become_broadcast=*/true},  //
      {/*input_spatials=*/{1, 1}, /*symmetric_pad_amount=*/{2, 2},
       /*reduce_window_spatials=*/{1, 1}, /*prepend_a=*/true,
       /*should_become_broadcast=*/false},  //
      {/*input_spatials=*/{5, 1}, /*symmetric_pad_amount=*/{0, 2},
       /*reduce_window_spatials=*/{2, 5}, /*prepend_a=*/true,
       /*should_become_broadcast=*/false},  //
  };
  return *cases;
}

INSTANTIATE_TEST_CASE_P(
    PadReduceWindowEffectiveBroadcastInstantiation,
    PadReduceWindowEffectiveBroadcastTest,
    ::testing::ValuesIn(PadReduceWindowEffectiveBroadcastCases()));

class DotStrengthReductionTest
    : public AlgebraicSimplifierTest,
      public ::testing::WithParamInterface<
          ::testing::tuple<int, int, int, bool, bool>> {};
TEST_P(DotStrengthReductionTest, DotStrengthReduction) {
  int m, k, n;
  bool transpose_lhs, transpose_rhs;
  std::tie(m, k, n, transpose_lhs, transpose_rhs) = GetParam();

  Shape dot_shape = ShapeUtil::MakeShape(F32, {m, n});
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {m, k});
  Shape transposed_lhs_shape = ShapeUtil::MakeShape(F32, {k, m});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {k, n});
  Shape transposed_rhs_shape = ShapeUtil::MakeShape(F32, {n, k});
  HloComputation::Builder builder(TestName());

  auto lhs = builder.AddInstruction(HloInstruction::CreateParameter(
      0, transpose_lhs ? transposed_lhs_shape : lhs_shape, "lhs"));
  if (transpose_lhs) {
    lhs = builder.AddInstruction(
        HloInstruction::CreateTranspose(lhs_shape, lhs, {1, 0}));
  }
  auto rhs = builder.AddInstruction(HloInstruction::CreateParameter(
      1, transpose_rhs ? transposed_rhs_shape : rhs_shape, "rhs"));
  if (transpose_rhs) {
    rhs = builder.AddInstruction(
        HloInstruction::CreateTranspose(rhs_shape, rhs, {1, 0}));
  }
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  builder.AddInstruction(
      HloInstruction::CreateDot(dot_shape, lhs, rhs, dot_dnums));
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, simplifier.Run(module.get()));
  const bool dot_should_be_transformed = m == 1 || k == 1 || n == 1;
  const bool computation_should_be_modified =
      dot_should_be_transformed || (transpose_lhs && transpose_rhs);
  EXPECT_EQ(changed, computation_should_be_modified);
  bool has_no_dot = true;
  for (const auto& hlo : computation->instructions()) {
    if (hlo->opcode() == HloOpcode::kDot) {
      has_no_dot = false;
      break;
    }
  }
  EXPECT_EQ(has_no_dot, dot_should_be_transformed);
}

INSTANTIATE_TEST_CASE_P(
    DotStrengthReductionTestInstantiation, DotStrengthReductionTest,
    ::testing::Combine(::testing::Values(1, 2), ::testing::Values(1, 2),
                       ::testing::Values(1, 2), ::testing::Bool(),
                       ::testing::Bool()));

struct DotOfConcatTestSpec {
  int64 m;
  int64 k;
  int64 n;
};

class DotOfConcatSimplificationTest
    : public HloTestBase,
      public ::testing::WithParamInterface<DotOfConcatTestSpec> {};

// Test that we transform
//  dot(const, concat(A, B, C))
// to
//  add(dot(const_0, A), dot(const_1, B),  dot(const_2, C))
TEST_P(DotOfConcatSimplificationTest, ConstantLHS) {
  HloComputation::Builder builder(TestName());

  DotOfConcatTestSpec spec = GetParam();

  ASSERT_GE(spec.k, 3);

  int64 k0 = spec.k / 3;
  int64 k1 = spec.k / 3;
  int64 k2 = spec.k - k0 - k1;

  Shape lhs_shape = ShapeUtil::MakeShape(F32, {spec.m, spec.k});
  auto* lhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/spec.m, /*cols=*/spec.k)));

  Shape rhs0_shape = ShapeUtil::MakeShape(F32, {k0, spec.n});
  Shape rhs1_shape = ShapeUtil::MakeShape(F32, {k1, spec.n});
  Shape rhs2_shape = ShapeUtil::MakeShape(F32, {k2, spec.n});

  HloInstruction* rhs0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, rhs0_shape, "rhs0"));
  HloInstruction* rhs1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, rhs1_shape, "rhs1"));
  HloInstruction* rhs2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, rhs2_shape, "rhs2"));

  Shape rhs_shape = ShapeUtil::MakeShape(F32, {spec.k, spec.n});
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateConcatenate(rhs_shape, {rhs0, rhs1, rhs2}, 0));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);

  Shape dot_shape = ShapeUtil::MakeShape(F32, {spec.m, spec.n});
  builder.AddInstruction(
      HloInstruction::CreateDot(dot_shape, lhs, rhs, dot_dnums));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  TF_ASSERT_OK_AND_ASSIGN(bool run_successful, simplifier.Run(module.get()));
  ASSERT_TRUE(run_successful);

  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), dot_shape));

  auto match_dot_0 = op::Dot(op::Slice(op::Constant()), op::Parameter(0));
  auto match_dot_1 = op::Dot(op::Slice(op::Constant()), op::Parameter(1));
  auto match_dot_2 = op::Dot(op::Slice(op::Constant()), op::Parameter(2));
  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Add(match_dot_0, match_dot_1), match_dot_2));
}

// Test that we transform
//  dot(concat(A, B, C), const)
// to
//  add(dot(A, const_0), dot(B, const_1),  dot(C, const_2))
TEST_P(DotOfConcatSimplificationTest, ConstantRHS) {
  HloComputation::Builder builder(TestName());

  DotOfConcatTestSpec spec = GetParam();

  ASSERT_GE(spec.k, 4);

  int64 k0 = spec.k / 4;
  int64 k1 = spec.k / 4;
  int64 k2 = spec.k / 4;
  int64 k3 = spec.k - k0 - k1 - k2;

  Shape lhs0_shape = ShapeUtil::MakeShape(F32, {spec.m, k0});
  Shape lhs1_shape = ShapeUtil::MakeShape(F32, {spec.m, k1});
  Shape lhs2_shape = ShapeUtil::MakeShape(F32, {spec.m, k2});
  Shape lhs3_shape = ShapeUtil::MakeShape(F32, {spec.m, k3});

  HloInstruction* lhs0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs0_shape, "lhs0"));
  HloInstruction* lhs1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, lhs1_shape, "lhs1"));
  HloInstruction* lhs2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, lhs2_shape, "lhs2"));
  HloInstruction* lhs3 = builder.AddInstruction(
      HloInstruction::CreateParameter(3, lhs2_shape, "lhs3"));

  Shape lhs_shape = ShapeUtil::MakeShape(F32, {spec.m, spec.k});
  HloInstruction* lhs =
      builder.AddInstruction(HloInstruction::CreateConcatenate(
          lhs_shape, {lhs0, lhs1, lhs2, lhs3}, 1));

  Shape rhs_shape = ShapeUtil::MakeShape(F32, {spec.k, spec.m});
  auto* rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/spec.k, /*cols=*/spec.m)));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);

  Shape dot_shape = ShapeUtil::MakeShape(F32, {spec.m, spec.n});
  builder.AddInstruction(
      HloInstruction::CreateDot(dot_shape, lhs, rhs, dot_dnums));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(/*is_layout_sensitive=*/false,
                                 non_bitcasting_callback());
  TF_ASSERT_OK_AND_ASSIGN(bool run_successful, simplifier.Run(module.get()));
  ASSERT_TRUE(run_successful);
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), dot_shape));

  auto match_dot_0 = op::Dot(op::Parameter(0), op::Slice(op::Constant()));
  auto match_dot_1 = op::Dot(op::Parameter(1), op::Slice(op::Constant()));
  auto match_dot_2 = op::Dot(op::Parameter(2), op::Slice(op::Constant()));
  auto match_dot_3 = op::Dot(op::Parameter(3), op::Slice(op::Constant()));
  EXPECT_THAT(computation->root_instruction(),
              op::Add(op::Add(op::Add(match_dot_0, match_dot_1), match_dot_2),
                      match_dot_3));
}

DotOfConcatTestSpec kDotOfConcatTestSpecs[] = {
    {/*m=*/3, /*k=*/9, /*n=*/3},    //
    {/*m=*/3, /*k=*/20, /*n=*/3},   //
    {/*m=*/1, /*k=*/18, /*n=*/5},   //
    {/*m=*/20, /*k=*/20, /*n=*/1},  //
    {/*m=*/1, /*k=*/16, /*n=*/1},   //
};

INSTANTIATE_TEST_CASE_P(DotOfConcatSimplificationTestInstantiation,
                        DotOfConcatSimplificationTest,
                        ::testing::ValuesIn(kDotOfConcatTestSpecs));
}  // namespace
}  // namespace xla

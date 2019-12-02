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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
namespace m = match;

class AlgebraicSimplifierTest : public HloTestBase {
 protected:
  AlgebraicSimplifierOptions default_options_;
};

// Test that A + 0 is simplified to A
TEST_F(AlgebraicSimplifierTest, AddZero) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, param0, zero));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

TEST_F(AlgebraicSimplifierTest, FactorIntegerAddition) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[8] parameter(0)
      p1 = s32[8] parameter(1)
      p2 = s32[8] parameter(2)
      x = s32[8] multiply(p0, p2)
      y = s32[8] multiply(p1, p2)
      ROOT sum = s32[8] add(x, y)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::MultiplyAnyOrder(
          m::AddAnyOrder(m::Parameter(0), m::Parameter(1)), m::Parameter(2))));
}

// A*C + B*C => (A+B)*C if C is a floating-point power of 2.
TEST_F(AlgebraicSimplifierTest, FactorFpAddition) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      c = f32[] constant(0.125)
      x = f32[] multiply(p0, c)
      y = f32[] multiply(p1, c)
      ROOT sum = f32[] add(x, y)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::MultiplyAnyOrder(
                  m::AddAnyOrder(m::Parameter(0), m::Parameter(1)),
                  m::ConstantScalar(0.125))));
}

// A*C + B*C => (A+B)*C if C is a broadcast of a floating-point power of 2.
TEST_F(AlgebraicSimplifierTest, FactorFpAdditionWithBroadcast) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      c = f32[] constant(0.125)
      b = f32[4] broadcast(c), dimensions={}
      x = f32[4] multiply(p0, b)
      y = f32[4] multiply(p1, b)
      ROOT sum = f32[4] add(x, y)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::MultiplyAnyOrder(
                  m::AddAnyOrder(m::Parameter(0), m::Parameter(1)),
                  m::Broadcast(m::ConstantScalar(0.125)))));
}

// A*C + B*C => (A+B)*C simplification should not happen if C is not a
// floating-point power of 2.
TEST_F(AlgebraicSimplifierTest, FactorFpAdditionNotPowerOf2) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      c = f32[] constant(0.3)
      x = f32[] multiply(p0, c)
      y = f32[] multiply(p1, c)
      ROOT sum = f32[] add(x, y)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  EXPECT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
}

// A*C + B*C => (A+B)*C simplification should not happen if A, B, and C are
// complex numbers.
TEST_F(AlgebraicSimplifierTest, FactorFpAdditionComplex) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = c64[8] parameter(0)
      p1 = c64[8] parameter(1)
      p2 = c64[8] parameter(2)
      x = c64[8] multiply(p0, p2)
      y = c64[8] multiply(p1, p2)
      ROOT sum = c64[8] add(x, y)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  EXPECT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
}

// A*C + B*C => (A+B)*C simplification is OK if A, B, and C are complex.
TEST_F(AlgebraicSimplifierTest, FactorFpAdditionBfloat16) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = bf16[4] parameter(0)
      p1 = bf16[4] parameter(1)
      c = bf16[] constant(0.125)
      b = bf16[4] broadcast(c), dimensions={}
      x = bf16[4] multiply(p0, b)
      y = bf16[4] multiply(p1, b)
      ROOT sum = bf16[4] add(x, y)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::MultiplyAnyOrder(
                  m::AddAnyOrder(m::Parameter(0), m::Parameter(1)),
                  m::Broadcast(m::ConstantScalar(0.125)))));
}

TEST_F(AlgebraicSimplifierTest, UnsignedDivideByPowerOf2) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = u32[4] parameter(0)
      c = u32[] constant(8)
      b = u32[4] broadcast(c), dimensions={}
      ROOT d = u32[4] divide(p, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ShiftRightLogical(
                  m::Parameter(0), m::Broadcast(m::ConstantScalar(3)))));
}

TEST_F(AlgebraicSimplifierTest, SignedDivideByPowerOf2) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = s32[4] parameter(0)
      c = s32[] constant(8)
      b = s32[4] broadcast(c), dimensions={}
      ROOT d = s32[4] divide(p, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  auto match_dividend_is_negative =
      m::Lt(m::Parameter(0), m::Broadcast(m::ConstantScalar(0)));
  auto match_abs = m::Select(match_dividend_is_negative,
                             m::Negate(m::Parameter(0)), m::Parameter(0));
  auto match_shift =
      m::ShiftRightLogical(match_abs, m::Broadcast(m::ConstantScalar(3)));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Select(match_dividend_is_negative,
                                   m::Negate(match_shift), match_shift)));
}

TEST_F(AlgebraicSimplifierTest, UnsignedRemainderByPowerOf2) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = u32[4] parameter(0)
      c = u32[] constant(8)
      b = u32[4] broadcast(c), dimensions={}
      ROOT r = u32[4] remainder(p, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::AndAnyOrder(m::Parameter(0),
                                        m::Broadcast(m::ConstantScalar(7)))));
}

TEST_F(AlgebraicSimplifierTest, SignedRemainderByPowerOf2) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = s32[4] parameter(0)
      c = s32[] constant(8)
      b = s32[4] broadcast(c), dimensions={}
      ROOT r = s32[4] remainder(p, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  auto match_dividend_is_negative =
      m::Lt(m::Parameter(0), m::Broadcast(m::ConstantScalar(0)));
  auto match_abs = m::Select(match_dividend_is_negative,
                             m::Negate(m::Parameter(0)), m::Parameter(0));
  auto match_and =
      m::AndAnyOrder(match_abs, m::Broadcast(m::ConstantScalar(7)));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Select(match_dividend_is_negative,
                                   m::Negate(match_and), match_and)));
}

// Test that A * 0 is simplified to 0
TEST_F(AlgebraicSimplifierTest, MulZero) {
  auto m = CreateNewVerifiedModule();
  Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0s32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0s32, HloOpcode::kMultiply, param0, zero));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kMultiply);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_EQ(computation->root_instruction(), zero);
}

TEST_F(AlgebraicSimplifierTest, MultiplyReassociateMergeConstants) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      c0 = f32[] constant(2.0)
      c1 = f32[] constant(3.0)
      multiply0 = f32[] multiply(p0, c0)
      ROOT multiply1 = f32[] multiply(multiply0, c1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0),
                                     m::Multiply(m::ConstantScalar(2.0),
                                                 m::ConstantScalar(3.0)))));
}

TEST_F(AlgebraicSimplifierTest, MultiplyReassociateMergeBroadcastedConstants) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4] parameter(0)
      c0 = f32[] constant(2.0)
      c1 = f32[] constant(3.0)
      b0 = f32[4] broadcast(c0), dimensions={}
      b1 = f32[4] broadcast(c1), dimensions={}
      multiply0 = f32[4] multiply(p0, b0)
      ROOT multiply1 = f32[4] multiply(multiply0, b1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Multiply(
          m::Parameter(0), m::Broadcast(m::Multiply(m::ConstantScalar(2.0),
                                                    m::ConstantScalar(3.0))))));
}

TEST_F(AlgebraicSimplifierTest,
       MultiplyReassociateMultiplyOfConstantAndBroadcast) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      c0 = f32[4] constant({2.0, 3.0, 4.0, 5.0})
      c1 = f32[] constant(3.0)
      c2 = f32[] constant(4.0)
      b0 = f32[4] broadcast(c1), dimensions={}
      b1 = f32[4] broadcast(c2), dimensions={}
      multiply0 = f32[4] multiply(c0, b0)
      ROOT multiply1 = f32[4] multiply(multiply0, b1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Multiply(
          m::Constant(), m::Broadcast(m::Multiply(m::ConstantScalar(3.0),
                                                  m::ConstantScalar(4.0))))));
}

// Test that select(true, a, b) is simplified to a
TEST_F(AlgebraicSimplifierTest, SelectTrue) {
  Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0s32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0s32, "param1"));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  builder.AddInstruction(HloInstruction::CreateTernary(
      r0s32, HloOpcode::kSelect, one, param0, param1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_EQ(computation->root_instruction(), param0);
}

// Test that select(false, a, b) is simplified to b
TEST_F(AlgebraicSimplifierTest, SelectFalse) {
  Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0s32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0s32, "param1"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateTernary(
      r0s32, HloOpcode::kSelect, zero, param0, param1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_EQ(computation->root_instruction(), param1);
}

// Test that select(a, b, b) is simplified to b
TEST_F(AlgebraicSimplifierTest, SelectIdentical) {
  Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0s32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0s32, "param1"));
  builder.AddInstruction(HloInstruction::CreateTernary(
      r0s32, HloOpcode::kSelect, param0, param1, param1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_EQ(computation->root_instruction(), param1);
}

// Test that Reduce(Reduce(A)) -> Reduce(A)
TEST_F(AlgebraicSimplifierTest, TwoReducesToOne) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  // Create add computation.
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
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
    add_computation = m->AddEmbeddedComputation(builder.Build());
  }
  Shape r4f32 = ShapeUtil::MakeShape(F32, {4, 5, 6, 7});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r4f32, "param"));
  std::vector<int64> dims0({0});
  Shape r3f32 = ShapeUtil::MakeShape(F32, {5, 6, 7});
  HloInstruction* reduce0 = builder.AddInstruction(
      HloInstruction::CreateReduce(r3f32, param, zero, dims0, add_computation));
  std::vector<int64> dims1({1, 2});
  Shape r1f32 = ShapeUtil::MakeShape(F32, {5});
  builder.AddInstruction(HloInstruction::CreateReduce(r1f32, reduce0, zero,
                                                      dims1, add_computation));
  m->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Reduce(m::Parameter(0), m::Op().Is(zero))));
  EXPECT_EQ(root->dimensions(), std::vector<int64>({0, 2, 3}));
}

// Test that Const + A is canonicalized to A + Const.
TEST_F(AlgebraicSimplifierTest, AddConstOnLHS) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(42.0f)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, constant, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Add(m::Parameter(0), m::Constant())));
}

// Test that [(A + C1) + C2] => [A + (C1 + C2)] for constants C1 and C2.
TEST_F(AlgebraicSimplifierTest, AddReassociateMergeConstants) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(42.0f)));
  HloInstruction* constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(3.14159f)));

  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, param0, constant1));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, add1, constant2));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Add(
                        m::Op().Is(param0),
                        m::Add(m::Op().Is(constant1), m::Op().Is(constant2)))));
}

TEST_F(AlgebraicSimplifierTest, AddReassociateMergeBroadcastedConstants) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4] parameter(0)
      c0 = f32[] constant(1.0)
      c1 = f32[] constant(2.0)
      b0 = f32[4] broadcast(c0), dimensions={}
      b1 = f32[4] broadcast(c1), dimensions={}
      add0 = f32[4] add(p0, b0)
      ROOT add1 = f32[4] add(add0, b1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Add(m::Parameter(0),
                                m::Broadcast(m::Add(m::ConstantScalar(1.0),
                                                    m::ConstantScalar(2.0))))));
}

TEST_F(AlgebraicSimplifierTest, AddBroadcastZeroR0Operand) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {3, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  HloInstruction* bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(r2f32, zero, {0, 1}));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kAdd, bcast, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

TEST_F(AlgebraicSimplifierTest, InlineTrivialMap) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
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
    add_computation = m->AddEmbeddedComputation(builder.Build());
  }
  Shape r2f32 = ShapeUtil::MakeShape(F32, {32, 1});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  builder.AddInstruction(HloInstruction::CreateMap(
      r2f32,
      {param0, builder.AddInstruction(
                   HloInstruction::CreateBroadcast(r2f32, zero, {}))},
      add_computation));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kMap);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Add(m::Parameter(0),
                                      m::Broadcast(m::Op().Is(zero)))));
}

TEST_F(AlgebraicSimplifierTest, KeepNontrivialMap) {
  const char* kModuleStr = R"(
    HloModule m
    fusion {
      x = f32[] parameter(0)
      c = f32[] constant(42)
      m = f32[] multiply(x, x)
      ROOT a = f32[] add(m, c)
    }

    map {
      x = f32[] parameter(0)
      ROOT f = f32[] fusion(x), kind=kLoop, calls=fusion
    }

    ENTRY test {
      p = f32[2,2] parameter(0)
      ROOT map = f32[2,2] map(p), dimensions={0,1}, to_apply=map
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
}

TEST_F(AlgebraicSimplifierTest, AddBroadcastZeroR1Operand) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {3, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({0, 0, 0})));
  HloInstruction* bcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(r2f32, zero, {1}));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kAdd, bcast, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

TEST_F(AlgebraicSimplifierTest, AddBroadcastZeroWithDynamicSlice) {
  auto m = CreateNewVerifiedModule();
  Shape full_shape = ShapeUtil::MakeShape(F32, {1800, 12, 512});

  Shape partial_shape = ShapeUtil::MakeShape(F32, {1, 12, 512});

  HloComputation::Builder builder(TestName());
  HloInstruction* full_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, full_shape, "param0"));
  HloInstruction* partial_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, partial_shape, "param1"));

  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
  HloInstruction* index = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));

  HloInstruction* bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(full_shape, zero, {}));

  HloInstruction* dynamic_update_slice =
      builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
          full_shape, bcast, partial_param, {index, index, index}));

  builder.AddInstruction(HloInstruction::CreateBinary(
      full_shape, HloOpcode::kAdd, full_param, dynamic_update_slice));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root->opcode(), HloOpcode::kDynamicUpdateSlice);
  EXPECT_THAT(root->operand(0), full_param);
  EXPECT_THAT(root->operand(1), GmockMatch(m::Add(m::DynamicSlice(), m::Op())));
}

TEST_F(AlgebraicSimplifierTest, ConstantToBroadcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({3.14f, 3.14f, 3.14f})));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Constant())));
  EXPECT_EQ(3.14f, root->operand(0)->literal().GetFirstElement<float>());
}

TEST_F(AlgebraicSimplifierTest, ConstantNotToBroadcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({3.14, 3.14, 4})));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
}

TEST_F(AlgebraicSimplifierTest, IotaToBroadcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({0.0f, 1.0f, 2.0f})));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Iota()));
}

// Test that A - 0 is simplified to A
TEST_F(AlgebraicSimplifierTest, SubZero) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kSubtract, param0, zero));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSubtract);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that A - Const is canonicalized to A + (-Const).
TEST_F(AlgebraicSimplifierTest, SubConstCanonicalization) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kSubtract, param0, constant));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSubtract);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Add(m::Parameter(0),
                                      m::Negate(m::Op().Is(constant)))));
}

// Test that A - Broadcast(Const) is canonicalized to A + Broadcast(-Const).
TEST_F(AlgebraicSimplifierTest, SubBroadcastConstCanonicalization) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4] parameter(0)
      c = f32[] constant(0.125)
      b = f32[4] broadcast(c), dimensions={}
      ROOT sub = f32[4] subtract(p0, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Add(m::Parameter(0),
                        m::Broadcast(m::Negate(m::ConstantScalar(0.125))))));
}

// Test that Broadcast(x) where x has degenerate dimensions first removes the
// degenerate dimensions.
TEST_F(AlgebraicSimplifierTest, DegenerateDimsInOperandRemovedFromBroadcast) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      c = f32[1,4] parameter(0)
      ROOT b = f32[5,1,4,3] broadcast(c), dimensions={1,2}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Reshape(m::Parameter(0)))));
}

// Test that (A/B)/C is simplified to A/(B*C).
TEST_F(AlgebraicSimplifierTest, LhsDivOfDiv) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Divide(m::Divide(m::Parameter(0), m::Parameter(1)),
                                   m::Parameter(2))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Parameter(0),
                           m::Multiply(m::Parameter(1), m::Parameter(2)))));
}

// Test that A/(B/C) is simplified to (A*C)/B.
TEST_F(AlgebraicSimplifierTest, RhsDivOfDiv) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Parameter(0),
                           m::Divide(m::Parameter(1), m::Parameter(2)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Multiply(m::Parameter(0), m::Parameter(2)),
                           m::Parameter(1))));
}

// Test that (A/B)/(C/D) is simplified to (A*D)/(B*C).
TEST_F(AlgebraicSimplifierTest, DivOfDivAndDiv) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {42, 123});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r2f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r2f32, "param2"));
  HloInstruction* param3 = builder.AddInstruction(
      HloInstruction::CreateParameter(3, r2f32, "param3"));
  HloInstruction* div0 = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kDivide, param0, param1));
  HloInstruction* div1 = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kDivide, param2, param3));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kDivide, div0, div1));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Divide(m::Parameter(0), m::Parameter(1)),
                           m::Divide(m::Parameter(2), m::Parameter(3)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Multiply(m::Parameter(0), m::Parameter(3)),
                           m::Multiply(m::Parameter(1), m::Parameter(2)))));
}

// Test that A/exp(B) is simplified to A*exp(-B).
TEST_F(AlgebraicSimplifierTest, DivOfExp) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Divide(m::Parameter(0), m::Exp(m::Parameter(1)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0),
                                     m::Exp(m::Negate(m::Parameter(1))))));
}

// Test that A/pow(B,C) is simplified to A*pow(B,-C).
TEST_F(AlgebraicSimplifierTest, DivOfPower) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Parameter(0),
                           m::Power(m::Parameter(1), m::Parameter(2)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(
                  m::Parameter(0),
                  m::Power(m::Parameter(1), m::Negate(m::Parameter(2))))));
}

// Test that broadcasting is done on the right step when simplifying A/pow(B,C)
// to A*pow(B,-C).
TEST_F(AlgebraicSimplifierTest, DivOfBroadcastingPower) {
  auto m = CreateNewVerifiedModule();
  Shape r1f32 = ShapeUtil::MakeShape(F32, {7});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r1f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r1f32, "param2"));
  HloInstruction* power = builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kPower, param1, param2));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kDivide, param0, power));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Parameter(0),
                           m::Power(m::Parameter(1), m::Parameter(2)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  ASSERT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(
                  m::Parameter(0),
                  m::Power(m::Parameter(1), m::Negate(m::Parameter(2))))));
}

// A / Const => A * InvertedConst
TEST_F(AlgebraicSimplifierTest, DivideByConstant) {
  auto m = CreateNewVerifiedModule();
  Shape r1f32 = ShapeUtil::MakeShape(F32, {3});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* constant =
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<float>({1.f, 2.f, 3.f})));
  builder.AddInstruction(HloInstruction::CreateBinary(r1f32, HloOpcode::kDivide,
                                                      param0, constant));

  auto computation = m->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Constant())));
}

// A / Broadcast(Const) => A * Broadcast(InvertedConst)
TEST_F(AlgebraicSimplifierTest, DivideByBroadcastedConstant) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = f32[4] parameter(0)
      c = f32[] constant(256.0)
      b = f32[4] broadcast(c), dimensions={}
      ROOT d = f32[4] divide(p, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());

  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(
                  m::Parameter(0),
                  m::Broadcast(m::Op().IsConstantScalar(1.0f / 256.0f)))));
}

// pow(pow(A, X), Y) => pow(A, X*Y)
TEST_F(AlgebraicSimplifierTest, PowerOfPower) {
  auto m = CreateNewVerifiedModule();
  Shape r1f32 = ShapeUtil::MakeShape(F32, {7});
  HloComputation::Builder builder(TestName());
  HloInstruction* base = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r1f32, "param1"));
  HloInstruction* exp2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r1f32, "param2"));
  HloInstruction* inner_power = builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kPower, base, exp1));
  builder.AddInstruction(HloInstruction::CreateBinary(r1f32, HloOpcode::kPower,
                                                      inner_power, exp2));

  auto computation = m->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Power(m::Op().Is(base),
                          m::Multiply(m::Op().Is(exp1), m::Op().Is(exp2)))));
}

// Don't simplify pow(pow(A, X), Y) => pow(A, X*Y) if X and Y are complex
// numbers.
TEST_F(AlgebraicSimplifierTest, PowerOfPowerComplex) {
  auto m = CreateNewVerifiedModule();
  Shape r1c64 = ShapeUtil::MakeShape(C64, {7});
  HloComputation::Builder builder(TestName());
  HloInstruction* base = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1c64, "param0"));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r1c64, "param1"));
  HloInstruction* exp2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r1c64, "param2"));
  HloInstruction* inner_power = builder.AddInstruction(
      HloInstruction::CreateBinary(r1c64, HloOpcode::kPower, base, exp1));
  builder.AddInstruction(HloInstruction::CreateBinary(r1c64, HloOpcode::kPower,
                                                      inner_power, exp2));

  m->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(m.get()).ValueOrDie());
}

// Test that A/1 is simplified to A for a scalar.
TEST_F(AlgebraicSimplifierTest, DivOneScalar) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, param0, one));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, div);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that A/1 is simplified to A for an array.
TEST_F(AlgebraicSimplifierTest, DivOneArray) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 2});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* one = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 1.0}, {1.0, 1.0}})));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kDivide, param0, one));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, div);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that complex(real(c), imag(c)) is simplified to c.
TEST_F(AlgebraicSimplifierTest, ComplexOfRealImagC) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, cplx);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that real(complex(r,i)) is simplified to r.
TEST_F(AlgebraicSimplifierTest, RealOfComplex) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, real);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that imag(complex(r,i)) is simplified to i.
TEST_F(AlgebraicSimplifierTest, ImagOfComplex) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, imag);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param1);
}

// Test that get_element(make_tuple({A,B}),1) is simplified to B
TEST_F(AlgebraicSimplifierTest, SelectMakeTuple) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, add);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Add(m::Parameter(1), m::Parameter(2))));
}

// Test that exp(A)/exp(B) is simplified to exp(A-B)
TEST_F(AlgebraicSimplifierTest, ExpDiv) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Exp(m::Parameter(0)), m::Exp(m::Parameter(1)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Exp(m::Subtract(m::Parameter(0), m::Parameter(1)))));
}

// Test that exp(A)*exp(B) is simplified to exp(A+B)
TEST_F(AlgebraicSimplifierTest, ExpMul) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Exp(m::Parameter(0)),
                                     m::Exp(m::Parameter(1)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Exp(m::Add(m::Parameter(0), m::Parameter(1)))));
}

// Test that pow(exp(A), B) is simplified to exp(A*B)
TEST_F(AlgebraicSimplifierTest, PowExp) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Exp(m::Parameter(0)), m::Parameter(1))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Exp(m::Multiply(m::Parameter(0), m::Parameter(1)))));
}

// Test that ln(pow(A, B)) is simplified to ln(A)*B
TEST_F(AlgebraicSimplifierTest, LnPow) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Log(m::Power(m::Parameter(0), m::Parameter(1)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Log(m::Abs(m::Parameter(0))),
                                     m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, LnSqrt) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* sqrt = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kSqrt, param0));
  builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kLog, sqrt));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Log(m::Sqrt(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Multiply(m::Log(m::Parameter(0)), m::ConstantScalar(0.5))));
}

TEST_F(AlgebraicSimplifierTest, LnRsqrt) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* rsqrt = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kRsqrt, param0));
  builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kLog, rsqrt));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Log(m::Rsqrt(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Log(m::Parameter(0)),
                                     m::ConstantScalar(-0.5))));
}

// Test that ln(exp(A)) is simplified to A
TEST_F(AlgebraicSimplifierTest, LnExp) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* exp0 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param0));
  builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kLog, exp0));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Log(m::Exp(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_EQ(computation->root_instruction(), param0);
}

// Test that ln(exp(A)/exp(B)) is simplified to A-B
TEST_F(AlgebraicSimplifierTest, LnExpDiv) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Log(m::Divide(m::Exp(m::Parameter(0)),
                                          m::Exp(m::Parameter(1))))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Subtract(m::Parameter(0), m::Parameter(1))));
}

// Test that pow(A, 0) where A is a scalar is simplified to the scalar
// constant 1.
TEST_F(AlgebraicSimplifierTest, Pow0Scalar) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, zero));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(zero))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
  EXPECT_EQ(root->literal().GetFirstElement<float>(), 1);
}

// Test that pow(A, 0) where A is not a scalar is simplified to broadcast(1).
TEST_F(AlgebraicSimplifierTest, Pow0Vector) {
  auto m = CreateNewVerifiedModule();
  Shape r1f32 = ShapeUtil::MakeShape(F32, {42});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kPower, param0, zero));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(zero))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast()));
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), r1f32))
      << ShapeUtil::HumanString(root->shape());
  EXPECT_EQ(root->dimensions().size(), 0);
  EXPECT_TRUE(ShapeUtil::IsScalar(root->operand(0)->shape()));
  EXPECT_EQ(root->operand(0)->literal().GetFirstElement<float>(), 1);
}

// Test that pow(A, 1) is simplified to A.
TEST_F(AlgebraicSimplifierTest, Pow1) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, one));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(one))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_EQ(computation->root_instruction(), param0);
}

// Test that pow(A, 2) is simplified to A*A.
TEST_F(AlgebraicSimplifierTest, Pow2) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* two = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, two));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(two))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(0))));
}

// Test that pow(A, -1) is simplified to 1/A.
TEST_F(AlgebraicSimplifierTest, PowNegative1) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* negative_one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(-1)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0f32, HloOpcode::kPower,
                                                      param0, negative_one));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(negative_one))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Divide(m::Constant(), m::Parameter(0))));
  EXPECT_EQ(root->operand(0)->literal().GetFirstElement<float>(), 1);
}

TEST_F(AlgebraicSimplifierTest, ZeroSizedConvolution) {
  auto m = CreateNewVerifiedModule();
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
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeUtil::MakeShape(F32, {3, 3, 3}), lhs, rhs, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m->AddEntryComputation(builder.Build());
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Convolution(m::Op().Is(lhs), m::Op().Is(rhs))));
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Constant())));
}

TEST_F(AlgebraicSimplifierTest, ReduceWindowIsReduceAndReshape) {
  auto m = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 2, 3, 4}), "param"));
  Window window;
  for (int64 i = 0; i < 4; ++i) {
    WindowDimension* dim = window.add_dimensions();
    // Makes 1x2x3x1 window.
    dim->set_size((i % 3) + 1);
    dim->set_stride(1);
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
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
    add_computation = m->AddEmbeddedComputation(builder.Build());
  }
  builder.AddInstruction(HloInstruction::CreateReduceWindow(
      ShapeUtil::MakeShape(F32, {1, 1, 1, 4}), param,
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f))),
      window, add_computation));
  m->AddEntryComputation(builder.Build());
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceWindow(m::Parameter(0), m::Constant())));
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Reshape(m::Reduce(m::Parameter(0), m::Constant()))));
}

TEST_F(AlgebraicSimplifierTest, ZeroSizedReduceWindow) {
  auto m = CreateNewVerifiedModule();
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
    add_computation = m->AddEmbeddedComputation(builder.Build());
  }
  builder.AddInstruction(HloInstruction::CreateReduceWindow(
      ShapeUtil::MakeShape(F32, {5, 2}), param,
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f))),
      window, add_computation));
  m->AddEntryComputation(builder.Build());
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceWindow(m::Parameter(0), m::Constant())));
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Constant())));
}

TEST_F(AlgebraicSimplifierTest, ZeroSizedPad) {
  auto m = CreateNewVerifiedModule();
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
          HloInstruction::CreateConstant(LiteralUtil::CreateR0(0.0f))),
      padding));
  m->AddEntryComputation(builder.Build());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Constant())));
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Constant())));
}

TEST_F(AlgebraicSimplifierTest, ReshapeBroadcast) {
  auto m = CreateNewVerifiedModule();
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
  m->AddEntryComputation(std::move(computation));

  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Reshape(m::Op().Is(op))))));

  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(m->entry_computation()->root_instruction(), op);
}

// Test that convert(A, $TYPE) is simplified to A if A is of type $TYPE.
TEST_F(AlgebraicSimplifierTest, ConvertBetweenSameType) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(
      HloInstruction::CreateConvert(ShapeUtil::MakeShape(F32, {}), input));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Convert(m::Op().Is(input))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), input);
}

// Test that convert(convert(A, $TYPE1), $TYPE2) is simplified to A if A is of
// $TYPE2 and convert(A, $TYP1) is an upcast.
TEST_F(AlgebraicSimplifierTest, EliminateConvertPairUpCast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* input =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShapeWithLayout(F16, {1, 14, 14, 64}, {3, 2, 1, 0}),
          "param"));
  HloInstruction* convert_1 =
      builder.AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(input->shape(), F32), input));
  builder.AddInstruction(HloInstruction::CreateConvert(
      ShapeUtil::ChangeElementType(convert_1->shape(), F16), convert_1));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Convert(m::Convert(m::Op().Is(input)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), input);
}

// Test that convert(convert(A, $TYPE1), $TYPE2) is NOT simplified to A even if
// A is of $TYPE2 since convert(A, $TYP1) is a downcast.
TEST_F(AlgebraicSimplifierTest, DoNotEliminateConvertPairDownCast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* input =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShapeWithLayout(F32, {1, 14, 14, 64}, {3, 2, 1, 0}),
          "param"));
  HloInstruction* convert_1 =
      builder.AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(input->shape(), F16), input));
  builder.AddInstruction(HloInstruction::CreateConvert(
      ShapeUtil::ChangeElementType(convert_1->shape(), F32), convert_1));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Convert(m::Convert(m::Op().Is(input)))));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Convert(m::Convert(m::Op().Is(input)))));
}

// Test that Tuple(convert(A, $TYPE1) , floor(convert(convert(A, $TYPE1),
// $TYPE2)), convert(convert(A, $TYPE1), $TYPE2)) is simplified to
// Tuple(convert(A, $TYPE1) , floor(A), A) showing a case where the first
// convert has a fan-out.
TEST_F(AlgebraicSimplifierTest, EliminateConvertPairMultiOut) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* input =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShapeWithLayout(F16, {1, 14, 14, 64}, {3, 2, 1, 0}),
          "param"));
  HloInstruction* convert_1 =
      builder.AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(input->shape(), F32), input));
  HloInstruction* convert_2 =
      builder.AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(convert_1->shape(), F16), convert_1));

  HloInstruction* floor = builder.AddInstruction(HloInstruction::CreateUnary(
      convert_2->shape(), HloOpcode::kFloor, convert_2));

  // Collect all the reshapes into a tuple so they are not dead.
  builder.AddInstruction(
      HloInstruction::CreateTuple({convert_1, convert_2, floor}));

  auto computation = m->AddEntryComputation(builder.Build());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Tuple(m::Op().Is(convert_1), m::Op().Is(convert_2),
                                  m::Op().Is(floor))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Tuple(m::Op().Is(convert_1), m::Op().Is(input),
                                  m::Floor(m::Op().Is(input)))));
}

// Test that copies are removed.
TEST_F(AlgebraicSimplifierTest, RemoveCopy) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  builder.AddInstruction(
      HloInstruction::CreateUnary(param0->shape(), HloOpcode::kCopy, param0));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), param0);
}

TEST_F(AlgebraicSimplifierTest, CopyOfReshapeOfCopyEqualsBitcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShapeWithLayout(F32, {1, 14, 14, 64}, {3, 2, 1, 0}),
          "param"));
  HloInstruction* copy = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShapeWithLayout(F32, {1, 14, 14, 64}, {0, 1, 2, 3}),
      HloOpcode::kCopy, param));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShapeWithLayout(F32, {14 * 14, 64}, {0, 1}), copy));
  builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShapeWithLayout(F32, {14 * 14, 64}, {1, 0}),
      HloOpcode::kCopy, reshape));
  auto computation = m->AddEntryComputation(builder.Build());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Reshape(m::Copy(m::Parameter(0))))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  // Verify that the copy of reshape of copy is replaced.
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Bitcast(m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, ReshapeOfCopyEqualsBitcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShapeWithLayout(F32, {1, 14, 14, 64}, {3, 2, 1, 0}),
          "param"));
  HloInstruction* copy = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShapeWithLayout(F32, {1, 14, 14, 64}, {0, 1, 2, 3}),
      HloOpcode::kCopy, param));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShapeWithLayout(F32, {14 * 14, 64}, {1, 0}), copy));

  auto computation = m->AddEntryComputation(builder.Build());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Copy(m::Parameter(0)))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  // Verify that the copy of reshape of copy is replaced.
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Bitcast(m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, CopyEqualsBitcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShapeWithLayout(F32, {1, 14, 14, 64}, {0, 1, 2, 3}),
          "param"));
  builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShapeWithLayout(F32, {1, 14, 14, 64}, {1, 2, 0, 3}),
      HloOpcode::kCopy, param));
  auto computation = m->AddEntryComputation(builder.Build());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));

  AlgebraicSimplifierOptions options(
      [](const Shape&, const Shape&) { return false; });
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier1(options);
  ASSERT_FALSE(simplifier1.Run(m.get()).ValueOrDie());
  // Verify that the copy is not replaced.
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));

  AlgebraicSimplifierOptions options2;
  options2.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier2(options2);
  EXPECT_TRUE(simplifier2.Run(m.get()).ValueOrDie());
  // Verify that the copy is replaced.
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Bitcast(m::Parameter(0))));
}

// Test that unary concatenates are removed.
TEST_F(AlgebraicSimplifierTest, RemoveUnaryConcatenate) {
  auto m = CreateNewVerifiedModule();
  Shape r1f32 = ShapeUtil::MakeShape(F32, {100});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  builder.AddInstruction(
      HloInstruction::CreateConcatenate(param0->shape(), {param0}, 0));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Concatenate(m::Parameter(0))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), param0);
}

// Test that empty operands of concatenates are removed.
TEST_F(AlgebraicSimplifierTest, RemoveEmptyConcatenateOperands) {
  auto m = CreateNewVerifiedModule();
  const int kParamLength = 100;
  Shape r1f32 = ShapeUtil::MakeShape(F32, {kParamLength});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r1f32, "param1"));
  HloInstruction* empty_literal = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({})));
  HloInstruction* empty_slice =
      builder.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {0}), param1, {42}, {42}, {1}));
  Shape result_shape = ShapeUtil::MakeShape(F32, {3 * kParamLength});
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      result_shape, {empty_literal, param0, param0, empty_slice, param1}, 0));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Concatenate(
                  m::Op().Is(empty_literal), m::Parameter(0), m::Parameter(0),
                  m::Op().Is(empty_slice), m::Parameter(1))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Concatenate(m::Parameter(0), m::Parameter(0),
                                        m::Parameter(1))));
}

// Test that reduce of concat is simplified.
TEST_F(AlgebraicSimplifierTest, SimplifyReduceOfConcat) {
  auto m = CreateNewVerifiedModule();
  const int kParamLength = 100;
  Shape r3f32 =
      ShapeUtil::MakeShape(F32, {kParamLength, kParamLength, kParamLength});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r3f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r3f32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r3f32, "param2"));
  Shape concat_shape =
      ShapeUtil::MakeShape(F32, {kParamLength, 3 * kParamLength, kParamLength});
  HloInstruction* Concatenate =
      builder.AddInstruction(HloInstruction::CreateConcatenate(
          concat_shape, {param0, param1, param2}, 1));
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
    add_computation = m->AddEmbeddedComputation(builder.Build());
  }
  Shape r4f32 = ShapeUtil::MakeShape(F32, {4, 5, 6, 7});
  Shape reduce_shape = ShapeUtil::MakeShape(F32, {kParamLength});

  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
  builder.AddInstruction(HloInstruction::CreateReduce(
      reduce_shape, Concatenate, zero, {1, 2}, add_computation));

  auto computation = m->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Map(m::Map(m::Reduce(m::Parameter(0), m::Op().Is(zero)),
                               m::Reduce(m::Parameter(1), m::Op().Is(zero))),
                        m::Reduce(m::Parameter(2), m::Op().Is(zero)))));
}

// Test a concatenate with only empty operands is removed.
TEST_F(AlgebraicSimplifierTest, OnlyEmptyConcatenateOperands) {
  auto m = CreateNewVerifiedModule();
  const int kParamLength = 100;
  Shape r1f32 = ShapeUtil::MakeShape(F32, {kParamLength});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* empty_literal = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({})));
  HloInstruction* empty_slice =
      builder.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {0}), param0, {42}, {42}, {1}));
  Shape result_shape = ShapeUtil::MakeShape(F32, {0});
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      result_shape, {empty_literal, empty_slice}, 0));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Concatenate(m::Op().Is(empty_literal),
                                        m::Op().Is(empty_slice))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_EQ(computation->root_instruction(), empty_literal);
}

// Test that concat with a scalar broadcast becomes a pad.
TEST_F(AlgebraicSimplifierTest, ConcatenateOfBroadcastBecomesPad) {
  auto m = CreateNewVerifiedModule();
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
      ShapeUtil::MakeShape(F32, {200}), {broadcast, param0}, 0));

  auto computation = m->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, SimplifyConcatenateOfSlices) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {100, 99});
  Shape concat_shape = ShapeUtil::MakeShape(F32, {50, 80});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r2f32, "param1"));

  HloInstruction* slice0 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param0, /*start_indices=*/{0, 0},
      /*limit_indices=*/{50, 10}, /*strides=*/{1, 1}));

  // Cannot merge 'slice0' and 'slice1' because of different start indices in
  // dimension 0.
  HloInstruction* slice1 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param0, /*start_indices=*/{50, 10},
      /*limit_indices=*/{100, 20}, /*strides=*/{1, 1}));

  // Cannot merge 'slice1' and 'slice2' because of stride in dimension 2.
  HloInstruction* slice2 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param0, /*start_indices=*/{50, 20},
      /*limit_indices=*/{100, 40}, /*strides=*/{1, 2}));

  // Cannot merge 'slice2' and 'slice3' because of stride in dimension 2.
  HloInstruction* slice3 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param0, /*start_indices=*/{50, 40},
      /*limit_indices=*/{100, 50}, /*strides=*/{1, 1}));

  // Can merge 'slice3' and 'slice4'.
  HloInstruction* slice4 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param0, /*start_indices=*/{50, 50},
      /*limit_indices=*/{100, 60}, /*strides=*/{1, 1}));

  // Can merge 'slice4' and 'slice5'.
  HloInstruction* slice5 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param0, /*start_indices=*/{50, 60},
      /*limit_indices=*/{100, 70}, /*strides=*/{1, 1}));

  // Cannot merge 'slice5' and 'slice6' because of overlap.
  HloInstruction* slice6 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param0, /*start_indices=*/{50, 69},
      /*limit_indices=*/{100, 79}, /*strides=*/{1, 1}));

  // Cannot merge 'slice6' and 'slice7' because of slicing from a different
  // parameter.
  HloInstruction* slice7 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param1, /*start_indices=*/{50, 79},
      /*limit_indices=*/{100, 89}, /*strides=*/{1, 1}));

  builder.AddInstruction(HloInstruction::CreateConcatenate(
      concat_shape,
      {slice0, slice1, slice2, slice3, slice4, slice5, slice6, slice7}, 1));
  auto computation = m->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  auto s = m::Slice(m::Parameter(0));
  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Concatenate(s, s, s, s, s, m::Slice(m::Parameter(1)))));
  // The operand 3 should be a merge of 'slice3', 'slice4' and 'slice5', so its
  // shape should have dimensions {50, 30}.
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->operand(3)->shape(),
                       ShapeUtil::MakeShape(F32, {50, 30})));
  EXPECT_EQ(computation->root_instruction()->operand(3)->slice_starts(1), 40);
}

// Test that a simplification which changes layouts is not performed if layout
// sensitive is true.
TEST_F(AlgebraicSimplifierTest, CopyWithDifferentLayout) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 2}), "param0"));
  HloInstruction* copy = builder.AddInstruction(
      HloInstruction::CreateUnary(param0->shape(), HloOpcode::kCopy, param0));

  auto computation = m->AddEntryComputation(builder.Build());

  // Set to different layouts.
  *param0->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  *copy->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({1, 0});

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(m.get()).ValueOrDie());

  // Copy has not been removed.
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));
}

// Test that a simplification which preserves layouts is performed if layout
// sensitive is true.
TEST_F(AlgebraicSimplifierTest, CopyWithSameLayout) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 2}), "param0"));
  HloInstruction* copy = builder.AddInstruction(
      HloInstruction::CreateUnary(param0->shape(), HloOpcode::kCopy, param0));

  auto computation = m->AddEntryComputation(builder.Build());

  // Set to same layouts.
  *param0->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  *copy->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  // Copy has been removed.
  EXPECT_THAT(computation->root_instruction(), param0);
}

// Test that a reshape which could be replaced with a bitcast is not if
// add_bitcasts is false.
TEST_F(AlgebraicSimplifierTest, NoBitcastAdded) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Parameter(0))));

  AlgebraicSimplifierOptions options(
      [](const Shape&, const Shape&) { return false; });
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(m.get()).ValueOrDie());

  // Reshape is not replaced with a bitcast.
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Parameter(0))));
}

// Test transforming reshapes and transposes of rng.
TEST_F(AlgebraicSimplifierTest, ReshapeOfTransposeOfRngToRng) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  HloInstruction* rng0 = builder.AddInstruction(
      HloInstruction::CreateRng(ShapeUtil::MakeShape(F32, {2, 2}),
                                RandomDistribution::RNG_UNIFORM, {zero, one}));

  HloInstruction* transpose = builder.AddInstruction(
      HloInstruction::CreateTranspose(rng0->shape(), rng0, {1, 0}));
  Shape reshape_shape = builder
                            .AddInstruction(HloInstruction::CreateReshape(
                                ShapeUtil::MakeShape(F32, {4}), transpose))
                            ->shape();

  auto computation = m->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(AlgebraicSimplifierOptions{});
  EXPECT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  // Verify that reshape(transpose(rng)) is replace by a single rng of the
  // same shape as the reshape.
  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Rng()));
  EXPECT_TRUE(ShapeUtil::Equal(computation->root_instruction()->shape(),
                               reshape_shape));
}

// Test transforming reshapes to bitcasts under various conditions.
TEST_F(AlgebraicSimplifierTest, ReshapeReplacedWithBitcast) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Tuple(m::Op().Is(transformable_reshape),
                                  m::Op().Is(dimensions_wrong_reshape),
                                  m::Op().Is(layout_wrong_reshape))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  simplifier.Run(m.get()).ValueOrDie();

  // Verify that only the first reshape is replaced.
  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Tuple(m::Bitcast(), m::Op().Is(dimensions_wrong_reshape),
                          m::Op().Is(layout_wrong_reshape))));
}

// Regression test for a bug where if we failed to sink a reshape, we'd set the
// 'changed' bit in AlgebraicSimplifier to false.
TEST_F(AlgebraicSimplifierTest, FailureToSinkReshapeDoesntAffectChangedBit) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  // This add (param0 + 0) can be simplified.
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd,
      builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape, "param0")),
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<float>({{0, 0}, {0, 0}})))));

  builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {4}), add));

  AlgebraicSimplifier simplifier(AlgebraicSimplifierOptions{});
  m->AddEntryComputation(builder.Build());
  EXPECT_TRUE(simplifier.Run(m.get()).ValueOrDie());
}

// Regression test for a bug where if we failed to sink a reshape, we'd set the
// 'changed' bit in AlgebraicSimplifier to false.
TEST_F(AlgebraicSimplifierTest, FailureToSinkBroadcastDoesntAffectChangedBit) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  // This add (param0 + 0) can be simplified.
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd,
      builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape, "param0")),
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<float>({{0, 0}, {0, 0}})))));

  builder.AddInstruction(
      HloInstruction::CreateBroadcast(ShapeUtil::MakeShape(F32, {2, 2, 2}), add,
                                      /*broadcast_dimensions=*/{0, 1}));

  AlgebraicSimplifier simplifier(AlgebraicSimplifierOptions{});
  m->AddEntryComputation(builder.Build());
  EXPECT_TRUE(simplifier.Run(m.get()).ValueOrDie());
}

TEST_F(AlgebraicSimplifierTest, TransposeEqualsBitcast1) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Transpose(m::Parameter(0))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  // Verify that the reshape is replaced.
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Bitcast(m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, TransposeEqualsBitcast2) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Transpose(m::Parameter(0))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  // Verify that the reshape is replaced.
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Bitcast(m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, ReshapesMerged) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 2}), "param0"));

  HloInstruction* reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {2, 1, 2}), param0));

  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {1, 2, 1, 1, 2, 1}), reshape1));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Reshape(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, CopiesMerged) {
  auto m = CreateNewVerifiedModule();
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

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Copy(m::Parameter(0)))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, TransposesMerged) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 3, 4}), "param0"));

  HloInstruction* transpose1 =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {3, 4, 2}), param0, {1, 2, 0}));

  builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 3, 2}), transpose1, {1, 0, 2}));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Transpose(m::Op().Is(transpose1))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Transpose(m::Parameter(0))));
  EXPECT_EQ(std::vector<int64>({2, 1, 0}),
            computation->root_instruction()->dimensions());
}

TEST_F(AlgebraicSimplifierTest, TransposeIsReshape) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      param = f32[10] parameter(0)
      reshaped = f32[1,1,10] reshape(f32[10] param)
      transposed = f32[10,1,1] transpose(f32[1,1,10] reshaped), dimensions={2,1,0}
      ROOT reshaped_again = f32[10] reshape(f32[10,1,1] transposed)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter()));
}

// Test merging reshape and broadcast.
TEST_F(AlgebraicSimplifierTest, ReshapeAndBroadcastMerged) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {5}), "param0"));
  auto reshape1 = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {1, 5, 1}), param0));
  builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {1, 2, 3, 5, 1}), reshape1, {0, 3, 2}));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Reshape(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Parameter(0))));
}

// Test merging broadcast and reshape.
TEST_F(AlgebraicSimplifierTest, BroadcastAndReshapeMerged) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {2, 3}), "param0"));
  auto broadcast1 = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {1, 2, 3, 7, 12, 1}), param0, {1, 2}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {2, 3, 7, 2, 1, 3, 2}), broadcast1));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, BroadcastAndReshape_1_3x1_3) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 1}), param, {1}));
  builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {3}), broadcast));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  EXPECT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Reshape(m::Parameter(0)))));
}

TEST_F(AlgebraicSimplifierTest, BroadcastAndReshape_4_3x2x4_6x1x1x4) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {4}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 2, 4}), param, {2}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 1, 1, 4}), broadcast));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Parameter(0))));
  EXPECT_THAT(computation->root_instruction()->dimensions(),
              ::testing::ElementsAre(3));
}

TEST_F(AlgebraicSimplifierTest, BroadcastAndReshape_1_3x2x1_6x1x1x1) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 2, 1}), param, {2}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 1, 1, 1}), broadcast));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Reshape(m::Parameter(0)))));
  EXPECT_EQ(0, computation->root_instruction()->dimensions().size());
}

TEST_F(AlgebraicSimplifierTest, BroadcastAndReshape_4_3x2x4x2_6x8) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto param = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {4}), "param"));
  auto broadcast = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(F32, {3, 2, 4, 2}), param, {2}));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 8}), broadcast));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  EXPECT_FALSE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));
}

TEST_F(AlgebraicSimplifierTest, IotaAndReshapeMerged) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(HloInstruction::CreateIota(
      ShapeUtil::MakeShape(F32, {1, 2, 3, 7, 12, 1}), 2));
  Shape result_shape = ShapeUtil::MakeShape(F32, {2, 3, 7, 2, 1, 3, 2});
  builder.AddInstruction(HloInstruction::CreateReshape(result_shape, iota));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Iota()));
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), result_shape));
}

TEST_F(AlgebraicSimplifierTest, IotaEffectiveScalar) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(F32, {1, 1}), 0));
  auto result_shape = iota->shape();

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Iota()));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  auto root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Constant())));
  EXPECT_EQ(0.0f, root->operand(0)->literal().GetFirstElement<float>());
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), result_shape));
}

TEST_F(AlgebraicSimplifierTest, IotaAndReshape_1_3x2_6) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(F32, {3, 2}), 1));
  builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {6}), iota));

  auto computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  AlgebraicSimplifier simplifier(default_options_);
  EXPECT_FALSE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));
}

TEST_F(AlgebraicSimplifierTest, IotaAndReshape_4_3x2x4_6x1x1x4) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(F32, {3, 2, 4}), 2));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 1, 1, 4}), iota));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Iota()));
  EXPECT_EQ(Cast<HloIotaInstruction>(computation->root_instruction())
                ->iota_dimension(),
            3);
}

TEST_F(AlgebraicSimplifierTest, IotaAndReshape_1_3x2x2_6x1x1x2) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(F32, {3, 2, 2}), 2));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {6, 1, 1, 2}), iota));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Iota()));
  const int64 iota_dim =
      Cast<HloIotaInstruction>(computation->root_instruction())
          ->iota_dimension();
  EXPECT_THAT(iota_dim, ::testing::AnyOf(1, 2, 3));
}

TEST_F(AlgebraicSimplifierTest, IotaAndReshape_4_3x2x4x2_6x8) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(F32, {3, 2, 4, 2}), 2));
  builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {6, 8}), iota));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  AlgebraicSimplifier simplifier(default_options_);
  EXPECT_FALSE(simplifier.Run(m.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));
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

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Op().Is(zero))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

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

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);

  auto has_negative_padding = [](const HloInstruction* pad) {
    for (auto& padding_dimension : pad->padding_config().dimensions()) {
      if (padding_dimension.edge_padding_low() < 0 ||
          padding_dimension.edge_padding_high() < 0) {
        return true;
      }
    }
    return false;
  };

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Op().Is(zero))));
  EXPECT_TRUE(has_negative_padding(pad));

  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Pad(m::Parameter(0), m::Op().Is(zero)))));
  EXPECT_FALSE(
      has_negative_padding(computation->root_instruction()->operand(0)));
}

TEST_F(AlgebraicSimplifierTest, TrivialInteriorPadding) {
  // Verify that a pad instruction with interior padding on one-sized
  // dimensions, removes the interior padding.
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 1}), "param"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  PaddingConfig padding;
  for (int i = 0; i < 2; ++i) {
    auto dimension = padding.add_dimensions();
    dimension->set_edge_padding_low(3);
    dimension->set_edge_padding_high(3);
    dimension->set_interior_padding(i * 3);
  }
  HloInstruction* pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(F32, {8, 7}), param, zero, padding));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);

  ASSERT_THAT(computation->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Op().Is(zero))));
  ASSERT_TRUE(HasInteriorPadding(pad->padding_config()));

  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Op().Is(zero))));
  EXPECT_FALSE(
      HasInteriorPadding(computation->root_instruction()->padding_config()));
}

TEST_F(AlgebraicSimplifierTest, RemoveNoopReshape) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 3}), "param"));
  builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {2, 3}), param));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Parameter(0))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

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

  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Parameter(0))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(), param);
}

TEST_F(AlgebraicSimplifierTest, SliceOfSliceToSlice) {
  HloComputation::Builder builder(TestName());
  const int64 dim0 = 11;
  const int64 dim1 = 12;
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {dim0, dim1}), "param"));
  HloInstruction* original_slice =
      builder.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {dim0 - 2, dim1 - 4}), param,
          /*start_indices=*/{1, 2},
          /*limit_indices=*/{dim0 - 1, dim1 - 2}, /*strides=*/{1, 1}));

  builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {dim0 - 5, dim1 - 9}), original_slice,
      /*start_indices=*/{2, 3},
      /*limit_indices=*/{dim0 - 3, dim1 - 6}, /*strides=*/{1, 1}));
  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Slice(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Parameter(0))));
  EXPECT_EQ(computation->root_instruction()->slice_starts(0), 3);
  EXPECT_EQ(computation->root_instruction()->slice_starts(1), 5);
  EXPECT_EQ(computation->root_instruction()->slice_limits(0), dim0 - 2);
  EXPECT_EQ(computation->root_instruction()->slice_limits(1), dim1 - 4);
}

TEST_F(AlgebraicSimplifierTest, SliceOfBroadcastToBroadcast) {
  HloComputation::Builder builder(TestName());
  const int64 dim0 = 11;
  const int64 dim1 = 12;
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {dim0}), "param"));
  HloInstruction* broadcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::MakeShape(F32, {dim0, dim1}), param, {0}));
  builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {dim0, dim1 - 9}), broadcast,
      /*start_indices=*/{0, 3},
      /*limit_indices=*/{dim0, dim1 - 6}, /*strides=*/{1, 1}));
  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Broadcast(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, SliceOfReshapeToReshapeOfSlice) {
  HloComputation::Builder builder(TestName());
  const int64 dim0 = 11;
  const int64 dim1 = 12;
  const int64 dim2 = 13;
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {dim0 * dim1, dim2}), "param"));
  HloInstruction* original_reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {dim0, dim1, dim2}), param));

  builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {dim0 - 2, dim1, dim2}), original_reshape,
      /*start_indices=*/{0, 0, 0},
      /*limit_indices=*/{dim0 - 2, dim1, dim2}, /*strides=*/{1, 1, 1}));
  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Reshape(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Slice(m::Parameter(0)))));
}

TEST_F(AlgebraicSimplifierTest, SliceOfReshapeUnchanged) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1, 144, 25, 1, 512}), "param"));
  HloInstruction* original_reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {3600, 512}), param));

  builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {960, 512}), original_reshape,
      /*start_indices=*/{0, 0},
      /*limit_indices=*/{960, 512}, /*strides=*/{1, 1}));
  auto module = CreateNewVerifiedModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Reshape(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(module.get()).ValueOrDie());
}

TEST_F(AlgebraicSimplifierTest, RemoveNoopSort) {
  auto builder = HloComputation::Builder(TestName());
  auto module = CreateNewVerifiedModule();

  Shape keys_shape = ShapeUtil::MakeShape(F32, {1});
  auto keys = builder.AddInstruction(
      HloInstruction::CreateParameter(0, keys_shape, "keys"));
  TF_ASSERT_OK(MakeSortHlo(keys_shape, {keys}, 0, /*is_stable=*/false, &builder,
                           module.get())
                   .status());
  HloComputation* computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), keys);
}

TEST_F(AlgebraicSimplifierTest, ReplaceEffectiveScalarKeyValueSortWithTuple) {
  auto builder = HloComputation::Builder(TestName());
  auto module = CreateNewVerifiedModule();

  Shape keys_shape = ShapeUtil::MakeShape(F32, {5, 0});
  Shape values_shape = ShapeUtil::MakeShape(S32, {5, 0});
  auto keys = builder.AddInstruction(
      HloInstruction::CreateParameter(0, keys_shape, "keys"));
  auto values0 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, values_shape, "values0"));
  auto values1 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, values_shape, "values1"));
  TF_ASSERT_OK(MakeSortHlo(ShapeUtil::MakeTupleShape(
                               {keys_shape, values_shape, values_shape}),
                           {keys, values0, values1}, 0, /*is_stable=*/false,
                           &builder, module.get())
                   .status());
  HloComputation* computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Tuple(m::Op().Is(keys), m::Op().Is(values0),
                                  m::Op().Is(values1))));
}

// Test that A && True is simplified to A
TEST_F(AlgebraicSimplifierTest, AndTrue) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_true = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0pred, HloOpcode::kAnd,
                                                      param0, const_true));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAnd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that True && A is simplified to A
TEST_F(AlgebraicSimplifierTest, AndTrue2) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_true = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0pred, HloOpcode::kAnd,
                                                      const_true, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAnd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that A && False is simplified to False
TEST_F(AlgebraicSimplifierTest, AndFalse) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_false = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0pred, HloOpcode::kAnd,
                                                      param0, const_false));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAnd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, const_false);
}

// Test that False && A is simplified to False
TEST_F(AlgebraicSimplifierTest, AndFalse2) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_false = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0pred, HloOpcode::kAnd,
                                                      const_false, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAnd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, const_false);
}

// Test that A || True is simplified to True
TEST_F(AlgebraicSimplifierTest, OrTrue) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_true = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0pred, HloOpcode::kOr, param0, const_true));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kOr);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, const_true);
}

// Test that True || A is simplified to True
TEST_F(AlgebraicSimplifierTest, OrTrue2) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_true = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0pred, HloOpcode::kOr, const_true, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kOr);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, const_true);
}

// Test that A || False is simplified to A
TEST_F(AlgebraicSimplifierTest, OrFalse) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_false = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0pred, HloOpcode::kOr,
                                                      param0, const_false));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kOr);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that False || A is simplified to A
TEST_F(AlgebraicSimplifierTest, OrFalse2) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_false = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateBinary(r0pred, HloOpcode::kOr,
                                                      const_false, param0));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kOr);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Used for TEST_Ps that test merging (or not) of a kPad instruction into a
// convolution's Window.
struct ConvPaddingTestcase {
  ConvPaddingTestcase(absl::string_view padding,
                      absl::string_view orig_conv_window,
                      absl::string_view expected_conv_window)
      : ConvPaddingTestcase(padding, orig_conv_window, expected_conv_window,
                            /*pad_value=*/0) {}

  ConvPaddingTestcase(absl::string_view padding,
                      absl::string_view orig_conv_window,
                      absl::string_view expected_conv_window, float pad_value)
      : padding(padding),
        orig_conv_window(orig_conv_window),
        expected_conv_window(expected_conv_window),
        pad_value(pad_value) {}

  string ToString() const {
    return absl::StrFormat(
        "padding=%s, orig_conv_window=%s, expected_conv_window=%s, "
        "pad_value=%f",
        padding, orig_conv_window, expected_conv_window, pad_value);
  }

  string padding;
  string orig_conv_window;
  string expected_conv_window;
  float pad_value;
};

// ConvInputPaddingTest (and its one associated TEST_P testcase) checks that a
// computation that does
//
//   conv(pad(param0, padding=padding), param1), window=orig_conv_window
//
// gets transformed by AlgebraicSimplifier to
//
//   conv(param0, param1), window=expected_conv_window
//
// or, if expected_conv_window is the empty string, checks that
// AlgebraicSimplifier does *not* transform the original convolution.
class ConvInputPaddingTest
    : public AlgebraicSimplifierTest,
      public ::testing::WithParamInterface<ConvPaddingTestcase> {};

INSTANTIATE_TEST_SUITE_P(
    ConvInputPaddingTestCases, ConvInputPaddingTest,
    ::testing::ValuesIn(std::vector<ConvPaddingTestcase>{
        // Merge this edge padding into the conv.
        {"0_0x0_0x1_1x2_2", "", "pad=1_1x2_2"},
        // Merge this edge padding with the conv's edge padding.
        {"0_0x0_0x1_2x3_4", "pad=10_10x20_20", "pad=11_12x23_24"},
        // Merge this interior-padded kPad with the unpadded conv.  The 3x6
        // interior padding gets transformed to 4x7 conv lhs dilation.
        {"0_0x0_0x1_2_3x4_5_6", "", "pad=1_2x4_5 lhs_dilate=4x7"},
        // kPad has dilation on one dim, conv has it on the other; merge them.
        {"0_0x0_0x0_0_1x0_0_0", "lhs_dilate=1x10", "lhs_dilate=2x10"},
        // kPad has dilation and edge padding on one dim, conv has them on the
        // other; merge them.
        {"0_0x0_0x0_1_1x0_0_0", "pad=0_0x3_0 lhs_dilate=1x10",
         "pad=0_1x3_0 lhs_dilate=2x10"},

        // Don't transform if the pad value is nonzero.
        {"0_0x0_0x1_1x2_2", "", "", /*pad_value=*/1},

        // We refuse to transform the following because on some dimension, one
        // of the kPad and conv has dilation and the other has some sort of
        // padding.
        {"0_0x0_0x0_0_1x0_0", "pad=1_0x0_0", ""},
        {"0_0x0_0x0_0_1x0_0", "pad=0_1x0_0", ""},
        {"0_0x0_0x0_0_1x0_0", "lhs_dilate=2x1", ""},
        {"0_0x0_0x1_0_0x0_0", "lhs_dilate=2x1", ""},
        {"0_0x0_0x0_1_0x0_0", "lhs_dilate=2x1", ""},
        {"0_0x0_0x0_0_1x0_0", "lhs_dilate=2x1", ""},

        // We can't merge feature or batch padding into the conv.
        {"1_0x0_0x0_0x0_0", "", ""},
        {"0_0x1_0x0_0x0_0", "", ""},
    }));

TEST_P(ConvInputPaddingTest, DoTest) {
  ConvPaddingTestcase testcase = GetParam();

  // It would be better to put the testcase's ToString into the test name, but
  // gUnit has constraints on what can go into test names, and any reasonable
  // implementation of ToString() seems to violate them.
  SCOPED_TRACE(testcase.ToString());

  auto builder = HloComputation::Builder(TestName());
  auto* input = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1024, 128, 100, 100}),  // bf01
      "input"));
  auto* pad_value = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR0(testcase.pad_value)));

  PaddingConfig padding_config =
      ParsePaddingConfig(testcase.padding).ValueOrDie();
  auto* lhs_pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeInference::InferPadShape(input->shape(), pad_value->shape(),
                                    padding_config)
          .ValueOrDie(),
      input, pad_value, padding_config));

  auto* filter = builder.AddInstruction(HloInstruction::CreateParameter(
      1,
      ShapeUtil::MakeShape(
          F32, {lhs_pad->shape().dimensions(1), 256, 3, 3}),  // io01
      "input"));

  ConvolutionDimensionNumbers dnums =
      ParseConvolutionDimensionNumbers("bf01_io01->bf01").ValueOrDie();
  Window window =
      ParseWindow(absl::StrCat("size=3x3 ", testcase.orig_conv_window))
          .ValueOrDie();
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeInference::InferConvolveShape(lhs_pad->shape(), filter->shape(),
                                         /*feature_group_count=*/1,
                                         /*batch_group_count=*/1, window, dnums)
          .ValueOrDie(),
      lhs_pad, filter, /*feature_group_count=*/1, /*batch_group_count=*/1,
      window, dnums, DefaultPrecisionConfig(2)));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  if (testcase.expected_conv_window.empty()) {
    ASSERT_FALSE(simplifier.Run(module.get()).ValueOrDie());
  } else {
    ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
    auto* conv = module->entry_computation()->root_instruction();
    SCOPED_TRACE(module->ToString());
    ASSERT_THAT(conv,
                GmockMatch(m::Convolution(m::Parameter(), m::Parameter())));
    EXPECT_EQ(window_util::ToString(conv->window()),
              absl::StrCat("size=3x3 ", testcase.expected_conv_window));
  }
}

// ConvFilterPaddingTest (and its one associated TEST_P) checks that a
// computation that does
//
//   conv(param0, pad(param1, padding=padding)), window=orig_conv_window
//
// gets transformed by AlgebraicSimplifier to
//
//   conv(param0, param1), window=expected_conv_window
//
// or, if expected_conv_window is the empty string, checks that
// AlgebraicSimplifier does *not* transform the original convolution.
class ConvFilterPaddingTest
    : public AlgebraicSimplifierTest,
      public ::testing::WithParamInterface<ConvPaddingTestcase> {};

INSTANTIATE_TEST_SUITE_P(
    ConvFilterPaddingTestCases, ConvFilterPaddingTest,
    ::testing::ValuesIn(std::vector<ConvPaddingTestcase>{
        // Can only merge interior padding on the filter's spatial dimensions;
        // all
        // other paddings (edge padding and interior padding on the channel
        // dims)
        // should be rejected out of hand.
        {"1_0_0x0_0_0x0_0x0_0", "", ""},
        {"0_1_0x0_0_0x0_0x0_0", "", ""},
        {"0_0_1x0_0_0x0_0x0_0", "", ""},
        {"0_0_0x1_0_0x0_0x0_0", "", ""},
        {"0_0_0x0_1_0x0_0x0_0", "", ""},
        {"0_0_0x0_0_1x0_0x0_0", "", ""},
        {"0_0_0x0_0_0x1_0x0_0", "", ""},
        {"0_0_0x0_0_0x0_1x0_0", "", ""},
        {"0_0_0x0_0_0x0_0x1_0", "", ""},
        {"0_0_0x0_0_0x0_0x0_1", "", ""},

        // Interior padding on channel dims can be merged into the conv, so long
        // as the conv and pad don't have interior padding on the same dim.
        {"0_0x0_0x0_0_5x0_0", "", "rhs_dilate=6x1"},
        {"0_0x0_0x0_0x0_0_10", "", "rhs_dilate=1x11"},
        {"0_0x0_0x0_0_10x0_0_100", "", "rhs_dilate=11x101"},
        {"0_0x0_0x0_0_1x0_0", "rhs_dilate=1x10", "rhs_dilate=2x10"},
        {"0_0x0_0x0_0x0_0_5", "rhs_dilate=10x1", "rhs_dilate=10x6"},

        // Can't merge if for a given dim there's interior padding on both the
        // pad and conv.
        {"0_0x0_0x0_0_1x0_0", "rhs_dilate=2x10", ""},
        {"0_0x0_0x0_0x0_0_5", "rhs_dilate=10x2", ""},

        // Don't transform if the pad value is nonzero.
        {"0_0x0_0x0_0_5x0_0", "", "", /*pad_value=*/1},
    }));

TEST_P(ConvFilterPaddingTest, DoIt) {
  ConvPaddingTestcase testcase = GetParam();

  // It would be better to put the testcase's ToString into the test name, but
  // gUnit has constraints on what can go into test names, and any reasonable
  // implementation of ToString() seems to violate them.
  SCOPED_TRACE(testcase.ToString());

  auto builder = HloComputation::Builder(TestName());
  auto* pad_value = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR0(testcase.pad_value)));
  auto* filter = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {128, 256, 3, 3}),  // io01
      "input"));
  PaddingConfig padding_config =
      ParsePaddingConfig(testcase.padding).ValueOrDie();
  auto* rhs_pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeInference::InferPadShape(filter->shape(), pad_value->shape(),
                                    padding_config)
          .ValueOrDie(),
      filter, pad_value, padding_config));

  auto* input = builder.AddInstruction(HloInstruction::CreateParameter(
      0,
      ShapeUtil::MakeShape(
          F32, {1024, rhs_pad->shape().dimensions(0), 100, 100}),  // bf01
      "input"));

  ConvolutionDimensionNumbers dnums =
      ParseConvolutionDimensionNumbers("bf01_io01->bf01").ValueOrDie();
  Window window = ParseWindow(absl::StrFormat("size=%dx%d %s",
                                              rhs_pad->shape().dimensions(2),
                                              rhs_pad->shape().dimensions(3),
                                              testcase.orig_conv_window))
                      .ValueOrDie();

  // Add a PrecisionConfig and check that AlgebraicSimplifier keeps it in place
  // after the transformation.
  PrecisionConfig precision_config;
  precision_config.add_operand_precision(PrecisionConfig::HIGH);
  precision_config.add_operand_precision(PrecisionConfig::HIGHEST);

  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeInference::InferConvolveShape(input->shape(), rhs_pad->shape(),
                                         /*feature_group_count=*/1,
                                         /*batch_group_count=*/1, window, dnums)
          .ValueOrDie(),
      input, rhs_pad, /*feature_group_count=*/1, /*batch_group_count=*/1,
      window, dnums, precision_config));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  if (testcase.expected_conv_window.empty()) {
    ASSERT_FALSE(simplifier.Run(module.get()).ValueOrDie());
  } else {
    ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
    auto* conv = module->entry_computation()->root_instruction();
    SCOPED_TRACE(module->ToString());
    ASSERT_THAT(conv,
                GmockMatch(m::Convolution(m::Parameter(), m::Parameter())));
    EXPECT_EQ(window_util::ToString(conv->window()),
              absl::StrFormat("size=%dx%d %s",
                              conv->operand(1)->shape().dimensions(2),
                              conv->operand(1)->shape().dimensions(3),
                              testcase.expected_conv_window));
    EXPECT_THAT(Cast<HloConvolutionInstruction>(conv)
                    ->precision_config()
                    .operand_precision(),
                ElementsAre(PrecisionConfig::HIGH, PrecisionConfig::HIGHEST));
  }
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
  auto build_and_simplify = [&]() -> string {
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

    auto make_shape = [](absl::Span<const int64> dims,
                         bool minor_to_major_layout) {
      if (minor_to_major_layout) {
        return ShapeUtil::MakeShapeWithLayout(F32, dims, {0, 1, 2, 3});
      } else {
        return ShapeUtil::MakeShape(F32, dims);
      }
    };
    auto in_shape = make_shape(in_dims, options.input_minor_to_major_layout);
    auto f_shape = make_shape(f_dims, options.filter_minor_to_major_layout);

    HloInstruction* input =
        b.AddInstruction(HloInstruction::CreateParameter(0, in_shape, "input"));
    HloInstruction* filter =
        b.AddInstruction(HloInstruction::CreateParameter(1, f_shape, "filter"));
    Shape out_shape = ShapeInference::InferConvolveShape(
                          in_shape, f_shape, /*feature_group_count=*/1,
                          /*batch_group_count=*/1, window, dnums)
                          .ValueOrDie();
    if (options.output_minor_to_major_layout) {
      out_shape = ShapeUtil::MakeShapeWithLayout(F32, out_shape.dimensions(),
                                                 {0, 1, 2, 3});
    }

    b.AddInstruction(HloInstruction::CreateConvolve(
        out_shape, input, filter,
        /*feature_group_count=*/1, /*batch_group_count=*/1, window, dnums,
        DefaultPrecisionConfig(2)));

    auto module = CreateNewVerifiedModule();
    auto* computation = module->AddEntryComputation(b.Build());

    AlgebraicSimplifierOptions simplifier_options;
    simplifier_options.set_is_layout_sensitive(true);
    AlgebraicSimplifier simplifier(simplifier_options);
    if (!simplifier.Run(module.get()).ValueOrDie()) {
      return "NO_CHANGE";
    }
    auto* root = computation->root_instruction();
    if (root->opcode() == HloOpcode::kBitcast &&
        root->operand(0)->opcode() == HloOpcode::kDot) {
      auto lhs_shape = root->operand(0)->operand(0)->shape();
      auto rhs_shape = root->operand(0)->operand(1)->shape();
      return absl::StrCat(absl::StrJoin(lhs_shape.dimensions(), "x"), " DOT ",
                          absl::StrJoin(rhs_shape.dimensions(), "x"));
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

// Test that slice(broadcast(/*scalar value*/)) simplifies to a single
// broadcast.
TEST_F(AlgebraicSimplifierTest, ScalarBroadcastToSlice) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* scalar_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "scalar_param"));

  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {4, 5, 6, 7});
  HloInstruction* broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(broadcast_shape, scalar_param, {}));

  Shape slice_shape = ShapeUtil::MakeShape(F32, {2, 2, 3, 3});
  HloInstruction* slice = builder.AddInstruction(HloInstruction::CreateSlice(
      slice_shape, broadcast, {0, 1, 2, 3}, {2, 3, 5, 6}, {1, 1, 1, 1}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, slice);
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), slice_shape));

  AlgebraicSimplifier simplifier(default_options_);

  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  // Running simplification again should not result in any further changes.
  ASSERT_FALSE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Op().Is(scalar_param))
                             .WithShapeEqualTo(&slice_shape)));
}

// Test that reshape(transpose(broadcast(/*scalar value*/))) simplifies to a
// single broadcast.
TEST_F(AlgebraicSimplifierTest, ScalarBroadcastToTransposeReshape) {
  HloComputation::Builder builder(TestName());
  HloInstruction* forty_two = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));

  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {4, 5, 6});
  HloInstruction* broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(broadcast_shape, forty_two, {}));

  HloInstruction* transpose =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {6, 5, 4}), broadcast, {2, 1, 0}));

  Shape reshape_shape = ShapeUtil::MakeShape(F32, {30, 1, 4});
  HloInstruction* reshape = builder.AddInstruction(
      HloInstruction::CreateReshape(reshape_shape, transpose));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, reshape);
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), reshape_shape));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Op().Is(forty_two))
                             .WithShapeEqualTo(&reshape_shape)));
}

// Test that ReduceWindow(Pad(op, x), y) can simplify to ReduceWindow(op, x).
TEST_F(AlgebraicSimplifierTest, FoldPadIntoReduceWindow) {
  auto module = CreateNewVerifiedModule();
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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(5.0f)));
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
    add_computation = module->AddEmbeddedComputation(builder.Build());
  }

  // Create the reduce-window.
  Window window;
  for (int64 i = 0; i < pad->shape().rank(); ++i) {
    auto* dim = window.add_dimensions();
    dim->set_size(1);
    dim->set_padding_low(10);
    dim->set_padding_high(100);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
    dim->set_stride(1);
  }
  const Shape reduce_window_shape =
      ShapeUtil::MakeShape(F32, {111, 113, 113, 116});
  HloInstruction* reduce_init_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(5.0f)));
  HloInstruction* reduce_window =
      builder.AddInstruction(HloInstruction::CreateReduceWindow(
          reduce_window_shape, pad, reduce_init_value, window,
          add_computation));

  // Build the computation and run the simplifier.
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, reduce_window);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  // Running simplification again should not result in any further changes.
  ASSERT_FALSE(simplifier.Run(module.get()).ValueOrDie());

  // Verify the result
  root = computation->root_instruction();
  EXPECT_THAT(root,
              GmockMatch(m::ReduceWindow(m::Op().Is(operand), m::Constant())));
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

// Test that ReduceWindow(Convert(Pad(op, x)), y) can simplify to
// ReduceWindow(Convert(op), x).
TEST_F(AlgebraicSimplifierTest, FoldConvertedPadIntoReduceWindow) {
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  // Create operand to the pad.
  HloInstruction* parameter =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(BF16, {1, 2, 3, 4}), "p0"));

  // Create the pad.
  PaddingConfig padding = MakeNoPaddingConfig(4);
  padding.mutable_dimensions(1)->set_edge_padding_low(1);
  padding.mutable_dimensions(3)->set_edge_padding_high(2);

  HloInstruction* pad_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(5.0f)));
  HloInstruction* pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(BF16, {1, 3, 3, 5}), parameter, pad_value, padding));

  HloInstruction* convert =
      builder.AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(pad->shape(), F32), pad));

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
    add_computation = module->AddEmbeddedComputation(builder.Build());
  }

  // Create the reduce-window.
  Window window;
  for (int64 i = 0; i < pad->shape().rank(); ++i) {
    auto* dim = window.add_dimensions();
    dim->set_size(1);
    dim->set_padding_low(10);
    dim->set_padding_high(100);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
    dim->set_stride(1);
  }
  const Shape reduce_window_shape =
      ShapeUtil::MakeShape(F32, {111, 113, 113, 116});
  HloInstruction* reduce_init_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(5.0f)));
  HloInstruction* reduce_window =
      builder.AddInstruction(HloInstruction::CreateReduceWindow(
          reduce_window_shape, convert, reduce_init_value, window,
          add_computation));

  // Build the computation and run the simplifier.
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, reduce_window);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  // Running simplification again should not result in any further changes.
  ASSERT_FALSE(simplifier.Run(module.get()).ValueOrDie());

  // Verify the result
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::ReduceWindow(m::Convert(m::Parameter(0)),
                                               m::Constant())));
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

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).ValueOrDie());

  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(a, root);
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), shape));
}

TEST_F(AlgebraicSimplifierTest, IteratorInvalidation) {
  // Dots add computations to the parent module. Test that, when the HloModule's
  // computations are updated, then iterator invalidation doesn't occur
  // when running on subsequent computations.
  auto m = CreateNewVerifiedModule();
  Shape r1f32 = ShapeUtil::MakeShape(F32, {1});
  HloComputation::Builder builder(TestName() + ".Dot");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r1f32, "x"));
  HloInstruction* y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r1f32, "y"));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(0);
  builder.AddInstruction(HloInstruction::CreateDot(r1f32, x, y, dot_dnums,
                                                   DefaultPrecisionConfig(2)));
  std::unique_ptr<HloComputation> dot_computation(builder.Build());

  HloComputation::Builder call_builder(TestName() + ".Call");
  HloInstruction* zero = call_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({0.0f})));
  HloInstruction* one = call_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({1.0f})));
  call_builder.AddInstruction(
      HloInstruction::CreateCall(r1f32, {zero, one}, dot_computation.get()));

  m->AddEmbeddedComputation(std::move(dot_computation));
  m->AddEntryComputation(call_builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
}

// Test that a constant with tuple shape becomes a tuple of constants.
TEST_F(AlgebraicSimplifierTest, ConstantTupleBecomesTupleOfConstants) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  const float constant_scalar = 7.3f;
  std::initializer_list<float> constant_vector = {1.1f, 2.0f, 3.3f};
  Literal elements[] = {LiteralUtil::CreateR0<float>(constant_scalar),
                        LiteralUtil::CreateR1<float>(constant_vector)};
  Literal value = LiteralUtil::MakeTuple({&elements[0], &elements[1]});
  builder.AddInstruction(HloInstruction::CreateConstant(std::move(value)));

  auto computation = m->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Tuple(m::Constant(), m::Constant())));
}

// A dynamic-slice is trivial if its start indices are all zeroes and the size
// of its input equals the size of its output.  In this case, the dynamic slice
// is equal to its input.
TEST_F(AlgebraicSimplifierTest, TrivialDynamicSlice) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape shape = ShapeUtil::MakeShape(F32, {10, 100, 1000});
  std::vector<HloInstruction*> params;
  for (int i = 0; i < 3; ++i) {
    params.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
        i + 1, ShapeUtil::MakeShape(U32, {}), "slice_indices")));
  }
  builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      shape,
      builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape, "slice_from")),
      params,
      /*slice_sizes=*/{10, 100, 1000}));

  auto computation = m->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Parameter()));
}

// A dynamic-update-slice is trivial if its start indices are all zeroes and the
// size of its "update" equals the size of its output.  In this case, the
// dynamic-update-slice is equal to its update.
TEST_F(AlgebraicSimplifierTest, TrivialDynamicUpdateSlice) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape full_shape = ShapeUtil::MakeShape(F32, {10, 100, 1000});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {10, 1, 1000});

  std::vector<HloInstruction*> slice_indices, update_indices;
  for (int i = 0; i < 3; ++i) {
    slice_indices.push_back(
        builder.AddInstruction(HloInstruction::CreateParameter(
            i + 1, ShapeUtil::MakeShape(U32, {}), "slice_indices")));
    update_indices.push_back(
        builder.AddInstruction(HloInstruction::CreateParameter(
            i + 5, ShapeUtil::MakeShape(U32, {}), "update_indices")));
  }
  HloInstruction* slice =
      builder.AddInstruction(HloInstruction::CreateDynamicSlice(
          slice_shape,
          builder.AddInstruction(
              HloInstruction::CreateParameter(0, full_shape, "slice_from")),
          slice_indices,
          /*slice_sizes=*/{10, 1, 1000}));

  builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      slice_shape,
      builder.AddInstruction(
          HloInstruction::CreateParameter(4, slice_shape, "to_update")),
      slice, update_indices));

  auto computation = m->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::DynamicSlice(m::Parameter(), m::Parameter(),
                                         m::Parameter(), m::Parameter())));
}

// Test that two consecutive broadcasts can be merged to one.
TEST_F(AlgebraicSimplifierTest, MergeBroadcasts) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 2});
  HloInstruction* input_array = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({3, 4})));
  HloInstruction* inner_bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(r2f32, input_array, {1}));
  Shape r3f32 = ShapeUtil::MakeShape(F32, {2, 2, 2});
  builder.AddInstruction(
      HloInstruction::CreateBroadcast(r3f32, inner_bcast, {0, 2}));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Constant())));
  EXPECT_THAT(root->dimensions(), ElementsAre(2));
}

// Test that two consecutive broadcasts can be merged to one.
TEST_F(AlgebraicSimplifierTest, MergeBroadcasts2) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 3});
  Shape r3f32 = ShapeUtil::MakeShape(F32, {2, 5, 3});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r2f32, "param0"));
  // The initial dimensions go to places 0 and 2 in the 3-dim array,
  // and to places 1 and 3 in the 4-dim array,
  HloInstruction* inner_bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(r3f32, param0, {0, 2}));
  Shape r4f32 = ShapeUtil::MakeShape(F32, {4, 2, 5, 3});
  builder.AddInstruction(
      HloInstruction::CreateBroadcast(r4f32, inner_bcast, {1, 2, 3}));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Parameter(0))));
  EXPECT_THAT(root->dimensions(), ElementsAre(1, 3));
}

// Test that a broadcast of an iota can be merged to one iota.
TEST_F(AlgebraicSimplifierTest, MergeBroadcastAndIota) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  Shape r2f32 = ShapeUtil::MakeShape(F32, {2, 2});
  HloInstruction* iota =
      builder.AddInstruction(HloInstruction::CreateIota(r2f32, 1));
  Shape r3f32 = ShapeUtil::MakeShape(F32, {2, 2, 2});
  builder.AddInstruction(HloInstruction::CreateBroadcast(r3f32, iota, {0, 2}));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Iota()));
  EXPECT_EQ(Cast<HloIotaInstruction>(root)->iota_dimension(), 2);
}

// Test that a broadcast of an iota can be merged to one iota.
TEST_F(AlgebraicSimplifierTest, MergeBroadcastAndIota2) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  Shape r3f32 = ShapeUtil::MakeShape(F32, {2, 5, 3});
  HloInstruction* iota =
      builder.AddInstruction(HloInstruction::CreateIota(r3f32, 1));
  Shape r4f32 = ShapeUtil::MakeShape(F32, {4, 2, 5, 3});
  builder.AddInstruction(
      HloInstruction::CreateBroadcast(r4f32, iota, {1, 2, 3}));

  auto computation = m->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Iota()));
  EXPECT_EQ(Cast<HloIotaInstruction>(root)->iota_dimension(), 2);
}

TEST_F(AlgebraicSimplifierTest, TransposeOfDot) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = f32[3,4,5] parameter(0)
      rhs = f32[6,3,4] parameter(1)
      dot = f32[5,6] dot(lhs,rhs), lhs_contracting_dims={0,1}, rhs_contracting_dims={1,2}
      ROOT transpose = f32[6,5] transpose(dot), dimensions={1,0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Dot(m::Parameter(1), m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, SliceOfPadLow) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      param = f32[3,4] parameter(0)
      constant = f32[] constant(0.0)
      pad = f32[8,10] pad(f32[3,4] param, f32[] constant), padding=3_2x1_5
      ROOT slice = f32[1,1] slice(f32[8,10] pad), slice={[2:3],[0:1]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Reshape(m::Constant())));
}

TEST_F(AlgebraicSimplifierTest, SliceOfPadHigh) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      param = f32[3,4] parameter(0)
      constant = f32[] constant(0.0)
      pad = f32[8,10] pad(f32[3,4] param, f32[] constant), padding=3_2x1_5
      ROOT slice = f32[1,1] slice(f32[8,10] pad), slice={[6:7],[9:10]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Reshape(m::Constant())));
}

TEST_F(AlgebraicSimplifierTest, SliceOfPadMidNonScalar) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      param = f32[3,4] parameter(0)
      constant = f32[] constant(0.0)
      pad = f32[8,10] pad(f32[3,4] param, f32[] constant), padding=3_2x1_5
      ROOT slice = f32[1,1] slice(f32[8,10] pad), slice={[5:6],[4:5]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(module.get()).ValueOrDie());
}

TEST_F(AlgebraicSimplifierTest, SliceOfPadMidScalarConstant) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      param = f32[3,4] parameter(0)
      constant = f32[] constant(0.0)
      pad = f32[8,10] pad(f32[3,4] param, f32[] constant), padding=3_2x1_5
      ROOT slice = f32[1,1] slice(f32[8,10] pad), slice={[5:6],[9:10]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Reshape(m::Constant())));
}

TEST_F(AlgebraicSimplifierTest, SliceOfPadMidScalar) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      param = f32[1,1] parameter(0)
      constant = f32[] constant(0.0)
      pad = f32[8,10] pad(f32[1,1] param, f32[] constant), padding=3_4x4_5
      ROOT slice = f32[1,1] slice(f32[8,10] pad), slice={[3:4],[4:5]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter()));
}

TEST_F(AlgebraicSimplifierTest, SliceOfPadSomeDimsInPadding) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY entry () -> f32[1]{0} {
      constant.val = f32[] constant(4)
      constant.pad = f32[] constant(-7)
      reshape.1 = f32[1,1,1]{2,1,0} reshape(f32[] constant.val)
      pad = f32[3,3,3]{2,1,0} pad(f32[1,1,1]{2,1,0} reshape.1, f32[] constant.pad), padding=0_2x0_2x2_0
      slice = f32[1,1,1]{2,1,0} slice(f32[3,3,3]{2,1,0} pad), slice={[0:1], [0:1], [0:1]}
      ROOT reshape.2 = f32[1]{0} reshape(f32[1,1,1]{2,1,0} slice)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Reshape(m::ConstantScalar(-7.0))));
}

TEST_F(AlgebraicSimplifierTest, SliceOfConcatScalarInput) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      param.0 = f32[2] parameter(0)
      param.1 = f32[1] parameter(1)
      param.2 = f32[3] parameter(2)
      concat = f32[6] concatenate(param.0, param.1, param.2), dimensions={0}
      ROOT slice = f32[1] slice(concat), slice={[2:3]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter(1)));
}

TEST_F(AlgebraicSimplifierTest, SliceOfConcatNonScalarInput) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      param.0 = f32[2] parameter(0)
      param.1 = f32[1] parameter(1)
      param.2 = f32[3] parameter(2)
      concat = f32[6] concatenate(param.0, param.1, param.2), dimensions={0}
      ROOT slice = f32[1] slice(concat), slice={[4:5]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Slice(m::Parameter(2))));
  EXPECT_EQ(root->slice_starts(0), 1);
  EXPECT_EQ(root->slice_limits(0), 2);
}

TEST_F(AlgebraicSimplifierTest, NegateNegate) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      param.0 = f32[2] parameter(0)
      neg.0 = f32[2] negate(param.0)
      ROOT neg.1 = f32[2] negate(neg.0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter(0)));
}

TEST_F(AlgebraicSimplifierTest, NotNot) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      param.0 = pred[2] parameter(0)
      not.0 = pred[2] not(param.0)
      ROOT not.1 = pred[2] not(not.0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter(0)));
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
    return absl::StrCat(absl::StrJoin(input_spatials, ","), ";",
                        absl::StrJoin(symmetric_pad_spatials, ","), ";",
                        absl::StrJoin(reduce_window_spatials, ","), ";",
                        prepend_a, ";", should_become_broadcast);
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
  auto m = CreateNewVerifiedModule();
  const auto& param = GetParam();

  // a and b are parallel bounds we can either turn into a B F S0 S1 or
  // `B S0 S1 F` kind of pattern.
  auto decorate_spatials = [&param](absl::Span<const int64> spatials, int64 a,
                                    int64 b) {
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
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape pad_shape,
      ShapeInference::InferPadShape(input->shape(),
                                    ShapeUtil::MakeShape(F32, {}), padding));
  HloInstruction* pad = builder.AddInstruction(HloInstruction::CreatePad(
      pad_shape, input,
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0(0.0f))),
      padding));

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
    add_computation = m->AddEmbeddedComputation(builder.Build());
  }

  Window window = window_util::MakeWindow(
      decorate_spatials(param.reduce_window_spatials, 1, 1));
  auto zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape,
                          ShapeInference::InferReduceWindowShape(
                              pad->shape(), zero->shape(), window,
                              add_computation->ComputeProgramShape()));
  builder.AddInstruction(HloInstruction::CreateReduceWindow(
      output_shape, pad, zero, window, add_computation));

  auto computation = m->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  TF_ASSERT_OK_AND_ASSIGN(bool run_successful, simplifier.Run(m.get()));
  ASSERT_TRUE(run_successful);

  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), output_shape));

  if (param.should_become_broadcast) {
    EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Broadcast()));
  } else {
    EXPECT_THAT(computation->root_instruction(),
                GmockMatch(m::ReduceWindow(m::Op(), m::Op().Is(zero))));
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
       /*reduce_window_spatials=*/{1, 1}, /*prepend_a=*/true,
       /*should_become_broadcast=*/false},  //
      {/*input_spatials=*/{5, 1}, /*symmetric_pad_amount=*/{0, 2},
       /*reduce_window_spatials=*/{2, 5}, /*prepend_a=*/true,
       /*should_become_broadcast=*/false},  //
  };
  return *cases;
}

INSTANTIATE_TEST_SUITE_P(
    PadReduceWindowEffectiveBroadcastInstantiation,
    PadReduceWindowEffectiveBroadcastTest,
    ::testing::ValuesIn(PadReduceWindowEffectiveBroadcastCases()));

class BatchDotStrengthReductionTest
    : public AlgebraicSimplifierTest,
      public ::testing::WithParamInterface<
          ::testing::tuple<int, int, int, PrimitiveType>> {};
TEST_P(BatchDotStrengthReductionTest, BatchDotStrengthReduction) {
  auto module = CreateNewVerifiedModule();
  int m, k, n;
  PrimitiveType element_type;
  std::tie(m, k, n, element_type) = GetParam();
  std::vector<int64> lhs_dims = {2, 3, 5};
  std::vector<int64> rhs_dims = lhs_dims;
  std::vector<int64> output_dims = lhs_dims;
  if (m > 0) {
    lhs_dims.push_back(m);
    output_dims.push_back(m);
  }
  if (k > 0) {
    lhs_dims.push_back(k);
    rhs_dims.push_back(k);
  }
  if (n > 0) {
    rhs_dims.push_back(n);
    output_dims.push_back(n);
  }
  Shape dot_shape = ShapeUtil::MakeShape(element_type, output_dims);
  Shape lhs_shape = ShapeUtil::MakeShape(element_type, lhs_dims);
  Shape rhs_shape = ShapeUtil::MakeShape(element_type, rhs_dims);
  HloComputation::Builder builder(TestName());

  auto lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "lhs"));
  auto rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, rhs_shape, "rhs"));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_lhs_batch_dimensions(1);
  dot_dnums.add_lhs_batch_dimensions(2);
  dot_dnums.add_rhs_batch_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(1);
  dot_dnums.add_rhs_batch_dimensions(2);
  if (k > 0) {
    dot_dnums.add_lhs_contracting_dimensions(m > 0 ? 4 : 3);
    dot_dnums.add_rhs_contracting_dimensions(3);
  }
  builder.AddInstruction(HloInstruction::CreateDot(
      dot_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));
  auto computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, simplifier.Run(module.get()));
  const bool dot_should_be_transformed =
      m == 1 || k == 1 || n == 1 || m == -1 || k == -1 || n == -1;
  EXPECT_EQ(changed, dot_should_be_transformed);
  TF_ASSERT_OK_AND_ASSIGN(changed, simplifier.Run(module.get()));
  bool has_no_dot = true;
  for (const auto& hlo : computation->instructions()) {
    if (hlo->opcode() == HloOpcode::kDot) {
      has_no_dot = false;
      break;
    }
  }
  EXPECT_EQ(has_no_dot, dot_should_be_transformed);
}

INSTANTIATE_TEST_SUITE_P(
    BatchDotStrengthReductionTestInstantiation, BatchDotStrengthReductionTest,
    ::testing::Combine(::testing::Values(-1, 1, 2), ::testing::Values(-1, 1, 2),
                       ::testing::Values(-1, 1, 2),
                       ::testing::Values(C128, C64, F64, F32, BF16)));

class DotStrengthReductionTest
    : public AlgebraicSimplifierTest,
      public ::testing::WithParamInterface<
          ::testing::tuple<int, int, int, bool, bool, PrimitiveType>> {};
TEST_P(DotStrengthReductionTest, DotStrengthReduction) {
  auto module = CreateNewVerifiedModule();
  int m, k, n;
  bool transpose_lhs, transpose_rhs;
  PrimitiveType element_type;
  std::tie(m, k, n, transpose_lhs, transpose_rhs, element_type) = GetParam();

  Shape dot_shape = ShapeUtil::MakeShape(element_type, {m, n});
  Shape lhs_shape = ShapeUtil::MakeShape(element_type, {m, k});
  Shape transposed_lhs_shape = ShapeUtil::MakeShape(element_type, {k, m});
  Shape rhs_shape = ShapeUtil::MakeShape(element_type, {k, n});
  Shape transposed_rhs_shape = ShapeUtil::MakeShape(element_type, {n, k});
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
  builder.AddInstruction(HloInstruction::CreateDot(
      dot_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));
  auto computation = module->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  // First pass of algebraic simplifier will remove degenerate dimensions
  // and optimize dot(transpose(x),transpose(y))
  TF_ASSERT_OK_AND_ASSIGN(bool changed, simplifier.Run(module.get()));
  const bool dot_should_be_transformed = m == 1 || k == 1 || n == 1;
  const bool computation_should_be_modified =
      dot_should_be_transformed || (transpose_lhs && transpose_rhs);
  EXPECT_EQ(changed, computation_should_be_modified);
  // The second pass of algebriac simplifer will remove dots without
  // non-contracting dimensions or contracting dimensions.
  TF_ASSERT_OK_AND_ASSIGN(changed, simplifier.Run(module.get()));
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

INSTANTIATE_TEST_SUITE_P(
    DotStrengthReductionTestInstantiation, DotStrengthReductionTest,
    ::testing::Combine(::testing::Values(1, 2), ::testing::Values(1, 2),
                       ::testing::Values(1, 2), ::testing::Bool(),
                       ::testing::Bool(),
                       ::testing::Values(C128, C64, F64, F32, BF16)));

struct DotOfConcatTestSpec {
  int64 m;
  int64 k;
  int64 n;
};

class DotOfConcatSimplificationTest
    : public AlgebraicSimplifierTest,
      public ::testing::WithParamInterface<DotOfConcatTestSpec> {};

// Test that we transform
//  dot(const, concat(A, B, C))
// to
//  add(dot(const_0, A), dot(const_1, B),  dot(const_2, C))
TEST_P(DotOfConcatSimplificationTest, ConstantLHS) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  DotOfConcatTestSpec spec = GetParam();

  ASSERT_GE(spec.k, 3);

  int64 k0 = spec.k / 3;
  int64 k1 = spec.k / 3;
  int64 k2 = spec.k - k0 - k1;

  Shape lhs_shape = ShapeUtil::MakeShape(F32, {spec.m, spec.k});
  auto* lhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
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
  builder.AddInstruction(HloInstruction::CreateDot(
      dot_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto computation = m->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  TF_ASSERT_OK_AND_ASSIGN(bool run_successful, simplifier.Run(m.get()));
  ASSERT_TRUE(run_successful);

  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), dot_shape));

  auto match_dot_0 = m::Dot(m::Slice(m::Constant()), m::Parameter(0));
  auto match_dot_1 = m::Dot(m::Slice(m::Constant()), m::Parameter(1));
  auto match_dot_2 = m::Dot(m::Slice(m::Constant()), m::Parameter(2));
  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Add(m::Add(match_dot_0, match_dot_1), match_dot_2)));
}

// Test that we transform
//  dot(concat(A, B, C), const)
// to
//  add(dot(A, const_0), dot(B, const_1),  dot(C, const_2))
TEST_P(DotOfConcatSimplificationTest, ConstantRHS) {
  auto m = CreateNewVerifiedModule();
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
      HloInstruction::CreateParameter(3, lhs3_shape, "lhs3"));

  Shape lhs_shape = ShapeUtil::MakeShape(F32, {spec.m, spec.k});
  HloInstruction* lhs =
      builder.AddInstruction(HloInstruction::CreateConcatenate(
          lhs_shape, {lhs0, lhs1, lhs2, lhs3}, 1));

  Shape rhs_shape = ShapeUtil::MakeShape(F32, {spec.k, spec.n});
  auto* rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/spec.k, /*cols=*/spec.n)));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);

  Shape dot_shape = ShapeUtil::MakeShape(F32, {spec.m, spec.n});
  builder.AddInstruction(HloInstruction::CreateDot(
      dot_shape, lhs, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto computation = m->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  TF_ASSERT_OK_AND_ASSIGN(bool run_successful, simplifier.Run(m.get()));
  ASSERT_TRUE(run_successful);
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), dot_shape));

  auto match_dot_0 = m::Dot(m::Parameter(0), m::Slice(m::Constant()));
  auto match_dot_1 = m::Dot(m::Parameter(1), m::Slice(m::Constant()));
  auto match_dot_2 = m::Dot(m::Parameter(2), m::Slice(m::Constant()));
  auto match_dot_3 = m::Dot(m::Parameter(3), m::Slice(m::Constant()));
  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Add(m::Add(m::Add(match_dot_0, match_dot_1), match_dot_2),
                        match_dot_3)));
}

DotOfConcatTestSpec kDotOfConcatTestSpecs[] = {
    {/*m=*/3, /*k=*/9, /*n=*/3},    //
    {/*m=*/3, /*k=*/20, /*n=*/3},   //
    {/*m=*/1, /*k=*/18, /*n=*/5},   //
    {/*m=*/20, /*k=*/20, /*n=*/1},  //
    {/*m=*/1, /*k=*/16, /*n=*/1},   //
};

TEST_F(DotOfConcatSimplificationTest, ConcatIntoScalarDot) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      param0 = f32[4] parameter(0)
      param1 = f32[1] parameter(1)
      constant = f32[5] constant({-0.38, 0.07, -0.62, 0.66, 0.20})
      concat = f32[5] concatenate(param0, param1), dimensions={0}
      ROOT dot = f32[] dot(concat, constant), lhs_contracting_dims={0},
                                              rhs_contracting_dims={0}
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options = default_options_;
  options.set_enable_dot_strength_reduction(false);
  ASSERT_FALSE(AlgebraicSimplifier(options).Run(m.get()).ValueOrDie());
}

// Test that DynamicUpdateSlice update param with any dimension equal to zero
// gets removed.
TEST_F(AlgebraicSimplifierTest, DynamicUpdateSliceZeroUpdate) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  const Shape dslice_shape = ShapeUtil::MakeShape(F32, {10});
  HloInstruction* const operand = builder.AddInstruction(
      HloInstruction::CreateParameter(0, dslice_shape, "operand"));
  const Shape update_shape = ShapeUtil::MakeShape(F32, {0});
  HloInstruction* const update = builder.AddInstruction(
      HloInstruction::CreateParameter(1, update_shape, "update"));
  HloInstruction* const start_indices = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>({})));
  builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      dslice_shape, operand, update,
      std::initializer_list<HloInstruction*>({start_indices})));
  const HloComputation* const computation =
      m->AddEntryComputation(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), operand);
}

INSTANTIATE_TEST_SUITE_P(DotOfConcatSimplificationTestInstantiation,
                         DotOfConcatSimplificationTest,
                         ::testing::ValuesIn(kDotOfConcatTestSpecs));

struct DotOfGatherTestSpec {
  int64 m;
  int64 k;
  int64 n;
  int s;      // start index for dynamic slice on the non-contracting dimension
  int64 lcd;  // left contracting dimension
  int64 rcd;  // right contracting dimension
  bool neg;   // is negative testcase
};

class DotOfGatherSimplificationTest
    : public AlgebraicSimplifierTest,
      public ::testing::WithParamInterface<DotOfGatherTestSpec> {};

// input: dot(DS(ctA), ctB))
// where DS(ctA) = DS({M x K}, {s, 0}, {1, K}) and ctB = {K x N}.
// => input dimensions: dot({1 x K}, {K x N}) => {1 x N}.
// output: DS(dot(ctA, ctB))
// => output dimensions: DS ({M x N}, {s, 0}, {1, N}) => {1 x N}.
TEST_P(DotOfGatherSimplificationTest, ConstantRHS) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  DotOfGatherTestSpec spec = GetParam();

  ASSERT_LE(spec.s, spec.m);

  // For negative tests, increase k of the dynamic slice argument to prevent the
  // optimization (constants ctA, ctB must have equal contracting dimensions).
  int64 k_increase = spec.neg ? 5 : 0;
  int64 lhs_rows = (spec.lcd == 0) ? (spec.k + k_increase) : spec.m;
  int64 lhs_cols = (spec.lcd == 0) ? spec.m : (spec.k + k_increase);
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {lhs_rows, lhs_cols});
  auto* lhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/lhs_rows,
          /*cols=*/lhs_cols)));

  int32 start_row = (spec.lcd == 0) ? 0 : spec.s;
  int32 start_col = (spec.lcd == 0) ? spec.s : 0;
  std::vector<HloInstruction*> start_indices = {
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32>(start_row))),
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32>(start_col)))};
  int64 slice_row_size = (spec.lcd == 0) ? spec.k : 1;
  int64 slice_col_size = (spec.lcd == 0) ? 1 : spec.k;
  std::vector<int64> slice_sizes = {slice_row_size, slice_col_size};
  Shape ds_shape = ShapeUtil::MakeShape(F32, slice_sizes);
  auto* ds = builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      ds_shape, lhs, start_indices, slice_sizes));

  int64 rhs_rows = (spec.rcd == 0) ? spec.k : spec.n;
  int64 rhs_cols = (spec.rcd == 0) ? spec.n : spec.k;
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {rhs_rows, rhs_cols});
  auto* rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/rhs_rows,
          /*cols=*/rhs_cols)));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(spec.lcd);
  dot_dnums.add_rhs_contracting_dimensions(spec.rcd);

  int64 dot_row_size = 1;
  int64 dot_col_size = spec.n;
  Shape dot_shape = ShapeUtil::MakeShape(F32, {dot_row_size, dot_col_size});
  builder.AddInstruction(HloInstruction::CreateDot(
      dot_shape, ds, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto computation = m->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  TF_ASSERT_OK_AND_ASSIGN(bool run_successful, simplifier.Run(m.get()));
  ASSERT_TRUE(run_successful);
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), dot_shape));

  if (spec.neg) {
    EXPECT_NE(computation->root_instruction()->opcode(),
              HloOpcode::kDynamicSlice);
  } else {
    EXPECT_THAT(computation->root_instruction(),
                GmockMatch(m::DynamicSlice(m::Dot(m::Constant(), m::Constant()),
                                           m::Constant(), m::Constant())));
  }
}

// input: dot(ctA, DS(ctB))
// where ctA = {M x K} and DS(ctB) = DS({K x N}, {0, s}, {K, 1}).
// => input dimensions: dot({M x K}, {K x 1}) => {M x 1}.
// output: DS(dot(ctA, ctB))
// => output dimensions: DS ({M x N}, {0, s}, {M, 1}) => {M x 1}.
TEST_P(DotOfGatherSimplificationTest, ConstantLHS) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  DotOfGatherTestSpec spec = GetParam();

  ASSERT_LE(spec.s, spec.n);

  int64 lhs_rows = (spec.lcd == 0) ? spec.k : spec.m;
  int64 lhs_cols = (spec.lcd == 0) ? spec.m : spec.k;
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {lhs_rows, lhs_cols});
  auto* lhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/lhs_rows,
          /*cols=*/lhs_cols)));

  // For negative tests increase k of the dynamic slice argument to prevent the
  // optimization
  int64 k_increase = spec.neg ? 5 : 0;
  int64 rhs_rows = (spec.rcd == 0) ? (spec.k + k_increase) : spec.n;
  int64 rhs_cols = (spec.rcd == 0) ? spec.n : (spec.k + k_increase);
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {rhs_rows, rhs_cols});
  auto* rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/rhs_rows,
          /*cols=*/rhs_cols)));

  int32 start_row = (spec.rcd == 0) ? 0 : spec.s;
  int32 start_col = (spec.rcd == 0) ? spec.s : 0;
  std::vector<HloInstruction*> start_indices = {
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32>(start_row))),
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32>(start_col)))};
  int64 slice_row_size = (spec.rcd == 0) ? spec.k : 1;
  int64 slice_col_size = (spec.rcd == 0) ? 1 : spec.k;
  std::vector<int64> slice_sizes = {slice_row_size, slice_col_size};
  Shape ds_shape = ShapeUtil::MakeShape(F32, slice_sizes);
  auto* ds = builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      ds_shape, rhs, start_indices, slice_sizes));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(spec.lcd);
  dot_dnums.add_rhs_contracting_dimensions(spec.rcd);

  int64 dot_row_size = spec.m;
  int64 dot_col_size = 1;
  Shape dot_shape = ShapeUtil::MakeShape(F32, {dot_row_size, dot_col_size});
  builder.AddInstruction(HloInstruction::CreateDot(
      dot_shape, lhs, ds, dot_dnums, DefaultPrecisionConfig(2)));

  auto computation = m->AddEntryComputation(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  TF_ASSERT_OK_AND_ASSIGN(bool run_successful, simplifier.Run(m.get()));
  ASSERT_TRUE(run_successful);
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), dot_shape));

  if (spec.neg) {
    EXPECT_NE(computation->root_instruction()->opcode(),
              HloOpcode::kDynamicSlice);
  } else {
    EXPECT_THAT(computation->root_instruction(),
                GmockMatch(m::DynamicSlice(m::Dot(m::Constant(), m::Constant()),
                                           m::Constant(), m::Constant())));
  }
}

std::vector<DotOfGatherTestSpec> DotOfGatherPositiveNegativeTests() {
  std::vector<DotOfGatherTestSpec> positives = {
      // "Classical dot", i.e. matrix multiply:
      {/*m=*/10, /*k=*/10, /*n=*/5, /*s=*/0, /*lcd=*/1, /*rcd=*/0,
       /*neg=*/false},
      {/*m=*/20, /*k=*/20, /*n=*/3, /*s=*/2, /*lcd=*/1, /*rcd=*/0,
       /*neg=*/false},
      {/*m=*/10, /*k=*/3, /*n=*/10, /*s=*/9, /*lcd=*/1, /*rcd=*/0,
       /*neg=*/false},
      // Note: testing for m=1 and n=1 is unnecessary, as this optimizes to
      // dot(ct, ct) before DotOfGather optimization kicks in.
      // Contract on rows:
      {/*m=*/10, /*k=*/10, /*n=*/5, /*s=*/0, /*lcd=*/0, /*rcd=*/0,
       /*neg=*/false},
      {/*m=*/20, /*k=*/20, /*n=*/3, /*s=*/2, /*lcd=*/0, /*rcd=*/0,
       /*neg=*/false},
      {/*m=*/10, /*k=*/3, /*n=*/10, /*s=*/9, /*lcd=*/0, /*rcd=*/0,
       /*neg=*/false},
      // Reverse matrix multiply:
      {/*m=*/10, /*k=*/10, /*n=*/5, /*s=*/0, /*lcd=*/0, /*rcd=*/1,
       /*neg=*/false},
      {/*m=*/20, /*k=*/20, /*n=*/3, /*s=*/2, /*lcd=*/0, /*rcd=*/1,
       /*neg=*/false},
      {/*m=*/10, /*k=*/3, /*n=*/10, /*s=*/9, /*lcd=*/0, /*rcd=*/1,
       /*neg=*/false},
      // Contract on columns:
      {/*m=*/10, /*k=*/10, /*n=*/5, /*s=*/0, /*lcd=*/1, /*rcd=*/1,
       /*neg=*/false},
      {/*m=*/20, /*k=*/20, /*n=*/3, /*s=*/2, /*lcd=*/1, /*rcd=*/1,
       /*neg=*/false},
      {/*m=*/10, /*k=*/3, /*n=*/10, /*s=*/9, /*lcd=*/1, /*rcd=*/1,
       /*neg=*/false},
  };
  std::vector<DotOfGatherTestSpec> all;
  for (int i = 0; i < positives.size(); i++) {
    DotOfGatherTestSpec positive_test = positives[i];
    all.push_back(positive_test);
    DotOfGatherTestSpec negative_test = positive_test;
    negative_test.neg = true;
    all.push_back(negative_test);
  }
  return all;
}

INSTANTIATE_TEST_SUITE_P(
    DotOfGatherSimplificationTestInstantiation, DotOfGatherSimplificationTest,
    ::testing::ValuesIn(DotOfGatherPositiveNegativeTests()));

TEST_F(AlgebraicSimplifierTest, TupleReduceReshape) {
  const char* hlo_string = R"(
HloModule module

reducer {
  parameter.1 = f32[] parameter(0)
  parameter.3 = f32[] parameter(2)
  add.2 = f32[] add(parameter.1, parameter.3)
  parameter.0 = f32[] parameter(1)
  parameter.2 = f32[] parameter(3)
  add.3 = f32[] add(parameter.0, parameter.2)
  ROOT tuple.4 = (f32[], f32[]) tuple(add.2, add.3)
}

ENTRY entry {
  parameter.6 = (f32[], f32[]) parameter(0)
  get-tuple-element.10 = f32[] get-tuple-element(parameter.6), index=0
  get-tuple-element.11 = f32[] get-tuple-element(parameter.6), index=1
  constant = f32[] constant(0)
  ROOT reduce = (f32[], f32[]) reduce(get-tuple-element.10, get-tuple-element.11, constant, constant), dimensions={}, to_apply=reducer
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Tuple(
                        m::Reshape(m::GetTupleElement(m::Parameter(), 0)),
                        m::Reshape(m::GetTupleElement(m::Parameter(), 1)))));
}

TEST_F(AlgebraicSimplifierTest, TupleReduceBroadcast) {
  const char* hlo_string = R"(
HloModule module

reducer {
  parameter.1 = f32[] parameter(0)
  parameter.3 = f32[] parameter(2)
  mul.2 = f32[] add(parameter.1, parameter.3)
  parameter.0 = f32[] parameter(1)
  parameter.2 = f32[] parameter(3)
  add.3 = f32[] add(parameter.0, parameter.2)
  ROOT tuple.4 = (f32[], f32[]) tuple(mul.2, add.3)
}

ENTRY entry {
  parameter.6 = (f32[0, 10, 10], f32[0, 10, 10]) parameter(0)
  get-tuple-element.10 = f32[0, 10, 10] get-tuple-element(parameter.6), index=0
  get-tuple-element.11 = f32[0, 10, 10] get-tuple-element(parameter.6), index=1
  constant.0 = f32[] constant(0)
  constant.1 = f32[] constant(1)
  ROOT reduce = (f32[10, 10], f32[10, 10]) reduce(get-tuple-element.10, get-tuple-element.11, constant.0, constant.1), dimensions={0}, to_apply=reducer
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Tuple(m::Broadcast(m::ConstantScalar(0)),
                                        m::Broadcast(m::ConstantScalar(1)))));
}

TEST_F(AlgebraicSimplifierTest, ZeroSizedReshapeWithoutLayout) {
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {1}), "param"));
  HloInstruction* broadcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::MakeShape(F32, {0, 1}), param, {1}));

  // Create a reshape with zero sized result and without layout.
  Shape reshaped_shape = ShapeUtil::MakeShape(F32, {0});
  reshaped_shape.clear_layout();
  builder.AddInstruction(
      HloInstruction::CreateReshape(reshaped_shape, broadcast));

  std::unique_ptr<VerifiedHloModule> module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
}

TEST_F(AlgebraicSimplifierTest, DividedByConstantInstructionWithoutLayout) {
  Shape shape = ShapeUtil::MakeShape(F32, {});
  shape.clear_layout();
  auto builder = HloComputation::Builder(TestName());
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));

  HloInstruction* const_value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(20.0f)));
  builder.AddInstruction(HloInstruction::CreateBinary(shape, HloOpcode::kDivide,
                                                      param, const_value));

  std::unique_ptr<VerifiedHloModule> module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).ValueOrDie());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Multiply()));
}

// Test that 1/sqrt(X) is simplified to rsqrt(X).
TEST_F(AlgebraicSimplifierTest, RecipSqrt) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      sqrt = f32[] sqrt(p0)
      ROOT div = f32[] divide(p1, sqrt)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::MultiplyAnyOrder(m::Parameter(1),
                                             m::Rsqrt(m::Parameter(0)))));
}

// Test that 1/rsqrt(X) is simplified to sqrt(X).
TEST_F(AlgebraicSimplifierTest, RecipRsqrt) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      rsqrt = f32[] rsqrt(p0)
      ROOT div = f32[] divide(p1, rsqrt)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::MultiplyAnyOrder(m::Parameter(1),
                                             m::Sqrt(m::Parameter(0)))));
}

TEST_F(AlgebraicSimplifierTest, CopyReshape) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[168,168,48,48]{3,2,1,0} parameter(0)
      r0 = f32[1,168,168,2304]{3,2,1,0} reshape(p0)
      ROOT c0 = f32[1,168,168,2304]{3,0,2,1} copy(r0)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  Shape result_shape = m->entry_computation()->root_instruction()->shape();
  AlgebraicSimplifierOptions options(
      [](const Shape&, const Shape&) { return false; });
  options.set_is_layout_sensitive(true);
  ASSERT_TRUE(AlgebraicSimplifier(options).Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Reshape(m::Parameter(0)).WithShapeEqualTo(&result_shape)));
}

TEST_F(AlgebraicSimplifierTest, DotContractingReorder_RL) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      rhs = f32[6, 2] constant({{1, 2},{3, 4},{5, 6},{1, 1},{1, 1},{1, 1}})
      t0 = f32[2, 2, 3] parameter(0)
      t1 = f32[2, 3, 2] transpose(t0), dimensions={0, 2, 1}
      lhs = f32[2, 6] reshape(t1)
      ROOT dot.5 = f32[2, 2] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  auto shape1 = ShapeUtil::MakeShape(F32, {2, 6});
  auto shape2 = ShapeUtil::MakeShape(F32, {3, 2, 2});
  auto shape3 = ShapeUtil::MakeShape(F32, {2, 3, 2});
  // The transformation of moving transpose and reshape to the constant side
  // is layout insensitive. We ignore layout when checking shapes.
  const HloInstruction* transpose;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(
                  m::Reshape(m::Parameter(0)).WithShapeCompatibleTo(&shape1),
                  m::Reshape(m::Transpose(&transpose,
                                          m::Reshape(m::Constant())
                                              .WithShapeCompatibleTo(&shape2))
                                 .WithShapeCompatibleTo(&shape3)))));
  EXPECT_THAT(transpose->dimensions(), ElementsAre(1, 0, 2));
}

TEST_F(AlgebraicSimplifierTest, DotContractingReorder_RR) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      rhs = f32[2, 6] constant({{1, 2, 3, 4, 5, 6},
                                {1, 1, 1, 1, 1, 1}})
      t0 = f32[2, 2, 3] parameter(0)
      t1 = f32[2, 3, 2] transpose(t0), dimensions={0, 2, 1}
      lhs = f32[2, 6] reshape(t1)
      ROOT dot.5 = f32[2, 2] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  auto shape1 = ShapeUtil::MakeShape(F32, {2, 6});
  auto shape2 = ShapeUtil::MakeShape(F32, {2, 3, 2});
  auto shape3 = ShapeUtil::MakeShape(F32, {2, 2, 3});
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(
                  m::Reshape(m::Parameter(0)).WithShapeCompatibleTo(&shape1),
                  m::Reshape(m::Transpose(m::Reshape(m::Constant())
                                              .WithShapeCompatibleTo(&shape2))
                                 .WithShapeCompatibleTo(&shape3)))));
}

TEST_F(AlgebraicSimplifierTest, DotContractingReorder_LR) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      rhs = f32[2, 6] constant({{1, 2, 3, 4, 5, 6},
                                {1, 1, 1, 1, 1, 1}})
      t0 = f32[2, 3, 2] parameter(0)
      t1 = f32[3, 2, 2] transpose(t0), dimensions={1, 0, 2}
      lhs = f32[6, 2] reshape(t1)
      ROOT dot.5 = f32[2, 2] dot(lhs, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  auto shape1 = ShapeUtil::MakeShape(F32, {6, 2});
  auto shape2 = ShapeUtil::MakeShape(F32, {2, 3, 2});
  auto shape3 = ShapeUtil::MakeShape(F32, {2, 2, 3});
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(
                  m::Reshape(m::Parameter(0)).WithShapeCompatibleTo(&shape1),
                  m::Reshape(m::Transpose(m::Reshape(m::Constant())
                                              .WithShapeCompatibleTo(&shape2))
                                 .WithShapeCompatibleTo(&shape3)))));
}

TEST_F(AlgebraicSimplifierTest, DotContractingReorder_LR2) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      rhs = f32[8, 2] constant({{1, 1},{2, 2},{3, 3},{4, 4},{5, 5},{6, 6},{7, 7},{8, 8}})
      t0 = f32[2, 2, 2, 2] parameter(0)
      t1 = f32[2, 2, 2, 2] transpose(t0), dimensions={0, 2, 3, 1}
      lhs = f32[2, 8] reshape(t1)
      ROOT dot.5 = f32[2, 2] dot(lhs, rhs), lhs_contracting_dims={1},
                                            rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  auto shape1 = ShapeUtil::MakeShape(F32, {2, 8});
  auto shape2 = ShapeUtil::MakeShape(F32, {2, 2, 2, 2});
  const HloInstruction* transpose;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Dot(
          m::Reshape(m::Parameter(0)).WithShapeCompatibleTo(&shape1),
          m::Reshape(m::Transpose(
              &transpose,
              m::Reshape(m::Constant()).WithShapeCompatibleTo(&shape2))))));
  EXPECT_THAT(transpose->dimensions(), ElementsAre(2, 0, 1, 3));
}

TEST_F(AlgebraicSimplifierTest, DotContractingReorder_MM) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      rhs = f32[2, 6, 2] constant({{{1, 1},{2, 2},{3, 3},{4, 4},{5, 5},{6, 6}},
                                   {{1, 1},{2, 2},{3, 3},{4, 4},{5, 5},{6, 6}}})
      t0 = f32[2, 2, 3, 2] parameter(0)
      t1 = f32[2, 3, 2, 2] transpose(t0), dimensions={0, 2, 1, 3}
      lhs = f32[2, 6, 2] reshape(t1)
      ROOT dot.5 = f32[2, 2, 2] dot(lhs, rhs), lhs_batch_dims={0}, lhs_contracting_dims={1},
                                               rhs_batch_dims={0}, rhs_contracting_dims={1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  auto shape1 = ShapeUtil::MakeShape(F32, {2, 6, 2});
  auto shape2 = ShapeUtil::MakeShape(F32, {2, 3, 2, 2});
  auto shape3 = ShapeUtil::MakeShape(F32, {2, 2, 3, 2});
  const HloInstruction* transpose;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(
                  m::Reshape(m::Parameter(0)).WithShapeCompatibleTo(&shape1),
                  m::Reshape(m::Transpose(&transpose,
                                          m::Reshape(m::Constant())
                                              .WithShapeCompatibleTo(&shape2))
                                 .WithShapeCompatibleTo(&shape3)))));
  EXPECT_THAT(transpose->dimensions(), ElementsAre(0, 2, 1, 3));
}

TEST_F(AlgebraicSimplifierTest, DotContractingReorder_NegTranspose) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      rhs = f32[12, 2] constant({{1, 1},{2, 2},{3, 3},{4, 4},{5, 5},{6, 6},{1, 1},{2, 2},{3, 3},{4, 4},{5, 5},{6, 6}})
      t0 = f32[3, 4, 2] parameter(0)
      t1 = f32[2, 3, 4] transpose(t0), dimensions={2, 0, 1}
      lhs = f32[2, 12] reshape(t1)
      ROOT dot.5 = f32[2, 2] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  // Transpose affects non-contracting dimension. The transpose and reshape
  // should not be moved to the constant side.
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
}

TEST_F(AlgebraicSimplifierTest, DotContractingReorder_NegReshape) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      rhs = f32[8, 2] constant({{1, 1},{2, 2},{3, 3},{4, 4},{1, 1},{2, 2},{3, 3},{4, 4}})
      t0 = f32[2, 4, 3] parameter(0)
      t1 = f32[2, 3, 4] transpose(t0), dimensions={0, 2, 1}
      lhs = f32[3, 8] reshape(t1)
      ROOT dot.5 = f32[3, 2] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  // Reshape affects non-contracting dimensions. The transpose and reshape
  // should not be moved to the constant side.
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
}

TEST_F(AlgebraicSimplifierTest, DotContractingReorder_NegConstant) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      t0 = f32[2, 3, 4] parameter(0)
      t1 = f32[2, 4, 3] transpose(t0), dimensions={0, 2, 1}
      lhs = f32[2, 12] reshape(t1)
      rhs = f32[12, 2] parameter(1)
      ROOT dot.5 = f32[2, 2] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  // Both operands are non-constant, so the optimization should not happen.
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
}

TEST_F(AlgebraicSimplifierTest, DotContractingReorder_NegLayout) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      rhs = f32[6, 2] constant({{1, 2},{3, 4},{5, 6},{1, 1},{1, 1},{1, 1}})
      t0 = f32[2, 2, 3] parameter(0)
      t1 = f32[2, 3, 2] transpose(t0), dimensions={0, 2, 1}
      lhs = f32[2, 6] reshape(t1)
      ROOT dot.5 = f32[2, 2] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  // We disable converting reshape to bitcast to make sure algsimp pass does
  // not catch the reshape in this test, then we can simply check if algsimp
  // pass does not make any change.
  AlgebraicSimplifierOptions options(
      [](const Shape&, const Shape&) { return false; });
  options.set_is_layout_sensitive(true);
  // The transformation of moving transpose and reshape to the constant side is
  // layout insensitive. It should not happen if AlgebraicSimplifier is set up
  // to be layout sensitive.
  ASSERT_FALSE(AlgebraicSimplifier(options).Run(m.get()).ValueOrDie());
}

TEST_F(AlgebraicSimplifierTest, DotContractingReorder_SizeOneDimsNoChange) {
  // This isn't transformed (notice that the relative order of the `2` and `3`
  // dims doesn't change, so there's no opportunity here), but it's nonetheless
  // an interesting testcase because of the presence of the size-1 dimensions.
  const char* kModuleStr = R"(
    HloModule m
    test {
     param = f32[1,2,5,3] parameter(0)
     transpose = f32[1,5,2,3] transpose(param), dimensions={0,2,1,3}
     reshape = f32[5,6] reshape(transpose)
     constant = f32[6,4] constant({{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4}})
     ROOT dot = f32[5,4] dot(reshape, constant),
       lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
}

TEST_F(AlgebraicSimplifierTest, DotContractingReorder_SizeOneDims) {
  const char* kModuleStr = R"(
    HloModule m
    test {
     param = f32[1,2,3,5] parameter(0)
     transpose = f32[1,3,2,5] transpose(param), dimensions={0,2,1,3}
     reshape = f32[6,5] reshape(transpose)
     constant = f32[6,4] constant({{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4}})
     ROOT dot = f32[5,4] dot(reshape, constant),
       lhs_contracting_dims={0}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  auto shape1 = ShapeUtil::MakeShape(F32, {6, 5});
  auto shape2 = ShapeUtil::MakeShape(F32, {1, 3, 2, 4});
  auto shape3 = ShapeUtil::MakeShape(F32, {1, 2, 3, 4});
  const HloInstruction* transpose;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(
                  m::Reshape(m::Parameter(0)).WithShapeCompatibleTo(&shape1),
                  m::Reshape(m::Transpose(&transpose,
                                          m::Reshape(m::Constant())
                                              .WithShapeCompatibleTo(&shape2))
                                 .WithShapeCompatibleTo(&shape3)))));
  EXPECT_THAT(transpose->dimensions(), ElementsAre(0, 2, 1, 3));
}

TEST_F(AlgebraicSimplifierTest,
       DotContractingReorder_NoChangeInContractingDimsOrder) {
  // No optimization opportunity here because the transpose does not reorder the
  // contracting dims.
  const char* kModuleStr = R"(
    HloModule m
    test {
      param = f32[2,5,1,3] parameter(0)
      transpose = f32[1,5,2,3] transpose(param), dimensions={2,1,0,3}
      reshape = f32[5,6] reshape(transpose)
      constant = f32[6,4] constant({{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4}})
      ROOT dot = f32[5,4] dot(reshape, constant),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
}

TEST_F(AlgebraicSimplifierTest, CompareIota) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = s32[] constant(0)
      iota = s32[128] iota(), iota_dimension=0
      broad = s32[128] broadcast(zero), dimensions={}
      ROOT compare = pred[128] compare(iota, broad), direction=LT
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::ConstantScalar(false))));
}

TEST_F(AlgebraicSimplifierTest, CompareSame) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      param = s32[123] parameter(0)
      ROOT compare = pred[123] compare(param, param), direction=GE
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::ConstantScalar(true))));
}

TEST_F(AlgebraicSimplifierTest, CanDisableDotToMultiplyRewrite) {
  // Some backends may have better performance by treating an outer product as a
  // Dot, rather than a broadcast Multiply
  const char* kModuleStr = R"(
    HloModule m
    test {
      param1 = f32[64] parameter(0)
      param2 = f32[64] parameter(1)
      ROOT compare = f32[64, 64] dot(param1, param2),
        lhs_contracting_dims={}, rhs_contracting_dims={}
    })";

  // Verify that the default is to re-write
  TF_ASSERT_OK_AND_ASSIGN(auto m1, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m1.get()).ValueOrDie());
  EXPECT_THAT(m1->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(m::Op(), m::Op())));

  // Verify that we can disable the re-write
  AlgebraicSimplifierOptions opts = default_options_;
  opts.set_enable_dot_to_multiply_rewrite(false);
  TF_ASSERT_OK_AND_ASSIGN(auto m2, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(opts).Run(m2.get()).ValueOrDie());
}

TEST_F(AlgebraicSimplifierTest, RemainderOfIota) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      iota = s32[5,1000] iota(), iota_dimension=0
      five = s32[] constant(5)
      five_bcast = s32[5,1000] broadcast(s32[] five), dimensions={}
      ROOT remainder = s32[5,1000] remainder(iota, s32[5,1000] five_bcast)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Iota()));
}

TEST_F(AlgebraicSimplifierTest, RemainderOfNPlusIota) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      iota = s32[5,1000] iota(), iota_dimension=0
      five = s32[] constant(5)
      five_bcast = s32[5,1000] broadcast(five), dimensions={}
      sum = s32[5,1000] add(iota, five_bcast)
      ROOT remainder = s32[5,1000] remainder(sum, s32[5,1000] five_bcast)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Remainder(m::Iota(), m::Broadcast())));
}

// No simplification because 125 + 5 overflows S8.
TEST_F(AlgebraicSimplifierTest, RemainderOfNPlusIotaOverflow) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      iota = s8[126] iota(), iota_dimension=0
      five = s8[] constant(5)
      five_bcast = s8[126] broadcast(five), dimensions={}
      sum = s8[126] add(iota, five_bcast)
      ROOT remainder = s8[126] remainder(sum, s8[126] five_bcast)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
}

TEST_F(AlgebraicSimplifierTest, RepeatedRemainder) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = s32[1000] parameter(0)
      q = s32[1000] parameter(1)
      r = s32[1000] remainder(p, q)
      ROOT rr = s32[1000] remainder(r, q)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Remainder(m::Parameter(), m::Parameter())));
}

TEST_F(AlgebraicSimplifierTest, SlicePadLayout) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      %param.0 = f32[128,9,9,1024]{0,3,2,1} parameter(0)
      %param.1 = f32[] parameter(1)
      %slice = f32[128,9,9,1024]{0,3,2,1} slice(%param.0),
        slice={[0:128], [0:9], [0:9], [0:1024]}
      ROOT %pad = f32[128,8,9,1024]{0,3,2,1} pad(%slice, %param.1),
        padding=0_0x-1_0x0_0x0_0
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  const Shape root_shape = m->entry_computation()->root_instruction()->shape();
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  ASSERT_TRUE(AlgebraicSimplifier(options).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Slice().WithShapeEqualTo(&root_shape)));
}

TEST_F(AlgebraicSimplifierTest, MinOfMaxToClamp) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4] parameter(0)
      c0 = f32[] constant(3.0)
      c1 = f32[] constant(4.0)
      b0 = f32[4] broadcast(c0), dimensions={}
      b1 = f32[4] broadcast(c1), dimensions={}
      m0 = f32[4] maximum(b0, p0)
      ROOT m1 = f32[4] minimum(m0, b1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Clamp(m::Broadcast(m::ConstantScalar(3.0)), m::Parameter(0),
                          m::Broadcast(m::ConstantScalar(4.0)))));
}

TEST_F(AlgebraicSimplifierTest, MaxOfMinToClamp) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4] parameter(0)
      c0 = f32[] constant(3.0)
      c1 = f32[] constant(4.0)
      b0 = f32[4] broadcast(c0), dimensions={}
      b1 = f32[4] broadcast(c1), dimensions={}
      m0 = f32[4] minimum(p0, b1)
      ROOT m1 = f32[4] maximum(b0, m0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Clamp(m::Broadcast(m::ConstantScalar(3.0)), m::Parameter(0),
                          m::Broadcast(m::ConstantScalar(4.0)))));
}

TEST_F(AlgebraicSimplifierTest, ClampOfClamp) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = f32[] parameter(2)
      c0 = f32[] clamp(p0, p1, p2)
      ROOT c1 = f32[] clamp(p0, c0, p2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Clamp(m::Parameter(0), m::Parameter(1), m::Parameter(2))));
}

TEST_F(AlgebraicSimplifierTest, MaxOfClamp) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = f32[] parameter(2)
      c0 = f32[] clamp(p0, p1, p2)
      ROOT m0 = f32[] maximum(p0, c0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Clamp(m::Parameter(0), m::Parameter(1), m::Parameter(2))));
}

TEST_F(AlgebraicSimplifierTest, SliceOfConcat) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[100,50] parameter(0)
      p1 = f32[50,50] parameter(1)
      c0 = f32[150,50] concatenate(p0, p1), dimensions={0}
      ROOT s0 = f32[50,50] slice(c0), slice={[100:150], [0:50]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(1)));
}

}  // namespace
}  // namespace xla

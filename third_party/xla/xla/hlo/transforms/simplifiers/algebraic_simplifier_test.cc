/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/layout_assignment.h"
#include "xla/service/memory_annotations.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::tsl::testing::IsOkAndHolds;
namespace m = match;
namespace op = xla::testing::opcode_matchers;

class AlgebraicSimplifierTest : public HloHardwareIndependentTestBase {
 public:
  AlgebraicSimplifierTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/true,
            /*allow_mixed_precision_in_hlo_verifier=*/true,
            LayoutAssignment::InstructionCanChangeLayout) {}

 protected:
  AlgebraicSimplifierOptions default_options_;
};

// clang-format off
const char* non_neg_ops[] = {"abs(p0)",
                             "constant(0.0)",
                             "constant(0.1)",
                             "constant(inf)",
                             "exponential(p0)",
                             "maximum(p0, a1)",
                             "maximum(a1, p0)",
                             "minimum(a0, a1)",
                             "multiply(p0, p0)",
                             "power(a0, p1)",
                             "select(pred0, a0, a1)",
                             "select(pred0, a1, a0)"};

const char* arb_sign_ops[] = {"constant(-0.0)",
                              "constant(-0.1)",
                              "constant(-inf)",
                              "constant(nan)",
                              "cosine(p0)",
                              "custom-call(a1), custom_call_target=\"foobar\"",
                              "maximum(p0, p1)",
                              "maximum(p1, p0)",
                              "minimum(a0, p1)",
                              "minimum(p0, a1)",
                              "minimum(p0, p1)",
                              "multiply(p0, a1)",
                              "multiply(a1, p0)",
                              "negate(p0)",
                              "power(p0, p1)",
                              "select(pred0, a1, p0)",
                              "select(pred0, p0, a1)"};
// clang-format on

// Test that the result of particular operations is always non-negative
TEST_F(AlgebraicSimplifierTest, IsNonNegative_Op) {
  for (const auto* op : non_neg_ops) {
    const auto kModuleStr = absl::StrFormat(R"(
      HloModule m
      test {
        p0 = f32[] parameter(0)
        a0 = f32[] abs(p0)
        p1 = f32[] parameter(1)
        a1 = f32[] abs(p1)
        pred0 = pred[] parameter(2)
        ROOT y = f32[] %s
      }
    )",
                                            op);
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
    ASSERT_TRUE(AlgebraicSimplifierVisitor::IsNonNegative(
        m->entry_computation()->root_instruction(), default_options_));
  }
}

// Test that the result of particular operations might be negative
TEST_F(AlgebraicSimplifierTest, IsNonNegative_Op_NegativeTestCase) {
  for (const auto op : arb_sign_ops) {
    const auto kModuleStr = absl::StrFormat(R"(
      HloModule m
      test {
        p0 = f32[] parameter(0)
        a0 = f32[] abs(p0)
        p1 = f32[] parameter(1)
        a1 = f32[] abs(p1)
        pred0 = pred[] parameter(2)
        ROOT y = f32[] %s
      }
    )",
                                            op);
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
    ASSERT_FALSE(AlgebraicSimplifierVisitor::IsNonNegative(
        m->entry_computation()->root_instruction(), default_options_));
  }
}

// Test that the result of Broadcast is non-negative if its operand is
// non-negative
TEST_F(AlgebraicSimplifierTest, IsNonNegative_Broadcast) {
  for (const auto op : non_neg_ops) {
    const auto kModuleStr = absl::StrFormat(R"(
      HloModule m
      test {
        p0 = f32[] parameter(0)
        a0 = f32[] abs(p0)
        p1 = f32[] parameter(1)
        a1 = f32[] abs(p1)
        pred0 = pred[] parameter(2)
        y = f32[] %s
        ROOT b = f32[4,8] broadcast(y), dimensions={}
      }
    )",
                                            op);
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
    ASSERT_TRUE(AlgebraicSimplifierVisitor::IsNonNegative(
        m->entry_computation()->root_instruction(), default_options_));
  }
}

// Test that the result of Broadcast might be negative if its oprand is
// not non-negative
TEST_F(AlgebraicSimplifierTest, IsNonNegative_Broadcast_NegativeTestCase) {
  for (const auto op : arb_sign_ops) {
    const auto kModuleStr = absl::StrFormat(R"(
      HloModule m
      test {
        p0 = f32[] parameter(0)
        a0 = f32[] abs(p0)
        p1 = f32[] parameter(1)
        a1 = f32[] abs(p1)
        pred0 = pred[] parameter(2)
        y = f32[] %s
        ROOT b = f32[4,8] broadcast(y), dimensions={}
      }
    )",
                                            op);
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
    ASSERT_FALSE(AlgebraicSimplifierVisitor::IsNonNegative(
        m->entry_computation()->root_instruction(), default_options_));
  }
}

// Test that the result #2 of custom-call batchNormalizationForwardTraining is
// non-negative
TEST_F(AlgebraicSimplifierTest,
       IsNonNegative_CustomCall_BatchNormalizationForwardTraining) {
  const auto kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[128,32,2,112] parameter(0)
      p1 = f32[32] parameter(1)
      p2 = f32[32] parameter(2)
      c0 = f32[] constant(0.001)
      c1 = s64[] constant(1)
      cc0 = (f32[128,32,2,112], f32[32], f32[32]) custom-call(p0, p1, p2, c0, c1), custom_call_target="__cudnn$batchNormalizationForwardTraining"
      ROOT t2 = f32[32] get-tuple-element(cc0), index=2
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  default_options_.set_cudnn_batchnorm_forward_training_metadata(
      "__cudnn$batchNormalizationForwardTraining");
  ASSERT_TRUE(AlgebraicSimplifierVisitor::IsNonNegative(
      m->entry_computation()->root_instruction(), default_options_));
}

// Test that the results #0 and #1 of custom-call
// batchNormalizationForwardTraining can have arbitrary sign
TEST_F(
    AlgebraicSimplifierTest,
    IsNonNegative_CustomCall_BatchNormalizationForwardTraining_NegativeTestCase_index1) {  // NOLINT(whitespace/line_length)
  for (const auto op : {"f32[128,32,2,112] get-tuple-element(cc0), index=0",
                        "f32[32] get-tuple-element(cc0), index=1"}) {
    const auto kModuleStr = absl::StrFormat(R"(
      HloModule m
      test {
        p0 = f32[128,32,2,112] parameter(0)
        p1 = f32[32] parameter(1)
        p2 = f32[32] parameter(2)
        c0 = f32[] constant(0.001)
        c1 = s64[] constant(1)
        cc0 = (f32[128,32,2,112], f32[32], f32[32]) custom-call(p0, p1, p2, c0, c1), custom_call_target="__cudnn$batchNormalizationForwardTraining"
        ROOT t1 = %s
      }
    )",
                                            op);
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
    default_options_.set_cudnn_batchnorm_forward_training_metadata(
        "__cudnn$batchNormalizationForwardTraining");
    ASSERT_FALSE(AlgebraicSimplifierVisitor::IsNonNegative(
        m->entry_computation()->root_instruction(), default_options_));
  }
}

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::MultiplyAnyOrder(
                  m::AddAnyOrder(m::Parameter(0), m::Parameter(1)),
                  m::ConstantScalar(0.125))));
}

// (Abs(A)) * (Abs(A)) => (A*A)
TEST_F(AlgebraicSimplifierTest, SquareOfAbs) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = f32[] parameter(0)
      a = f32[] abs(p)
      ROOT z = f32[] multiply(a, a)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(0))));
}

// (A*C1) * (B*C2) => (A*B)*(C1*C2)
TEST_F(AlgebraicSimplifierTest, MultiplyChain) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      c = f32[] constant(2)
      d = f32[] constant(4)
      x = f32[] multiply(p0, c)
      y = f32[] multiply(p1, d)
      ROOT z = f32[] multiply(x, y)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::MultiplyAnyOrder(
          m::MultiplyAnyOrder(m::Parameter(0), m::Parameter(1)),
          m::MultiplyAnyOrder(m::ConstantScalar(2), m::ConstantScalar(4)))));
}

// (a*C1)*C2 => a*(C1*C2)
TEST_F(AlgebraicSimplifierTest, MultiplyChain2) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      a = f32[] constant(2)
      b = f32[] constant(4)
      c = f32[] multiply(p0, a)
      ROOT y = f32[] multiply(c, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::MultiplyAnyOrder(
                  m::Parameter(0), m::MultiplyAnyOrder(m::ConstantScalar(2),
                                                       m::ConstantScalar(4)))));
}

// MUL(MUL(X, BROADCAST(constant)), BROADCAST(Y)) ==>
// MUL(X, BROADCAST(MUL(Y, BROADCAST(constant))))
TEST_F(AlgebraicSimplifierTest, MultiplyBroadcastReassoc) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[2,2] parameter(0)
      p1 = f32[] parameter(1)
      b = f32[] constant(2)
      c = f32[2, 2] broadcast(b), dimensions={}
      x = f32[2,2] multiply(p0, c)
      y = f32[2,2] broadcast(p1), dimensions={}
      ROOT z = f32[2,2] multiply(y, x)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::MultiplyAnyOrder(
                  m::Parameter(0), m::Broadcast(m::MultiplyAnyOrder(
                                       m::Parameter(1), m::Constant())))));
}

// Mul(Add(Conv(input, filter), bias), Broadcast(constant)) => Conv(input,
// Mul(filter, Broadcast(constant))), Mul(bias, Broadcast(constant)))
TEST_F(AlgebraicSimplifierTest, ReorderConvAddMul) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      input = f32[5,4,4,1] parameter(0)
      filter = f32[2,2,1,2] constant({{{{1.1, 1.2}}, {{2.1, 2.2}}},
                                      {{{3.1, 3.2}}, {{4.1, 4.2}}}})
      conv = f32[5,3,3,2] convolution(input, filter),
               window={size=2x2}, dim_labels=b01f_01io->b01f
      bias = f32[5,3,3,2] parameter(1)
      add = f32[5,3,3,2] add(conv, bias)
      constant = f32[2] constant({1.0, 1.1})
      bcast = f32[5,3,3,2] broadcast(constant), dimensions={3}
      ROOT multiply = f32[5,3,3,2] multiply(add, bcast)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions opts = default_options_;
  opts.set_enable_conv_add_multiply_reorder(true);
  ASSERT_TRUE(AlgebraicSimplifier(opts).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::AddAnyOrder(
                  m::Convolution(
                      m::Parameter(0),
                      m::Multiply(m::Constant(), m::Broadcast(m::Constant()))),
                  m::Multiply(m::Parameter(1), m::Broadcast(m::Constant())))));
}

TEST_F(AlgebraicSimplifierTest, DoNotReorderConvAddMulWhenDisabled) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      input = f32[5,4,4,1] parameter(0)
      filter = f32[2,2,1,2] constant({{{{1.1, 1.2}}, {{2.1, 2.2}}},
                                      {{{3.1, 3.2}}, {{4.1, 4.2}}}})
      conv = f32[5,3,3,2] convolution(input, filter),
               window={size=2x2}, dim_labels=b01f_01io->b01f
      bias = f32[5,3,3,2] parameter(1)
      add = f32[5,3,3,2] add(conv, bias)
      constant = f32[2] constant({1.0, 1.1})
      bcast = f32[5,3,3,2] broadcast(constant), dimensions={3}
      ROOT multiply = f32[5,3,3,2] multiply(add, bcast)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions opts = default_options_;
  opts.set_enable_conv_add_multiply_reorder(false);
  EXPECT_FALSE(AlgebraicSimplifier(opts).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest,
       DoNotReorderConvAddMulWithUnmatchingOutputFeatureDimension) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      input = f32[5,3,3,1] parameter(0)
      filter = f32[2,2,1,2] constant({{{{1.1, 1.2}}, {{2.1, 2.2}}},
                                      {{{3.1, 3.2}}, {{4.1, 4.2}}}})
      conv = f32[5,2,2,2] convolution(input, filter),
               window={size=2x2}, dim_labels=b01f_01io->b01f
      bias = f32[5,2,2,2] parameter(1)
      add = f32[5,2,2,2] add(conv, bias)
      constant = f32[2] constant({1.0, 1.1})
      bcast = f32[5,2,2,2] broadcast(constant), dimensions={2}
      ROOT multiply = f32[5,2,2,2] multiply(add, bcast)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions opts = default_options_;
  opts.set_enable_conv_add_multiply_reorder(true);
  EXPECT_FALSE(AlgebraicSimplifier(opts).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  EXPECT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  EXPECT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0s32, HloOpcode::kMultiply, param0, zero));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kMultiply);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Multiply(
          m::Parameter(0), m::Broadcast(m::Multiply(m::ConstantScalar(2.0),
                                                    m::ConstantScalar(3.0))))));
}

TEST_F(AlgebraicSimplifierTest, ElementwiseSinkMultipleBroadcastsScalar) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      b0 = f32[4] broadcast(p0), dimensions={}
      b1 = f32[4] broadcast(p1), dimensions={}
      ROOT multiply = f32[4] multiply(b1, b0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Broadcast(m::Multiply(m::Broadcast(m::Parameter(1)),
                                          m::Broadcast(m::Parameter(0))))));
}

TEST_F(AlgebraicSimplifierTest, ElementwiseSinkMultipleBroadcastsConstantMix) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4] parameter(0)
      c0 = f32[] constant(2.0)
      b0 = f32[4,2] broadcast(c0), dimensions={}
      b1 = f32[4,2] broadcast(p0), dimensions={0}
      ROOT multiply = f32[4,2] multiply(b1, b0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Multiply(
                  m::Parameter(0), m::Broadcast(m::ConstantScalar(2.0))))));
}

TEST_F(AlgebraicSimplifierTest, ElementwiseSinkMultipleBroadcastsNonScalar) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      b0 = f32[4,2] broadcast(p0), dimensions={0}
      b1 = f32[4,2] broadcast(p1), dimensions={0}
      ROOT multiply = f32[4,2] multiply(b1, b0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Broadcast(m::Multiply(m::Parameter(1), m::Parameter(0)))));
}

TEST_F(AlgebraicSimplifierTest, ElementwiseNoSinkBroadcastsDifferentDims) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4] parameter(0)
      p1 = f32[8] parameter(1)
      b0 = f32[4,8] broadcast(p0), dimensions={0}
      b1 = f32[4,8] broadcast(p1), dimensions={1}
      ROOT multiply = f32[4,8] multiply(b1, b0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(m::Broadcast(m::Parameter(1)),
                                     m::Broadcast(m::Parameter(0)))));
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  auto computation = module->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_EQ(computation->root_instruction(), param0);
}

// Test that select(true, a, b) is not simplified to a when mixed precision
TEST_F(AlgebraicSimplifierTest, SelectTrueMixedPrecision) {
  Shape r0bf16 = ShapeUtil::MakeShape(BF16, {});
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0bf16, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "param1"));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  builder.AddInstruction(HloInstruction::CreateTernary(
      r0f32, HloOpcode::kSelect, one, param0, param1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(module.get()).value());
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
  auto computation = module->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_EQ(computation->root_instruction(), param1);
}

// Test that select(false a, b) is not simplified to a when mixed precision
TEST_F(AlgebraicSimplifierTest, SelectFalseMixedPrecision) {
  Shape r0bf16 = ShapeUtil::MakeShape(BF16, {});
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0bf16, "param1"));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateTernary(
      r0f32, HloOpcode::kSelect, one, param0, param1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(module.get()).value());
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
  auto computation = module->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_EQ(computation->root_instruction(), param1);
}

// Test that select(a, b, b) is not simplified to a when mixed precision
TEST_F(AlgebraicSimplifierTest, SelectIdenticalMixedPrecision) {
  Shape r0bf16 = ShapeUtil::MakeShape(BF16, {});
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0bf16, "param1"));
  builder.AddInstruction(HloInstruction::CreateTernary(
      r0f32, HloOpcode::kSelect, param0, param1, param1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(module.get()).value());
}

// Test that select(not(pred), a, b) is simplified to select(pred, b, a)
TEST_F(AlgebraicSimplifierTest, SelectWithNotPred) {
  Shape pred_ty = ShapeUtil::MakeShape(PRED, {});
  Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, pred_ty, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0s32, "param1"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, r0s32, "param2"));
  HloInstruction* pred_instr = builder.AddInstruction(
      HloInstruction::CreateUnary(pred_ty, HloOpcode::kNot, param0));
  builder.AddInstruction(HloInstruction::CreateTernary(
      r0s32, HloOpcode::kSelect, pred_instr, param1, param2));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  const auto& operands = computation->root_instruction()->operands();
  EXPECT_EQ(operands[0], param0);
  EXPECT_EQ(operands[1], param2);
  EXPECT_EQ(operands[2], param1);
}

// Test that select(a, true, false) is simplified to a
TEST_F(AlgebraicSimplifierTest, SelectPredPred) {
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateTernary(
      r0pred, HloOpcode::kSelect, param0, one, zero));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_EQ(computation->root_instruction(), param0);
}

// Test that select(a, false, true) is simplified to not(a)
TEST_F(AlgebraicSimplifierTest, SelectPredPred2) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  builder.AddInstruction(HloInstruction::CreateTernary(
      r0pred, HloOpcode::kSelect, param0, zero, one));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSelect);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Not(m::Parameter(0))));
}

// select(compare(a, b, GT/GE), a, b) => or(a, b),   a,b ∈ PRED
TEST_F(AlgebraicSimplifierTest, SelectGtCompare) {
  for (const auto cmp_dir : {"GT", "GE"}) {
    const auto kModuleStr = absl::StrFormat(R"(
      HloModule m
      test {
        p0 = pred[8]{0} parameter(0)
        p1 = pred[8]{0} parameter(1)
        compare = pred[8]{0} compare(p0, p1), direction=%s
        ROOT select = pred[8]{0} select(compare, p0, p1)
      }
    )",
                                            cmp_dir);
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
    ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
    EXPECT_THAT(m->entry_computation()->root_instruction(),
                GmockMatch(m::Or(m::Parameter(0), m::Parameter(1))));
  }
}

// select(compare(a, b, LT/LE), a, b) => and(a, b),   a,b ∈ PRED
TEST_F(AlgebraicSimplifierTest, SelectLtCompare) {
  for (const auto cmp_dir : {"LT", "LE"}) {
    const auto kModuleStr = absl::StrFormat(R"(
      HloModule m
      test {
        p0 = pred[8]{0} parameter(0)
        p1 = pred[8]{0} parameter(1)
        compare = pred[8]{0} compare(p0, p1), direction=%s
        ROOT select = pred[8]{0} select(compare, p0, p1)
      }
    )",
                                            cmp_dir);
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
    ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
    EXPECT_THAT(m->entry_computation()->root_instruction(),
                GmockMatch(m::And(m::Parameter(0), m::Parameter(1))));
  }
}

// select(compare(a, b, EQ), a, b) => b,   a,b ∈ PRED
TEST_F(AlgebraicSimplifierTest, SelectEqCompare) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = pred[8]{0} parameter(0)
      p1 = pred[8]{0} parameter(1)
      compare = pred[8]{0} compare(p0, p1), direction=EQ
      ROOT select = pred[8]{0} select(compare, p0, p1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(1)));
}

// select(compare(a, b, NE), a, b) => a,   a,b ∈ PRED
TEST_F(AlgebraicSimplifierTest, SelectNeCompare) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = pred[8]{0} parameter(0)
      p1 = pred[8]{0} parameter(1)
      compare = pred[8]{0} compare(p0, p1), direction=NE
      ROOT select = pred[8]{0} select(compare, p0, p1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

// select(compare(a, b, NE), b, a) ≠> a - wrong operands order
TEST_F(AlgebraicSimplifierTest, SelectNeCompare_NegativeTestCase) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = pred[8]{0} parameter(0)
      p1 = pred[8]{0} parameter(1)
      compare = pred[8]{0} compare(p0, p1), direction=NE
      ROOT select = pred[8]{0} select(compare, p1, p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

// Test that select(pred, xs, dynamic_update_slice(xs, x, i)) is simplified
// to dynamic_update_slice(xs, select(pred, dynamic_slice(xs, i), x), i)
TEST_F(AlgebraicSimplifierTest, SelectDUSWithShapedPred) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = pred[8] parameter(0)
      xs = f32[8] parameter(1)
      x = f32[2] parameter(2)
      i = u32[] parameter(3)
      dus = f32[8] dynamic-update-slice(xs, x, i)
      ROOT out = f32[8] select(p, dus, xs)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicUpdateSlice(
                  m::Parameter(1),
                  m::Select(m::DynamicSlice(m::Parameter(0), m::Parameter(3)),
                            m::Parameter(2),
                            m::DynamicSlice(m::Parameter(1), m::Parameter(3))),
                  m::Parameter(3))));
}

TEST_F(AlgebraicSimplifierTest, ReverseSelectDUSWithShapedPred) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = pred[8] parameter(0)
      xs = f32[8] parameter(1)
      x = f32[2] parameter(2)
      i = u32[] parameter(3)
      dus = f32[8] dynamic-update-slice(xs, x, i)
      ROOT out = f32[8] select(p, xs, dus)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicUpdateSlice(
                  m::Parameter(1),
                  m::Select(m::DynamicSlice(m::Parameter(0), m::Parameter(3)),
                            m::DynamicSlice(m::Parameter(1), m::Parameter(3)),
                            m::Parameter(2)),
                  m::Parameter(3))));
}

TEST_F(AlgebraicSimplifierTest, SelectDUSNotTriggering) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = pred[8] parameter(0)
      xs = f32[8] parameter(1)
      x = f32[2] parameter(2)
      i = u32[] parameter(3)
      ys = f32[8] parameter(4)
      dus = f32[8] dynamic-update-slice(xs, x, i)
      ROOT out = f32[8] select(p, dus, ys)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Select(
                  m::Parameter(0),
                  m::DynamicUpdateSlice(m::Parameter(1), m::Parameter(2),
                                        m::Parameter(3)),
                  m::Parameter(4))));
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
  std::vector<int64_t> dims0({0});
  Shape r3f32 = ShapeUtil::MakeShape(F32, {5, 6, 7});
  HloInstruction* reduce0 = builder.AddInstruction(
      HloInstruction::CreateReduce(r3f32, param, zero, dims0, add_computation));
  std::vector<int64_t> dims1({1, 2});
  Shape r1f32 = ShapeUtil::MakeShape(F32, {5});
  builder.AddInstruction(HloInstruction::CreateReduce(r1f32, reduce0, zero,
                                                      dims1, add_computation));
  m->AddEntryComputationWithLayouts(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Reduce(m::Parameter(0), m::Op().Is(zero))));
  EXPECT_EQ(root->dimensions(), std::vector<int64_t>({0, 2, 3}));
}

TEST_F(AlgebraicSimplifierTest, ReduceOfMergeNoncontractingDims) {
  const char* kModuleStr = R"(
    HloModule m
    add_f32 {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT r = f32[] add(p0, p1)
    }

    ENTRY test {
      p = f32[3,5,7] parameter(0)
      reshape = f32[15,7] reshape(p)
      ROOT reduce = f32[15] reduce(reshape, f32[] constant(0)), dimensions={1}, to_apply=add_f32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options = default_options_;
  options.set_unconditionally_simplify_reduce_of_transpose_or_reshape(true);
  ASSERT_TRUE(AlgebraicSimplifier(options).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Reshape(m::Reduce()
                                .WithShape(F32, {3, 5})
                                .WithPredicate([](const HloInstruction* instr) {
                                  return instr->dimensions() ==
                                         std::vector<int64_t>({2});
                                }))
                     .WithShape(F32, {15})));
}

TEST_F(AlgebraicSimplifierTest, ReduceOfSplitNoncontractingDims) {
  const char* kModuleStr = R"(
    HloModule m
    add_f32 {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT r = f32[] add(p0, p1)
    }

    ENTRY test {
      p = f32[3,35] parameter(0)
      reshape = f32[3,5,7] reshape(p)
      ROOT reduce = f32[5,7] reduce(reshape, f32[] constant(0)), dimensions={0}, to_apply=add_f32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options = default_options_;
  options.set_unconditionally_simplify_reduce_of_transpose_or_reshape(true);
  ASSERT_TRUE(AlgebraicSimplifier(options).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Reshape(m::Reduce().WithShape(F32, {35}).WithPredicate(
                                [](const HloInstruction* instr) {
                                  return instr->dimensions() ==
                                         std::vector<int64_t>({0});
                                }))
                     .WithShape(F32, {5, 7})));
}

TEST_F(AlgebraicSimplifierTest,
       ReduceOfReshapeOfContractingAndNoncontractingDims) {
  const char* kModuleStr = R"(
    HloModule m
    add_f32 {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT r = f32[] add(p0, p1)
    }

    ENTRY test {
      ROOT reduce = f32[8] reduce(
        f32[8,4] reshape(f32[32] parameter(0)), f32[] constant(0)),
        dimensions={1}, to_apply=add_f32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options = default_options_;
  options.set_unconditionally_simplify_reduce_of_transpose_or_reshape(true);
  ASSERT_FALSE(AlgebraicSimplifier(options).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, ReduceOfNegate) {
  const char* kModuleStr = R"(
    HloModule m
    add_f32 {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT r = f32[] add(p0, p1)
    }

    ENTRY test {
      p = f32[15,7] parameter(0)
      n = negate(p)
      ROOT reduce = f32[15] reduce(n, f32[] constant(0)), dimensions={1}, to_apply=add_f32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options = default_options_;
  options.set_unconditionally_simplify_reduce_of_transpose_or_reshape(true);
  ASSERT_TRUE(AlgebraicSimplifier(options).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Negate(m::Reduce(m::Parameter(0), m::ConstantScalar(0)))));
}

TEST_F(AlgebraicSimplifierTest, ReduceBroadcastOfScalar) {
  // Test Reduce(Broadcast(x), a, Max)
  const char* kModuleStrForMax = R"(
    HloModule m
    max_f32 {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT r = f32[] maximum(p0, p1)
    }

    ENTRY test {
      p = f32[] parameter(0)
      b = f32[1000,1000] broadcast(p), dimensions={}
      ROOT reduce = f32[] reduce(b, f32[] constant(0)), dimensions={0,1}, to_apply=max_f32
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnVerifiedModule(kModuleStrForMax));
  AlgebraicSimplifierOptions options = default_options_;
  ASSERT_TRUE(AlgebraicSimplifier(options).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::MaximumAnyOrder(m::Parameter(0), m::ConstantScalar(0))));

  // Test Reduce(Broadcast(x), a, And)
  const char* kModuleStrForAnd = R"(
    HloModule m
    and_u4 {
      p0 = u4[] parameter(0)
      p1 = u4[] parameter(1)
      ROOT r = u4[] and(p0, p1)
    }

    ENTRY test {
      p = u4[] parameter(0)
      b = u4[1000,1000] broadcast(p), dimensions={}
      ROOT reduce = u4[] reduce(b, u4[] constant(0)), dimensions={0,1}, to_apply=and_u4
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(m, ParseAndReturnVerifiedModule(kModuleStrForAnd));
  ASSERT_TRUE(AlgebraicSimplifier(options).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::AndAnyOrder(m::Parameter(0), m::ConstantScalar(0))));
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Add(m::Parameter(0),
                                m::Broadcast(m::Add(m::ConstantScalar(1.0),
                                                    m::ConstantScalar(2.0))))));
}

TEST_F(AlgebraicSimplifierTest, ReplaceSubtractOfEqualOperandsWithZero) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      ROOT sub = f32[] subtract(p0, p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_enable_fast_math(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ConstantScalar(0.0)));
}

TEST_F(AlgebraicSimplifierTest,
       ReplaceSubtractOfEqualOperandsWithBroadcastZero) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[512,20] parameter(0)
      ROOT sub = f32[512,20] subtract(p0, p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_enable_fast_math(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast()));
}

TEST_F(AlgebraicSimplifierTest, SubAddReassociateMergeConstants) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      c0 = f32[] constant(1.0)
      c1 = f32[] constant(2.0)
      sub = f32[] subtract(c0, p0)
      ROOT add = f32[] add(sub, c1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Subtract(
                  m::Add(m::ConstantScalar(1.0), m::ConstantScalar(2.0)),
                  m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, ExpOfZero) {
  const char* m = R"(
  HloModule m
    ENTRY main{
      %constant = bf16[] constant(0)
      %broadcast = bf16[6,512]{1,0} broadcast(bf16[] %constant), dimensions={}
      ROOT exponential.11278 = bf16[6,512] exponential(%broadcast)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(m));
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::ConstantScalar(1.0))));
}

TEST_F(AlgebraicSimplifierTest, SubAddReassociateMergeBroadcastedConstants) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[4] parameter(0)
      c0 = f32[] constant(1.0)
      c1 = f32[] constant(2.0)
      b0 = f32[4] broadcast(c0), dimensions={}
      b1 = f32[4] broadcast(c1), dimensions={}
      sub = f32[4] subtract(b0, p0)
      ROOT add = f32[4] add(sub, b1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Subtract(
          m::Broadcast(m::Add(m::ConstantScalar(1.0), m::ConstantScalar(2.0))),
          m::Parameter(0))));
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kMap);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

TEST_F(AlgebraicSimplifierTest, ConstantToBroadcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({3.14f, 3.14f, 3.14f})));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Constant())));
  EXPECT_EQ(3.14f, root->operand(0)->literal().GetFirstElement<float>());
}

TEST_F(AlgebraicSimplifierTest, ConstantNotToBroadcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({3.14, 3.14, 4})));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(m.get()).value());
  root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
}

TEST_F(AlgebraicSimplifierTest, IotaToBroadcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({0.0f, 1.0f, 2.0f})));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSubtract);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kSubtract);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Add(m::Parameter(0),
                        m::Broadcast(m::Negate(m::ConstantScalar(0.125))))));
}

// Test that A - A is simplified to 0.
TEST_F(AlgebraicSimplifierTest, SubSame) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[2] parameter(0)
      ROOT sub = s32[2] subtract(p0, p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::ConstantScalar(0))));
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Reshape(m::Parameter(0)))));
}

// Test to catch a crash where we were overshooting the reshaped_dimensions
// vector.
TEST_F(AlgebraicSimplifierTest, ArrayOvershootTest) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      param0 = f32[18,18,2,1,1,128]{1,0,5,2,4,3} parameter(0)
      cpy1 = f32[18,18,2,1,1,128]{5,2,1,0,4,3} copy(f32[18,18,2,1,1,128]{1,0,5,2,4,3} param0)
      bitcast = f32[648,128,1,1]{3,2,1,0} bitcast(cpy1)
      ROOT cpy2 = f32[648,128,1,1]{3,2,0,1} copy(f32[648,128,1,1]{3,2,1,0} bitcast)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  // Assert false because algebraic simplifier - at the time of adding this
  // test - does not change anything. Motivation of the test to make sure it
  // does not crash the compiler.
  ASSERT_FALSE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Divide(m::Divide(m::Parameter(0), m::Parameter(1)),
                                   m::Parameter(2))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Parameter(0),
                           m::Divide(m::Parameter(1), m::Parameter(2)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Divide(m::Parameter(0), m::Parameter(1)),
                           m::Divide(m::Parameter(2), m::Parameter(3)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Divide(m::Parameter(0), m::Exp(m::Parameter(1)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Parameter(0),
                           m::Power(m::Parameter(1), m::Parameter(2)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Parameter(0),
                           m::Power(m::Parameter(1), m::Parameter(2)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());

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

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(m.get()).value());
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

  m->AddEntryComputationWithLayouts(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, div);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, div);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, cplx);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, real);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, imag);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, add);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Divide(m::Exp(m::Parameter(0)), m::Exp(m::Parameter(1)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Exp(m::Parameter(0)),
                                     m::Exp(m::Parameter(1)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Exp(m::Parameter(0)), m::Parameter(1))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Log(m::Power(m::Parameter(0), m::Parameter(1)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Select(
          m::Eq(m::Parameter(1), m::ConstantScalar(0.0f)),
          m::ConstantScalar(0.0f),
          m::Multiply(m::Log(m::Abs(m::Parameter(0))), m::Parameter(1)))));
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Log(m::Sqrt(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Log(m::Rsqrt(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Log(m::Exp(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Log(m::Divide(m::Exp(m::Parameter(0)),
                                          m::Exp(m::Parameter(1))))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(zero))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(zero))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(one))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(two))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(0))));
}

// Test that pow(A, 3) is simplified to A*A*A.
TEST_F(AlgebraicSimplifierTest, Pow3) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));
  HloInstruction* three = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(3)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kPower, param0, three));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(three))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Multiply(m::Parameter(0),
                             m::Multiply(m::Parameter(0), m::Parameter(0)))));
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Power(m::Parameter(0), m::Op().Is(negative_one))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  HloInstruction* root = computation->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Divide(m::Constant(), m::Parameter(0))));
  EXPECT_EQ(root->operand(0)->literal().GetFirstElement<float>(), 1);
}

// pow(A, 0.5) => sqrt(A), for A >= 0
TEST_F(AlgebraicSimplifierTest, PowHalf) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[1,32] parameter(0)
      c0 = f32[] constant(0.5)
      br0 = f32[1,32] broadcast(f32[] c0), dimensions={}
      abs0 = f32[1,32] abs(p0)
      ROOT pow = f32[1,32] power(abs0, br0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Sqrt(m::Abs(m::Parameter(0)))));
}

// pow(A, 0.5) ≠> sqrt(A)
// if A is arbitrary number - no simplification
TEST_F(AlgebraicSimplifierTest, PowHalf_NegativeTestCase) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[1,32] parameter(0)
      c0 = f32[] constant(0.5)
      br0 = f32[1,32] broadcast(f32[] c0), dimensions={}
      ROOT pow = f32[1,32] power(p0, br0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  m->AddEntryComputationWithLayouts(builder.Build());
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Convolution(m::Op().Is(lhs), m::Op().Is(rhs))));
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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
  for (int64_t i = 0; i < 4; ++i) {
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
  m->AddEntryComputationWithLayouts(builder.Build());
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceWindow(m::Parameter(0), m::Constant())));
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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
  for (int64_t i = 0; i < 2; ++i) {
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
  m->AddEntryComputationWithLayouts(builder.Build());
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceWindow(m::Parameter(0), m::Constant())));
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Constant())));
}

TEST_F(AlgebraicSimplifierTest, ZeroSizedVariadicReduceWindow) {
  const char* const hlo_string = R"(
HloModule ZeroSizedVariadicReduceWindow

ZeroSizedVariadicReduceWindow.add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  p2 = f32[] parameter(2)
  p3 = f32[] parameter(3)
  add.0 = f32[] add(p0, p1)
  add.1 = f32[] add(p2, p3)
  ROOT r = tuple(add.0, add.1)
}

ENTRY ZeroSizedReduceWindow {
  op = f32[3,0] parameter(0)
  constant = f32[] constant(0)
  ROOT reduce-window = (f32[5,2], f32[5,2]) reduce-window(op, op, constant, constant), window={size=1x1 pad=1_1x1_1}, to_apply=ZeroSizedVariadicReduceWindow.add
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ReduceWindow(m::Parameter(0), m::Parameter(0),
                                         m::Constant(), m::Constant())));
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Broadcast(m::Constant()),
                                  m::Broadcast(m::Constant()))));
}

TEST_F(AlgebraicSimplifierTest, NopMax) {
  const char* const hlo_string = R"(
HloModule test

ENTRY test {
  p_s8   = s8[]   parameter(0)
  p_u8   = u8[]   parameter(1)
  p_s16  = s16[]  parameter(2)
  p_u16  = u16[]  parameter(3)
  p_s32  = s32[]  parameter(4)
  p_u32  = u32[]  parameter(5)
  p_s64  = s64[]  parameter(6)
  p_u64  = u64[]  parameter(7)
  p_f16  = f16[]  parameter(8)
  p_bf16 = bf16[] parameter(9)
  p_f32  = f32[]  parameter(10)
  p_f64  = f64[]  parameter(11)

  const_s8   = s8[]   constant(-128)
  const_u8   = u8[]   constant(0)
  const_s16  = s16[]  constant(-32768)
  const_u16  = u16[]  constant(0)
  const_s32  = s32[]  constant(-2147483648)
  const_u32  = u32[]  constant(0)
  const_s64  = s64[]  constant(-9223372036854775808)
  const_u64  = u64[]  constant(0)
  const_f16  = f16[]  constant(-inf)
  const_bf16 = bf16[] constant(-inf)
  const_f32  = f32[]  constant(-inf)
  const_f64  = f64[]  constant(-inf)

  max_s8   = s8[]   maximum(p_s8, const_s8)
  max_u8   = u8[]   maximum(p_u8, const_u8)
  max_s16  = s16[]  maximum(p_s16, const_s16)
  max_u16  = u16[]  maximum(p_u16, const_u16)
  max_s32  = s32[]  maximum(p_s32, const_s32)
  max_u32  = u32[]  maximum(p_u32, const_u32)
  max_s64  = s64[]  maximum(p_s64, const_s64)
  max_u64  = u64[]  maximum(p_u64, const_u64)
  max_f16  = f16[]  maximum(p_f16, const_f16)
  max_bf16 = bf16[] maximum(p_bf16, const_bf16)
  max_f32  = f32[]  maximum(p_f32, const_f32)
  max_f64  = f64[]  maximum(p_f64, const_f64)

  max2_s8   = s8[]   maximum(const_s8, p_s8)
  max2_u8   = u8[]   maximum(const_u8, p_u8)
  max2_s16  = s16[]  maximum(const_s16, p_s16)
  max2_u16  = u16[]  maximum(const_u16, p_u16)
  max2_s32  = s32[]  maximum(const_s32, p_s32)
  max2_u32  = u32[]  maximum(const_u32, p_u32)
  max2_s64  = s64[]  maximum(const_s64, p_s64)
  max2_u64  = u64[]  maximum(const_u64, p_u64)
  max2_f16  = f16[]  maximum(const_f16, p_f16)
  max2_bf16 = bf16[] maximum(const_bf16, p_bf16)
  max2_f32  = f32[]  maximum(const_f32, p_f32)
  max2_f64  = f64[]  maximum(const_f64, p_f64)

  ROOT tuple = tuple(max_s8, max_u8, max_s16, max_u16, max_s32, max_u32,
                     max_s64, max_u64, max_f16, max_bf16, max_f32, max_f64,
                     max2_s8, max2_u8, max2_s16, max2_u16, max2_s32, max2_u32,
                     max2_s64, max2_u64, max2_f16, max2_bf16, max2_f32, max2_f64)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  // We can't write GmockMatch(m::Tuple(m::Parameter(0), m::Parameter(1), ...)
  // because this generates a template expression that's too complicated for our
  // MSVC to compile.  :(
  SCOPED_TRACE(m->ToString());
  const HloInstruction* root;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::Op(&root).WithOpcode(HloOpcode::kTuple).WithNumOperands(24)));
  for (int i = 0; i < root->operand_count(); i++) {
    SCOPED_TRACE(absl::StrCat("operand ", i));
    const HloInstruction* operand = root->operand(i);
    ASSERT_EQ(operand->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(operand->parameter_number(), i % 12);
  }
}

TEST_F(AlgebraicSimplifierTest, NopMin) {
  const char* const hlo_string = R"(
HloModule test

ENTRY test {
  p_s8   = s8[]   parameter(0)
  p_u8   = u8[]   parameter(1)
  p_s16  = s16[]  parameter(2)
  p_u16  = u16[]  parameter(3)
  p_s32  = s32[]  parameter(4)
  p_u32  = u32[]  parameter(5)
  p_s64  = s64[]  parameter(6)
  p_u64  = u64[]  parameter(7)
  p_f16  = f16[]  parameter(8)
  p_bf16 = bf16[] parameter(9)
  p_f32  = f32[]  parameter(10)
  p_f64  = f64[]  parameter(11)

  const_s8   = s8[]   constant(127)
  const_u8   = u8[]   constant(255)
  const_s16  = s16[]  constant(32767)
  const_u16  = u16[]  constant(65535)
  const_s32  = s32[]  constant(2147483647)
  const_u32  = u32[]  constant(4294967295)
  const_s64  = s64[]  constant(9223372036854775807)
  const_u64  = u64[]  constant(18446744073709551615)
  const_f16  = f16[]  constant(inf)
  const_bf16 = bf16[] constant(inf)
  const_f32  = f32[]  constant(inf)
  const_f64  = f64[]  constant(inf)

  min_s8   = s8[]   minimum(p_s8, const_s8)
  min_u8   = u8[]   minimum(p_u8, const_u8)
  min_s16  = s16[]  minimum(p_s16, const_s16)
  min_u16  = u16[]  minimum(p_u16, const_u16)
  min_s32  = s32[]  minimum(p_s32, const_s32)
  min_u32  = u32[]  minimum(p_u32, const_u32)
  min_s64  = s64[]  minimum(p_s64, const_s64)
  min_u64  = u64[]  minimum(p_u64, const_u64)
  min_f16  = f16[]  minimum(p_f16, const_f16)
  min_bf16 = bf16[] minimum(p_bf16, const_bf16)
  min_f32  = f32[]  minimum(p_f32, const_f32)
  min_f64  = f64[]  minimum(p_f64, const_f64)

  min2_s8   = s8[]   minimum(const_s8, p_s8)
  min2_u8   = u8[]   minimum(const_u8, p_u8)
  min2_s16  = s16[]  minimum(const_s16, p_s16)
  min2_u16  = u16[]  minimum(const_u16, p_u16)
  min2_s32  = s32[]  minimum(const_s32, p_s32)
  min2_u32  = u32[]  minimum(const_u32, p_u32)
  min2_s64  = s64[]  minimum(const_s64, p_s64)
  min2_u64  = u64[]  minimum(const_u64, p_u64)
  min2_f16  = f16[]  minimum(const_f16, p_f16)
  min2_bf16 = bf16[] minimum(const_bf16, p_bf16)
  min2_f32  = f32[]  minimum(const_f32, p_f32)
  min2_f64  = f64[]  minimum(const_f64, p_f64)

  ROOT tuple = tuple(min_s8, min_u8, min_s16, min_u16, min_s32, min_u32,
                     min_s64, min_u64, min_f16, min_bf16, min_f32, min_f64,
                     min2_s8, min2_u8, min2_s16, min2_u16, min2_s32, min2_u32,
                     min2_s64, min2_u64, min2_f16, min2_bf16, min2_f32, min2_f64)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  SCOPED_TRACE(m->ToString());

  // We can't write GmockMatch(m::Tuple(m::Parameter(0), m::Parameter(1), ...)
  // because this generates a template expression that's too complicated for our
  // MSVC to compile.  :(
  SCOPED_TRACE(m->ToString());
  const HloInstruction* root;
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::Op(&root).WithOpcode(HloOpcode::kTuple).WithNumOperands(24)));
  for (int i = 0; i < root->operand_count(); i++) {
    SCOPED_TRACE(absl::StrCat("operand ", i));
    const HloInstruction* operand = root->operand(i);
    ASSERT_EQ(operand->opcode(), HloOpcode::kParameter);
    EXPECT_EQ(operand->parameter_number(), i % 12);
  }
}

TEST_F(AlgebraicSimplifierTest, MinimumWithInfinityLhs) {
  const char* const kHloString = R"(
HloModule test

ENTRY main {
  x = f16[] constant(inf)
  y = f32[] parameter(0)
  ROOT min = f32[] minimum(x, y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHloString));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(AlgebraicSimplifierTest, MinimumWithInfinityLhsAndRhsF16Type) {
  const char* const kHloString = R"(
HloModule test

ENTRY main {
  x = f32[] constant(inf)
  y = f16[] parameter(0)
  ROOT min = f32[] minimum(x, y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHloString));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, MinimumWithInfinityRhs) {
  const char* const kHloString = R"(
HloModule test

ENTRY main {
  x = f32[] parameter(0)
  y = f16[] constant(inf)
  ROOT min = f32[] minimum(x, y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHloString));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(AlgebraicSimplifierTest, MinimumWithInfinityRhsAndLhsF16Type) {
  const char* const kHloString = R"(
HloModule test

ENTRY main {
  x = f16[] parameter(0)
  y = f32[] constant(inf)
  ROOT min = f32[] minimum(x, y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHloString));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, MinimumOfMinimum1) {
  const char* const hlo_string = R"(
HloModule test

ENTRY main {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  min1 = f32[] minimum(x, y)
  ROOT min = f32[] minimum(min1, y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Minimum(m::Parameter(0), m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, MinimumOfMinimum2) {
  const char* const hlo_string = R"(
HloModule test

ENTRY main {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  min1 = f32[] minimum(x, y)
  ROOT min = f32[] minimum(min1, x)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Minimum(m::Parameter(0), m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, MinimumOfMinimum3) {
  const char* const hlo_string = R"(
HloModule test

ENTRY main {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  min1 = f32[] minimum(x, y)
  ROOT min = f32[] minimum(y, min1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Minimum(m::Parameter(1), m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, MinimumOfMinimum4) {
  const char* const hlo_string = R"(
HloModule test

ENTRY main {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  min1 = f32[] minimum(x, y)
  ROOT min = f32[] minimum(x, min1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Minimum(m::Parameter(0), m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, MaximumWithNegativeInfinityLhs) {
  const char* const kHloString = R"(
HloModule test

ENTRY main {
  x = f16[] constant(-inf)
  y = f32[] parameter(0)
  ROOT max = f32[] maximum(x, y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHloString));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(AlgebraicSimplifierTest, MaximumWithNegativeInfinityLhsAndRhsF16Type) {
  const char* const kHloString = R"(
HloModule test

ENTRY main {
  x = f32[] constant(-inf)
  y = f16[] parameter(0)
  ROOT max = f32[] maximum(x, y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHloString));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, MaximumWithNegativeInfinityRhs) {
  const char* const kHloString = R"(
HloModule test

ENTRY main {
  x = f32[] parameter(0)
  y = f16[] constant(-inf)
  ROOT max = f32[] maximum(x, y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHloString));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(AlgebraicSimplifierTest, MaximumWithNegativeInfinityRhsAndLhsF16Type) {
  const char* const kHloString = R"(
HloModule test

ENTRY main {
  x = f16[] parameter(0)
  y = f32[] constant(-inf)
  ROOT max = f32[] maximum(x, y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHloString));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, MaximumOfMaximum1) {
  const char* const hlo_string = R"(
HloModule test

ENTRY main {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  max1 = f32[] maximum(x, y)
  ROOT max = f32[] maximum(max1, y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Maximum(m::Parameter(0), m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, MaximumOfMaximum2) {
  const char* const hlo_string = R"(
HloModule test

ENTRY main {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  max1 = f32[] maximum(x, y)
  ROOT max = f32[] maximum(max1, x)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Maximum(m::Parameter(0), m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, MaximumOfMaximum3) {
  const char* const hlo_string = R"(
HloModule test

ENTRY main {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  max1 = f32[] maximum(x, y)
  ROOT max = f32[] maximum(y, max1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Maximum(m::Parameter(1), m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, MaximumOfMaximum4) {
  const char* const hlo_string = R"(
HloModule test

ENTRY main {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  max1 = f32[] maximum(x, y)
  ROOT max = f32[] maximum(x, max1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Maximum(m::Parameter(0), m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, TrivialReduceWindow_Add) {
  const char* const hlo_string = R"(
HloModule test

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

ENTRY test {
  p = f32[16,32] parameter(0)
  constant = f32[] constant(0)
  ROOT reduce-window = reduce-window(p, constant), window={size=1x1}, to_apply=add
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::AddAnyOrder(m::Parameter(),
                                m::Broadcast(m::ConstantEffectiveScalar(0)))));
}

TEST_F(AlgebraicSimplifierTest, TrivialReduceWindow_Min) {
  const char* const hlo_string = R"(
HloModule test

min {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT min = f32[] minimum(p0, p1)
}

ENTRY test {
  p = f32[16,32] parameter(0)
  constant = f32[] constant(inf)
  ROOT reduce-window = reduce-window(p, constant), window={size=1x1}, to_apply=min
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::MinimumAnyOrder(
          m::Parameter(), m::Broadcast(m::ConstantEffectiveScalar(
                              std::numeric_limits<float>::infinity())))));
}

TEST_F(AlgebraicSimplifierTest, TrivialReduceWindow_Max) {
  const char* const hlo_string = R"(
HloModule test

max {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT max = f32[] maximum(p0, p1)
}

ENTRY test {
  p = f32[16,32] parameter(0)
  constant = f32[] constant(-inf)
  ROOT reduce-window = reduce-window(p, constant), window={size=1x1}, to_apply=max
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::MaximumAnyOrder(
          m::Parameter(), m::Broadcast(m::ConstantEffectiveScalar(
                              -std::numeric_limits<float>::infinity())))));
}

TEST_F(AlgebraicSimplifierTest, TrivialReduceWindowWithPad) {
  const char* const hlo_string = R"(
HloModule test

max {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT max = f32[] maximum(p0, p1)
}

ENTRY test {
  p = f32[16,32] parameter(0)
  constant = f32[] constant(-inf)
  ROOT reduce-window = reduce-window(p, constant), window={size=1x1 pad=1_2x3_4}, to_apply=max
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Pad(
          m::MaximumAnyOrder(m::Parameter(),
                             m::Broadcast(m::ConstantEffectiveScalar(
                                 -std::numeric_limits<float>::infinity()))),
          m::ConstantEffectiveScalar(
              -std::numeric_limits<float>::infinity()))));
}

TEST_F(AlgebraicSimplifierTest, TrivialReduceWindowWithUnsupported) {
  const char* const hlo_string = R"(
HloModule test

max {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT max = f32[] maximum(p0, p1)
}

unsupported_fn {
  p0 = f32[] parameter(0)
  ROOT p1 = f32[] parameter(1)
}

ENTRY test {
  p = f32[16,32] parameter(0)
  constant = f32[] constant(-inf)
  a = reduce-window(p, constant), window={size=1x1 pad=1_2x3_4 stride=1x2}, to_apply=max
  b = reduce-window(p, constant), window={size=1x1 pad=1_2x3_4 lhs_dilate=2x1}, to_apply=max
  c = reduce-window(p, constant), window={size=1x1 pad=1_2x3_4 rhs_dilate=2x1}, to_apply=max
  d = reduce-window(p, constant), window={size=1x1 pad=1_2x3_4 rhs_reversal=1x1}, to_apply=max
  e = reduce-window(p, constant), window={size=1x1}, to_apply=unsupported_fn
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(RunHloPass(&simplifier, m.get()).value());
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
  m->AddEntryComputationWithLayouts(builder.Build());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Constant())));
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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
  m->AddEntryComputationWithLayouts(std::move(computation));

  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Reshape(m::Op().Is(op))))));

  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Convert(m::Op().Is(input))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  EXPECT_THAT(computation->root_instruction(), input);
}

// Test that convert(convert(A, $TYPE1), $TYPE2) is simplified to A if A is of
// $TYPE2 and convert(A, $TYP1) is an upcast.
TEST_F(AlgebraicSimplifierTest, EliminateConvertPairUpCast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0,
                                      ShapeUtil::MakeShapeWithDenseLayout(
                                          F16, {1, 14, 14, 64}, {3, 2, 1, 0}),
                                      "param"));
  HloInstruction* convert_1 =
      builder.AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(input->shape(), F32), input));
  builder.AddInstruction(HloInstruction::CreateConvert(
      ShapeUtil::ChangeElementType(convert_1->shape(), F16), convert_1));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Convert(m::Convert(m::Op().Is(input)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  EXPECT_THAT(computation->root_instruction(), input);
}

// Test that convert(convert(A, $TYPE1), $TYPE2) is NOT simplified to A even if
// A is of $TYPE2 since convert(A, $TYP1) is a downcast.
TEST_F(AlgebraicSimplifierTest, DoNotEliminateConvertPairDownCast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0,
                                      ShapeUtil::MakeShapeWithDenseLayout(
                                          F32, {1, 14, 14, 64}, {3, 2, 1, 0}),
                                      "param"));
  HloInstruction* convert_1 =
      builder.AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(input->shape(), F16), input));
  builder.AddInstruction(HloInstruction::CreateConvert(
      ShapeUtil::ChangeElementType(convert_1->shape(), F32), convert_1));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Convert(m::Convert(m::Op().Is(input)))));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(m.get()).value());
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
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0,
                                      ShapeUtil::MakeShapeWithDenseLayout(
                                          F16, {1, 14, 14, 64}, {3, 2, 1, 0}),
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Tuple(m::Op().Is(convert_1), m::Op().Is(convert_2),
                                  m::Op().Is(floor))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  EXPECT_THAT(computation->root_instruction(), param0);
}

TEST_F(AlgebraicSimplifierTest, CopyOfReshapeOfCopyEqualsBitcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0,
                                      ShapeUtil::MakeShapeWithDenseLayout(
                                          F32, {1, 14, 14, 64}, {3, 2, 1, 0}),
                                      "param"));
  HloInstruction* copy = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 14, 14, 64}, {0, 1, 2, 3}),
      HloOpcode::kCopy, param));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShapeWithDenseLayout(F32, {14 * 14, 64}, {0, 1}),
          copy));
  builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {14 * 14, 64}, {1, 0}),
      HloOpcode::kCopy, reshape));
  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Reshape(m::Copy(m::Parameter(0))))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  // Verify that the copy of reshape of copy is replaced.
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Bitcast(m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, ReshapeOfCopyEqualsBitcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0,
                                      ShapeUtil::MakeShapeWithDenseLayout(
                                          F32, {1, 14, 14, 64}, {3, 2, 1, 0}),
                                      "param"));
  HloInstruction* copy = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 14, 14, 64}, {0, 1, 2, 3}),
      HloOpcode::kCopy, param));
  builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {14 * 14, 64}, {1, 0}), copy));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Copy(m::Parameter(0)))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  // Verify that the copy of reshape of copy is replaced.
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Bitcast(m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, CopyEqualsBitcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0,
                                      ShapeUtil::MakeShapeWithDenseLayout(
                                          F32, {1, 14, 14, 64}, {0, 1, 2, 3}),
                                      "param"));
  builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 14, 14, 64}, {1, 2, 0, 3}),
      HloOpcode::kCopy, param));
  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));

  AlgebraicSimplifierOptions options(
      [](const Shape&, const Shape&) { return false; });
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier1(options);
  ASSERT_FALSE(simplifier1.Run(m.get()).value());
  // Verify that the copy is not replaced.
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));

  AlgebraicSimplifierOptions options2;
  options2.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier2(options2);
  EXPECT_TRUE(simplifier2.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Concatenate(m::Parameter(0))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  EXPECT_THAT(computation->root_instruction(), param0);
}

// Test that unary concatenates are removed.
TEST_F(AlgebraicSimplifierTest, DoNotRemoveUnaryConcatenateWithCtrlDep) {
  auto m = CreateNewVerifiedModule();
  Shape r1f32 = ShapeUtil::MakeShape(F32, {100});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1f32, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r1f32, "param1"));

  HloInstruction* concat = builder.AddInstruction(
      HloInstruction::CreateConcatenate(param0->shape(), {param0}, 0));
  TF_ASSERT_OK(param1->AddControlDependencyTo(concat));
  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Concatenate(m::Parameter(0))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, SliceReverse) {
  const char* const hlo_string = R"(
HloModule module

ENTRY test {
  param = f32[6,7,32] parameter(0)
  constant = f32[] constant(0)
  pad = f32[8,7,32] pad(param, constant), padding=1_1x0_0x0_0
  rev = f32[8,7,32] reverse(pad), dimensions={0,2}
  slice = f32[1,7,32] slice(rev), slice={[2:3:1], [0:7:1], [0:32:1]}
  ROOT tuple = (f32[1,7,32]) tuple(slice)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  HloComputation* computation = module->entry_computation();
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Tuple(m::Reverse(m::Slice(m::Pad())))));
  const HloInstruction* slice =
      computation->root_instruction()->operand(0)->operand(0);
  EXPECT_TRUE(
      ShapeUtil::Equal(slice->shape(), ShapeUtil::MakeShape(F32, {1, 7, 32})));
  // slice start,limit of 0th and 2nd dimensions are changed
  // while 1st dimension's slice start, limit remains the same since
  // it is not reversed.
  EXPECT_EQ(slice->slice_starts(0), 5);
  EXPECT_EQ(slice->slice_limits(0), 6);
  EXPECT_EQ(slice->slice_starts(1), 0);
  EXPECT_EQ(slice->slice_limits(1), 7);
  EXPECT_EQ(slice->slice_starts(2), 0);
  EXPECT_EQ(slice->slice_limits(2), 32);
  EXPECT_EQ(slice->slice_strides(0), 1);
  EXPECT_EQ(slice->slice_strides(1), 1);
  EXPECT_EQ(slice->slice_strides(2), 1);
}

TEST_F(AlgebraicSimplifierTest, SliceReverseNonUnitEvenOddStrides) {
  const char* const hlo_string = R"(
HloModule module

ENTRY test {
  param = f32[6,7,32] parameter(0)
  constant = f32[] constant(0)
  pad = f32[8,7,32] pad(param, constant), padding=1_1x0_0x0_0
  rev = f32[8,7,32] reverse(pad), dimensions={0,1,2}
  slice = f32[1,2,7] slice(rev), slice={[2:3:2], [0:7:4], [0:32:5]}
  ROOT tuple = (f32[1,2,7]) tuple(slice)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  HloComputation* computation = module->entry_computation();
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Tuple(m::Reverse(m::Slice(m::Pad())))));
  const HloInstruction* slice =
      computation->root_instruction()->operand(0)->operand(0);
  EXPECT_TRUE(
      ShapeUtil::Equal(slice->shape(), ShapeUtil::MakeShape(F32, {1, 2, 7})));
  // slice start,limit of all dimensions are changed
  EXPECT_EQ(slice->slice_starts(0), 5);
  EXPECT_EQ(slice->slice_limits(0), 6);
  EXPECT_EQ(slice->slice_starts(1), 2);
  EXPECT_EQ(slice->slice_limits(1), 7);
  EXPECT_EQ(slice->slice_starts(2), 1);
  EXPECT_EQ(slice->slice_limits(2), 32);
  EXPECT_EQ(slice->slice_strides(0), 2);
  EXPECT_EQ(slice->slice_strides(1), 4);
  EXPECT_EQ(slice->slice_strides(2), 5);
}

TEST_F(AlgebraicSimplifierTest, SliceRedundantStrides) {
  constexpr absl::string_view kHloString = R"(
HloModule module

ENTRY test {
  param.0 = f32[6,7,32] parameter(0)
  param.1 = f32[6,7,32,187] parameter(1)
  slice.0 = f32[1,2,7] slice(param.0), slice={[2:3:2], [0:7:4], [0:32:5]}
  slice.1 = f32[2,2,7] slice(param.0), slice={[2:6:3], [0:7:4], [0:32:5]}
  slice.2 = f32[2,2,1] slice(param.0), slice={[2:6:3], [0:7:4], [0:32:32]}
  slice.3 = f32[2,2,1,1] slice(param.1), slice={[2:6:3], [0:7:4], [0:32:32], [3:187:187]}
  ROOT tuple = (f32[1,2,7], f32[2,2,7], f32[2,2,1], f32[2,2,1,1]) tuple(slice.0, slice.1, slice.2, slice.3)
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_THAT(simplifier.Run(module.get()), IsOkAndHolds(true));
  const HloInstruction* slice_0 = FindInstruction(module.get(), "slice.0");
  EXPECT_EQ(slice_0->slice_starts(0), 2);
  EXPECT_EQ(slice_0->slice_limits(0), 3);
  EXPECT_EQ(slice_0->slice_strides(0), 1);

  // No change to slice.1.
  const HloInstruction* slice_1 = FindInstruction(module.get(), "slice.1");
  EXPECT_EQ(slice_1->slice_starts(0), 2);
  EXPECT_EQ(slice_1->slice_limits(0), 6);
  EXPECT_EQ(slice_1->slice_strides(0), 3);

  const HloInstruction* slice_2 = FindInstruction(module.get(), "slice.2");
  EXPECT_EQ(slice_2->slice_starts(2), 0);
  EXPECT_EQ(slice_2->slice_limits(2), 1);
  EXPECT_EQ(slice_2->slice_strides(2), 1);

  const HloInstruction* slice_3 = FindInstruction(module.get(), "slice.3");
  EXPECT_EQ(slice_3->slice_starts(2), 0);
  EXPECT_EQ(slice_3->slice_limits(2), 1);
  EXPECT_EQ(slice_3->slice_strides(2), 1);
  EXPECT_EQ(slice_3->slice_starts(3), 3);
  EXPECT_EQ(slice_3->slice_limits(3), 4);
  EXPECT_EQ(slice_3->slice_strides(3), 1);
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Concatenate(
                  m::Op().Is(empty_literal), m::Parameter(0), m::Parameter(0),
                  m::Op().Is(empty_slice), m::Parameter(1))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Map(m::Map(m::Reduce(m::Parameter(0), m::Op().Is(zero)),
                               m::Reduce(m::Parameter(1), m::Op().Is(zero))),
                        m::Reduce(m::Parameter(2), m::Op().Is(zero)))));
}

// Test that reduce of concat is simplified if the concat operand shapes
// differ and enable_unconditional_reduce_of_concat_replacement() is true.
TEST_F(AlgebraicSimplifierTest, SimplifyReduceOfConcatWithDifferentShapes) {
  const char* kModuleStr = R"(
    HloModule m
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    ENTRY test {
      p0 = f32[100,100,100]{2,1,0} parameter(0)
      p1 = f32[100,100,100]{2,1,0} parameter(1)
      p2 = f32[100,90,100]{2,1,0} parameter(2)
      cat = f32[100,290,100]{2,1,0} concatenate(p0, p1, p2), dimensions={1}
      cst = f32[] constant(0)
      ROOT reduce = f32[100]{0} reduce(cat, cst), dimensions={1,2}, to_apply=add
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());

  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Map(m::Map(m::Reduce(m::Parameter(0), m::Constant()),
                               m::Reduce(m::Parameter(1), m::Constant())),
                        m::Reduce(m::Parameter(2), m::Constant()))));
}

// Test that reduce of concat is not simplified if the concat operand shapes
// differ and enable_unconditional_reduce_of_concat_replacement() is false.
TEST_F(AlgebraicSimplifierTest,
       DoNotSimplifyReduceOfConcatBecauseShapesDiffer) {
  const char* kModuleStr = R"(
    HloModule m
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    ENTRY test {
      p0 = f32[100,100,100]{2,1,0} parameter(0)
      p1 = f32[100,100,100]{2,1,0} parameter(1)
      p2 = f32[100,90,100]{2,1,0} parameter(2)
      cat = f32[100,290,100]{2,1,0} concatenate(p0, p1, p2), dimensions={1}
      cst = f32[] constant(0)
      ROOT reduce = f32[100]{0} reduce(cat, cst), dimensions={1,2}, to_apply=add
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options = default_options_;
  options.set_enable_unconditional_reduce_of_concat_replacement(false);
  ASSERT_FALSE(AlgebraicSimplifier(options).Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Concatenate(m::Op().Is(empty_literal),
                                        m::Op().Is(empty_slice))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, SimplifyConcatenateOfSlices) {
  auto m = CreateNewVerifiedModule();
  Shape r2f32 = ShapeUtil::MakeShape(F32, {100, 99});
  Shape concat_shape = ShapeUtil::MakeShape(F32, {50, 90});
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
  // Can merge 'slice7' and 'slice8'.
  HloInstruction* slice8 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {50, 10}), param1, /*start_indices=*/{50, 89},
      /*limit_indices=*/{100, 99}, /*strides=*/{1, 1}));

  builder.AddInstruction(HloInstruction::CreateConcatenate(
      concat_shape,
      {slice0, slice1, slice2, slice3, slice4, slice5, slice6, slice7, slice8},
      1));
  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  // The operand 6 should be  merge of 'slice7' and 'slice8', so its
  // shape should have dimensions {50, 20}
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->operand(5)->shape(),
                       ShapeUtil::MakeShape(F32, {50, 20})));
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

  // Set to different layouts.
  *param0->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  *copy->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({1, 0});

  HloComputation* computation =
      m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(m.get()).value());

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

  // Set to same layouts.
  *param0->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  *copy->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  HloComputation* computation =
      m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  // Copy has been removed.
  EXPECT_THAT(computation->root_instruction(), param0);
}

// Test that a simplification which changes copy to a bitcast is not performed
// if layout sensitive is true.
TEST_F(AlgebraicSimplifierTest, CopyWithDifferentMemorySpaces) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 2}), "param0"));
  HloInstruction* copy = builder.AddInstruction(
      HloInstruction::CreateUnary(param0->shape(), HloOpcode::kCopy, param0));

  // Set to different memory spaces.
  param0->mutable_shape()->mutable_layout()->set_memory_space(0);
  copy->mutable_shape()->mutable_layout()->set_memory_space(123);

  HloComputation* computation =
      m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(m.get()).value());

  // Copy has not been removed.
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Parameter(0))));
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Parameter(0))));

  AlgebraicSimplifierOptions options(
      [](const Shape&, const Shape&) { return false; });
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  AlgebraicSimplifier simplifier(AlgebraicSimplifierOptions{});
  EXPECT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Tuple(m::Op().Is(transformable_reshape),
                                  m::Op().Is(dimensions_wrong_reshape),
                                  m::Op().Is(layout_wrong_reshape))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  simplifier.Run(m.get()).value();

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
  m->AddEntryComputationWithLayouts(builder.Build());
  EXPECT_TRUE(simplifier.Run(m.get()).value());
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
  m->AddEntryComputationWithLayouts(builder.Build());
  EXPECT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Transpose(m::Parameter(0))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  // Verify that the transpose is replaced.
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Transpose(m::Parameter(0))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  // Verify that the transpose is replaced.
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Reshape(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 2, 2}, {0, 1, 2}),
      HloOpcode::kCopy, param0));

  builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 2, 2}, {0, 2, 1}),
      HloOpcode::kCopy, copy1));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Copy(m::Copy(m::Parameter(0)))));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Transpose(m::Op().Is(transpose1))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Transpose(m::Parameter(0))));
  EXPECT_EQ(std::vector<int64_t>({2, 1, 0}),
            computation->root_instruction()->dimensions());
}

TEST_F(AlgebraicSimplifierTest, SliceOfBroadcast) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      p0 = f32[10,20] parameter(0)
      b = f32[10,30,20] broadcast(p0), dimensions={0,2}
      ROOT s = f32[5,5,5] slice(b), slice={[0:5:1], [5:25:4], [5:15:2]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Slice(m::Parameter(0)))));
}

TEST_F(AlgebraicSimplifierTest, SliceOfBroadcastPreserveLayout) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      p0 = f32[10,20] parameter(0)
      b = f32[10,30,20]{2,0,1:T(256)} broadcast(p0), dimensions={0,2}
      ROOT s = f32[5,5,5]{2,0,1:T(256)} slice(b), slice={[0:5:1], [5:25:4], [5:15:2]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  const Shape original_slice_shape =
      module->entry_computation()->root_instruction()->shape();
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Slice(m::Parameter(0)))));
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), original_slice_shape));
}

TEST_F(AlgebraicSimplifierTest, DynamicSliceOfBroadcast) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      p0 = f32[10,20] parameter(0)
      i0 = s32[] parameter(1)
      i1 = s32[] parameter(2)
      i2 = s32[] parameter(3)
      b = f32[10,30,20] broadcast(p0), dimensions={0,2}
      ROOT ds = f32[5,5,5] dynamic-slice(b, i0, i1, i2), dynamic_slice_sizes={5,5,5}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::DynamicSlice(
                        m::Parameter(0), m::Parameter(1), m::Parameter(3)))));
}

TEST_F(AlgebraicSimplifierTest, DynamicSliceOfBroadcastPreserveLayout) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      p0 = f32[10,20] parameter(0)
      i0 = s32[] parameter(1)
      i1 = s32[] parameter(2)
      i2 = s32[] parameter(3)
      b = f32[10,30,20]{2,0,1:T(256)} broadcast(p0), dimensions={0,2}
      ROOT ds = f32[5,5,5]{2,0,1:T(256)} dynamic-slice(b, i0, i1, i2), dynamic_slice_sizes={5,5,5}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  const Shape original_dynslice_shape =
      module->entry_computation()->root_instruction()->shape();
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::DynamicSlice(
                        m::Parameter(0), m::Parameter(1), m::Parameter(3)))));
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), original_dynslice_shape));
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
  EXPECT_TRUE(simplifier.Run(module.get()).value());
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
      ShapeUtil::MakeShape(F32, {1, 2, 3, 5, 1}), reshape1, {0, 3, 4}));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Reshape(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  EXPECT_TRUE(simplifier.Run(m.get()).value());

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

  HloComputation* computation =
      m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  HloComputation* computation =
      m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  HloComputation* computation =
      m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Broadcast(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  EXPECT_FALSE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Iota()));
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), result_shape));
}

TEST_F(AlgebraicSimplifierTest, IotaAndReshapeToMixedRadix) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(F32, {21}), 0));
  Shape result_shape = ShapeUtil::MakeShape(F32, {7, 3});
  builder.AddInstruction(HloInstruction::CreateReshape(result_shape, iota));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Add(
                  m::Iota(),
                  m::Multiply(m::Iota(), m::Broadcast(m::ConstantScalar())))));
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), result_shape));
}
TEST_F(AlgebraicSimplifierTest, IotaAndReshapeToMixedRadixExtraDims) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(F32, {42, 24, 15}), 1));
  Shape result_shape = ShapeUtil::MakeShape(F32, {3, 14, 4, 3, 2, 5, 3});
  builder.AddInstruction(HloInstruction::CreateReshape(result_shape, iota));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  EXPECT_THAT(
      computation->root_instruction(),
      GmockMatch(m::Add(
          m::Add(m::Iota(),
                 m::Multiply(m::Iota(), m::Broadcast(m::ConstantScalar()))),
          m::Multiply(m::Iota(), m::Broadcast(m::ConstantScalar())))));
  EXPECT_TRUE(
      ShapeUtil::Equal(computation->root_instruction()->shape(), result_shape));
}
TEST_F(AlgebraicSimplifierTest, IotaEffectiveScalar) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  auto iota = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(F32, {1, 1}), 0));
  auto result_shape = iota->shape();

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Iota()));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  AlgebraicSimplifier simplifier(default_options_);
  EXPECT_FALSE(simplifier.Run(m.get()).value());

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

  HloComputation* computation =
      m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

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

  HloComputation* computation =
      m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());

  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Iota()));
  const int64_t iota_dim =
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

  HloComputation* computation =
      m->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Iota())));

  AlgebraicSimplifier simplifier(default_options_);
  EXPECT_FALSE(simplifier.Run(m.get()).value());

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
  HloComputation* computation =
      module->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Op().Is(zero))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());

  EXPECT_THAT(computation->root_instruction(), param);
}

TEST_F(AlgebraicSimplifierTest, RemoveNoopSliceOfPad) {
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {2, 2}), "param"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  PaddingConfig no_padding;
  for (int i = 0; i < 2; ++i) {
    auto dimension = no_padding.add_dimensions();
    dimension->set_edge_padding_low(2);
    dimension->set_edge_padding_high(0);
    dimension->set_interior_padding(1);
  }
  auto pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(F32, {5, 5}), param, zero, no_padding));
  builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {2, 2}), pad, /*start_indices=*/{2, 2},
      /*limit_indices=*/{5, 5}, /*strides=*/{2, 2}));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Pad(m::Parameter(0), m::Op().Is(zero)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());

  EXPECT_THAT(computation->root_instruction(), param);
}

TEST_F(AlgebraicSimplifierTest, NegativePadding) {
  // Verify that a pad instruction with negative padding is replaced with a
  // pad with non-negative padding followed by a slice. Also verify that the
  // type of the operand and the result can be different.
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {10, 10}), "param"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  PaddingConfig padding;
  int64_t low_padding[2] = {-1, -2};
  int64_t high_padding[2] = {2, -3};
  for (int i = 0; i < 2; ++i) {
    auto dimension = padding.add_dimensions();
    dimension->set_edge_padding_low(low_padding[i]);
    dimension->set_edge_padding_high(high_padding[i]);
    dimension->set_interior_padding(0);
  }
  HloInstruction* pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(BF16, {11, 5}), param, zero, padding));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputationWithLayouts(builder.Build());

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

  ASSERT_TRUE(simplifier.Run(module.get()).value());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Pad(m::Parameter(0), m::Op().Is(zero)))));
  EXPECT_FALSE(
      has_negative_padding(computation->root_instruction()->operand(0)));
}

TEST_F(AlgebraicSimplifierTest, BroadcastSinking) {
  constexpr absl::string_view kModuleStr = R"(
      HloModule m

      main {
        p0 = u32[2]{0} parameter(0)
        p1 = u32[2]{0} parameter(1)
        b0 = u32[1024,1,2]{2,1,0} broadcast(p0), dimensions={2}
        b1 = u32[1024,1,2]{2,1,0} broadcast(p1), dimensions={2}
        ROOT o = u32[1024,1,2]{2,1,0} or(b0, b1)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  AlgebraicSimplifier simplifier(default_options_);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, simplifier.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Broadcast(op::Or(op::Parameter(0), op::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, CanDisableBroadcastSinking) {
  // Some broadcasts can be sunk (or delayed). This test verifies that we can
  // disable this behavior when necessary.
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {}), "scalar"));
  HloInstruction* broadcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::MakeShape(F32, {512, 16}), param, {}));
  builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {512, 16}), HloOpcode::kNegate, broadcast));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Negate(m::Broadcast(m::Parameter(0)))));

  // Verify that we can disable the broadcast sinking optimization.
  AlgebraicSimplifierOptions opts = default_options_;
  opts.set_enable_sink_broadcast(false);
  AlgebraicSimplifier simplifier(opts);

  // Nothing has changed since broadcast sinking is disabled.
  ASSERT_FALSE(simplifier.Run(module.get()).value());
}

TEST_F(AlgebraicSimplifierTest, CanDisableNegativePadding) {
  // Verify that a pad instruction with negative padding is replaced with a
  // pad with non-negative padding followed by a slice.
  HloComputation::Builder builder(TestName());
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {10, 10}), "param"));
  HloInstruction* zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  PaddingConfig padding;
  int64_t low_padding[2] = {-1, -2};
  int64_t high_padding[2] = {2, -3};
  for (int i = 0; i < 2; ++i) {
    auto dimension = padding.add_dimensions();
    dimension->set_edge_padding_low(low_padding[i]);
    dimension->set_edge_padding_high(high_padding[i]);
    dimension->set_interior_padding(0);
  }
  HloInstruction* pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(F32, {11, 5}), param, zero, padding));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputationWithLayouts(builder.Build());

  // Verify that we can disable the negative padding optimization.
  AlgebraicSimplifierOptions opts = default_options_;
  opts.set_enable_negative_padding_replacement(false);

  AlgebraicSimplifier simplifier(opts);

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

  // Nothing has changed since the negative padding replacement is disabled.
  ASSERT_FALSE(simplifier.Run(module.get()).value());
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
  HloComputation* computation =
      module->AddEntryComputationWithLayouts(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);

  ASSERT_THAT(computation->root_instruction(),
              GmockMatch(m::Pad(m::Parameter(0), m::Op().Is(zero))));
  ASSERT_TRUE(HasInteriorPadding(pad->padding_config()));

  EXPECT_TRUE(simplifier.Run(module.get()).value());

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
  HloComputation* computation =
      module->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Parameter(0))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());

  EXPECT_THAT(computation->root_instruction(), param);
}

TEST_F(AlgebraicSimplifierTest, RemoveNoopSlice) {
  HloComputation::Builder builder(TestName());
  const int64_t dim0 = 2;
  const int64_t dim1 = 3;
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {dim0, dim1}), "param"));
  builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {dim0, dim1}), param, /*start_indices=*/{0, 0},
      /*limit_indices=*/{dim0, dim1}, /*strides=*/{1, 1}));

  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Parameter(0))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());

  EXPECT_THAT(computation->root_instruction(), param);
}

TEST_F(AlgebraicSimplifierTest, SliceOfSliceToSlice) {
  HloComputation::Builder builder(TestName());
  const int64_t dim0 = 11;
  const int64_t dim1 = 12;
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
  HloComputation* computation =
      module->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Slice(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Parameter(0))));
  EXPECT_EQ(computation->root_instruction()->slice_starts(0), 3);
  EXPECT_EQ(computation->root_instruction()->slice_starts(1), 5);
  EXPECT_EQ(computation->root_instruction()->slice_limits(0), dim0 - 2);
  EXPECT_EQ(computation->root_instruction()->slice_limits(1), dim1 - 4);
}

TEST_F(AlgebraicSimplifierTest, SliceOfBroadcastToBroadcast) {
  HloComputation::Builder builder(TestName());
  const int64_t dim0 = 11;
  const int64_t dim1 = 12;
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
  HloComputation* computation =
      module->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Broadcast(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, SliceOfReshapeToReshapeOfSlice) {
  HloComputation::Builder builder(TestName());
  const int64_t dim0 = 11;
  const int64_t dim1 = 12;
  const int64_t dim2 = 13;
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
  HloComputation* computation =
      module->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Reshape(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());

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
  HloComputation* computation =
      module->AddEntryComputationWithLayouts(builder.Build());

  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Reshape(m::Parameter(0)))));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(module.get()).value());
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
  HloComputation* computation =
      module->AddEntryComputationWithLayouts(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
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
  HloComputation* computation =
      module->AddEntryComputationWithLayouts(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAnd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAnd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAnd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAnd);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kOr);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kOr);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kOr);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kOr);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  std::string ToString() const {
    return absl::StrFormat(
        "padding=%s, orig_conv_window=%s, expected_conv_window=%s, "
        "pad_value=%f",
        padding, orig_conv_window, expected_conv_window, pad_value);
  }

  std::string padding;
  std::string orig_conv_window;
  std::string expected_conv_window;
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

  PaddingConfig padding_config = ParsePaddingConfig(testcase.padding).value();
  auto* lhs_pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeInference::InferPadShape(input->shape(), pad_value->shape(),
                                    padding_config)
          .value(),
      input, pad_value, padding_config));

  auto* filter = builder.AddInstruction(HloInstruction::CreateParameter(
      1,
      ShapeUtil::MakeShape(
          F32, {lhs_pad->shape().dimensions(1), 256, 3, 3}),  // io01
      "input"));

  ConvolutionDimensionNumbers dnums =
      ParseConvolutionDimensionNumbers("bf01_io01->bf01").value();
  Window window =
      ParseWindow(absl::StrCat("size=3x3 ", testcase.orig_conv_window)).value();
  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeInference::InferConvolveShape(
          lhs_pad->shape(), filter->shape(),
          /*feature_group_count=*/1,
          /*batch_group_count=*/1, window, dnums,
          /*preferred_element_type=*/std::nullopt)
          .value(),
      lhs_pad, filter, /*feature_group_count=*/1, /*batch_group_count=*/1,
      window, dnums, DefaultPrecisionConfig(2)));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputationWithLayouts(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  if (testcase.expected_conv_window.empty()) {
    ASSERT_FALSE(simplifier.Run(module.get()).value());
  } else {
    ASSERT_TRUE(simplifier.Run(module.get()).value());
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
  PaddingConfig padding_config = ParsePaddingConfig(testcase.padding).value();
  auto* rhs_pad = builder.AddInstruction(HloInstruction::CreatePad(
      ShapeInference::InferPadShape(filter->shape(), pad_value->shape(),
                                    padding_config)
          .value(),
      filter, pad_value, padding_config));

  auto* input = builder.AddInstruction(HloInstruction::CreateParameter(
      0,
      ShapeUtil::MakeShape(
          F32, {1024, rhs_pad->shape().dimensions(0), 100, 100}),  // bf01
      "input"));

  ConvolutionDimensionNumbers dnums =
      ParseConvolutionDimensionNumbers("bf01_io01->bf01").value();
  Window window = ParseWindow(absl::StrFormat("size=%dx%d %s",
                                              rhs_pad->shape().dimensions(2),
                                              rhs_pad->shape().dimensions(3),
                                              testcase.orig_conv_window))
                      .value();

  // Add a PrecisionConfig and check that AlgebraicSimplifier keeps it in place
  // after the transformation.
  PrecisionConfig precision_config;
  precision_config.add_operand_precision(PrecisionConfig::HIGH);
  precision_config.add_operand_precision(PrecisionConfig::HIGHEST);

  builder.AddInstruction(HloInstruction::CreateConvolve(
      ShapeInference::InferConvolveShape(
          input->shape(), rhs_pad->shape(),
          /*feature_group_count=*/1,
          /*batch_group_count=*/1, window, dnums,
          /*preferred_element_type=*/std::nullopt)
          .value(),
      input, rhs_pad, /*feature_group_count=*/1, /*batch_group_count=*/1,
      window, dnums, precision_config));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputationWithLayouts(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  if (testcase.expected_conv_window.empty()) {
    ASSERT_FALSE(simplifier.Run(module.get()).value());
  } else {
    ASSERT_TRUE(simplifier.Run(module.get()).value());
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
  auto build_and_simplify = [&]() -> std::string {
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
    std::vector<int64_t> in_dims;
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

    std::vector<int64_t> f_dims;
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

    auto make_shape = [](absl::Span<const int64_t> dims,
                         bool minor_to_major_layout) {
      if (minor_to_major_layout) {
        return ShapeUtil::MakeShapeWithDenseLayout(F32, dims, {0, 1, 2, 3});
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
                          /*batch_group_count=*/1, window, dnums,
                          /*preferred_element_type=*/std::nullopt)
                          .value();
    if (options.output_minor_to_major_layout) {
      out_shape = ShapeUtil::MakeShapeWithDenseLayout(
          F32, out_shape.dimensions(), {0, 1, 2, 3});
    }

    b.AddInstruction(HloInstruction::CreateConvolve(
        out_shape, input, filter,
        /*feature_group_count=*/1, /*batch_group_count=*/1, window, dnums,
        DefaultPrecisionConfig(2)));

    auto module = CreateNewVerifiedModule();
    auto* computation = module->AddEntryComputationWithLayouts(b.Build());

    AlgebraicSimplifierOptions simplifier_options;
    simplifier_options.set_is_layout_sensitive(true);
    AlgebraicSimplifier simplifier(simplifier_options);
    if (!simplifier.Run(module.get()).value()) {
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

struct ConvTestOptions {
  int input_width = 4;
  int input_height = 2;
  int input_features = 10;
  int input_batch = 1;
  int kernel_width = 1;
  int kernel_height = 1;
  int kernel_input_channels = 1;
  int kernel_output_channels = 10;
  int feature_group_count = 10;
  absl::string_view input_order = "bf01";
  absl::string_view kernel_order = "io01";
  absl::string_view output_order = "bf01";
  PrimitiveType input_type = F32;
  PrimitiveType kernel_type = F32;
  PrimitiveType output_type = F32;
  struct WindowOptions {
    int col_stride = 1;
    int col_padding = 0;
    int col_dilation = 1;
    int row_stride = 1;
    int row_padding = 0;
    int row_dilation = 1;
  } window_options;

  std::unique_ptr<HloComputation> Build(const std::string& test_name) const {
    auto create_dimension = [](int size, int stride, int padding,
                               int dilation) {
      WindowDimension dim;
      dim.set_size(size);
      dim.set_stride(stride);
      dim.set_padding_low(padding);
      dim.set_padding_high(padding);
      dim.set_window_dilation(dilation);
      dim.set_base_dilation(dilation);
      return dim;
    };
    auto vertical = create_dimension(kernel_height, window_options.row_stride,
                                     window_options.row_padding,
                                     window_options.row_dilation);
    auto horizontal = create_dimension(kernel_width, window_options.col_stride,
                                       window_options.col_padding,
                                       window_options.col_dilation);
    bool width_first = kernel_order.find('0') < kernel_order.find('1');
    Window window;
    *window.add_dimensions() = width_first ? horizontal : vertical;
    *window.add_dimensions() = width_first ? vertical : horizontal;

    auto create_shape = [this](PrimitiveType type, absl::string_view order,
                               int width, int height) {
      DimensionVector dim;
      for (char ch : order) {
        if (ch == '0') {
          dim.push_back(width);
        } else if (ch == '1') {
          dim.push_back(height);
        } else if (ch == 'f') {
          dim.push_back(input_features);
        } else if (ch == 'b') {
          dim.push_back(input_batch);
        } else if (ch == 'i') {
          dim.push_back(kernel_input_channels);
        } else if (ch == 'o') {
          dim.push_back(kernel_output_channels);
        } else {
          LOG(FATAL) << "Incorrect order value: " << ch;
        }
      }
      return ShapeUtil::MakeShape(type, dim);
    };
    Shape input_shape =
        create_shape(input_type, input_order, input_width, input_height);
    Shape kernel_shape =
        create_shape(kernel_type, kernel_order, kernel_width, kernel_height);

    ConvolutionDimensionNumbers dnums =
        ParseConvolutionDimensionNumbers(
            absl::StrCat(input_order, "_", kernel_order, "->", output_order))
            .value();
    Shape inferred_shape =
        ShapeInference::InferConvolveShape(
            input_shape, kernel_shape, feature_group_count,
            /*batch_group_count=*/1, window, dnums, output_type)
            .value();

    HloComputation::Builder b(test_name);
    HloInstruction* input = b.AddInstruction(
        HloInstruction::CreateParameter(0, input_shape, "input"));
    HloInstruction* kernel = b.AddInstruction(
        HloInstruction::CreateParameter(1, kernel_shape, "kernel"));
    b.AddInstruction(HloInstruction::CreateConvolve(
        inferred_shape, input, kernel, feature_group_count,
        /*batch_group_count=*/1, window, dnums,
        HloHardwareIndependentTestBase::DefaultPrecisionConfig(2)));
    return b.Build();
  }
};

class ConvTestBase : public HloHardwareIndependentTestBase {
 public:
  std::unique_ptr<HloModule> Simplify(ConvTestOptions options) {
    auto module = CreateNewVerifiedModule();
    module->AddEntryComputationWithLayouts(options.Build(TestName()));
    AlgebraicSimplifierOptions simplifier_options;
    AlgebraicSimplifier simplifier{simplifier_options};
    bool result = simplifier.Run(module.get()).value();
    return result ? std::move(module) : nullptr;
  }

  template <typename T>
  void ExpectSimplifyMatch(ConvTestOptions options, const T& match) {
    auto module = Simplify(std::move(options));
    ASSERT_NE(module.get(), nullptr);
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                GmockMatch(match));
  }

  void ExpectSimplifyNoMatch(ConvTestOptions options) {
    auto module = Simplify(std::move(options));
    EXPECT_EQ(module.get(), nullptr);
  }
};

TEST_F(ConvTestBase, DefaultOptions) {
  ExpectSimplifyMatch(
      ConvTestOptions{},
      m::Multiply(m::Parameter(0), m::Broadcast(m::Reshape(m::Parameter(1)))));
}

TEST_F(ConvTestBase, FeatureDimension) {
  ExpectSimplifyNoMatch(
      ConvTestOptions{.kernel_input_channels = 2, .feature_group_count = 5});
}

TEST_F(ConvTestBase, ContractedDimension) {
  ExpectSimplifyMatch(
      ConvTestOptions{.kernel_height = 2},
      m::Reshape(m::Reduce(m::Multiply(m::Parameter(0), m::Broadcast()),
                           m::Convert(m::Constant()))));
}

TEST_F(ConvTestBase, InputConvert) {
  ExpectSimplifyMatch(ConvTestOptions{.input_type = F16},
                      m::Multiply(m::Convert(m::Parameter(0)),
                                  m::Broadcast(m::Reshape(m::Parameter(1)))));
}

TEST_F(ConvTestBase, KernelConvert) {
  ExpectSimplifyMatch(
      ConvTestOptions{.kernel_type = F16},
      m::Multiply(m::Parameter(0),
                  m::Broadcast(m::Reshape(m::Convert(m::Parameter(1))))));
}

TEST_F(ConvTestBase, OutputConvert) {
  ExpectSimplifyMatch(
      ConvTestOptions{.output_type = F16},
      m::Multiply(m::Convert(m::Parameter(0)),
                  m::Broadcast(m::Reshape(m::Convert(m::Parameter(1))))));
}

class ConvTestTranspose : public ConvTestBase,
                          public ::testing::WithParamInterface<const char*> {};

TEST_P(ConvTestTranspose, InputLayout) {
  ExpectSimplifyMatch(ConvTestOptions{.input_order = GetParam()},
                      m::Multiply(m::Transpose(m::Parameter(0)),
                                  m::Broadcast(m::Reshape(m::Parameter(1)))));
}

TEST_P(ConvTestTranspose, KernelLayout) {
  ExpectSimplifyMatch(
      ConvTestOptions{.kernel_order = absl::StrReplaceAll(
                          GetParam(), {{"b", "i"}, {"f", "o"}})},
      m::Multiply(m::Parameter(0),
                  m::Broadcast(m::Reshape(m::Transpose(m::Parameter(1))))));
}

TEST_P(ConvTestTranspose, OutputLayout) {
  ExpectSimplifyMatch(
      ConvTestOptions{.output_order = GetParam()},
      m::Multiply(m::Transpose(m::Parameter(0)),
                  m::Broadcast(m::Reshape(m::Transpose(m::Parameter(1))))));
}

INSTANTIATE_TEST_SUITE_P(
    ConvTestSuite, ConvTestTranspose,
    ::testing::ValuesIn({"10fb", "10bf", "1fb0", "1f0b", "1bf0", "1b0f",
                         "01fb", "01bf", "0fb1", "0f1b", "0bf1", "0b1f",
                         "fb01", "fb10", "f01b", "f0b1", "f10b", "f1b0",
                         "b01f", "b0f1", "b10f", "b1f0", "bf10"}));

class ConvTestWindow : public ConvTestBase,
                       public ::testing::WithParamInterface<ConvTestOptions> {};

TEST_P(ConvTestWindow, NoMatch) { ExpectSimplifyNoMatch(GetParam()); }

INSTANTIATE_TEST_SUITE_P(
    ConvTestSuite, ConvTestWindow,
    ::testing::ValuesIn({
        ConvTestOptions{.kernel_width = 3},
        ConvTestOptions{.kernel_height = 3},
        ConvTestOptions{.window_options = {.col_stride = 2}},
        ConvTestOptions{.window_options = {.col_padding = 1}},
        ConvTestOptions{.window_options = {.col_dilation = 2}},
        ConvTestOptions{.window_options = {.row_stride = 2}},
        ConvTestOptions{.window_options = {.row_padding = 1}},
        ConvTestOptions{.window_options = {.row_dilation = 2}},
    }));

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
  auto computation = module->AddEntryComputationWithLayouts(builder.Build());

  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, slice);
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), slice_shape));

  AlgebraicSimplifier simplifier(default_options_);

  ASSERT_TRUE(simplifier.Run(module.get()).value());

  // Running simplification again should not result in any further changes.
  ASSERT_FALSE(simplifier.Run(module.get()).value());
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
  auto computation = module->AddEntryComputationWithLayouts(builder.Build());

  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root, reshape);
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), reshape_shape));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Broadcast(m::Op().Is(forty_two))
                             .WithShapeEqualTo(&reshape_shape)));
}

// Test that a depth-to-space transformation expressed as
// reshape(transpose(reshape(op))) can simplify to
// reshape(concat(slice(op), ..., slice(op))).
TEST_F(AlgebraicSimplifierTest, TransposeReshapeToConcatSlice) {
  const std::string& hlo_string = R"(
HloModule TransposeReshapeDepthToSpace

ENTRY entry {
  %param = f32[8,14,14,128] parameter(0)
  %reshape.1 = f32[8,14,14,2,64] reshape(%param)
  %transpose = transpose(%reshape.1), dimensions={0,1,3,2,4}
  ROOT %reshape.2 = f32[8,28,14,64] reshape(%transpose)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());

  Shape result_shape = ShapeUtil::MakeShape(F32, {8, 28, 14, 64});
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(m::Concatenate(m::Slice(m::Parameter(0)),
                                                   m::Slice(m::Parameter(0))))
                             .WithShapeEqualTo(&result_shape)));
}

// Test that the transformation above doesn't happen when disabled.
TEST_F(AlgebraicSimplifierTest, DisabledTransposeReshapeToConcatSlice) {
  const std::string& hlo_string = R"(
HloModule TransposeReshapeDepthToSpace

ENTRY entry {
  %param = f32[8,14,14,128] parameter(0)
  %reshape.1 = f32[8,14,14,2,64] reshape(%param)
  %transpose = transpose(%reshape.1), dimensions={0,1,3,2,4}
  ROOT %reshape.2 = f32[8,28,14,64] reshape(%transpose)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options = default_options_;
  options.set_rewrite_reshape_transpose_as_slice_concatenate(false);
  AlgebraicSimplifier simplifier(options);
  ASSERT_FALSE(simplifier.Run(module.get()).value());

  Shape result_shape = ShapeUtil::MakeShape(F32, {8, 28, 14, 64});
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Reshape(m::Transpose(m::Reshape(m::Parameter(0))))));
}

// Test that a depth-to-space transformation expressed as
// reshape(transpose(reshape(op))) with a large number of chunks
// is not rewritten.
TEST_F(AlgebraicSimplifierTest, TransposeReshapeTooLarge) {
  const std::string& hlo_string = R"(
HloModule TransposeReshapeDepthToSpaceBig

ENTRY entry {
  %param = f32[8,14,14,128] parameter(0)
  %reshape.1 = f32[8,14,14,8,16] reshape(%param)
  %transpose = transpose(%reshape.1), dimensions={0,1,3,2,4}
  ROOT %reshape.2 = f32[8,112,14,16] reshape(%transpose)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(module.get()).value());
}

// Test that a reshape(transpose(reshape(op))) that does not constitute a
// depth-to-space transformation is not rewritten.
TEST_F(AlgebraicSimplifierTest, TransposeReshapeNotDepthToSpace) {
  const std::string& hlo_string = R"(
HloModule TransposeReshapeDepthToSpace

ENTRY entry {
  %param = f32[8,14,14,128] parameter(0)
  %reshape.1 = f32[8,14,14,2,64] reshape(%param)
  %transpose = transpose(%reshape.1), dimensions={0,3,1,2,4}
  ROOT %reshape.2 = f32[8,28,14,64] reshape(%transpose)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(module.get()).value());
}

// Test that ReduceWindow(Pad(op, x), y) can simplify to ReduceWindow(op, x).
TEST_F(AlgebraicSimplifierTest, FoldPadIntoReduceWindow) {
  const std::string& hlo_string = R"(
HloModule test
fn {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}
ENTRY entry {
  param = f32[1,2,3,4] parameter(0)
  const = f32[] constant(5)
  pad = pad(param, const), padding=0_0x1_0x0_0x0_2
  ROOT r = reduce-window(pad, const), to_apply=fn, window={size=2x2x2x2 lhs_dilate=1x1x1x3 pad=10_100x10_100x10_100x10_100}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(RunHloPass(&simplifier, module.get()).value());
  // Running simplification again should not result in any further changes.
  ASSERT_FALSE(RunHloPass(&simplifier, module.get()).value());

  // Verify the result
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              GmockMatch(m::ReduceWindow(m::Parameter(0), m::Constant())));
  EXPECT_EQ(root->window().dimensions(0).padding_low(), 10);
  EXPECT_EQ(root->window().dimensions(1).padding_low(), 11);
  EXPECT_EQ(root->window().dimensions(2).padding_low(), 10);
  EXPECT_EQ(root->window().dimensions(3).padding_low(), 10);
  EXPECT_EQ(root->window().dimensions(0).padding_high(), 100);
  EXPECT_EQ(root->window().dimensions(1).padding_high(), 100);
  EXPECT_EQ(root->window().dimensions(2).padding_high(), 100);
  EXPECT_EQ(root->window().dimensions(3).padding_high(), 106);
}

// Test that ReduceWindow(Convert(Pad(op, x)), y) can simplify to
// ReduceWindow(Convert(op), x).
TEST_F(AlgebraicSimplifierTest, FoldConvertedPadIntoReduceWindow) {
  const std::string& hlo_string = R"(
HloModule test
fn {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}
ENTRY entry {
  param = bf16[1,2,3,4] parameter(0)
  const = bf16[] constant(5)
  pad = pad(param, const), padding=0_0x1_0x0_0x0_2
  converted = f32[1,3,3,6] convert(pad)
  ROOT r = reduce-window(converted, const), to_apply=fn, window={size=2x2x2x2 pad=10_100x10_100x10_100x10_100}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(RunHloPass(&simplifier, module.get()).value());
  // Running simplification again should not result in any further changes.
  ASSERT_FALSE(RunHloPass(&simplifier, module.get()).value());

  // Verify the result
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::ReduceWindow(m::Convert(m::Parameter(0)),
                                               m::Constant())));
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
  auto computation = module->AddEntryComputationWithLayouts(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());

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
  m->AddEntryComputationWithLayouts(call_builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Parameter()));
}

TEST_F(AlgebraicSimplifierTest, ConstantDynamicSlice) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape shape = ShapeUtil::MakeShape(F32, {10, 100, 1000});
  std::vector<HloInstruction*> params;
  for (int i = 0; i < 3; ++i) {
    params.push_back(builder.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32_t>(2 << (i + 1)))));
  }
  Shape ds_shape = ShapeUtil::MakeShape(F32, {2, 20, 200});
  builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      ds_shape,
      builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape, "operand")),
      params,
      /*slice_sizes=*/{2, 20, 200}));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Slice(m::Parameter())));
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::DynamicSlice(m::Parameter(), m::Constant(),
                                         m::Parameter(), m::Constant())));
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
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
      dot = f32[5,6] dot(lhs,rhs), lhs_contracting_dims={0,1}, rhs_contracting_dims={1,2}, operand_precision={highest,high}
      ROOT transpose = f32[6,5] transpose(dot), dimensions={1,0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  const HloInstruction* dot;
  ASSERT_THAT(root, GmockMatch(m::Dot(&dot, m::Parameter(1), m::Parameter(0))));
  EXPECT_EQ(dot->precision_config().operand_precision()[0],
            PrecisionConfig::HIGH);
  EXPECT_EQ(dot->precision_config().operand_precision()[1],
            PrecisionConfig::HIGHEST);
}

TEST_F(AlgebraicSimplifierTest, DotAssociativeReorder) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
        a = f32[2,3,4,5] parameter(0)
        b = f32[6,7,5] parameter(1)
        c = f32[4,7] parameter(2)
        inner_dot = f32[2,3,4,6,7] dot(a,b),
                    lhs_contracting_dims={3},
                    rhs_contracting_dims={2}
        ROOT outer_dot = f32[2,3,6] dot(inner_dot,c),
                         lhs_contracting_dims={2,4},
                         rhs_contracting_dims={0,1}
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_use_associative_reordering(true);
  options.set_associative_reordering_threshold(1.5);
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Parameter(0),
                                m::Dot(m::Parameter(1), m::Parameter(2)))));
}

TEST_F(AlgebraicSimplifierTest, DotLeftDotSharedBatchReorder) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
        a = f32[5,150,5] parameter(0)
        b = f32[5,5,5] parameter(1)
        c = f32[5,5,5] parameter(2)

        inner = f32[5,150,5] dot(a,b),
                lhs_batch_dims={0}, lhs_contracting_dims={2},
                rhs_batch_dims={0}, rhs_contracting_dims={2}
        ROOT outer = f32[5,150,5] dot(inner,c),
                lhs_batch_dims={0}, lhs_contracting_dims={2},
                rhs_batch_dims={0}, rhs_contracting_dims={2}
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_use_associative_reordering(true);
  options.set_associative_reordering_threshold(1.5);
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Parameter(0),
                                m::Dot(m::Parameter(1), m::Parameter(2)))));
}

TEST_F(AlgebraicSimplifierTest, DotRightDotSharedBatchReorder) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
        a = f32[2,3,3] parameter(0)
        b = f32[2,3,3] parameter(1)
        c = f32[2,3,16] parameter(2)

        inner = f32[2,3,16] dot(b,c),
                lhs_batch_dims={0}, lhs_contracting_dims={2},
                rhs_batch_dims={0}, rhs_contracting_dims={1}
        ROOT outer = f32[2,3,16] dot(a,inner),
                lhs_batch_dims={0}, lhs_contracting_dims={2},
                rhs_batch_dims={0}, rhs_contracting_dims={1}
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_use_associative_reordering(true);
  options.set_associative_reordering_threshold(1.5);
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Dot(m::Parameter(0), m::Parameter(1)),
                                m::Parameter(2))));
}

TEST_F(AlgebraicSimplifierTest, DotRightDotContractBatchReorder) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      a = f32[80,38,1536] parameter(0)
      b = f32[80,38,4] parameter(1)
      c = f32[80,4,1536] parameter(2)
      inner = f32[80,38,1536] dot(b, c),
              lhs_batch_dims={0},
              lhs_contracting_dims={2},
              rhs_batch_dims={0},
              rhs_contracting_dims={1}
      ROOT outer = f32[1536,1536] dot(a, inner),
                  lhs_contracting_dims={0,1},
                  rhs_contracting_dims={0,1}
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_use_associative_reordering(true);
  options.set_associative_reordering_threshold(1.5);
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(module.get()).value());
}

TEST_F(AlgebraicSimplifierTest, DotReverseLeftReorder) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
        a = f32[512,8] parameter(0)
        b = f32[8,2] parameter(1)
        reverse = f32[512,8] reverse(a), dimensions={1}
        ROOT dot = f32[512,2] dot(reverse,b),
                         lhs_contracting_dims={1},
                         rhs_contracting_dims={0}
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_use_associative_reordering(true);
  options.set_associative_reordering_threshold(1.1);
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Parameter(0), m::Reverse(m::Parameter(1)))));
}

TEST_F(AlgebraicSimplifierTest, DotReverseRightReorder) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
        a = f32[3,2048] parameter(0)
        b = f32[2048,2048] parameter(1)
        reverse = f32[2048,2048] reverse(b), dimensions={0,1}
        ROOT dot = f32[3,2048] dot(a, reverse),
                   lhs_contracting_dims={1},
                   rhs_contracting_dims={0}
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_use_associative_reordering(true);
  options.set_associative_reordering_threshold(1.1);
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Reverse(m::Parameter(0)),
                                m::Reverse(m::Parameter(1)))));
}

TEST_F(AlgebraicSimplifierTest, DotPadLeftReorder) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
        zero = f32[] constant(0.0)
        a = f32[8,5] parameter(0)
        b = f32[10,6] parameter(1)
        pad = f32[8,10] pad(a,zero), padding=0_0x0_5
        ROOT dot = f32[8,6] dot(pad,b),
                         lhs_contracting_dims={1},
                         rhs_contracting_dims={0}
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_use_associative_reordering(true);
  options.set_associative_reordering_threshold(1.1);
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Parameter(0), m::Slice(m::Parameter(1)))));
}

TEST_F(AlgebraicSimplifierTest, DotPadRightReorder) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
        zero = f32[] constant(0.0)
        a = f32[16,384] parameter(0)
        b = f32[128,128] parameter(1)
        pad = f32[384,384] pad(b,zero), padding=128_128x128_128
        ROOT dot = f32[16,384] dot(a,pad),
                         lhs_contracting_dims={1},
                         rhs_contracting_dims={0}
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_use_associative_reordering(true);
  options.set_associative_reordering_threshold(1.1);
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Slice(m::Parameter(0)),
                                m::Pad(m::Parameter(1), m::Constant()))));
}

// This pattern appears in translate_inference_bnmt_v15_vf_lite_execution_test
TEST_F(AlgebraicSimplifierTest, DotBroadcastLeftReorder) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
        a = f32[] parameter(0)
        b = f32[1024,1024] parameter(1)
        broadcast = f32[4,8,1024] broadcast(a), dimensions={}
        ROOT dot = f32[4,8,1024] dot(broadcast,b),
                         lhs_contracting_dims={2},
                         rhs_contracting_dims={0}
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_use_associative_reordering(true);
  options.set_associative_reordering_threshold(1.1);
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Broadcast(m::Parameter(0)),
                                m::Reduce(m::Parameter(1), m::Constant()))));
}

// This pattern appears in waymo_bp_omnipath_vf_lite_execution_test
TEST_F(AlgebraicSimplifierTest, DotBroadcastRightReorder) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
        a = f32[128,200,4] parameter(0)
        b = f32[128,128] parameter(1)
        broadcast = f32[128,200,128] broadcast(b), dimensions={0,2}
        ROOT dot = f32[4,128] dot(a,broadcast),
                         lhs_contracting_dims={0,1},
                         rhs_contracting_dims={0,1}
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_use_associative_reordering(true);
  options.set_associative_reordering_threshold(1.1);
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Reduce(m::Parameter(0), m::Constant()),
                                m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, ReduceDotReorder) {
  const char* hlo_string = R"(
    HloModule module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT r = f32[] add(p0, p1)
    }

    ENTRY test {
        a = f32[2,3,128,5] parameter(0)
        b = f32[3,5,256] parameter(1)
        inner_dot = f32[2,128,256] dot(a,b),
                    lhs_contracting_dims={1,3},
                    rhs_contracting_dims={0,1}
        c = f32[] constant(0)
        ROOT reduce = f32[2,256] reduce(inner_dot, c),
                      dimensions={1}, to_apply=add
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_raise_slice_and_reduce_through_dot(true);
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Dot(m::Reduce(m::Parameter(0), m::ConstantScalar(0)),
                        m::Reduce(m::Parameter(1), m::ConstantScalar(0)))));
}

TEST_F(AlgebraicSimplifierTest, SliceDotReorder) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
        a = f32[2048,2] parameter(0)
        b = f32[2,2048] parameter(1)
        dot = f32[2048,2048] dot(a,b),
              lhs_contracting_dims={1},
              rhs_contracting_dims={0}
        ROOT slice = f32[128,2048] slice(dot), slice={[0:128],[0:2048]}
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_raise_slice_and_reduce_through_dot(true);
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Slice(m::Parameter(0)), m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, SliceDotReorderWithStrides) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
        a = f32[2048,2] parameter(0)
        b = f32[2,2048] parameter(1)
        dot = f32[2048,2048] dot(a,b),
              lhs_contracting_dims={1},
              rhs_contracting_dims={0}
        ROOT slice = f32[16,256] slice(dot), slice={[0:128:8],[0:2048:8]}
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_raise_slice_and_reduce_through_dot(true);
  EXPECT_TRUE(AlgebraicSimplifier(options).Run(module.get()).value());
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Dot(m::Slice(m::Parameter(0)), m::Slice(m::Parameter(1)))));
}

TEST_F(AlgebraicSimplifierTest, TransposeOfBatchDot) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = f32[10,20,30,40]{3,2,1,0} parameter(0)
      rhs = f32[10,20,50,30]{3,2,1,0} parameter(1)
      dot = f32[10,20,40,50]{3,2,1,0} dot(lhs,rhs),
              lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
              lhs_contracting_dims={2}, rhs_contracting_dims={3},
              operand_precision={high, default}
      ROOT transpose = f32[10,20,50,40]{2,3,1,0} transpose(dot),
              dimensions={0,1,3,2}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifier simplifier(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&simplifier, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(&dot, m::Parameter(1), m::Parameter(0))));
  DotDimensionNumbers dnums = dot->dot_dimension_numbers();
  EXPECT_THAT(dnums.lhs_batch_dimensions(), ElementsAre(0, 1));
  EXPECT_THAT(dnums.rhs_batch_dimensions(), ElementsAre(0, 1));
  EXPECT_THAT(dnums.lhs_contracting_dimensions(), ElementsAre(3));
  EXPECT_THAT(dnums.rhs_contracting_dimensions(), ElementsAre(2));
  EXPECT_EQ(dot->precision_config().operand_precision()[0],
            PrecisionConfig::DEFAULT);
  EXPECT_EQ(dot->precision_config().operand_precision()[1],
            PrecisionConfig::HIGH);
}

TEST_F(AlgebraicSimplifierTest, TransposeOfBatchDimsInBatchDotCantSimplify) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = f32[10,20,30,40] parameter(0)
      rhs = f32[10,20,50,30] parameter(1)
      dot = dot(lhs,rhs), lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
                          lhs_contracting_dims={2}, rhs_contracting_dims={3}
      ROOT transpose = transpose(dot), dimensions={1,0,3,2}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto options = AlgebraicSimplifierOptions();

  options.set_supports_non_canonical_dots(false);
  AlgebraicSimplifier simplifier(options);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&simplifier, module.get()));
  EXPECT_FALSE(changed);

  options.set_supports_non_canonical_dots(true);
  AlgebraicSimplifier simplifier2(options);
  TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&simplifier2, module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(AlgebraicSimplifierTest, TransposeOfNonCanonicalBatchDotCantSimplify) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      p0 = f32[13,11,2,3] parameter(0)
      p1 = f32[13,11,3,7,5] parameter(1)
      dot1 = dot(p0, p1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
      dot2 = dot(p1, p0), rhs_batch_dims={0,1}, rhs_contracting_dims={3}, lhs_batch_dims={0,1}, lhs_contracting_dims={2}
      trans1 = transpose(dot1), dimensions={0,1,2,4,3}
      trans2 = transpose(dot2), dimensions={0,1,2,4,3}
      ROOT root = tuple(trans1, trans2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifier simplifier(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&simplifier, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AlgebraicSimplifierTest, DynamicSliceOfTranspose) {
  // This test is without layouts so we have to set the verifier to be layout
  // insensitive.
  set_verifier_layout_sensitive(false);
  set_instruction_can_change_layout_func({});

  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      param = f32[12,10,8] parameter(0)
      i0 = s32[] parameter(1)
      i1 = s32[] parameter(2)
      i2 = s32[] parameter(3)
      transpose = f32[12,8,10] transpose(param), dimensions={0,2,1}
      ROOT slice = f32[2,3,5] dynamic-slice(transpose, i0, i1, i2),
        dynamic_slice_sizes={2,3,5}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Transpose(
                        m::DynamicSlice(m::Parameter(0), m::Parameter(1),
                                        m::Parameter(3), m::Parameter(2)))));
}

TEST_F(AlgebraicSimplifierTest, DynamicSliceOfTrivialReshape) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      param = f32[12,10,1,8] parameter(0)
      i0 = s32[] parameter(1)
      i1 = s32[] parameter(2)
      i2 = s32[] parameter(3)
      z = s32[] constant(0)
      reshape = f32[1,12,10,8] reshape(param)
      ROOT slice = f32[1,2,3,5] dynamic-slice(reshape, z, i0, i1, i2),
        dynamic_slice_sizes={1,2,3,5}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(false);
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Reshape(m::DynamicSlice(
                        m::Parameter(0), m::Parameter(1), m::Parameter(2),
                        m::Constant(), m::Parameter(3)))));
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
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Constant())));
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
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Constant())));
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
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Slice(m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, SliceOfPad) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      param = f32[3,4] parameter(0)
      constant = f32[] constant(0.0)
      pad = f32[8,10] pad(f32[3,4] param, f32[] constant), padding=3_2x1_5
      ROOT slice = f32[2,3] slice(f32[8,10] pad), slice={[4:6],[2:5]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Slice(m::Parameter(0))));
  EXPECT_THAT(root->slice_starts(), ElementsAre(1, 1));
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
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Constant())));
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
  EXPECT_TRUE(simplifier.Run(module.get()).value());
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
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::ConstantScalar(-7.0))));
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
  EXPECT_TRUE(simplifier.Run(module.get()).value());
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
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Slice(m::Parameter(2))));
  EXPECT_EQ(root->slice_starts(0), 1);
  EXPECT_EQ(root->slice_limits(0), 2);
}

TEST_F(AlgebraicSimplifierTest, SliceOfReduceWindowOneReduceDim) {
  const char* hlo = R"(
    HloModule m
    Add.1 {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT r = s32[] add(p0, p1)
    }
    ENTRY test {
      p0 = s32[2,8] parameter(0)
      c0 = s32[] constant(0)
      r = s32[2,8] reduce-window(s32[2,8] p0, s32[] c0), window={size=1x8 pad=0_0x7_0}, to_apply=Add.1
      ROOT s = s32[2,1] slice(s32[2,8] r), slice={[0:2], [7:8]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  auto root = m->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      GmockMatch(m::Reshape(m::Reduce(m::Parameter(0), m::Constant())
                                .WithShape(S32, {2})
                                .WithPredicate([](const HloInstruction* instr) {
                                  return instr->dimensions() ==
                                         std::vector<int64_t>({1});
                                }))
                     .WithShape(S32, {2, 1})));
}

TEST_F(AlgebraicSimplifierTest, SliceOfReduceWindowTwoReduceDims) {
  const char* hlo = R"(
    HloModule m
    Add.1 {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT r = s32[] add(p0, p1)
    }
    ENTRY test {
      p0 = s32[3,4,2] parameter(0)
      c0 = s32[] constant(0)
      r = s32[3,4,2] reduce-window(s32[3,4,2] p0, s32[] c0), window={size=1x4x2 pad=0_0x3_0x1_0}, to_apply=Add.1
      ROOT s = s32[3,1,1] slice(s32[3,4,2] r), slice={[0:3], [3:4], [1:2]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  auto root = m->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      GmockMatch(m::Reshape(m::Reduce(m::Parameter(0), m::Constant())
                                .WithShape(S32, {3})
                                .WithPredicate([](const HloInstruction* instr) {
                                  return instr->dimensions() ==
                                         std::vector<int64_t>({1, 2});
                                }))
                     .WithShape(S32, {3, 1, 1})));
}

TEST_F(AlgebraicSimplifierTest, ConcatToBroadcast) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      p = f32[2,1,4] parameter(0)
      ROOT concat = f32[2,6,4] concatenate(p,p,p,p,p,p), dimensions={1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Reshape(m::Parameter(0)))));
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
  EXPECT_TRUE(simplifier.Run(module.get()).value());
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
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter(0)));
}

TEST_F(AlgebraicSimplifierTest, BatchDotTransposeOperands) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = f32[10,20,30,40] parameter(0)
      rhs = f32[10,20,50,30] parameter(1)
      lhs_t = transpose(lhs), dimensions={0,1,3,2}
      rhs_t = transpose(rhs), dimensions={0,1,3,2}
      ROOT root = dot(lhs_t, rhs_t),
                  lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
                  lhs_contracting_dims={3}, rhs_contracting_dims={2},
                  operand_precision={default, high}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifier simplifier(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&simplifier, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Transpose(m::Dot(&dot, m::Parameter(1), m::Parameter(0)))));
  EXPECT_EQ(dot->precision_config().operand_precision()[0],
            PrecisionConfig::HIGH);
  EXPECT_EQ(dot->precision_config().operand_precision()[1],
            PrecisionConfig::DEFAULT);
}

TEST_F(AlgebraicSimplifierTest, BatchDotTransposeBatchDims) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = f32[10,20,40,30] parameter(0)
      rhs = f32[10,20,30,50] parameter(1)
      lhs_t = transpose(lhs), dimensions={1,0,2,3}
      rhs_t = transpose(rhs), dimensions={1,0,2,3}
      ROOT root = dot(lhs_t, rhs_t),
                  lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
                  lhs_contracting_dims={3}, rhs_contracting_dims={2},
                  operand_precision={default, high}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifier simplifier(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&simplifier, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Transpose(m::Dot(&dot, m::Parameter(0), m::Parameter(1)))));
  EXPECT_EQ(dot->precision_config().operand_precision()[0],
            PrecisionConfig::DEFAULT);
  EXPECT_EQ(dot->precision_config().operand_precision()[1],
            PrecisionConfig::HIGH);
}

TEST_F(AlgebraicSimplifierTest, BatchDotTransposeBatchDimsAndOperands) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = f32[10,20,30,40] parameter(0)
      rhs = f32[10,20,50,30] parameter(1)
      lhs_t = transpose(lhs), dimensions={1,0,3,2}
      rhs_t = transpose(rhs), dimensions={1,0,3,2}
      ROOT root = dot(lhs_t, rhs_t),
                  lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
                  lhs_contracting_dims={3}, rhs_contracting_dims={2},
                  operand_precision={high, default}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifier simplifier(AlgebraicSimplifierOptions{});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&simplifier, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Transpose(m::Dot(&dot, m::Parameter(1), m::Parameter(0)))));
  EXPECT_EQ(dot->precision_config().operand_precision()[0],
            PrecisionConfig::DEFAULT);
  EXPECT_EQ(dot->precision_config().operand_precision()[1],
            PrecisionConfig::HIGH);
}

struct PadReduceWindowEffectiveBroadcastCase {
  std::vector<int64_t> input_spatials;
  std::vector<int64_t> symmetric_pad_spatials;
  std::vector<int64_t> reduce_window_spatials;
  // Whether to use `B F S0 S1` form vs `B S0 S1 F` form.
  //
  // This doesn't test any different functionality but is useful for making sure
  // kBroadcast nodes are well formed.
  bool prepend_a;
  bool should_become_broadcast;

  std::string ToTestCaseName() const {
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
  auto decorate_spatials = [&param](absl::Span<const int64_t> spatials,
                                    int64_t a, int64_t b) {
    std::vector<int64_t> result;
    if (param.prepend_a) {
      result.push_back(a);
    }
    for (int64_t s : spatials) {
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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  TF_ASSERT_OK_AND_ASSIGN(bool run_successful, simplifier.Run(m.get()));
  ASSERT_TRUE(run_successful);
  SCOPED_TRACE(m->ToString());

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
       /*reduce_window_spatials=*/{2, 2}, /*prepend_a=*/true,
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
  std::vector<int64_t> lhs_dims = {2, 3, 5};
  std::vector<int64_t> rhs_dims = lhs_dims;
  std::vector<int64_t> output_dims = lhs_dims;
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
  auto computation = module->AddEntryComputationWithLayouts(builder.Build());
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
  auto computation = module->AddEntryComputationWithLayouts(builder.Build());
  AlgebraicSimplifier simplifier(default_options_);
  // First pass of algebraic simplifier will remove degenerate dimensions
  // and optimize dot(transpose(x),transpose(y))
  TF_ASSERT_OK_AND_ASSIGN(bool changed, simplifier.Run(module.get()));
  const bool dot_should_be_transformed = m == 1 || k == 1 || n == 1;
  const bool computation_should_be_modified =
      dot_should_be_transformed || (transpose_lhs && transpose_rhs);
  EXPECT_EQ(changed, computation_should_be_modified);
  // The second pass of algebraic simplifier will remove dots without
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
  int64_t m;
  int64_t k;
  int64_t n;
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

  int64_t k0 = spec.k / 3;
  int64_t k1 = spec.k / 3;
  int64_t k2 = spec.k - k0 - k1;

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
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

  int64_t k0 = spec.k / 4;
  int64_t k1 = spec.k / 4;
  int64_t k2 = spec.k / 4;
  int64_t k3 = spec.k - k0 - k1 - k2;

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

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
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
  ASSERT_FALSE(AlgebraicSimplifier(options).Run(m.get()).value());
}

TEST_F(DotOfConcatSimplificationTest, UnnestConcatenate) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[2,10] parameter(0)
      p1 = f32[3,10] parameter(1)
      p2 = f32[4,10] parameter(2)
      c0 = f32[5,10] concatenate(p0, p1), dimensions={0}
      ROOT c1 = f32[9,10] concatenate(c0, p2), dimensions={0}
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifier simplifier(default_options_);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&simplifier, m.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Concatenate(m::Parameter(0), m::Parameter(1),
                                        m::Parameter(2))));
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
      m->AddEntryComputationWithLayouts(builder.Build());

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(computation->root_instruction(), operand);
}

// Test that dynamic-update-slice with a scalar broadcast becomes a pad.
TEST_F(AlgebraicSimplifierTest, DynamicUpdateSliceOfBroadcastToPad) {
  const char* hlo_string = R"(
HloModule AddBroadcastZeroWithDynamicSlice

ENTRY AddBroadcastZeroWithDynamicSlice {
  param0 = f32[1800,12,512]{2,1,0} parameter(0)
  constant = f32[] constant(0)
  broadcast = f32[1800,12,512]{2,1,0} broadcast(constant), dimensions={}
  param1 = f32[1,12,512]{2,1,0} parameter(1)
  constant.1 = s32[] constant(0)
  dynamic-update-slice = f32[1800,12,512]{2,1,0} dynamic-update-slice(broadcast, param1, constant.1, constant.1, constant.1)
  ROOT add = f32[1800,12,512]{2,1,0} add(param0, dynamic-update-slice)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  VLOG(2) << "Before rewrite dus->pad\n" << module->ToString();
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite dus->pad\n" << module->ToString();
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root->opcode(), HloOpcode::kAdd);
  EXPECT_THAT(root->operand(1)->opcode(), HloOpcode::kPad);
}

// Test that dynamic-update-slice with a scalar broadcast does not become a pad
// if the dynamic-update-slice is for host memory offload.
TEST_F(AlgebraicSimplifierTest, DynamicUpdateSliceOfBroadcastToPadHostOffload) {
  const std::string hlo_string = absl::StrFormat(
      R"(
HloModule DynamicUpdateSliceOfBroadcastToPadHostOffload

ENTRY DynamicUpdateSliceOfBroadcastToPadHostOffload {
  constant_bf16_0 = bf16[] constant(0)
  broadcast_0 = bf16[56,2,2048,2,128] broadcast(constant_bf16_0), dimensions={}
  param_0 = bf16[1,2,2048,2,128] parameter(0)
  custom_call = bf16[1,2,2048,2,128] custom-call(param_0), custom_call_target="%s"
  constant_s32_0 = s32[] constant(0)
  ROOT dynamic_update_slice = bf16[56,2,2048,2,128] dynamic-update-slice(broadcast_0, custom_call, constant_s32_0, constant_s32_0, constant_s32_0, constant_s32_0, constant_s32_0)
}
)",
      memory_annotations::kMoveToHostCustomCallTarget);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  VLOG(2) << "Before rewrite dus->pad\n" << module->ToString();
  AlgebraicSimplifier simplifier(default_options_);
  EXPECT_FALSE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite dus->pad\n" << module->ToString();
  // Look for the following pattern:
  // constant(0)   param(0)
  //      |           |
  // broadcast   custom-call  constant(0)
  //      |           |      /
  //      |           |     /
  //      |           |    /
  //      |           |   /
  //      |           |  /
  // dynamic-update-slice
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::DynamicUpdateSlice(
          m::Broadcast(m::ConstantScalar(0)),
          m::CustomCall({memory_annotations::kMoveToHostCustomCallTarget},
                        m::Parameter(0)),
          m::ConstantScalar(0), m::ConstantScalar(0), m::ConstantScalar(0),
          m::ConstantScalar(0), m::ConstantScalar(0))));
}

// Test that dynamic-update-slice with a scalar broadcast does not become a pad
// if the dynamic-update-slice is for host memory offload. Also disable
// optimization if there is a reshape between the custom-call and the
// dynamic-update-slice.
TEST_F(AlgebraicSimplifierTest,
       DynamicUpdateSliceOfBroadcastToPadHostOffloadWithReshape) {
  const std::string hlo_string = absl::StrFormat(
      R"(
HloModule DynamicUpdateSliceOfBroadcastToPadHostOffloadWithReshape

ENTRY DynamicUpdateSliceOfBroadcastToPadHostOffloadWithReshape {
  constant_bf16_0 = bf16[] constant(0)
  broadcast_0 = bf16[56,2,2048,2,128] broadcast(constant_bf16_0), dimensions={}
  param_0 = bf16[2,2048,2,128] parameter(0)
  custom_call = bf16[2,2048,2,128] custom-call(param_0), custom_call_target="%s"
  reshape = bf16[1,2,2048,2,128] reshape(custom_call)
  constant_s32_0 = s32[] constant(0)
  ROOT dynamic_update_slice = bf16[56,2,2048,2,128] dynamic-update-slice(broadcast_0, reshape, constant_s32_0, constant_s32_0, constant_s32_0, constant_s32_0, constant_s32_0)
}
)",
      memory_annotations::kMoveToHostCustomCallTarget);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  VLOG(2) << "Before rewrite dus->pad\n" << module->ToString();
  AlgebraicSimplifier simplifier(default_options_);
  EXPECT_FALSE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite dus->pad\n" << module->ToString();
  // Look for the following pattern:
  //              param(0)
  //                  |
  // constant(0)  custom-call
  //      |           |
  // broadcast    reshape   constant(0)
  //      |           |     /
  //      |           |    /
  //      |           |   /
  //      |           |  /
  // dynamic-update-slice
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::DynamicUpdateSlice(
          m::Broadcast(m::ConstantScalar(0)),
          m::Reshape(
              m::CustomCall({memory_annotations::kMoveToHostCustomCallTarget},
                            m::Parameter(0))),
          m::ConstantScalar(0), m::ConstantScalar(0), m::ConstantScalar(0),
          m::ConstantScalar(0), m::ConstantScalar(0))));
}

// Test that dynamic-update-slice with a scalar broadcast does not become a pad
// if the dynamic-update-slice is for host memory offload. Also disable
// optimization if there is a bitcast between the custom-call and the
// dynamic-update-slice.
TEST_F(AlgebraicSimplifierTest,
       DynamicUpdateSliceOfBroadcastToPadHostOffloadWithBitcast) {
  const std::string hlo_string = absl::StrFormat(
      R"(
HloModule DynamicUpdateSliceOfBroadcastToPadHostOffloadWithBitcast

ENTRY DynamicUpdateSliceOfBroadcastToPadHostOffloadWithBitcast {
  constant_bf16_0 = bf16[] constant(0)
  broadcast_0 = bf16[56,2,2048,2,128] broadcast(constant_bf16_0), dimensions={}
  param_0 = bf16[2,2048,2,128] parameter(0)
  custom_call = bf16[2,2048,2,128] custom-call(param_0), custom_call_target="%s"
  bitcast = bf16[1,2,2048,2,128] bitcast(custom_call)
  constant_s32_0 = s32[] constant(0)
  ROOT dynamic_update_slice = bf16[56,2,2048,2,128] dynamic-update-slice(broadcast_0, bitcast, constant_s32_0, constant_s32_0, constant_s32_0, constant_s32_0, constant_s32_0)
}
)",
      memory_annotations::kMoveToHostCustomCallTarget);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  VLOG(2) << "Before rewrite dus->pad\n" << module->ToString();
  AlgebraicSimplifier simplifier(default_options_);
  EXPECT_FALSE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite dus->pad\n" << module->ToString();
  // Look for the following pattern:
  //              param(0)
  //                  |
  // constant(0)  custom-call
  //      |           |
  // broadcast    bitcast   constant(0)
  //      |           |     /
  //      |           |    /
  //      |           |   /
  //      |           |  /
  // dynamic-update-slice
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::DynamicUpdateSlice(
          m::Broadcast(m::ConstantScalar(0)),
          m::Bitcast(
              m::CustomCall({memory_annotations::kMoveToHostCustomCallTarget},
                            m::Parameter(0))),
          m::ConstantScalar(0), m::ConstantScalar(0), m::ConstantScalar(0),
          m::ConstantScalar(0), m::ConstantScalar(0))));
}

// Test that dynamic-update-slice with a scalar broadcast does not become a pad
// if the dynamic-update-slice is for host memory offload. Also disable
// optimization if there are multiple formatting ops between the custom-call and
// the dynamic-update-slice.
TEST_F(AlgebraicSimplifierTest,
       DynamicUpdateSliceOfBroadcastToPadHostOffloadMultiLevel) {
  const std::string hlo_string = absl::StrFormat(
      R"(
HloModule DynamicUpdateSliceOfBroadcastToPadHostOffloadMultiLevel

ENTRY DynamicUpdateSliceOfBroadcastToPadHostOffloadMultiLevel {
  constant_bf16_0 = bf16[] constant(0)
  broadcast_0 = bf16[56,2,2048,2,128]{4,3,2,1,0} broadcast(constant_bf16_0), dimensions={}
  param_0 = bf16[2,2048,2,128]{1,2,3,0} parameter(0)
  custom_call = bf16[2,2048,2,128]{1,2,3,0} custom-call(param_0), custom_call_target="%s"
  copy = bf16[2,2048,2,128]{3,2,1,0} copy(custom_call)
  bitcast = bf16[1,2,2048,2,128]{4,3,2,1,0} bitcast(copy)
  constant_s32_0 = s32[] constant(0)
  ROOT dynamic_update_slice = bf16[56,2,2048,2,128]{4,3,2,1,0} dynamic-update-slice(broadcast_0, bitcast, constant_s32_0, constant_s32_0, constant_s32_0, constant_s32_0, constant_s32_0)
}
)",
      memory_annotations::kMoveToHostCustomCallTarget);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  VLOG(2) << "Before rewrite dus->pad\n" << module->ToString();
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite dus->pad\n" << module->ToString();
  // Look for the following pattern:
  //              param(0)
  //                  |
  // constant(0)  custom-call
  //      |           |
  //      |         copy
  //      |           |
  // broadcast    bitcast   constant(0)
  //      |           |     /
  //      |           |    /
  //      |           |   /
  //      |           |  /
  // dynamic-update-slice
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::DynamicUpdateSlice(
          m::Broadcast(m::ConstantScalar(0)),
          m::Bitcast(m::Copy(
              m::CustomCall({memory_annotations::kMoveToHostCustomCallTarget},
                            m::Parameter(0)))),
          m::ConstantScalar(0), m::ConstantScalar(0), m::ConstantScalar(0),
          m::ConstantScalar(0), m::ConstantScalar(0))));
}

// Test of dynamic-update-slice with dims where update and result have the same
// size so we can replace indices to 0.
TEST_F(AlgebraicSimplifierTest, DynamicUpdateSliceTrivialIndices) {
  const char* hlo_string = R"(
HloModule DynamicUpdateSliceTrivialIndices

ENTRY DynamicUpdateSliceTrivialIndices {
  param0 = f32[1800,12,512] parameter(0)
  param1 = f32[1,12,512] parameter(1)
  param2 = u32[] parameter(2)
  ROOT dynamic-update-slice = f32[1800,12,512] dynamic-update-slice(
    param0, param1, param2, param2, param2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite\n" << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicUpdateSlice(m::Parameter(0), m::Parameter(1),
                                               m::Parameter(2), m::Constant(),
                                               m::Constant())));
}

TEST_F(AlgebraicSimplifierTest, AddDynamicUpdateSliceToAddSlice) {
  const char* hlo_string = R"(
HloModule AddDynamicUpdateSliceToAddSlice

ENTRY AddDynamicUpdateSliceToAddSlice {
  param0 = f32[1,4,12,512,1,1] parameter(0)
  constant = f32[] constant(0)
  broadcast = f32[4,12,512] broadcast(constant), dimensions={}
  param1 = f32[1,12,512] parameter(1)
  param2 = s32[] parameter(2)
  constant.1 = s32[] constant(0)
  dynamic-update-slice = f32[4,12,512] dynamic-update-slice(
    broadcast, param1, param2, constant.1, constant.1)
  reshape = f32[1,4,12,512,1,1] reshape(dynamic-update-slice)
  ROOT add = f32[1,4,12,512,1,1] add(param0, reshape)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  VLOG(2) << "Before rewrite reshape\n" << module->ToString();
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite to add slice\n" << module->ToString();
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      GmockMatch(m::DynamicUpdateSlice(
          m::Parameter(0),
          m::Add(m::DynamicSlice(m::Parameter(0), m::Constant(),
                                 m::Parameter(2), m::Constant(), m::Constant(),
                                 m::Constant(), m::Constant()),
                 m::Reshape(m::Parameter(1))),
          m::Constant(), m::Parameter(2), m::Constant(), m::Constant(),
          m::Constant(), m::Constant())));
}

TEST_F(AlgebraicSimplifierTest, AddDynamicUpdateSliceToAddSliceOnLhs) {
  const char* hlo_string = R"(
HloModule AddDynamicUpdateSliceToAddSlice

ENTRY AddDynamicUpdateSliceToAddSlice {
  param0 = f32[4,12,512] parameter(0)
  constant = f32[] constant(0)
  broadcast = f32[4,12,512] broadcast(constant), dimensions={}
  param1 = f32[1,12,512] parameter(1)
  param2 = s32[] parameter(2)
  constant.1 = s32[] constant(0)
  dynamic-update-slice = f32[4,12,512] dynamic-update-slice(
    broadcast, param1, param2, constant.1, constant.1)
  ROOT add = f32[4,12,512] add(dynamic-update-slice, param0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite to add slice\n" << module->ToString();
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::DynamicUpdateSlice(
                        m::Parameter(0),
                        m::Add(m::DynamicSlice(m::Parameter(0), m::Parameter(2),
                                               m::Constant(), m::Constant()),
                               m::Parameter(1)),
                        m::Parameter(2), m::Constant(), m::Constant())));
}

TEST_F(AlgebraicSimplifierTest, ScalarMultiplyReduction) {
  const char* hlo_string = R"(
HloModule ConstScalarMultiply
ENTRY ConstScalarMultiply {
  param0 = f32[16,512,4096]{2,1,0} parameter(0)
  constant.0 = f32[] constant(0.5)
  broadcast.0 = f32[16,512,4096] broadcast(constant.0), dimensions={}
  multiply.0 = f32[16,512,4096]{2,1,0} multiply(param0, broadcast.0)
  param1 = f32[16,512,4096]{2,1,0} parameter(1)
  multiply.1 = f32[16,512,4096]{2,1,0} multiply(multiply.0, param1)
  param2 = f32[16,512,1024]{2,1,0} parameter(2)
  constant.1 = f32[] constant(1.109)
  broadcast.1 = f32[16,512,1024] broadcast(constant.1), dimensions={}
  multiply.2 = f32[16,512,1024]{2,1,0} multiply(param2, broadcast.1)
  ROOT convolution = f32[4096,1024,1]{1,0,2} convolution(multiply.1, multiply.2), window={size=16}, dim_labels=0fb_0io->bf0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifierOptions options;
  options.set_enable_scalar_multiply_reduction(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kMultiply);
  EXPECT_THAT(root,
              GmockMatch(m::MultiplyAnyOrder(
                  m::Op(), m::Broadcast(m::ConstantScalar(0.5f * 1.109f)))));
}

TEST_F(AlgebraicSimplifierTest, ScalarMultiplyReductionMultiUser) {
  const char* hlo_string = R"(
HloModule ConstScalarMultiply
ENTRY ConstScalarMultiply {
  param0 = f32[16,512,1024] parameter(0)
  param1 = f32[4096,1024,1] parameter(1)
  convolution = f32[16,512,4096] convolution(param0, param1), window={size=1}, dim_labels=0bf_oi0->0bf
  constant.1 = f32[] constant(0.5)
  broadcast.1 = f32[16,512,4096] broadcast(constant.1), dimensions={}
  multiply.1 = f32[16,512,4096] multiply(convolution, broadcast.1)
  param2 = f32[16,512,4096] parameter(2)
  multiply.2 = f32[16,512,4096] multiply(convolution, param2)
  ROOT add.1 = f32[16,512,4096] add(multiply.1, multiply.2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifierOptions options;
  options.set_enable_scalar_multiply_reduction(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_FALSE(simplifier.Run(module.get()).value());
}

INSTANTIATE_TEST_SUITE_P(DotOfConcatSimplificationTestInstantiation,
                         DotOfConcatSimplificationTest,
                         ::testing::ValuesIn(kDotOfConcatTestSpecs));

struct DotOfGatherTestSpec {
  int64_t m;
  int64_t k;
  int64_t n;
  int s;  // start index for dynamic slice on the non-contracting dimension
  int64_t lcd;  // left contracting dimension
  int64_t rcd;  // right contracting dimension
  bool neg;     // is negative testcase
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
  int64_t k_increase = spec.neg ? 5 : 0;
  int64_t lhs_rows = (spec.lcd == 0) ? (spec.k + k_increase) : spec.m;
  int64_t lhs_cols = (spec.lcd == 0) ? spec.m : (spec.k + k_increase);
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {lhs_rows, lhs_cols});
  auto* lhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/lhs_rows,
          /*cols=*/lhs_cols)));

  int32_t start_row = (spec.lcd == 0) ? 0 : spec.s;
  int32_t start_col = (spec.lcd == 0) ? spec.s : 0;
  std::vector<HloInstruction*> start_indices = {
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32_t>(start_row))),
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32_t>(start_col)))};
  int64_t slice_row_size = (spec.lcd == 0) ? spec.k : 1;
  int64_t slice_col_size = (spec.lcd == 0) ? 1 : spec.k;
  std::vector<int64_t> slice_sizes = {slice_row_size, slice_col_size};
  Shape ds_shape = ShapeUtil::MakeShape(F32, slice_sizes);
  auto* ds = builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      ds_shape, lhs, start_indices, slice_sizes));

  int64_t rhs_rows = (spec.rcd == 0) ? spec.k : spec.n;
  int64_t rhs_cols = (spec.rcd == 0) ? spec.n : spec.k;
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {rhs_rows, rhs_cols});
  auto* rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/rhs_rows,
          /*cols=*/rhs_cols)));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(spec.lcd);
  dot_dnums.add_rhs_contracting_dimensions(spec.rcd);

  int64_t dot_row_size = 1;
  int64_t dot_col_size = spec.n;
  Shape dot_shape = ShapeUtil::MakeShape(F32, {dot_row_size, dot_col_size});
  builder.AddInstruction(HloInstruction::CreateDot(
      dot_shape, ds, rhs, dot_dnums, DefaultPrecisionConfig(2)));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
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

  int64_t lhs_rows = (spec.lcd == 0) ? spec.k : spec.m;
  int64_t lhs_cols = (spec.lcd == 0) ? spec.m : spec.k;
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {lhs_rows, lhs_cols});
  auto* lhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/lhs_rows,
          /*cols=*/lhs_cols)));

  // For negative tests increase k of the dynamic slice argument to prevent the
  // optimization
  int64_t k_increase = spec.neg ? 5 : 0;
  int64_t rhs_rows = (spec.rcd == 0) ? (spec.k + k_increase) : spec.n;
  int64_t rhs_cols = (spec.rcd == 0) ? spec.n : (spec.k + k_increase);
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {rhs_rows, rhs_cols});
  auto* rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2F32Linspace(
          /*from=*/10.0, /*to=*/10000.0, /*rows=*/rhs_rows,
          /*cols=*/rhs_cols)));

  int32_t start_row = (spec.rcd == 0) ? 0 : spec.s;
  int32_t start_col = (spec.rcd == 0) ? spec.s : 0;
  std::vector<HloInstruction*> start_indices = {
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32_t>(start_row))),
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32_t>(start_col)))};
  int64_t slice_row_size = (spec.rcd == 0) ? spec.k : 1;
  int64_t slice_col_size = (spec.rcd == 0) ? 1 : spec.k;
  std::vector<int64_t> slice_sizes = {slice_row_size, slice_col_size};
  Shape ds_shape = ShapeUtil::MakeShape(F32, slice_sizes);
  auto* ds = builder.AddInstruction(HloInstruction::CreateDynamicSlice(
      ds_shape, rhs, start_indices, slice_sizes));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(spec.lcd);
  dot_dnums.add_rhs_contracting_dimensions(spec.rcd);

  int64_t dot_row_size = spec.m;
  int64_t dot_col_size = 1;
  Shape dot_shape = ShapeUtil::MakeShape(F32, {dot_row_size, dot_col_size});
  builder.AddInstruction(HloInstruction::CreateDot(
      dot_shape, lhs, ds, dot_dnums, DefaultPrecisionConfig(2)));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
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
  const std::vector<DotOfGatherTestSpec>::size_type positives_size =
      positives.size();
  all.reserve(positives_size * 2);
  for (std::vector<DotOfGatherTestSpec>::size_type i = 0; i < positives_size;
       i++) {
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

TEST_F(AlgebraicSimplifierTest, GatherOfScalarToBroadcast) {
  const char* hlo_string = R"(
  HloModule repeat

  ENTRY main {
    o = f32[1,1] parameter(0)
    i = s32[100,2] parameter(1)
    ROOT g = f32[100] gather(o, i), collapsed_slice_dims={0,1},
                                  start_index_map={0,1},
                                  index_vector_dim=1,
                                  offset_dims={},
                                  slice_sizes={1,1}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Reshape(m::Parameter(0)))));
}

TEST_F(AlgebraicSimplifierTest, GatherOfPad) {
  const char* hlo_string = R"(
HloModule module

ENTRY %entry {
  par.0 = f32[25165824,32]{1,0} parameter(0)
  constant.0 = f32[] constant(0)
  pad = f32[25165824,128]{1,0} pad(par.0, constant.0), padding=0_0x0_96
  start_indices = s32[20447232,1]{1,0} parameter(1)
  gather = f32[20447232,128]{1,0} gather(pad, start_indices),
    offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0},
    index_vector_dim=1, slice_sizes={1,128}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite \n" << module->ToString();
  auto root = module->entry_computation()->root_instruction();
  const HloInstruction* gather_instr;
  EXPECT_THAT(root, GmockMatch(m::Pad(m::Gather(&gather_instr, m::Parameter(0),
                                                m::Parameter(1)),
                                      m::ConstantScalar(0))));
  EXPECT_THAT(Cast<HloGatherInstruction>(gather_instr)->gather_slice_sizes(),
              ElementsAre(1, 32));
}

TEST_F(AlgebraicSimplifierTest, GatherOfPadWithBatchDims) {
  const char* hlo_string = R"(
HloModule module

ENTRY %entry {
  iota = s32[4,1]{1,0} iota(), iota_dimension=0
  constant.0 = s32[] constant(0)
  pad = s32[4,2]{1,0} pad(iota, constant.0), padding=0_0x0_1
  start_indices = s32[4,40,1]{2,1,0} parameter(0)
  gather = s32[4,40,2]{2,1,0} gather(pad, start_indices),
    offset_dims={2}, collapsed_slice_dims={}, start_index_map={0},
    operand_batching_dims={0}, start_indices_batching_dims={0},
    index_vector_dim=2, slice_sizes={1,2}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite \n" << module->ToString();
  auto root = module->entry_computation()->root_instruction();
  const HloInstruction* gather_instr;
  EXPECT_THAT(root, GmockMatch(m::Pad(
                        m::Gather(&gather_instr, m::Iota(), m::Parameter(0)),
                        m::ConstantScalar(0))));
  auto gather = Cast<HloGatherInstruction>(gather_instr);
  EXPECT_THAT(gather->gather_slice_sizes(), ElementsAre(1, 1));
  EXPECT_THAT(gather->gather_dimension_numbers().operand_batching_dims(),
              ElementsAre(0));
  EXPECT_THAT(gather->gather_dimension_numbers().start_indices_batching_dims(),
              ElementsAre(0));
}

TEST_F(AlgebraicSimplifierTest, GatherOfPadStartIndicesPaddedMoreDims) {
  const char* hlo_string = R"(
  HloModule module

  ENTRY %entry {
    par0 = f32[98,3,38,150,197] parameter(0)
    f0 = f32[] constant(0)
    operand = f32[100,3,40,150,200] pad(par0, f0), padding=2_0x0_0x0_2x0_0x2_1
    par1 = s32[197,98,16,150] parameter(1)
    s0 = s32[] constant(0)
    indices = s32[200,100,20,150] pad(par1, s0), padding=2_1x2_0x4_0x0_0
    ROOT gather = f32[200,40,100,20,150] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=4,
      slice_sizes={1,1,40,1,1},
      operand_batching_dims={0,3,4},
      start_indices_batching_dims={1,3,0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(module.get()).value());
}

TEST_F(AlgebraicSimplifierTest, GatherOfPadStartIndicesPaddedDifferently) {
  const char* hlo_string = R"(
  HloModule module

  ENTRY %entry {
    par0 = f32[98,3,38,150,197] parameter(0)
    f0 = f32[] constant(0)
    operand = f32[100,3,40,150,200] pad(par0, f0), padding=2_0x0_0x0_2x0_0x2_1
    par1 = s32[197,98,20,150] parameter(1)
    s0 = s32[] constant(0)
    indices = s32[200,100,20,150] pad(par1, s0), padding=1_2x2_0x0_0x0_0
    ROOT gather = f32[200,40,100,20,150] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=4,
      slice_sizes={1,1,40,1,1},
      operand_batching_dims={0,3,4},
      start_indices_batching_dims={1,3,0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(module.get()).value());
}

TEST_F(AlgebraicSimplifierTest, GatherOfPadWithPaddedBatchDims) {
  const char* hlo_string = R"(
  HloModule module

  ENTRY %entry {
    par0 = f32[98,3,38,150,197] parameter(0)
    f0 = f32[] constant(0)
    operand = f32[100,3,40,150,200] pad(par0, f0), padding=2_0x0_0x0_2x0_0x2_1
    par1 = s32[197,98,20,150] parameter(1)
    s0 = s32[] constant(0)
    indices = s32[200,100,20,150] pad(par1, s0), padding=2_1x2_0x0_0x0_0
    ROOT gather = f32[200,40,100,20,150] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=4,
      slice_sizes={1,1,40,1,1},
      operand_batching_dims={0,3,4},
      start_indices_batching_dims={1,3,0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite \n" << module->ToString();

  auto root = module->entry_computation()->root_instruction();
  const HloInstruction* pad_instr;
  const HloInstruction* gather_instr;
  EXPECT_THAT(root, GmockMatch(m::Pad(&pad_instr,
                                      m::Gather(&gather_instr, m::Parameter(0),
                                                m::Parameter(1)),
                                      m::ConstantScalar(0))));
  auto gather = Cast<HloGatherInstruction>(gather_instr);
  EXPECT_THAT(gather->gather_slice_sizes(), ElementsAre(1, 1, 38, 1, 1));
  EXPECT_THAT(gather->gather_dimension_numbers().operand_batching_dims(),
              ElementsAre(0, 3, 4));
  EXPECT_THAT(gather->gather_dimension_numbers().start_indices_batching_dims(),
              ElementsAre(1, 3, 0));
  EXPECT_EQ(PaddingConfigToString(pad_instr->padding_config()),
            "2_1x0_2x2_0x0_0x0_0");
}

TEST_F(AlgebraicSimplifierTest, GatherOfReshapeOfPad) {
  const char* hlo_string = R"(
ENTRY %entry {
  reshape.17992 = f32[64,393216,32]{2,1,0} parameter(0)
  constant.31700 = f32[] constant(0)
  pad.921 = f32[64,393216,128]{2,1,0} pad(reshape.17992, constant.31700), padding=0_0x0_0x0_96
  reshape.100261 = f32[25165824,128]{1,0} reshape(pad.921)
  reshape.40561 = s32[20447232,1]{1,0} parameter(1)
  gather.100277 = f32[20447232,128]{1,0} gather(reshape.100261, reshape.40561),
    offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0},
    index_vector_dim=1, slice_sizes={1,128}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite \n" << module->ToString();
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Pad(
                        m::Gather(m::Reshape(m::Parameter(0)), m::Parameter(1)),
                        m::ConstantScalar(0))));
}

TEST_F(AlgebraicSimplifierTest, GatherOfReshapeOfPad2) {
  const char* hlo_string = R"(
HloModule module

ENTRY %entry {
  iota.3 = s32[2,4,1]{2,1,0} iota(), iota_dimension=0
  constant.36 = s32[] constant(0)
  pad = s32[2,4,2]{2,1,0} pad(iota.3, constant.36), padding=0_0x0_0x0_1
  reshape = s32[8,2]{1,0} reshape(pad)
  reshape.300 = s32[3,40,1]{2,1,0} parameter(0)
  gather.363 = s32[3,40,2]{2,1,0} gather(reshape, reshape.300),
    offset_dims={2}, collapsed_slice_dims={0}, start_index_map={0},
    index_vector_dim=2, slice_sizes={1,2}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite \n" << module->ToString();
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, GmockMatch(m::Pad(m::Gather(m::Reshape(m::Iota()), m::Parameter(0)),
                              m::ConstantScalar(0))));
}

TEST_F(AlgebraicSimplifierTest, GatherOfReshapeOfPad3) {
  const char* hlo_string = R"(
HloModule module

ENTRY %entry {
  parameter.0 = f32[2,4256]{1,0} parameter(0)
  constant = f32[] constant(0)
  pad.264 = f32[2,4480]{1,0} pad(parameter.0, constant), padding=0_0x0_224
  slice.267 = f32[2,4480]{1,0} slice(pad.264), slice={[0:2], [0:4480]}
  reshape.269 = f32[2,28,160]{2,1,0} reshape(slice.267)
  parameter.1 = s32[27,2]{1,0} parameter(1)
  ROOT gather.271 = f32[2,27,2,160]{3,2,1,0} gather(reshape.269, parameter.1), offset_dims={0,3}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=2, slice_sizes={2,1,160}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  VLOG(2) << "After rewrite \n" << module->ToString();
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      GmockMatch(m::Gather(
          m::Reshape(m::Slice(m::Pad(m::Parameter(0), m::ConstantScalar(0)))),
          m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, GatherOfReshapeOfPad4) {
  const char* hlo_string = R"(
HloModule module

ENTRY %entry {
  dot.165 = bf16[2048,8192]{1,0} parameter(0)
  constant.16 = bf16[] constant(0)
  reshape.60 = s32[16,1]{1,0} parameter(1)
  pad.6 = bf16[4096,8192]{1,0} pad(
    bf16[2048,8192]{1,0} %dot.165, bf16[] %constant.16), padding=0_2048x0_0
  reshape.170 = bf16[4096,16,512]{2,1,0} reshape(bf16[4096,8192]{1,0} %pad.6)
  gather.175 = bf16[4096,16,512]{2,1,0} gather(
    bf16[4096,16,512]{2,1,0} %reshape.170, s32[16,1]{1,0} %reshape.60),
    offset_dims={0,2}, collapsed_slice_dims={1}, start_index_map={1},
    index_vector_dim=1, slice_sizes={4096,1,512}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  VLOG(0) << "After rewrite \n" << module->ToString();
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Pad(m::Gather(m::Reshape(), m::Parameter(1)),
                                      m::ConstantScalar(0))));
  EXPECT_EQ(root->padding_config().dimensions(0).edge_padding_high(), 2048);
  EXPECT_EQ(root->padding_config().dimensions(1).edge_padding_high(), 0);
  EXPECT_EQ(root->padding_config().dimensions(2).edge_padding_high(), 0);
}

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
  EXPECT_TRUE(simplifier.Run(module.get()).value());
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
  EXPECT_TRUE(simplifier.Run(module.get()).value());
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
  EXPECT_TRUE(simplifier.Run(module.get()).value());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Constant()));
}

TEST_F(AlgebraicSimplifierTest, DividedByConstantInstructionWithoutLayout) {
  // This test is without layouts so we have to set the verifier to be layout
  // insensitive.
  set_verifier_layout_sensitive(false);
  set_instruction_can_change_layout_func({});

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
  module->AddEntryComputationWithLayouts(builder.Build());

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(module.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(options).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_FALSE(AlgebraicSimplifier(options).Run(m.get()).value());
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
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, CompareGtMaxA) {
  // Gt(Max(a,b), a) -> Gt(b,a)
  const char* kModuleStr = R"(
    HloModule m
    test {
      a = f32[4] parameter(0)
      b = f32[4] parameter(1)
      m0 = f32[4] maximum(a, b)
      ROOT compare = pred[4] compare(m0, a), direction=GT
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Compare(m::Parameter(1), m::Parameter(0))
                     .WithComparisonDirection(ComparisonDirection::kGt)));
}

TEST_F(AlgebraicSimplifierTest, CompareGtMaxB) {
  // Gt(Max(a,b), b) -> Gt(a,b)
  const char* kModuleStr = R"(
    HloModule m
    test {
      a = f32[4] parameter(0)
      b = f32[4] parameter(1)
      m0 = f32[4] maximum(a, b)
      ROOT compare = pred[4] compare(m0, b), direction=GT
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Compare(m::Parameter(0), m::Parameter(1))
                     .WithComparisonDirection(ComparisonDirection::kGt)));
}

TEST_F(AlgebraicSimplifierTest, CompareGtAMin) {
  // Gt(a, Min(a,b)) -> Gt(a,b)
  const char* kModuleStr = R"(
    HloModule m
    test {
      a = f32[4] parameter(0)
      b = f32[4] parameter(1)
      m0 = f32[4] minimum(a, b)
      ROOT compare = pred[4] compare(a, m0), direction=GT
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Compare(m::Parameter(0), m::Parameter(1))
                     .WithComparisonDirection(ComparisonDirection::kGt)));
}

TEST_F(AlgebraicSimplifierTest, CompareGtBMin) {
  // Gt(b, Min(a,b)) -> Gt(b,a)
  const char* kModuleStr = R"(
    HloModule m
    test {
      a = f32[4] parameter(0)
      b = f32[4] parameter(1)
      m0 = f32[4] minimum(a, b)
      ROOT compare = pred[4] compare(b, m0), direction=GT
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Compare(m::Parameter(1), m::Parameter(0))
                     .WithComparisonDirection(ComparisonDirection::kGt)));
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::ConstantScalar(false))));
}

TEST_F(AlgebraicSimplifierTest, CompareAbsLtZeroBecomesFalse) {
  // |x| < 0  ->  false
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(R"(
m {
  p = s32[5] parameter(0)
  a = s32[5] abs(p)
  z = s32[] constant(0)
  b = s32[5] broadcast(z)
  ROOT r = pred[5] compare(a, b), direction=LT
})"));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::ConstantScalar(false))));
}

TEST_F(AlgebraicSimplifierTest, CompareLtZero) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(param, zero), direction=LT
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ConstantScalar(false)));
}

TEST_F(AlgebraicSimplifierTest, CompareLeZero) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(param, zero), direction=LE
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Le(m::Parameter(0), m::ConstantEffectiveScalar(0))));
}

TEST_F(AlgebraicSimplifierTest, CompareGeZero) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(param, zero), direction=GE
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ConstantScalar(true)));
}

TEST_F(AlgebraicSimplifierTest, CompareGtZero) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(param, zero), direction=GT
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Gt(m::Parameter(0), m::ConstantEffectiveScalar(0))));
}

TEST_F(AlgebraicSimplifierTest, CompareZeroGt) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(zero, param), direction=GT
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ConstantScalar(false)));
}

TEST_F(AlgebraicSimplifierTest, CompareZeroGe) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(zero, param), direction=GE
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Ge(m::ConstantEffectiveScalar(0), m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, CompareZeroLe) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(zero, param), direction=LE
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::ConstantScalar(true)));
}

TEST_F(AlgebraicSimplifierTest, CompareZeroLt) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      zero = u32[] constant(0)
      param = u32[] parameter(0)
      ROOT compare = pred[] compare(zero, param), direction=LT
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Lt(m::ConstantEffectiveScalar(0), m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, CompareSame) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      param = s32[123] parameter(0)
      ROOT compare = pred[123] compare(param, param), direction=GE
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::ConstantScalar(true))));
}

TEST_F(AlgebraicSimplifierTest, CompareSimplified) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      param = s32[] parameter(0)
      c1 = s32[] constant(10)
      c2 = s32[] constant(100)
      cmp1 = pred[] compare(param, c1), direction=LT
      cmp2 = pred[] compare(param, c2), direction=LT
      ROOT out = pred[] and(cmp1, cmp2)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Compare(m::Op(), m::Op().IsConstantScalar(10))
                     .WithComparisonDirection(ComparisonDirection::kLt)));
}

TEST_F(AlgebraicSimplifierTest, CompareSimplifiedReversed) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      param = s32[] parameter(0)
      c1 = s32[] constant(10)
      c2 = s32[] constant(100)
      cmp1 = pred[] compare(param, c1), direction=LT
      cmp2 = pred[] compare(c2, param), direction=GT
      ROOT out = pred[] and(cmp1, cmp2)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Compare(m::Op(), m::Op().IsConstantScalar(10))
                     .WithComparisonDirection(ComparisonDirection::kLt)));
}

// Test that A != False is simplified to A
TEST_F(AlgebraicSimplifierTest, NeFalse) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_false = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateCompare(
      r0pred, param0, const_false, ComparisonDirection::kNe));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCompare);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that False != A is simplified to A
TEST_F(AlgebraicSimplifierTest, NeFalse2) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_false = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  builder.AddInstruction(HloInstruction::CreateCompare(
      r0pred, const_false, param0, ComparisonDirection::kNe));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCompare);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that A == True is simplified to A
TEST_F(AlgebraicSimplifierTest, EqTrue) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_true = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  builder.AddInstruction(HloInstruction::CreateCompare(
      r0pred, param0, const_true, ComparisonDirection::kEq));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCompare);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

// Test that True == A is simplified to A
TEST_F(AlgebraicSimplifierTest, EqTrue2) {
  auto m = CreateNewVerifiedModule();
  Shape r0pred = ShapeUtil::MakeShape(PRED, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0pred, "param0"));
  HloInstruction* const_true = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  builder.AddInstruction(HloInstruction::CreateCompare(
      r0pred, const_true, param0, ComparisonDirection::kEq));

  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCompare);
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  root = computation->root_instruction();
  EXPECT_EQ(root, param0);
}

TEST_F(AlgebraicSimplifierTest, CompareSelectCompare) {
  // Causal mask suboptimal HLO simplification
  // Ne(select(Ge(a, b), ones, zeros), zeros) -> Ge(a, b)
  const char* kModuleStr = R"(
    HloModule m
    test {
      a = s32[4,4] parameter(0)
      b = s32[4,4] parameter(1)
      %cmp0 = pred[4,4] compare(a, b), direction=GE
      %c1 = f32[] constant(1)
      %ones = f32[4,4] broadcast(f32[] %c1)
      %c0 = f32[] constant(0)
      %zeros = f32[4,4] broadcast(f32[] %c0)
      %sel0 = f32[4,4] select(%cmp0, %ones, %zeros)
      ROOT %cmp1 = pred[4,4] compare(%sel0, %zeros), direction=NE
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Compare(m::Parameter(0), m::Parameter(1))
                     .WithComparisonDirection(ComparisonDirection::kGe)));
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m1.get()).value());
  EXPECT_THAT(m1->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(m::Op(), m::Op())));

  // Verify that we can disable the re-write
  AlgebraicSimplifierOptions opts = default_options_;
  opts.set_enable_dot_to_multiply_rewrite(false);
  TF_ASSERT_OK_AND_ASSIGN(auto m2, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(opts).Run(m2.get()).value());
}

TEST_F(AlgebraicSimplifierTest,
       NoDotToMultiplyRewriteWithPrecisionConfigAlgorithm) {
  constexpr char kModuleStr[] = R"(
HloModule test
ENTRY dot {
 a = f32[128]{0} parameter(0)
 b = f32[128]{0} parameter(1)
 ROOT dot = f32[128,128]{1,0} dot(a, b), algorithm=dot_tf32_tf32_f32
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest,
       NoDotToMultiplyRewriteZeroContractingDimWithPrecisionConfigAlgorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT dot = f32[] dot(a, b), algorithm=dot_tf32_tf32_f32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(options).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
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
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(1)));
}

TEST_F(AlgebraicSimplifierTest, SliceOfMultipleConcatOperands) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[50,50] parameter(0)
      p1 = f32[50,50] parameter(1)
      p2 = f32[50,50] parameter(2)
      p3 = f32[50,50] parameter(3)
      c0 = f32[200,50] concatenate(p0, p1, p2, p3), dimensions={0}
      ROOT s0 = f32[98,50] slice(c0), slice={[51:149], [0:50]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Slice(m::Concatenate(m::Parameter(1), m::Parameter(2)))));
  EXPECT_THAT(m->entry_computation()->root_instruction()->slice_starts(),
              ElementsAre(1, 0));
  EXPECT_THAT(m->entry_computation()->root_instruction()->slice_limits(),
              ElementsAre(99, 50));
}

TEST_F(AlgebraicSimplifierTest, SqrtOfSelfMultiply) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[32]{0} parameter(0)
      m0 = f32[32]{0} multiply(f32[32]{0} p0, f32[32]{0} p0)
      ROOT s0 = f32[32]{0} sqrt(f32[32]{0} m0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Abs(m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, ReduceOfBatchDotToContractingDimension) {
  const char* kModuleStr = R"(
    HloModule m
    a {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT r = f32[] add(p0, p1)
    }
    test {
      p0 = f32[32,8,5,6] parameter(0)
      p1 = f32[8,32,6,7] parameter(1)
      d = f32[32,8,5,7] dot(p0, p1),
        lhs_batch_dims={0,1},
        rhs_batch_dims={1,0},
        rhs_contracting_dims={2},
        lhs_contracting_dims={3}
     c = f32[] constant(0)
     ROOT r = f32[8,5,7] reduce(d,c), dimensions={0}, to_apply=a
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Parameter(0), m::Parameter(1))));
}

// Same test as above, but with the option supports_non_canonical_dots set to
// false.
TEST_F(AlgebraicSimplifierTest,
       ReduceOfBatchDotToContractingDimensionDisabled) {
  const char* kModuleStr = R"(
    HloModule m
    a {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT r = f32[] add(p0, p1)
    }
    test {
      p0 = f32[32,8,5,6] parameter(0)
      p1 = f32[8,32,6,7] parameter(1)
      d = f32[32,8,5,7] dot(p0, p1),
        lhs_batch_dims={0,1},
        rhs_batch_dims={1,0},
        rhs_contracting_dims={2},
        lhs_contracting_dims={3}
     c = f32[] constant(0)
     ROOT r = f32[8,5,7] reduce(d,c), dimensions={0}, to_apply=a
    }
  )";
  AlgebraicSimplifierOptions options = default_options_;
  options.set_supports_non_canonical_dots(false);
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(options).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, ReduceAddIsCommutative) {
  const char* kModuleStr = R"(
    HloModule m
    fn1 {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT r = f32[] add(p0, p1)
    }
    fn2 {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT r = f32[] add(p1, p0)
    }
    test {
      p0 = f32[10,10,10] parameter(0)
      zero = f32[] constant(0)
      r1 = f32[10,10] reduce(p0, zero), dimensions={0}, to_apply=fn1
      ROOT r2 = f32[10] reduce(r1, zero), dimensions={0}, to_apply=fn2
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Reduce(m::Parameter(0), m::ConstantScalar(0))));
}

// rsqrt(pow(x, -2)) => x, for x >= 0
TEST_F(AlgebraicSimplifierTest, RsqrtOfRPower2) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[1,32] parameter(0)
      abs0 = f32[1,32] abs(p0)
      c1 = f32[] constant(-2.0)
      broadcast = f32[1,32] broadcast(c1), dimensions={}
      power = f32[1,32] power(abs0, broadcast)
      ROOT rsqrt = f32[1, 32] rsqrt(power)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Abs(m::Parameter(0))));
}

// rsqrt(pow(x, -2)) => x
// if x is arbitrary number - no simplification
TEST_F(AlgebraicSimplifierTest, RsqrtOfRPower2_NegativeTestCase) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[1,32] parameter(0)
      c1 = f32[] constant(-2.0)
      broadcast = f32[1,32] broadcast(c1), dimensions={}
      power = f32[1,32] power(p0, broadcast)
      ROOT rsqrt = f32[1, 32] rsqrt(power)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

// rsqrt(1/x) => sqrt(x), for x >= 0
TEST_F(AlgebraicSimplifierTest, RsqrtDivide) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[1,32] parameter(0)
      abs0 = f32[1,32] abs(p0)
      c1 = f32[] constant(1.0)
      broadcast.1 = f32[1,32] broadcast(c1), dimensions={}
      divide = f32[1,32] divide(broadcast.1, abs0)
      ROOT rsqrt = f32[1,32] rsqrt(divide)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Sqrt(m::Abs(m::Parameter(0)))));
}

// rsqrt(1/x) => sqrt(x)
// if x is arbitrary number - no simplification
TEST_F(AlgebraicSimplifierTest, RsqrtDivide_NegativeTestCase) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[1,32] parameter(0)
      c1 = f32[] constant(1.0)
      broadcast.1 = f32[1,32] broadcast(c1), dimensions={}
      divide = f32[1,32] divide(broadcast.1, p0)
      ROOT rsqrt = f32[1,32] rsqrt(divide)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

// sqrt(x) * sqrt(x) => x, for x >= 0
TEST_F(AlgebraicSimplifierTest, MultiplySelfSqrt) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[1,32] parameter(0)
      abs0 = f32[1,32] abs(p0)
      sqrt = f32[1,32] sqrt(abs0)
      ROOT mul = f32[1,32] multiply(sqrt, sqrt)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Abs(m::Parameter(0))));
}

// sqrt(x) * sqrt(y) is not simplified.
TEST_F(AlgebraicSimplifierTest, MultiplySqrtDifferentOperands) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[1,32] parameter(0)
      abs = f32[1,32] abs(p0)
      exp = f32[1,32] exponential(p0)
      sqrt = f32[1,32] sqrt(abs)
      sqrt2 = f32[1,32] sqrt(exp)
      ROOT mul = f32[1,32] multiply(sqrt, sqrt2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

// sqrt(x) * sqrt(x) ≠> x
// if x is arbitrary number - no simplification
TEST_F(AlgebraicSimplifierTest, MultiplySelfSqrt_NegativeTestCase) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[1,32] parameter(0)
      sqrt = f32[1,32] sqrt(p0)
      ROOT mul = f32[1,32] multiply(sqrt, sqrt)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

// rsqrt(x) * rsqrt(x) -> 1/x, for x >= 0
TEST_F(AlgebraicSimplifierTest, MultiplySelfRsqrt) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[1,32] parameter(0)
      abs0 = f32[1,32] abs(p0)
      rsqrt = f32[1,32] rsqrt(abs0)
      ROOT mul = f32[1,32] multiply(rsqrt, rsqrt)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Divide(m::Broadcast(m::ConstantScalar(1.0)),
                                   m::Abs(m::Parameter(0)))));
}

// rsqrt(x) * rsqrt(y) is not simplified.
TEST_F(AlgebraicSimplifierTest, MultiplyRsqrtDifferentOperands) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[1,32] parameter(0)
      abs = f32[1,32] abs(p0)
      exp = f32[1,32] exponential(p0)
      rsqrt = f32[1,32] rsqrt(abs)
      rsqrt2 = f32[1,32] rsqrt(exp)
      ROOT mul = f32[1,32] multiply(rsqrt, rsqrt2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

// rsqrt(x) * rsqrt(x) -> 1/x
// if x is arbitrary number - no simplification
TEST_F(AlgebraicSimplifierTest, MultiplySelfRsqrt_NegativeTestCase) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[1,32] parameter(0)
      rsqrt = f32[1,32] rsqrt(p0)
      ROOT mul = f32[1,32] multiply(rsqrt, rsqrt)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, MultiplyNegateNegate) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      neg0 = f32[] negate(p0)
      neg1 = f32[] negate(p1)
      ROOT mul = f32[] multiply(neg0, neg1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, AbsEliminationBatchnormTraining) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[128,32,2,112]{3,2,1,0} parameter(0)
      p1 = f32[32]{0} parameter(1)
      p2 = f32[32]{0} parameter(2)
      constant = f32[] constant(0.001)
      constant.1 = s64[] constant(1)
      custom-call.1 = (f32[128,32,2,112]{3,2,1,0}, f32[32]{0}, f32[32]{0}) custom-call(p0, p1, p2, constant, constant.1), custom_call_target="__cudnn$batchNormalizationForwardTraining"
      get-tuple-element.1 = f32[128,32,2,112]{3,2,1,0} get-tuple-element(custom-call.1), index=0
      get-tuple-element.2 = f32[32]{0} get-tuple-element(custom-call.1), index=1
      get-tuple-element = f32[32]{0} get-tuple-element(custom-call.1), index=2
      abs = f32[32]{0} abs(get-tuple-element)
      ROOT %tuple = (f32[128,32,2,112]{3,2,1,0}, f32[32]{0}, f32[32]{0}) tuple(get-tuple-element.1, get-tuple-element.2, abs)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  default_options_.set_cudnn_batchnorm_forward_training_metadata(
      "__cudnn$batchNormalizationForwardTraining");
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  // Verify that the module doesn't have any abs node.
  EXPECT_EQ(FindInstruction(m.get(), HloOpcode::kAbs), nullptr);
  EXPECT_EQ(m->entry_computation()->root_instruction()->operand(2)->opcode(),
            HloOpcode::kGetTupleElement);
}

TEST_F(AlgebraicSimplifierTest,
       AbsEliminationBatchnormTraining_NegativeTestCase) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[128,32,2,112]{3,2,1,0} parameter(0)
      p1 = f32[32]{0} parameter(1)
      p2 = f32[32]{0} parameter(2)
      constant = f32[] constant(0.001)
      constant.1 = s64[] constant(1)
      custom-call.1 = (f32[128,32,2,112]{3,2,1,0}, f32[32]{0}, f32[32]{0}) custom-call(p0, p1, p2, constant, constant.1), custom_call_target="__cudnn$batchNormalizationForwardTraining"
      get-tuple-element.1 = f32[128,32,2,112]{3,2,1,0} get-tuple-element(custom-call.1), index=0
      get-tuple-element.2 = f32[32]{0} get-tuple-element(custom-call.1), index=1
      get-tuple-element = f32[32]{0} get-tuple-element(custom-call.1), index=2
      abs = f32[32]{0} abs(get-tuple-element)
      ROOT %tuple = (f32[128,32,2,112]{3,2,1,0}, f32[32]{0}, f32[32]{0}) tuple(get-tuple-element.1, get-tuple-element.2, abs)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  default_options_.set_cudnn_batchnorm_forward_training_metadata(
      "__cudnn$batchNormalizationForwardInference");
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_NE(FindInstruction(m.get(), HloOpcode::kAbs), nullptr);
}

TEST_F(AlgebraicSimplifierTest, AbsEliminationMultiply) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = f32[32]{0} parameter(0)
      m = f32[32]{0} multiply(p, p)
      ROOT a = f32[32]{0} abs(m)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, AbsEliminationPower2) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[32]{0} parameter(0)
      c0 = f32[] constant(2)
      b0 = f32[32]{0} broadcast(c0), dimensions={}
      pow = f32[32]{0} power(p0, b0)
      ROOT a = f32[32]{0} abs(pow)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  // Pow(A, 2) is transformed to AA. As a result, Abs(Power(A, 2)) is
  // transformed to AA.
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, ScatterAddCombined) {
  const char* hlo_string = R"(
  HloModule m
  apply {
   a = f32[] parameter(0)
   b = f32[] parameter(1)
   ROOT c = f32[] add(a, b)
  }
  test {
    z  = f32[] constant(0)
    init = f32[100,4] broadcast(z), dimensions={}
    shared = f32[100,4] parameter(0)
    index0 = s32[20] parameter(1)
    index1 = s32[10] parameter(2)
    update0 = f32[20,4] parameter(3)
    update1 = f32[10,4] parameter(4)
    scatter.0 = f32[100,4] scatter(init, index0, update0),
              to_apply=apply,
              update_window_dims={1},
              inserted_window_dims={0},
              scatter_dims_to_operand_dims={0},
              index_vector_dim=1
    scatter.1 = f32[100,4] scatter(init, index1, update1),
              to_apply=apply,
              update_window_dims={1},
              inserted_window_dims={0},
              scatter_dims_to_operand_dims={0},
              index_vector_dim=1
    add.0 = f32[100,4] add(shared, scatter.0)
    ROOT add.1 = f32[100,4] add(add.0, scatter.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  // Combine Scatters
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  // Optimize Add with 0
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Scatter(m::Parameter(0),
                            m::Concatenate(m::Parameter(1), m::Parameter(2)),
                            m::Concatenate(m::Parameter(3), m::Parameter(4)))));
}

TEST_F(AlgebraicSimplifierTest, ScatterAddCombinedSwapped) {
  const char* hlo_string = R"(
  HloModule m
  apply {
   a = f32[] parameter(0)
   b = f32[] parameter(1)
   ROOT c = f32[] add(a, b)
  }
  test {
    z  = f32[] constant(0)
    init = f32[100,4] broadcast(z), dimensions={}
    shared = f32[100,4] parameter(0)
    index0 = s32[20] parameter(1)
    index1 = s32[10] parameter(2)
    update0 = f32[20,4] parameter(3)
    update1 = f32[10,4] parameter(4)
    scatter.0 = f32[100,4] scatter(init, index0, update0),
              to_apply=apply,
              update_window_dims={1},
              inserted_window_dims={0},
              scatter_dims_to_operand_dims={0},
              index_vector_dim=1
    scatter.1 = f32[100,4] scatter(init, index1, update1),
              to_apply=apply,
              update_window_dims={1},
              inserted_window_dims={0},
              scatter_dims_to_operand_dims={0},
              index_vector_dim=1
    add.0 = f32[100,4] add(shared, scatter.0)
    ROOT add.1 = f32[100,4] add(scatter.1, add.0)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  // Combine Scatters
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  // Optimize Add with 0
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Scatter(m::Parameter(0),
                            m::Concatenate(m::Parameter(2), m::Parameter(1)),
                            m::Concatenate(m::Parameter(4), m::Parameter(3)))));
}

TEST_F(AlgebraicSimplifierTest, ScatterAddCombinedWeirdDnums) {
  const char* hlo_string = R"(
  HloModule m
  apply {
   a = f32[] parameter(0)
   b = f32[] parameter(1)
   ROOT c = f32[] add(a, b)
  }
  test {
    z  = f32[] constant(0)
    init = f32[100,4] broadcast(z), dimensions={}
    shared = f32[100,4] parameter(0)
    index0 = s32[1,4,5] parameter(1)
    index1 = s32[1,2,5] parameter(2)
    update0 = f32[4,4,5] parameter(3)
    update1 = f32[2,4,5] parameter(4)
    scatter.0 = f32[100,4] scatter(init, index0, update0),
              to_apply=apply,
              update_window_dims={1},
              inserted_window_dims={0},
              scatter_dims_to_operand_dims={0},
              index_vector_dim=0
    scatter.1 = f32[100,4] scatter(init, index1, update1),
              to_apply=apply,
              update_window_dims={1},
              inserted_window_dims={0},
              scatter_dims_to_operand_dims={0},
              index_vector_dim=0
    ROOT add.1 = f32[100,4] add(scatter.0, scatter.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  // Combine Scatters
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  // Simplify Add
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Scatter(m::Broadcast(),
                            m::Concatenate(m::Parameter(1), m::Parameter(2)),
                            m::Concatenate(m::Parameter(3), m::Parameter(4)))));
}

TEST_F(AlgebraicSimplifierTest, ScatterAddCombinedWeirdDnums2) {
  const char* hlo_string = R"(
  HloModule m
  apply {
   a = f32[] parameter(0)
   b = f32[] parameter(1)
   ROOT c = f32[] add(a, b)
  }
  test {
    z  = f32[] constant(0)
    init = f32[100,4] broadcast(z), dimensions={}
    shared = f32[100,4] parameter(0)
    index0 = s32[4,3,1] parameter(1)
    index1 = s32[4,5,1] parameter(2)
    update0 = f32[4,4,3] parameter(3)
    update1 = f32[4,4,5] parameter(4)
    scatter.0 = f32[100,4] scatter(init, index0, update0),
              to_apply=apply,
              update_window_dims={0},
              inserted_window_dims={0},
              scatter_dims_to_operand_dims={0},
              index_vector_dim=2
    scatter.1 = f32[100,4] scatter(init, index1, update1),
              to_apply=apply,
              update_window_dims={0},
              inserted_window_dims={0},
              scatter_dims_to_operand_dims={0},
              index_vector_dim=2
    ROOT add.1 = f32[100,4] add(scatter.0, scatter.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  // Combine Scatters
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  // Simplify Add
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Scatter(m::Broadcast(),
                            m::Concatenate(m::Parameter(1), m::Parameter(2)),
                            m::Concatenate(m::Parameter(3), m::Parameter(4)))));
}

TEST_F(AlgebraicSimplifierTest, ScatterAddCombinedWithBatchDim) {
  const char* hlo_string = R"(
  HloModule m
  apply {
   a = f32[] parameter(0)
   b = f32[] parameter(1)
   ROOT c = f32[] add(a, b)
  }
  test {
    z  = f32[] constant(0)
    init = f32[3,100,4] broadcast(z), dimensions={}
    index0 = s32[3,1,4,5] parameter(0)
    index1 = s32[3,1,2,5] parameter(1)
    update0 = f32[3,4,4,5] parameter(2)
    update1 = f32[3,2,4,5] parameter(3)
    scatter.0 = f32[3,100,4] scatter(init, index0, update0),
              to_apply=apply,
              update_window_dims={2},
              inserted_window_dims={1},
              scatter_dims_to_operand_dims={1},
              index_vector_dim=1,
              input_batching_dims={0},
              scatter_indices_batching_dims={0}
    scatter.1 = f32[3,100,4] scatter(init, index1, update1),
              to_apply=apply,
              update_window_dims={2},
              inserted_window_dims={1},
              scatter_dims_to_operand_dims={1},
              index_vector_dim=1,
              input_batching_dims={0},
              scatter_indices_batching_dims={0}
    ROOT add.1 = f32[3,100,4] add(scatter.0, scatter.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  // Combine Scatters
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  // Simplify Add
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  const HloInstruction* concat1;
  const HloInstruction* concat2;
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Scatter(
                  m::Broadcast(),
                  m::Concatenate(&concat1, m::Parameter(0), m::Parameter(1)),
                  m::Concatenate(&concat2, m::Parameter(2), m::Parameter(3)))));
  EXPECT_EQ(Cast<HloConcatenateInstruction>(concat1)->concatenate_dimension(),
            2);
  EXPECT_EQ(Cast<HloConcatenateInstruction>(concat2)->concatenate_dimension(),
            1);
}

TEST_F(AlgebraicSimplifierTest, ScatterAddCombinedWithBatchDim2) {
  const char* hlo_string = R"(
  HloModule m
  apply {
   a = f32[] parameter(0)
   b = f32[] parameter(1)
   ROOT c = f32[] add(a, b)
  }
   test {
    z  = f32[] constant(0)
    init = f32[100,3,4] broadcast(z), dimensions={}
    index0 = s32[4,3,5,1] parameter(0)
    index1 = s32[2,3,5,1] parameter(1)
    update0 = f32[4,3,4,5] parameter(2)
    update1 = f32[2,3,4,5] parameter(3)
    scatter.0 = f32[100,3,4] scatter(init, index0, update0),
              to_apply=apply,
              update_window_dims={2},
              inserted_window_dims={0},
              scatter_dims_to_operand_dims={0},
              index_vector_dim=3,
              input_batching_dims={1},
              scatter_indices_batching_dims={1}
    scatter.1 = f32[100,3,4] scatter(init, index1, update1),
              to_apply=apply,
              update_window_dims={2},
              inserted_window_dims={0},
              scatter_dims_to_operand_dims={0},
              index_vector_dim=3,
              input_batching_dims={1},
              scatter_indices_batching_dims={1}
    ROOT add.1 = f32[100,3,4] add(scatter.0, scatter.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  // Combine Scatters
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  // Simplify Add
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  const HloInstruction* concat1;
  const HloInstruction* concat2;
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Scatter(
                  m::Broadcast(),
                  m::Concatenate(&concat1, m::Parameter(0), m::Parameter(1)),
                  m::Concatenate(&concat2, m::Parameter(2), m::Parameter(3)))));
  EXPECT_EQ(Cast<HloConcatenateInstruction>(concat1)->concatenate_dimension(),
            0);
  EXPECT_EQ(Cast<HloConcatenateInstruction>(concat2)->concatenate_dimension(),
            0);
}

TEST_F(AlgebraicSimplifierTest, ScalarScatter) {
  const char* hlo_string = R"(
  HloModule m
  apply {
   a = f32[] parameter(0)
   b = f32[] parameter(1)
   ROOT c = f32[] add(a, b)
  }
  test {
    z  = f32[] constant(0)
    init = f32[100,4,20] broadcast(z), dimensions={}
    shared = f32[100,4,20] parameter(0)
    index0 = s32[1] parameter(1)
    index1 = s32[1] parameter(2)
    update0 = f32[4,20] parameter(3)
    update1 = f32[4,20] parameter(4)
    scatter.0 = f32[100,4,20] scatter(init, index0, update0),
              to_apply=apply,
              update_window_dims={0, 1},
              inserted_window_dims={0},
              scatter_dims_to_operand_dims={0},
              index_vector_dim=0
    scatter.1 = f32[100,4,20] scatter(init, index1, update1),
              to_apply=apply,
              update_window_dims={0, 1},
              inserted_window_dims={0},
              scatter_dims_to_operand_dims={0},
              index_vector_dim=0
    ROOT add.1 = f32[100,4,20] add(scatter.0, scatter.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  // Combine Scatters
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, SwapConvOperands) {
  const char* hlo_string = R"(
  HloModule m
  test {
    a = f32[3,3,160,160] parameter(0)
    b = f32[128,32,32,160] parameter(1)
    ROOT c = f32[128,32,32,160] convolution(a,b),
     window={size=32x32 pad=30_30x30_30 rhs_reversal=1x1},
     dim_labels=01bf_o01i->f01b
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  // Combine Scatters
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  const HloInstruction* conv = m->entry_computation()->root_instruction();
  EXPECT_THAT(conv,
              GmockMatch(m::Convolution(m::Parameter(1), m::Parameter(0))));
  EXPECT_EQ(conv->window().dimensions(0).size(), 3);
  EXPECT_EQ(conv->window().dimensions(1).size(), 3);
  EXPECT_EQ(conv->window().dimensions(0).window_reversal(), true);
  EXPECT_EQ(conv->window().dimensions(1).window_reversal(), true);
  EXPECT_EQ(conv->window().dimensions(0).padding_low(), 1);
  EXPECT_EQ(conv->window().dimensions(1).padding_low(), 1);
  EXPECT_EQ(conv->window().dimensions(0).padding_high(), 1);
  EXPECT_EQ(conv->window().dimensions(1).padding_high(), 1);
}

TEST_F(AlgebraicSimplifierTest, ScalarDividePredicate) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = pred[2] parameter(0)
      cvt = f32[2] convert(p0)
      p1 = f32[] parameter(1)
      bcast = f32[2] broadcast(p1), dimensions={}
      ROOT div = f32[2] divide(cvt, bcast)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::MultiplyAnyOrder(
          m::Convert(m::Parameter(0)),
          m::Broadcast(m::Divide(m::ConstantScalar(1), m::Parameter(1))))));
}

TEST_F(AlgebraicSimplifierTest, MultipleDotStrengthReductions) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY test {
      a = c64[2,2] parameter(0)
      b = c64[2] parameter(1)
      cd = c64[2] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      c = f64[2,2] parameter(2)
      d = f64[2] parameter(3)
      dd = f64[2] dot(c, d), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT tuple = (c64[2], f64[2]) tuple(cd, dd)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_EQ(3, m->computation_count());
}

TEST_F(AlgebraicSimplifierTest,
       NoDotStrengthReductionWithPrecisionConfigAlgorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
      a = f32[128,2]{1,0} parameter(0)
      b = f32[2]{0} parameter(1)
      ROOT dot = f32[128]{0} dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_tf32_tf32_f32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, UnaryVariadicReduce) {
  const char* kModuleStr = R"(
    HloModule m
    fn {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      a = f32[] add(p0, p1)
      ROOT t = (f32[]) tuple(a)
    }
    test {
      p0 = f32[32,8,6,7] parameter(0)
      c = f32[] constant(0)
      ROOT r = (f32[8,6,7]) reduce(p0, c), dimensions={0}, to_apply=fn
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  ASSERT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Reduce(m::Parameter(0), m::ConstantScalar(0)))));
  ASSERT_EQ(m->entry_computation()
                ->root_instruction()
                ->operand(0)
                ->called_computations()
                .size(),
            1);
  EXPECT_THAT(m->entry_computation()
                  ->root_instruction()
                  ->operand(0)
                  ->called_computations()[0]
                  ->root_instruction(),
              GmockMatch(m::Add(m::Parameter(0), m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, ReplaceReduceSumOfConstantBroadcast) {
  const char* kModuleStr = R"(
  HloModule ReplaceReduceSumOfConstantBroadcast

  add_f32 {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    ROOT r = f32[] add(p0, p1)
  }

  ENTRY main {
    init_value = f32[] constant(0)
    const_value = f32[] constant(1)
    const_bcast = f32[8, 128] broadcast(f32[] const_value), dimensions={}
    ROOT reduce = f32[8] reduce(f32[8, 128] const_bcast, f32[] init_value), dimensions={1}, to_apply=add_f32
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_TRUE(simplifier.Run(m.get()).value());
  int64_t reduce_count =
      absl::c_count_if(m->entry_computation()->instructions(),
                       HloPredicateIsOp<HloOpcode::kReduce>);
  // Expect no Reduce operation after simplification.
  EXPECT_EQ(0, reduce_count);
}

TEST_F(AlgebraicSimplifierTest, ReplaceReduceMaxWithReduceArgMax) {
  const char* kModuleStr = R"(
HloModule ReplaceReduceMaxWithReduceArgMax

%reduction_computation__1.25287 (parameter.25288: bf16[], parameter.25289: s32[], parameter.25290: bf16[], parameter.25291: s32[]) -> (bf16[], s32[]) {
  %constant.25292 = pred[] constant(false)
  %parameter.25288 = bf16[] parameter(0)
  %parameter.25290 = bf16[] parameter(2)
  %compare.25293 = pred[] compare(bf16[] %parameter.25288, bf16[] %parameter.25290), direction=GT
  %compare.25294 = pred[] compare(bf16[] %parameter.25288, bf16[] %parameter.25288), direction=NE
  %or.25295 = pred[] or(pred[] %compare.25293, pred[] %compare.25294)
  %select.25300 = bf16[] select(pred[] %or.25295, bf16[] %parameter.25288, bf16[] %parameter.25290)
  %compare.25296 = pred[] compare(bf16[] %parameter.25288, bf16[] %parameter.25290), direction=EQ
  %parameter.25289 = s32[] parameter(1)
  %parameter.25291 = s32[] parameter(3)
  %compare.25297 = pred[] compare(s32[] %parameter.25289, s32[] %parameter.25291), direction=LT
  %and.25298 = pred[] and(pred[] %compare.25296, pred[] %compare.25297)
  %or.25299 = pred[] or(pred[] %or.25295, pred[] %and.25298)
  %select.25301 = s32[] select(pred[] %or.25299, s32[] %parameter.25289, s32[] %parameter.25291)
  ROOT %tuple.25302 = (bf16[], s32[]) tuple(bf16[] %select.25300, s32[] %select.25301)
}

%primitive_computation_max.25303 (parameter.25304: bf16[], parameter.25305: bf16[]) -> bf16[] {
  %parameter.25304 = bf16[] parameter(0), metadata={op_type="max" op_name="max"}
  %parameter.25305 = bf16[] parameter(1), metadata={op_type="max" op_name="max"}
  ROOT %maximum.25306 = bf16[] maximum(bf16[] %parameter.25304, bf16[] %parameter.25305), metadata={op_type="max" op_name="max"}
}

ENTRY %main {
  %p0 = bf16[384,128,19392]{2,1,0} parameter(0)

  // Variadic Reduce (ArgMax)
  %iota.25376 = s32[384,128,19392] iota(), iota_dimension=2
  %constant.25377 = bf16[] constant(-inf)
  %constant.25378 = s32[] constant(0)
  %reduce.25379 = (bf16[384,128]{1,0}, s32[384,128]{1,0}) reduce(bf16[384,128,19392]{2,1,0} %p0, s32[384,128,19392] %iota.25376, bf16[] %constant.25377, s32[] %constant.25378), dimensions={2}, to_apply=%reduction_computation__1.25287

  %get-tuple-element.25381 = s32[384,128]{1,0} get-tuple-element((bf16[384,128]{1,0}, s32[384,128]{1,0}) %reduce.25379), index=1

  // Reduce (Max)
  %constant.25382 = bf16[] constant(-inf)
  %reduce.25383 = bf16[384,128]{1,0} reduce(bf16[384,128,19392]{2,1,0} %p0, bf16[] %constant.25382), dimensions={2}, to_apply=%primitive_computation_max.25303

  ROOT %tuple.0 = (bf16[384,128]{1,0}, s32[384,128]{1,0}) tuple(%reduce.25383, %get-tuple-element.25381)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  int64_t reduce_count =
      absl::c_count_if(m->entry_computation()->instructions(),
                       HloPredicateIsOp<HloOpcode::kReduce>);
  // Expect one Reduce operation after simplification.
  EXPECT_EQ(1, reduce_count);
  auto variadic_reduce = m::Reduce().WithShape(m::Shape().IsTuple());
  auto root = m->entry_computation()->root_instruction();
  // Expect that both outputs are fed by 'variadic_reduce'.
  ASSERT_THAT(root,
              GmockMatch(m::Tuple(m::GetTupleElement(variadic_reduce, 0),
                                  m::GetTupleElement(variadic_reduce, 1))));
}

TEST_F(AlgebraicSimplifierTest, ReplaceReduceMinWithReduceArgMin) {
  const char* kModuleStr = R"(
HloModule ReplaceReduceMinWithReduceArgMin

%region_3.84 (Arg_0.85: bf16[], Arg_1.86: s32[], Arg_2.87: bf16[], Arg_3.88: s32[]) -> (bf16[], s32[]) {
  %Arg_3.88 = s32[]{:T(256)} parameter(3)
  %Arg_2.87 = bf16[]{:T(512)} parameter(2)
  %Arg_1.86 = s32[]{:T(256)} parameter(1)
  %compare.93 = pred[]{:T(1024)S(6)} compare(s32[]{:T(256)} %Arg_1.86, s32[]{:T(256)} %Arg_3.88), direction=LT, metadata={op_name="lt" source_file="<ipython-input-4-4f3bd086a82e>" source_line=12}
  %Arg_0.85 = bf16[]{:T(512)} parameter(0)
  %compare.92 = pred[]{:T(1024)S(6)} compare(bf16[]{:T(512)} %Arg_0.85, bf16[]{:T(512)} %Arg_2.87), direction=EQ, metadata={op_name="eq" source_file="<ipython-input-4-4f3bd086a82e>" source_line=12}
  %and.94 = pred[]{:T(1024)S(6)} and(pred[]{:T(1024)S(6)} %compare.92, pred[]{:T(1024)S(6)} %compare.93), metadata={op_name="and" source_file="<ipython-input-4-4f3bd086a82e>" source_line=12}
  %compare.90 = pred[]{:T(1024)S(6)} compare(bf16[]{:T(512)} %Arg_0.85, bf16[]{:T(512)} %Arg_0.85), direction=NE, metadata={op_name="ne" source_file="<ipython-input-4-4f3bd086a82e>" source_line=12}
  %compare.89 = pred[]{:T(1024)S(6)} compare(bf16[]{:T(512)} %Arg_0.85, bf16[]{:T(512)} %Arg_2.87), direction=LT, metadata={op_name="lt" source_file="<ipython-input-4-4f3bd086a82e>" source_line=12}
  %or.91 = pred[]{:T(1024)S(6)} or(pred[]{:T(1024)S(6)} %compare.89, pred[]{:T(1024)S(6)} %compare.90), metadata={op_name="or" source_file="<ipython-input-4-4f3bd086a82e>" source_line=12}
  %select.96 = bf16[]{:T(512)} select(pred[]{:T(1024)S(6)} %or.91, bf16[]{:T(512)} %Arg_0.85, bf16[]{:T(512)} %Arg_2.87), metadata={op_name="select_n" source_file="<ipython-input-4-4f3bd086a82e>" source_line=12}
  %or.95 = pred[]{:T(1024)S(6)} or(pred[]{:T(1024)S(6)} %or.91, pred[]{:T(1024)S(6)} %and.94), metadata={op_name="or" source_file="<ipython-input-4-4f3bd086a82e>" source_line=12}
  %select.97 = s32[]{:T(256)} select(pred[]{:T(1024)S(6)} %or.95, s32[]{:T(256)} %Arg_1.86, s32[]{:T(256)} %Arg_3.88), metadata={op_name="select_n" source_file="<ipython-input-4-4f3bd086a82e>" source_line=12}
  ROOT %tuple.98 = (bf16[]{:T(512)}, s32[]{:T(256)}) tuple(bf16[]{:T(512)} %select.96, s32[]{:T(256)} %select.97)
}

%region_0.8 (Arg_0.9: bf16[], Arg_1.10: bf16[]) -> bf16[] {
  %Arg_1.10 = bf16[]{:T(512)} parameter(1)
  %Arg_0.9 = bf16[]{:T(512)} parameter(0)
  ROOT %minimum.11 = bf16[]{:T(512)} minimum(bf16[]{:T(512)} %Arg_0.9, bf16[]{:T(512)} %Arg_1.10), metadata={op_name="jit(ScaMTPUTopK)/jit(main)/jit(ScaMTPUTopK)/jit(jit_ScaMTPUTopK)/reduce_min[axes=(2,)]" source_file="<ipython-input-4-4f3bd086a82e>" source_line=8}
}

ENTRY %main {
  %param_0.3 = bf16[1024,1024,2048]{2,0,1:T(8,128)(2,1)} parameter(0)

  // ArgMin
  %iota.5.clone.1 = s32[1024,1024,2048]{2,0,1:T(8,128)} iota(), iota_dimension=2, metadata={op_name="jit(ScaMTPUTopK)/jit(main)/jit(ScaMTPUTopK)/jit(jit_ScaMTPUTopK)/iota[dtype=int32 shape=(1024, 1024, 2048) dimension=2]" source_file="<ipython-input-4-4f3bd086a82e>" source_line=12}
  %constant.24 = bf16[]{:T(512)} constant(inf)
  %constant.23 = s32[]{:T(256)} constant(0)
  %reduce.3 = (bf16[1024,1024]{0,1:T(8,128)(2,1)}, s32[1024,1024]{0,1:T(8,128)}) reduce(bf16[1024,1024,2048]{2,0,1:T(8,128)(2,1)} %param_0.3, s32[1024,1024,2048]{2,0,1:T(8,128)} %iota.5.clone.1, bf16[]{:T(512)} %constant.24, s32[]{:T(256)} %constant.23), dimensions={2}, to_apply=%region_3.84

  %gte.0 = s32[1024,1024]{0,1:T(8,128)} get-tuple-element(%reduce.3), index=1

  // ReduceMin
  %constant.25 = bf16[]{:T(512)} constant(inf)
  %reduce.4 = bf16[1024,1024]{0,1:T(8,128)(2,1)} reduce(bf16[1024,1024,2048]{2,0,1:T(8,128)(2,1)} %param_0.3, bf16[]{:T(512)} %constant.25), dimensions={2}, to_apply=%region_0.8

  ROOT %tuple.0 = (bf16[1024,1024]{0,1:T(8,128)(2,1)}, s32[1024,1024]{0,1:T(8,128)}) tuple(%reduce.4, %gte.0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  int64_t reduce_count =
      absl::c_count_if(m->entry_computation()->instructions(),
                       HloPredicateIsOp<HloOpcode::kReduce>);
  // Expect one Reduce operation after simplification.
  EXPECT_EQ(1, reduce_count);
  auto variadic_reduce = m::Reduce().WithShape(m::Shape().IsTuple());
  auto root = m->entry_computation()->root_instruction();
  // Expect that both outputs are fed by 'variadic_reduce'.
  ASSERT_THAT(root,
              GmockMatch(m::Tuple(m::GetTupleElement(variadic_reduce, 0),
                                  m::GetTupleElement(variadic_reduce, 1))));
}

TEST_F(AlgebraicSimplifierTest, UnaryVariadicReduceWindow) {
  const char* kModuleStr = R"(
    HloModule m
    fn {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      a = f32[] add(p0, p1)
      ROOT t = (f32[]) tuple(a)
    }
    test {
      p0 = f32[32,8,6,7] parameter(0)
      c = f32[] constant(0)
      ROOT r = (f32[32,8,6,7]) reduce-window(p0, c), to_apply=fn, window={size=1x1x1x1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::ReduceWindow(m::Parameter(0), m::ConstantScalar(0)))));
  ASSERT_EQ(m->entry_computation()
                ->root_instruction()
                ->operand(0)
                ->called_computations()
                .size(),
            1);
  EXPECT_THAT(m->entry_computation()
                  ->root_instruction()
                  ->operand(0)
                  ->called_computations()[0]
                  ->root_instruction(),
              GmockMatch(m::Add(m::Parameter(0), m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, BroadcastAndPadReorder) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      c1 = pred[] constant(true)
      b2 = pred[32,1,768]{2,1,0} broadcast(pred[] c1), dimensions={}
      c3 = pred[] constant(false)
      ROOT p4 = pred[4096,1,768]{2,1,0} pad(pred[32,1,768]{2,1,0} b2, pred[] c3), padding=0_4064x0_0x0_0
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(
                  m::Pad(m::Broadcast(m::Constant()), m::Constant()))));
}

TEST_F(AlgebraicSimplifierTest, BroadcastAndPadReorderWithUse) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      c1 = pred[] constant(true)
      b2 = pred[1,768,32]{2,1,0} broadcast(pred[] c1), dimensions={}
      c3 = pred[] constant(false)
      p4 = pred[1,768,4096]{2,1,0} pad(pred[1,768,32]{2,1,0} b2, pred[] c3), padding=0_0x0_0x0_4064
      ROOT p5 = (pred[1,768,4096]{2,1,0}) tuple(pred[1,768,4096]{2,1,0} p4)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Broadcast(
                  m::Pad(m::Broadcast(m::Constant()), m::Constant())))));
}

TEST_F(AlgebraicSimplifierTest, BroadcastAndPadReorderWithNonScalar) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      c1 = pred[32] parameter(0)
      b2 = pred[1,768,32]{2,1,0} broadcast(pred[32] c1), dimensions={2}
      c3 = pred[] constant(false)
      p4 = pred[1,768,4096]{2,1,0} pad(pred[1,768,32]{2,1,0} b2, pred[] c3), padding=0_0x0_0x0_4064
      ROOT p5 = (pred[1,768,4096]{2,1,0}) tuple(pred[1,768,4096]{2,1,0} p4)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Broadcast(
                  m::Pad(m::Broadcast(m::Parameter()), m::Constant())))));
}

// Test that dynamic-update-slice with a scalar broadcast becomes a pad when the
// start_indices are too big.
TEST_F(AlgebraicSimplifierTest, DynamicUpdateSliceOfBroadcastToPadOob) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  constant.546 = f32[] constant(0)
  broadcast.467 = f32[2]{0} broadcast(constant.546), dimensions={}
  parameter.1 = f32[1]{0} parameter(0)
  constant.551 = s32[] constant(2)
  ROOT dynamic-update-slice.44 = f32[2]{0} dynamic-update-slice(broadcast.467, parameter.1, constant.551)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  VLOG(2) << "Before rewrite dus->pad\n" << module->ToString();
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite dus->pad\n" << module->ToString();
  auto* pad = module->entry_computation()->root_instruction();
  EXPECT_THAT(pad,
              GmockMatch(m::Pad(m::Parameter(0), m::ConstantScalar(0.0f))));
  EXPECT_FALSE(HasInteriorPadding(pad->padding_config()));
  ASSERT_EQ(pad->padding_config().dimensions_size(), 1);
  EXPECT_EQ(pad->padding_config().dimensions(0).edge_padding_low(), 1);
  EXPECT_EQ(pad->padding_config().dimensions(0).edge_padding_high(), 0);
}

// Test folding of dynamic_slice(iota, index) -> clamp(index, 0, size-1)
TEST_F(AlgebraicSimplifierTest, DynamicSliceOfIota) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  %cst = s32[2]{0} constant({0, 1})
  %index = u32[] parameter(0)
  ROOT %dynamic-slice = s32[1]{0} dynamic-slice(s32[2]{0} %cst, u32[] %index),
                                  dynamic_slice_sizes={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite \n" << module->ToString();

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(m::Convert(
                  m::Clamp(m::Constant(), m::Parameter(0), m::Constant())))));
}

// Test of converting dynamic-slice indices to zeros on dims with the full size.
TEST_F(AlgebraicSimplifierTest, DynamicSliceTrivialIndices) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  %operand = s32[2,2] parameter(0)
  %index = u32[] parameter(1)
  ROOT %dynamic-slice = s32[2,1] dynamic-slice(%operand, %index, %index),
    dynamic_slice_sizes={2,1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite \n" << module->ToString();

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicSlice(m::Parameter(0), m::Constant(),
                                         m::Parameter(1))));
}

// Test folding of clamp(pid, 0, limit) -> pid
TEST_F(AlgebraicSimplifierTest, ClampOfPartitionId) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  %pid = u32[] partition-id()
  %low = u32[] constant(0)
  %high = u32[] constant(5)
  ROOT %c = u32[] clamp(%low, %pid, %high)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo_string, /*replica_count=*/1,
                                                /*num_partitions=*/6));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite \n" << module->ToString();

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::PartitionId()));
}

TEST_F(AlgebraicSimplifierTest, ConstantToIota) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  %cst = s32[4] constant({0, 25, 50, 75})
  ROOT %s = s32[4] copy(s32[4] %cst)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite \n" << module->ToString();

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(m::Iota(), m::Broadcast())));
}

TEST_F(AlgebraicSimplifierTest, DynamicSliceOfStridedIota) {
  const char* hlo_string = R"(
HloModule module

ENTRY f {
  %cst = s32[4] constant({0, 25, 50, 75})
  %index = u32[] parameter(0)
  ROOT %dynamic-slice = s32[1]{0} dynamic-slice(s32[4]{0} %cst, u32[] %index),
                                  dynamic_slice_sizes={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());
  VLOG(2) << "After rewrite \n" << module->ToString();

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(
                  m::Multiply(m::Convert(m::Clamp()), m::Constant()))));
}

TEST_F(AlgebraicSimplifierTest, AbsEliminationSelMaxBcast) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[32]{0} parameter(0)
      p1 = pred[32]{0} parameter(1)
      zero = f32[] constant(0.0)
      bcast = f32[32] broadcast(zero), dimensions={}
      m = f32[32]{0} maximum(p0, bcast)
      s = f32[32]{0} select(p1, bcast, m)
      ROOT a = f32[32]{0} abs(s)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Select(
                  m::Parameter(1), m::Broadcast(m::ConstantScalar()),
                  m::MaximumAnyOrder(m::Parameter(0),
                                     m::Broadcast(m::ConstantScalar())))));
}

TEST_F(AlgebraicSimplifierTest, AbsEliminationIota) {
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(R"(
    e {
      i = s32[3,2] iota(), iota_dimension=0
      ROOT a = s32[3,2] abs(i)
    }
  )"));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Iota()));
}

TEST_F(AlgebraicSimplifierTest, SimplifyRedundantBitcastConvert) {
  const char* kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = s32[10] parameter(0)
      p1 = s32[10] parameter(1)
      b0 = u32[10] bitcast-convert(p0)
      b1 = u32[10] bitcast-convert(p1)
      c = u32[20] concatenate(b0, b1), dimensions={0}
      ROOT out = s32[20] bitcast-convert(c)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Concatenate(m::Parameter(0), m::Parameter(1))));
}

TEST_F(AlgebraicSimplifierTest, SimplifyTautologicalBitcastConvert) {
  const char* kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = bf16[10] parameter(0)
      ROOT out = bf16[10] bitcast-convert(p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(AlgebraicSimplifierTest, SimplifyBitcastConvertChain) {
  const char* kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = s16[10] parameter(0)
      b1 = bf16[10] bitcast-convert(p0)
      b2 = u16[10] bitcast-convert(b1)
      b3 = f16[10] bitcast-convert(b2)
      ROOT out = s16[10] bitcast-convert(b3)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::BitcastConvert(m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest,
       DoNotSimplifyRedundantBitcastConvertWithControlDep) {
  const char* kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = s32[10] parameter(0)
      p1 = s32[10] parameter(1)
      p2 = s32[10] parameter(2)
      p3 = s32[10] parameter(3)
      add0 = s32[10] add(p2, p3)
      b0 = s32[1, 10] bitcast(p0)
      ROOT b1 = s32[1, 1, 10] bitcast(b0), control-predecessors={add0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  // b1 has a control dep, we don't expect the graph to change.
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, SimplifyOptimizationBarrier) {
  const char* kModuleStr = R"(
    HloModule m

    ENTRY entry {
      param.0 = f32[] parameter(0)
      param.1 = f32[] parameter(1)
      add.0 = f32[] add(param.0, param.1)
      sub.0 = f32[] subtract(param.0, param.1)
      mul.0 = f32[] multiply(param.0, param.1)
      tuple.0 = (f32[], f32[], f32[]) tuple(mul.0, sub.0, add.0)
      b = (f32[], f32[], f32[]) opt-barrier(tuple.0)
      gte.0 = f32[] get-tuple-element(b), index=1
      ROOT  t = (f32[], f32[]) tuple(mul.0,gte.0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  EXPECT_EQ(m->entry_computation()
                ->root_instruction()
                ->operand(1)
                ->operand(0)
                ->operand(0)
                ->operand_count(),
            3);
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_EQ(m->entry_computation()
                ->root_instruction()
                ->operand(1)
                ->operand(0)
                ->operand(0)
                ->operand_count(),
            2);
}

TEST_F(AlgebraicSimplifierTest, GTETupleShardingLoss) {
  // Verify the gte(tuple) folding does not happen if it loses sharding info.
  const char* kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = s32[10] parameter(0), sharding={devices=[2]0,1}
      t = (s32[10]) tuple(p0)
      ROOT %gte = s32[10] get-tuple-element(t), index=0, sharding={replicated}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, DynamicSliceShapeLayout) {
  // Verify we maintain layout when optimizing dynamic-slice
  const char* kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = u32[]{:T(128)} parameter(0)
      %constant.1 = s32[4]{0:T(128)} constant({0, 16, 32, 48})
      %dynamic-slice = s32[1]{0:T(128)} dynamic-slice(s32[4]{0:T(128)} %constant.1, u32[] %p0), dynamic_slice_sizes={1}
      ROOT t = (s32[1]{0:T(128)}) tuple(dynamic-slice)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  const Shape& slice_shape =
      m->entry_computation()->root_instruction()->operand(0)->shape();
  EXPECT_TRUE(slice_shape.has_layout());
  EXPECT_EQ(slice_shape.layout().tiles_size(), 1);
}

// Fold a sequence of copy bitcast copy
TEST_F(AlgebraicSimplifierTest, CopyBitcastCopy) {
  const char* kModuleStr = R"(
    HloModule m

    ENTRY test {
     fusion.1235 = bf16[1600,50,512]{2,0,1:T(8,128)(2,1)} parameter(0)
     copy.3038 = bf16[1600,50,512]{0,2,1:T(8,128)(2,1)} copy(fusion.1235)
     bitcast.8 = bf16[1600,50,16,32]{0,3,2,1:T(8,128)(2,1)} bitcast(copy.3038)
     copy.3045 = bf16[1600,50,16,32]{1,3,2,0:T(8,128)(2,1)} copy(bitcast.8)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Bitcast(m::Copy(m::Parameter()))));
}

TEST_F(AlgebraicSimplifierTest, CopyBitcastCopyDimSize1) {
  const char* kModuleStr = R"(
    HloModule m

    ENTRY test {
     param.8 = f32[9, 1, 12]{2,1,0} parameter(0)
     transpose.1 = f32[1,12,9]{1,0,2} transpose(param.8), dimensions={1,2,0}
     copy.4 = f32[1,12,9]{2,1,0} copy(transpose.1)
     bitcast.15 = f32[1,108]{1,0} bitcast(copy.4)
     copy.1 = f32[1,108]{0,1} copy(bitcast.15)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(m::Bitcast(m::Bitcast(m::Copy(m::Bitcast(m::Parameter()))))));
}

TEST_F(AlgebraicSimplifierTest, CopyBitcastCopy2) {
  const char* kModuleStr = R"(
    HloModule m

    ENTRY test {
     %Arg_0.1 = f32[8,3,3,7,7]{4,0,3,2,1:T(8,128)} parameter(0)
     %copy.1 = f32[8,3,3,7,7]{4,3,2,1,0:T(8,128)} copy(f32[8,3,3,7,7]{4,0,3,2,1:T(8,128)} %Arg_0.1)
     %bitcast = f32[1,72,7,7]{3,2,1,0:T(8,128)} bitcast(f32[8,3,3,7,7]{4,3,2,1,0:T(8,128)} %copy.1)
     %copy.2 = f32[1,72,7,7]{1,3,2,0:T(8,128)} copy(f32[1,72,7,7]{3,2,1,0:T(8,128)} %bitcast)
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_FALSE(simplifier.Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, CopyReshapeCopy3) {
  const char* kModuleStr = R"(
   HloModule m

  ENTRY main {
  p = f32[2,3]{0,1} parameter(0)
  copy = f32[2,3]{1,0} copy(p)
  reshape = f32[3,2]{1,0} bitcast(copy)
  ROOT copy.1 = f32[3,2]{0,1} copy(reshape)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_FALSE(simplifier.Run(m.get()).value());
  VLOG(3) << "Module " << m->ToString();
}

TEST_F(AlgebraicSimplifierTest, CopyReshapeCopy4) {
  const char* kModuleStr = R"(
   HloModule m

  ENTRY main {
    p = f32[6,2,3]{0,1,2} parameter(0)
    copy.0 = f32[6,2,3]{0,2,1} copy(p)
    reshape = f32[2,3,6]{1,0,2} bitcast(copy.0)
    ROOT copy.1 = f32[2,3,6]{0,1,2} copy(reshape)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  ASSERT_FALSE(simplifier.Run(m.get()).value());
  VLOG(3) << "Module " << m->ToString();
}

TEST_F(AlgebraicSimplifierTest, BitcastCopyChain) {
  const char* kModuleStr = R"(
   HloModule m

  ENTRY main {
   p.0 = f32[4,32,32,1]{2,3,1,0} parameter(0)
   reshape.30 = f32[4,32,1,1,32]{4,1,0,3,2} reshape(p.0)
   transpose.1757 = f32[4,1,1,32,32]{3,4,0,1,2} transpose(reshape.30), dimensions={0,3,2,4,1}
   copy.3 = f32[4,1,1,32,32]{4,3,0,2,1} copy(transpose.1757)
   reshape.1758 = f32[4,1,1,1024]{3,2,1,0} reshape(copy.3)
   transpose.61 = f32[1024,4,1,1]{0,3,2,1} transpose(reshape.1758), dimensions={3,0,1,2}
   copy.4 = f32[1024,4,1,1]{0,1,3,2} copy(transpose.61)
   ROOT reshape.107 = f32[1024,4]{0,1} reshape(copy.4)
   })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  auto result = simplifier.Run(m.get()).value();
  SCOPED_TRACE(m->ToString());
  ASSERT_TRUE(result);
  EXPECT_NE(FindInstruction(m.get(), "copy.3"), nullptr);
}

// Make sure that the following copy-bitcast-copy is not transformed via
// SwapCopyBitcastCopy function. If SwapCopyBitcastCopy does not fire, in this
// case, the last copy will be turned into a bitcast by HandleCopy.
TEST_F(AlgebraicSimplifierTest, BitcastCopyChainSmall) {
  const char* kModuleStr = R"(
   HloModule m
   ENTRY %main (para.0: f32[4,1,1,32,32]) -> f32[1024,4,1,1] {
    %para.0 = f32[4,1,1,32,32]{3,4,0,1,2} parameter(0)
    %copy.0 = f32[4,1,1,32,32]{4,3,0,2,1} copy(%para.0)
    %bitcast.0 = f32[1024,4,1,1]{0,3,2,1} bitcast(%copy.0)
    ROOT %copy.1 = f32[1024,4,1,1]{0,1,3,2} copy(%bitcast.0)
  }

  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  SCOPED_TRACE(m->ToString());
  auto result = simplifier.Run(m.get()).value();
  SCOPED_TRACE(m->ToString());
  ASSERT_TRUE(result);
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Bitcast(m::Bitcast(m::Copy(m::Parameter(0))))));
}

TEST_F(AlgebraicSimplifierTest, BitcastUndoesBitcast) {
  const char* kModuleStr = R"(
   HloModule m
   ENTRY %main (p0: f32[32]) -> f32[1, 32, 32] {
    %p0 = f32[32]{0} parameter(0)
    %bitcast.0 = f32[1,32]{1, 0} bitcast(%p0)
    %bitcast.1 = f32[32]{0} bitcast(%bitcast.0)
    ROOT %broadcast = f32[1, 32, 32]{2, 1, 0} broadcast(%bitcast.1), dimensions={1}
  }

  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  SCOPED_TRACE(m->ToString());
  auto result = simplifier.Run(m.get()).value();
  SCOPED_TRACE(m->ToString());
  ASSERT_TRUE(result);
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Parameter(0))));
}

// Reverse(Reverse(A)) ==> A.
TEST_F(AlgebraicSimplifierTest, RemoveIdenticalNestedReverse) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = bf16[100,1,100,512] parameter(0)
      r0 = bf16[100,1,100,512] reverse(p0), dimensions={1,2}
      ROOT r1 = bf16[100,1,100,512] reverse(r0), dimensions={1,2}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  SCOPED_TRACE("Before rewrite\n" + m->ToString());
  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  auto g = simplifier.Run(m.get()).value();
  SCOPED_TRACE("After rewrite\n" + m->ToString());
  ASSERT_TRUE(g);
  auto* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter(0)));
}

// Reverse(Reverse(A)) ==> Reverse(A).
TEST_F(AlgebraicSimplifierTest, ShrinkNestedReverse) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = bf16[100,1,100,512] parameter(0)
      r0 = bf16[100,1,100,512] reverse(p0), dimensions={1,2,3}
      ROOT r1 = bf16[100,1,100,512] reverse(r0), dimensions={1,2}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  SCOPED_TRACE("Before rewrite\n" + m->ToString());
  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  auto g = simplifier.Run(m.get()).value();
  SCOPED_TRACE("After rewrite\n" + m->ToString());
  ASSERT_TRUE(g);
  auto* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Reverse(m::Parameter(0))));
}

// reverse(ElementWiseBinOp(x, constant)) ==> ElementWiseBinOp(reverse(x),
// constant)
TEST_F(AlgebraicSimplifierTest, SwapConstantEwboWithReverse) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = bf16[100,1,100,512] parameter(0)
      constant0 = bf16[] constant(1)
      const = bf16[100,1,100,512] broadcast(constant0), dimensions={}
      ewbo0 = bf16[100,1,100,512] add(p0, const)
      ROOT r0 = bf16[100,1,100,512] reverse(ewbo0), dimensions={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  SCOPED_TRACE("Before rewrite\n" + m->ToString());
  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  auto g = simplifier.Run(m.get()).value();
  SCOPED_TRACE("After rewrite\n" + m->ToString());
  ASSERT_TRUE(g);
  auto* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Add(m::Reverse(m::Parameter(0)),
                                      m::Broadcast(m::Constant()))));
}

// reverse(ElementWiseBinOp(constant, x)) ==> ElementWiseBinOp(constant,
// reverse(x))
TEST_F(AlgebraicSimplifierTest, SwapConstantEwboWithReverse2) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = bf16[100,1,100,512] parameter(0)
      constant0 = bf16[] constant(1)
      const = bf16[100,1,100,512] broadcast(constant0), dimensions={}
      ewbo0 = bf16[100,1,100,512] add(const, p0)
      ROOT r0 = bf16[100,1,100,512] reverse(ewbo0), dimensions={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  SCOPED_TRACE("Before rewrite\n" + m->ToString());
  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  auto g = simplifier.Run(m.get()).value();
  SCOPED_TRACE("After rewrite\n" + m->ToString());
  ASSERT_TRUE(g);
  auto* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Add(m::Broadcast(m::Constant()),
                                      m::Reverse(m::Parameter(0)))));
}

TEST_F(AlgebraicSimplifierTest, SquaredComplexSqrtIsFloat) {
  const char* const kModuleStr = R"(
  HloModule module

  ENTRY entry {
    arg = c64[7]{0} parameter(0)
    multiply = c64[7]{0} multiply(arg, arg)
    ROOT sqrt = c64[7]{0} sqrt(multiply)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  SCOPED_TRACE("Before rewrite\n" + m->ToString());
  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  auto g = simplifier.Run(m.get()).value();
  SCOPED_TRACE("After rewrite\n" + m->ToString());
  ASSERT_TRUE(g);
  auto* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Convert(m::Abs(m::Parameter(0)))));
}

// Don't replace root instruction with the copy-to-operand optimization if
// sharding is applied.
TEST_F(AlgebraicSimplifierTest, RootCopySharding) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = bf16[8] parameter(0)
      ROOT r0 = bf16[8] copy(p0), sharding={devices=[8]0,1,2,3,4,5,6,7}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  SCOPED_TRACE("Before rewrite\n" + m->ToString());
  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  auto returned = simplifier.Run(m.get()).value();
  SCOPED_TRACE("After rewrite\n" + m->ToString());
  ASSERT_FALSE(returned);
  auto* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Copy(m::Parameter(0))));
}

// non-contiguous degenerate dimensions adding
// reverse(DegenerateDimensionAddingReshape(x)) ==>
// DegenerateDimensionAddingReshape(reverse(x))
TEST_F(AlgebraicSimplifierTest,
       SwapNonContiguousDegenerateDimensionAddingReshapeAndReverse) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {3, 4, 5}), "param0"));

  HloInstruction* reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {1, 3, 1, 4, 1, 5}), param0));
  std::vector<int64_t> dims{0, 1, 3, 4};
  absl::Span<const int64_t> rev_dims = absl::MakeSpan(dims);
  builder.AddInstruction(HloInstruction::CreateReverse(
      ShapeUtil::MakeShape(F32, {1, 3, 1, 4, 1, 5}), reshape1, rev_dims));
  SCOPED_TRACE("Before rewrite\n" + m->ToString());
  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reverse(m::Reshape(m::Parameter(0)))));
  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  auto g = simplifier.Run(m.get()).value();
  ASSERT_TRUE(g);
  SCOPED_TRACE("After rewrite\n" + m->ToString());
  std::vector<int64_t> after_rewrite_dims{0, 1};
  absl::Span<const int64_t> after_rewrite_rev_dims =
      absl::MakeSpan(after_rewrite_dims);
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Reverse())));
  EXPECT_THAT(computation->root_instruction()->operand(0)->dimensions(),
              after_rewrite_rev_dims);
}

// contiguous degenerate dimensions adding
// reverse(DegenerateDimensionAddingReshape(x)) ==>
// DegenerateDimensionAddingReshape(reverse(x))
TEST_F(AlgebraicSimplifierTest,
       SwapContiguousDegenerateDimensionAddingReshapeAndReverse) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(F32, {3, 4, 5}), "param0"));

  HloInstruction* reshape1 =
      builder.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {3, 1, 1, 1, 4, 1, 5}), param0));
  std::vector<int64_t> dims{0, 1, 3, 4};
  absl::Span<const int64_t> rev_dims = absl::MakeSpan(dims);
  builder.AddInstruction(HloInstruction::CreateReverse(
      ShapeUtil::MakeShape(F32, {3, 1, 1, 1, 4, 1, 5}), reshape1, rev_dims));
  SCOPED_TRACE("Before rewrite\n" + m->ToString());
  auto computation = m->AddEntryComputationWithLayouts(builder.Build());
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reverse(m::Reshape(m::Parameter(0)))));
  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  auto g = simplifier.Run(m.get()).value();
  ASSERT_TRUE(g);
  SCOPED_TRACE("After rewrite\n" + m->ToString());
  std::vector<int64_t> after_rewrite_dims{0, 1};
  absl::Span<const int64_t> after_rewrite_rev_dims =
      absl::MakeSpan(after_rewrite_dims);
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Reshape(m::Reverse())));
  EXPECT_THAT(computation->root_instruction()->operand(0)->dimensions(),
              after_rewrite_rev_dims);
}

// Make sure the optimization for reshape(dynamic-update-slice) does not more
// forward if the dus has multiple users.
TEST_F(AlgebraicSimplifierTest, ReshapeOfDupDoNotCloneMultiUserDup) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[128,1184,1,128]{3,2,1,0} parameter(0)
      p1 = f32[128,1,1,128]{3,2,1,0} parameter(1)
      p2 = s32[] parameter(2)
      constant.6030 = s32[] constant(0)
      dynamic-update-slice.1854 = f32[128,1184,1,128]{3,2,1,0} dynamic-update-slice(p0, p1, constant.6030, p2, constant.6030, constant.6030)
      reshape.33672 = f32[128,1,1184,128]{3,1,2,0} reshape(dynamic-update-slice.1854)
      ROOT tuple.0 = (f32[128,1,1184,128]{3,1,2,0}, f32[128,1184,1,128]{3,2,1,0}) tuple(reshape.33672, dynamic-update-slice.1854)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  SCOPED_TRACE("Before rewrite\n" + m->ToString());
  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  auto g = simplifier.Run(m.get()).value();
  SCOPED_TRACE("After rewrite\n" + m->ToString());
  ASSERT_FALSE(g);
}

TEST_F(AlgebraicSimplifierTest, MultiplyOfConvertedPred) {
  const char* kModuleStr = R"(
   HloModule m
   test {
     p = pred[2,2]{0,1} parameter(0)
     convert = f32[2,2]{0,1} convert(p)
     p2 = f32[2,2]{0,1} parameter(1)
     ROOT multiply = f32[2,2]{0,1} multiply(p2, convert)
   }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  SCOPED_TRACE("Before rewrite\n" + m->ToString());
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  auto g = simplifier.Run(m.get()).value();
  SCOPED_TRACE("After rewrite\n" + m->ToString());
  ASSERT_TRUE(g);
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Select(m::Parameter(0), m::Parameter(1),
                                   m::Broadcast(m::ConstantScalar(0)))));
  // Also run the HloVerifier on the resulting module to check that the
  // generated instructions don't have an invalid layout change now.
  EXPECT_TRUE(verifier().Run(m.get()).status().ok());
}

TEST_F(AlgebraicSimplifierTest, TransposeOfBroadcast) {
  const char* kModuleStr = R"(
   HloModule m
   test {
     bcast = f32[10,2,3,4] broadcast(f32[2,4] parameter(0)), dimensions={1,3}
     ROOT trans = f32[2,3,10,4] transpose(bcast), dimensions={1,2,0,3}
   }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  EXPECT_TRUE(
      RunHloPass(AlgebraicSimplifier(default_options_), m.get()).value());
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Parameter(0))
                             .WithPredicate([](const HloInstruction* instr) {
                               return instr->dimensions() ==
                                      std::vector<int64_t>({0, 3});
                             })));
}

TEST_F(AlgebraicSimplifierTest, TransposeBitcastOfBroadcast) {
  const char* kModuleStr = R"(
   HloModule m
   test {
     bcast = f32[10,2,3,4]{3,2,1,0} broadcast(f32[2,4]{1,0} parameter(0)), dimensions={1,3}
     ROOT trans = f32[2,3,10,4]{3,1,0,2} bitcast(bcast)
   }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  EXPECT_TRUE(RunHloPass(AlgebraicSimplifier(options), m.get()).value());
  SCOPED_TRACE(m->ToString());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Parameter(0))
                             .WithPredicate([](const HloInstruction* instr) {
                               return instr->dimensions() ==
                                      std::vector<int64_t>({0, 3});
                             })));
}

TEST_F(AlgebraicSimplifierTest, TransposeOfBroadcastWithLayoutCheckSkipped) {
  const char* kModuleStr = R"(
   HloModule m
   test {
     bcast = f32[10,2,3,4]{3,2,1,0} broadcast(f32[2,4]{1,0} parameter(0)), dimensions={1,3}
     ROOT trans = f32[2,3,10,4]{0,1,2,3} transpose(bcast), dimensions={1,2,0,3}
   }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  EXPECT_FALSE(RunHloPass(AlgebraicSimplifier(options), m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, TransposeOfBroadcastSkipped) {
  const char* kModuleStr = R"(
   HloModule m
   test {
     bcast = f32[10,2,3,4] broadcast(f32[2,4] parameter(0)), dimensions={1,3}
     ROOT trans = f32[4,2,3,10] transpose(bcast), dimensions={3,1,2,0}
   }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  bool changed =
      RunHloPass(AlgebraicSimplifier(default_options_), m.get()).value();
  SCOPED_TRACE(m->ToString());
  EXPECT_FALSE(changed);
}

TEST_F(AlgebraicSimplifierTest, DontSinkInstructionsInDUSAsyncComputation) {
  const char* kModuleStr = R"(
   HloModule m
   test {
     %param_0 = f32[1]{0} parameter(0)
     %param_1 = f32[10]{0} parameter(1)
     %constant_1 = s32[] constant(0)
     %dynamic-update-slice-start = ((f32[10]{0}, f32[1]{0}, s32[]),
      f32[10]{0}, u32[]) dynamic-update-slice-start(f32[10]{0} %param_1,
     f32[1]{0} %param_0, s32[] %constant_1)
     ROOT %dynamic-update-slice-done =
     f32[10]{0} dynamic-update-slice-done(((f32[10]{0}, f32[1]{0}, s32[]),
     f32[10]{0}, u32[]) %dynamic-update-slice-start)
   }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  bool changed =
      RunHloPass(AlgebraicSimplifier(default_options_), m.get()).value();
  SCOPED_TRACE(m->ToString());
  EXPECT_FALSE(changed);
}

TEST_F(AlgebraicSimplifierTest, DontSinkInstructionsInDSAsyncComputation) {
  const char* kModuleStr = R"(
   HloModule m
   test {
     %param_0 = f32[10]{0} parameter(0)
     %constant_1 = s32[] constant(0)
     %dynamic-slice-start = ((f32[10]{0}, s32[]), f32[1]{0}, u32[])
      dynamic-slice-start(f32[10]{0} %param_0, s32[] %constant_1),
      dynamic_slice_sizes={1}
     ROOT %dynamic-slice-done = f32[1]{0}
      dynamic-slice-done(((f32[10]{0}, s32[]), f32[1]{0}, u32[])
        %dynamic-slice-start)
   }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  bool changed =
      RunHloPass(AlgebraicSimplifier(default_options_), m.get()).value();
  SCOPED_TRACE(m->ToString());
  EXPECT_FALSE(changed);
}

TEST_F(AlgebraicSimplifierTest, NoOpSliceToDynamicOfPadToStatic) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[<=512] parameter(0)
      c = (f32[512], s32[]) custom-call(p0), custom_call_target="PadToStatic"
      gte0 = f32[512] get-tuple-element(c), index=0
      gte1 = s32[] get-tuple-element(c), index=1
      ROOT c2 = f32[<=512] custom-call(gte0, gte1), custom_call_target="SliceToDynamic"
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(AlgebraicSimplifierTest, DiffShapeSliceToDynamicOfPadToStatic) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[<=512] parameter(0)
      c = (f32[512], s32[]) custom-call(p0), custom_call_target="PadToStatic"
      gte0 = f32[512] get-tuple-element(c), index=0
      gte1 = s32[] get-tuple-element(c), index=1
      ROOT c2 = f32[<=1024] custom-call(gte0, gte1), custom_call_target="SliceToDynamic"
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, DiffShapeSliceToDynamicDifferentPadToStatic) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[<=512] parameter(0)
      c = (f32[512], s32[]) custom-call(p0), custom_call_target="PadToStatic"
      p1 = f32[<=512] parameter(1)
      c1 = (f32[512], s32[]) custom-call(p1), custom_call_target="PadToStatic"
      gte0 = f32[512] get-tuple-element(c), index=0
      gte1 = s32[] get-tuple-element(c1), index=1
      ROOT c2 = f32[<=512] custom-call(gte0, gte1), custom_call_target="SliceToDynamic"
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, NotPadToStaticSizeDynamicDifferentPadToStatic) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[<=512] parameter(0)
      c = (f32[512], s32[]) custom-call(p0), custom_call_target="PadToStatic"
      gte0 = f32[512] get-tuple-element(c), index=0
      gte1 = s32[] parameter(1)
      ROOT c2 = f32[<=512] custom-call(gte0, gte1), custom_call_target="SliceToDynamic"
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}
class AlgebraicSimplifierUpcastDowncastTest
    : public AlgebraicSimplifierTest,
      public ::testing::WithParamInterface<
          std::tuple<HloOpcode, PrimitiveType, PrimitiveType, bool>> {
 public:
  AlgebraicSimplifierUpcastDowncastTest()
      : AlgebraicSimplifierTest(),
        binary_opcode_(std::get<0>(GetParam())),
        outer_type_(std::get<1>(GetParam())),
        binary_op_type_(std::get<2>(GetParam())),
        should_rewrite_(std::get<3>(GetParam())) {}

 protected:
  const HloOpcode binary_opcode_;
  const PrimitiveType outer_type_;
  const PrimitiveType binary_op_type_;
  const bool should_rewrite_;
};

TEST_P(AlgebraicSimplifierUpcastDowncastTest,
       CheckUpcastingAndDowncastingConvertsAreRemoved) {
  const std::string& src_type_str =
      primitive_util::LowercasePrimitiveTypeName(outer_type_);
  const std::string& dest_type_str =
      primitive_util::LowercasePrimitiveTypeName(binary_op_type_);
  absl::string_view op_str = HloOpcodeString(binary_opcode_);
  const std::string kModuleStr =
      absl::StrFormat(R"(
    HloModule m
    test {
      p1 = %s[] parameter(0)
      p2 = %s[] parameter(1)
      c1 = %s[] convert(p1)
      c2 = %s[] convert(p2)
      res = %s[] %s(c1, c2)
      ROOT cres = %s[] convert(res)
    }
  )",
                      src_type_str, src_type_str, dest_type_str, dest_type_str,
                      dest_type_str, op_str, src_type_str);
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, RunHloPass(AlgebraicSimplifier(default_options_), m.get()));
  SCOPED_TRACE(m->ToString());
  EXPECT_EQ(changed, should_rewrite_);
  if (should_rewrite_) {
    EXPECT_THAT(
        m->entry_computation()->root_instruction(),
        GmockMatch(
            m::Op()
                .WithOpcode(binary_opcode_)
                .WithOperand(0, m::Parameter(0).WithElementType(outer_type_))
                .WithOperand(1, m::Parameter(1).WithElementType(outer_type_))));
  }
}

std::vector<std::tuple<HloOpcode, PrimitiveType, PrimitiveType, bool>>
GetUpcastDowncastTestCases() {
  std::vector<std::tuple<HloOpcode, PrimitiveType, PrimitiveType, bool>> result;
  const std::vector<PrimitiveType> types = {
      S8, S16, S32, S64, U8, U16, U32, U64, F16, F32, BF16, F64, C64, C128};
  for (const auto op :
       {HloOpcode::kAdd, HloOpcode::kSubtract, HloOpcode::kMultiply,
        HloOpcode::kDivide, HloOpcode::kRemainder}) {
    for (const auto original_type : types) {
      for (const auto upcast_type : types) {
        const bool should_rewrite = [&] {
          if (original_type == upcast_type) {
            // Even though the function we're targeting does not support certain
            // types, something else in AlgebraicSimplifier will handle this
            // case.
            return true;
          }
          if ((primitive_util::IsSignedIntegralType(original_type) !=
               primitive_util::IsSignedIntegralType(upcast_type)) ||
              (primitive_util::IsUnsignedIntegralType(original_type) !=
               primitive_util::IsUnsignedIntegralType(upcast_type)) ||
              (primitive_util::IsFloatingPointType(original_type) !=
               primitive_util::IsFloatingPointType(upcast_type)) ||
              (primitive_util::IsComplexType(original_type) !=
               primitive_util::IsComplexType(upcast_type))) {
            // Not yet handling conversions from one class of types to another
            // class of types (ex. integer to floating point).
            return false;
          }
          if (primitive_util::IsComplexType(original_type) ||
              primitive_util::IsComplexType(upcast_type)) {
            // Not yet handling complex types.
            return false;
          }
          if (primitive_util::IsFloatingPointType(original_type) ||
              primitive_util::IsFloatingPointType(upcast_type)) {
            // Not yet handling floating point types.
            return false;
          }
          if (!primitive_util::CastPreservesValues(original_type,
                                                   upcast_type)) {
            // We are looking for upcast->bin_op->downcast; this is the opposite
            // direction.
            return false;
          }
          if (op == HloOpcode::kDivide || op == HloOpcode::kRemainder) {
            if (primitive_util::IsSignedIntegralType(original_type)) {
              // This transformation is not safe for divide or remainder with
              // signed integers.
              return false;
            }
          }
          return true;
        }();
        result.emplace_back(op, original_type, upcast_type, should_rewrite);
      }
    }
  }
  return result;
}

INSTANTIATE_TEST_SUITE_P(AllTypes, AlgebraicSimplifierUpcastDowncastTest,
                         ::testing::ValuesIn(GetUpcastDowncastTestCases()));

template <typename Arg0, typename Arg1, typename Arg2>
auto SparseDotMatcher(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2) {
  return match::Op()
      .WithOpcode(HloOpcode::kDot)
      .WithOperand(0, std::forward<Arg0>(arg0))
      .WithOperand(1, std::forward<Arg1>(arg1))
      .WithOperand(2, std::forward<Arg2>(arg2));
}

TEST_F(AlgebraicSimplifierTest, SparseDotRemoveDegenerateDimensions) {
  const char* kHlo = R"(
    HloModule m
    ENTRY test {
      %lhs = f32[1,5,10,16,1] parameter(0)
      %rhs = f32[5,1,20,1,32] parameter(1)
      %meta = u16[1,5,10,2,1] parameter(2)
      ROOT %dot = f32[1,5,10,20] dot(%lhs, %rhs, %meta),
          lhs_batch_dims={0,1}, rhs_batch_dims={1,0},
          lhs_contracting_dims={3,4}, rhs_contracting_dims={4,3},
          sparsity=L.3@2:4
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(module.get()).value());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, GmockMatch(m::Reshape(SparseDotMatcher(m::Reshape(m::Parameter(0)),
                                                   m::Reshape(m::Parameter(1)),
                                                   m::Reshape(m::Parameter(2)))
                                      .WithShape(F32, {5, 10, 20}))));
  auto dot = Cast<HloDotInstruction>(root->operand(0));
  auto descriptor = dot->sparsity().front();
  EXPECT_EQ(descriptor.index(), 0);
  EXPECT_EQ(descriptor.dimension(), 2);
}

TEST_F(AlgebraicSimplifierTest, SparseDotMoveSliceToOperands) {
  const char* kHlo = R"(
    HloModule m
    ENTRY test {
      %lhs = f32[7,12,16] parameter(0)
      %rhs = f32[7,22,32] parameter(1)
      %meta = u16[7,12,2] parameter(2)
      %dot = f32[7,12,22] dot(%lhs, %rhs, %meta),
          lhs_batch_dims={0}, rhs_batch_dims={0},
          lhs_contracting_dims={2}, rhs_contracting_dims={2},
          sparsity=L.2@2:4
      ROOT %slice = f32[5,10,20] slice(%dot), slice={[0:5], [0:10], [0:20]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  AlgebraicSimplifierOptions options;
  options.set_raise_slice_and_reduce_through_dot(true);
  ASSERT_TRUE(AlgebraicSimplifier(options).Run(module.get()).value());
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(SparseDotMatcher(m::Slice(m::Parameter(0)),
                                                m::Slice(m::Parameter(1)),
                                                m::Slice(m::Parameter(2)))
                                   .WithShape(F32, {5, 10, 20})));
  auto dot = Cast<HloDotInstruction>(root);
  auto descriptor = dot->sparsity().front();
  EXPECT_EQ(descriptor.index(), 0);
  EXPECT_EQ(descriptor.dimension(), 2);
}

TEST_F(AlgebraicSimplifierTest, SparseDotKeepTranspose) {
  const char* hlo_string = R"(
    HloModule m
    ENTRY test {
      %lhs = f32[10,16] parameter(0)
      %rhs = f32[32,20] parameter(1)
      %meta = u16[10,2] parameter(2)
      %dot = f32[10,20] dot(%lhs, %rhs, %meta),
          lhs_contracting_dims={1}, rhs_contracting_dims={0},
          sparsity=L.1@2:4
      ROOT %transpose = f32[20,10] transpose(%dot), dimensions={1,0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto options = AlgebraicSimplifierOptions();

  options.set_supports_non_canonical_dots(false);
  AlgebraicSimplifier simplifier1(options);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&simplifier1, module.get()));
  EXPECT_FALSE(changed);

  options.set_supports_non_canonical_dots(true);
  AlgebraicSimplifier simplifier2(options);
  TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&simplifier2, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AlgebraicSimplifierTest, SparseDotKeepOperandsTranspose) {
  const char* hlo_string = R"(
    HloModule m
    ENTRY test {
      %lhs = f32[10,20,30,16] parameter(0)
      %rhs = f32[10,20,32,40] parameter(1)
      %lhs_t = f32[20,10,30,16] transpose(%lhs), dimensions={1,0,2,3}
      %rhs_t = f32[20,10,32,40] transpose(%rhs), dimensions={1,0,2,3}
      %meta = u16[20,10,30,2] parameter(2)
      ROOT %root = dot(%lhs_t, %rhs_t, %meta),
          lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
          lhs_contracting_dims={3}, rhs_contracting_dims={2}, sparsity=L.3@2:4
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifier simplifier(default_options_);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&simplifier, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AlgebraicSimplifierTest, SparseDotNoAssociativeReorderOuter) {
  const char* hlo_string = R"(
    HloModule m
    ENTRY test {
      %a = f32[10,5] parameter(0)
      %b = f32[5,32] parameter(1)
      %c = f32[64,20] parameter(2)
      %meta = u16[10,4] parameter(3)
      %inner = f32[10,32] dot(%a, %b),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT %outer = f32[10,20] dot(%inner, %c, %meta),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=L.1@2:4
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_use_associative_reordering(true);
  options.set_associative_reordering_threshold(0);
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(module.get()).value());
}

TEST_F(AlgebraicSimplifierTest, SparseDotNoAssociativeReorderInner) {
  const char* hlo_string = R"(
    HloModule m
    ENTRY test {
      %a = f32[10,64] parameter(0)
      %b = f32[128,32] parameter(1)
      %c = f32[32,20] parameter(2)
      %meta = u16[10,8] parameter(3)
      %inner = f32[10,32] dot(%a, %b, %meta),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=L.1@2:4
      ROOT %outer = f32[10,20] dot(%inner, %c),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_use_associative_reordering(true);
  options.set_associative_reordering_threshold(0);
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(module.get()).value());
}

TEST_F(AlgebraicSimplifierTest, SparseDotNoAssociativeReorderReduce) {
  const char* hlo_string = R"(
    HloModule m
    add {
      %p0 = f32[] parameter(0)
      %p1 = f32[] parameter(1)
      ROOT %add = f32[] add(p0, p1)
    }
    ENTRY test {
      %a = f32[10,16] parameter(0)
      %b = f32[32,20] parameter(1)
      %meta = u16[10,2] parameter(2)
      %dot = f32[10,20] dot(%a, %b, %meta),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=L.1@2:4
      %c = f32[] constant(0)
      ROOT %reduce = f32[10] reduce(%dot, %c), dimensions={1}, to_apply=add
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_use_associative_reordering(true);
  options.set_associative_reordering_threshold(0);
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(module.get()).value());
}

TEST_F(AlgebraicSimplifierTest, SparseDotNoAssociativeReorderOther) {
  const char* hlo_string = R"(
    HloModule m
    ENTRY test {
      %a = f32[10,16] parameter(0)
      %b = f32[32,20] parameter(1)
      %meta = u16[10,2] parameter(2)
      %reverse = f32[10,16] reverse(%a), dimensions={1}
      ROOT %dot = f32[10,20] dot(%reverse, %b, %meta),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=L.1@2:4
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_use_associative_reordering(true);
  options.set_associative_reordering_threshold(0);
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(module.get()).value());
}

TEST_F(AlgebraicSimplifierTest, SparseDotReduceBatchDimension) {
  const char* kModuleStr = R"(
    HloModule m
    add {
      %p0 = f32[] parameter(0)
      %p1 = f32[] parameter(1)
      ROOT %add = f32[] add(%p0, %p1)
    }
    ENTRY test {
      %p0 = f32[32,8,5,64] parameter(0)
      %p1 = f32[8,32,128,7] parameter(1)
      %meta = u16[32,8,5,8] parameter(2)
      %dot = f32[32,8,5,7] dot(%p0, %p1, %meta),
          lhs_batch_dims={0,1}, rhs_batch_dims={1,0},
          lhs_contracting_dims={3}, rhs_contracting_dims={2}, sparsity=L.3@2:4
      %c = f32[] constant(0)
      ROOT %r = f32[8,5,7] reduce(%dot, %c), dimensions={0}, to_apply=add
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(SparseDotMatcher(m::Parameter(0), m::Parameter(1),
                                          m::Parameter(2))));
}

TEST_F(AlgebraicSimplifierTest, SparseDotNoContractingReorder) {
  const char* kModuleStr = R"(
    HloModule m
    ENTRY test {
      %lhs = f32[2,8] constant({{1,2,3,4,5,6,7,8},{9,10,11,12,13,14,15,16}})
      %meta = u16[2,1] constant({{0},{1}})
      %t0 = f32[5,2,8] parameter(0)
      %t1 = f32[5,8,2] transpose(%t0), dimensions={0,2,1}
      %rhs = f32[5,16] reshape(t1)
      ROOT %dot = f32[2,5] dot(%lhs, %rhs, %meta),
          lhs_contracting_dims={1}, rhs_contracting_dims={1}, sparsity=L.1@2:4
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, SparseDotOfConcat) {
  const char* kModuleStr = R"(
    HloModule m
    ENTRY test {
      %a = f32[2,4] parameter(0)
      %b = f32[2,4] parameter(1)
      %lhs = f32[2,8] concatenate(%a, %b), dimensions={1}
      %meta = u16[2,1] constant({{0},{1}})
      %rhs = f32[16,2] constant({
          {0,1},{2,3},{4,5},{6,7},{8,9},{10,11},{12,13},{14,15},
          {16,17},{18,19},{20,21},{22,23},{24,25},{26,27},{28,29},{30,31}})
      ROOT %dot = f32[2,2] dot(%lhs, %rhs, %meta),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=L.1@2:4
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, BroadcastToTranspose) {
  const std::string hlo_string = R"(
  HloModule broadcast_module
    ENTRY %main {
      input = f32[6,4,3] parameter(0)
      ROOT output = f32[4,3,6] broadcast(input), dimensions={2,0,1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Transpose(m::Parameter(0))));
  EXPECT_EQ(root->dimensions(), std::vector<int64_t>({1, 2, 0}));
}

TEST_F(AlgebraicSimplifierTest, BroadcastToTranspose2) {
  const std::string hlo_string = R"(
  HloModule broadcast_module
    ENTRY %main {
      input = f32[6,4,3] parameter(0)
      ROOT output = f32[4,6,3] broadcast(input), dimensions={1,0,2}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Transpose(m::Parameter(0))));
  EXPECT_EQ(root->dimensions(), std::vector<int64_t>({1, 0, 2}));
}

TEST_F(AlgebraicSimplifierTest, LayoutConstraintToNoop) {
  const std::string hlo_string = R"(
  HloModule layout_constraint
    ENTRY %main {
      input = f32[6,4,3]{0,1,2} parameter(0)
      ROOT output = f32[6,4,3]{0,1,2} custom-call(input), custom_call_target="LayoutConstraint"
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(AlgebraicSimplifier(options).Run(m.get()).value());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter(0)));
}

TEST_F(AlgebraicSimplifierTest, LayoutConstraintToCopy) {
  const std::string hlo_string = R"(
  HloModule layout_constraint
    ENTRY %main {
      input = f32[6,4,3]{0,1,2} parameter(0)
      ROOT output = f32[6,4,3]{2,0,1} custom-call(input), custom_call_target="LayoutConstraint"
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnUnverifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(AlgebraicSimplifier(options).Run(m.get()).value());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Copy(m::Parameter(0))));
}

TEST_F(AlgebraicSimplifierTest, KeepLayoutConstraint) {
  const std::string hlo_string = R"(
  HloModule layout_constraint
    ENTRY %main {
      input = f32[6,4,3]{0,1,2} parameter(0)
      ROOT output = f32[6,4,3]{2,0,1} custom-call(input), custom_call_target="LayoutConstraint"
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnUnverifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(false);
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(AlgebraicSimplifier(options).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, PreserveSharding) {
  const std::string hlo_string = R"(
  HloModule jit_matmul, entry_computation_layout={(f64[8,3]{1,0}, f64[])->f64[8,3]{1,0}}, allow_spmd_sharding_propagation_to_parameters={false,true}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=2
    ENTRY %main.4 (Arg_0.1: f64[8,3], Arg_1.2: f64[]) -> f64[8,3] {
      %Arg_1.2 = f64[] parameter(1)
      %Arg_0.1 = f64[8,3]{1,0} parameter(0), sharding={devices=[2,1]0,1}
      ROOT %dot.3 = f64[8,3]{1,0} dot(f64[] %Arg_1.2, f64[8,3]{1,0} %Arg_0.1), lhs_contracting_dims={}, rhs_contracting_dims={}, metadata={op_name="jit(matmul)/jit(main)/dot_general[dimension_numbers=(((), ()), ((), ())) precision=None preferred_element_type=float64]" source_file="third_party/py/jax/tests/pjit_test.py" source_line=4021}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_TRUE(m->entry_computation()->parameter_instruction(0)->has_sharding());
}

// Move parameter from the LHS of a dot to the RHS.
TEST_F(AlgebraicSimplifierTest, SwapDotOperands) {
  set_verifier_layout_sensitive(false);
  set_instruction_can_change_layout_func({});
  const std::string hlo_string = R"(
HloModule main

ENTRY main.1 {
  param_0 = s8[1024,1024] parameter(0)
  param_1 = bf16[1024,1024] parameter(1)
  activations = bf16[1024,1024] add(param_1, param_1)
  weights = bf16[1024,1024] convert(param_0)
  ROOT dot = bf16[1024,1024] dot(weights, activations), lhs_batch_dims={}, lhs_contracting_dims={1}, rhs_batch_dims={}, rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  default_options_.set_enable_move_dot_param_to_rhs(true);
  EXPECT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTranspose);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kDot);
  EXPECT_NE(root->operand(0)->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(0)->operand(1)->operand(0)->opcode(),
            HloOpcode::kParameter);
}

TEST_F(AlgebraicSimplifierTest, ReduceOfConstantBroadcastS32) {
  const std::string hlo_string = R"(
  HloModule test
    add_s32 {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT r = s32[] add(p0, p1)
    }
    ENTRY test.1 {
      one = s32[] constant(2)
      init = s32[] constant(10)
      bcast = s32[1,7,7,1] broadcast(one), dimensions={}
      ROOT out = s32[1,7,1] reduce(bcast, init), dimensions={1}, to_apply=add_s32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_TRUE(simplifier.Run(m.get()).value());
  int64_t reduce_count =
      absl::c_count_if(m->entry_computation()->instructions(),
                       HloPredicateIsOp<HloOpcode::kReduce>);
  // Expect no Reduce operation after simplification.
  EXPECT_EQ(0, reduce_count);
}

TEST_F(AlgebraicSimplifierTest, TrivialReduce) {
  const std::string hlo_string = R"(
  HloModule test
    add_s32 {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT r = s32[] add(p0, p1)
    }
    ENTRY test.1 {
      bcast = s32[1,7,7,1] parameter(0)
      init = s32[] constant(0)
      ROOT out = s32[1,7,7] reduce(bcast, init), dimensions={3}, to_apply=add_s32
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  AlgebraicSimplifierOptions options = default_options_;
  options.set_is_layout_sensitive(false);
  HloPassFix<AlgebraicSimplifier> simplifier(options);
  EXPECT_TRUE(simplifier.Run(m.get()).value());
  int64_t reduce_count =
      absl::c_count_if(m->entry_computation()->instructions(),
                       HloPredicateIsOp<HloOpcode::kReduce>);
  // Expect no Reduce operation after simplification.
  EXPECT_EQ(0, reduce_count);
}

TEST_F(AlgebraicSimplifierTest, ReduceOfConstantBroadcastBF16) {
  const std::string hlo_string = R"(
  HloModule test
    add_bf16 {
      p0 = bf16[] parameter(0)
      p1 = bf16[] parameter(1)
      ROOT r = bf16[] add(p0, p1)
    }
    ENTRY test.1 {
      one = bf16[] constant(2.12)
      init = bf16[] constant(10.34)
      bcast = bf16[1,7,7,1] broadcast(one), dimensions={}
      ROOT out = bf16[1,7,1] reduce(bcast, init), dimensions={1}, to_apply=add_bf16
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_TRUE(simplifier.Run(m.get()).value());
  int64_t reduce_count =
      absl::c_count_if(m->entry_computation()->instructions(),
                       HloPredicateIsOp<HloOpcode::kReduce>);
  // Expect no Reduce operation after simplification.
  EXPECT_EQ(0, reduce_count);
}

TEST_F(AlgebraicSimplifierTest, ReduceOfNonScalarBroadcast) {
  const std::string hlo_string = R"(
    HloModule module
    add {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT sum = f32[] add(a, b)
    }

    ENTRY test {
        a = f32[64,1001] parameter(0)
        broadcast = f32[64,7,7,1001] broadcast(a), dimensions={0,3}
        zero = f32[] constant(0)
        ROOT reduce = f32[64,7,1001] reduce(broadcast, zero), dimensions={2},
                  to_apply=add
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_TRUE(simplifier.Run(m.get()).value());
  HloInstruction* root = m->entry_computation()->root_instruction();
  int64_t reduce_count =
      absl::c_count_if(m->entry_computation()->instructions(),
                       HloPredicateIsOp<HloOpcode::kReduce>);
  // Expect no Reduce operation after simplification.
  EXPECT_EQ(0, reduce_count);
  EXPECT_THAT(root, GmockMatch(m::Broadcast(m::Multiply())));
}

TEST_F(AlgebraicSimplifierTest, RemoveConvertConstant) {
  const std::string hlo_string = R"(
    HloModule module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT r = f32[] add(p0, p1)
    }

    ENTRY test {
        a = f32[32,64] parameter(0)
        b = s32[] constant(0)
        c = f32[] convert(b)
        ROOT reduce = f32[32] reduce(a, c),
                      dimensions={1}, to_apply=add
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  default_options_.set_use_convert_constant_folding(true);
  EXPECT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Reduce(m::Parameter(0),
                                         m::Constant().WithShape(F32, {}))));
}

TEST_F(AlgebraicSimplifierTest, KeepInt4ConvertConstant) {
  const std::string hlo_string = R"(
    HloModule module

    ENTRY test {
        a = s8[] constant(0)
        ROOT b = s4[] convert(a)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  default_options_.set_use_convert_constant_folding(true);
  ASSERT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, ReduceBroadcastScalarToBroadcastMultiply) {
  const std::string hlo_string = R"(
    HloModule module
    add_bf16 {
      x = bf16[] parameter(0)
      y = bf16[] parameter(1)
      ROOT sum = bf16[] add(x, y)
    }

    ENTRY test {
        a = bf16[] parameter(0)
        negate = bf16[] negate(a)
        broadcast = bf16[2,5,11,17,19] broadcast(negate), dimensions={}
        zero = bf16[] constant(0)
        ROOT reduce = bf16[2] reduce(broadcast, zero),
                      dimensions={1,2,3,4}, to_apply=add_bf16
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloPassFix<AlgebraicSimplifier> simplifier(default_options_);
  EXPECT_TRUE(simplifier.Run(m.get()).value());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kMultiply);
}

TEST_F(AlgebraicSimplifierTest, SinkCbrtThroughMax) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY test {
        a = bf16[17,96,120] parameter(0)
        b = bf16[17,96,120] parameter(1)
        cbrt_a = bf16[17,96,120] cbrt(a)
        cbrt_b = bf16[17,96,120] cbrt(b)
        ROOT max = bf16[17,96,120] maximum(cbrt_a, cbrt_b)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, GmockMatch(m::Cbrt(m::Maximum(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(AlgebraicSimplifierTest,
       DynamicSlicePreservedWithTrivialConstantIndices) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY f {
      %operand = s32[2,2] parameter(0)
      %constant = u32[] constant(0)
      ROOT %dynamic-slice = s32[2,1] dynamic-slice(%operand, %constant, %constant),
        dynamic_slice_sizes={2,1}
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Disable dynamic-slice to slice conversion
  default_options_.set_disable_dynamic_slice_to_slice_conversion(true);

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_FALSE(simplifier.Run(module.get()).value());

  // Expect the dynamic-slice to be preserved
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicSlice(m::Parameter(0), m::Constant(),
                                         m::Constant())));
}

TEST_F(AlgebraicSimplifierTest,
       DynamicSliceConvertedToConstantSliceWithConstantIndices) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY f {
      %operand = s32[2,2] parameter(0)
      %constant = u32[] constant(0)
      ROOT %dynamic-slice = s32[2,1] dynamic-slice(%operand, %constant, %constant),
        dynamic_slice_sizes={2,1}
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Enable dynamic-slice to slice conversion (default behavior)
  ASSERT_FALSE(default_options_.disable_dynamic_slice_to_slice_conversion());

  AlgebraicSimplifier simplifier(default_options_);
  ASSERT_TRUE(simplifier.Run(module.get()).value());

  // Expect the dynamic-slice to be converted to a constant slice
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Slice(m::Parameter(0))));
}

// Bitcast of broadcast is not simplified if the layouts are different.
// TransposeBitcastOfBroadcast is a simplified example.
TEST_F(AlgebraicSimplifierTest, BitcastBroadcastDifferentLayout) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY f {
      %operand = f32[200001]{0:T(1024)} parameter(0)
      %broadcast.91 = f32[200001,128]{1,0:T(8,128)} broadcast(f32[200001]{0:T(1024)} %operand), dimensions={0}
      %bitcast.8 = f32[200001,128]{1,0:T(8)L(1024)} bitcast(f32[200001,128]{1,0:T(8,128)} %broadcast.91)
      ROOT %add = f32[200001,128]{1,0:T(8)L(1024)} add(f32[200001,128]{1,0:T(8)L(1024)} %bitcast.8, f32[200001,128]{1,0:T(8)L(1024)} %bitcast.8)
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(module.get()).value());
}

TEST_F(AlgebraicSimplifierTest, AllGatherOfBroadcast) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      z = f32[] constant(0)
      b = f32[4,4] broadcast(z), dimensions={}
      ROOT ag = f32[16,4] all-gather(b), dimensions={0}, replica_groups={{0, 1, 2, 3}}
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Constant())));
}

TEST_F(AlgebraicSimplifierTest, TrivialMin) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      a = f32[4,4] parameter(0)
      ROOT %min = f32[4,4] minimum(%a, %a)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(AlgebraicSimplifierTest, TrivialMax) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      a = f32[4,4] parameter(0)
      ROOT %min = f32[4,4] maximum(%a, %a)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(AlgebraicSimplifierTest, PathologicalComplexity) {
  // Without replacing min(x,x)->x, the algorithmic recursion complexity is
  // O(2^n).
  const char* kModuleStr = R"(
    HloModule m
    test {
      a = s32[4,4] parameter(0)
      b = s32[4,4] parameter(1)
      %cmp0 = pred[4,4] compare(a, b), direction=GE
      %c1 = f32[] constant(1)
      %ones = f32[4,4] broadcast(f32[] %c1)
      %c0 = f32[] constant(0)
      %zeros = f32[4,4] broadcast(f32[] %c0)
      %min = f32[4,4] minimum(%ones, %zeros)
      %min0 = f32[4,4] minimum(%min, %min)
      %min1 = f32[4,4] minimum(%min0, %min0)
      %min2 = f32[4,4] minimum(%min1, %min1)
      %min3 = f32[4,4] minimum(%min2, %min2)
      %min4 = f32[4,4] minimum(%min3, %min3)
      %min5 = f32[4,4] minimum(%min4, %min4)
      %min6 = f32[4,4] minimum(%min5, %min5)
      %min7 = f32[4,4] minimum(%min6, %min6)
      %min8 = f32[4,4] minimum(%min7, %min7)
      %min9 = f32[4,4] minimum(%min8, %min8)
      %min10 = f32[4,4] minimum(%min9, %min9)
      %min11 = f32[4,4] minimum(%min10, %min10)
      %min12 = f32[4,4] minimum(%min11, %min11)
      %min13 = f32[4,4] minimum(%min12, %min12)
      %min14 = f32[4,4] minimum(%min13, %min13)
      %min15 = f32[4,4] minimum(%min14, %min14)
      %min16 = f32[4,4] minimum(%min15, %min15)
      %min17 = f32[4,4] minimum(%min16, %min16)
      %min18 = f32[4,4] minimum(%min17, %min17)
      %min19 = f32[4,4] minimum(%min18, %min18)
      %min20 = f32[4,4] minimum(%min19, %min19)
      %min21 = f32[4,4] minimum(%min20, %min20)
      %min22 = f32[4,4] minimum(%min21, %min21)
      %min23 = f32[4,4] minimum(%min22, %min22)
      %min24 = f32[4,4] minimum(%min23, %min23)
      %min25 = f32[4,4] minimum(%min24, %min24)
      %min26 = f32[4,4] minimum(%min25, %min25)
      %min27 = f32[4,4] minimum(%min26, %min26)
      %min28 = f32[4,4] minimum(%min27, %min27)
      %min29 = f32[4,4] minimum(%min28, %min28)
      ROOT %cmp1 = pred[4,4] compare(%min29, %zeros), direction=LT
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  ASSERT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(m::Constant())));
}

TEST_F(AlgebraicSimplifierTest, RespectHostOffloadingcopies) {
  const char* hlo_string = R"(
    HloModule m
    ENTRY test {
      param0 = f32[16384,126]{1,0:T(8,128)S(5)} parameter(0), sharding={replicated}
      copy0 = f32[16384,126]{1,0:T(8,128)} copy(param0)
      bitcast0 = f32[126,16384]{0,1:T(8,128)} bitcast(copy0)
      ROOT copy1 = f32[16384,126]{1,0:T(8,128)} copy(bitcast0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(module.get()).value());
}

TEST_F(AlgebraicSimplifierTest,
       ReducePrecisionWithSamePrecisionAsOperandIsRemovedIfRemoveNoOpIsSet) {
  const char* hlo = R"(
  HloModule test
  ENTRY main {
    p0 = bf16[64]{0} parameter(0)
    ROOT reduce-precision = bf16[64] reduce-precision(p0), exponent_bits=8, mantissa_bits=7
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo));
  default_options_.set_enable_remove_no_op_reduce_precision(true);
  EXPECT_TRUE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter()));
}

TEST_F(AlgebraicSimplifierTest,
       ReducePrecisionWithDifferentPrecisionFromOperandIsNotModifiedByDefault) {
  const char* hlo = R"(
  HloModule test
  ENTRY main {
    p0 = bf16[64]{0} parameter(0)
    ROOT reduce-precision = bf16[64] reduce-precision(p0), exponent_bits=7, mantissa_bits=8
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo));

  default_options_.set_enable_remove_no_op_reduce_precision(true);
  EXPECT_FALSE(AlgebraicSimplifier(default_options_).Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, TestWithControlDependencies) {
  const char* hlo = R"(
  HloModule extracted, entry_computation_layout={((s8[2]{0}, s8[]))->s8[]}

  inner_body (p.1: (s8[2], s8[])) -> (s8[2], s8[]) {
    z.1 = s8[2]{0} constant({0, 0})
    p.1 = (s8[2]{0}, s8[]) parameter(0)
    s = s8[] get-tuple-element((s8[2]{0}, s8[]) p.1), index=1
    sz = s8[] constant(0)
    add = s8[] add(s8[] s, s8[] sz), control-predecessors={z.1}
    ROOT r.1 = (s8[2]{0}, s8[]) tuple(s8[2]{0} z.1, s8[] add)
  }

  cond (p: (s8[2], s8[])) -> pred[] {
    p = (s8[2]{0}, s8[]) parameter(0)
    i = s8[] get-tuple-element((s8[2]{0}, s8[]) p), index=1
    z = s8[] constant(0)
    ROOT r = pred[] compare(s8[] i, s8[] z), direction=LT
  }

  ENTRY outer_body (p.2: (s8[2], s8[])) -> s8[] {
    z.2 = s8[2]{0} constant({0, 0})
    p.2 = (s8[2]{0}, s8[]) parameter(0)
    s.1 = s8[] get-tuple-element((s8[2]{0}, s8[]) p.2), index=1
    t = (s8[2]{0}, s8[]) tuple(s8[2]{0} z.2, s8[] s.1)
    w = (s8[2]{0}, s8[]) while((s8[2]{0}, s8[]) t), condition=cond, body=inner_body
    ROOT ss = s8[] get-tuple-element((s8[2]{0}, s8[]) w), index=1
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, CopyReshapeToReshapeCopyWithHostCopies) {
  const char* hlo = R"(
  HloModule module

  ENTRY main {
    param.251 = f32[128,8]{0,1:T(8,128)S(5)} parameter(0), sharding={devices=[1,16,16]<=[256] last_tile_dim_replicate}
    copy.11654 = f32[128,8]{0,1:T(8,128)} copy(param.251)
    reshape.37527 = f32[16,8,8]{1,0,2:T(8,128)} reshape(copy.11654)
    ROOT copy.10970 = f32[16,8,8]{1,2,0:T(8,128)} copy(reshape.37527)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
  auto reshape_is_bitcast = [](const Shape& from_shape, const Shape& to_shape) {
    return false;
  };
  AlgebraicSimplifierOptions options(reshape_is_bitcast);
  options.set_enable_floats_are_real(true);
  options.set_is_layout_sensitive(true);
  options.set_enable_conv_simplification(false);
  AlgebraicSimplifier simplifier(options);
  EXPECT_FALSE(simplifier.Run(m.get()).value());
}

TEST_F(AlgebraicSimplifierTest, SimplifyShardedPad) {
  const char* hlo = R"(
HloModule test, num_partitions=4

ENTRY main {
  c0 = f32[] constant(0)
  c1 = f32[] constant(1)
  b0 = f32[512,34,5]{2,1,0} broadcast(c0), dimensions={}, sharding={devices=[1,2,2]<=[2,2]T(1,0)}
  ROOT pad = f32[512,46,5]{2,1,0} pad(b0, c1), padding=0_0x6_6x0_0, sharding={devices=[1,2,2]<=[2,2]T(1,0)}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));

  AlgebraicSimplifierOptions options;
  AlgebraicSimplifier simplifier(options);
  EXPECT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Broadcast(
                  m::Pad(m::Broadcast(m::Constant()), m::Constant()))));
  TF_EXPECT_OK(VerifyHloModule(m.get(),
                               /*layout_sensitive=*/true,
                               /*allow_mixed_precision=*/true));
}

}  // namespace
}  // namespace xla

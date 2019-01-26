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
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_element_type_converter.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

static std::array<bool, 2> use_bf16_params{true, false};

// Test fixture for the HloEvaluator.
//
// In bf16 mode, all f32 shapes are converted to bf16 before running.
class HloEvaluatorTest : public HloTestBase {
 public:
  HloEvaluatorTest() : use_bfloat16_(false) {}

  Literal Evaluate(absl::Span<const Literal* const> arg_literals = {}) {
    if (use_bfloat16_) {
      HloElementTypeConverter(F32, BF16).Run(m_.get()).ValueOrDie();
    }
    return evaluator_.Evaluate(*m_->entry_computation(), arg_literals)
        .ConsumeValueOrDie();
  }

  // Evaluate function that takes in a local module instead of using m_
  // that is in HloTestBase. Once m_ in HloTestBase is
  // removed, this should be the default Evaluate function.
  Literal EvaluateWithModule(
      HloModule* module, absl::Span<const Literal* const> arg_literals = {}) {
    if (use_bfloat16_) {
      HloElementTypeConverter(F32, BF16).Run(m_.get()).ValueOrDie();
    }
    return evaluator_.Evaluate(*module->entry_computation(), arg_literals)
        .ConsumeValueOrDie();
  }

  void TestUnaryOp(HloOpcode opcode, Literal expected, Literal input,
                   float aabs = 0) {
    HloComputation::Builder b(TestName());
    auto c1 =
        b.AddInstruction(HloInstruction::CreateConstant(std::move(input)));
    b.AddInstruction(HloInstruction::CreateUnary(expected.shape(), opcode, c1));
    m_->AddEntryComputation(b.Build());

    Literal result = Evaluate();

    auto element_type = expected.shape().element_type();
    if (element_type == F32 || element_type == F64) {
      ErrorSpec error(aabs);
      EXPECT_TRUE(LiteralTestUtil::Near(expected, result, error));
    } else {
      EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
    }
  }

  void TestBinaryOp(HloOpcode opcode, Literal expected, Literal lhs,
                    Literal rhs) {
    HloComputation::Builder b(TestName());
    auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs)));
    auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs)));
    b.AddInstruction(
        HloInstruction::CreateBinary(expected.shape(), opcode, c1, c2));
    m_->AddEntryComputation(b.Build());

    Literal result = Evaluate();

    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

 protected:
  explicit HloEvaluatorTest(bool use_bfloat16) : use_bfloat16_(use_bfloat16) {}
  HloEvaluator evaluator_;

  const bool use_bfloat16_;
  std::unique_ptr<HloModule> m_ = CreateNewVerifiedModule();
};

// Lets you write TEST_Ps that run twice, once with and once without bf16.
class HloEvaluatorBf16Test : public ::testing::WithParamInterface<bool>,
                             public HloEvaluatorTest {
 protected:
  HloEvaluatorBf16Test() : HloEvaluatorTest(/*use_bfloat16=*/GetParam()) {}
};

INSTANTIATE_TEST_SUITE_P(HloEvaluatorTest_Instantiation, HloEvaluatorBf16Test,
                         ::testing::ValuesIn(use_bf16_params));

// Verifies that HloEvaluator evaluates a HLO instruction that performs clamp
// with 3 operands.
TEST_P(HloEvaluatorBf16Test, DoesClamp) {
  auto low = LiteralUtil::CreateR2<float>({{0.f, 2.f}, {2.f, 4.f}});
  auto value = LiteralUtil::CreateR2<float>({{0.f, 5.f}, {0.f, 4.f}});
  auto high = LiteralUtil::CreateR2<float>({{2.f, 4.f}, {4.f, 4.f}});

  Shape shape = low.shape();
  HloComputation::Builder b(TestName());
  auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(low)));
  auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(value)));
  auto c3 = b.AddInstruction(HloInstruction::CreateConstant(std::move(high)));
  b.AddInstruction(
      HloInstruction::CreateTernary(shape, HloOpcode::kClamp, c1, c2, c3));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected = LiteralUtil::CreateR2<float>({{0, 4}, {2, 4}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Verifies that clamping of int64 does not cause loss of precision
TEST_P(HloEvaluatorBf16Test, DoesClampInt64) {
  auto ones = [](int bits) -> int64 { return (1LL << bits) - 1; };

  auto low =
      LiteralUtil::CreateR2<int64>({{0, ones(54)}, {ones(54), ones(58)}});
  auto value = LiteralUtil::CreateR2<int64>({{0, ones(56)}, {0, ones(58)}});
  auto high = LiteralUtil::CreateR2<int64>(
      {{ones(54), ones(55)}, {ones(56), ones(58)}});

  Shape shape = low.shape();
  HloComputation::Builder b(TestName());
  auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(low)));
  auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(value)));
  auto c3 = b.AddInstruction(HloInstruction::CreateConstant(std::move(high)));
  b.AddInstruction(
      HloInstruction::CreateTernary(shape, HloOpcode::kClamp, c1, c2, c3));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected =
      LiteralUtil::CreateR2<int64>({{0, ones(55)}, {ones(54), ones(58)}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DISABLED_DoesClampSpecialBroadcast) {
  auto low = LiteralUtil::CreateR0<float>(0.f);
  auto value = LiteralUtil::CreateR2<float>({{-1.f, 0.f}, {1.f, 2.f}});
  auto high = LiteralUtil::CreateR0<float>(1.f);

  Shape shape = value.shape();
  HloComputation::Builder b(TestName());
  auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(low)));
  auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(value)));
  auto c3 = b.AddInstruction(HloInstruction::CreateConstant(std::move(high)));
  b.AddInstruction(
      HloInstruction::CreateTernary(shape, HloOpcode::kClamp, c1, c2, c3));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected = LiteralUtil::CreateR2<float>({{0, 0}, {1, 1}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs select
// with 3 operands.
TEST_P(HloEvaluatorBf16Test, DoesSelect) {
  auto pred = LiteralUtil::CreateR2<bool>({{true, false}, {false, true}});
  auto on_true = LiteralUtil::CreateR2<float>({{2.f, 4.f}, {4.f, 4.f}});
  auto on_false = LiteralUtil::CreateR2<float>({{0.f, 5.f}, {0.f, 4.f}});

  Shape shape = on_true.shape();
  HloComputation::Builder b(TestName());
  auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(pred)));
  auto c2 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(on_true)));
  auto c3 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(on_false)));
  b.AddInstruction(
      HloInstruction::CreateTernary(shape, HloOpcode::kSelect, c1, c2, c3));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate({});

  auto expected = LiteralUtil::CreateR2<float>({{2, 5}, {0, 4}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise addition with 2 operands.
TEST_F(HloEvaluatorTest, DoesAdd) {
  auto lhs = LiteralUtil::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64>({{3, 4}, {-96, 8}});
  TestBinaryOp(HloOpcode::kAdd, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise and with 2 operands.
TEST_P(HloEvaluatorBf16Test, DoesAnd) {
  auto lhs = LiteralUtil::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64>({{0, 0}, {4, 4}});
  TestBinaryOp(HloOpcode::kAnd, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise or with 2 operands.
TEST_F(HloEvaluatorTest, DoesOr) {
  auto lhs = LiteralUtil::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64>({{3, 4}, {-100, 4}});
  TestBinaryOp(HloOpcode::kOr, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise or with 2 operands.
TEST_F(HloEvaluatorTest, DoesXor) {
  auto lhs = LiteralUtil::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64>({{3, 4}, {-104, 0}});
  TestBinaryOp(HloOpcode::kXor, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise multiply with 2 operands.
TEST_F(HloEvaluatorTest, DoesMultiply) {
  auto lhs = LiteralUtil::CreateR2<int32>({{-1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int32>(
      {{std::numeric_limits<int32>::min(), 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int32>(
      {{std::numeric_limits<int32>::min(), 0}, {-400, 16}});
  TestBinaryOp(HloOpcode::kMultiply, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise divide with 2 operands.
TEST_F(HloEvaluatorTest, DoesDivideInt64) {
  auto lhs = LiteralUtil::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64>({{0, 0}, {-25, 1}});
  TestBinaryOp(HloOpcode::kDivide, std::move(expected), std::move(lhs),
               std::move(rhs));
}
TEST_P(HloEvaluatorBf16Test, DoesDivideDouble) {
  auto lhs = LiteralUtil::CreateR2<double>({{1.0, 0.0}, {-100.0, 4.0}});
  auto rhs = LiteralUtil::CreateR2<double>({{2.2, 4.0}, {4.0, 4.0}});
  auto expected =
      LiteralUtil::CreateR2<double>({{0.45454545454545453, 0}, {-25, 1}});
  TestBinaryOp(HloOpcode::kDivide, std::move(expected), std::move(lhs),
               std::move(rhs));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise abs op with 1 operand.
TEST_F(HloEvaluatorTest, DoesAbsR2) {
  auto operand = LiteralUtil::CreateR2<int64>({{1, -20}, {-100, 4}});
  auto expected = LiteralUtil::CreateR2<int64>({{1, 20}, {100, 4}});
  TestUnaryOp(HloOpcode::kAbs, std::move(expected), std::move(operand));
}
TEST_P(HloEvaluatorBf16Test, DoesAbsR0) {
  auto operand = LiteralUtil::CreateR0<float>(-1.0f);
  auto expected = LiteralUtil::CreateR0<float>(1.0f);
  TestUnaryOp(HloOpcode::kAbs, std::move(expected), std::move(operand));
}
TEST_P(HloEvaluatorBf16Test, DoesAbsR1WithZeroSize) {
  auto operand = LiteralUtil::CreateR1<float>({});
  auto expected = LiteralUtil::CreateR1<float>({});
  TestUnaryOp(HloOpcode::kAbs, std::move(expected), std::move(operand));
}
TEST_F(HloEvaluatorTest, DoesNegateR2) {
  auto operand = LiteralUtil::CreateR2<int32>(
      {{0, std::numeric_limits<int32>::min()}, {-1, 4}});
  auto expected = LiteralUtil::CreateR2<int32>(
      {{0, std::numeric_limits<int>::min()}, {1, -4}});
  TestUnaryOp(HloOpcode::kNegate, std::move(expected), std::move(operand));
}
TEST_P(HloEvaluatorBf16Test, DoesCosR2) {
  auto operand = LiteralUtil::CreateR2<float>({{0, M_PI}, {-M_PI, 2 * M_PI}});
  auto expected = LiteralUtil::CreateR2<float>({{1, -1}, {-1, 1}});
  TestUnaryOp(HloOpcode::kCos, std::move(expected), std::move(operand),
              use_bfloat16_ ? 0.031250 : 9.5367431640625E-7);
}
TEST_P(HloEvaluatorBf16Test, DoesSinR2) {
  auto operand = LiteralUtil::CreateR2<float>({{0, M_PI}, {-M_PI, 2 * M_PI}});
  auto expected = LiteralUtil::CreateR2<float>({{0, 0}, {0, 0}});
  TestUnaryOp(HloOpcode::kSin, std::move(expected), std::move(operand),
              use_bfloat16_ ? 0.031250 : 9.5367431640625E-7);
}
TEST_F(HloEvaluatorTest, DoesNotR2) {
  auto operand =
      LiteralUtil::CreateR2<int32>({{0, std::numeric_limits<int>::min()},
                                    {-1, std::numeric_limits<int>::max()}});
  auto expected =
      LiteralUtil::CreateR2<int32>({{-1, std::numeric_limits<int>::max()},
                                    {0, std::numeric_limits<int>::min()}});
  TestUnaryOp(HloOpcode::kNot, std::move(expected), std::move(operand));
}

TEST_F(HloEvaluatorTest, DoesRealC128) {
  auto x = LiteralUtil::CreateR1<complex128>({{1, 0}, {-100, 4}});
  auto expected_real = LiteralUtil::CreateR1<double>({1, -100});
  TestUnaryOp(HloOpcode::kReal, std::move(expected_real), std::move(x));
}

TEST_F(HloEvaluatorTest, DoesImagC128) {
  auto x = LiteralUtil::CreateR1<complex128>({{1, 0}, {-100, 4}});
  auto expected_imag = LiteralUtil::CreateR1<double>({0, 4});
  TestUnaryOp(HloOpcode::kImag, std::move(expected_imag), std::move(x));
}

// Verifies that HloEvaluator evaluates a HLO Computation with non-parameter nor
// constant operands.
TEST_F(HloEvaluatorTest, DoesTraverseInstructions) {
  auto lhs = LiteralUtil::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64>({{2, 4}, {4, 4}});
  auto rhs2 = LiteralUtil::CreateR2<int64>({{1, -20}, {-100, 4}});
  std::vector<const Literal*> args = {&lhs, &rhs, &rhs2};

  Shape shape = ShapeUtil::MakeShape(S64, {2, 2});

  HloComputation::Builder b(TestName());
  auto param_lhs =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "lhs"));
  auto param_rhs =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "rhs"));
  auto lhs_instruction = b.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, param_lhs, param_rhs));

  auto param_rhs2 =
      b.AddInstruction(HloInstruction::CreateParameter(2, shape, "rhs2"));
  b.AddInstruction(HloInstruction::CreateBinary(shape, HloOpcode::kAdd,
                                                lhs_instruction, param_rhs2));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate(args);

  auto expected = LiteralUtil::CreateR2<int64>({{4, -16}, {-196, 12}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Verifies Reshape operation is correctly evaluated.
TEST_F(HloEvaluatorTest, DoesReshape) {
  HloComputation::Builder b(TestName());
  const int64 dimensions[] = {11, 8, 7, 5, 9};
  TF_ASSERT_OK_AND_ASSIGN(auto literal,
                          LiteralUtil::CreateRandomLiteral<F32>(
                              ShapeUtil::MakeShape(F32, dimensions), 0.0, 1.0));
  auto literal_clone = literal.Clone();
  HloInstruction* literal_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {8, 7, 11, 9, 5});
  const int64 permutation[] = {1, 2, 0, 4, 3};
  b.AddInstruction(
      HloInstruction::CreateTranspose(shape, literal_instruction, permutation));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate({});

  using NativeT = typename primitive_util::PrimitiveTypeToNative<F32>::type;
  result.EachCell<NativeT>([&](absl::Span<const int64> indices, NativeT value) {
    std::vector<int64> rindexes = Permute(permutation, indices);
    EXPECT_NEAR(value, literal_clone.Get<NativeT>(rindexes), 0.031250);
  });
}

// Verifies Broadcast operation is correctly evaluated.
TEST_F(HloEvaluatorTest, DoesBroadcast) {
  HloComputation::Builder b(TestName());
  auto input_literal = LiteralUtil::CreateR2<int32>({{1, 2}, {3, 4}, {5, 6}});
  auto output_literal = LiteralUtil::CreateR3<int32>(
      {{{1, 2}, {3, 4}, {5, 6}}, {{1, 2}, {3, 4}, {5, 6}}});
  HloInstruction* literal_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  b.AddInstruction(HloInstruction::CreateBroadcast(
      output_literal.shape(), literal_instruction, {1, 2}));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate({});

  EXPECT_TRUE(LiteralTestUtil::Equal(result, output_literal));
}

TEST_F(HloEvaluatorTest, DoesBroadcastScalar) {
  HloComputation::Builder b(TestName());
  auto input_literal = LiteralUtil::CreateR0<int32>(111);
  auto output_literal = LiteralUtil::CreateR2<int32>(
      {{111, 111}, {111, 111}, {111, 111}, {111, 111}, {111, 111}, {111, 111}});

  HloInstruction* literal_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  // Broadcast dimension should be empty in the case of scalars.
  b.AddInstruction(HloInstruction::CreateBroadcast(
      output_literal.shape(), literal_instruction,
      /*broadcast_dimensions=*/{}));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate({});

  EXPECT_TRUE(LiteralTestUtil::Equal(result, output_literal));
}

TEST_F(HloEvaluatorTest, DoesConcatenateSimple) {
  HloComputation::Builder b(TestName());

  HloInstruction* operand1 = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int64>({{-1, -2}, {100, 200}})));
  HloInstruction* operand2 = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int64>({{-2, -3}, {-100, -200}})));

  std::vector<HloInstruction*> operands = {operand1, operand2};

  Shape shape = ShapeUtil::MakeShape(S64, {4, 2});
  b.AddInstruction(HloInstruction::CreateConcatenate(shape, operands, 0));

  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected = LiteralUtil::CreateR2<int64>(
      {{-1, -2}, {100, 200}, {-2, -3}, {-100, -200}});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, ConcatenateHandlesShapeWithZeroElement) {
  HloComputation::Builder b(TestName());

  HloInstruction* operand1 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int64>({100, 200})));
  HloInstruction* operand2 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int64>({})));

  std::vector<HloInstruction*> operands = {operand1, operand2};

  Shape shape = ShapeUtil::MakeShape(S64, {2});
  b.AddInstruction(HloInstruction::CreateConcatenate(shape, operands, 0));

  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected = LiteralUtil::CreateR1<int64>({100, 200});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, ConvertWithSameLayout) {
  HloComputation::Builder b(TestName());

  auto input_literal = LiteralUtil::CreateR2<int32>({{1, 2}, {3, 4}, {5, 6}});
  auto expected =
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  ASSERT_TRUE(LayoutUtil::LayoutsInShapesEqual(input_literal.shape(),
                                               expected.shape()));

  HloInstruction* constant = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  b.AddInstruction(HloInstruction::CreateConvert(expected.shape(), constant));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  EXPECT_TRUE(LiteralTestUtil::Equal(result, expected));
}

TEST_P(HloEvaluatorBf16Test, ConvertWithDifferentLayout) {
  HloComputation::Builder b(TestName());

  auto input_literal = LiteralUtil::CreateR2WithLayout<int32>(
      {{1, 2}, {3, 4}, {5, 6}}, LayoutUtil::MakeLayout({0, 1}));
  auto expected = LiteralUtil::CreateR2WithLayout<float>(
      {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}, LayoutUtil::MakeLayout({1, 0}));
  ASSERT_FALSE(LayoutUtil::LayoutsInShapesEqual(input_literal.shape(),
                                                expected.shape()));

  HloInstruction* constant = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  b.AddInstruction(HloInstruction::CreateConvert(expected.shape(), constant));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  EXPECT_TRUE(LiteralTestUtil::Equal(result, expected));
}

PaddingConfig CreatePaddingConfig(
    std::initializer_list<std::array<int64, 3>> padding_dimensions) {
  PaddingConfig padding_config;

  for (auto& paddings_per_dim : padding_dimensions) {
    auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(paddings_per_dim[0]);
    dimension->set_edge_padding_high(paddings_per_dim[1]);
    dimension->set_interior_padding(paddings_per_dim[2]);
  }
  return padding_config;
}

TEST_F(HloEvaluatorTest, Pad2DIntegerArrayWithZeroDimension) {
  auto operand = LiteralUtil::CreateR2<int32>({{}, {}});
  HloComputation::Builder b(TestName());
  auto operand_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(operand)));

  constexpr int32 kPadValue = 10;
  auto pad_value = LiteralUtil::CreateR0<int32>(kPadValue);
  auto padding_value_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(pad_value)));

  auto padding_config = CreatePaddingConfig({{{1, 0, 2}}, {{0, 2, 1}}});
  Shape shape = ShapeUtil::MakeShape(S32, {5, 2});
  b.AddInstruction(HloInstruction::CreatePad(
      shape, operand_instruction, padding_value_instruction, padding_config));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected = LiteralUtil::CreateR2<int32>(
      {{10, 10}, {10, 10}, {10, 10}, {10, 10}, {10, 10}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, Pad4DFloatArrayWithInteriorPadding) {
  HloComputation::Builder b(TestName());

  Array4D<float> input_array(3, 2, 1, 1, {1, 2, 3, 4, 5, 6});
  auto input = LiteralUtil::CreateR4FromArray4D<float>(input_array);
  HloInstruction* input_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(input)));
  constexpr float kPadValue = 1.5;
  auto pad_value = LiteralUtil::CreateR0<float>(kPadValue);
  HloInstruction* pad_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(pad_value)));

  Shape shape = ShapeUtil::MakeShape(F32, {8, 5, 1, 1});
  auto r4_padding_on_dim0_dim1 =
      CreatePaddingConfig({{{1, 0, 2}}, {{0, 2, 1}}, {{0, 0, 0}}, {{0, 0, 0}}});
  b.AddInstruction(HloInstruction::CreatePad(
      shape, input_instruction, pad_instruction, r4_padding_on_dim0_dim1));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected_array = absl::make_unique<Array4D<float>>(8, 5, 1, 1);
  expected_array->Fill(kPadValue);
  (*expected_array)(1, 0, 0, 0) = 1.0f;
  (*expected_array)(1, 2, 0, 0) = 2.0f;
  (*expected_array)(4, 0, 0, 0) = 3.0f;
  (*expected_array)(4, 2, 0, 0) = 4.0f;
  (*expected_array)(7, 0, 0, 0) = 5.0f;
  (*expected_array)(7, 2, 0, 0) = 6.0f;

  auto expected = LiteralUtil::CreateR4FromArray4D<float>(*expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, NegativePadding2D) {
  HloComputation::Builder b(TestName());

  // input_array:
  // f32[4,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  //  { 9, 10, 11 },
  //  { 13, 14, 15 },
  // }
  auto input_array = absl::make_unique<Array2D<float>>(4, 3);
  input_array->FillUnique(1.0f);
  auto input = LiteralUtil::CreateR2FromArray2D<float>(*input_array);
  HloInstruction* input_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(input)));

  auto pad_value_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.718f)));

  auto r2_padding_on_dim0_dim1 =
      CreatePaddingConfig({{{-1, -2, 0}}, {{-2, 4, 0}}});
  Shape shape = ShapeUtil::MakeShape(F32, {1, 5});
  b.AddInstruction(HloInstruction::CreatePad(shape, input_instruction,
                                             pad_value_instruction,
                                             r2_padding_on_dim0_dim1));

  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  // f32[1,5] { 7.0, 2.718, 2.718, 2.718, 2.718 }
  auto expected_array = absl::make_unique<Array2D<float>>(1, 5);
  (*expected_array)(0, 0) = 7.0f;
  (*expected_array)(0, 1) = 2.718f;
  (*expected_array)(0, 2) = 2.718f;
  (*expected_array)(0, 3) = 2.718f;
  (*expected_array)(0, 4) = 2.718f;
  auto expected = LiteralUtil::CreateR2FromArray2D<float>(*expected_array);

  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, ErrorSpec(0.031250)));
}

TEST_P(HloEvaluatorBf16Test, NegativeAndInteriorPadding2D) {
  HloComputation::Builder b(TestName());

  // f32[4,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  //  { 9, 10, 11 },
  //  { 13, 14, 15 },
  // }
  auto input_array = absl::make_unique<Array2D<float>>(4, 3);
  input_array->FillUnique(1.0f);
  auto input = LiteralUtil::CreateR2FromArray2D<float>(*input_array);
  HloInstruction* input_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(input)));

  auto pad_value_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.718f)));

  PaddingConfig padding_config = MakeNoPaddingConfig(2);

  // Negative padding that results in zero dimensions.
  auto r2_padding_on_dim0_dim1 =
      CreatePaddingConfig({{{-2, -5, 1}}, {{-2, 4, 2}}});

  Shape shape = ShapeUtil::MakeShape(F32, {0, 9});
  b.AddInstruction(HloInstruction::CreatePad(shape, input_instruction,
                                             pad_value_instruction,
                                             r2_padding_on_dim0_dim1));

  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected_array = absl::make_unique<Array2D<float>>(0, 9);
  auto expected = LiteralUtil::CreateR2FromArray2D<float>(*expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DotRank2AndRank1) {
  HloComputation::Builder b(TestName());

  // lhs:
  // f32[4,1] {
  //  { 1 },
  //  { 2 },
  //  { 3 },
  //  { 4 },
  // }
  auto lhs_array = absl::make_unique<Array2D<float>>(4, 1);
  lhs_array->FillUnique(1.0f);
  auto lhs_literal = LiteralUtil::CreateR2FromArray2D<float>(*lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  // rhs:
  // f32[2] { 1, 2 },
  auto rhs_literal = LiteralUtil::CreateR2<float>({{1, 2}});
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {4, 2});
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  b.AddInstruction(HloInstruction::CreateDot(shape, lhs_instruction,
                                             rhs_instruction, dot_dnums,
                                             DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  // clang-format off
  auto expected_array = Array2D<float>({
      {1.f, 2.f},
      {2.f, 4.f},
      {3.f, 6.f},
      {4.f, 8.f},
  });
  // clang-format on
  auto expected = LiteralUtil::CreateR2FromArray2D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DotRank1AndRank2) {
  HloComputation::Builder b(TestName());

  // lhs:
  // f32[3]
  //  { 1, 2, 3 },
  auto lhs_literal = LiteralUtil::CreateR1<float>({1, 2, 3});
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  // rhs:
  // f32[3,2] {
  //  { 1, 2 },
  //  { 3, 4 },
  //  { 5, 6 },
  // }
  auto rhs_array = absl::make_unique<Array2D<float>>(3, 2);
  rhs_array->FillUnique(1.0f);
  auto rhs_literal = LiteralUtil::CreateR2FromArray2D<float>(*rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {2});
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  b.AddInstruction(HloInstruction::CreateDot(shape, lhs_instruction,
                                             rhs_instruction, dot_dnums,
                                             DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected = LiteralUtil::CreateR1<float>({22.f, 28.f});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DotRank2AndRank2) {
  HloComputation::Builder b(TestName());

  // lhs:
  // f32[4,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  //  { 9, 10, 11 },
  //  { 13, 14, 15 },
  // }
  auto lhs_array = absl::make_unique<Array2D<float>>(4, 3);
  lhs_array->FillUnique(1.0f);
  auto lhs_literal = LiteralUtil::CreateR2FromArray2D<float>(*lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  // rhs:
  // f32[3,2] {
  //  { 1, 2 },
  //  { 3, 4 },
  //  { 5, 6 },
  // }
  auto rhs_array = absl::make_unique<Array2D<float>>(3, 2);
  rhs_array->FillUnique(1.0f);
  auto rhs_literal = LiteralUtil::CreateR2FromArray2D<float>(*rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {4, 2});
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  b.AddInstruction(HloInstruction::CreateDot(shape, lhs_instruction,
                                             rhs_instruction, dot_dnums,
                                             DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected_array = Array2D<float>({
      {22.f, 28.f},
      {58.f, 76.f},
      {94.f, 124.f},
      {130.f, 172.f},
  });
  auto expected = LiteralUtil::CreateR2FromArray2D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DotRank4AndRank4) {
  HloComputation::Builder b(TestName());

  auto lhs_array = absl::make_unique<Array4D<float>>(2, 2, 3, 1);
  lhs_array->FillIota(1.0f);
  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(*lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  auto rhs_array = absl::make_unique<Array4D<float>>(2, 2, 3, 1);
  rhs_array->FillIota(2.0f);
  auto rhs_literal = LiteralUtil::CreateR4FromArray4D<float>(*rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {2, 1, 1});
  DotDimensionNumbers dot_dnums;

  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(0);
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_rhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(2);
  b.AddInstruction(HloInstruction::CreateDot(shape, lhs_instruction,
                                             rhs_instruction, dot_dnums,
                                             DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();
  float expected_1 = 0;
  for (float i = 1.0f; i < 7.0f; ++i) {
    expected_1 += i * i + i;
  }
  float expected_2 = 0;
  for (float i = 7.0f; i < 13.0f; ++i) {
    expected_2 += i * i + i;
  }
  auto expected_array = Array3D<float>({{{expected_1}}, {{expected_2}}});
  auto expected = LiteralUtil::CreateR3FromArray3D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, SimpleConv1D) {
  HloComputation::Builder b(TestName());

  Array3D<float> lhs_array = {{{1, 2, 3}}};
  auto lhs_literal = LiteralUtil::CreateR3FromArray3D<float>(lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  Array3D<float> rhs_array = {{{3.f, 4.f}}};
  auto rhs_literal = LiteralUtil::CreateR3FromArray3D<float>(rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(1);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.set_input_feature_dimension(1);
  dnums.set_output_feature_dimension(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);

  dnums.set_kernel_output_feature_dimension(0);
  dnums.set_kernel_input_feature_dimension(1);
  dnums.add_kernel_spatial_dimensions(2);

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 3});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  Array3D<float> expected_array = {{{11.f, 18.f, 9.f}}};
  auto expected = LiteralUtil::CreateR3FromArray3D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, Simple4x4Conv2DWith2x2Kernel) {
  HloComputation::Builder b(TestName());

  Array4D<float> lhs_array(1, 1, 4, 4);
  // clang-format off
  lhs_array.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  Array4D<float> rhs_array(1, 1, 2, 2);
  // clang-format off
  rhs_array.FillWithYX(Array2D<float>({
    {5, 6},
    {7, 8},
  }));
  // clang-format on
  auto rhs_literal = LiteralUtil::CreateR4FromArray4D<float>(rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(1);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;

  ConvolutionDimensionNumbers dnums =
      XlaBuilder::CreateDefaultConvDimensionNumbers(2);

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 4, 4});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  Array4D<float> expected_array(1, 1, 4, 4);
  // clang-format off
  expected_array.FillWithYX(Array2D<float>({
    {100, 126, 152,  76},
    {204, 230, 256, 124},
    {308, 334, 360, 172},
    {149, 160, 171,  80},
  }));
  // clang-format on
  auto expected = LiteralUtil::CreateR4FromArray4D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, Conv2DGeneralDimensionsReversed) {
  HloComputation::Builder b(TestName());

  // clang-format off
  // Input dimensions: [feature=2, height=3, batch=1, width=4]
  Array4D<float> input({
    {{{1, 2, 3, 4}},
     {{5, 6, 7, 8}},
     {{9, 10, 11, 12}}},
    {{{13, 14, 15, 16}},
     {{17, 18, 19, 20}},
     {{21, 22, 23, 24}}}
  });
  // Weight dimensions:
  // [kernel_output_feature=1, width=3, kernel_input_feature=2, height=3]
  Array4D<float> weight({{
    {{1, 7, 13},
     {4, 10, 16}},
    {{2, 8, 14},
     {5, 11, 17}},
    {{3, 9, 15},
     {6, 12, 18}}
  }});
  // clang-format on

  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(input);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  auto rhs_literal = LiteralUtil::CreateR4FromArray4D<float>(weight);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));
  rhs_instruction = b.AddInstruction(HloInstruction::CreateReverse(
      rhs_instruction->shape(), rhs_instruction, {3, 1}));

  Window window;
  WindowDimension dim;
  dim.set_size(3);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(0);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  dim.set_window_reversal(true);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(2);
  dnums.set_output_batch_dimension(2);
  dnums.set_input_feature_dimension(0);
  dnums.set_output_feature_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(3);
  dnums.add_output_spatial_dimensions(3);

  dnums.set_kernel_output_feature_dimension(0);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.add_kernel_spatial_dimensions(1);

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 2});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  // clang-format off
  // Result dimensions: [feature=1, height=1, batch=1, width=2]
  Array4D<float> expected_array({{{{2514, 2685}}}});
  Array4D<float> expected_array_bf16({{{{2512, 2688}}}});
  // clang-format on
  auto expected = LiteralUtil::CreateR4FromArray4D<float>(
      use_bfloat16_ ? expected_array_bf16 : expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, Conv2DGeneralDimensions) {
  HloComputation::Builder b(TestName());

  // clang-format off
  // Input dimensions: [feature=2, height=3, batch=1, width=4]
  Array4D<float> input({
    {{{1, 2, 3, 4}},
     {{5, 6, 7, 8}},
     {{9, 10, 11, 12}}},
    {{{13, 14, 15, 16}},
     {{17, 18, 19, 20}},
     {{21, 22, 23, 24}}}
  });
  // Weight dimensions:
  // [kernel_output_feature=1, width=3, kernel_input_feature=2, height=3]
  Array4D<float> weight({{
    {{1, 7, 13},
     {4, 10, 16}},
    {{2, 8, 14},
     {5, 11, 17}},
    {{3, 9, 15},
     {6, 12, 18}}
  }});
  // clang-format on

  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(input);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  auto rhs_literal = LiteralUtil::CreateR4FromArray4D<float>(weight);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Window window;
  WindowDimension dim;
  dim.set_size(3);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(0);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(2);
  dnums.set_output_batch_dimension(2);
  dnums.set_input_feature_dimension(0);
  dnums.set_output_feature_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(3);
  dnums.add_output_spatial_dimensions(3);

  dnums.set_kernel_output_feature_dimension(0);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.add_kernel_spatial_dimensions(1);

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 2});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  // clang-format off
  // Result dimensions: [feature=1, height=1, batch=1, width=2]
  Array4D<float> expected_array({{{{2514, 2685}}}});
  Array4D<float> expected_array_bf16({{{{2512, 2688}}}});
  // clang-format on
  auto expected = LiteralUtil::CreateR4FromArray4D<float>(
      use_bfloat16_ ? expected_array_bf16 : expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DilatedBaseConv2DWithHighPadding) {
  HloComputation::Builder b(TestName());

  Array4D<float> lhs_array(1, 1, 4, 4);
  // clang-format off
  lhs_array.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  Array4D<float> rhs_array(1, 1, 2, 2);
  // clang-format off
  rhs_array.FillWithYX(Array2D<float>({
    {5, 6},
    {7, 8},
  }));
  // clang-format on
  auto rhs_literal = LiteralUtil::CreateR4FromArray4D<float>(rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(1);
  dim.set_window_dilation(1);
  dim.set_base_dilation(2);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;

  ConvolutionDimensionNumbers dnums =
      XlaBuilder::CreateDefaultConvDimensionNumbers(2);

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 7, 7});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  Array4D<float> expected_array(1, 1, 7, 7);
  expected_array.FillWithYX(Array2D<float>({
      {5, 12, 10, 18, 15, 24, 20},
      {35, 48, 42, 56, 49, 64, 56},
      {25, 36, 30, 42, 35, 48, 40},
      {63, 80, 70, 88, 77, 96, 84},
      {45, 60, 50, 66, 55, 72, 60},
      {91, 112, 98, 120, 105, 128, 112},
      {65, 84, 70, 90, 75, 96, 80},
  }));
  auto expected = LiteralUtil::CreateR4FromArray4D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DilatedBaseConv2DWithLowAndHighPadding) {
  HloComputation::Builder b(TestName());

  Array4D<float> lhs_array(1, 1, 4, 4);
  // clang-format off
  lhs_array.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  Array4D<float> rhs_array(1, 1, 2, 2);
  // clang-format off
  rhs_array.FillWithYX(Array2D<float>({
    {5, 6},
    {7, 8},
  }));
  // clang-format on
  auto rhs_literal = LiteralUtil::CreateR4FromArray4D<float>(rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(1);
  dim.set_padding_high(1);
  dim.set_window_dilation(1);
  dim.set_base_dilation(2);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;

  ConvolutionDimensionNumbers dnums =
      XlaBuilder::CreateDefaultConvDimensionNumbers(2);

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 8, 8});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  Array4D<float> expected_array(1, 1, 8, 8);
  expected_array.FillWithYX(Array2D<float>({
      {8, 7, 16, 14, 24, 21, 32, 28},
      {6, 5, 12, 10, 18, 15, 24, 20},
      {40, 35, 48, 42, 56, 49, 64, 56},
      {30, 25, 36, 30, 42, 35, 48, 40},
      {72, 63, 80, 70, 88, 77, 96, 84},
      {54, 45, 60, 50, 66, 55, 72, 60},
      {104, 91, 112, 98, 120, 105, 128, 112},
      {78, 65, 84, 70, 90, 75, 96, 80},
  }));
  auto expected = LiteralUtil::CreateR4FromArray4D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test,
       DilatedWindowAndBaseConv2DWithDifferentLowAndHighPaddingAndStrides) {
  HloComputation::Builder b(TestName());

  Array4D<float> lhs_array(1, 1, 4, 4);
  // clang-format off
  lhs_array.FillWithYX(Array2D<float>({
    {1,  2,  3,  4 },
    {5,  6,  7,  8 },
    {9,  10, 11, 12},
    {13, 14, 15, 16},
  }));
  // clang-format on
  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  Array4D<float> rhs_array(1, 1, 2, 3);
  // clang-format off
  rhs_array.FillWithYX(Array2D<float>({
    {5, 6, 7},
    {8, 9, 10},
  }));
  // clang-format on
  auto rhs_literal = LiteralUtil::CreateR4FromArray4D<float>(rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(2);
  dim.set_padding_high(2);
  dim.set_window_dilation(2);
  dim.set_base_dilation(2);
  *window.add_dimensions() = dim;
  dim.set_size(3);
  dim.set_stride(3);
  dim.set_padding_low(2);
  dim.set_padding_high(-1);
  dim.set_window_dilation(1);
  dim.set_base_dilation(3);
  *window.add_dimensions() = dim;

  ConvolutionDimensionNumbers dnums =
      XlaBuilder::CreateDefaultConvDimensionNumbers(2);

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 9, 3});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  Array4D<float> expected_array(1, 1, 9, 3);
  expected_array.FillWithYX(Array2D<float>({
      {10, 20, 30},
      {0, 0, 0},
      {57, 74, 91},
      {0, 0, 0},
      {125, 142, 159},
      {0, 0, 0},
      {193, 210, 227},
      {0, 0, 0},
      {91, 98, 105},
  }));
  auto expected = LiteralUtil::CreateR4FromArray4D<float>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, Conv2DGroupedConvolution) {
  HloComputation::Builder b(TestName());
  std::vector<int64> input_dims = {1, 2, 2, 4};
  std::vector<int64> filter_dims = {2, 2, 2, 8};
  Shape input_shape = ShapeUtil::MakeShapeWithType<float>(input_dims);
  Shape filter_shape = ShapeUtil::MakeShapeWithType<float>(filter_dims);
  // Tensorflow dimension numbers for 2D convolution.
  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.set_input_feature_dimension(3);
  dnums.set_output_feature_dimension(3);
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);

  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(0);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;

  std::vector<float> input_elems(ShapeUtil::ElementsIn(input_shape));
  std::iota(input_elems.begin(), input_elems.end(), -7);
  auto input_r1 = LiteralUtil::CreateR1<float>(input_elems);
  auto input_r4 = input_r1.Reshape(input_dims).ConsumeValueOrDie();
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(input_r4)));

  std::vector<float> filter_elems(ShapeUtil::ElementsIn(filter_shape));
  std::iota(filter_elems.begin(), filter_elems.end(), -31);
  auto filter_r1 = LiteralUtil::CreateR1<float>(filter_elems);
  auto filter_r4 = filter_r1.Reshape(filter_dims).ConsumeValueOrDie();
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(filter_r4)));

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 8});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction,
      /*feature_group_count=*/2, /*batch_group_count=*/1, window, dnums,
      DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  Array4D<float> expected_array(1, 1, 1, 8);
  expected_array.FillWithYX(
      Array2D<float>({{668, 664, 660, 656, 668, 680, 692, 704}}));
  auto expected = LiteralUtil::CreateR4FromArray4D<float>(expected_array);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

class HloEvaluatorPreciseReduceTest : public HloTestBase {};

// Tests that Reduce doesn't lose precision when adding many numbers (because
// it accumulates its result in a double).
TEST_F(HloEvaluatorPreciseReduceTest, AddReductionPrecisionTest) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder b(TestName());

  constexpr int kNumElements = 1 << 25;  // float += 1 saturates at 1<<24
  std::vector<float> v(kNumElements, 1.0f);
  HloInstruction* arg_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(v)));
  HloInstruction* init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));
  auto add_func = m->AddEmbeddedComputation(add_computation.Build());

  HloInstruction* reduce_instruction = b.AddInstruction(
      HloInstruction::CreateReduce(scalar_shape, arg_instruction, init_value,
                                   /*dimensions_to_reduce=*/{0}, add_func));
  m->AddEntryComputation(b.Build());

  HloEvaluator hlo_eval;
  Literal result = hlo_eval.Evaluate(reduce_instruction).ConsumeValueOrDie();
  LiteralTestUtil::ExpectR0Equal<float>(kNumElements, result);
}

// Reducing many numbers should be fast because it doesn't create
// intermediate Literals; the microbenchmark should finish in < 1 msec.
void BM_ReducePrecisely(int num_iters) {
  tensorflow::testing::StopTiming();
  HloComputation::Builder b("BM_ReducePrecisely");
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsFromFlags());
  HloModule module("BM_ReducePrecisely", config);

  constexpr int kNumElements = 1 << 25;  // float += 1 saturates at 1<<24
  std::vector<float> v(kNumElements, 1.0f);
  HloInstruction* arg_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(v)));
  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));
  auto add_func = module.AddEmbeddedComputation(add_computation.Build());

  HloInstruction* reduce_instruction = b.AddInstruction(
      HloInstruction::CreateReduce(scalar_shape, arg_instruction, init_value,
                                   /*dimensions_to_reduce=*/{0}, add_func));
  module.AddEntryComputation(b.Build());

  HloEvaluator hlo_eval;
  tensorflow::testing::StartTiming();
  hlo_eval.Evaluate(reduce_instruction).ConsumeValueOrDie();
  tensorflow::testing::StopTiming();
}

BENCHMARK(BM_ReducePrecisely);

TEST_P(HloEvaluatorBf16Test, ReduceAdd) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto arg_array = absl::make_unique<Array2D<float>>(2, 3);
  arg_array->FillUnique(1.0f);
  auto arg_literal = LiteralUtil::CreateR2FromArray2D<float>(*arg_array);

  HloInstruction* arg_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));
  auto add_func = m_->AddEmbeddedComputation(add_computation.Build());

  Shape shape = ShapeUtil::MakeShape(F32, {2});
  b.AddInstruction(
      HloInstruction::CreateReduce(shape, arg_instruction, init_value,
                                   /*dimensions_to_reduce=*/{1}, add_func));

  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected = LiteralUtil::CreateR1<float>({6, 18});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowMax) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto arg_array = absl::make_unique<Array2D<float>>(2, 3);
  arg_array->FillUnique(1.0f);
  auto arg_literal = LiteralUtil::CreateR2FromArray2D<float>(*arg_array);

  HloInstruction* arg_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));

  HloComputation::Builder max_computation("max");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = max_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = max_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  max_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kMaximum, param_lhs, param_rhs));
  auto max_func = m_->AddEmbeddedComputation(max_computation.Build());

  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(0);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;

  Shape shape = ShapeUtil::MakeShape(F32, {1, 2});
  b.AddInstruction(HloInstruction::CreateReduceWindow(
      shape, arg_instruction, init_value, window, max_func));

  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected = LiteralUtil::CreateR2<float>({{6, 7}});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowMaxWindowDilation) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[3,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  //  { 9, 10, 11 },
  // }
  auto arg_array = absl::make_unique<Array2D<float>>(3, 3);
  arg_array->FillUnique(1.0f);
  auto arg_literal = LiteralUtil::CreateR2FromArray2D<float>(*arg_array);

  HloInstruction* arg_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));

  HloComputation::Builder max_computation("max");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = max_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = max_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  max_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kMaximum, param_lhs, param_rhs));
  auto max_func = m_->AddEmbeddedComputation(max_computation.Build());

  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(0);
  dim.set_window_dilation(2);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1});
  b.AddInstruction(HloInstruction::CreateReduceWindow(
      shape, arg_instruction, init_value, window, max_func));

  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected = LiteralUtil::CreateR2<float>({{11}});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowAdd) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto arg_array = absl::make_unique<Array2D<float>>(2, 3);
  arg_array->FillUnique(1.0f);
  auto arg_literal = LiteralUtil::CreateR2FromArray2D<float>(*arg_array);

  HloInstruction* arg_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));
  auto add_func = m_->AddEmbeddedComputation(add_computation.Build());

  Window window;
  WindowDimension dim;
  dim.set_size(1);
  dim.set_stride(1);
  dim.set_padding_low(0);
  dim.set_padding_high(0);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_low(1);
  dim.set_padding_high(0);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;

  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  b.AddInstruction(HloInstruction::CreateReduceWindow(
      shape, arg_instruction, init_value, window, add_func));

  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected = LiteralUtil::CreateR2<float>({{1, 3, 5}, {5, 11, 13}});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowAdd6D) {
  HloComputation::Builder b(TestName());

  // arg: f32[4,4,4,4,4,4] full of ones. Using small dims to limit run-time.
  std::vector<int64> input_dims(6, 4);
  Literal arg_literal =
      LiteralUtil::CreateFullWithDescendingLayout<float>(input_dims, 1.0f);

  HloInstruction* arg_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));
  auto add_func = m_->AddEmbeddedComputation(add_computation.Build());

  Window window;

  WindowDimension trivial_dim;
  trivial_dim.set_size(1);
  trivial_dim.set_stride(1);
  trivial_dim.set_padding_low(0);
  trivial_dim.set_padding_high(0);
  trivial_dim.set_window_dilation(1);
  trivial_dim.set_base_dilation(1);

  WindowDimension active_dim;
  active_dim.set_size(2);
  active_dim.set_stride(1);
  active_dim.set_padding_low(0);
  active_dim.set_padding_high(0);
  active_dim.set_window_dilation(1);
  active_dim.set_base_dilation(1);

  *window.add_dimensions() = trivial_dim;
  *window.add_dimensions() = active_dim;
  *window.add_dimensions() = active_dim;
  *window.add_dimensions() = active_dim;
  *window.add_dimensions() = trivial_dim;
  *window.add_dimensions() = trivial_dim;

  Shape shape = ShapeUtil::MakeShape(F32, {4, 3, 3, 3, 4, 4});
  b.AddInstruction(HloInstruction::CreateReduceWindow(
      shape, arg_instruction, init_value, window, add_func));

  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  std::vector<int64> output_dims = {4, 3, 3, 3, 4, 4};
  Literal result_literal =
      LiteralUtil::CreateFullWithDescendingLayout<float>(output_dims, 8.0f);
  EXPECT_TRUE(LiteralTestUtil::Equal(result_literal, result));
}

TEST_P(HloEvaluatorBf16Test, StridedSlice) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[3,5] {
  //  { 1, 2, 3, 4, 5 },
  //  { 9, 10, 11, 12, 13 },
  //  { 17, 18, 19, 20, 21 },
  // }
  auto operand_array = absl::make_unique<Array2D<float>>(3, 5);
  operand_array->FillUnique(1.0f);
  auto operand_literal =
      LiteralUtil::CreateR2FromArray2D<float>(*operand_array);

  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {2, 1});
  b.AddInstruction(HloInstruction::CreateSlice(shape, operand,
                                               /*start_indices=*/{0, 2},
                                               /*limit_indices=*/{3, 5},
                                               /*strides=*/{2, 3}));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected = LiteralUtil::CreateR2<float>({
      {3},
      {19},
  });

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DynamicSlice) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,4] {
  //  { 1, 2, 3, 4 },
  //  { 5, 6, 7, 8 },
  // }
  auto operand_array = absl::make_unique<Array2D<float>>(2, 4);
  operand_array->FillUnique(1.0f);
  auto operand_literal =
      LiteralUtil::CreateR2FromArray2D<float>(*operand_array);

  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  auto zero = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto one = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));

  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  b.AddInstruction(
      HloInstruction::CreateDynamicSlice(shape, operand, {zero, one}, {2, 3}));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected = LiteralUtil::CreateR2<float>({
      {2, 3, 4},
      {6, 7, 8},
  });

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Verifies that the HloEvaluator's implementation goes along with existing
// backends' behavior, although this is not required by the spec.
TEST_P(HloEvaluatorBf16Test, DynamicSliceModSlice) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,4] {
  //  { 1, 2, 3, 4 },
  //  { 5, 6, 7, 8 },
  // }
  auto operand_array = absl::make_unique<Array2D<float>>(2, 4);
  operand_array->FillUnique(1.0f);
  auto operand_literal =
      LiteralUtil::CreateR2FromArray2D<float>(*operand_array);

  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  auto two = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(2)));
  auto one = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));

  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  b.AddInstruction(
      HloInstruction::CreateDynamicSlice(shape, operand, {two, one}, {2, 3}));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected = LiteralUtil::CreateR2<float>({
      {2, 3, 4},
      {6, 7, 8},
  });

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, DynamicSliceUpdate) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto operand_array = absl::make_unique<Array2D<double>>(2, 3);
  operand_array->FillUnique(1.0);
  auto operand_literal =
      LiteralUtil::CreateR2FromArray2D<double>(*operand_array);

  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  auto zero = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto one = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));

  auto update = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<double>({{-2.0, -3.0}, {-6.0, -7.0}})));

  Shape shape = ShapeUtil::MakeShape(F64, {2, 3});
  b.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      shape, operand, update, {zero, one}));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected = LiteralUtil::CreateR2<double>({
      {1, -2, -3},
      {5, -6, -7},
  });

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, SetAndGetTuples) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto operand_array = absl::make_unique<Array2D<double>>(2, 3);
  operand_array->FillUnique(1.0);
  auto operand_literal2 =
      LiteralUtil::CreateR2FromArray2D<double>(*operand_array);

  HloInstruction* operand2 = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal2)));
  HloInstruction* operand1 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int64>({0, 1})));

  auto tuple =
      b.AddInstruction(HloInstruction::CreateTuple({operand1, operand2}));

  Shape shape = ShapeUtil::MakeShape(F64, {2, 3});
  b.AddInstruction(HloInstruction::CreateGetTupleElement(shape, tuple, 1));

  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto expected = LiteralUtil::CreateR2<double>({
      {1, 2, 3},
      {5, 6, 7},
  });

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, SetAndGetNestedTuples) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto operand_array = absl::make_unique<Array2D<double>>(2, 3);
  operand_array->FillUnique(1.0);

  HloInstruction* operand2 = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2FromArray2D<double>(*operand_array)));
  HloInstruction* operand1 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int64>({0, 1})));

  auto tuple1 =
      b.AddInstruction(HloInstruction::CreateTuple({operand1, operand2}));
  auto tuple2 =
      b.AddInstruction(HloInstruction::CreateTuple({operand2, operand2}));

  auto outer_tuple =
      b.AddInstruction(HloInstruction::CreateTuple({tuple1, tuple2}));

  b.AddInstruction(
      HloInstruction::CreateGetTupleElement(tuple2->shape(), outer_tuple, 1));

  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  auto result_inner_literal =
      LiteralUtil::CreateR2FromArray2D<double>(*operand_array);
  auto expected =
      LiteralUtil::MakeTuple({&result_inner_literal, &result_inner_literal});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, Reverse) {
  HloComputation::Builder b(TestName());

  // Input shape is float[4x3x2x1].
  // clang-format off
  Array4D<float> input({
    {{{1.0f}, {2.0f}},
     {{3.0f}, {4.0f}},
     {{5.0f}, {6.0f}}},
    {{{7.0f}, {8.0f}},
     {{9.0f}, {10.0f}},
     {{11.0f}, {12.0f}}},
    {{{13.0f}, {14.0f}},
     {{15.0f}, {16.0f}},
     {{17.0f}, {18.0f}}},
    {{{19.0f}, {20.0f}},
     {{21.0f}, {22.0f}},
     {{23.0f}, {24.0f}}},
  });
  // clang-format on
  auto operand_literal = LiteralUtil::CreateR4FromArray4D<float>(input);
  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  const Shape shape = ShapeUtil::MakeShape(F32, {4, 3, 2, 1});
  b.AddInstruction(HloInstruction::CreateReverse(shape, operand, {0, 1}));
  m_->AddEntryComputation(b.Build());

  Literal result = Evaluate();

  // clang-format off
  auto expected = LiteralUtil::CreateR4FromArray4D<float>({
    {{{23.0f}, {24.0f}},
     {{21.0f}, {22.0f}},
     {{19.0f}, {20.0f}}},

    {{{17.0f}, {18.0f}},
     {{15.0f}, {16.0f}},
     {{13.0f}, {14.0f}}},

    {{{11.0f}, {12.0f}},
     {{9.0f}, {10.0f}},
     {{7.0f}, {8.0f}}},

    {{{5.0f}, {6.0f}},
     {{3.0f}, {4.0f}},
     {{1.0f}, {2.0f}}},
  });
  // clang-format on

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, EvaluateWithSubstitutions) {
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  HloInstruction* param0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param0"));
  HloInstruction* square = b.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kMultiply, param0, param0));
  HloInstruction* add = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, square));

  // Evaluate add with param0 = {1, 2, 3, 4}, square = {10, 20, 30, 40}.
  HloEvaluator evaluator;
  Literal param0_literal = LiteralUtil::CreateR1<float>({1, 2, 3, 4});
  Literal square_literal = LiteralUtil::CreateR1<float>({10, 20, 30, 40});
  auto result = evaluator.EvaluateWithSubstitutions(
      add, {{param0, &param0_literal}, {square, &square_literal}});
  TF_ASSERT_OK(result.status());
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({11, 22, 33, 44}), result.ValueOrDie()));
}

// Check that EvaluateWithSubstitutions works if one of the operands to the op
// we're evaluating is a constant.
TEST_P(HloEvaluatorBf16Test, EvaluateWithSubstitutionsWithConstantOperand) {
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  HloInstruction* param0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param0"));
  HloInstruction* square = b.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kMultiply, param0, param0));
  HloInstruction* constant = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1, 2, 3, 4})));
  HloInstruction* add = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, constant, square));

  // Evaluate add with square = {10, 20, 30, 40}.
  HloEvaluator evaluator;
  Literal square_literal = LiteralUtil::CreateR1<float>({10, 20, 30, 40});
  auto result =
      evaluator.EvaluateWithSubstitutions(add, {{square, &square_literal}});
  TF_ASSERT_OK(result.status());
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({11, 22, 33, 44}), result.ValueOrDie()));
}

TEST_F(HloEvaluatorTest, EvaluateGather_TensorFlowGatherV1) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,3] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1, 3}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32>({0, 2});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {7, 8, 9}}),
      Evaluate({&operand, &start_indices})));
}

TEST_F(HloEvaluatorTest, EvaluateGather_TensorFlowGatherV2) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherV2

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[3,2] gather(operand, indices),
      offset_dims={0},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3, 1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32>({0, 2});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{1, 3}, {4, 6}, {7, 9}}),
      Evaluate({&operand, &start_indices})));
}

TEST_F(HloEvaluatorTest, EvaluateGather_TensorFlowGatherMultipleBatchDims) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,3,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=2,
      slice_sizes={3, 1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{0, 2}, {2, 1}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR3<int32>(
          {{{1, 3}, {4, 6}, {7, 9}}, {{3, 2}, {6, 5}, {9, 8}}}),
      Evaluate({&operand, &start_indices})));
}

TEST_F(HloEvaluatorTest, EvaluateGather_TensorFlowGatherNd) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=1,
      slice_sizes={1,1,2}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{0, 0}, {1, 0}});
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR2<int32>({{-1, 1}, {-4, 4}}),
                             Evaluate({&operand, &start_indices})));
}

TEST_F(HloEvaluatorTest,
       EvaluateGather_TensorFlowGatherNdNonDefaultIndexVectorDim) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1,2}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{0, 0}, {1, 0}});
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR2<int32>({{-2, 2}, {-1, 1}}),
                             Evaluate({&operand, &start_indices})));
}

TEST_F(HloEvaluatorTest, EvaluateGather_DynamicSlice) {
  const char* hlo_text = R"(
HloModule DynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[1,1] gather(operand, indices),
      offset_dims={0,1},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32>({1, 1});
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR2<int32>({{5}}),
                                     Evaluate({&operand, &start_indices})));
}

TEST_F(HloEvaluatorTest, EvaluateGather_BatchDynamicSlice) {
  const char* hlo_text = R"(
HloModule BatchDynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,1,1] gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32>({{2, 1}, {1, 1}});
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR3<int32>({{{8}}, {{5}}}),
                             Evaluate({&operand, &start_indices})));
}

TEST_F(HloEvaluatorTest, EvaluateGather_ZeroDimBounds) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,0] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,0] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1, 0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand = LiteralUtil::CreateR2<int32>({{}, {}, {}});
  Literal start_indices = LiteralUtil::CreateR1<int32>({0, 2});
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR2<int32>({{}, {}}),
                                     Evaluate({&operand, &start_indices})));
}

TEST_F(HloEvaluatorTest, EvaluateGather_NoOutputWindowDims) {
  const string hlo_text = R"(
HloModule GatherXd

ENTRY main {
  operand = s32[3] parameter(0)
  indices = s32[2,2,1] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      offset_dims={},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=2,
      slice_sizes={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  Literal operand = LiteralUtil::CreateR1<int32>({0, 1, 2});
  Literal start_indices =
      LiteralUtil::CreateR3<int32>({{{0}, {1}}, {{2}, {1}}});
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR2<int32>({{0, 1}, {2, 1}}),
                             Evaluate({&operand, &start_indices})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatterV1_Update) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{10, 20, 30}, {4, 5, 6}, {70, 80, 90}}),
      Evaluate({&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatterV2_Update) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterV2

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[3,2] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={0},
      inserted_window_dims={1},
      scatter_dims_to_operand_dims={1},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32>({{10, 30}, {40, 60}, {70, 90}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{10, 2, 30}, {40, 5, 60}, {70, 8, 90}}),
      Evaluate({&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatter_Add) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{11, 22, 33}, {4, 5, 6}, {77, 88, 99}}),
      Evaluate({&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatter_Mul) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter

mul_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT mul = s32[] multiply(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=mul_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{10, 40, 90}, {4, 5, 6}, {490, 640, 810}}),
      Evaluate({&operand, &scatter_indices, &updates})));
}

TEST_P(HloEvaluatorBf16Test, EvaluateScatter_TensorFlowScatter_F32) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter

add_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(f32[] lhs, f32[] rhs)
}

ENTRY main {
  operand = f32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = f32[2,3] parameter(2)
  ROOT scatter = f32[3,3] scatter(operand, indices, updates),
      to_apply=add_f32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand = LiteralUtil::CreateR2<float>(
      {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({2, 1});
  Literal updates =
      LiteralUtil::CreateR2<float>({{0.4, 1.1, 0.7}, {2.3, 3.1, 1.6}});
  EXPECT_TRUE(LiteralTestUtil::Near(
      LiteralUtil::CreateR2<float>(
          {{1.1, 2.2, 3.3}, {6.7, 8.6, 8.2}, {8.1, 9.9, 10.6}}),
      Evaluate({&operand, &scatter_indices, &updates}), ErrorSpec{0.1, 0.01}));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatter_RepeatedIndices) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({1, 1});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {84, 105, 126}, {7, 8, 9}}),
      Evaluate({&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatter_MultipleBatchDims) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterMultipleBatchDims

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,3,2] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={1},
      scatter_dims_to_operand_dims={1},
      index_vector_dim=2
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{0, 2}, {2, 1}});
  Literal updates = LiteralUtil::CreateR3<int32>(
      {{{10, 30}, {40, 60}, {70, 90}}, {{5, 5}, {5, 5}, {5, 5}}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{11, 7, 38}, {44, 10, 71}, {77, 13, 104}}),
      Evaluate({&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatterNd) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterNd

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,2] parameter(2)
  ROOT scatter = s32[3,3,2] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{0, 0}, {1, 0}});
  Literal updates = LiteralUtil::CreateR2<int32>({{-10, 10}, {-40, 40}});
  Literal expected =
      LiteralUtil::CreateR3<int32>({{{-10, 10}, {-2, 2}, {-3, 3}},  //
                                    {{-40, 40}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      expected, Evaluate({&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest,
       EvaluateScatter_TensorFlowScatterNd_NonDefaultIndexVectorDim) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterNdNonDefaultIndexVectorDim

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,2] parameter(2)
  ROOT scatter = s32[3,3,2] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{0, 0}, {1, 0}});
  Literal updates = LiteralUtil::CreateR2<int32>({{-10, 10}, {-20, 20}});
  Literal expected =
      LiteralUtil::CreateR3<int32>({{{-20, 20}, {-10, 10}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},      //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      expected, Evaluate({&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_DynamicUpdateSlice) {
  const char* hlo_text = R"(
HloModule DynamicUpdateSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[1,1] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={0,1},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({1, 1});
  Literal updates = LiteralUtil::CreateR2<int32>({{10}});
  Literal expected =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 10, 6}, {7, 8, 9}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      expected, Evaluate({&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_BatchDynamicUpdateSlice) {
  const char* hlo_text = R"(
HloModule BatchDynamicUpdateSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,1,1] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{2, 1}, {1, 1}});
  Literal updates = LiteralUtil::CreateR3<int32>({{{10}}, {{20}}});
  Literal expected =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 20, 6}, {7, 10, 9}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      expected, Evaluate({&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_ZeroDimBounds) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter_ZeroDimBounds

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,0] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,0] parameter(2)
  ROOT scatter = s32[3,0] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand = LiteralUtil::CreateR2<int32>({{}, {}, {}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32>({{}, {}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      operand, Evaluate({&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_NoUpdateWindowDims) {
  const string hlo_text = R"(
HloModule Scatter_NoUpdateWindowDims

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3] parameter(0)
  indices = s32[2,2,1] parameter(1)
  updates = s32[2,2] parameter(2)
  ROOT scatter = s32[3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=2
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  Literal operand = LiteralUtil::CreateR1<int32>({0, 1, 2});
  Literal scatter_indices =
      LiteralUtil::CreateR3<int32>({{{0}, {1}}, {{2}, {1}}});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20}, {30, 40}});
  Literal expected = LiteralUtil::CreateR1<int32>({10, 61, 32});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      expected, Evaluate({&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_NegativeIndices) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter_NegativeIndices

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  // No updates should happen for the negative indices.
  Literal scatter_indices = LiteralUtil::CreateR1<int32>({-1, 2});
  Literal updates = LiteralUtil::CreateR2<int32>({{10, 20, 30}, {70, 80, 90}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {77, 88, 99}}),
      EvaluateWithModule(module.get(),
                         {&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_OobIndices) {
  const string hlo_text = R"(
HloModule BatchDynamicUpdateSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3]{1,0} parameter(0)
  indices = s32[6,2]{1,0} parameter(1)
  updates = s32[6,1,1]{2,1,0} parameter(2)
  ROOT scatter = s32[3,3]{1,0} scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  // No updates should happen for the OOB indices.
  Literal scatter_indices = LiteralUtil::CreateR2<int32>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {2147483647, 1}, {1, 2}});
  Literal updates = LiteralUtil::CreateR3<int32>(
      {{{10}}, {{20}}, {{30}}, {{40}}, {{50}}, {{60}}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 30, 60}, {7, 20, 9}}),
      EvaluateWithModule(module.get(),
                         {&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_OobUpdateWindow) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterNd_OobUpdateWindow

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[1,2] parameter(1)
  updates = s32[1,2,2] parameter(2)
  ROOT scatter = s32[3,3,2] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                    {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                    {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32>({{0, 2}});
  Literal updates = LiteralUtil::CreateR3<int32>({{{-10, 10}, {-40, 40}}});
  // Given the update window size of 2,2 and the index of 0,2, the update window
  // will be OOB. So, nothing should be updated.
  Literal expected = operand.Clone();
  EXPECT_TRUE(LiteralTestUtil::Equal(
      expected, EvaluateWithModule(module.get(),
                                   {&operand, &scatter_indices, &updates})));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise comparison with 2 bfloat16 operands.
TEST_F(HloEvaluatorTest, DoesCompareBF16) {
  // lhs >= rhs
  auto lhs = LiteralUtil::CreateR2<bfloat16>(
      {{bfloat16(0.25), bfloat16(0.35), bfloat16(0.125)},
       {bfloat16(-0.25), bfloat16(-0.35), bfloat16(-0.125)}});
  auto rhs = LiteralUtil::CreateR2<bfloat16>(
      {{bfloat16(0.5), bfloat16(0.125), bfloat16(0.125)},
       {bfloat16(0.25), bfloat16(-0.375), bfloat16(-0.127)}});
  auto expected =
      LiteralUtil::CreateR2<bool>({{false, true, true}, {false, true, true}});
  TestBinaryOp(HloOpcode::kGe, std::move(expected), std::move(lhs),
               std::move(rhs));
}

TEST_P(HloEvaluatorBf16Test, Bf16Reduction) {
  const string hlo_text = R"(
HloModule Bf16Reduction

add_bf16 (lhs: bf16[], rhs: bf16[]) -> bf16[] {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(bf16[] lhs, bf16[] rhs)
}

ENTRY main {
  arg0 = bf16[4]{0} parameter(0)
  init = bf16[] constant(0)
  ROOT %reduce = bf16[] reduce(arg0, init), dimensions={0}, to_apply=add_bf16
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  Literal arg = LiteralUtil::CreateR1<bfloat16>(
      {bfloat16(1.0f), bfloat16(3.0f), bfloat16(-2.0f), bfloat16(42.0f)});
  Literal expected = LiteralUtil::CreateR0<bfloat16>(bfloat16(44.0f));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, Evaluate({&arg})));
}

TEST_P(HloEvaluatorBf16Test, SliceWithDifferentLayout) {
  // Regression test for b/114735354.
  const string hlo_text = R"(
HloModule SliceWithDifferentLayout

ENTRY main {
  arg = f32[2,2,2]{0,1,2} parameter(0)
  ROOT %slice = f32[2,2,2]{1,0,2} slice(f32[2,2,2]{0,1,2} %arg), slice={[0:2], [0:2], [0:2]}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  Literal arg = LiteralUtil::CreateR3WithLayout<float>(
      {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}},
      LayoutUtil::MakeLayout({0, 1, 2}));
  Literal actual = Evaluate({&arg});
  EXPECT_TRUE(LiteralTestUtil::Equal(arg, actual));
}

TEST_P(HloEvaluatorBf16Test, Bitcast) {
  // Regression test for b/114735354.
  constexpr absl::string_view hlo_text_base = R"(
HloModule Bitcast

ENTRY main {
  param = %s[32,121]{1,0} parameter(0)
  ROOT bitcast = %s[121,32,1]{0,1,2} bitcast(%s[32,121]{1,0} param)
}
)";
  string hlo_text;
  if (use_bfloat16_) {
    hlo_text = absl::StrFormat(hlo_text_base, "bf16", "bf16", "bf16");
  } else {
    hlo_text = absl::StrFormat(hlo_text_base, "f32", "f32", "f32");
  }
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto args = MakeFakeArguments(m_.get()).ConsumeValueOrDie();
  Literal actual = Evaluate({&args[0]});
  if (use_bfloat16_) {
    EXPECT_TRUE(
        absl::c_equal(args[0].data<bfloat16>(), actual.data<bfloat16>()));
  } else {
    EXPECT_TRUE(absl::c_equal(args[0].data<float>(), actual.data<float>()));
  }
}

// Check that s32 under/overflow doesn't trigger a ubsan failure.
TEST_F(HloEvaluatorTest, Int32Overflow) {
  constexpr absl::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  c1 = s32[] constant(1073741824)  // 2^30
  sum = s32[] add(c1, c1)  // 2^31, i.e. INT_MIN

  c2 = s32[] constant(-2147483648)  // -2^31
  sub = s32[] subtract(c2, c1)  // -2^31 - 2^30, underflows

  mul = s32[] multiply(c1, c1)
  ROOT tuple = (s32[], s32[], s32[]) tuple(sum, sub, mul)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  std::vector<Literal> actual = Evaluate({}).DecomposeTuple();
  ASSERT_EQ(actual.size(), 3);

  uint32 pow30 = uint32{1} << 30;
  uint32 pow31 = uint32{1} << 31;
  EXPECT_EQ(actual[0].GetFirstElement<int32>(), static_cast<int32>(pow31));
  EXPECT_EQ(actual[1].GetFirstElement<int32>(),
            static_cast<int32>(-(pow31 + pow30)));
  EXPECT_EQ(actual[2].GetFirstElement<int32>(),
            static_cast<int32>(pow31 * pow31));
}

TEST_F(HloEvaluatorTest, GetDimensionSize) {
  constexpr absl::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  size = u32[] parameter(0)

  data = s32[4] parameter(1)

  sum = s32[4] add(data, data)

  ROOT dynamic_size = u32[] get-dimension-size(sum), dimensions={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  // Set up dynamic parameter binding.
  TF_CHECK_OK(m_->dynamic_parameter_binding().Bind(
      DynamicParameterBinding::DynamicParameter{0, {}},
      DynamicParameterBinding::DynamicDimension{1, {}, 0}));

  TF_ASSERT_OK_AND_ASSIGN(DynamicDimensionInference dynamic_dimension_inference,
                          DynamicDimensionInference::Run(m_.get()));

  evaluator_.set_dynamic_dimension_inference(&dynamic_dimension_inference);
  Literal size_arg = LiteralUtil::CreateR0<uint32>(3);
  Literal data_arg = LiteralUtil::CreateR1<int32>({1, 2, 3, 4});

  Literal actual = Evaluate({&size_arg, &data_arg});

  EXPECT_EQ(actual.GetFirstElement<uint32>(), static_cast<uint32>(3));
}

// Check that we get a useful error if we pass inputs of the wrong shape.
TEST_F(HloEvaluatorTest, EvaluateWithWrongInputShapes) {
  constexpr absl::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  p0 = s32[1] parameter(0)
  ROOT sum = s32[1] add(p0, p0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal input_wrong_shape = LiteralUtil::CreateR1<int32>({0, 1});

  EXPECT_EQ(HloEvaluator()
                .Evaluate(*m_, {&input_wrong_shape})
                .status()
                .error_message(),
            "Shape mismatch at parameter 0. Computation expected s32[1]{0}, "
            "but arg was s32[2].");
  EXPECT_EQ(HloEvaluator()
                .Evaluate(*m_->entry_computation(), {&input_wrong_shape})
                .status()
                .error_message(),
            "Shape mismatch at parameter 0. Computation expected s32[1]{0}, "
            "but arg was s32[2].");
}

// Check that we get a useful error if we pass too many or too few inputs.
TEST_F(HloEvaluatorTest, EvaluateWithWrongNumberOfInputs) {
  constexpr absl::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  p0 = s32[1] parameter(0)
  ROOT sum = s32[1] add(p0, p0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal input = LiteralUtil::CreateR1<int32>({0});

  EXPECT_EQ(
      HloEvaluator().Evaluate(*m_, {&input, &input}).status().error_message(),
      "Expected 1 argument, but got 2.");
  EXPECT_EQ(HloEvaluator()
                .Evaluate(*m_->entry_computation(), {&input, &input})
                .status()
                .error_message(),
            "Expected 1 argument, but got 2.");
}

TEST_F(HloEvaluatorTest, PreserveFusionInputLayout) {
  constexpr absl::string_view hlo_text = R"(
    HloModule FusionInputLayout

    fused_computation {
      param_0 = f32[20,20]{0,1} parameter(0)
      ROOT bitcast = f32[20,20]{1,0} bitcast(param_0)
    }

    ENTRY kernel_entry {
      parameter.0 = f32[20,20]{0,1} parameter(0)
      ROOT fusion = f32[20,20]{1,0} fusion(parameter.0),
        kind=kLoop, calls=fused_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto args = MakeFakeArguments(m_.get()).ConsumeValueOrDie();
  Literal actual = Evaluate({&args[0]});
  EXPECT_TRUE(absl::c_equal(args[0].data<float>(), actual.data<float>()));
}

TEST_F(HloEvaluatorTest, PreserveFusionOutputLayout) {
  constexpr absl::string_view hlo_text = R"(
    HloModule FusionOutputLayout

    fused_computation {
      param_0 = f32[20,20]{1,0} parameter(0)
      ROOT bitcast = f32[20,20]{0,1} bitcast(param_0)
    }

    ENTRY kernel_entry {
      parameter.0 = f32[20,20]{1,0} parameter(0)
      ROOT fusion = f32[20,20]{0,1} fusion(parameter.0),
        kind=kLoop, calls=fused_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto args = MakeFakeArguments(m_.get()).ConsumeValueOrDie();
  Literal actual = Evaluate({&args[0]});
  EXPECT_TRUE(absl::c_equal(args[0].data<float>(), actual.data<float>()));
}

TEST_F(HloEvaluatorTest, PreserveMOFusionOutputLayout) {
  constexpr absl::string_view hlo_text = R"(
    HloModule MOFusionOutputLayout

    fused_computation {
      param_0 = f32[20,20]{1,0} parameter(0)
      bitcast = f32[20,20]{0,1} bitcast(param_0)
      ROOT tuple = (f32[20,20]{0,1}) tuple(bitcast)
    }

    ENTRY kernel_entry {
      parameter.0 = f32[20,20]{1,0} parameter(0)
      ROOT fusion = (f32[20,20]{0,1}) fusion(parameter.0),
        kind=kLoop, calls=fused_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto args = MakeFakeArguments(m_.get()).ConsumeValueOrDie();
  Literal actual_tuple = Evaluate({&args[0]});
  std::vector<Literal> actual_literals = actual_tuple.DecomposeTuple();
  EXPECT_TRUE(
      absl::c_equal(args[0].data<float>(), actual_literals[0].data<float>()));
}

}  // namespace
}  // namespace xla

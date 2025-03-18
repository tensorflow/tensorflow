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
#include "xla/hlo/evaluator/hlo_evaluator.h"

#include <array>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/internal/endian.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/array4d.h"
#include "xla/comparison_util.h"
#include "xla/debug_options_flags.h"
#include "xla/error_spec.h"
#include "xla/hlo/analysis/tuple_points_to_analysis.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/transforms/simplifiers/hlo_element_type_converter.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

static std::array<bool, 2> use_bf16_params{true, false};

// Test fixture for the HloEvaluator.
//
// In bf16 mode, all f32 shapes are converted to bf16 before running.
class HloEvaluatorTest : public HloHardwareIndependentTestBase {
 public:
  HloEvaluatorTest() : use_bfloat16_(false) { InitializeFftData(); }

  absl::StatusOr<Literal> Evaluate(
      absl::Span<const Literal* const> arg_literals = {}) {
    if (use_bfloat16_) {
      HloElementTypeConverter(F32, BF16).Run(m_.get()).value();
    }
    return evaluator_.Evaluate(*m_->entry_computation(), arg_literals);
  }

  // Evaluate function that takes in a local module instead of using m_
  // that is in HloHardwareIndependentTestBase. Once m_ in
  // HloHardwareIndependentTestBase is removed, this should be the default
  // Evaluate function.
  Literal EvaluateWithModule(
      HloModule* module, absl::Span<const Literal* const> arg_literals = {}) {
    if (use_bfloat16_) {
      HloElementTypeConverter(F32, BF16).Run(m_.get()).value();
    }
    return evaluator_.Evaluate(*module->entry_computation(), arg_literals)
        .value();
  }

  void TestUnaryOp(HloOpcode opcode, Literal expected, Literal input,
                   float aabs = 0) {
    HloComputation::Builder b(TestName());
    auto c1 =
        b.AddInstruction(HloInstruction::CreateConstant(std::move(input)));
    b.AddInstruction(HloInstruction::CreateUnary(expected.shape(), opcode, c1));
    m_->AddEntryComputation(b.Build());

    TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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

    TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

  void TestTernaryOp(HloOpcode opcode, Literal expected, Literal src0,
                     Literal src1, Literal src2) {
    HloComputation::Builder b(TestName());
    auto operand0 =
        b.AddInstruction(HloInstruction::CreateConstant(std::move(src0)));
    auto operand1 =
        b.AddInstruction(HloInstruction::CreateConstant(std::move(src1)));
    auto operand2 =
        b.AddInstruction(HloInstruction::CreateConstant(std::move(src2)));
    b.AddInstruction(HloInstruction::CreateTernary(
        expected.shape(), opcode, operand0, operand1, operand2));
    m_->AddEntryComputation(b.Build());

    TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

  void TestEvaluateInstruction(HloInstruction* instruction,
                               const Literal& expected) {
    TF_ASSERT_OK_AND_ASSIGN(Literal result, evaluator_.Evaluate(instruction));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

  void TestEvaluationFailure(HloInstruction* instruction) {
    absl::StatusOr<Literal> result = evaluator_.Evaluate(instruction);
    EXPECT_TRUE(!result.ok());
  }

  void TestRecursivelyEvaluateInstruction(HloInstruction* instruction,
                                          const Literal& expected) {
    TF_ASSERT_OK_AND_ASSIGN(
        Literal result,
        evaluator_.Evaluate(
            instruction, /*precomputed_analyses=*/{},
            /*recursively_evaluate_nonconstant_operands=*/true));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

  void TestRecursiveEvaluationFailure(HloInstruction* instruction) {
    absl::StatusOr<Literal> result =
        evaluator_.Evaluate(instruction, /*precomputed_analyses=*/{},
                            /*recursively_evaluate_nonconstant_operands=*/true);
    EXPECT_TRUE(!result.ok());
  }

  std::unique_ptr<HloComputation> MaxComputationScalarF32() {
    HloComputation::Builder max_computation("max");
    Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
    auto param_lhs = max_computation.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
    auto param_rhs = max_computation.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
    max_computation.AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kMaximum, param_lhs, param_rhs));
    return max_computation.Build();
  }

  void ReduceWindowMaxIotaTest(int window_size, int padding, int stride,
                               int window_dilation, int base_dilation,
                               const Literal& expected) {
    HloComputation::Builder b(TestName());

    // arg:
    // f32[4,4] {
    //  {  0,  1,  2,  3 },
    //  {  4,  5,  6,  7 },
    //  {  8,  9, 10, 11 },
    //  { 12, 13, 14, 15 }
    // }
    auto arg_array = std::make_unique<Array2D<float>>(4, 4);
    arg_array->FillIota(0);
    auto arg_literal = LiteralUtil::CreateR2FromArray2D<float>(*arg_array);

    HloInstruction* arg_instruction = b.AddInstruction(
        HloInstruction::CreateConstant(std::move(arg_literal)));
    auto init_value = b.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));
    auto max_func = m_->AddEmbeddedComputation(MaxComputationScalarF32());

    Window window;
    WindowDimension dim;
    dim.set_size(window_size);
    dim.set_stride(stride);
    dim.set_padding_low(padding);
    dim.set_padding_high(padding);
    dim.set_window_dilation(window_dilation);
    dim.set_base_dilation(base_dilation);
    *window.add_dimensions() = dim;
    *window.add_dimensions() = dim;

    int dim0 = expected.shape().dimensions(0);
    int dim1 = expected.shape().dimensions(1);
    Shape shape = ShapeUtil::MakeShape(F32, {dim0, dim1});
    b.AddInstruction(HloInstruction::CreateReduceWindow(
        shape, arg_instruction, init_value, window, max_func));

    m_->AddEntryComputation(b.Build());
    TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

 protected:
  explicit HloEvaluatorTest(bool use_bfloat16) : use_bfloat16_(use_bfloat16) {
    InitializeFftData();
  }

  // Initializes data sets used in FFT tests below.
  void InitializeFftData();

  HloEvaluator evaluator_;

  const bool use_bfloat16_;
  std::unique_ptr<HloModule> m_ = CreateNewVerifiedModule();

  // Data sets used in FFT tests below.
  ErrorSpec fft_error_ = ErrorSpec(1e-4, 1e-5);
  Literal fft_c64x2x4x8_;
  Literal fft_c64x2x4x8_1d_;
  Literal fft_c64x2x4x8_2d_;
  Literal fft_c64x2x4x8_3d_;
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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<float>({{0, 4}, {2, 4}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Verifies that clamping of int64_t does not cause loss of precision
TEST_P(HloEvaluatorBf16Test, DoesClampInt64) {
  auto ones = [](int bits) { return (int64_t{1} << bits) - 1; };

  auto low =
      LiteralUtil::CreateR2<int64_t>({{0, ones(54)}, {ones(54), ones(58)}});
  auto value = LiteralUtil::CreateR2<int64_t>({{0, ones(56)}, {0, ones(58)}});
  auto high = LiteralUtil::CreateR2<int64_t>(
      {{ones(54), ones(55)}, {ones(56), ones(58)}});

  Shape shape = low.shape();
  HloComputation::Builder b(TestName());
  auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(low)));
  auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(value)));
  auto c3 = b.AddInstruction(HloInstruction::CreateConstant(std::move(high)));
  b.AddInstruction(
      HloInstruction::CreateTernary(shape, HloOpcode::kClamp, c1, c2, c3));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected =
      LiteralUtil::CreateR2<int64_t>({{0, ones(55)}, {ones(54), ones(58)}});

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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({}));

  auto expected = LiteralUtil::CreateR2<float>({{2, 5}, {0, 4}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise addition with 2 operands.
TEST_F(HloEvaluatorTest, DoesAdd) {
  auto lhs = LiteralUtil::CreateR2<int64_t>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64_t>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64_t>({{3, 4}, {-96, 8}});
  TestBinaryOp(HloOpcode::kAdd, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise and with 2 operands.
TEST_P(HloEvaluatorBf16Test, DoesAnd) {
  auto lhs = LiteralUtil::CreateR2<int64_t>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64_t>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64_t>({{0, 0}, {4, 4}});
  TestBinaryOp(HloOpcode::kAnd, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise or with 2 operands.
TEST_F(HloEvaluatorTest, DoesOr) {
  auto lhs = LiteralUtil::CreateR2<int64_t>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64_t>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64_t>({{3, 4}, {-100, 4}});
  TestBinaryOp(HloOpcode::kOr, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise or with 2 operands.
TEST_F(HloEvaluatorTest, DoesXor) {
  auto lhs = LiteralUtil::CreateR2<int64_t>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64_t>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64_t>({{3, 4}, {-104, 0}});
  TestBinaryOp(HloOpcode::kXor, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise multiply with 2 operands.
TEST_F(HloEvaluatorTest, DoesMultiply) {
  auto lhs = LiteralUtil::CreateR2<int32_t>({{-1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int32_t>(
      {{std::numeric_limits<int32_t>::min(), 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int32_t>(
      {{std::numeric_limits<int32_t>::min(), 0}, {-400, 16}});
  TestBinaryOp(HloOpcode::kMultiply, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise divide with 2 operands.
TEST_F(HloEvaluatorTest, DoesDivideInt64) {
  auto lhs = LiteralUtil::CreateR2<int64_t>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64_t>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64_t>({{0, 0}, {-25, 1}});
  TestBinaryOp(HloOpcode::kDivide, std::move(expected), std::move(lhs),
               std::move(rhs));
}

TEST_F(HloEvaluatorTest, DoesClampS64) {
  auto low = LiteralUtil::CreateR1<int64_t>(
      {-8616761059752331528LL, 6780561065411491190LL, -8616761059752331528LL});
  auto value = LiteralUtil::CreateR1<int64_t>(
      {-6780561065411491190LL, 6780561065411491180LL, 4241131823772864090LL});
  auto high = LiteralUtil::CreateR1<int64_t>(
      {-6780561065411491180LL, 8616761059752331528LL, 3832151243857508051LL});
  auto expected = LiteralUtil::CreateR1<int64_t>(
      {-6780561065411491190LL, 6780561065411491190LL, 3832151243857508051LL});
  TestTernaryOp(HloOpcode::kClamp, std::move(expected), std::move(low),
                std::move(value), std::move(high));
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
  auto operand = LiteralUtil::CreateR2<int64_t>({{1, -20}, {-100, 4}});
  auto expected = LiteralUtil::CreateR2<int64_t>({{1, 20}, {100, 4}});
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

TEST_F(HloEvaluatorTest, DoesAbsC128) {
  auto x = LiteralUtil::CreateR0<complex128>({1, 2});
  auto expected_real = LiteralUtil::CreateR0<double>(2.23607);
  TestUnaryOp(HloOpcode::kAbs, std::move(expected_real), std::move(x), 3e-06);
}

TEST_F(HloEvaluatorTest, DoesNegateR2) {
  auto operand = LiteralUtil::CreateR2<int32_t>(
      {{0, std::numeric_limits<int32_t>::min()}, {-1, 4}});
  auto expected = LiteralUtil::CreateR2<int32_t>(
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
TEST_P(HloEvaluatorBf16Test, DoesTanR2) {
  auto operand = LiteralUtil::CreateR2<float>({{0, M_PI}, {-M_PI, 2 * M_PI}});
  auto expected = LiteralUtil::CreateR2<float>({{0, 0}, {0, 0}});
  TestUnaryOp(HloOpcode::kTan, std::move(expected), std::move(operand),
              use_bfloat16_ ? 0.031250 : 9.5367431640625E-7);
}
TEST_F(HloEvaluatorTest, DoesNotR2) {
  auto operand =
      LiteralUtil::CreateR2<int32_t>({{0, std::numeric_limits<int>::min()},
                                      {-1, std::numeric_limits<int>::max()}});
  auto expected =
      LiteralUtil::CreateR2<int32_t>({{-1, std::numeric_limits<int>::max()},
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

TEST_P(HloEvaluatorBf16Test, DoesImagF32AndBf16) {
  auto x = LiteralUtil::CreateR1<float>({1, -100});
  auto expected_imag = LiteralUtil::CreateR1<float>({0, 0});
  TestUnaryOp(HloOpcode::kImag, std::move(expected_imag), std::move(x));
}

TEST_F(HloEvaluatorTest, DoesImagF64) {
  auto x = LiteralUtil::CreateR1<double>({1, -100});
  auto expected_imag = LiteralUtil::CreateR1<double>({0, 0});
  TestUnaryOp(HloOpcode::kImag, std::move(expected_imag), std::move(x));
}

// Verifies that HloEvaluator evaluates a HLO Computation with non-parameter nor
// constant operands.
TEST_F(HloEvaluatorTest, DoesTraverseInstructions) {
  auto lhs = LiteralUtil::CreateR2<int64_t>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64_t>({{2, 4}, {4, 4}});
  auto rhs2 = LiteralUtil::CreateR2<int64_t>({{1, -20}, {-100, 4}});
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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate(args));

  auto expected = LiteralUtil::CreateR2<int64_t>({{4, -16}, {-196, 12}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Verifies Reshape operation is correctly evaluated.
TEST_F(HloEvaluatorTest, DoesReshape) {
  HloComputation::Builder b(TestName());
  const int64_t dimensions[] = {11, 8, 7, 5, 9};
  TF_ASSERT_OK_AND_ASSIGN(auto literal,
                          LiteralUtil::CreateRandomLiteral<F32>(
                              ShapeUtil::MakeShape(F32, dimensions), 0.0, 1.0));
  auto literal_clone = literal.Clone();
  HloInstruction* literal_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {8, 7, 11, 9, 5});
  const int64_t permutation[] = {1, 2, 0, 4, 3};
  b.AddInstruction(
      HloInstruction::CreateTranspose(shape, literal_instruction, permutation));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({}));

  using NativeT = typename primitive_util::PrimitiveTypeToNative<F32>::type;
  result.EachCell<NativeT>(
      [&](absl::Span<const int64_t> indices, NativeT value) {
        std::vector<int64_t> rindexes = PermuteInverse(indices, permutation);
        EXPECT_NEAR(value, literal_clone.Get<NativeT>(rindexes), 0.031250);
      });
}

// Verifies Broadcast operation is correctly evaluated.
TEST_F(HloEvaluatorTest, DoesBroadcast) {
  HloComputation::Builder b(TestName());
  auto input_literal = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}, {5, 6}});
  auto output_literal = LiteralUtil::CreateR3<int32_t>(
      {{{1, 2}, {3, 4}, {5, 6}}, {{1, 2}, {3, 4}, {5, 6}}});
  HloInstruction* literal_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  b.AddInstruction(HloInstruction::CreateBroadcast(
      output_literal.shape(), literal_instruction, {1, 2}));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({}));

  EXPECT_TRUE(LiteralTestUtil::Equal(result, output_literal));
}

TEST_F(HloEvaluatorTest, DoesBroadcastScalar) {
  HloComputation::Builder b(TestName());
  auto input_literal = LiteralUtil::CreateR0<int32_t>(111);
  auto output_literal = LiteralUtil::CreateR2<int32_t>(
      {{111, 111}, {111, 111}, {111, 111}, {111, 111}, {111, 111}, {111, 111}});

  HloInstruction* literal_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  // Broadcast dimension should be empty in the case of scalars.
  b.AddInstruction(HloInstruction::CreateBroadcast(
      output_literal.shape(), literal_instruction,
      /*broadcast_dimensions=*/{}));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({}));

  EXPECT_TRUE(LiteralTestUtil::Equal(result, output_literal));
}

TEST_F(HloEvaluatorTest, DoesConcatenateSimple) {
  HloComputation::Builder b(TestName());

  HloInstruction* operand1 = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int64_t>({{-1, -2}, {100, 200}})));
  HloInstruction* operand2 = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int64_t>({{-2, -3}, {-100, -200}})));

  std::vector<HloInstruction*> operands = {operand1, operand2};

  Shape shape = ShapeUtil::MakeShape(S64, {4, 2});
  b.AddInstruction(HloInstruction::CreateConcatenate(shape, operands, 0));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<int64_t>(
      {{-1, -2}, {100, 200}, {-2, -3}, {-100, -200}});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, ConcatenateHandlesShapeWithZeroElement) {
  HloComputation::Builder b(TestName());

  HloInstruction* operand1 = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int64_t>({100, 200})));
  HloInstruction* operand2 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int64_t>({})));

  std::vector<HloInstruction*> operands = {operand1, operand2};

  Shape shape = ShapeUtil::MakeShape(S64, {2});
  b.AddInstruction(HloInstruction::CreateConcatenate(shape, operands, 0));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR1<int64_t>({100, 200});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, ConvertWithSameLayout) {
  HloComputation::Builder b(TestName());

  auto input_literal = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}, {5, 6}});
  auto expected =
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  ASSERT_TRUE(LayoutUtil::LayoutsInShapesEqual(input_literal.shape(),
                                               expected.shape()));

  HloInstruction* constant = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  b.AddInstruction(HloInstruction::CreateConvert(expected.shape(), constant));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  EXPECT_TRUE(LiteralTestUtil::Equal(result, expected));
}

TEST_P(HloEvaluatorBf16Test, ConvertWithDifferentLayout) {
  HloComputation::Builder b(TestName());

  auto input_literal = LiteralUtil::CreateR2WithLayout<int32_t>(
      {{1, 2}, {3, 4}, {5, 6}}, LayoutUtil::MakeLayout({0, 1}));
  auto expected = LiteralUtil::CreateR2WithLayout<float>(
      {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}, LayoutUtil::MakeLayout({1, 0}));
  ASSERT_FALSE(LayoutUtil::LayoutsInShapesEqual(input_literal.shape(),
                                                expected.shape()));

  HloInstruction* constant = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  b.AddInstruction(HloInstruction::CreateConvert(expected.shape(), constant));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  EXPECT_TRUE(LiteralTestUtil::Equal(result, expected));
}

PaddingConfig CreatePaddingConfig(
    std::initializer_list<std::array<int64_t, 3>> padding_dimensions) {
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
  auto operand = LiteralUtil::CreateR2<int32_t>({{}, {}});
  HloComputation::Builder b(TestName());
  auto operand_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(operand)));

  constexpr int32_t kPadValue = 10;
  auto pad_value = LiteralUtil::CreateR0<int32_t>(kPadValue);
  auto padding_value_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(pad_value)));

  auto padding_config = CreatePaddingConfig({{{1, 0, 2}}, {{0, 2, 1}}});
  Shape shape = ShapeUtil::MakeShape(S32, {5, 2});
  b.AddInstruction(HloInstruction::CreatePad(
      shape, operand_instruction, padding_value_instruction, padding_config));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<int32_t>(
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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected_array = std::make_unique<Array4D<float>>(8, 5, 1, 1);
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
  auto input_array = std::make_unique<Array2D<float>>(4, 3);
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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  // f32[1,5] { 7.0, 2.718, 2.718, 2.718, 2.718 }
  auto expected_array = std::make_unique<Array2D<float>>(1, 5);
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
  auto input_array = std::make_unique<Array2D<float>>(4, 3);
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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected_array = std::make_unique<Array2D<float>>(0, 9);
  auto expected = LiteralUtil::CreateR2FromArray2D<float>(*expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, Pad2DFloatArrayDifferentTypes) {
  HloComputation::Builder b(TestName());
  b.AddInstruction(HloInstruction::CreatePad(
      ShapeUtil::MakeShape(BF16, {5, 2}),
      /*operand=*/
      b.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<bfloat16>({{}, {}}))),
      /*padding_value=*/
      b.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(10.0f))),
      CreatePaddingConfig({{{1, 0, 2}}, {{0, 2, 1}}})));
  m_->AddEntryComputation(b.Build());
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  bfloat16 bf16_c(10.0f);
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<bfloat16>({{bf16_c, bf16_c},
                                       {bf16_c, bf16_c},
                                       {bf16_c, bf16_c},
                                       {bf16_c, bf16_c},
                                       {bf16_c, bf16_c}}),
      result));
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
  auto lhs_array = std::make_unique<Array2D<float>>(4, 1);
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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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
  auto rhs_array = std::make_unique<Array2D<float>>(3, 2);
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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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
  auto lhs_array = std::make_unique<Array2D<float>>(4, 3);
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
  auto rhs_array = std::make_unique<Array2D<float>>(3, 2);
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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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

  auto lhs_array = std::make_unique<Array4D<float>>(2, 2, 3, 1);
  lhs_array->FillIota(1.0f);
  auto lhs_literal = LiteralUtil::CreateR4FromArray4D<float>(*lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  auto rhs_array = std::make_unique<Array4D<float>>(2, 2, 3, 1);
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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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
  std::vector<int64_t> input_dims = {1, 2, 2, 4};
  std::vector<int64_t> filter_dims = {2, 2, 2, 8};
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
  auto input_r4 = input_r1.Reshape(input_dims).value();
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(input_r4)));

  std::vector<float> filter_elems(ShapeUtil::ElementsIn(filter_shape));
  std::iota(filter_elems.begin(), filter_elems.end(), -31);
  auto filter_r1 = LiteralUtil::CreateR1<float>(filter_elems);
  auto filter_r4 = filter_r1.Reshape(filter_dims).value();
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(filter_r4)));

  Shape shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 8});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction,
      /*feature_group_count=*/2, /*batch_group_count=*/1, window, dnums,
      DefaultPrecisionConfig(2)));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  Array4D<float> expected_array(1, 1, 1, 8);
  expected_array.FillWithYX(
      Array2D<float>({{668, 664, 660, 656, 668, 680, 692, 704}}));
  auto expected = LiteralUtil::CreateR4FromArray4D<float>(expected_array);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Initialization of data sets for FFT tests:

void HloEvaluatorTest::InitializeFftData() {
  // clang-format off
  fft_c64x2x4x8_ = LiteralUtil::CreateR3<complex64>({
    {{{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0},
      {4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0}, {7.0, 0.0}},
     {{0.0, 0.0}, {0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0},
      {0.0, 4.0}, {0.0, 5.0}, {0.0, 6.0}, {0.0, 7.0}},
     {{0.0, 7.0}, {1.0, 6.0}, {2.0, 5.0}, {3.0, 4.0},
      {4.0, 3.0}, {5.0, 2.0}, {6.0, 1.0}, {7.0, 0.0}},
     {{7.0, 0.0}, {6.0, 1.0}, {5.0, 2.0}, {4.0, 3.0},
      {3.0, 4.0}, {2.0, 5.0}, {1.0, 6.0}, {0.0, 7.0}}},
    {{{-4.0, 0.0}, {-3.0, 0.0}, {-2.0, 0.0}, {-1.0, 0.0},
      {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}},
     {{0.0, -4.0}, {0.0, -3.0}, {0.0, -2.0}, {0.0, -1.0},
      {0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}, {0.0, 4.0}},
     {{3.5, 3.5}, {-1.707107, -0.707107}, {-1.0, -0.0}, {-0.707107, 0.292893},
      {-0.5, 0.5}, {-0.292893, 0.707107}, {0.0, 1.0}, {0.707107, 1.707107}},
     {{3.5, 3.5}, {1.707107, 0.707107}, {1.0, 0.0}, {0.707107, -0.292893},
      {0.5, -0.5}, {0.292893, -0.707107}, {-0.0, -1.0}, {-0.707107, -1.707107}}}
  });
  fft_c64x2x4x8_1d_ = LiteralUtil::CreateR3<complex64>({
    {{{28.0, 0.0}, {-4.0, 9.656854}, {-4.0, 4.0}, {-4.0, 1.656854},
      {-4.0, 0.0}, {-4.0, -1.656854}, {-4.0, -4.0}, {-4.0, -9.656854}},
     {{0.0, 28.0}, {-9.656854, -4.0}, {-4.0, -4.0}, {-1.656854, -4.0},
      {0.0, -4.0}, {1.656854, -4.0}, {4.0, -4.0}, {9.656854, -4.0}},
     {{28.0, 28.0}, {5.656854, 13.656854}, {0.0, 8.0}, {-2.343146, 5.656854},
      {-4.0, 4.0}, {-5.656854, 2.343146}, {-8.0, -0.0}, {-13.656854, -5.656854}},  // NOLINT
     {{28.0, 28.0}, {-5.656854, -13.656854}, {-0.0, -8.0}, {2.343146, -5.656854},  // NOLINT
      {4.0, -4.0}, {5.656854, -2.343146}, {8.0, 0.0}, {13.656854, 5.656854}}},
    {{{0.0, 0.0}, {-5.0, 12.071068}, {-4.0, 4.0}, {-5.0, 2.071068},
      {-4.0, 0.0}, {-5.0, -2.071068}, {-4.0, -4.0}, {-5.0, -12.071068}},
     {{0.0, 0.0}, {-12.071068, -5.0}, {-4.0, -4.0}, {-2.071068, -5.0},
      {0.0, -4.0}, {2.071068, -5.0}, {4.0, -4.0}, {12.071068, -5.0}},
     {{0.0, 7.0}, {1.0, 6.0}, {2.0, 5.0}, {3.0, 4.0},
      {4.0, 3.0}, {5.0, 2.0}, {6.0, 1.0}, {7.0, 0.0}},
     {{7.0, 0.0}, {6.0, 1.0}, {5.0, 2.0}, {4.0, 3.0},
      {3.0, 4.0}, {2.0, 5.0}, {1.0, 6.0}, {0.0, 7.0}}}
  });
  fft_c64x2x4x8_2d_ = LiteralUtil::CreateR3<complex64>({
    {{{84.0, 84.0}, {-13.656854, 5.656854}, {-8.0, 0.0}, {-5.656854, -2.343146},
      {-4.0, -4.0}, {-2.343146, -5.656854}, {0.0, -8.0}, {5.656854, -13.656854}},  // NOLINT
     {{0.0, 0.0}, {0.0, -0.0}, {0.0, 0.0}, {0.0, 0.0},
      {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
     {{28.0, -28.0}, {16.970562, 40.970562}, {0.0, 24.0}, {-7.029438, 16.970562},      // NOLINT
      {-12.0, 12.0}, {-16.970562, 7.029438}, {-24.0, 0.0}, {-40.970562, -16.970562}},  // NOLINT
     {{0.0, -56.0}, {-19.313708, -8.0}, {-8.0, -8.0}, {-3.313708, -8.0},
      {0.0, -8.0}, {3.313708, -8.0}, {8.0, -8.0}, {19.313708, -8.0}}},
    {{{7.0, 7.0}, {-10.071068, 14.071068}, {-1.0, 7.0}, {-0.071068, 4.071068},
      {3.0, 3.0}, {4.071068, -0.071068}, {7.0, -1.0}, {14.071068, -10.071068}},
     {{0.0, 0.0}, {-12.0, 24.142136}, {-12.0, 8.0}, {-16.0, 4.142136},
      {-16.0, 0.0}, {-20.0, -4.142136}, {-20.0, -8.0}, {-24.0, -24.142136}},
     {{-7.0, 7.0}, {2.071068, 22.071068}, {-3.0, 11.0}, {-3.928932, 8.071068},
      {-3.0, 3.0}, {-4.071068, -0.071068}, {-3.0, -5.0}, {-10.071068, -14.071068}},  // NOLINT
     {{0.0, -14.0}, {0.0, -12.0}, {0.0, -10.0}, {0.0, -8.0},
      {0.0, -6.0}, {0.0, -4.0}, {0.0, -2.0}, {0.0, 0.0}}}
  });
  fft_c64x2x4x8_3d_ = LiteralUtil::CreateR3<complex64>({
    {{{91.0, 91.0}, {-23.727922, 19.727922}, {-9.0, 7.0}, {-5.727922, 1.727922},
      {-1.0, -1.0}, {1.727922, -5.727922}, {7.0, -9}, {19.727922, -23.727922}},
     {{0.0, 0.0}, {-12.0, 24.142136}, {-12.0, 8.0}, {-16.0, 4.142136},
      {-16.0, 0.0}, {-20.0, -4.142136}, {-20.0, -8.0}, {-24.0, -24.142136}},
     {{21.0, -21.0}, {19.041630, 63.041630}, {-3.0, 35.0}, {-10.958370, 25.041630},     // NOLINT
      {-15.0, 15.0}, {-21.041630, 6.958370}, {-27.0, -5.0}, {-51.041630, -31.041630}},  // NOLINT
     {{0.0, -70.0}, {-19.313708, -20.0}, {-8.0, -18.0}, {-3.313708, -16.0},
      {0.0, -14.0}, {3.313708, -12.0}, {8.0, -10.0}, {19.313708, -8.0}}},
    {{{77.0, 77.0}, {-3.585786, -8.414214}, {-7.0, -7.0}, {-5.585786, -6.414214},   // NOLINT
      {-7.0, -7.0}, {-6.414214, -5.585786}, {-7.0, -7.0}, {-8.414214, -3.585786}},  // NOLINT
     {{0.0, 0.0}, {12.0, -24.142136}, {12.0, -8.0}, {16.0, -4.142136},
      {16.0, 0.0}, {20.0, 4.142136}, {20.0, 8.0}, {24.0, 24.142136}},
     {{35.0, -35.0}, {14.899494, 18.899494}, {3.0, 13.0}, {-3.100506, 8.899494},
      {-9.0, 9.0}, {-12.899494, 7.100506}, {-21.0, 5.0}, {-30.899494, -2.899494}},  // NOLINT
     {{0.0, -42.0}, {-19.313708, 4.0}, {-8.0, 2.0}, {-3.313708, 0.0},
      {0.0, -2.0}, {3.313708, -4.0}, {8.0, -6.0}, {19.313708, -8.0}}}
  });
  // clang-format on
}

// Simple FFT tests:

TEST_F(HloEvaluatorTest, 1D_FFT_4_on_c64x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[4] parameter(0)
  ROOT fft = c64[4] fft(operand), fft_type=FFT, fft_length={4}
}
)";
  auto input = LiteralUtil::CreateR1<complex64>(
      {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}});
  auto expected = LiteralUtil::CreateR1<complex64>(
      {{10.0, 0.0}, {-2.0, 2.0}, {-2.0, 0.0}, {-2.0, -2.0}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_IFFT_4_on_c64x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[4] parameter(0)
  ROOT ifft = c64[4] fft(operand), fft_type=IFFT, fft_length={4}
}
)";
  auto input = LiteralUtil::CreateR1<complex64>(
      {{10.0, 0.0}, {-2.0, 2.0}, {-2.0, 0.0}, {-2.0, -2.0}});
  auto expected = LiteralUtil::CreateR1<complex64>(
      {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_RFFT_4_on_f32x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[4] parameter(0)
  ROOT rfft = c64[3] fft(operand), fft_type=RFFT, fft_length={4}
}
)";
  auto input = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, 4.0});
  auto expected =
      LiteralUtil::CreateR1<complex64>({{10.0, 0.0}, {-2.0, 2.0}, {-2.0, 0.0}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_IRFFT_4_on_c64x3) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3] parameter(0)
  ROOT irfft = f32[4] fft(operand), fft_type=IRFFT, fft_length={4}
}
)";
  auto input =
      LiteralUtil::CreateR1<complex64>({{10.0, 0.0}, {-2.0, 2.0}, {-2.0, 0.0}});
  auto expected = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, 4.0});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// 1D FFT tests:

TEST_F(HloEvaluatorTest, 1D_FFT_8_on_c64x2x4x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8] parameter(0)
  ROOT fft = c64[2, 4, 8] fft(operand), fft_type=FFT, fft_length={8}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&fft_c64x2x4x8_}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_1d_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_1d_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_IFFT_8_on_c64x2x4x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8] parameter(0)
  ROOT ifft = c64[2, 4, 8] fft(operand), fft_type=IFFT, fft_length={8}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&fft_c64x2x4x8_1d_}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_RFFT_8_on_f32x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[8] parameter(0)
  ROOT rfft = c64[5] fft(operand), fft_type=RFFT, fft_length={8}
}
)";
  auto input =
      LiteralUtil::CreateR1<float>({1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1});
  auto expected = LiteralUtil::CreateR1<complex64>({{39.6, 0.0},
                                                    {-3.6, 8.691169},
                                                    {-3.6, 3.6},
                                                    {-3.6, 1.491169},
                                                    {-3.6, 0.0}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_IRFFT_8_on_c64x5) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[5] parameter(0)
  ROOT irfft = f32[8] fft(operand), fft_type=IRFFT, fft_length={8}
}
)";
  auto input = LiteralUtil::CreateR1<complex64>({{39.6, 0.0},
                                                 {-3.6, 8.691169},
                                                 {-3.6, 3.6},
                                                 {-3.6, 1.491169},
                                                 {-3.6, 0.0}});
  auto expected =
      LiteralUtil::CreateR1<float>({1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_RFFT_9_on_f32x9) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[9] parameter(0)
  ROOT rfft = c64[5] fft(operand), fft_type=RFFT, fft_length={9}
}
)";
  auto input = LiteralUtil::CreateR1<float>(
      {1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1, 9.9});
  auto expected = LiteralUtil::CreateR1<complex64>({{49.5, 0.0},
                                                    {-3.360560, 11.705792},
                                                    {-3.893717, 5.712929},
                                                    {-4.5, 3.117691},
                                                    {-4.895723, 1.021942}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 1D_IRFFT_9_on_c64x5) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[5] parameter(0)
  ROOT irfft = f32[9] fft(operand), fft_type=IRFFT, fft_length={9}
}
)";
  auto input = LiteralUtil::CreateR1<complex64>({{49.5, 0.0},
                                                 {-3.360560, 11.705792},
                                                 {-3.893717, 5.712929},
                                                 {-4.5, 3.117691},
                                                 {-4.895723, 1.021942}});
  auto expected = LiteralUtil::CreateR1<float>(
      {1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1, 9.9});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// 2D FFT tests:

TEST_F(HloEvaluatorTest, 2D_FFT_4x8_on_c64x2x4x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8] parameter(0)
  ROOT fft = c64[2, 4, 8] fft(operand), fft_type=FFT, fft_length={4, 8}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&fft_c64x2x4x8_}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_2d_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_2d_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 2D_IFFT_4x8_on_c64x2x4x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8] parameter(0)
  ROOT ifft = c64[2, 4, 8] fft(operand), fft_type=IFFT, fft_length={4, 8}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&fft_c64x2x4x8_2d_}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 2D_RFFT_3x8_on_f32x3x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[3, 8] parameter(0)
  ROOT rfft = c64[3, 5] fft(operand), fft_type=RFFT, fft_length={3, 8}
}
)";
  auto input =
      LiteralUtil::CreateR2<float>({{1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1},
                                    {8.1, 7.2, 6.3, 5.4, 4.5, 3.6, 2.7, 1.8},
                                    {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8}});
  auto expected = LiteralUtil::CreateR2<complex64>({{{118.8, 0.0},
                                                     {-4.4, 10.622540},
                                                     {-4.4, 4.4},
                                                     {-4.4, 1.822540},
                                                     {-4.4, 0.0}},
                                                    {{0.0, 0.0},
                                                     {-19.926162, 0.797280},
                                                     {-10.128203, -3.728203},
                                                     {-6.069756, -5.602720},
                                                     {-3.2, -6.928203}},
                                                    {{0.0, 0.0},
                                                     {13.526162, 14.653687},
                                                     {3.728203, 10.128203},
                                                     {-0.330244, 8.253687},
                                                     {-3.2, 6.928203}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 2D_IRFFT_3x8_on_c64x3x5) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 5] parameter(0)
  ROOT irfft = f32[3, 8] fft(operand), fft_type=IRFFT, fft_length={3, 8}
}
)";
  auto input = LiteralUtil::CreateR2<complex64>({{{118.8, 0.0},
                                                  {-4.4, 10.622540},
                                                  {-4.4, 4.4},
                                                  {-4.4, 1.822540},
                                                  {-4.4, 0.0}},
                                                 {{0.0, 0.0},
                                                  {-19.926162, 0.797280},
                                                  {-10.128203, -3.728203},
                                                  {-6.069756, -5.602720},
                                                  {-3.2, -6.928203}},
                                                 {{0.0, 0.0},
                                                  {13.526162, 14.653687},
                                                  {3.728203, 10.128203},
                                                  {-0.330244, 8.253687},
                                                  {-3.2, 6.928203}}});
  auto expected =
      LiteralUtil::CreateR2<float>({{1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1},
                                    {8.1, 7.2, 6.3, 5.4, 4.5, 3.6, 2.7, 1.8},
                                    {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 2D_RFFT_3x9_on_f32x3x9) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[3, 9] parameter(0)
  ROOT rfft = c64[3, 5] fft(operand), fft_type=RFFT, fft_length={3, 9}
}
)";
  auto input = LiteralUtil::CreateR2<float>(
      {{1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3, 8.2, 9.1},
       {9.1, 8.2, 7.3, 6.4, 5.5, 4.6, 3.7, 2.8, 1.9},
       {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9}});
  auto expected = LiteralUtil::CreateR2<complex64>({{{148.5, 0.0},
                                                     {-4.95, 13.600013},
                                                     {-4.95, 5.899180},
                                                     {-4.95, 2.857884},
                                                     {-4.95, 0.872819}},
                                                    {{0.0, 0.0},
                                                     {-25.014467, 2.096690},
                                                     {-12.888800, -3.503916},
                                                     {-8.1, -5.715768},
                                                     {-4.974333, -7.159452}},
                                                    {{0.0, 0.0},
                                                     {17.814467, 17.685147},
                                                     {5.688800, 12.084542},
                                                     {0.9, 9.872690},
                                                     {-2.225667, 8.429006}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 2D_IRFFT_3x9_on_c64x3x5) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 5] parameter(0)
  ROOT irfft = f32[3, 9] fft(operand), fft_type=IRFFT, fft_length={3, 9}
}
)";
  auto input = LiteralUtil::CreateR2<complex64>({{{148.5, 0.0},
                                                  {-4.95, 13.600013},
                                                  {-4.95, 5.899180},
                                                  {-4.95, 2.857884},
                                                  {-4.95, 0.872819}},
                                                 {{0.0, 0.0},
                                                  {-25.014467, 2.096690},
                                                  {-12.888800, -3.503916},
                                                  {-8.1, -5.715768},
                                                  {-4.974333, -7.159452}},
                                                 {{0.0, 0.0},
                                                  {17.814467, 17.685147},
                                                  {5.688800, 12.084542},
                                                  {0.9, 9.872690},
                                                  {-2.225667, 8.429006}}});
  auto expected = LiteralUtil::CreateR2<float>(
      {{1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3, 8.2, 9.1},
       {9.1, 8.2, 7.3, 6.4, 5.5, 4.6, 3.7, 2.8, 1.9},
       {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// 3D FFT tests:

TEST_F(HloEvaluatorTest, 3D_FFT_2x4x8_on_c64x2x4x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8] parameter(0)
  ROOT fft = c64[2, 4, 8] fft(operand), fft_type=FFT, fft_length={2, 4, 8}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&fft_c64x2x4x8_}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_3d_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_3d_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 3D_IFFT_2x4x8_on_c64x2x4x8) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8] parameter(0)
  ROOT ifft = c64[2, 4, 8] fft(operand), fft_type=IFFT, fft_length={2, 4, 8}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&fft_c64x2x4x8_3d_}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 3D_RFFT_3x3x4_on_f32x3x3x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[3, 3, 4] parameter(0)
  ROOT rfft = c64[3, 3, 3] fft(operand), fft_type=RFFT, fft_length={3, 3, 4}
}
)";
  auto input = LiteralUtil::CreateR3<float>(
      {{{1.8, 2.7, 3.6, 4.5}, {8.1, 7.2, 6.3, 5.4}, {1.1, 2.2, 3.3, 4.4}},
       {{5.4, 6.3, 7.2, 8.1}, {4.5, 3.6, 2.7, 1.8}, {5.5, 6.6, 7.7, 8.8}},
       {{-1.8, -2.7, -3.6, -4.5},
        {-5.4, -6.3, -7.2, -8.1},
        {1.9, 2.9, 3.9, 4.9}}});
  auto expected = LiteralUtil::CreateR3<complex64>(
      {{{{92.8, 0.0}, {-2.8, 2.8}, {-2.8, 0.0}},
        {{-5.9, 35.160631}, {-11.519100, -8.919100}, {-1.3, -10.219100}},
        {{-5.9, -35.160631}, {8.919100, 11.519100}, {-1.3, 10.219100}}},
       {{{29.5, -81.579593}, {1.390897, 5.190897}, {-1.9, 3.290897}},
        {{-25.1, -49.017038}, {1.044486, 4.844486}, {-1.9, 2.944486}},
        {{11.8, 27.712813}, {1.517691, 4.717691}, {-1.6, 3.117691}}},
       {{{29.5, 81.579593}, {-5.190897, -1.390897}, {-1.9, -3.290897}},
        {{11.8, -27.712813}, {-4.717691, -1.517691}, {-1.6, -3.117691}},
        {{-25.1, 49.017038}, {-4.844486, -1.044486}, {-1.9, -2.944486}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 3D_IRFFT_3x3x4_on_c64x3x3x3) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 3, 3] parameter(0)
  ROOT irfft = f32[3, 3, 4] fft(operand), fft_type=IRFFT, fft_length={3, 3, 4}
}
)";
  auto input = LiteralUtil::CreateR3<complex64>(
      {{{{92.8, 0.0}, {-2.8, 2.8}, {-2.8, 0.0}},
        {{-5.9, 35.160631}, {-11.519100, -8.919100}, {-1.3, -10.219100}},
        {{-5.9, -35.160631}, {8.919100, 11.519100}, {-1.3, 10.219100}}},
       {{{29.5, -81.579593}, {1.390897, 5.190897}, {-1.9, 3.290897}},
        {{-25.1, -49.017038}, {1.044486, 4.844486}, {-1.9, 2.944486}},
        {{11.8, 27.712813}, {1.517691, 4.717691}, {-1.6, 3.117691}}},
       {{{29.5, 81.579593}, {-5.190897, -1.390897}, {-1.9, -3.290897}},
        {{11.8, -27.712813}, {-4.717691, -1.517691}, {-1.6, -3.117691}},
        {{-25.1, 49.017038}, {-4.844486, -1.044486}, {-1.9, -2.944486}}}});
  auto expected = LiteralUtil::CreateR3<float>(
      {{{1.8, 2.7, 3.6, 4.5}, {8.1, 7.2, 6.3, 5.4}, {1.1, 2.2, 3.3, 4.4}},
       {{5.4, 6.3, 7.2, 8.1}, {4.5, 3.6, 2.7, 1.8}, {5.5, 6.6, 7.7, 8.8}},
       {{-1.8, -2.7, -3.6, -4.5},
        {-5.4, -6.3, -7.2, -8.1},
        {1.9, 2.9, 3.9, 4.9}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 3D_RFFT_3x3x5_on_f32x3x3x5) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[3, 3, 5] parameter(0)
  ROOT rfft = c64[3, 3, 3] fft(operand), fft_type=RFFT, fft_length={3, 3, 5}
}
)";
  auto input = LiteralUtil::CreateR3<float>({{{1.8, 2.7, 3.6, 4.5, 5.4},
                                              {8.1, 7.2, 6.3, 5.4, 4.5},
                                              {1.1, 2.2, 3.3, 4.4, 5.5}},
                                             {{5.4, 6.3, 7.2, 8.1, 9.0},
                                              {4.5, 3.6, 2.7, 1.8, 0.9},
                                              {5.5, 6.6, 7.7, 8.8, 9.9}},
                                             {{-1.8, -2.7, -3.6, -4.5, -5.4},
                                              {-5.4, -6.3, -7.2, -8.1, -9.0},
                                              {1.9, 2.9, 3.9, 4.9, 5.9}}});
  auto expected = LiteralUtil::CreateR3<complex64>(
      {{{{119.5, 0.0}, {-3.5, 4.817337}, {-3.5, 1.137219}},
        {{-5.75, 56.724664}, {-19.206730, -10.537254}, {-5.775483, -12.245880}},
        {{-5.75, -56.724664}, {15.956730, 15.010495}, {2.525483, 13.301869}}},
       {{{39.25, -106.088112}, {3.286913, 7.382528}, {-1.038404, 4.885305}},
        {{-29.0, -64.951905}, {2.690922, 6.949515}, {-1.179098, 4.452292}},
        {{16.75, 30.743902}, {3.363918, 6.649878}, {-0.733751, 4.546954}}},
       {{{39.25, 106.088112}, {-8.036913, -0.844714}, {-3.711596, -3.341936}},
        {{16.75, -30.743902}, {-7.363918, -1.144350}, {-3.266249, -3.247275}},
        {{-29.0, 64.951905}, {-7.440922, -0.411701}, {-3.570902, -2.908924}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 3D_IRFFT_3x3x5_on_c64x3x3x3) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 3, 3] parameter(0)
  ROOT irfft = f32[3, 3, 5] fft(operand), fft_type=IRFFT, fft_length={3, 3, 5}
}
)";
  auto input = LiteralUtil::CreateR3<complex64>(
      {{{{119.5, 0.0}, {-3.5, 4.817337}, {-3.5, 1.137219}},
        {{-5.75, 56.724664}, {-19.206730, -10.537254}, {-5.775483, -12.245880}},
        {{-5.75, -56.724664}, {15.956730, 15.010495}, {2.525483, 13.301869}}},
       {{{39.25, -106.088112}, {3.286913, 7.382528}, {-1.038404, 4.885305}},
        {{-29.0, -64.951905}, {2.690922, 6.949515}, {-1.179098, 4.452292}},
        {{16.75, 30.743902}, {3.363918, 6.649878}, {-0.733751, 4.546954}}},
       {{{39.25, 106.088112}, {-8.036913, -0.844714}, {-3.711596, -3.341936}},
        {{16.75, -30.743902}, {-7.363918, -1.144350}, {-3.266249, -3.247275}},
        {{-29.0, 64.951905}, {-7.440922, -0.411701}, {-3.570902, -2.908924}}}});
  auto expected = LiteralUtil::CreateR3<float>({{{1.8, 2.7, 3.6, 4.5, 5.4},
                                                 {8.1, 7.2, 6.3, 5.4, 4.5},
                                                 {1.1, 2.2, 3.3, 4.4, 5.5}},
                                                {{5.4, 6.3, 7.2, 8.1, 9.0},
                                                 {4.5, 3.6, 2.7, 1.8, 0.9},
                                                 {5.5, 6.6, 7.7, 8.8, 9.9}},
                                                {{-1.8, -2.7, -3.6, -4.5, -5.4},
                                                 {-5.4, -6.3, -7.2, -8.1, -9.0},
                                                 {1.9, 2.9, 3.9, 4.9, 5.9}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// FFT tests with non-default data layout:

TEST_F(HloEvaluatorTest, 1D_FFT_8_on_c64x2x4x8_with_layout) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8]{0, 2, 1} parameter(0)
  ROOT fft = c64[2, 4, 8]{1, 2, 0} fft(operand), fft_type=FFT, fft_length={8}
}
)";
  auto input = fft_c64x2x4x8_.Relayout(LayoutUtil::MakeLayout({0, 2, 1}));
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_1d_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_1d_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 2D_FFT_4x8_on_c64x2x4x8_with_layout) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8]{2, 0, 1} parameter(0)
  ROOT fft = c64[2, 4, 8]{1, 0, 2} fft(operand), fft_type=FFT, fft_length={4, 8}
}
)";
  auto input = fft_c64x2x4x8_.Relayout(LayoutUtil::MakeLayout({2, 0, 1}));
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_2d_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_2d_, result, fft_error_));
}

TEST_F(HloEvaluatorTest, 3D_FFT_2x4x8_on_c64x2x4x8_with_layout) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[2, 4, 8]{1, 2, 0} parameter(0)
  ROOT fft =
    c64[2, 4, 8]{0, 2, 1} fft(operand), fft_type=FFT, fft_length={2, 4, 8}
}
)";
  auto input = fft_c64x2x4x8_.Relayout(LayoutUtil::MakeLayout({1, 2, 0}));
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), fft_c64x2x4x8_3d_.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(fft_c64x2x4x8_3d_, result, fft_error_));
}

// FFT tests with unusual parameters:

// Zero-length transform.
TEST_F(HloEvaluatorTest, 1D_FFT_0_on_c64x1x1x1x1) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[1, 1, 1, 1] parameter(0)
  ROOT fft = c64[1, 1, 1, 1] fft(operand), fft_type=FFT, fft_length={0}
}
)";
  auto input = LiteralUtil::CreateR4<complex64>({{{{{42.24, 24.42}}}}});
  auto expected = LiteralUtil::CreateR4<complex64>({{{{{0.0, 0.0}}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// Zero-length axis.
TEST_F(HloEvaluatorTest, 1D_FFT_1_on_c64x1x1x1x0) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[1, 1, 1, 0] parameter(0)
  ROOT fft = c64[1, 1, 1, 0] fft(operand), fft_type=FFT, fft_length={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto input,
      LiteralUtil::CreateR4<complex64>({{{{}}}}).Reshape({1, 1, 1, 0}));
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), input.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(input, result, fft_error_));
}

// Some/all dimensions have length 1.
TEST_F(HloEvaluatorTest, 1D_FFT_1_on_c64x1x1x1x1) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[1, 1, 1, 1] parameter(0)
  ROOT fft = c64[1, 1, 1, 1] fft(operand), fft_type=FFT, fft_length={1}
}
)";
  auto input = LiteralUtil::CreateR4<complex64>({{{{{42.24, 24.42}}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), input.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(input, result, fft_error_));
}

// Zero-length transform.
TEST_F(HloEvaluatorTest, 3D_FFT_1x0x1_on_c64x1x1x1x1) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[1, 1, 1, 1] parameter(0)
  ROOT fft = c64[1, 1, 1, 1] fft(operand), fft_type=FFT, fft_length={1, 0, 1}
}
)";
  auto input = LiteralUtil::CreateR4<complex64>({{{{{42.24, 24.42}}}}});
  auto expected = LiteralUtil::CreateR4<complex64>({{{{{0.0, 0.0}}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// Zero-length axis.
TEST_F(HloEvaluatorTest, 3D_FFT_1x1x1_on_c64x0x1x0x1) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[0, 1, 0, 1] parameter(0)
  ROOT fft = c64[0, 1, 0, 1] fft(operand), fft_type=FFT, fft_length={1, 1, 1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto input,
      LiteralUtil::CreateR4<complex64>({{{{}}}}).Reshape({0, 1, 0, 1}));
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), input.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(input, result, fft_error_));
}

// Some/all dimensions have length 1.
TEST_F(HloEvaluatorTest, 3D_FFT_1x1x1_on_c64x1x1x1x1) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[1, 1, 1, 1] parameter(0)
  ROOT fft = c64[1, 1, 1, 1] fft(operand), fft_type=FFT, fft_length={1, 1, 1}
}
)";
  auto input = LiteralUtil::CreateR4<complex64>({{{{{42.24, 24.42}}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), input.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(input, result, fft_error_));
}

// Some/all dimensions have length 1.
TEST_F(HloEvaluatorTest, 3D_FFT_3x1x1_on_c64x1x3x1x1) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[1, 3, 1, 1] parameter(0)
  ROOT fft = c64[1, 3, 1, 1] fft(operand), fft_type=FFT, fft_length={3, 1, 1}
}
)";
  auto input = LiteralUtil::CreateR4<complex64>(
      {{{{{42.24, 24.42}}}, {{{-42.24, 24.42}}}, {{{42.24, -24.42}}}}});
  auto expected =
      LiteralUtil::CreateR4<complex64>({{{{{42.24, 24.42}}},
                                         {{{84.5367, 97.5818}}},
                                         {{{-0.0566792, -48.7418}}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// Some/all dimensions have length 1.
TEST_F(HloEvaluatorTest, 3D_IFFT_3x1x1_on_c64x1x3x1x1) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[1, 3, 1, 1] parameter(0)
  ROOT ifft = c64[1, 3, 1, 1] fft(operand), fft_type=IFFT, fft_length={3, 1, 1}
}
)";
  auto input = LiteralUtil::CreateR4<complex64>({{{{{42.24, 24.42}}},
                                                  {{{84.5367, 97.5818}}},
                                                  {{{-0.0566792, -48.7418}}}}});
  auto expected = LiteralUtil::CreateR4<complex64>(
      {{{{{42.24, 24.42}}}, {{{-42.24, 24.42}}}, {{{42.24, -24.42}}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// Odd transform length.
TEST_F(HloEvaluatorTest, 1D_FFT_5_on_c64x5) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[5] parameter(0)
  ROOT fft = c64[5] fft(operand), fft_type=FFT, fft_length={5}
}
)";
  auto input = LiteralUtil::CreateR1<complex64>(
      {{1.0, 5.0}, {2.0, 4.0}, {3.0, 3.0}, {4.0, 2.0}, {5.0, 1.0}});
  auto expected = LiteralUtil::CreateR1<complex64>({{15.0, 15.0},
                                                    {0.940955, 5.94095},
                                                    {-1.6877, 3.3123},
                                                    {-3.3123, 1.6877},
                                                    {-5.94095, -0.940955}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// Odd transform length.
TEST_F(HloEvaluatorTest, 1D_IFFT_5_on_c64x5) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[5] parameter(0)
  ROOT ifft = c64[5] fft(operand), fft_type=IFFT, fft_length={5}
}
)";
  auto input = LiteralUtil::CreateR1<complex64>({{15.0, 15.0},
                                                 {0.940955, 5.94095},
                                                 {-1.6877, 3.3123},
                                                 {-3.3123, 1.6877},
                                                 {-5.94095, -0.940955}});
  auto expected = LiteralUtil::CreateR1<complex64>(
      {{1.0, 5.0}, {2.0, 4.0}, {3.0, 3.0}, {4.0, 2.0}, {5.0, 1.0}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// All input values are zero.
TEST_F(HloEvaluatorTest, 1D_FFT_4_on_zero_c64x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[4] parameter(0)
  ROOT fft = c64[4] fft(operand), fft_type=FFT, fft_length={4}
}
)";
  auto input = LiteralUtil::CreateR1<complex64>(
      {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), input.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(input, result, fft_error_));
}

// All input values are zero.
TEST_F(HloEvaluatorTest, 3D_FFT_3x3x4_on_zero_c64x3x3x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 3, 4] parameter(0)
  ROOT fft = c64[3, 3, 4] fft(operand), fft_type=FFT, fft_length={3, 3, 4}
}
)";
  auto input = LiteralUtil::CreateR3<complex64>(
      {{{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), input.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(input, result, fft_error_));
}

// All input values are zero.
TEST_F(HloEvaluatorTest, 3D_IFFT_3x3x4_on_zero_c64x3x3x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 3, 4] parameter(0)
  ROOT ifft = c64[3, 3, 4] fft(operand), fft_type=IFFT, fft_length={3, 3, 4}
}
)";
  auto input = LiteralUtil::CreateR3<complex64>(
      {{{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), input.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(input, result, fft_error_));
}

// All input values are zero.
TEST_F(HloEvaluatorTest, 3D_RFFT_3x3x4_on_zero_f32x3x3x4) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = f32[3, 3, 4] parameter(0)
  ROOT rfft = c64[3, 3, 3] fft(operand), fft_type=RFFT, fft_length={3, 3, 4}
}
)";
  auto input = LiteralUtil::CreateR3<float>(
      {{{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}},
       {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}},
       {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}}});
  auto expected = LiteralUtil::CreateR3<complex64>(
      {{{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// All input values are zero.
TEST_F(HloEvaluatorTest, 3D_IRFFT_3x3x4_on_zero_c64x3x3x3) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 3, 3] parameter(0)
  ROOT irfft = f32[3, 3, 4] fft(operand), fft_type=IRFFT, fft_length={3, 3, 4}
}
)";
  auto input = LiteralUtil::CreateR3<complex64>(
      {{{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}},
       {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}}});
  auto expected = LiteralUtil::CreateR3<float>(
      {{{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}},
       {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}},
       {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

// Input values, for which IRFFT discards non-zero imaginary parts.
TEST_F(HloEvaluatorTest, 2D_IRFFT_3x4_on_c64x3x3) {
  const char* hlo_text = R"(
HloModule Fft

ENTRY main {
  operand = c64[3, 3] parameter(0)
  ROOT irfft = f32[3, 4] fft(operand), fft_type=IRFFT, fft_length={3, 4}
}
)";
  auto input =
      LiteralUtil::CreateR2<complex64>({{{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}},
                                        {{3.0, 0.0}, {4.0, 0.0}, {5.0, 0.0}},
                                        {{6.0, 0.0}, {7.0, 0.0}, {8.0, 0.0}}});
  auto expected =
      LiteralUtil::CreateR2<float>({{4.0, -0.5, 0.0, -0.5},
                                    {-1.5, 0.433013, 0.0, -0.433013},
                                    {-1.5, -0.433013, 0.0, 0.433013}});
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&input}));
  EXPECT_TRUE(ShapeUtil::Compatible(result.shape(), expected.shape()));
  EXPECT_TRUE(LiteralTestUtil::Near(expected, result, fft_error_));
}

class HloEvaluatorPreciseReduceTest : public HloHardwareIndependentTestBase {};

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
  Literal result = hlo_eval.Evaluate(reduce_instruction).value();
  LiteralTestUtil::ExpectR0Equal<float>(kNumElements, result);
}

TEST_P(HloEvaluatorBf16Test, ReduceAdd) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto arg_array = std::make_unique<Array2D<float>>(2, 3);
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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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
  auto arg_array = std::make_unique<Array2D<float>>(2, 3);
  arg_array->FillUnique(1.0f);
  auto arg_literal = LiteralUtil::CreateR2FromArray2D<float>(*arg_array);

  HloInstruction* arg_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.f)));
  auto max_func = m_->AddEmbeddedComputation(MaxComputationScalarF32());

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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<float>({{6, 7}});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowMaxIotaWindowDilation) {
  auto expected = LiteralUtil::CreateR2<float>({{10, 11}, {14, 15}});
  ReduceWindowMaxIotaTest(
      /*window_size=*/2,
      /*padding=*/0,
      /*stride=*/1,
      /*window_dilation=*/2,
      /*base_dilation=*/1,
      /*expected=*/expected);
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowMaxIotaStrideWindowDilation) {
  auto expected = LiteralUtil::CreateR2<float>({{10}});
  ReduceWindowMaxIotaTest(
      /*window_size=*/2,
      /*padding=*/0,
      /*stride=*/2,
      /*window_dilation=*/2,
      /*base_dilation=*/1,
      /*expected=*/expected);
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowMaxIotaBaseDilation) {
  auto expected = LiteralUtil::CreateR2<float>({{0, 1, 1, 2, 2, 3},
                                                {4, 5, 5, 6, 6, 7},
                                                {4, 5, 5, 6, 6, 7},
                                                {8, 9, 9, 10, 10, 11},
                                                {8, 9, 9, 10, 10, 11},
                                                {12, 13, 13, 14, 14, 15}});
  ReduceWindowMaxIotaTest(
      /*window_size=*/2,
      /*padding=*/0,
      /*stride=*/1,
      /*window_dilation=*/1,
      /*base_dilation=*/2,
      /*expected=*/expected);
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowMaxIotaStrideBaseDilation) {
  auto expected =
      LiteralUtil::CreateR2<float>({{0, 1, 2}, {4, 5, 6}, {8, 9, 10}});
  ReduceWindowMaxIotaTest(
      /*window_size=*/2,
      /*padding=*/0,
      /*stride=*/2,
      /*window_dilation=*/1,
      /*base_dilation=*/2,
      /*expected=*/expected);
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowMaxIotaStrideBothDilation) {
  auto expected =
      LiteralUtil::CreateR2<float>({{5, 6, 7}, {9, 10, 11}, {13, 14, 15}});
  ReduceWindowMaxIotaTest(
      /*window_size=*/2,
      /*padding=*/0,
      /*stride=*/2,
      /*window_dilation=*/2,
      /*base_dilation=*/2,
      /*expected=*/expected);
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowMaxIotaPaddingStrideBaseDilation) {
  // The base is dilated first, and then padding is applied, hence this result.
  auto expected =
      LiteralUtil::CreateR2<float>({{0, 2, 3}, {8, 10, 11}, {12, 14, 15}});
  ReduceWindowMaxIotaTest(
      /*window_size=*/3,
      /*padding=*/1,
      /*stride=*/3,
      /*window_dilation=*/1,
      /*base_dilation=*/2,
      /*expected=*/expected);
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowAdd) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto arg_array = std::make_unique<Array2D<float>>(2, 3);
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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<float>({{1, 3, 5}, {5, 11, 13}});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, ReduceWindowAdd6D) {
  HloComputation::Builder b(TestName());

  // arg: f32[4,4,4,4,4,4] full of ones. Using small dims to limit run-time.
  std::vector<int64_t> input_dims(6, 4);
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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  std::vector<int64_t> output_dims = {4, 3, 3, 3, 4, 4};
  Literal result_literal =
      LiteralUtil::CreateFullWithDescendingLayout<float>(output_dims, 8.0f);
  EXPECT_TRUE(LiteralTestUtil::Equal(result_literal, result));
}

TEST_P(HloEvaluatorBf16Test, Min3In5Stride2Tuple) {
  HloComputation::Builder builder("main");
  auto input1 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({10000, 1000, 100, 10, 1})));
  auto input2 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({10000, 1000, 100, 10, 1})));
  HloComputation::Builder bcompute("ComputeFunction");
  auto shape1 = ShapeUtil::MakeShape(F32, {});
  auto shape2 = ShapeUtil::MakeShape(F32, {});
  auto p2 =
      bcompute.AddInstruction(HloInstruction::CreateParameter(0, shape1, "x0"));
  auto p3 =
      bcompute.AddInstruction(HloInstruction::CreateParameter(1, shape2, "x1"));
  auto p4 =
      bcompute.AddInstruction(HloInstruction::CreateParameter(2, shape1, "y0"));
  auto p5 =
      bcompute.AddInstruction(HloInstruction::CreateParameter(3, shape2, "y1"));
  std::vector<HloInstruction*> compute_vec = {
      bcompute.AddInstruction(
          HloInstruction::CreateBinary(shape1, HloOpcode::kMinimum, p2, p4)),
      bcompute.AddInstruction(
          HloInstruction::CreateBinary(shape2, HloOpcode::kMinimum, p3, p5))};
  bcompute.AddInstruction(HloInstruction::CreateTuple(compute_vec));
  auto compute_tuple = m_->AddEmbeddedComputation(bcompute.Build());
  std::vector<HloInstruction*> input_vec = {input1, input2};
  auto init1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::MaxValue(F32)));
  auto init2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::MaxValue(F32)));
  std::vector<HloInstruction*> init_vec = {init1, init2};
  auto padding = std::pair<int64_t, int64_t>(0, 0);
  TF_ASSERT_OK_AND_ASSIGN(auto window,
                          ShapeInference::InferWindowFromDimensions(
                              {3}, {2}, absl::MakeSpan(&padding, 1),
                              /*lhs_dilation=*/{},
                              /*rhs_dilation=*/{}));
  std::vector<const Shape*> input_shapes = {&input1->shape(), &input2->shape()};
  std::vector<const Shape*> init_shapes = {&init1->shape(), &init2->shape()};
  TF_ASSERT_OK_AND_ASSIGN(Shape shape,
                          ShapeInference::InferReduceWindowShape(
                              input_shapes, init_shapes, window,
                              compute_tuple->ComputeProgramShape()));
  builder.AddInstruction(HloInstruction::CreateReduceWindow(
      shape, input_vec, init_vec, window, compute_tuple));
  auto r1 = LiteralUtil::CreateR1<float>({100, 1});
  auto expected = LiteralUtil::MakeTuple({&r1, &r1});
  m_->AddEntryComputation(builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, Min3In5Stride2TupleDiffInput) {
  HloComputation::Builder builder("main");
  auto input1 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({10000, 1000, 100, 10, 1})));
  auto input2 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int>({15, 28, 300, 107, 12})));
  HloComputation::Builder bcompute("ComputeFunction");
  auto shape1 = ShapeUtil::MakeShape(F32, {});
  auto shape2 = ShapeUtil::MakeShape(S32, {});
  auto p2 =
      bcompute.AddInstruction(HloInstruction::CreateParameter(0, shape1, "x0"));
  auto p3 =
      bcompute.AddInstruction(HloInstruction::CreateParameter(1, shape2, "x1"));
  auto p4 =
      bcompute.AddInstruction(HloInstruction::CreateParameter(2, shape1, "y0"));
  auto p5 =
      bcompute.AddInstruction(HloInstruction::CreateParameter(3, shape2, "y1"));
  std::vector<HloInstruction*> compute_vec = {
      bcompute.AddInstruction(
          HloInstruction::CreateBinary(shape1, HloOpcode::kMinimum, p2, p4)),
      bcompute.AddInstruction(
          HloInstruction::CreateBinary(shape2, HloOpcode::kMinimum, p3, p5))};
  bcompute.AddInstruction(HloInstruction::CreateTuple(compute_vec));
  auto compute_tuple = m_->AddEmbeddedComputation(bcompute.Build());
  std::vector<HloInstruction*> input_vec = {input1, input2};
  auto init1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::MaxValue(F32)));
  auto init2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::MaxValue(S32)));
  std::vector<HloInstruction*> init_vec = {init1, init2};
  auto padding = std::pair<int64_t, int64_t>(0, 0);
  TF_ASSERT_OK_AND_ASSIGN(auto window,
                          ShapeInference::InferWindowFromDimensions(
                              {3}, {2}, absl::MakeSpan(&padding, 1),
                              /*lhs_dilation=*/{},
                              /*rhs_dilation=*/{}));
  std::vector<const Shape*> input_shapes = {&input1->shape(), &input2->shape()};
  std::vector<const Shape*> init_shapes = {&init1->shape(), &init2->shape()};
  TF_ASSERT_OK_AND_ASSIGN(Shape shape,
                          ShapeInference::InferReduceWindowShape(
                              input_shapes, init_shapes, window,
                              compute_tuple->ComputeProgramShape()));
  builder.AddInstruction(HloInstruction::CreateReduceWindow(
      shape, input_vec, init_vec, window, compute_tuple));
  auto r1 = LiteralUtil::CreateR1<float>({100, 1});
  auto r2 = LiteralUtil::CreateR1<int>({15, 12});
  auto expected = LiteralUtil::MakeTuple({&r1, &r2});
  m_->AddEntryComputation(builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, StridedSlice) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[3,5] {
  //  { 1, 2, 3, 4, 5 },
  //  { 9, 10, 11, 12, 13 },
  //  { 17, 18, 19, 20, 21 },
  // }
  auto operand_array = std::make_unique<Array2D<float>>(3, 5);
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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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
  auto operand_array = std::make_unique<Array2D<float>>(2, 4);
  operand_array->FillUnique(1.0f);
  auto operand_literal =
      LiteralUtil::CreateR2FromArray2D<float>(*operand_array);

  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  auto zero = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  auto one = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));

  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  b.AddInstruction(
      HloInstruction::CreateDynamicSlice(shape, operand, {zero, one}, {2, 3}));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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
  auto operand_array = std::make_unique<Array2D<float>>(2, 4);
  operand_array->FillUnique(1.0f);
  auto operand_literal =
      LiteralUtil::CreateR2FromArray2D<float>(*operand_array);

  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  auto two = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(2)));
  auto one = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));

  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  b.AddInstruction(
      HloInstruction::CreateDynamicSlice(shape, operand, {two, one}, {2, 3}));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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
  auto operand_array = std::make_unique<Array2D<double>>(2, 3);
  operand_array->FillUnique(1.0);
  auto operand_literal =
      LiteralUtil::CreateR2FromArray2D<double>(*operand_array);

  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  auto zero = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  auto one = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));

  auto update = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<double>({{-2.0, -3.0}, {-6.0, -7.0}})));

  Shape shape = ShapeUtil::MakeShape(F64, {2, 3});
  b.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      shape, operand, update, {zero, one}));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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
  auto operand_array = std::make_unique<Array2D<double>>(2, 3);
  operand_array->FillUnique(1.0);
  auto operand_literal2 =
      LiteralUtil::CreateR2FromArray2D<double>(*operand_array);

  HloInstruction* operand2 = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal2)));
  HloInstruction* operand1 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int64_t>({0, 1})));

  auto tuple =
      b.AddInstruction(HloInstruction::CreateTuple({operand1, operand2}));

  Shape shape = ShapeUtil::MakeShape(F64, {2, 3});
  b.AddInstruction(HloInstruction::CreateGetTupleElement(shape, tuple, 1));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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
  auto operand_array = std::make_unique<Array2D<double>>(2, 3);
  operand_array->FillUnique(1.0);

  HloInstruction* operand2 = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2FromArray2D<double>(*operand_array)));
  HloInstruction* operand1 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int64_t>({0, 1})));

  auto tuple1 =
      b.AddInstruction(HloInstruction::CreateTuple({operand1, operand2}));
  auto tuple2 =
      b.AddInstruction(HloInstruction::CreateTuple({operand2, operand2}));

  auto outer_tuple =
      b.AddInstruction(HloInstruction::CreateTuple({tuple1, tuple2}));

  b.AddInstruction(
      HloInstruction::CreateGetTupleElement(tuple2->shape(), outer_tuple, 1));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator.EvaluateWithSubstitutions(
          add, {{param0, &param0_literal}, {square, &square_literal}}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({11, 22, 33, 44}), result));
}

TEST_F(HloEvaluatorTest, EvaluateWithSubstitutionsRecursive) {
  const char* hlo = R"(
  HloModule test

  ENTRY main {
    param = s32[] parameter(0)
    c1 = s32[] constant(1)
    c2 = s32[] constant(2)
    add.1 = s32[] add(c1, c2)
    ROOT add.2 = s32[] add(param, add.1)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  Literal param_value = LiteralUtil::CreateR0(PrimitiveType::S32, 3);
  HloInstruction* param = module->entry_computation()->parameter_instruction(0);
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      evaluator_.EvaluateWithSubstitutions(
          /*instruction=*/module->entry_computation()->root_instruction(),
          /*substitutions=*/{{param, &param_value}},
          /*recursively_evaluate_nonconstant_operands=*/true));
  EXPECT_EQ(result, LiteralUtil::CreateR0(PrimitiveType::S32, 1 + 2 + 3));
}

TEST_F(HloEvaluatorTest,
       EvaluateWithSubstitutionsRecursiveWithDeepSubstitutions) {
  const char* hlo = R"(
  HloModule test
  ENTRY main {
    param = s32[] parameter(0)
    c1 = s32[] constant(1)
    c2 = s32[] constant(2)
    add.1 = s32[] add(param, c1)
    add.2 = s32[] add(add.1, c2)
    ROOT add.3 = s32[] add(add.2, c1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  Literal param_value = LiteralUtil::CreateR0(PrimitiveType::S32, 4);
  HloInstruction* param = module->entry_computation()->parameter_instruction(0);
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator_.EvaluateWithSubstitutions(
          /*instruction=*/module->entry_computation()->root_instruction(),
          /*substitutions=*/{{param, &param_value}},
          /*recursively_evaluate_nonconstant_operands=*/true));
  EXPECT_EQ(result, LiteralUtil::CreateR0(PrimitiveType::S32, 4 + 1 + 2 + 1));
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
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator.EvaluateWithSubstitutions(add, {{square, &square_literal}}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({11, 22, 33, 44}), result));
}

// Check that EvaluateWithSubstitutions works if the thing we're evaluating is
// being substituted.
TEST_P(HloEvaluatorBf16Test, EvaluateSubstitutedInstruction) {
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  HloInstruction* param =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param0"));

  HloEvaluator evaluator;
  Literal literal = LiteralUtil::CreateR1<float>({10, 20, 30, 40});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, evaluator.EvaluateWithSubstitutions(
                                              param, {{param, &literal}}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({10, 20, 30, 40}), result));
}

TEST_F(HloEvaluatorTest, EvaluateWithSubstitutionsLiteralBase) {
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(S64, {3});

  HloInstruction* param0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param0"));
  HloInstruction* square = b.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kMultiply, param0, param0));

  int64_t int64_values[] = {1, 2, 3};
  const Shape literal_shape = ShapeUtil::MakeShape(S64, {3});

  BorrowingLiteral literal(reinterpret_cast<const char*>(int64_values),
                           literal_shape);
  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(Literal result, evaluator.EvaluateWithSubstitutions(
                                              square, {{param0, &literal}}));
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int64_t>({1, 4, 9}),
                                     result));
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {7, 8, 9}}), result));
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{1, 3}, {4, 6}, {7, 9}}), result));
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 1}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR3<int32_t>(
          {{{1, 3}, {4, 6}, {7, 9}}, {{3, 2}, {6, 5}, {9, 8}}}),
      result));
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
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{-1, 1}, {-4, 4}}), result));
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
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{-2, 2}, {-1, 1}}), result));
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({1, 1});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR2<int32_t>({{5}}), result));
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{2, 1}, {1, 1}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR3<int32_t>({{{8}}, {{5}}}), result));
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
  Literal operand = LiteralUtil::CreateR2<int32_t>({{}, {}, {}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR2<int32_t>({{}, {}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_NoOutputWindowDims) {
  const std::string hlo_text = R"(
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

  Literal operand = LiteralUtil::CreateR1<int32_t>({0, 1, 2});
  Literal start_indices =
      LiteralUtil::CreateR3<int32_t>({{{0}, {1}}, {{2}, {1}}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{0, 1}, {2, 1}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_ExplicitBatchDims) {
  const std::string hlo_text = R"(
HloModule gather

ENTRY main {
  operand = s32[3,2,1,3] parameter(0)
  indices = s32[3,2] parameter(1)
  ROOT gather = s32[3,2,2] gather(operand, indices),
      offset_dims={2},
      collapsed_slice_dims={2},
      start_index_map={0},
      index_vector_dim=2,
      slice_sizes={2,1,1,1},
      operand_batching_dims={1,3},
      start_indices_batching_dims={1,0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  Literal operand =
      LiteralUtil::CreateR4<int32_t>({{{{1, 2, 3}}, {{4, 5, 6}}},
                                      {{{7, 8, 9}}, {{10, 11, 12}}},
                                      {{{13, 14, 15}}, {{16, 17, 18}}}});
  Literal start_indices =
      LiteralUtil::CreateR2<int32_t>({{1, 0}, {0, 1}, {1, 0}});
  Literal expected_result = LiteralUtil::CreateR3<int32_t>(
      {{{7, 13}, {4, 10}}, {{2, 8}, {11, 17}}, {{9, 15}, {6, 12}}});

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_result, result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_GetDiagonal) {
  const std::string hlo_text = R"(
HloModule module

ENTRY %module {
  %operand = f32[4,4] parameter(0)
  %indices = s32[4,1] iota(), iota_dimension=0
  ROOT %gather = f32[4,1] gather(%operand, %indices), offset_dims={},
    collapsed_slice_dims={1}, start_index_map={1}, operand_batching_dims={0},
    start_indices_batching_dims={0}, index_vector_dim=2, slice_sizes={1,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  Literal operand = LiteralUtil::CreateR2<float>({{0.0, 0.1, 0.2, 0.3},
                                                  {1.0, 1.1, 1.2, 1.3},
                                                  {2.0, 2.1, 2.2, 2.3},
                                                  {3.0, 3.1, 3.2, 3.3}});
  Literal expected_result =
      LiteralUtil::CreateR2<float>({{0.0}, {1.1}, {2.2}, {3.3}});

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_result, result));
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {4, 5, 6}, {70, 80, 90}}),
      result));
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 30}, {40, 60}, {70, 90}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{10, 2, 30}, {40, 5, 60}, {70, 8, 90}}),
      result));
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{11, 22, 33}, {4, 5, 6}, {77, 88, 99}}),
      result));
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR2<int32_t>(
                                 {{10, 40, 90}, {4, 5, 6}, {490, 640, 810}}),
                             result));
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
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({2, 1});
  Literal updates =
      LiteralUtil::CreateR2<float>({{0.4, 1.1, 0.7}, {2.3, 3.1, 1.6}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Near(
      LiteralUtil::CreateR2<float>(
          {{1.1, 2.2, 3.3}, {6.7, 8.6, 8.2}, {8.1, 9.9, 10.6}}),
      result, ErrorSpec{0.1, 0.01}));
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({1, 1});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {84, 105, 126}, {7, 8, 9}}),
      result));
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 1}});
  Literal updates = LiteralUtil::CreateR3<int32_t>(
      {{{10, 30}, {40, 60}, {70, 90}}, {{5, 5}, {5, 5}, {5, 5}}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR2<int32_t>(
                                 {{11, 7, 38}, {44, 10, 71}, {77, 13, 104}}),
                             result));
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
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{-10, 10}, {-40, 40}});
  Literal expected =
      LiteralUtil::CreateR3<int32_t>({{{-10, 10}, {-2, 2}, {-3, 3}},  //
                                      {{-40, 40}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
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
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{-10, 10}, {-20, 20}});
  Literal expected =
      LiteralUtil::CreateR3<int32_t>({{{-20, 20}, {-10, 10}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},      //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({1, 1});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{10}});
  Literal expected =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 10, 6}, {7, 8, 9}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{2, 1}, {1, 1}});
  Literal updates = LiteralUtil::CreateR3<int32_t>({{{10}}, {{20}}});
  Literal expected =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 20, 6}, {7, 10, 9}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
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
  Literal operand = LiteralUtil::CreateR2<int32_t>({{}, {}, {}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{}, {}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(operand, result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_NoUpdateWindowDims) {
  const std::string hlo_text = R"(
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

  Literal operand = LiteralUtil::CreateR1<int32_t>({0, 1, 2});
  Literal scatter_indices =
      LiteralUtil::CreateR3<int32_t>({{{0}, {1}}, {{2}, {1}}});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{10, 20}, {30, 40}});
  Literal expected = LiteralUtil::CreateR1<int32_t>({10, 61, 32});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  // No updates should happen for the negative indices.
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({-1, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {77, 88, 99}}),
      EvaluateWithModule(module.get(),
                         {&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_OobIndices) {
  const std::string hlo_text = R"(
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
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  // No updates should happen for the OOB indices.
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {2147483647, 1}, {1, 2}});
  Literal updates = LiteralUtil::CreateR3<int32_t>(
      {{{10}}, {{20}}, {{30}}, {{40}}, {{50}}, {{60}}});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 30, 60}, {7, 20, 9}}),
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
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 2}});
  Literal updates = LiteralUtil::CreateR3<int32_t>({{{-10, 10}, {-40, 40}}});
  // Given the update window size of 2,2 and the index of 0,2, the update window
  // will be OOB. So, nothing should be updated.
  Literal expected = operand.Clone();
  EXPECT_TRUE(LiteralTestUtil::Equal(
      expected, EvaluateWithModule(module.get(),
                                   {&operand, &scatter_indices, &updates})));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_Multioutput) {
  const char* hlo_text = R"(
HloModule MultioutputScatter

update {
  lhs0 = s32[] parameter(0)
  lhs1 = f32[] parameter(1)
  rhs0 = s32[] parameter(2)
  rhs1 = f32[] parameter(3)
  ROOT tuple = (s32[], f32[]) tuple(rhs0, rhs1)
}

ENTRY main {
  operand0 = s32[3,3,2] parameter(0)
  operand1 = f32[3,3,2] parameter(1)
  indices = s32[2,2] parameter(2)
  updates0 = s32[2,2] parameter(3)
  updates1 = f32[2,2] parameter(4)
  ROOT scatter = (s32[3,3,2], f32[3,3,2]) scatter(operand0, operand1, indices, updates0, updates1),
      to_apply=update,
      update_window_dims={1},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand0 =
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal operand1 =
      LiteralUtil::CreateR3<float>({{{-2, 2}, {-3, 3}, {-4, 4}},  //
                                    {{-5, 5}, {-6, 6}, {-7, 7}},  //
                                    {{-8, 8}, {-9, 9}, {-10, 10}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  Literal updates0 = LiteralUtil::CreateR2<int32_t>({{-10, 10}, {-40, 40}});
  Literal updates1 = LiteralUtil::CreateR2<float>({{-11, 11}, {-41, 41}});
  Literal expected = LiteralUtil::MakeTupleOwned(
      LiteralUtil::CreateR3<int32_t>({{{-10, 10}, {-2, 2}, {-3, 3}},  //
                                      {{-40, 40}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}}),
      LiteralUtil::CreateR3<float>({{{-11, 11}, {-3, 3}, {-4, 4}},  //
                                    {{-41, 41}, {-6, 6}, {-7, 7}},  //
                                    {{-8, 8}, {-9, 9}, {-10, 10}}}));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      Evaluate({&operand0, &operand1, &scatter_indices, &updates0, &updates1}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_ExplicitBatchDims) {
  const char* hlo_text = R"(
  HloModule ScatterExplicitBatchDims
  add_s32 {
    x = s32[] parameter(0)
    y = s32[] parameter(1)
    ROOT s = s32[] add(x,y)
  }

  ENTRY main {
    indices = s32[2,3,5] parameter(0)
    update = s32[2,3,2,5] parameter(1)
    z = s32[] constant(0)
    input = s32[5,3,2,2] broadcast(z), dimensions={}
    ROOT s = s32[5,3,2,2] scatter(input, indices, update),
      update_window_dims={2},
      inserted_window_dims={1},
      scatter_dims_to_operand_dims={1},
      index_vector_dim=3,
      input_batching_dims={0,3},
      scatter_indices_batching_dims={2,0},
      to_apply=add_s32
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto indices =
      std::make_unique<Literal>(ShapeUtil::MakeShape(S32, {2, 3, 5}));
  indices
      ->Populate<int>([](absl::Span<const int64_t> indices) {
        return static_cast<int>((indices[1] + 1) % 3);
      })
      .IgnoreError();
  auto updates =
      std::make_unique<Literal>(ShapeUtil::MakeShape(S32, {2, 3, 2, 5}));
  updates
      ->Populate<int>([](absl::Span<const int64_t> indices) {
        return static_cast<int>(indices[0] * 1000 + indices[1] * 100 +
                                indices[2] * 10 + indices[3]);
      })
      .IgnoreError();
  Literal expected =
      LiteralUtil::CreateR4<int32_t>({{{{200, 1200}, {210, 1210}},
                                       {{0, 1000}, {10, 1010}},
                                       {{100, 1100}, {110, 1110}}},
                                      {{{201, 1201}, {211, 1211}},
                                       {{1, 1001}, {11, 1011}},
                                       {{101, 1101}, {111, 1111}}},
                                      {{{202, 1202}, {212, 1212}},
                                       {{2, 1002}, {12, 1012}},
                                       {{102, 1102}, {112, 1112}}},
                                      {{{203, 1203}, {213, 1213}},
                                       {{3, 1003}, {13, 1013}},
                                       {{103, 1103}, {113, 1113}}},
                                      {{{204, 1204}, {214, 1214}},
                                       {{4, 1004}, {14, 1014}},
                                       {{104, 1104}, {114, 1114}}}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({indices.get(), updates.get()}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
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

  HloComputation::Builder b(TestName());
  auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs)));
  auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs)));
  b.AddInstruction(HloInstruction::CreateCompare(expected.shape(), c1, c2,
                                                 ComparisonDirection::kGe));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_P(HloEvaluatorBf16Test, Bf16Reduction) {
  const std::string hlo_text = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&arg}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, MixedPrecisionReduction) {
  const std::string hlo_text = R"(
HloModule MixedPrecisionReduction

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main {
  arg0 = f32[4]{0} parameter(0)
  init = f32[] constant(0)
  ROOT %reduce = bf16[] reduce(arg0, init), dimensions={0}, to_apply=add_f32
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  Literal arg = LiteralUtil::CreateR1<float>({1.0f, 3.0f, -2.0f, 42.0f});
  Literal expected = LiteralUtil::CreateR0<bfloat16>(bfloat16(44.0f));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&arg}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, DontFailOnCallUnimplementedOps) {
  // Outfeed triggers unimplemented error within HandleCall, and we verify that
  // the Evaluator does fail in such case.
  const std::string hlo_text = R"(
HloModule DontFailOnCall

call {
  token0 = token[] after-all()
  constant = u32[3]{0} constant({1,2,3})
  ROOT  outfeed = token[] outfeed(constant, token0), outfeed_shape=u32[3]{0}
}

ENTRY main {
  ROOT result = token[] call(), to_apply=call
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto statusor = Evaluate();
  EXPECT_FALSE(statusor.status().ok());
}

TEST_F(HloEvaluatorTest, DontFailOnFusionWithUnimplementedOps) {
  // Outfeed triggers unimplemented error within HandleFusion, and we verify
  // that the Evaluator does fail in such case.
  const std::string hlo_text = R"(
HloModule DontFailOnFusion

fused_computation {
  token0 = token[] after-all()
  constant = u32[3]{0} constant({1,2,3})
  ROOT  outfeed = token[] outfeed(constant, token0), outfeed_shape=u32[3]{0}
}

ENTRY main {
  ROOT result = token[] fusion(), kind=kLoop, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto statusor = Evaluate();
  EXPECT_FALSE(statusor.status().ok());
}

TEST_P(HloEvaluatorBf16Test, SliceWithDifferentLayout) {
  // Regression test for b/114735354.
  const std::string hlo_text = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, Evaluate({&arg}));
  EXPECT_TRUE(LiteralTestUtil::Equal(arg, actual));
}

TEST_P(HloEvaluatorBf16Test, Bitcast) {
  // Regression test for b/114735354.
  const absl::string_view hlo_text_base = R"(
HloModule Bitcast

ENTRY main {
  param = %s[32,121]{1,0} parameter(0)
  ROOT bitcast = %s[121,32,1]{0,1,2} bitcast(%s[32,121]{1,0} param)
}
)";
  std::string hlo_text;
  if (use_bfloat16_) {
    hlo_text = absl::StrFormat(hlo_text_base, "bf16", "bf16", "bf16");
  } else {
    hlo_text = absl::StrFormat(hlo_text_base, "f32", "f32", "f32");
  }
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto args = MakeFakeArguments(m_.get()).value();
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, Evaluate({&args[0]}));
  if (use_bfloat16_) {
    EXPECT_TRUE(
        absl::c_equal(args[0].data<bfloat16>(), actual.data<bfloat16>()));
  } else {
    EXPECT_TRUE(absl::c_equal(args[0].data<float>(), actual.data<float>()));
  }
}

TEST_P(HloEvaluatorBf16Test, BitcastWithoutLayout) {
  const absl::string_view hlo_text_base = R"(
HloModule Bitcast

ENTRY main {
  param = %s[2,4] parameter(0)
  ROOT bitcast = %s[4,2,1] bitcast(%s[2,4] param)
}
)";
  std::string hlo_text;
  Literal arg;
  if (use_bfloat16_) {
    hlo_text = absl::StrFormat(hlo_text_base, "bf16", "bf16", "bf16");
    arg = LiteralUtil::CreateR2<bfloat16>(
        {{bfloat16(1), bfloat16(2), bfloat16(3), bfloat16(4)},
         {bfloat16(5), bfloat16(6), bfloat16(7), bfloat16(8)}});
  } else {
    hlo_text = absl::StrFormat(hlo_text_base, "f32", "f32", "f32");
    arg = LiteralUtil::CreateR2<float>({{1., 2., 3., 4.}, {5., 6., 7., 8.}});
  }

  HloParserOptions parser_config;
  parser_config.set_fill_missing_layouts(false);
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnUnverifiedModule(
                                  hlo_text, HloModuleConfig(), parser_config));

  absl::StatusOr<Literal> actual = Evaluate({&arg});
  EXPECT_FALSE(actual.ok());
  EXPECT_EQ(actual.status().message(),
            "Evaluator cannot evaluate bitcast for non-scalar operand without "
            "assigned layout.");
}

TEST_P(HloEvaluatorBf16Test, EffectiveScalarBitcastWithoutLayout) {
  const absl::string_view hlo_text_base = R"(
HloModule Bitcast

ENTRY main {
  param = %s[1,1] parameter(0)
  ROOT bitcast = %s[1,1,1] bitcast(%s[1,1] param)
}
)";
  std::string hlo_text;
  Literal arg;
  if (use_bfloat16_) {
    hlo_text = absl::StrFormat(hlo_text_base, "bf16", "bf16", "bf16");
    arg = LiteralUtil::CreateR2<bfloat16>({{bfloat16(2)}});
  } else {
    hlo_text = absl::StrFormat(hlo_text_base, "f32", "f32", "f32");
    arg = LiteralUtil::CreateR2<float>({{2.}});
  }

  HloParserOptions parser_config;
  parser_config.set_fill_missing_layouts(false);
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnUnverifiedModule(
                                  hlo_text, HloModuleConfig(), parser_config));

  TF_ASSERT_OK_AND_ASSIGN(Literal actual, Evaluate({&arg}));
  if (use_bfloat16_) {
    EXPECT_TRUE(absl::c_equal(arg.data<bfloat16>(), actual.data<bfloat16>()));
  } else {
    EXPECT_TRUE(absl::c_equal(arg.data<float>(), actual.data<float>()));
  }
}

// Check that s32 under/overflow doesn't trigger a ubsan failure.
TEST_F(HloEvaluatorTest, Int32Overflow) {
  const absl::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  c1 = s32[] constant(1073741824)  // 2^30
  sum = s32[] add(c1, c1)  // 2^31, i.e. INT_MIN

  c2 = s32[] constant(-2147483648)  // -2^31
  sub = s32[] subtract(c2, c1)  // -2^31 - 2^30, underflows

  c3 = u32[] constant(4294967295)
  c4 = u32[] constant(33)

  mul = s32[] multiply(c1, c1)

  pow = u32[] power(c3, c4)
  ROOT tuple = (s32[], s32[], s32[], u32[]) tuple(sum, sub, mul, pow)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(auto literal, Evaluate({}));
  std::vector<Literal> actual = literal.DecomposeTuple();
  ASSERT_EQ(actual.size(), 4);

  uint32_t pow30 = uint32_t{1} << 30;
  uint32_t pow31 = uint32_t{1} << 31;
  EXPECT_EQ(actual[0].GetFirstElement<int32_t>(), static_cast<int32_t>(pow31));
  EXPECT_EQ(actual[1].GetFirstElement<int32_t>(),
            static_cast<int32_t>(-(pow31 + pow30)));
  EXPECT_EQ(actual[2].GetFirstElement<int32_t>(),
            static_cast<int32_t>(pow31 * pow31));
  EXPECT_EQ(actual[3].GetFirstElement<uint32_t>(), uint32_t{4294967295});
}

TEST_F(HloEvaluatorTest, GetDimensionSize) {
  const absl::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  size = s32[] parameter(0)

  data = s32[4] parameter(1)

  data_dynamic = s32[<=4] set-dimension-size(data, size), dimensions={0}

  sum = s32[<=4] add(data_dynamic, data)

  ROOT dynamic_size = s32[] get-dimension-size(sum), dimensions={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(DynamicDimensionInference dynamic_dimension_inference,
                          DynamicDimensionInference::Run(m_.get()));

  evaluator_.set_dynamic_dimension_inference(&dynamic_dimension_inference);
  Literal size_arg = LiteralUtil::CreateR0<int32_t>(3);
  Literal data_arg = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4});

  TF_ASSERT_OK_AND_ASSIGN(Literal actual, Evaluate({&size_arg, &data_arg}));

  EXPECT_EQ(actual.GetFirstElement<int32_t>(), static_cast<int32_t>(3));
}

// Check that we get a useful error if we pass inputs of the wrong shape.
TEST_F(HloEvaluatorTest, EvaluateWithWrongInputShapes) {
  const absl::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  p0 = s32[1] parameter(0)
  ROOT sum = s32[1] add(p0, p0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal input_wrong_shape = LiteralUtil::CreateR1<int32_t>({0, 1});

  EXPECT_EQ(
      HloEvaluator().Evaluate(*m_, {&input_wrong_shape}).status().message(),
      "Shape mismatch at parameter 0. Computation expected s32[1]{0}, "
      "but arg was s32[2]{0}.");
  EXPECT_EQ(HloEvaluator()
                .Evaluate(*m_->entry_computation(), {&input_wrong_shape})
                .status()
                .message(),
            "Shape mismatch at parameter 0. Computation expected s32[1]{0}, "
            "but arg was s32[2]{0}.");
}

// Check that we get a useful error if we pass too many or too few inputs.
TEST_F(HloEvaluatorTest, EvaluateWithWrongNumberOfInputs) {
  const absl::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  p0 = s32[1] parameter(0)
  ROOT sum = s32[1] add(p0, p0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal input = LiteralUtil::CreateR1<int32_t>({0});

  EXPECT_EQ(HloEvaluator().Evaluate(*m_, {&input, &input}).status().message(),
            "Expected 1 argument, but got 2.");
  EXPECT_EQ(HloEvaluator()
                .Evaluate(*m_->entry_computation(), {&input, &input})
                .status()
                .message(),
            "Expected 1 argument, but got 2.");
}

TEST_F(HloEvaluatorTest, PreserveFusionInputLayout) {
  const absl::string_view hlo_text = R"(
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
  auto args = MakeFakeArguments(m_.get()).value();

  TF_ASSERT_OK_AND_ASSIGN(Literal actual, Evaluate({&args[0]}));
  EXPECT_TRUE(absl::c_equal(args[0].data<float>(), actual.data<float>()));
}

TEST_F(HloEvaluatorTest, PreserveFusionOutputLayout) {
  const absl::string_view hlo_text = R"(
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
  auto args = MakeFakeArguments(m_.get()).value();
  TF_ASSERT_OK_AND_ASSIGN(Literal actual, Evaluate({&args[0]}));
  EXPECT_TRUE(absl::c_equal(args[0].data<float>(), actual.data<float>()));
}

TEST_F(HloEvaluatorTest, PreserveMOFusionOutputLayout) {
  const absl::string_view hlo_text = R"(
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
  auto args = MakeFakeArguments(m_.get()).value();
  TF_ASSERT_OK_AND_ASSIGN(Literal actual_tuple, Evaluate({&args[0]}));
  std::vector<Literal> actual_literals = actual_tuple.DecomposeTuple();
  EXPECT_TRUE(
      absl::c_equal(args[0].data<float>(), actual_literals[0].data<float>()));
}

// Tests that custom_calls fail to evaluate when no handler is specified.
TEST_F(HloEvaluatorTest, EvaluateCustomCall_NoHandler) {
  const absl::string_view hlo_text = R"(
    HloModule EvaluateCustomCall_NoHandler
    ENTRY kernel_entry {
      parameter.0 = u32[2,2]{1,0} parameter(0)
      ROOT test_root = (u32[2,2]{1,0}) custom-call(parameter.0),
          custom_call_target="_my_custom_call"
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto args = MakeFakeArguments(m_.get()).value();
  EXPECT_EQ(HloEvaluator().Evaluate(*m_, {&args[0]}).status().code(),
            ::tsl::error::UNIMPLEMENTED);
}

// Tests when a custom_call handler returns an error.
TEST_F(HloEvaluatorTest, EvaluateCustomCall_HandlerError) {
  const absl::string_view hlo_text = R"(
    HloModule EvaluateCustomCall_HandlerError
    ENTRY kernel_entry {
      parameter.0 = u32[2,2]{1,0} parameter(0)
      ROOT test_root = (u32[2,2]{1,0}) custom-call(parameter.0),
          custom_call_target="_my_custom_call"
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto args = MakeFakeArguments(m_.get()).value();
  HloEvaluator evaluator;
  evaluator.set_custom_call_handler([](const HloInstruction* custom_call,
                                       absl::Span<const Literal*> operands) {
    return Internal("Test error");
  });
  EXPECT_EQ(evaluator.Evaluate(*m_, {&args[0]}).status().code(),
            ::tsl::error::INTERNAL);
}

// Tests the custom_call handler on calls with many inputs.
// We sum the operands so that we can verify the operand and output literals
// are properly mapped for access.
TEST_F(HloEvaluatorTest, EvaluateCustomCall_ManyInputs) {
  const absl::string_view hlo_text = R"(
    HloModule EvaluateCustomCall_ManyInputs
    ENTRY kernel_entry {
      parameter.0 = u32[1]{0} parameter(0)
      parameter.1 = u32[1]{0} parameter(1)
      ROOT test_root = u32[1]{0} custom-call(parameter.0, parameter.1),
          custom_call_target="_my_custom_call"
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto args = MakeFakeArguments(m_.get()).value();
  HloEvaluator evaluator;
  evaluator.set_custom_call_handler([](const HloInstruction* custom_call,
                                       absl::Span<const Literal*> operands) {
    EXPECT_EQ(HloOpcode::kCustomCall, custom_call->opcode());
    EXPECT_EQ("_my_custom_call", custom_call->custom_call_target());
    EXPECT_EQ(2, custom_call->operand_count());
    EXPECT_EQ(2, operands.size());
    auto output = Literal::CreateFromShape(custom_call->shape());
    auto operand0_data = operands[0]->data<uint32_t>();
    auto operand1_data = operands[1]->data<uint32_t>();
    auto output_data = output.data<uint32_t>();
    output_data[0] = operand0_data[0] + operand1_data[0];
    return output;
  });
  TF_ASSERT_OK_AND_ASSIGN(
      Literal actual_literal,
      evaluator.Evaluate(*m_->entry_computation(), {&args[0], &args[1]}));
  auto arg0_data = args[0].data<uint32_t>();
  auto arg1_data = args[1].data<uint32_t>();
  std::vector<uint32_t> expected_data = {arg0_data[0] + arg1_data[0]};
  EXPECT_TRUE(absl::c_equal(expected_data, actual_literal.data<uint32_t>()));
}

TEST_F(HloEvaluatorTest, EvaluateCustomCallInFusion) {
  const absl::string_view hlo_text = R"(
fusion1 {
  p = f32[] parameter(0)
  ROOT c = f32[] custom-call(p), custom_call_target="__cchandler1"
}

ENTRY e {
  p = f32[] parameter(0)
  ROOT f = f32[] fusion(p), kind=kCustom, calls=fusion1
})";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  auto input = LiteralUtil::CreateR0<float>(0);
  HloEvaluator evaluator;
  evaluator.set_custom_call_handler([](const HloInstruction* custom_call,
                                       absl::Span<const Literal*> operands) {
    return LiteralUtil::CreateR0<float>(1 -
                                        operands[0]->GetFirstElement<float>());
  });
  TF_ASSERT_OK_AND_ASSIGN(auto output, evaluator.Evaluate(*m_, {&input}));
  EXPECT_EQ(output, LiteralUtil::CreateR0<float>(1));
}

TEST_F(HloEvaluatorTest, IsFiniteF16) {
  const absl::string_view hlo_text = R"(
  HloModule test

  ENTRY IsFiniteTest {
    c = f16[6] constant({nan, 7, nan, -1, inf, -inf})
    ROOT is-finite = pred[6] is-finite(c)
  })";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal actual_literal,
      HloEvaluator().Evaluate(*m_->entry_computation(), {}));
  EXPECT_THAT(actual_literal.data<bool>(),
              ::testing::ElementsAre(false, true, false, true, false, false));
}

TEST_F(HloEvaluatorTest, IsFiniteBf16) {
  const absl::string_view hlo_text = R"(
  HloModule test

  ENTRY IsFiniteTest {
    c = bf16[6] constant({nan, 7, nan, -1, inf, -inf})
    ROOT is-finite = pred[6] is-finite(c)
  })";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal actual_literal,
      HloEvaluator().Evaluate(*m_->entry_computation(), {}));
  EXPECT_THAT(actual_literal.data<bool>(),
              ::testing::ElementsAre(false, true, false, true, false, false));
}

// Check that evaluating `f32[<huge>, 0] iota` doesn't oom (it's an empty
// array!).
TEST_F(HloEvaluatorTest, ZeroSizedIotaWithHugeDimension) {
  const absl::string_view hlo_text = R"(
  HloModule test
  ENTRY t {
    ROOT i = f32[1000000000000, 0] iota(), iota_dimension=0
  })";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal actual_literal,
      HloEvaluator().Evaluate(*m_->entry_computation(), {}));
  EXPECT_THAT(actual_literal.data<float>(), ::testing::IsEmpty());
}

TEST_F(HloEvaluatorTest, CopyStartCopyDone) {
  const absl::string_view hlo_text = R"(
  HloModule test
  ENTRY CopyStartCopyDone {
    init = f32[] constant(42.0)
    copy-start = (f32[]{:S(1)}, f32[], u32[]) copy-start(init)
    ROOT copy-done = f32[] copy-done(copy-start)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal expected = LiteralUtil::CreateR0<float>(42.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result, HloEvaluator().Evaluate(*m_->entry_computation(), {}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, CopyDifferentTypes) {
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(/*hlo_text=*/R"(
  HloModule test

  ENTRY CopyDifferentTypes {
    c = bf16[3] constant({1, 2, 3})
    ROOT copy = f32[3] copy(bf16[3] c)
  }
  )"));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result, HloEvaluator().Evaluate(*m_->entry_computation(), {}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<float>({1.f, 2.f, 3.f}), result));
}

TEST_F(HloEvaluatorTest, AsyncOps) {
  const absl::string_view hlo_text = R"(
  HloModule test
  ENTRY AsyncOps {
    init = f32[] constant(42.0)
    async-start = ((f32[]), f32[], u32[]) negate-start(init)
    async-update = ((f32[]), f32[], u32[]) negate-update(async-start)
    ROOT async-done = f32[] negate-done(async-update)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal expected = LiteralUtil::CreateR0<float>(-42.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result, HloEvaluator().Evaluate(*m_->entry_computation(), {}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, MapBF16) {
  const absl::string_view hlo_text = R"(
  HloModule test

  map_computation {
    p = bf16[] parameter(0)
    add = bf16[] add(p, p)
    ROOT conv = f32[] convert(add)
  }

  ENTRY CopyStartCopyDone {
    c = bf16[3] constant({1, 2, 3})
    ROOT map = f32[3] map(c), to_apply=map_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal expected = LiteralUtil::CreateR1<float>({2.f, 4.f, 6.f});
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result, HloEvaluator().Evaluate(*m_->entry_computation(), {}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, MapS16) {
  const absl::string_view hlo_text = R"(
  HloModule test

  map_computation {
    p = s16[] parameter(0)
    add = s16[] add(p, p)
    ROOT conv = f32[] convert(add)
  }

  ENTRY CopyStartCopyDone {
    c = s16[3] constant({1, 2, 3})
    ROOT map = f32[3] map(c), to_apply=map_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal expected = LiteralUtil::CreateR1<float>({2.f, 4.f, 6.f});
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result, HloEvaluator().Evaluate(*m_->entry_computation(), {}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, MapU16) {
  const absl::string_view hlo_text = R"(
  HloModule test

  map_computation {
    p = u16[] parameter(0)
    add = u16[] add(p, p)
    ROOT conv = f32[] convert(add)
  }

  ENTRY CopyStartCopyDone {
    c = u16[3] constant({1, 2, 3})
    ROOT map = f32[3] map(c), to_apply=map_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal expected = LiteralUtil::CreateR1<float>({2.f, 4.f, 6.f});
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result, HloEvaluator().Evaluate(*m_->entry_computation(), {}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, MapMixed) {
  const absl::string_view hlo_text = R"(
  HloModule test

  map_computation {
    p0 = u16[] parameter(0)
    p1 = f32[] parameter(1)
    c0 = f32[] convert(p0)
    ROOT add = f32[] add(c0, p1)
  }

  ENTRY CopyStartCopyDone {
    c0 = u16[3] constant({1, 2, 3})
    c1 = f32[3] constant({1.5, 2.5, 3.5})
    ROOT map = f32[3] map(c0, c1), to_apply=map_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal expected = LiteralUtil::CreateR1<float>({2.5f, 4.5f, 6.5f});
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result, HloEvaluator().Evaluate(*m_->entry_computation(), {}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, DotUpcast) {
  const absl::string_view hlo_text = R"(
  HloModule test
  ENTRY DotUpcast {
    l = s16[4,3]{1,0} parameter(0)
    r = s8[3,2]{1,0} parameter(1)
    ROOT result = s32[4,2] dot(l, r), lhs_contracting_dims={1},
                                      rhs_contracting_dims={0}
  }
  )";
  // lhs:
  // s16[4,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  //  { 9, 10, 11 },
  //  { 13, 14, 15 },
  // }
  auto lhs_array = std::make_unique<Array2D<int16_t>>(4, 3);
  lhs_array->FillUnique(1);
  auto lhs_literal = LiteralUtil::CreateR2FromArray2D<int16_t>(*lhs_array);

  // rhs:
  // s8[3,2] {
  //  { 1, 2 },
  //  { 3, 4 },
  //  { 5, 6 },
  // }
  auto rhs_array = std::make_unique<Array2D<int8_t>>(3, 2);
  rhs_array->FillUnique(1);
  auto rhs_literal = LiteralUtil::CreateR2FromArray2D<int8_t>(*rhs_array);
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&lhs_literal, &rhs_literal}));

  auto expected_array =
      Array2D<int32_t>({{22, 28}, {58, 76}, {94, 124}, {130, 172}});
  auto expected = LiteralUtil::CreateR2FromArray2D<int32_t>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, SortC64) {
  const absl::string_view hlo_text = R"(
  HloModule m

  sort_lt_comparator {
    parameter.0 = c64[] parameter(0)
    real.0 = f32[] real(parameter.0)
    parameter.1 = c64[] parameter(1)
    real.1 = f32[] real(parameter.1)
    ROOT compare = pred[] compare(real.0, real.1), direction=LT
  }

  ENTRY main {
    c = c64[3] constant({(2, 0), (4, 0), (6, 0)})
    ROOT sort = c64[3]{0} sort(c), dimensions={0}, to_apply=sort_lt_comparator
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal expected =
      LiteralUtil::CreateR1<std::complex<float>>({2.f, 4.f, 6.f});
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result, HloEvaluator().Evaluate(*m_->entry_computation(), {}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, ConvertC128ToC64) {
  const absl::string_view hlo_text = R"(
  HloModule m

  ENTRY main {
    c = c128[3] constant({(2, 0), (4, 0), (6, 0)})
    ROOT sort = c64[3]{0} convert(c)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal expected =
      LiteralUtil::CreateR1<std::complex<float>>({2.f, 4.f, 6.f});
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result, HloEvaluator().Evaluate(*m_->entry_computation(), {}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Tests that HloEvaluator can evaluate an instruction even when its operands
// are not constant.
TEST_F(HloEvaluatorTest, RecursivelyEvaluateNonConstantOperands) {
  Literal c0_literal = LiteralUtil::CreateR2<float>({{0.f, 2.f}, {2.f, 4.f}});
  Literal c1_literal = LiteralUtil::CreateR2<float>({{0.f, 5.f}, {0.f, 4.f}});
  Literal c2_literal = LiteralUtil::CreateR2<float>({{2.f, 4.f}, {4.f, 4.f}});

  Shape shape = c0_literal.shape();
  HloComputation::Builder b(TestName());
  HloInstruction* c0 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(c0_literal)));
  HloInstruction* c1 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(c1_literal)));
  HloInstruction* c2 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(c2_literal)));

  HloInstruction* add0 = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, c0, c1));
  HloInstruction* add1 = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, c1, c2));
  HloInstruction* add2 = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, add0, add1));

  m_->AddEntryComputation(b.Build());
  Literal expected = LiteralUtil::CreateR2<float>({{2, 16}, {6, 16}});
  TestRecursivelyEvaluateInstruction(add2, expected);
}

// Tests that HloEvaluator can evaluate a GetTupleElement even when its operand
// Tuple instruction cannot be fully evaluated. Note that this requires that the
//  tuple element at the given tuple index can be evaluated.
TEST_F(HloEvaluatorTest, GetTupleElementOnPartiallyKnownTupleSucceeds) {
  Literal c0_literal = LiteralUtil::CreateR2<float>({{0.f, 2.f}, {2.f, 4.f}});

  Shape shape = c0_literal.shape();
  HloComputation::Builder b(TestName());
  HloInstruction* c0 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(c0_literal)));
  HloInstruction* p0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param.0"));
  HloInstruction* p1 =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "param.1"));

  HloInstruction* tuple =
      b.AddInstruction(HloInstruction::CreateTuple({p0, p1, c0}));
  HloInstruction* gte =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(tuple, 2));

  m_->AddEntryComputation(b.Build());
  Literal expected = LiteralUtil::CreateR2<float>({{0.f, 2.f}, {2.f, 4.f}});
  TestRecursivelyEvaluateInstruction(gte, expected);
}

// Tests that Infeed cannot be evaluated.
TEST_F(HloEvaluatorTest, InfeedFailure) {
  HloComputation::Builder b(TestName());
  HloInstruction* token = b.AddInstruction(HloInstruction::CreateToken());
  HloInstruction* infeed = b.AddInstruction(HloInstruction::CreateInfeed(
      ShapeUtil::MakeShape(F32, {4, 4}), token, ""));

  m_->AddEntryComputation(b.Build());
  TestRecursiveEvaluationFailure(infeed);
}

// Tests that GetTupleElement cannot be evaluated if the corresponding tuple
// element cannot be evaluated.
TEST_F(HloEvaluatorTest, GetUnknownTupleElementFails) {
  Literal c0_literal = LiteralUtil::CreateR2<float>({{0.f, 2.f}, {2.f, 4.f}});

  Shape shape = c0_literal.shape();
  HloComputation::Builder b(TestName());
  HloInstruction* c0 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(c0_literal)));
  HloInstruction* p0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param.0"));
  HloInstruction* p1 =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "param.1"));

  HloInstruction* tuple =
      b.AddInstruction(HloInstruction::CreateTuple({p0, p1, c0}));
  HloInstruction* gte =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(tuple, 0));

  m_->AddEntryComputation(b.Build());
  TestRecursiveEvaluationFailure(gte);
}

// Tests that partial evaluation works for nested tuples.
TEST_F(HloEvaluatorTest, GetTupleElementFromNestedTupleSucceeds) {
  Literal c0_literal = LiteralUtil::CreateR2<float>({{0.f, 2.f}, {2.f, 4.f}});

  Shape shape = c0_literal.shape();
  HloComputation::Builder b(TestName());
  HloInstruction* c0 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(c0_literal)));
  HloInstruction* p0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param.0"));
  HloInstruction* p1 =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "param.1"));

  HloInstruction* tuple0 =
      b.AddInstruction(HloInstruction::CreateTuple({p0, c0}));
  HloInstruction* tuple1 =
      b.AddInstruction(HloInstruction::CreateTuple({tuple0, p1}));
  HloInstruction* gte0 =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(tuple1, 0));
  HloInstruction* gte1 =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(gte0, 1));

  m_->AddEntryComputation(b.Build());
  Literal expected = LiteralUtil::CreateR2<float>({{0.f, 2.f}, {2.f, 4.f}});
  TestRecursivelyEvaluateInstruction(gte1, expected);
}

// Tests that partial evaluation works when the GetTupleElement is interleaved
// with other Tuple instructions.
TEST_F(HloEvaluatorTest, GetTupleElementInterleavedWithTupleSucceeds) {
  Literal c0_literal = LiteralUtil::CreateR2<float>({{0.f, 2.f}, {2.f, 4.f}});

  Shape shape = c0_literal.shape();
  HloComputation::Builder b(TestName());
  HloInstruction* c0 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(c0_literal)));
  HloInstruction* p0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param.0"));
  HloInstruction* p1 =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "param.1"));
  HloInstruction* p2 =
      b.AddInstruction(HloInstruction::CreateParameter(2, shape, "param.2"));

  HloInstruction* tuple0 =
      b.AddInstruction(HloInstruction::CreateTuple({p0, c0}));
  HloInstruction* tuple1 =
      b.AddInstruction(HloInstruction::CreateTuple({tuple0, p1}));
  HloInstruction* gte0 =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(tuple1, 0));
  HloInstruction* tuple2 =
      b.AddInstruction(HloInstruction::CreateTuple({gte0, p2}));
  HloInstruction* gte1 =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(tuple2, 0));
  HloInstruction* gte2 =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(gte1, 1));

  m_->AddEntryComputation(b.Build());
  Literal expected = LiteralUtil::CreateR2<float>({{0.f, 2.f}, {2.f, 4.f}});
  TestRecursivelyEvaluateInstruction(gte2, expected);
}

// Tests that we can evaluate a parameter instruction through the call graph.
TEST_F(HloEvaluatorTest, ParameterThroughCallSucceeds) {
  constexpr absl::string_view kHloModule = R"(
    HloModule parameter_through_call

    %identity {
      ROOT %param = s32[] parameter(0)
    }

    ENTRY parameter_through_call {
      %constant = s32[] constant(42)
      ROOT %call = s32[] call(s32[] %constant), to_apply=%identity
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  const HloInstruction* parameter_instruction = nullptr;
  for (const auto* computation : hlo_module->computations()) {
    for (const auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kParameter) {
        parameter_instruction = instruction;
      }
    }
  }
  ASSERT_NE(parameter_instruction, nullptr);

  Literal expected = LiteralUtil::CreateR0<int32_t>(42);
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator_.Evaluate(parameter_instruction, /*precomputed_analyses=*/{},
                          /*recursively_evaluate_nonconstant_operands=*/true));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// As above, but with analyses precomputed.
TEST_F(HloEvaluatorTest, ParameterThroughCallSucceedsWithPrecomputation) {
  constexpr absl::string_view kHloModule = R"(
    HloModule parameter_through_call

    %identity {
      ROOT %param = s32[] parameter(0)
    }

    ENTRY parameter_through_call {
      %constant = s32[] constant(42)
      ROOT %call = s32[] call(s32[] %constant), to_apply=%identity
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  const HloInstruction* parameter_instruction = nullptr;
  for (const auto* computation : hlo_module->computations()) {
    for (const auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kParameter) {
        parameter_instruction = instruction;
      }
    }
  }
  ASSERT_NE(parameter_instruction, nullptr);

  Literal expected = LiteralUtil::CreateR0<int32_t>(42);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TuplePointsToAnalysis> tuple_points_to,
      TuplePointsToAnalysis::Run(hlo_module.get()));
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(hlo_module.get());
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator_.Evaluate(parameter_instruction,
                          {tuple_points_to.get(), call_graph.get()},
                          /*recursively_evaluate_nonconstant_operands=*/true));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

class PatternMatchParseWhileLoopTest : public HloHardwareIndependentTestBase {};

TEST_F(PatternMatchParseWhileLoopTest, LoopBoundDefinedInsideOfCond) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %while_condition {
      %param = (s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %loop_bound = s32[] constant(5)
      ROOT result = pred[] compare(%gte.0, %loop_bound), direction=LT
    }

    %while_body {
      %param = (s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = f32[1024, 1024] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %gte.1, f32[1024, 1024] %gte.2)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.1 = f32[1024, 1024] parameter(0)
      %constant.0 = s32[] constant(0)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      %while = (s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
      ROOT %result = f32[1024, 1024] get-tuple-element((s32[], f32[1024, 1024], f32[1024, 1024]) %while), index=2
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  HloInstruction* while_op =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  std::optional<ParsedWhileLoop> parsed_while_loop =
      PatternMatchParseWhileLoop(while_op);
  ASSERT_TRUE(parsed_while_loop.has_value());
  EXPECT_FALSE(parsed_while_loop->is_dynamic());
  EXPECT_EQ(parsed_while_loop->static_while_loop->trip_count, 5);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_index, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_init_value, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->step_size, 1);
  EXPECT_EQ(parsed_while_loop->static_while_loop->loop_bound, 5);
}

TEST_F(PatternMatchParseWhileLoopTest,
       LoopBoundDefinedInsideOfCondWithPrecomputation) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %while_condition {
      %param = (s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %loop_bound = s32[] constant(5)
      ROOT result = pred[] compare(%gte.0, %loop_bound), direction=LT
    }

    %while_body {
      %param = (s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = f32[1024, 1024] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %gte.1, f32[1024, 1024] %gte.2)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.1 = f32[1024, 1024] parameter(0)
      %constant.0 = s32[] constant(0)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      %while = (s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
      ROOT %result = f32[1024, 1024] get-tuple-element((s32[], f32[1024, 1024], f32[1024, 1024]) %while), index=2
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TuplePointsToAnalysis> tuple_points_to,
      TuplePointsToAnalysis::Run(hlo_module.get()));
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(hlo_module.get());

  HloInstruction* while_op =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  std::optional<ParsedWhileLoop> parsed_while_loop = PatternMatchParseWhileLoop(
      while_op, {tuple_points_to.get(), call_graph.get()});
  ASSERT_TRUE(parsed_while_loop.has_value());
  EXPECT_FALSE(parsed_while_loop->is_dynamic());
  EXPECT_EQ(parsed_while_loop->static_while_loop->trip_count, 5);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_index, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_init_value, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->step_size, 1);
  EXPECT_EQ(parsed_while_loop->static_while_loop->loop_bound, 5);
}

TEST_F(PatternMatchParseWhileLoopTest, LoopBoundDefinedOutsideOfCond) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %gte.2, f32[1024, 1024] %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.1 = f32[1024, 1024] parameter(0)
      %constant.0 = s32[] constant(0)
      %constant.1 = s32[] constant(10)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %constant.1, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      %while = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
      ROOT %result = f32[1024, 1024] get-tuple-element((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %while), index=3
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  HloInstruction* while_op =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  std::optional<ParsedWhileLoop> parsed_while_loop =
      PatternMatchParseWhileLoop(while_op);
  ASSERT_TRUE(parsed_while_loop.has_value());
  EXPECT_FALSE(parsed_while_loop->is_dynamic());
  EXPECT_EQ(parsed_while_loop->static_while_loop->trip_count, 10);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_index, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_init_value, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->step_size, 1);
  EXPECT_EQ(parsed_while_loop->static_while_loop->loop_bound, 10);
}

TEST_F(PatternMatchParseWhileLoopTest, LoopBoundComputedOutsideOfCond) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %gte.2, f32[1024, 1024] %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.1 = f32[1024, 1024] parameter(0)
      %constant.0 = s32[] constant(0)
      %constant.1 = s32[] constant(10)
      %constant.2 = s32[] constant(4)
      %loop_bound = s32[] multiply(s32[] %constant.1, s32[] %constant.2)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %loop_bound, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      %while = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
      ROOT %result = f32[1024, 1024] get-tuple-element((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %while), index=3
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  HloInstruction* while_op =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  std::optional<ParsedWhileLoop> parsed_while_loop =
      PatternMatchParseWhileLoop(while_op);
  ASSERT_TRUE(parsed_while_loop.has_value());
  EXPECT_FALSE(parsed_while_loop->is_dynamic());
  EXPECT_EQ(parsed_while_loop->static_while_loop->trip_count, 40);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_index, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_init_value, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->step_size, 1);
  EXPECT_EQ(parsed_while_loop->static_while_loop->loop_bound, 40);
}

TEST_F(PatternMatchParseWhileLoopTest, StepSizeNotOne) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %gte.2, f32[1024, 1024] %gte.3)
      %constant = s32[] constant(4)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.1 = f32[1024, 1024] parameter(0)
      %constant.0 = s32[] constant(0)
      %constant.1 = s32[] constant(10)
      %constant.2 = s32[] constant(4)
      %loop_bound = s32[] multiply(s32[] %constant.1, s32[] %constant.2)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %loop_bound, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      %while = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
      ROOT %result = f32[1024, 1024] get-tuple-element((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %while), index=3
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  HloInstruction* while_op =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  std::optional<ParsedWhileLoop> parsed_while_loop =
      PatternMatchParseWhileLoop(while_op);
  ASSERT_TRUE(parsed_while_loop.has_value());
  EXPECT_FALSE(parsed_while_loop->is_dynamic());
  EXPECT_EQ(parsed_while_loop->static_while_loop->trip_count, 10);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_index, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_init_value, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->step_size, 4);
  EXPECT_EQ(parsed_while_loop->static_while_loop->loop_bound, 40);
}

// The loop condition comparison is computed by a call to another computation.
TEST_F(PatternMatchParseWhileLoopTest, RecursiveCond) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %compute_pred {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %compare = pred[] compare(gte.0, %gte.1), direction=LT
      ROOT %tuple = (pred[]) tuple(pred[] %compare)
    }

    %while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %call = (pred[]) call((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %param), to_apply=%compute_pred
      ROOT %gte.4 = pred[] get-tuple-element((pred[]) %call), index=0
    }

    %while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %gte.2, f32[1024, 1024] %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.1 = f32[1024, 1024] parameter(0)
      %constant.0 = s32[] constant(0)
      %loop_bound = s32[] constant(10)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %loop_bound, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      %while = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
      ROOT %result = f32[1024, 1024] get-tuple-element((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %while), index=3
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  HloInstruction* while_op =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  std::optional<ParsedWhileLoop> parsed_while_loop =
      PatternMatchParseWhileLoop(while_op);
  ASSERT_TRUE(parsed_while_loop.has_value());
  EXPECT_FALSE(parsed_while_loop->is_dynamic());
  EXPECT_EQ(parsed_while_loop->static_while_loop->trip_count, 10);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_index, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_init_value, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->step_size, 1);
  EXPECT_EQ(parsed_while_loop->static_while_loop->loop_bound, 10);
}

// The loop condition comparison is computed by a call to another computation.
// The called computation could be calling another computation and could use
// get-tuple-element to extract the result.
TEST_F(PatternMatchParseWhileLoopTest, RecursiveCondGetTupleElement) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %compute_pred {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %compare = pred[] compare(gte.0, %gte.1), direction=LT
      ROOT %tuple = (pred[]) tuple(pred[] %compare)
    }

    %get_tuple_element {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %call = (pred[]) call((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %param), to_apply=%compute_pred
      %gte.4 = pred[] get-tuple-element((pred[]) %call), index=0
      ROOT %tuple.1 = (pred[]) tuple(pred[] %gte.4)
    }
    %while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %call = (pred[]) call((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %param), to_apply=%get_tuple_element
      ROOT %gte.4 = pred[] get-tuple-element((pred[]) %call), index=0
    }

    %while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %gte.2, f32[1024, 1024] %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.1 = f32[1024, 1024] parameter(0)
      %constant.0 = s32[] constant(0)
      %loop_bound = s32[] constant(10)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %loop_bound, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      %while = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
      ROOT %result = f32[1024, 1024] get-tuple-element((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %while), index=3
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  HloInstruction* while_op =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  std::optional<ParsedWhileLoop> parsed_while_loop =
      PatternMatchParseWhileLoop(while_op);
  ASSERT_TRUE(parsed_while_loop.has_value());
  EXPECT_FALSE(parsed_while_loop->is_dynamic());
  EXPECT_EQ(parsed_while_loop->static_while_loop->trip_count, 10);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_index, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_init_value, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->step_size, 1);
  EXPECT_EQ(parsed_while_loop->static_while_loop->loop_bound, 10);
}

TEST_F(PatternMatchParseWhileLoopTest, LoopBoundDependsOnAnotherLoop) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %compute_pred.0 {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %compare = pred[] compare(gte.0, %gte.1), direction=LT
      ROOT %tuple = (pred[]) tuple(pred[] %compare)
    }

    %while_condition.0 {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %call = (pred[]) call((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %param), to_apply=%compute_pred.0
      ROOT %gte.4 = pred[] get-tuple-element((pred[]) %call), index=0
    }

    %while_body.0 {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %gte.2, f32[1024, 1024] %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    %compute_pred.1 {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %compare = pred[] compare(gte.0, %gte.1), direction=LT
      ROOT %tuple = (pred[]) tuple(pred[] %compare)
    }

    %while_condition.1 {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %call = (pred[]) call((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %param), to_apply=%compute_pred.1
      ROOT %gte.4 = pred[] get-tuple-element((pred[]) %call), index=0
    }

    %while_body.1 {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %gte.2, f32[1024, 1024] %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.1 = f32[1024, 1024] parameter(0)
      %param.2 = f32[1024, 1024] parameter(1)
      %constant.0 = s32[] constant(0)
      %loop_bound = s32[] constant(10)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init.0 = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %loop_bound, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      %while.0 = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init.0), condition=%while_condition.0, body=%while_body.0
      %result.0 = f32[1024, 1024] get-tuple-element((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %while.0), index=3
      %new_loop_bound = s32[] get-tuple-element((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %while.0), index=0
      %while_init.1 = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %new_loop_bound, f32[1024, 1024] %param.2, f32[1024, 1024] %result.0)
      %while.1 = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init.1), condition=%while_condition.1, body=%while_body.1
      ROOT %result.1 = f32[1024, 1024] get-tuple-element((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %while.1), index=3
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  HloInstruction* while_op =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  std::optional<ParsedWhileLoop> parsed_while_loop =
      PatternMatchParseWhileLoop(while_op);
  ASSERT_TRUE(parsed_while_loop.has_value());
  EXPECT_FALSE(parsed_while_loop->is_dynamic());
  EXPECT_EQ(parsed_while_loop->static_while_loop->trip_count, 10);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_index, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_init_value, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->step_size, 1);
  EXPECT_EQ(parsed_while_loop->static_while_loop->loop_bound, 10);
}

TEST_F(PatternMatchParseWhileLoopTest, DynamicLoop) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %gte.2, f32[1024, 1024] %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.1 = f32[1024, 1024] parameter(0)
      %param.2 = s32[] parameter(1)
      %loop_bound = s32[] constant(10)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %param.2, s32[] %loop_bound, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      %while = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
      ROOT %result = f32[1024, 1024] get-tuple-element((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %while), index=3
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  HloInstruction* while_op =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  std::optional<ParsedWhileLoop> parsed_while_loop =
      PatternMatchParseWhileLoop(while_op);
  ASSERT_TRUE(parsed_while_loop.has_value());
  EXPECT_TRUE(parsed_while_loop->is_dynamic());
}

// The loop condition comparison is computed by a call to another computation.
TEST_F(PatternMatchParseWhileLoopTest, BooleanCond) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce
    %while_condition {
      %param = (pred[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
       ROOT %gte.0 = pred[] get-tuple-element(%param), index=0
    }

    %while_body {
      %param = (pred[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = pred[] get-tuple-element(%param), index=0
      %gte.1 = f32[1024, 1024] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %gte.1, f32[1024, 1024] %gte.2)
      %new_loop_cond = pred[] constant(false)
      ROOT %loop_result = (pred[], f32[1024, 1024], f32[1024, 1024]) tuple(%new_loop_cond, %gte.1, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.1 = f32[1024, 1024] parameter(0)
      %constant.0 = pred[] constant(true)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (pred[], f32[1024, 1024], f32[1024, 1024]) tuple(pred[] %constant.0, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      %while = (pred[], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
      ROOT %result = f32[1024, 1024] get-tuple-element((pred[], f32[1024, 1024], f32[1024, 1024]) %while), index=2
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  HloInstruction* while_op =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  std::optional<ParsedWhileLoop> parsed_while_loop =
      PatternMatchParseWhileLoop(while_op);
  ASSERT_TRUE(parsed_while_loop.has_value());
  EXPECT_FALSE(parsed_while_loop->is_dynamic());
  EXPECT_EQ(parsed_while_loop->static_while_loop->trip_count, 1);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_index, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_init_value, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->step_size, 1);
  EXPECT_EQ(parsed_while_loop->static_while_loop->loop_bound, 1);
}

TEST_F(PatternMatchParseWhileLoopTest, NestedLoop) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %nested_while_condition {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %nested_while_body {
      %param = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %gte.2, f32[1024, 1024] %gte.3)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %accumulation)
    }

    %while_condition {
      %param = (s32[], s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      ROOT result = pred[] compare(%gte.0, %gte.1), direction=LT
    }

    %while_body {
      %param = (s32[], s32[], s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = s32[] get-tuple-element(%param), index=1
      %gte.2 = s32[] get-tuple-element(%param), index=2
      %gte.3 = f32[1024, 1024] get-tuple-element(%param), index=3
      %gte.4 = f32[1024, 1024] get-tuple-element(%param), index=4
      %constant.4 = s32[] constant(0)
      %nested_while_init = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.4, s32[] %gte.2, f32[1024, 1024] %gte.3, f32[1024, 1024] %gte.4)
      %nested_while = (s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%nested_while_init), condition=%nested_while_condition, body=%nested_while_body
      %nested_while_result = f32[1024, 1024] get-tuple-element((s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %nested_while), index=3
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %gte.2, %gte.3, %nested_while_result)
    }

    ENTRY accumulated_all_reduce {
      %param.1 = f32[1024, 1024] parameter(0)
      %param.2 = s32[] parameter(1)
      %constant.0 = s32[] constant(0)
      %constant.2 = s32[] constant(4)
      %loop_bound = s32[] multiply(s32[] %param.2, s32[] %constant.2)
      %constant.3 = s32[] constant(5)
      %nested_loop_bound = s32[] multiply(s32[] %constant.3, s32[] %constant.2)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], s32[], s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, s32[] %loop_bound, s32[] %nested_loop_bound, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      %while = (s32[], s32[], s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
      ROOT %result = f32[1024, 1024] get-tuple-element((s32[], s32[], s32[], f32[1024, 1024], f32[1024, 1024]) %while), index=4
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  HloInstruction* while_op =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);
  HloComputation* while_body = while_op->while_body();
  HloInstruction* nested_while =
      while_body->root_instruction()->mutable_operand(4)->mutable_operand(0);
  CHECK_EQ(nested_while->opcode(), HloOpcode::kWhile);
  std::optional<ParsedWhileLoop> parsed_while_loop =
      PatternMatchParseWhileLoop(nested_while);
  ASSERT_TRUE(parsed_while_loop.has_value());
  EXPECT_FALSE(parsed_while_loop->is_dynamic());
  EXPECT_EQ(parsed_while_loop->static_while_loop->trip_count, 20);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_index, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_init_value, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->step_size, 1);
  EXPECT_EQ(parsed_while_loop->static_while_loop->loop_bound, 20);
}

TEST_F(PatternMatchParseWhileLoopTest, CopiedLoopCond) {
  constexpr absl::string_view kHloModule = R"(
    HloModule accumulated_all_reduce

    %while_condition {
      %param = (s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %copy.0 = s32[] copy(s32[] %gte.0)
      %loop_bound = s32[] constant(5)
      %result = pred[] compare(%gte.0, %loop_bound), direction=LT
      ROOT %copy.1 = pred[] copy(pred[] %result)
    }

    %while_body {
      %param = (s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
      %gte.0 = s32[] get-tuple-element(%param), index=0
      %gte.1 = f32[1024, 1024] get-tuple-element(%param), index=1
      %gte.2 = f32[1024, 1024] get-tuple-element(%param), index=2
      %accumulation = f32[1024, 1024] add(f32[1024, 1024] %gte.1, f32[1024, 1024] %gte.2)
      %constant = s32[] constant(1)
      %increment_iteration = s32[] add(s32[] %gte.0, s32[] %constant)
      ROOT %loop_result = (s32[], f32[1024, 1024], f32[1024, 1024]) tuple(%increment_iteration, %gte.1, %accumulation)
    }

    ENTRY accumulated_all_reduce {
      %param.1 = f32[1024, 1024] parameter(0)
      %constant.0 = s32[] constant(0)
      %accumulation_buffer_init = f32[] constant(0)
      %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}
      %while_init = (s32[], f32[1024, 1024], f32[1024, 1024]) tuple(s32[] %constant.0, f32[1024, 1024] %param.1, f32[1024, 1024] %accumulation_buffer)
      %while = (s32[], f32[1024, 1024], f32[1024, 1024]) while(%while_init), condition=%while_condition, body=%while_body
      ROOT %result = f32[1024, 1024] get-tuple-element((s32[], f32[1024, 1024], f32[1024, 1024]) %while), index=2
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  HloInstruction* while_op =
      hlo_module->entry_computation()->root_instruction()->mutable_operand(0);
  std::optional<ParsedWhileLoop> parsed_while_loop =
      PatternMatchParseWhileLoop(while_op);
  ASSERT_TRUE(parsed_while_loop.has_value());
  EXPECT_FALSE(parsed_while_loop->is_dynamic());
  EXPECT_EQ(parsed_while_loop->static_while_loop->trip_count, 5);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_index, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->induction_var_init_value, 0);
  EXPECT_EQ(parsed_while_loop->static_while_loop->step_size, 1);
  EXPECT_EQ(parsed_while_loop->static_while_loop->loop_bound, 5);
}

TEST_F(HloEvaluatorTest, DotTraced) {
  const absl::string_view hlo_text = R"(
  HloModule test
  ENTRY DotUpcast {
    l = s16[4,3]{1,0} parameter(0)
    r = s8[3,2]{1,0} parameter(1)
    ROOT result = s32[4,2] dot(l, r), lhs_contracting_dims={1},
                                      rhs_contracting_dims={0}
  }
  )";
  // lhs:
  // s16[4,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  //  { 9, 10, 11 },
  //  { 13, 14, 15 },
  // }
  auto lhs_array = std::make_unique<Array2D<int16_t>>(4, 3);
  lhs_array->FillUnique(1);
  auto lhs_literal = LiteralUtil::CreateR2FromArray2D<int16_t>(*lhs_array);

  // rhs:
  // s8[3,2] {
  //  { 1, 2 },
  //  { 3, 4 },
  //  { 5, 6 },
  // }
  auto rhs_array = std::make_unique<Array2D<int8_t>>(3, 2);
  rhs_array->FillUnique(1);
  auto rhs_literal = LiteralUtil::CreateR2FromArray2D<int8_t>(*rhs_array);
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  absl::flat_hash_set<std::array<int64_t, 3>> macs_traced;
  auto mac_handler = [&macs_traced](int64_t result_index, int64_t lhs_index,
                                    int64_t rhs_index) -> void {
    macs_traced.insert(
        std::array<int64_t, 3>{result_index, lhs_index, rhs_index});
  };
  evaluator_.set_trace_mac_handler(mac_handler);
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&lhs_literal, &rhs_literal}));

  auto expected_array =
      Array2D<int32_t>({{22, 28}, {58, 76}, {94, 124}, {130, 172}});
  auto expected = LiteralUtil::CreateR2FromArray2D<int32_t>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));

  const absl::flat_hash_set<std::array<int64_t, 3>> macs_expected = {
      {1, 0, 1},  {0, 0, 0},  {2, 4, 2}, {5, 6, 1}, {2, 5, 4}, {4, 7, 2},
      {2, 3, 0},  {5, 7, 3},  {5, 8, 5}, {4, 6, 0}, {6, 9, 0}, {7, 10, 3},
      {7, 11, 5}, {1, 1, 3},  {0, 2, 4}, {3, 4, 3}, {1, 2, 5}, {7, 9, 1},
      {6, 10, 2}, {6, 11, 4}, {3, 5, 5}, {4, 8, 4}, {0, 1, 2}, {3, 3, 1}};

  EXPECT_EQ(macs_traced, macs_expected);
}

TEST_F(HloEvaluatorTest, SimpleConvTraced) {
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

  absl::flat_hash_set<std::array<int64_t, 3>> macs_traced;
  auto mac_handler = [&macs_traced](int64_t result_index, int64_t lhs_index,
                                    int64_t rhs_index) -> void {
    macs_traced.insert(
        std::array<int64_t, 3>{result_index, lhs_index, rhs_index});
  };
  evaluator_.set_trace_mac_handler(mac_handler);

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

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

  const absl::flat_hash_set<std::array<int64_t, 3>> macs_expected = {
      {10, 14, 2}, {7, 7, 0},   {11, 15, 2}, {4, 4, 0},   {3, 7, 2},
      {5, 9, 2},   {8, 9, 1},   {12, 12, 0}, {6, 10, 2},  {5, 6, 1},
      {13, 14, 1}, {15, 15, 0}, {11, 11, 0}, {0, 5, 3},   {10, 10, 0},
      {2, 7, 3},   {13, 13, 0}, {1, 6, 3},   {0, 0, 0},   {4, 9, 3},
      {8, 12, 2},  {8, 13, 3},  {9, 9, 0},   {6, 7, 1},   {9, 13, 2},
      {2, 6, 2},   {0, 1, 1},   {6, 6, 0},   {5, 10, 3},  {10, 15, 3},
      {14, 14, 0}, {7, 11, 2},  {0, 4, 2},   {10, 11, 1}, {6, 11, 3},
      {2, 2, 0},   {3, 3, 0},   {9, 14, 3},  {12, 13, 1}, {1, 5, 2},
      {5, 5, 0},   {14, 15, 1}, {1, 1, 0},   {2, 3, 1},   {4, 5, 1},
      {4, 8, 2},   {9, 10, 1},  {8, 8, 0},   {1, 2, 1},
  };

  EXPECT_EQ(macs_traced, macs_expected);
}

TEST(EvalErrorTest, OK) {
  EXPECT_EQ(std::nullopt, internal::ParseEvalErrorDetail(absl::OkStatus()));
}

TEST(EvalErrorTest, NoPayload) {
  EXPECT_EQ(std::nullopt,
            internal::ParseEvalErrorDetail(absl::InternalError("hmm")));
}

TEST(EvalErrorTest, Payload) {
  absl::Status s = absl::InternalError("hmm");
  std::string payload;
  payload.resize(sizeof(internal::EvalErrorDetail));
  absl::little_endian::Store32(
      const_cast<char*>(payload.data()),
      static_cast<uint32_t>(
          internal::EvalErrorDetail::kDynamicValueDependence));
  s.SetPayload(internal::kEvalErrorDetailUrl, absl::Cord(payload));

  EXPECT_EQ(internal::ParseEvalErrorDetail(s),
            internal::EvalErrorDetail::kDynamicValueDependence);
}

//===----------------------------------------------------------------------===//
// Perfrormance benchmarks below.
//===----------------------------------------------------------------------===//

// Reducing many numbers should be fast because it doesn't create
// intermediate Literals; the microbenchmark should finish in < 1 msec.
void BM_ReducePrecisely(::testing::benchmark::State& state) {
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

  // Benchmark loop
  for (auto s : state) {
    HloEvaluator hlo_eval;
    hlo_eval.Evaluate(reduce_instruction).value();
  }
}

BENCHMARK(BM_ReducePrecisely);

static void BM_UnaryOp(benchmark::State& state) {
  int64_t d = state.range(0);

  std::unique_ptr<HloInstruction> input =
      HloInstruction::CreateConstant(LiteralUtil::CreateFull({d, d}, 1.0f));

  std::unique_ptr<HloInstruction> unary = HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {d, d}), HloOpcode::kExp, input.get());

  HloEvaluator evaluator;
  for (auto s : state) {
    CHECK_OK(evaluator.Evaluate(unary.get()).status());
  }
}

BENCHMARK(BM_UnaryOp)
    ->MeasureProcessCPUTime()
    ->Arg(64)
    ->Arg(128)
    ->Arg(512)
    ->Arg(1024);

static void BM_BinaryOp(benchmark::State& state) {
  int64_t d = state.range(0);

  std::unique_ptr<HloInstruction> lhs =
      HloInstruction::CreateConstant(LiteralUtil::CreateFull({d, d}, 1.0f));
  std::unique_ptr<HloInstruction> rhs =
      HloInstruction::CreateConstant(LiteralUtil::CreateFull({d, d}, 2.0f));

  std::unique_ptr<HloInstruction> binary = HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {d, d}), HloOpcode::kAdd, lhs.get(), rhs.get());

  HloEvaluator evaluator;
  for (auto s : state) {
    CHECK_OK(evaluator.Evaluate(binary.get()).status());
  }
}

BENCHMARK(BM_BinaryOp)
    ->MeasureProcessCPUTime()
    ->Arg(64)
    ->Arg(128)
    ->Arg(512)
    ->Arg(1024);

}  // namespace
}  // namespace xla

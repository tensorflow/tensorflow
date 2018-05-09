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

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_element_type_converter.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
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

class HloEvaluatorTest : public ::testing::WithParamInterface<bool>,
                         public HloVerifiedTestBase {
 protected:
  HloEvaluatorTest() : use_bfloat16_(GetParam()) {
    evaluator_ = MakeUnique<HloEvaluator>();
  }

  std::unique_ptr<Literal> Evaluate(
      tensorflow::gtl::ArraySlice<const Literal*> arg_literals = {}) {
    if (use_bfloat16_) {
      // In BF16 mode, we convert all F32 type to BF16 and evaluate the module.
      auto type_converter = HloElementTypeConverter(F32, BF16);
      type_converter.Run(&module()).ValueOrDie();
    }
    return evaluator_->Evaluate(*module().entry_computation(), arg_literals)
        .ConsumeValueOrDie();
  }

  std::unique_ptr<HloEvaluator> evaluator_;

  void TestUnaryOp(HloOpcode opcode, std::unique_ptr<Literal> expected,
                   std::unique_ptr<Literal> input, float aabs = 0) {
    HloComputation::Builder b(TestName());
    auto c1 =
        b.AddInstruction(HloInstruction::CreateConstant(std::move(input)));
    b.AddInstruction(
        HloInstruction::CreateUnary(expected->shape(), opcode, c1));
    module().AddEntryComputation(b.Build());

    std::unique_ptr<Literal> result = Evaluate();

    auto element_type = expected->shape().element_type();
    if (element_type == F32 || element_type == F64) {
      ErrorSpec error(aabs);
      LiteralTestUtil::ExpectNear(*expected, *result, error);
    } else {
      LiteralTestUtil::ExpectEqual(*expected, *result);
    }
  }

  void TestBinaryOp(HloOpcode opcode, std::unique_ptr<Literal> expected,
                    std::unique_ptr<Literal> lhs,
                    std::unique_ptr<Literal> rhs) {
    HloComputation::Builder b(TestName());
    auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs)));
    auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs)));
    b.AddInstruction(
        HloInstruction::CreateBinary(expected->shape(), opcode, c1, c2));
    module().AddEntryComputation(b.Build());

    std::unique_ptr<Literal> result = Evaluate();

    LiteralTestUtil::ExpectEqual(*expected, *result);
  }

  bool use_bfloat16_;
};

#define XLA_TYPED_TEST_P(test_case_name, test_name, test_type1) \
  TEST_P(test_case_name, test_name)

// Verifies that HloEvaluator evaluates a HLO instruction that performs clamp
// with 3 operands.
TEST_P(HloEvaluatorTest, DoesClamp) {
  auto low = Literal::CreateR2<float>({{0.f, 2.f}, {2.f, 4.f}});
  auto value = Literal::CreateR2<float>({{0.f, 5.f}, {0.f, 4.f}});
  auto high = Literal::CreateR2<float>({{2.f, 4.f}, {4.f, 4.f}});

  Shape shape = low->shape();
  HloComputation::Builder b(TestName());
  auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(low)));
  auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(value)));
  auto c3 = b.AddInstruction(HloInstruction::CreateConstant(std::move(high)));
  b.AddInstruction(
      HloInstruction::CreateTernary(shape, HloOpcode::kClamp, c1, c2, c3));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected = Literal::CreateR2<float>({{0, 4}, {2, 4}});

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, DISABLED_DoesClampSpecialBroadcast) {
  auto low = Literal::CreateR0<float>(0.f);
  auto value = Literal::CreateR2<float>({{-1.f, 0.f}, {1.f, 2.f}});
  auto high = Literal::CreateR0<float>(1.f);

  Shape shape = value->shape();
  HloComputation::Builder b(TestName());
  auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(low)));
  auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(value)));
  auto c3 = b.AddInstruction(HloInstruction::CreateConstant(std::move(high)));
  b.AddInstruction(
      HloInstruction::CreateTernary(shape, HloOpcode::kClamp, c1, c2, c3));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected = Literal::CreateR2<float>({{0, 0}, {1, 1}});

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs select
// with 3 operands.
TEST_P(HloEvaluatorTest, DoesSelect) {
  auto pred = Literal::CreateR2<bool>({{true, false}, {false, true}});
  auto on_true = Literal::CreateR2<float>({{2.f, 4.f}, {4.f, 4.f}});
  auto on_false = Literal::CreateR2<float>({{0.f, 5.f}, {0.f, 4.f}});

  Shape shape = on_true->shape();
  HloComputation::Builder b(TestName());
  auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(pred)));
  auto c2 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(on_true)));
  auto c3 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(on_false)));
  b.AddInstruction(
      HloInstruction::CreateTernary(shape, HloOpcode::kSelect, c1, c2, c3));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate({});

  auto expected = Literal::CreateR2<float>({{2, 5}, {0, 4}});

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise addition with 2 operands.
TEST_P(HloEvaluatorTest, DoesAdd) {
  auto lhs = Literal::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = Literal::CreateR2<int64>({{2, 4}, {4, 4}});
  auto expected = Literal::CreateR2<int64>({{3, 4}, {-96, 8}});
  TestBinaryOp(HloOpcode::kAdd, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise and with 2 operands.
TEST_P(HloEvaluatorTest, DoesAnd) {
  auto lhs = Literal::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = Literal::CreateR2<int64>({{2, 4}, {4, 4}});
  auto expected = Literal::CreateR2<int64>({{0, 0}, {4, 4}});
  TestBinaryOp(HloOpcode::kAnd, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise or with 2 operands.
TEST_P(HloEvaluatorTest, DoesOr) {
  auto lhs = Literal::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = Literal::CreateR2<int64>({{2, 4}, {4, 4}});
  auto expected = Literal::CreateR2<int64>({{3, 4}, {-100, 4}});
  TestBinaryOp(HloOpcode::kOr, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise multiply with 2 operands.
TEST_P(HloEvaluatorTest, DoesMultiply) {
  auto lhs = Literal::CreateR2<int32>({{-1, 0}, {-100, 4}});
  auto rhs = Literal::CreateR2<int32>(
      {{std::numeric_limits<int32>::min(), 4}, {4, 4}});
  auto expected = Literal::CreateR2<int32>(
      {{std::numeric_limits<int32>::min(), 0}, {-400, 16}});
  TestBinaryOp(HloOpcode::kMultiply, std::move(expected), std::move(lhs),
               std::move(rhs));
}
// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise divide with 2 operands.
TEST_P(HloEvaluatorTest, DoesDivideInt64) {
  auto lhs = Literal::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = Literal::CreateR2<int64>({{2, 4}, {4, 4}});
  auto expected = Literal::CreateR2<int64>({{0, 0}, {-25, 1}});
  TestBinaryOp(HloOpcode::kDivide, std::move(expected), std::move(lhs),
               std::move(rhs));
}
TEST_P(HloEvaluatorTest, DoesDivideDouble) {
  auto lhs = Literal::CreateR2<double>({{1.0, 0.0}, {-100.0, 4.0}});
  auto rhs = Literal::CreateR2<double>({{2.2, 4.0}, {4.0, 4.0}});
  auto expected =
      Literal::CreateR2<double>({{0.45454545454545453, 0}, {-25, 1}});
  TestBinaryOp(HloOpcode::kDivide, std::move(expected), std::move(lhs),
               std::move(rhs));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise abs op with 1 operand.
TEST_P(HloEvaluatorTest, DoesAbsR2) {
  auto operand = Literal::CreateR2<int64>({{1, -20}, {-100, 4}});
  auto expected = Literal::CreateR2<int64>({{1, 20}, {100, 4}});
  TestUnaryOp(HloOpcode::kAbs, std::move(expected), std::move(operand));
}
TEST_P(HloEvaluatorTest, DoesAbsR0) {
  auto operand = Literal::CreateR0<float>(-1.0f);
  auto expected = Literal::CreateR0<float>(1.0f);
  TestUnaryOp(HloOpcode::kAbs, std::move(expected), std::move(operand));
}
TEST_P(HloEvaluatorTest, DoesAbsR1WithZeroSize) {
  auto operand = Literal::CreateR1<float>({});
  auto expected = Literal::CreateR1<float>({});
  TestUnaryOp(HloOpcode::kAbs, std::move(expected), std::move(operand));
}
TEST_P(HloEvaluatorTest, DoesNegateR2) {
  auto operand = Literal::CreateR2<int32>(
      {{0, std::numeric_limits<int32>::min()}, {-1, 4}});
  auto expected =
      Literal::CreateR2<int32>({{0, std::numeric_limits<int>::min()}, {1, -4}});
  TestUnaryOp(HloOpcode::kNegate, std::move(expected), std::move(operand));
}
TEST_P(HloEvaluatorTest, DoesCosR2) {
  auto operand = Literal::CreateR2<float>({{0, M_PI}, {-M_PI, 2 * M_PI}});
  auto expected = Literal::CreateR2<float>({{1, -1}, {-1, 1}});
  TestUnaryOp(HloOpcode::kCos, std::move(expected), std::move(operand),
              use_bfloat16_ ? 0x1.0P-5 : 0x1.0P-20);
}
TEST_P(HloEvaluatorTest, DoesSinR2) {
  auto operand = Literal::CreateR2<float>({{0, M_PI}, {-M_PI, 2 * M_PI}});
  auto expected = Literal::CreateR2<float>({{0, 0}, {0, 0}});
  TestUnaryOp(HloOpcode::kSin, std::move(expected), std::move(operand),
              use_bfloat16_ ? 0x1.0P-5 : 0x1.0P-20);
}
TEST_P(HloEvaluatorTest, DoesNotR2) {
  auto operand =
      Literal::CreateR2<int32>({{0, std::numeric_limits<int>::min()},
                                {-1, std::numeric_limits<int>::max()}});
  auto expected =
      Literal::CreateR2<int32>({{-1, std::numeric_limits<int>::max()},
                                {0, std::numeric_limits<int>::min()}});
  TestUnaryOp(HloOpcode::kNot, std::move(expected), std::move(operand));
}
// Verifies that HloEvaluator evaluates a HLO Computation with non-parameter nor
// constant operands.
TEST_P(HloEvaluatorTest, DoesTraverseInstructions) {
  auto lhs = Literal::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = Literal::CreateR2<int64>({{2, 4}, {4, 4}});
  auto rhs2 = Literal::CreateR2<int64>({{1, -20}, {-100, 4}});
  std::vector<const Literal*> args = {lhs.get(), rhs.get(), rhs2.get()};

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
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate(args);

  auto expected = Literal::CreateR2<int64>({{4, -16}, {-196, 12}});

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

// Verifies Reshape operation is correctly evaluated.
TEST_P(HloEvaluatorTest, DoesReshape) {
  HloComputation::Builder b(TestName());
  const int64 dimensions[] = {11, 8, 7, 5, 9};
  TF_ASSERT_OK_AND_ASSIGN(auto literal,
                          LiteralTestUtil::CreateRandomLiteral<F32>(
                              ShapeUtil::MakeShape(F32, dimensions), 0.0, 1.0));
  auto literal_clone = literal->CloneToUnique();
  HloInstruction* literal_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {8, 7, 11, 9, 5});
  const int64 permutation[] = {1, 2, 0, 4, 3};
  b.AddInstruction(
      HloInstruction::CreateTranspose(shape, literal_instruction, permutation));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate({});

  using NativeT = typename primitive_util::PrimitiveTypeToNative<F32>::type;
  result->EachCell<NativeT>(
      [&](tensorflow::gtl::ArraySlice<int64> indices, NativeT value) {
        std::vector<int64> rindexes = Permute(permutation, indices);
        EXPECT_NEAR(value, literal_clone->Get<NativeT>(rindexes), 0x1.0P-5);
      });
}

// Verifies Broadcast operation is correctly evaluated.
TEST_P(HloEvaluatorTest, DoesBroadcast) {
  HloComputation::Builder b(TestName());
  auto input_literal = Literal::CreateR2<int32>({{1, 2}, {3, 4}, {5, 6}});
  auto output_literal = Literal::CreateR3<int32>(
      {{{1, 2}, {3, 4}, {5, 6}}, {{1, 2}, {3, 4}, {5, 6}}});
  HloInstruction* literal_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  b.AddInstruction(HloInstruction::CreateBroadcast(
      output_literal->shape(), literal_instruction, {1, 2}));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate({});

  LiteralTestUtil::ExpectEqual(*result, *output_literal);
}

TEST_P(HloEvaluatorTest, DoesBroadcastScalar) {
  HloComputation::Builder b(TestName());
  auto input_literal = Literal::CreateR0<int32>(111);
  auto output_literal = Literal::CreateR2<int32>(
      {{111, 111}, {111, 111}, {111, 111}, {111, 111}, {111, 111}, {111, 111}});

  HloInstruction* literal_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  // Broadcast dimension should be empty in the case of scalars.
  b.AddInstruction(HloInstruction::CreateBroadcast(
      output_literal->shape(), literal_instruction,
      /*broadcast_dimensions=*/{}));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate({});

  LiteralTestUtil::ExpectEqual(*result, *output_literal);
}

TEST_P(HloEvaluatorTest, DoesConcatenateSimple) {
  HloComputation::Builder b(TestName());

  HloInstruction* operand1 = b.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR2<int64>({{-1, -2}, {100, 200}})));
  HloInstruction* operand2 = b.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR2<int64>({{-2, -3}, {-100, -200}})));

  std::vector<HloInstruction*> operands = {operand1, operand2};

  Shape shape = ShapeUtil::MakeShape(S64, {4, 2});
  b.AddInstruction(HloInstruction::CreateConcatenate(shape, operands, 0));

  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected =
      Literal::CreateR2<int64>({{-1, -2}, {100, 200}, {-2, -3}, {-100, -200}});
  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, ConcatenateHandlesShapeWithZeroElement) {
  HloComputation::Builder b(TestName());

  HloInstruction* operand1 = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<int64>({100, 200})));
  HloInstruction* operand2 = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<int64>({})));

  std::vector<HloInstruction*> operands = {operand1, operand2};

  Shape shape = ShapeUtil::MakeShape(S64, {2});
  b.AddInstruction(HloInstruction::CreateConcatenate(shape, operands, 0));

  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected = Literal::CreateR1<int64>({100, 200});
  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, ConvertWithSameLayout) {
  HloComputation::Builder b(TestName());

  auto input_literal = Literal::CreateR2<int32>({{1, 2}, {3, 4}, {5, 6}});
  auto expected =
      Literal::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  ASSERT_TRUE(LayoutUtil::LayoutsInShapesEqual(input_literal->shape(),
                                               expected->shape()));

  HloInstruction* constant = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  b.AddInstruction(HloInstruction::CreateConvert(expected->shape(), constant));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  LiteralTestUtil::ExpectEqual(*result, *expected);
}

TEST_P(HloEvaluatorTest, ConvertWithDifferentLayout) {
  HloComputation::Builder b(TestName());

  auto input_literal = Literal::CreateR2WithLayout<int32>(
      {{1, 2}, {3, 4}, {5, 6}}, LayoutUtil::MakeLayout({0, 1}));
  auto expected = Literal::CreateR2WithLayout<float>(
      {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}, LayoutUtil::MakeLayout({1, 0}));
  ASSERT_FALSE(LayoutUtil::LayoutsInShapesEqual(input_literal->shape(),
                                                expected->shape()));

  HloInstruction* constant = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  b.AddInstruction(HloInstruction::CreateConvert(expected->shape(), constant));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  LiteralTestUtil::ExpectEqual(*result, *expected);
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

TEST_P(HloEvaluatorTest, Pad2DIntegerArrayWithZeroDimension) {
  auto operand = Literal::CreateR2<int32>({{}, {}});
  HloComputation::Builder b(TestName());
  auto operand_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(operand)));

  constexpr int32 kPadValue = 10;
  auto pad_value = Literal::CreateR0<int32>(kPadValue);
  auto padding_value_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(pad_value)));

  auto padding_config = CreatePaddingConfig({{{1, 0, 2}}, {{0, 2, 1}}});
  Shape shape = ShapeUtil::MakeShape(S32, {5, 2});
  b.AddInstruction(HloInstruction::CreatePad(
      shape, operand_instruction, padding_value_instruction, padding_config));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected = Literal::CreateR2<int32>(
      {{10, 10}, {10, 10}, {10, 10}, {10, 10}, {10, 10}});

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, Pad4DFloatArrayWithInteriorPadding) {
  HloComputation::Builder b(TestName());

  Array4D<float> input_array(3, 2, 1, 1, {1, 2, 3, 4, 5, 6});
  auto input = Literal::CreateR4FromArray4D<float>(input_array);
  HloInstruction* input_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(input)));
  constexpr float kPadValue = 1.5;
  auto pad_value = Literal::CreateR0<float>(kPadValue);
  HloInstruction* pad_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(pad_value)));

  Shape shape = ShapeUtil::MakeShape(F32, {8, 5, 1, 1});
  auto r4_padding_on_dim0_dim1 =
      CreatePaddingConfig({{{1, 0, 2}}, {{0, 2, 1}}, {{0, 0, 0}}, {{0, 0, 0}}});
  b.AddInstruction(HloInstruction::CreatePad(
      shape, input_instruction, pad_instruction, r4_padding_on_dim0_dim1));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected_array = MakeUnique<Array4D<float>>(8, 5, 1, 1);
  expected_array->Fill(kPadValue);
  (*expected_array)(1, 0, 0, 0) = 1.0f;
  (*expected_array)(1, 2, 0, 0) = 2.0f;
  (*expected_array)(4, 0, 0, 0) = 3.0f;
  (*expected_array)(4, 2, 0, 0) = 4.0f;
  (*expected_array)(7, 0, 0, 0) = 5.0f;
  (*expected_array)(7, 2, 0, 0) = 6.0f;

  auto expected = Literal::CreateR4FromArray4D<float>(*expected_array);

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, NegativePadding2D) {
  HloComputation::Builder b(TestName());

  // input_array:
  // f32[4,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  //  { 9, 10, 11 },
  //  { 13, 14, 15 },
  // }
  auto input_array = MakeUnique<Array2D<float>>(4, 3);
  input_array->FillUnique(1.0f);
  auto input = Literal::CreateR2FromArray2D<float>(*input_array);
  HloInstruction* input_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(input)));

  auto pad_value_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.718f)));

  auto r2_padding_on_dim0_dim1 =
      CreatePaddingConfig({{{-1, -2, 0}}, {{-2, 4, 0}}});
  Shape shape = ShapeUtil::MakeShape(F32, {1, 5});
  b.AddInstruction(HloInstruction::CreatePad(shape, input_instruction,
                                             pad_value_instruction,
                                             r2_padding_on_dim0_dim1));

  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  // f32[1,5] { 7.0, 2.718, 2.718, 2.718, 2.718 }
  auto expected_array = MakeUnique<Array2D<float>>(1, 5);
  (*expected_array)(0, 0) = 7.0f;
  (*expected_array)(0, 1) = 2.718f;
  (*expected_array)(0, 2) = 2.718f;
  (*expected_array)(0, 3) = 2.718f;
  (*expected_array)(0, 4) = 2.718f;
  auto expected = Literal::CreateR2FromArray2D<float>(*expected_array);

  LiteralTestUtil::ExpectNear(*expected, *result, ErrorSpec(0x1.0P-5));
}

TEST_P(HloEvaluatorTest, NegativeAndInteriorPadding2D) {
  HloComputation::Builder b(TestName());

  // f32[4,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  //  { 9, 10, 11 },
  //  { 13, 14, 15 },
  // }
  auto input_array = MakeUnique<Array2D<float>>(4, 3);
  input_array->FillUnique(1.0f);
  auto input = Literal::CreateR2FromArray2D<float>(*input_array);
  HloInstruction* input_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(input)));

  auto pad_value_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(2.718f)));

  PaddingConfig padding_config = MakeNoPaddingConfig(2);

  // Negative padding that results in zero dimensions.
  auto r2_padding_on_dim0_dim1 =
      CreatePaddingConfig({{{-2, -5, 1}}, {{-2, 4, 2}}});

  Shape shape = ShapeUtil::MakeShape(F32, {0, 9});
  b.AddInstruction(HloInstruction::CreatePad(shape, input_instruction,
                                             pad_value_instruction,
                                             r2_padding_on_dim0_dim1));

  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected_array = MakeUnique<Array2D<float>>(0, 9);
  auto expected = Literal::CreateR2FromArray2D<float>(*expected_array);

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, DotRank2AndRank1) {
  HloComputation::Builder b(TestName());

  // lhs:
  // f32[4,1] {
  //  { 1 },
  //  { 2 },
  //  { 3 },
  //  { 4 },
  // }
  auto lhs_array = MakeUnique<Array2D<float>>(4, 1);
  lhs_array->FillUnique(1.0f);
  auto lhs_literal = Literal::CreateR2FromArray2D<float>(*lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  // rhs:
  // f32[2] { 1, 2 },
  auto rhs_literal = Literal::CreateR2<float>({{1, 2}});
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {4, 2});
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  b.AddInstruction(HloInstruction::CreateDot(shape, lhs_instruction,
                                             rhs_instruction, dot_dnums));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  // clang-format off
  auto expected_array = Array2D<float>({
      {1.f, 2.f},
      {2.f, 4.f},
      {3.f, 6.f},
      {4.f, 8.f},
  });
  // clang-format on
  auto expected = Literal::CreateR2FromArray2D<float>(expected_array);

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, DotRank1AndRank2) {
  HloComputation::Builder b(TestName());

  // lhs:
  // f32[3]
  //  { 1, 2, 3 },
  auto lhs_literal = Literal::CreateR1<float>({1, 2, 3});
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  // rhs:
  // f32[3,2] {
  //  { 1, 2 },
  //  { 3, 4 },
  //  { 5, 6 },
  // }
  auto rhs_array = MakeUnique<Array2D<float>>(3, 2);
  rhs_array->FillUnique(1.0f);
  auto rhs_literal = Literal::CreateR2FromArray2D<float>(*rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {2});
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  b.AddInstruction(HloInstruction::CreateDot(shape, lhs_instruction,
                                             rhs_instruction, dot_dnums));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected = Literal::CreateR1<float>({22.f, 28.f});

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, DotRank2AndRank2) {
  HloComputation::Builder b(TestName());

  // lhs:
  // f32[4,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  //  { 9, 10, 11 },
  //  { 13, 14, 15 },
  // }
  auto lhs_array = MakeUnique<Array2D<float>>(4, 3);
  lhs_array->FillUnique(1.0f);
  auto lhs_literal = Literal::CreateR2FromArray2D<float>(*lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  // rhs:
  // f32[3,2] {
  //  { 1, 2 },
  //  { 3, 4 },
  //  { 5, 6 },
  // }
  auto rhs_array = MakeUnique<Array2D<float>>(3, 2);
  rhs_array->FillUnique(1.0f);
  auto rhs_literal = Literal::CreateR2FromArray2D<float>(*rhs_array);
  HloInstruction* rhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs_literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {4, 2});
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  b.AddInstruction(HloInstruction::CreateDot(shape, lhs_instruction,
                                             rhs_instruction, dot_dnums));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected_array = Array2D<float>({
      {22.f, 28.f},
      {58.f, 76.f},
      {94.f, 124.f},
      {130.f, 172.f},
  });
  auto expected = Literal::CreateR2FromArray2D<float>(expected_array);

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, SimpleConv1D) {
  HloComputation::Builder b(TestName());

  Array3D<float> lhs_array = {{{1, 2, 3}}};
  auto lhs_literal = Literal::CreateR3FromArray3D<float>(lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  Array3D<float> rhs_array = {{{3.f, 4.f}}};
  auto rhs_literal = Literal::CreateR3FromArray3D<float>(rhs_array);
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

  const Shape& shape = ShapeUtil::MakeShape(F32, {1, 1, 3});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, window, dnums));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  Array3D<float> expected_array = {{{11.f, 18.f, 9.f}}};
  auto expected = Literal::CreateR3FromArray3D<float>(expected_array);

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, Simple4x4Conv2DWith2x2Kernel) {
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
  auto lhs_literal = Literal::CreateR4FromArray4D<float>(lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  Array4D<float> rhs_array(1, 1, 2, 2);
  // clang-format off
  rhs_array.FillWithYX(Array2D<float>({
    {5, 6},
    {7, 8},
  }));
  // clang-format on
  auto rhs_literal = Literal::CreateR4FromArray4D<float>(rhs_array);
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

  const Shape& shape = ShapeUtil::MakeShape(F32, {1, 1, 4, 4});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, window, dnums));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  Array4D<float> expected_array(1, 1, 4, 4);
  // clang-format off
  expected_array.FillWithYX(Array2D<float>({
    {100, 126, 152,  76},
    {204, 230, 256, 124},
    {308, 334, 360, 172},
    {149, 160, 171,  80},
  }));
  // clang-format on
  auto expected = Literal::CreateR4FromArray4D<float>(expected_array);

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, Conv2DGeneralDimensionsReversed) {
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

  auto lhs_literal = Literal::CreateR4FromArray4D<float>(input);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  auto rhs_literal = Literal::CreateR4FromArray4D<float>(weight);
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

  const Shape& shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 2});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, window, dnums));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  // clang-format off
  // Result dimensions: [feature=1, height=1, batch=1, width=2]
  Array4D<float> expected_array({{{{2514, 2685}}}});
  Array4D<float> expected_array_bf16({{{{2512, 2672}}}});
  // clang-format on
  auto expected = Literal::CreateR4FromArray4D<float>(
      use_bfloat16_ ? expected_array_bf16 : expected_array);

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, Conv2DGeneralDimensions) {
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

  auto lhs_literal = Literal::CreateR4FromArray4D<float>(input);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  auto rhs_literal = Literal::CreateR4FromArray4D<float>(weight);
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

  const Shape& shape = ShapeUtil::MakeShape(F32, {1, 1, 1, 2});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, window, dnums));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  // clang-format off
  // Result dimensions: [feature=1, height=1, batch=1, width=2]
  Array4D<float> expected_array({{{{2514, 2685}}}});
  Array4D<float> expected_array_bf16({{{{2512, 2672}}}});
  // clang-format on
  auto expected = Literal::CreateR4FromArray4D<float>(
      use_bfloat16_ ? expected_array_bf16 : expected_array);

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, DilatedBaseConv2DWithHighPadding) {
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
  auto lhs_literal = Literal::CreateR4FromArray4D<float>(lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  Array4D<float> rhs_array(1, 1, 2, 2);
  // clang-format off
  rhs_array.FillWithYX(Array2D<float>({
    {5, 6},
    {7, 8},
  }));
  // clang-format on
  auto rhs_literal = Literal::CreateR4FromArray4D<float>(rhs_array);
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

  const Shape& shape = ShapeUtil::MakeShape(F32, {1, 1, 7, 7});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, window, dnums));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

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
  auto expected = Literal::CreateR4FromArray4D<float>(expected_array);

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, DilatedBaseConv2DWithLowAndHighPadding) {
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
  auto lhs_literal = Literal::CreateR4FromArray4D<float>(lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  Array4D<float> rhs_array(1, 1, 2, 2);
  // clang-format off
  rhs_array.FillWithYX(Array2D<float>({
    {5, 6},
    {7, 8},
  }));
  // clang-format on
  auto rhs_literal = Literal::CreateR4FromArray4D<float>(rhs_array);
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

  const Shape& shape = ShapeUtil::MakeShape(F32, {1, 1, 8, 8});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, window, dnums));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

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
  auto expected = Literal::CreateR4FromArray4D<float>(expected_array);

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest,
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
  auto lhs_literal = Literal::CreateR4FromArray4D<float>(lhs_array);
  HloInstruction* lhs_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs_literal)));

  Array4D<float> rhs_array(1, 1, 2, 3);
  // clang-format off
  rhs_array.FillWithYX(Array2D<float>({
    {5, 6, 7},
    {8, 9, 10},
  }));
  // clang-format on
  auto rhs_literal = Literal::CreateR4FromArray4D<float>(rhs_array);
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

  const Shape& shape = ShapeUtil::MakeShape(F32, {1, 1, 9, 3});
  b.AddInstruction(HloInstruction::CreateConvolve(
      shape, lhs_instruction, rhs_instruction, window, dnums));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

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
  auto expected = Literal::CreateR4FromArray4D<float>(expected_array);

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

class HloEvaluatorPreciseReduceTest : public HloVerifiedTestBase {};

// Tests that Reduce doesn't lose precision when adding many numbers (because
// it accumulates its result in a double).
TEST_F(HloEvaluatorPreciseReduceTest, AddReductionPrecisionTest) {
  HloComputation::Builder b(TestName());

  constexpr int kNumElements = 1 << 25;  // float += 1 saturates at 1<<24
  std::vector<float> v(kNumElements, 1.0f);
  HloInstruction* arg_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<float>(v)));
  HloInstruction* init_value = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));
  auto add_func = module().AddEmbeddedComputation(add_computation.Build());

  HloInstruction* reduce_instruction = b.AddInstruction(
      HloInstruction::CreateReduce(scalar_shape, arg_instruction, init_value,
                                   /*dimensions_to_reduce=*/{0}, add_func));
  module().AddEntryComputation(b.Build());

  HloEvaluator hlo_eval;
  std::unique_ptr<Literal> result =
      hlo_eval.Evaluate(reduce_instruction).ConsumeValueOrDie();
  LiteralTestUtil::ExpectR0Equal<float>(kNumElements, *result);
}

// Reducing many numbers should be fast because it doesn't create
// intermediate Literals; the microbenchmark should finish in < 1 msec.
void BM_ReducePrecisely(int num_iters) {
  tensorflow::testing::StopTiming();
  HloComputation::Builder b("BM_ReducePrecisely");
  HloModuleConfig config;
  config.set_debug_options(legacy_flags::GetDebugOptionsFromFlags());
  HloModule module("BM_ReducePrecisely", VersionedComputationHandle(), config);

  constexpr int kNumElements = 1 << 25;  // float += 1 saturates at 1<<24
  std::vector<float> v(kNumElements, 1.0f);
  HloInstruction* arg_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<float>(v)));
  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.f)));

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

TEST_P(HloEvaluatorTest, ReduceAdd) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto arg_array = MakeUnique<Array2D<float>>(2, 3);
  arg_array->FillUnique(1.0f);
  auto arg_literal = Literal::CreateR2FromArray2D<float>(*arg_array);

  HloInstruction* arg_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));
  auto add_func = module().AddEmbeddedComputation(add_computation.Build());

  Shape shape = ShapeUtil::MakeShape(F32, {2});
  b.AddInstruction(
      HloInstruction::CreateReduce(shape, arg_instruction, init_value,
                                   /*dimensions_to_reduce=*/{1}, add_func));

  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected = Literal::CreateR1<float>({6, 18});

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, ReduceWindowMax) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto arg_array = MakeUnique<Array2D<float>>(2, 3);
  arg_array->FillUnique(1.0f);
  auto arg_literal = Literal::CreateR2FromArray2D<float>(*arg_array);

  HloInstruction* arg_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.f)));

  HloComputation::Builder max_computation("max");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = max_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = max_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  max_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kMaximum, param_lhs, param_rhs));
  auto max_func = module().AddEmbeddedComputation(max_computation.Build());

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

  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected = Literal::CreateR2<float>({{6, 7}});
  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, ReduceWindowAdd) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto arg_array = MakeUnique<Array2D<float>>(2, 3);
  arg_array->FillUnique(1.0f);
  auto arg_literal = Literal::CreateR2FromArray2D<float>(*arg_array);

  HloInstruction* arg_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));
  auto add_func = module().AddEmbeddedComputation(add_computation.Build());

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

  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected = Literal::CreateR2<float>({{1, 3, 5}, {5, 11, 13}});
  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, ReduceWindowAdd6D) {
  HloComputation::Builder b(TestName());

  // arg: f32[4,4,4,4,4,4] full of ones. Using small dims to limit run-time.
  std::vector<int64> input_dims(6, 4);
  std::unique_ptr<Literal> arg_literal =
      Literal::CreateFullWithDescendingLayout<float>(input_dims, 1.0f);

  HloInstruction* arg_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(arg_literal)));

  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(0.f)));

  HloComputation::Builder add_computation("add");
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto param_lhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
  auto param_rhs = add_computation.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
  add_computation.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kAdd, param_lhs, param_rhs));
  auto add_func = module().AddEmbeddedComputation(add_computation.Build());

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

  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  std::vector<int64> output_dims = {4, 3, 3, 3, 4, 4};
  std::unique_ptr<Literal> result_literal =
      Literal::CreateFullWithDescendingLayout<float>(output_dims, 8.0f);
  LiteralTestUtil::ExpectEqual(*result_literal, *result);
}

TEST_P(HloEvaluatorTest, StridedSlice) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[3,5] {
  //  { 1, 2, 3, 4, 5 },
  //  { 9, 10, 11, 12, 13 },
  //  { 17, 18, 19, 20, 21 },
  // }
  auto operand_array = MakeUnique<Array2D<float>>(3, 5);
  operand_array->FillUnique(1.0f);
  auto operand_literal = Literal::CreateR2FromArray2D<float>(*operand_array);

  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  Shape shape = ShapeUtil::MakeShape(F32, {2, 1});
  b.AddInstruction(HloInstruction::CreateSlice(shape, operand,
                                               /*start_indices=*/{0, 2},
                                               /*limit_indices=*/{3, 5},
                                               /*strides=*/{2, 3}));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected = Literal::CreateR2<float>({
      {3},
      {19},
  });

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, DynamicSlice) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,4] {
  //  { 1, 2, 3, 4 },
  //  { 5, 6, 7, 8 },
  // }
  auto operand_array = MakeUnique<Array2D<float>>(2, 4);
  operand_array->FillUnique(1.0f);
  auto operand_literal = Literal::CreateR2FromArray2D<float>(*operand_array);

  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  auto start_indices = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<int32>({0, 1})));

  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  b.AddInstruction(HloInstruction::CreateDynamicSlice(shape, operand,
                                                      start_indices, {2, 3}));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected = Literal::CreateR2<float>({
      {2, 3, 4},
      {6, 7, 8},
  });

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

// Verifies that the HloEvaluator's implementation goes along with existing
// backends' behavior, although this is not required by the spec.
TEST_P(HloEvaluatorTest, DynamicSliceModSlice) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,4] {
  //  { 1, 2, 3, 4 },
  //  { 5, 6, 7, 8 },
  // }
  auto operand_array = MakeUnique<Array2D<float>>(2, 4);
  operand_array->FillUnique(1.0f);
  auto operand_literal = Literal::CreateR2FromArray2D<float>(*operand_array);

  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  auto start_indices = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<int32>({2, 1})));

  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  b.AddInstruction(HloInstruction::CreateDynamicSlice(shape, operand,
                                                      start_indices, {2, 3}));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected = Literal::CreateR2<float>({
      {2, 3, 4},
      {6, 7, 8},
  });

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, DynamicSliceUpdate) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto operand_array = MakeUnique<Array2D<double>>(2, 3);
  operand_array->FillUnique(1.0);
  auto operand_literal = Literal::CreateR2FromArray2D<double>(*operand_array);

  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  auto start_indices = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<int64>({0, 1})));

  auto update = b.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR2<double>({{-2.0, -3.0}, {-6.0, -7.0}})));

  Shape shape = ShapeUtil::MakeShape(F64, {2, 3});
  b.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      shape, operand, update, start_indices));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected = Literal::CreateR2<double>({
      {1, -2, -3},
      {5, -6, -7},
  });

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, SetAndGetTuples) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto operand_array = MakeUnique<Array2D<double>>(2, 3);
  operand_array->FillUnique(1.0);
  auto operand_literal2 = Literal::CreateR2FromArray2D<double>(*operand_array);

  HloInstruction* operand2 = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal2)));
  HloInstruction* operand1 = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<int64>({0, 1})));

  auto tuple =
      b.AddInstruction(HloInstruction::CreateTuple({operand1, operand2}));

  Shape shape = ShapeUtil::MakeShape(F64, {2, 3});
  b.AddInstruction(HloInstruction::CreateGetTupleElement(shape, tuple, 1));

  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto expected = Literal::CreateR2<double>({
      {1, 2, 3},
      {5, 6, 7},
  });

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, SetAndGetNestedTuples) {
  HloComputation::Builder b(TestName());

  // arg:
  // f32[2,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  // }
  auto operand_array = MakeUnique<Array2D<double>>(2, 3);
  operand_array->FillUnique(1.0);

  HloInstruction* operand2 = b.AddInstruction(HloInstruction::CreateConstant(
      Literal::CreateR2FromArray2D<double>(*operand_array)));
  HloInstruction* operand1 = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<int64>({0, 1})));

  auto tuple1 =
      b.AddInstruction(HloInstruction::CreateTuple({operand1, operand2}));
  auto tuple2 =
      b.AddInstruction(HloInstruction::CreateTuple({operand2, operand2}));

  auto outer_tuple =
      b.AddInstruction(HloInstruction::CreateTuple({tuple1, tuple2}));

  b.AddInstruction(
      HloInstruction::CreateGetTupleElement(tuple2->shape(), outer_tuple, 1));

  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  auto result_inner_literal =
      Literal::CreateR2FromArray2D<double>(*operand_array);
  auto expected = Literal::MakeTuple({
      result_inner_literal.get(),
      result_inner_literal.get(),
  });

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, Reverse) {
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
  auto operand_literal = Literal::CreateR4FromArray4D<float>(input);
  HloInstruction* operand = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(operand_literal)));

  const Shape shape = ShapeUtil::MakeShape(F32, {4, 3, 2, 1});
  b.AddInstruction(HloInstruction::CreateReverse(shape, operand, {0, 1}));
  module().AddEntryComputation(b.Build());

  std::unique_ptr<Literal> result = Evaluate();

  // clang-format off
  auto expected = Literal::CreateR4FromArray4D<float>({
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

  LiteralTestUtil::ExpectEqual(*expected, *result);
}

TEST_P(HloEvaluatorTest, EvaluateWithSubstitutions) {
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
  auto result = evaluator.EvaluateWithSubstitutions(
      add, {{param0, Literal::CreateR1<float>({1, 2, 3, 4}).get()},
            {square, Literal::CreateR1<float>({10, 20, 30, 40}).get()}});
  TF_ASSERT_OK(result.status());
  LiteralTestUtil::ExpectEqual(*Literal::CreateR1<float>({11, 22, 33, 44}),
                               *result.ValueOrDie());
}

// Check that EvaluateWithSubstitutions works if one of the operands to the op
// we're evaluating is a constant.
TEST_P(HloEvaluatorTest, EvaluateWithSubstitutionsWithConstantOperand) {
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  HloInstruction* param0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param0"));
  HloInstruction* square = b.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kMultiply, param0, param0));
  HloInstruction* constant = b.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR1<float>({1, 2, 3, 4})));
  HloInstruction* add = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, constant, square));

  // Evaluate add with square = {10, 20, 30, 40}.
  HloEvaluator evaluator;
  auto result = evaluator.EvaluateWithSubstitutions(
      add, {{square, Literal::CreateR1<float>({10, 20, 30, 40}).get()}});
  TF_ASSERT_OK(result.status());
  LiteralTestUtil::ExpectEqual(*Literal::CreateR1<float>({11, 22, 33, 44}),
                               *result.ValueOrDie());
}

TEST_P(HloEvaluatorTest, EvaluateGather_TensorFlowGatherV1) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,3] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1, 3}
}
)";
  ParseAndVerifyModule(hlo_text);
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR1<int32>({0, 2});
  LiteralTestUtil::ExpectEqual(
      *Literal::CreateR2<int32>({{1, 2, 3}, {7, 8, 9}}),
      *Evaluate({operand.get(), gather_indices.get()}));
}

TEST_P(HloEvaluatorTest, EvaluateGather_TensorFlowGatherV2) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherV2

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[3,2] gather(operand, indices),
      output_window_dims={0},
      elided_window_dims={1},
      gather_dims_to_operand_dims={1},
      index_vector_dim=1,
      window_bounds={3, 1}
}
)";
  ParseAndVerifyModule(hlo_text);
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR1<int32>({0, 2});
  LiteralTestUtil::ExpectEqual(
      *Literal::CreateR2<int32>({{1, 3}, {4, 6}, {7, 9}}),
      *Evaluate({operand.get(), gather_indices.get()}));
}

TEST_P(HloEvaluatorTest, EvaluateGather_TensorFlowGatherMultipleBatchDims) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,3,2] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={1},
      gather_dims_to_operand_dims={1},
      index_vector_dim=2,
      window_bounds={3, 1}
}
)";
  ParseAndVerifyModule(hlo_text);
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR2<int32>({{0, 2}, {2, 1}});
  LiteralTestUtil::ExpectEqual(
      *Literal::CreateR3<int32>(
          {{{1, 3}, {4, 6}, {7, 9}}, {{3, 2}, {6, 5}, {9, 8}}}),
      *Evaluate({operand.get(), gather_indices.get()}));
}

TEST_P(HloEvaluatorTest, EvaluateGather_TensorFlowGatherNd) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0,1},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=1,
      window_bounds={1,1,2}
}
)";
  ParseAndVerifyModule(hlo_text);
  std::unique_ptr<Literal> operand =
      Literal::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                {{-7, 7}, {-8, 8}, {-9, 9}}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR2<int32>({{0, 0}, {1, 0}});
  LiteralTestUtil::ExpectEqual(
      *Literal::CreateR2<int32>({{-1, 1}, {-4, 4}}),
      *Evaluate({operand.get(), gather_indices.get()}));
}

TEST_P(HloEvaluatorTest,
       EvaluateGather_TensorFlowGatherNdNonDefaultIndexVectorDim) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0,1},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=0,
      window_bounds={1,1,2}
}
)";
  ParseAndVerifyModule(hlo_text);
  std::unique_ptr<Literal> operand =
      Literal::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                {{-7, 7}, {-8, 8}, {-9, 9}}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR2<int32>({{0, 0}, {1, 0}});
  LiteralTestUtil::ExpectEqual(
      *Literal::CreateR2<int32>({{-2, 2}, {-1, 1}}),
      *Evaluate({operand.get(), gather_indices.get()}));
}

TEST_P(HloEvaluatorTest, EvaluateGather_DynamicSlice) {
  const char* hlo_text = R"(
HloModule DynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[1,1] gather(operand, indices),
      output_window_dims={0,1},
      elided_window_dims={},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=0,
      window_bounds={1,1}
}
)";
  ParseAndVerifyModule(hlo_text);
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR1<int32>({1, 1});
  LiteralTestUtil::ExpectEqual(
      *Literal::CreateR2<int32>({{5}}),
      *Evaluate({operand.get(), gather_indices.get()}));
}

TEST_P(HloEvaluatorTest, EvaluateGather_BatchDynamicSlice) {
  const char* hlo_text = R"(
HloModule BatchDynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,1,1] gather(operand, indices),
      output_window_dims={1,2},
      elided_window_dims={},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=0,
      window_bounds={1,1}
}
)";
  ParseAndVerifyModule(hlo_text);
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR2<int32>({{2, 1}, {1, 1}});
  LiteralTestUtil::ExpectEqual(
      *Literal::CreateR3<int32>({{{8}}, {{5}}}),
      *Evaluate({operand.get(), gather_indices.get()}));
}

TEST_P(HloEvaluatorTest, EvaluateGather_ZeroDimBounds) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,0] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,0] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1, 0}
}
)";
  ParseAndVerifyModule(hlo_text);
  std::unique_ptr<Literal> operand = Literal::CreateR2<int32>({{}, {}, {}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR1<int32>({0, 2});
  LiteralTestUtil::ExpectEqual(
      *Literal::CreateR2<int32>({{}, {}}),
      *Evaluate({operand.get(), gather_indices.get()}));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise comparison with 2 bfloat16 operands.
TEST_P(HloEvaluatorTest, DoesCompareBF16) {
  // lhs >= rhs
  auto lhs = Literal::CreateR2<bfloat16>(
      {{bfloat16(0.25), bfloat16(0.35), bfloat16(0.125)},
       {bfloat16(-0.25), bfloat16(-0.35), bfloat16(-0.125)}});
  auto rhs = Literal::CreateR2<bfloat16>(
      {{bfloat16(0.5), bfloat16(0.125), bfloat16(0.125)},
       {bfloat16(0.25), bfloat16(-0.375), bfloat16(-0.127)}});
  auto expected =
      Literal::CreateR2<bool>({{false, true, true}, {false, true, true}});
  TestBinaryOp(HloOpcode::kGe, std::move(expected), std::move(lhs),
               std::move(rhs));
}

INSTANTIATE_TEST_CASE_P(HloEvaluatorTest_Instantiation, HloEvaluatorTest,
                        ::testing::ValuesIn(use_bf16_params));

}  // namespace
}  // namespace xla

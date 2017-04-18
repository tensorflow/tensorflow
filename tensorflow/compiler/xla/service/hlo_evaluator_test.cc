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

#include <string>
#include <utility>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

// Verifies that HloEvaluator evaluates a HLO instruction that performs clamp
// with 3 operands.
TEST(HloEvaluatorTest, DoesClamp) {
  auto low = LiteralUtil::CreateR2<float>({{0.f, 2.f}, {2.f, 4.f}});
  auto high = LiteralUtil::CreateR2<float>({{2.f, 4.f}, {4.f, 4.f}});
  auto value = LiteralUtil::CreateR2<float>({{0.f, 5.f}, {0.f, 4.f}});

  Shape shape = low->shape();
  auto c1 = HloInstruction::CreateConstant(std::move(low));
  auto c2 = HloInstruction::CreateConstant(std::move(high));
  auto c3 = HloInstruction::CreateConstant(std::move(value));
  auto instruction = HloInstruction::CreateTernary(
      shape, HloOpcode::kClamp, c1.get(), c2.get(), c3.get());

  std::unique_ptr<Literal> result =
      HloEvaluator::EvaluateOpForLiteral(instruction.get()).ConsumeValueOrDie();

  auto expected = LiteralUtil::CreateR2<float>({{0, 4}, {2, 4}});

  EXPECT_TRUE(LiteralUtil::Equal(*result, *expected));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs select
// with 3 operands.
TEST(HloEvaluatorTest, DoesSelect) {
  auto pred = LiteralUtil::CreateR2<bool>({{true, false}, {false, true}});
  auto on_true = LiteralUtil::CreateR2<float>({{2.f, 4.f}, {4.f, 4.f}});
  auto on_false = LiteralUtil::CreateR2<float>({{0.f, 5.f}, {0.f, 4.f}});

  Shape shape = on_true->shape();
  auto c1 = HloInstruction::CreateConstant(std::move(pred));
  auto c2 = HloInstruction::CreateConstant(std::move(on_true));
  auto c3 = HloInstruction::CreateConstant(std::move(on_false));
  auto instruction = HloInstruction::CreateTernary(
      shape, HloOpcode::kSelect, c1.get(), c2.get(), c3.get());

  std::unique_ptr<Literal> result =
      HloEvaluator::EvaluateOpForLiteral(instruction.get()).ConsumeValueOrDie();

  auto expected = LiteralUtil::CreateR2<float>({{2, 5}, {0, 4}});

  EXPECT_TRUE(LiteralUtil::Equal(*result, *expected));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise addition with 2 operands.
TEST(HloEvaluatorTest, DoesAdd) {
  auto lhs = LiteralUtil::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64>({{2, 4}, {4, 4}});

  Shape shape = ShapeUtil::MakeShape(S64, {2, 2});
  auto c1 = HloInstruction::CreateConstant(std::move(lhs));
  auto c2 = HloInstruction::CreateConstant(std::move(rhs));
  auto instruction =
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, c1.get(), c2.get());

  std::unique_ptr<Literal> result =
      HloEvaluator::EvaluateOpForLiteral(instruction.get()).ConsumeValueOrDie();

  auto expected = LiteralUtil::CreateR2<int64>({{3, 4}, {-96, 8}});

  EXPECT_TRUE(LiteralUtil::Equal(*result, *expected));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise divide with 2 operands.
TEST(HloEvaluatorTest, DoesDivide) {
  auto lhs_s64 = LiteralUtil::CreateR2<int64>({{1, 0}, {-100, 4}});
  auto rhs_s64 = LiteralUtil::CreateR2<int64>({{2, 4}, {4, 4}});

  Shape shape_s64 = ShapeUtil::MakeShape(S64, {2, 2});
  auto c1_s64 = HloInstruction::CreateConstant(std::move(lhs_s64));
  auto c2_s64 = HloInstruction::CreateConstant(std::move(rhs_s64));
  auto instruction = HloInstruction::CreateBinary(shape_s64, HloOpcode::kDivide,
                                                  c1_s64.get(), c2_s64.get());

  std::unique_ptr<Literal> result =
      HloEvaluator::EvaluateOpForLiteral(instruction.get()).ConsumeValueOrDie();

  auto expected = LiteralUtil::CreateR2<int64>({{0, 0}, {-25, 1}});

  EXPECT_TRUE(LiteralUtil::Equal(*result, *expected));

  auto lhs_f64 = LiteralUtil::CreateR2<double>({{1.0, 0.0}, {-100.0, 4.0}});
  auto rhs_f64 = LiteralUtil::CreateR2<double>({{2.2, 4.0}, {4.0, 4.0}});

  Shape shape_f64 = ShapeUtil::MakeShape(F64, {2, 2});
  auto c1_f64 = HloInstruction::CreateConstant(std::move(lhs_f64));
  auto c2_f64 = HloInstruction::CreateConstant(std::move(rhs_f64));
  instruction = HloInstruction::CreateBinary(shape_f64, HloOpcode::kDivide,
                                             c1_f64.get(), c2_f64.get());

  result =
      HloEvaluator::EvaluateOpForLiteral(instruction.get()).ConsumeValueOrDie();

  expected =
      LiteralUtil::CreateR2<double>({{0.45454545454545453, 0}, {-25, 1}});

  EXPECT_TRUE(LiteralUtil::Equal(*result, *expected));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise abs op with 1 operand.
TEST(HloEvaluatorTest, DoesAbs) {
  auto operand = LiteralUtil::CreateR2<int64>({{1, -20}, {-100, 4}});

  Shape shape = ShapeUtil::MakeShape(S64, {2, 2});
  auto c1 = HloInstruction::CreateConstant(std::move(operand));
  auto instruction =
      HloInstruction::CreateUnary(shape, HloOpcode::kAbs, c1.get());

  std::unique_ptr<Literal> result =
      HloEvaluator::EvaluateOpForLiteral(instruction.get()).ConsumeValueOrDie();

  auto expected = LiteralUtil::CreateR2<int64>({{1, 20}, {100, 4}});

  EXPECT_TRUE(LiteralUtil::Equal(*result, *expected));
}

}  // namespace
}  // namespace xla

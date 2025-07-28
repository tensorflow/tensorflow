/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/model/experimental/symbolic_expr.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/analysis/indexing_test_utils.h"

namespace xla {
namespace gpu {
namespace {
using ::testing::Combine;
using ::testing::Values;

// Test fixture to hold the context for all tests.
class SymbolicExprTest : public ::testing::Test {
 protected:
  SymbolicExprContext ctx_;
};

TEST_F(SymbolicExprTest, CreateAndPrint) {
  SymbolicExpr* v0 = ctx_.CreateVariable(0);
  SymbolicExpr* v1 = ctx_.CreateVariable(1);
  SymbolicExpr* c0 = ctx_.CreateConstant(0);
  SymbolicExpr* c2 = ctx_.CreateConstant(2);
  SymbolicExpr* c42 = ctx_.CreateConstant(42);
  SymbolicExpr* add = ctx_.CreateBinaryOp(SymbolicExprType::kAdd, v0, c42);
  SymbolicExpr* min = ctx_.CreateBinaryOp(SymbolicExprType::kMin, v1, c2);
  SymbolicExpr* max = ctx_.CreateBinaryOp(SymbolicExprType::kMax, min, c0);
  SymbolicExpr* mul = ctx_.CreateBinaryOp(SymbolicExprType::kMul, add, max);
  SymbolicExpr* floordiv =
      ctx_.CreateBinaryOp(SymbolicExprType::kFloorDiv, mul, c2);
  SymbolicExpr* ceildiv =
      ctx_.CreateBinaryOp(SymbolicExprType::kCeilDiv, floordiv, c2);

  ASSERT_NE(ceildiv, nullptr);
  EXPECT_THAT(ceildiv->ToString(),
              MatchIndexingString(
                  "((((v0 + 42) * max(min(v1, 2), 0)) floordiv 2) ceildiv 2)"));
}

TEST_F(SymbolicExprTest, ParseAndPrint) {
  const std::string kStringContainingAllOperators =
      "((((v0 + 42) * max(min(v1, 2), 0)) floordiv 2) ceildiv 2)";
  SymbolicExpr* parsed_expr = ctx_.Parse(kStringContainingAllOperators);
  ASSERT_NE(parsed_expr, nullptr);
  EXPECT_THAT(parsed_expr->ToString(),
              MatchIndexingString(kStringContainingAllOperators));
}

TEST_F(SymbolicExprTest, ParseAndPrint_Invalid) {
  EXPECT_DEATH(ctx_.Parse("1 + "), "Unexpected end of expression");
  EXPECT_DEATH(ctx_.Parse("max(1, )"), "Failed to parse expression");
  EXPECT_DEATH(ctx_.Parse("(1 + 2"), "Missing parenthesis");
  EXPECT_DEATH(ctx_.Parse("foo(3, 4)"), "Failed to parse expression");
}

TEST_F(SymbolicExprTest, Evaluate) {
  SymbolicExpr* v0 = ctx_.CreateVariable(0);
  SymbolicExpr* v1 = ctx_.CreateVariable(1);
  SymbolicExpr* c0 = ctx_.CreateConstant(0);
  SymbolicExpr* c2 = ctx_.CreateConstant(2);
  SymbolicExpr* c42 = ctx_.CreateConstant(42);
  SymbolicExpr* add = ctx_.CreateBinaryOp(SymbolicExprType::kAdd, v0, c42);
  SymbolicExpr* min = ctx_.CreateBinaryOp(SymbolicExprType::kMin, v1, c2);
  SymbolicExpr* max = ctx_.CreateBinaryOp(SymbolicExprType::kMax, min, c0);
  SymbolicExpr* mul = ctx_.CreateBinaryOp(SymbolicExprType::kMul, add, max);
  SymbolicExpr* floordiv =
      ctx_.CreateBinaryOp(SymbolicExprType::kFloorDiv, mul, c2);
  SymbolicExpr* ceildiv =
      ctx_.CreateBinaryOp(SymbolicExprType::kCeilDiv, floordiv, c2);

  // ((((5 + 42) * max(min(1, 2), 0)) / 2) ceildiv 2) = 23 ceildiv 2 = 12
  EXPECT_EQ(ceildiv->Evaluate({5, 1}), 12);
}

TEST_F(SymbolicExprTest, Evaluate_Invalid) {
  SymbolicExpr* v0 = ctx_.CreateVariable(0);
  SymbolicExpr* v1 = ctx_.CreateVariable(1);
  SymbolicExpr* add = ctx_.CreateBinaryOp(SymbolicExprType::kAdd, v0, v1);

  EXPECT_DEATH(add->Evaluate({5}),
               "Evaluate has not provided a value for VariableID 1.");
}

class SymbolicExprEvaluateDivModTest
    : public SymbolicExprTest,
      public ::testing::WithParamInterface<std::tuple<int64_t, int64_t>> {};

TEST_P(SymbolicExprEvaluateDivModTest, EvaluateDivMod) {
  const auto& params = GetParam();
  const int64_t numerator_val = std::get<0>(params);
  const int64_t denominator_val = std::get<1>(params);
  SymbolicExpr* numerator = ctx_.CreateConstant(numerator_val);
  SymbolicExpr* denominator = ctx_.CreateConstant(denominator_val);

  if (numerator_val % denominator_val == 0) {
    EXPECT_EQ(
        ctx_.CreateBinaryOp(SymbolicExprType::kMod, numerator, denominator)
            ->Evaluate({}),
        0);
    EXPECT_EQ(
        ctx_.CreateBinaryOp(SymbolicExprType::kFloorDiv, numerator, denominator)
            ->Evaluate({}),
        ctx_.CreateBinaryOp(SymbolicExprType::kCeilDiv, numerator, denominator)
            ->Evaluate({}));
  } else {
    EXPECT_GT(
        ctx_.CreateBinaryOp(SymbolicExprType::kMod, numerator, denominator)
            ->Evaluate({}),
        0);
    EXPECT_EQ(
        ctx_.CreateBinaryOp(SymbolicExprType::kFloorDiv, numerator, denominator)
                ->Evaluate({}) +
            1,
        ctx_.CreateBinaryOp(SymbolicExprType::kCeilDiv, numerator, denominator)
            ->Evaluate({}));
  }
}

INSTANTIATE_TEST_SUITE_P(PositiveAndNegative, SymbolicExprEvaluateDivModTest,
                         Combine(Values(5, -5, 4, -4), Values(2, -2)));

TEST_F(SymbolicExprTest, ReplaceVariables) {
  SymbolicExpr* expr_to_sub = ctx_.Parse("(v0 + v1)");
  std::vector<SymbolicExpr*> substitutions{nullptr, ctx_.Parse("(v2 * 10)")};
  SymbolicExpr* result = expr_to_sub->ReplaceVariables(substitutions, &ctx_);
  EXPECT_EQ(result->ToString(), "(v0 + (v2 * 10))");
}

}  // namespace
}  // namespace gpu
}  // namespace xla

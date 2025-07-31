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
struct SymbolicExprTest : public ::testing::Test {
 protected:
  SymbolicExprContext ctx;
  SymbolicExpr v0 = ctx.CreateVariable(0);
  SymbolicExpr v1 = ctx.CreateVariable(1);
};

TEST_F(SymbolicExprTest, CreateAndPrint) {
  SymbolicExpr expr = (((v0 + 42) * v1.min(2).max(0)) / 2).ceilDiv(2);

  ASSERT_NE(expr, nullptr);
  EXPECT_THAT(expr.ToString(),
              MatchIndexingString(
                  "((((v0 + 42) * max(min(v1, 2), 0)) floordiv 2) ceildiv 2)"));
}

TEST_F(SymbolicExprTest, ParseAndPrint) {
  const std::string kStringContainingAllOperators =
      "((((v0 + 42) * max(min(v1, 2), 0)) floordiv 2) ceildiv 2)";
  SymbolicExpr parsed_expr = ctx.Parse(kStringContainingAllOperators);
  ASSERT_NE(parsed_expr, nullptr);
  EXPECT_THAT(parsed_expr.ToString(),
              MatchIndexingString(kStringContainingAllOperators));
}

TEST_F(SymbolicExprTest, ParseAndPrint_Invalid) {
  EXPECT_DEATH(ctx.Parse("1 + "), "Unexpected end of expression");
  EXPECT_DEATH(ctx.Parse("max(1, )"), "Failed to parse expression");
  EXPECT_DEATH(ctx.Parse("(1 + 2"), "Missing parenthesis");
  EXPECT_DEATH(ctx.Parse("foo(3, 4)"), "Failed to parse expression");
}

TEST_F(SymbolicExprTest, Evaluate) {
  SymbolicExpr expr = (((v0 + 42) * v1.min(2).max(0)) / 2).ceilDiv(2);

  // ((((5 + 42) * max(min(1, 2), 0)) / 2) ceildiv 2) = 23 ceildiv 2 = 12
  EXPECT_EQ(expr.Evaluate({5, 1}), 12);
}

TEST_F(SymbolicExprTest, Evaluate_Invalid) {
  SymbolicExpr add = v0 + v1;

  EXPECT_DEATH(add.Evaluate({5}),
               "Evaluate has not provided a value for VariableID 1.");
}

class SymbolicExprEvaluateDivModTest
    : public SymbolicExprTest,
      public ::testing::WithParamInterface<std::tuple<int64_t, int64_t>> {};

TEST_P(SymbolicExprEvaluateDivModTest, EvaluateDivMod) {
  const auto& params = GetParam();
  const int64_t numerator_val = std::get<0>(params);
  const int64_t denominator_val = std::get<1>(params);
  SymbolicExpr numerator = ctx.CreateConstant(numerator_val);
  SymbolicExpr denominator = ctx.CreateConstant(denominator_val);

  if (numerator_val % denominator_val == 0) {
    EXPECT_EQ((numerator % denominator).Evaluate({}), 0);
    EXPECT_EQ((numerator / denominator).Evaluate({}),
              numerator.ceilDiv(denominator).Evaluate({}));
  } else {
    EXPECT_GT((numerator % denominator).Evaluate({}), 0);
    EXPECT_EQ((numerator / denominator).Evaluate({}) + 1,
              numerator.ceilDiv(denominator).Evaluate({}));
  }
}

INSTANTIATE_TEST_SUITE_P(PositiveAndNegative, SymbolicExprEvaluateDivModTest,
                         Combine(Values(5, -5, 4, -4), Values(2, -2)));

TEST_F(SymbolicExprTest, ReplaceVariables) {
  SymbolicExpr expr_to_sub = ctx.Parse("(v0 + v1)");
  std::vector<SymbolicExpr> substitutions{{}, ctx.Parse("(v2 * 10)")};
  SymbolicExpr result = expr_to_sub.ReplaceVariables(substitutions, &ctx);
  EXPECT_EQ(result.ToString(), "(v0 + (v2 * 10))");
}

TEST_F(SymbolicExprTest, UniquingWorks) {
  SymbolicExpr c1 = ctx.CreateConstant(42);
  SymbolicExpr c2 = ctx.CreateConstant(42);
  EXPECT_EQ(c1, c2);
  SymbolicExpr c3 = ctx.CreateConstant(99);
  EXPECT_NE(c1, c3);

  SymbolicExpr add1 = v0 + 42;
  SymbolicExpr add2 = v0 + 42;
  EXPECT_EQ(add1, add2);
  SymbolicExpr add3 = v0 + 99;
  EXPECT_NE(add1, add3);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

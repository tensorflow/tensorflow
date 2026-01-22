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

#include "xla/hlo/analysis/symbolic_expr.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_test_utils.h"

namespace xla {
namespace {
using ::testing::Combine;
using ::testing::Values;

// Test fixture to hold the context for all tests.
struct SymbolicExprTest : public ::testing::Test {
 protected:
  SymbolicExprTest() {
    RegisterSymbolicExprStorage(&ctx);
    v0 = CreateSymbolicVariable(0, &ctx);
    v1 = CreateSymbolicVariable(1, &ctx);
    c1 = CreateSymbolicConstant(1, &ctx);
    c3 = CreateSymbolicConstant(3, &ctx);
    c2 = CreateSymbolicConstant(2, &ctx);
    c5 = CreateSymbolicConstant(5, &ctx);
  }

  mlir::MLIRContext ctx;
  SymbolicExpr v0;
  SymbolicExpr v1;
  SymbolicExpr c1;
  SymbolicExpr c3;
  SymbolicExpr c2;
  SymbolicExpr c5;
};

TEST_F(SymbolicExprTest, CreateAndPrint) {
  SymbolicExpr expr = (((v0 + 42) * v1.min(2).max(0)) / 2).ceilDiv(2);

  ASSERT_NE(expr, nullptr);
  EXPECT_THAT(expr.ToString(),
              MatchIndexingString(
                  "((((v0 + 42) * max(min(v1, 2), 0)) floordiv 2) ceildiv 2)"));
}

TEST_F(SymbolicExprTest, PrintWithDifferentNumDimensions) {
  SymbolicExpr expr = v0 * 2 + v1;

  EXPECT_THAT(expr.ToString(), MatchIndexingString("((v0 * 2) + v1)"));
  // Only symbols
  EXPECT_THAT(expr.ToString(0), MatchIndexingString("((s0 * 2) + s1)"));
  // One dimension and one symbol
  EXPECT_THAT(expr.ToString(1), MatchIndexingString("((d0 * 2) + s0)"));
  // Only dimensions
  EXPECT_THAT(expr.ToString(2), MatchIndexingString("((d0 * 2) + d1)"));
}

TEST_F(SymbolicExprTest, ParseAndPrint) {
  const std::string kStringContainingAllOperators =
      "(((((v0 + 42) * max(min(v1, 2), 0)) floordiv 2) ceildiv 2) mod 5)";
  SymbolicExpr parsed_expr =
      ParseSymbolicExpr(kStringContainingAllOperators, &ctx);
  ASSERT_NE(parsed_expr, nullptr);
  EXPECT_THAT(parsed_expr.ToString(),
              MatchIndexingString(kStringContainingAllOperators));
}

TEST_F(SymbolicExprTest, ParseAndPrint_Invalid) {
  absl::ScopedMockLog log(absl::MockLogDefault::kDisallowUnexpected);
  log.StartCapturingLogs();

  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_,
                       "Unexpected end of expression at: \"\""));
  EXPECT_EQ(ParseSymbolicExpr("1 + ", &ctx), SymbolicExpr());

  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_,
                       "Failed to parse expression at: \")\""));
  EXPECT_EQ(ParseSymbolicExpr("max(1, )", &ctx), SymbolicExpr());

  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_,
                       "Missing parenthesis at: \"\""));
  EXPECT_EQ(ParseSymbolicExpr("(1 + 2", &ctx), SymbolicExpr());

  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_,
                       "Failed to parse expression at: \"foo(3, 4)\""));
  EXPECT_EQ(ParseSymbolicExpr("foo(3, 4)", &ctx), SymbolicExpr());
}

TEST_F(SymbolicExprTest, ParseWithVariableMap) {
  llvm::DenseMap<llvm::StringRef, SymbolicExpr> variable_map;
  variable_map["foo"] = v0;
  // Purposely use a variable name that starts with a 'd' to test that the
  // dim/symbol parsing is not triggered when the variable map is provided.
  variable_map["dim_bar"] = v1;

  absl::string_view expr_str = "foo + dim_bar * 2";
  SymbolicExpr expr =
      ParseSymbolicExprAndAdvance(&expr_str, &ctx, variable_map);
  EXPECT_EQ(expr, v0 + v1 * 2);
  EXPECT_TRUE(expr_str.empty());

  absl::ScopedMockLog log(absl::MockLogDefault::kDisallowUnexpected);
  log.StartCapturingLogs();
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_,
                       "Failed to parse expression at: \"baz\""));
  expr_str = "baz";
  EXPECT_EQ(ParseSymbolicExprAndAdvance(&expr_str, &ctx, variable_map),
            SymbolicExpr());
}

TEST_F(SymbolicExprTest, ParseDimsAndSymbols) {
  EXPECT_EQ(ParseSymbolicExpr("d0", &ctx), v0);
  EXPECT_EQ(ParseSymbolicExpr("s0", &ctx, /*num_dims=*/2),
            CreateSymbolicVariable(2, &ctx));
  EXPECT_EQ(ParseSymbolicExpr("s0", &ctx, /*num_dims=*/0), v0);

  absl::ScopedMockLog log(absl::MockLogDefault::kDisallowUnexpected);
  log.StartCapturingLogs();
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_,
                       "Symbol cannot be parsed because number of dimensions "
                       "is not set. at: \"0\""));
  EXPECT_EQ(ParseSymbolicExpr("s0", &ctx), SymbolicExpr());
}

TEST_F(SymbolicExprTest, ConstantFolding) {
  // Expressions are simplified at creation if possible.
  EXPECT_EQ(c2 + c3, c5);
  EXPECT_EQ(c5 - c2, c3);
  EXPECT_EQ(c2 * c3, CreateSymbolicConstant(6, &ctx));
  EXPECT_EQ(c5 / c2, c2);
  EXPECT_EQ(c5 % c2, c1);
  EXPECT_EQ(c5.floorDiv(c2), c2);
  EXPECT_EQ(c5.ceilDiv(c2), c3);
  EXPECT_EQ(c2.min(c5), c2);
  EXPECT_EQ(c5.min(c2), c2);
  EXPECT_EQ(c2.max(c5), c5);
  EXPECT_EQ(c5.max(c2), c5);
  EXPECT_EQ(((c2 + c3) * c2).ceilDiv(c5), c2);
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
  SymbolicExpr numerator = CreateSymbolicConstant(numerator_val, &ctx);
  SymbolicExpr denominator = CreateSymbolicConstant(denominator_val, &ctx);

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
  SymbolicExpr expr_to_sub = ParseSymbolicExpr("(v0 + v1)", &ctx);
  std::vector<SymbolicExpr> substitutions{{},
                                          ParseSymbolicExpr("(v2 * 10)", &ctx)};
  SymbolicExpr result = expr_to_sub.ReplaceVariables(substitutions);
  EXPECT_EQ(result.ToString(), "(v0 + (v2 * 10))");
}

TEST_F(SymbolicExprTest, ReplaceSymbols) {
  SymbolicExpr d0 = CreateSymbolicVariable(0, &ctx);
  SymbolicExpr s0 = CreateSymbolicVariable(1, &ctx);
  SymbolicExpr s1 = CreateSymbolicVariable(2, &ctx);
  SymbolicExpr c7 = CreateSymbolicConstant(7, &ctx);
  SymbolicExpr expr_to_sub = (d0 + s0 * 2) * s1;
  SymbolicExpr result = expr_to_sub.ReplaceSymbols({d0, c7}, /*num_dims=*/1);
  EXPECT_EQ(result, ((d0 + (d0 * 2)) * c7));
}

TEST_F(SymbolicExprTest, ReplaceDimsAndSymbols) {
  SymbolicExpr d0 = CreateSymbolicVariable(0, &ctx);
  SymbolicExpr s0 = CreateSymbolicVariable(1, &ctx);
  SymbolicExpr s1 = CreateSymbolicVariable(2, &ctx);
  SymbolicExpr c7 = CreateSymbolicConstant(7, &ctx);
  SymbolicExpr expr_to_sub = (d0 + s0 * 2) * s1;
  SymbolicExpr result = expr_to_sub.ReplaceDimsAndSymbols({s0}, {d0, c7});
  EXPECT_EQ(result, ((s0 + (d0 * 2)) * c7));
}

TEST_F(SymbolicExprTest, UniquingWorks) {
  SymbolicExpr c1 = CreateSymbolicConstant(42, &ctx);
  SymbolicExpr c2 = CreateSymbolicConstant(42, &ctx);
  EXPECT_EQ(c1, c2);
  SymbolicExpr c3 = CreateSymbolicConstant(99, &ctx);
  EXPECT_NE(c1, c3);

  SymbolicExpr add1 = v0 + 42;
  SymbolicExpr add2 = v0 + 42;
  EXPECT_EQ(add1, add2);
  SymbolicExpr add3 = v0 + 99;
  EXPECT_NE(add1, add3);
}

TEST_F(SymbolicExprTest, UniquingDoesNotCrashWithCombinedAffineExpr) {
  mlir::AffineExpr affine_expr = mlir::getAffineDimExpr(0, &ctx);
  SymbolicExpr c1 = CreateSymbolicConstant(42, &ctx);
  EXPECT_EQ(affine_expr, mlir::getAffineDimExpr(0, &ctx));
  EXPECT_EQ(c1, CreateSymbolicConstant(42, &ctx));
}

TEST_F(SymbolicExprTest, Replace) {
  SymbolicExpr d0 = CreateSymbolicVariable(0, &ctx);
  SymbolicExpr d1 = CreateSymbolicVariable(1, &ctx);
  SymbolicExpr c2 = CreateSymbolicConstant(2, &ctx);
  SymbolicExpr c5 = CreateSymbolicConstant(5, &ctx);

  SymbolicExpr expr = (d0 + c2) * (d1 + c2);
  EXPECT_EQ(expr.Replace(d0 + c2, c5), (c5 * (d1 + c2)));
  EXPECT_EQ(expr.Replace(d1, d0), (d0 + c2) * (d0 + c2));
  EXPECT_EQ(expr.Replace(c2, c5), (d0 + c5) * (d1 + c5));
  EXPECT_EQ(expr.Replace(expr, c2), c2);
  EXPECT_EQ(expr.Replace(d1, d1), expr);
  EXPECT_EQ(expr.Replace(CreateSymbolicConstant(42, &ctx), d1), expr);
}

TEST_F(SymbolicExprTest, ReplaceWithMap) {
  SymbolicExpr d0 = CreateSymbolicVariable(0, &ctx);
  SymbolicExpr d1 = CreateSymbolicVariable(1, &ctx);
  SymbolicExpr c2 = CreateSymbolicConstant(2, &ctx);
  SymbolicExpr c5 = CreateSymbolicConstant(5, &ctx);

  SymbolicExpr expr = (d0 + c2) * (d1 + c2);

  llvm::DenseMap<SymbolicExpr, SymbolicExpr> replace_expression;
  replace_expression[d0 + c2] = c5;
  replace_expression[d1] = d0;
  EXPECT_EQ(expr.Replace(replace_expression), c5 * (d0 + c2));

  llvm::DenseMap<SymbolicExpr, SymbolicExpr> replace_constant;
  replace_constant[c2] = d0;
  EXPECT_EQ(expr.Replace(replace_constant), (d0 + d0) * (d1 + d0));

  llvm::DenseMap<SymbolicExpr, SymbolicExpr> swap_variables;
  swap_variables[d0] = d1;
  swap_variables[d1] = d0;
  EXPECT_EQ(expr.Replace(swap_variables), (d1 + c2) * (d0 + c2));

  llvm::DenseMap<SymbolicExpr, SymbolicExpr> no_change;
  no_change[CreateSymbolicVariable(99, &ctx)] = c5;
  EXPECT_EQ(expr.Replace(no_change), expr);
}

TEST_F(SymbolicExprTest, BasicSimplificationsAtCreationTime) {
  auto c0 = CreateSymbolicConstant(0, &ctx);
  auto c1 = CreateSymbolicConstant(1, &ctx);
  auto c3 = CreateSymbolicConstant(3, &ctx);

  // x + 0 = x
  EXPECT_EQ(v0 + c0, v0);
  EXPECT_EQ(c0 + v0, v0);
  EXPECT_EQ(c2 + c1, c3);

  // TODO(b/459357586): This will be canonicalized to (v0 + 2) in the future.
  EXPECT_NE(v0 + c2, c2 + v0);

  // x * 0 = 0
  EXPECT_EQ(v0 * c0, c0);
  EXPECT_EQ(c0 * v0, c0);
  EXPECT_EQ(c2 * c0, c0);

  // x * 1 = x
  EXPECT_EQ(v0 * c1, v0);
  EXPECT_EQ(c1 * v0, v0);
  EXPECT_EQ(c2 * c1, c2);

  // Associativity: (X * C1) * C2 = X * (C1 * C2)
  EXPECT_EQ(((v0 * 2) * 3), v0 * 6);

  // No associativity if constant is on LHS of outer mul.
  // TODO(b/459357586): This will be canonicalized to (v0 * 6) in the future.
  SymbolicExpr mul_2_v0 = CreateSymbolicConstant(2, &ctx) * v0;
  SymbolicExpr mul_2_v0_3 = mul_2_v0 * 3;
  EXPECT_EQ(mul_2_v0_3.ToString(), "((2 * v0) * 3)");
}

TEST_F(SymbolicExprTest, Canonicalization_Basic) {
  SymbolicExpr constants = (c2 * 3) + 5;
  EXPECT_EQ(constants.Canonicalize().ToString(), "11");

  SymbolicExpr add_commutativity = c2 + v0;
  EXPECT_EQ(add_commutativity.Canonicalize().ToString(), "(v0 + 2)");

  SymbolicExpr neutral_element = (v0 + 0) * 1 + (v1 * 0);
  EXPECT_EQ(neutral_element.Canonicalize().ToString(), "v0");

  SymbolicExpr add_combining_constants = (c2 + v0) + 3;
  EXPECT_EQ(add_combining_constants.Canonicalize().ToString(), "(v0 + 5)");

  SymbolicExpr mul_combining_constants = (c2 * v0) * -1;
  EXPECT_EQ(mul_combining_constants.Canonicalize().ToString(), "(v0 * -2)");

  SymbolicExpr combination = (v0 * 3) + (v0 * 2);
  EXPECT_EQ(combination.Canonicalize().ToString(), "(v0 * 5)");

  SymbolicExpr subtraction = (v0 * 5) - (v0 * 2);
  EXPECT_EQ(subtraction.Canonicalize().ToString(), "(v0 * 3)");

  SymbolicExpr subtraction_with_zero = (v0 * 5) - 0;
  EXPECT_EQ(subtraction_with_zero.Canonicalize().ToString(), "(v0 * 5)");

  SymbolicExpr equal_subtraction = (v0 * 5) - (v0 * 5);
  EXPECT_EQ(equal_subtraction.Canonicalize().ToString(), "0");

  SymbolicExpr distribute_mul_over_add = (v0 + 2) * 3;
  EXPECT_EQ(distribute_mul_over_add.Canonicalize().ToString(),
            "((v0 * 3) + 6)");

  SymbolicExpr term_sorting = (v1 * 3) + (v0 * 2);
  EXPECT_EQ(term_sorting.Canonicalize().ToString(), "((v0 * 2) + (v1 * 3))");

  SymbolicExpr add_associativity_and_commutativity = v0 + v1 + v0 + v1;
  EXPECT_EQ(add_associativity_and_commutativity.Canonicalize().ToString(),
            "((v0 * 2) + (v1 * 2))");

  SymbolicExpr complex_expression = ((v1 * 2) + 5) + ((v0 - v1) * 3);
  EXPECT_EQ(complex_expression.Canonicalize().ToString(),
            "(((v0 * 3) + (v1 * -1)) + 5)");

  SymbolicExpr nested_dist = (c2 * (v0 + 1) + 3) * 4;
  EXPECT_EQ(nested_dist.Canonicalize().ToString(), "((v0 * 8) + 20)");
}

TEST_F(SymbolicExprTest, Canonicalization_MinMax) {
  // Min - Max
  EXPECT_EQ((c2.min(5) + c2.max(7)).Canonicalize().ToString(), "9");
  EXPECT_EQ((v0.max(v0)).Canonicalize().ToString(), "v0");
  EXPECT_EQ((v0.min(v0)).Canonicalize().ToString(), "v0");
  EXPECT_EQ((v0.max(v0 + 1)).Canonicalize().ToString(), "(v0 + 1)");
  EXPECT_EQ((v0.min(v0 - 1)).Canonicalize().ToString(), "(v0 + -1)");
  EXPECT_EQ((v0.min(v1) + v0.min(v1)).Canonicalize().ToString(),
            "(min(v0, v1) * 2)");
}

TEST_F(SymbolicExprTest, Canonicalization_DivMod) {
  // FloorDiv, CeilDiv, and Mod simplifications.
  EXPECT_EQ((v0.floorDiv(1)).Canonicalize().ToString(), "v0");
  EXPECT_EQ((v0.ceilDiv(1)).Canonicalize().ToString(), "v0");
  EXPECT_EQ((v0 % 1).Canonicalize().ToString(), "0");

  EXPECT_EQ(((v0 * 8).floorDiv(4)).Canonicalize().ToString(), "(v0 * 2)");
  EXPECT_EQ(((v0 * 8).ceilDiv(4)).Canonicalize().ToString(), "(v0 * 2)");
  EXPECT_EQ(((v0 * 8 + 3).floorDiv(4)).Canonicalize().ToString(), "(v0 * 2)");
  EXPECT_EQ(((v0 * 8 + 3).ceilDiv(4)).Canonicalize().ToString(),
            "((v0 * 2) + 1)");

  EXPECT_EQ(((v0 * 8 + 4).floorDiv(4)).Canonicalize().ToString(),
            "((v0 * 2) + 1)");
  EXPECT_EQ(((v0 * 8 + 4).ceilDiv(4)).Canonicalize().ToString(),
            "((v0 * 2) + 1)");

  EXPECT_EQ(((v0 * 8) % 4).Canonicalize().ToString(), "0");
  EXPECT_EQ(((v0 * 8 + 3) % 4).Canonicalize().ToString(), "3");

  // Test ceilDiv with negative divisor.
  EXPECT_EQ((v0.ceilDiv(-1)).Canonicalize().ToString(), "(v0 * -1)");
  EXPECT_EQ((v0.ceilDiv(-2)).Canonicalize().ToString(),
            "((v0 floordiv 2) * -1)");
  EXPECT_EQ(((v0 * 6).floorDiv(-3)).Canonicalize().ToString(), "(v0 * -2)");
  EXPECT_EQ(((v0 * 6).ceilDiv(-3)).Canonicalize().ToString(), "(v0 * -2)");
}

TEST_F(SymbolicExprTest, Walk) {
  SymbolicExpr expr = (v0 + 42) * v1;
  std::vector<std::string> visited_exprs;
  expr.Walk([&](SymbolicExpr e) { visited_exprs.push_back(e.ToString()); });

  EXPECT_THAT(visited_exprs, ::testing::ElementsAre("v0", "42", "(v0 + 42)",
                                                    "v1", "((v0 + 42) * v1)"));
}

TEST_F(SymbolicExprTest, Hashing) {
  absl::flat_hash_set<SymbolicExpr> set;

  SymbolicExpr c42_1 = CreateSymbolicConstant(42, &ctx);
  SymbolicExpr c42_2 = CreateSymbolicConstant(42, &ctx);
  SymbolicExpr c3 = CreateSymbolicConstant(3, &ctx);

  set.insert(c42_1);
  set.insert(c42_2);
  set.insert(c3);
  EXPECT_EQ(set.size(), 2);

  SymbolicExpr v0_1 = CreateSymbolicVariable(0, &ctx);
  SymbolicExpr v0_2 = CreateSymbolicVariable(0, &ctx);
  SymbolicExpr v1 = CreateSymbolicVariable(1, &ctx);

  set.insert(v0_1);
  set.insert(v0_2);
  set.insert(v1);
  EXPECT_EQ(set.size(), 4);

  SymbolicExpr add1 = v0_1 + c42_1;
  SymbolicExpr add2 = v0_2 + c42_2;
  SymbolicExpr add3 = v1 + c3;

  set.insert(add1);
  set.insert(add2);
  set.insert(add3);
  EXPECT_EQ(set.size(), 6);
}

}  // namespace
}  // namespace xla

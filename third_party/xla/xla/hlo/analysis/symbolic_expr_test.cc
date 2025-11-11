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
#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/DenseMap.h"
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
  mlir::MLIRContext mlir_context;
  SymbolicExprContext ctx{&mlir_context};
  SymbolicExpr v0 = ctx.CreateVariable(0);
  SymbolicExpr v1 = ctx.CreateVariable(1);
  SymbolicExpr c2 = ctx.CreateConstant(2);
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
  SymbolicExpr result = expr_to_sub.ReplaceVariables(substitutions);
  EXPECT_EQ(result.ToString(), "(v0 + (v2 * 10))");
}

TEST_F(SymbolicExprTest, ReplaceSymbols) {
  SymbolicExpr d0 = ctx.CreateVariable(0);
  SymbolicExpr s0 = ctx.CreateVariable(1);
  SymbolicExpr s1 = ctx.CreateVariable(2);
  SymbolicExpr c7 = ctx.CreateConstant(7);
  SymbolicExpr expr_to_sub = (d0 + s0 * 2) * s1;
  SymbolicExpr result = expr_to_sub.ReplaceSymbols({d0, c7}, /*num_dims=*/1);
  EXPECT_EQ(result, ((d0 + (d0 * 2)) * c7));
}

TEST_F(SymbolicExprTest, ReplaceDimsAndSymbols) {
  SymbolicExpr d0 = ctx.CreateVariable(0);
  SymbolicExpr s0 = ctx.CreateVariable(1);
  SymbolicExpr s1 = ctx.CreateVariable(2);
  SymbolicExpr c7 = ctx.CreateConstant(7);
  SymbolicExpr expr_to_sub = (d0 + s0 * 2) * s1;
  SymbolicExpr result =
      expr_to_sub.ReplaceDimsAndSymbols({s0}, {d0, c7}, /*num_dims=*/1);
  EXPECT_EQ(result, ((s0 + (d0 * 2)) * c7));

  SymbolicExpr replace_only_dims =
      expr_to_sub.ReplaceDimsAndSymbols({s0}, {}, /*num_dims=*/1);
  EXPECT_EQ(replace_only_dims, ((s0 + (s0 * 2)) * s1));

  SymbolicExpr replace_only_symbols =
      expr_to_sub.ReplaceDimsAndSymbols({}, {d0, c7}, /*num_dims=*/1);
  EXPECT_EQ(replace_only_symbols, ((d0 + (d0 * 2)) * c7));
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

TEST_F(SymbolicExprTest, UniquingDoesNotCrashWithCombinedAffineExpr) {
  mlir::AffineExpr affine_expr = mlir::getAffineDimExpr(0, &mlir_context);
  SymbolicExpr c1 = ctx.CreateConstant(42);
  EXPECT_EQ(affine_expr, mlir::getAffineDimExpr(0, &mlir_context));
  EXPECT_EQ(c1, ctx.CreateConstant(42));
}

TEST_F(SymbolicExprTest, UniquingWorksAcrossDifferentContexts) {
  SymbolicExprContext ctx2(&mlir_context);

  EXPECT_EQ(ctx.CreateConstant(42), ctx2.CreateConstant(42));
  EXPECT_EQ(ctx.CreateVariable(0), ctx2.CreateVariable(0));
}

TEST_F(SymbolicExprTest, Replace) {
  SymbolicExpr d0 = ctx.CreateVariable(0);
  SymbolicExpr d1 = ctx.CreateVariable(1);
  SymbolicExpr c2 = ctx.CreateConstant(2);
  SymbolicExpr c5 = ctx.CreateConstant(5);

  SymbolicExpr expr = (d0 + c2) * (d1 + c2);
  EXPECT_EQ(expr.Replace(d0 + c2, c5), (c5 * (d1 + c2)));
  EXPECT_EQ(expr.Replace(d1, d0), (d0 + c2) * (d0 + c2));
  EXPECT_EQ(expr.Replace(c2, c5), (d0 + c5) * (d1 + c5));
  EXPECT_EQ(expr.Replace(expr, c2), c2);
  EXPECT_EQ(expr.Replace(d1, d1), expr);
  EXPECT_EQ(expr.Replace(ctx.CreateConstant(42), d1), expr);
}

TEST_F(SymbolicExprTest, ReplaceWithMap) {
  SymbolicExpr d0 = ctx.CreateVariable(0);
  SymbolicExpr d1 = ctx.CreateVariable(1);
  SymbolicExpr c2 = ctx.CreateConstant(2);
  SymbolicExpr c5 = ctx.CreateConstant(5);

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
  no_change[ctx.CreateVariable(99)] = c5;
  EXPECT_EQ(expr.Replace(no_change), expr);
}

TEST_F(SymbolicExprTest, BasicSimplificationsAtCreationTime) {
  auto c0 = ctx.CreateConstant(0);
  auto c1 = ctx.CreateConstant(1);
  auto c3 = ctx.CreateConstant(3);

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
  SymbolicExpr mul_2_v0 = ctx.CreateConstant(2) * v0;
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

  SymbolicExpr c42_1 = ctx.CreateConstant(42);
  SymbolicExpr c42_2 = ctx.CreateConstant(42);
  SymbolicExpr c3 = ctx.CreateConstant(3);

  set.insert(c42_1);
  set.insert(c42_2);
  set.insert(c3);
  EXPECT_EQ(set.size(), 2);

  SymbolicExpr v0_1 = ctx.CreateVariable(0);
  SymbolicExpr v0_2 = ctx.CreateVariable(0);
  SymbolicExpr v1 = ctx.CreateVariable(1);

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

TEST_F(SymbolicExprTest, SymbolicExprContextEq) {
  mlir::MLIRContext mlir_context2;
  SymbolicExprContext ctx2(&mlir_context2);

  // Different MLIRContexts should result in different SymbolicExprContexts.
  EXPECT_NE(ctx, ctx2);

  // Same MLIRContext should result in same StorageUniquer and thus equal
  // SymbolicExprContexts.
  SymbolicExprContext ctx3(&mlir_context);
  EXPECT_EQ(ctx, ctx3);
}

}  // namespace
}  // namespace xla

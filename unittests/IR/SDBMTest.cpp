//===- SDBMTest.cpp - SDBM expression unit tests --------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SDBMExpr.h"
#include "gtest/gtest.h"

using namespace mlir;

static MLIRContext *ctx() {
  static thread_local MLIRContext context;
  return &context;
}

namespace {

TEST(SDBMExpr, Constant) {
  // We can create consants and query them.
  auto expr = SDBMConstantExpr::get(ctx(), 42);
  EXPECT_EQ(expr.getValue(), 42);

  // Two separately created constants with identical values are trivially equal.
  auto expr2 = SDBMConstantExpr::get(ctx(), 42);
  EXPECT_EQ(expr, expr2);

  // Hierarchy is okay.
  auto generic = static_cast<SDBMExpr>(expr);
  EXPECT_TRUE(generic.isa<SDBMConstantExpr>());
}

TEST(SDBMExpr, Dim) {
  // We can create dimension expressions and query them.
  auto expr = SDBMDimExpr::get(ctx(), 0);
  EXPECT_EQ(expr.getPosition(), 0);

  // Two separately created dimensions with the same position are trivially
  // equal.
  auto expr2 = SDBMDimExpr::get(ctx(), 0);
  EXPECT_EQ(expr, expr2);

  // Hierarchy is okay.
  auto generic = static_cast<SDBMExpr>(expr);
  EXPECT_TRUE(generic.isa<SDBMDimExpr>());
  EXPECT_TRUE(generic.isa<SDBMInputExpr>());
  EXPECT_TRUE(generic.isa<SDBMPositiveExpr>());
  EXPECT_TRUE(generic.isa<SDBMVaryingExpr>());

  // Dimensions are not Symbols.
  auto symbol = SDBMSymbolExpr::get(ctx(), 0);
  EXPECT_NE(expr, symbol);
  EXPECT_FALSE(expr.isa<SDBMSymbolExpr>());
}

TEST(SDBMExpr, Symbol) {
  // We can create symbol expressions and query them.
  auto expr = SDBMSymbolExpr::get(ctx(), 0);
  EXPECT_EQ(expr.getPosition(), 0);

  // Two separately created symbols with the same position are trivially equal.
  auto expr2 = SDBMSymbolExpr::get(ctx(), 0);
  EXPECT_EQ(expr, expr2);

  // Hierarchy is okay.
  auto generic = static_cast<SDBMExpr>(expr);
  EXPECT_TRUE(generic.isa<SDBMSymbolExpr>());
  EXPECT_TRUE(generic.isa<SDBMInputExpr>());
  EXPECT_TRUE(generic.isa<SDBMPositiveExpr>());
  EXPECT_TRUE(generic.isa<SDBMVaryingExpr>());

  // Dimensions are not Symbols.
  auto symbol = SDBMDimExpr::get(ctx(), 0);
  EXPECT_NE(expr, symbol);
  EXPECT_FALSE(expr.isa<SDBMDimExpr>());
}

TEST(SDBMExpr, Stripe) {
  auto cst2 = SDBMConstantExpr::get(ctx(), 2);
  auto cst0 = SDBMConstantExpr::get(ctx(), 0);
  auto var = SDBMSymbolExpr::get(ctx(), 0);

  // We can create stripe expressions and query them.
  auto expr = SDBMStripeExpr::get(var, cst2);
  EXPECT_EQ(expr.getVar(), var);
  EXPECT_EQ(expr.getStripeFactor(), cst2);

  // Two separately created stripe expressions with the same LHS and RHS are
  // trivially equal.
  auto expr2 = SDBMStripeExpr::get(SDBMSymbolExpr::get(ctx(), 0), cst2);
  EXPECT_EQ(expr, expr2);

  // Stripes can be nested.
  SDBMStripeExpr::get(expr, SDBMConstantExpr::get(ctx(), 4));

  // Non-positive stripe factors are not allowed.
  EXPECT_DEATH(SDBMStripeExpr::get(var, cst0), "non-positive");

  // Hierarchy is okay.
  auto generic = static_cast<SDBMExpr>(expr);
  EXPECT_TRUE(generic.isa<SDBMStripeExpr>());
  EXPECT_TRUE(generic.isa<SDBMPositiveExpr>());
  EXPECT_TRUE(generic.isa<SDBMVaryingExpr>());
}

TEST(SDBMExpr, Neg) {
  auto cst2 = SDBMConstantExpr::get(ctx(), 2);
  auto var = SDBMSymbolExpr::get(ctx(), 0);
  auto stripe = SDBMStripeExpr::get(var, cst2);

  // We can create negation expressions and query them.
  auto expr = SDBMNegExpr::get(var);
  EXPECT_EQ(expr.getVar(), var);
  auto expr2 = SDBMNegExpr::get(stripe);
  EXPECT_EQ(expr2.getVar(), stripe);

  // Neg expressions are trivially comparable.
  EXPECT_EQ(expr, SDBMNegExpr::get(var));

  // Hierarchy is okay.
  auto generic = static_cast<SDBMExpr>(expr);
  EXPECT_TRUE(generic.isa<SDBMNegExpr>());
  EXPECT_TRUE(generic.isa<SDBMVaryingExpr>());
}

TEST(SDBMExpr, Sum) {
  auto cst2 = SDBMConstantExpr::get(ctx(), 2);
  auto var = SDBMSymbolExpr::get(ctx(), 0);
  auto stripe = SDBMStripeExpr::get(var, cst2);

  // We can create sum expressions and query them.
  auto expr = SDBMSumExpr::get(var, cst2);
  EXPECT_EQ(expr.getLHS(), var);
  EXPECT_EQ(expr.getRHS(), cst2);
  auto expr2 = SDBMSumExpr::get(stripe, cst2);
  EXPECT_EQ(expr2.getLHS(), stripe);
  EXPECT_EQ(expr2.getRHS(), cst2);

  // Sum expressions are trivially comparable.
  EXPECT_EQ(expr, SDBMSumExpr::get(var, cst2));

  // Hierarchy is okay.
  auto generic = static_cast<SDBMExpr>(expr);
  EXPECT_TRUE(generic.isa<SDBMSumExpr>());
  EXPECT_TRUE(generic.isa<SDBMVaryingExpr>());
}

TEST(SDBMExpr, Diff) {
  auto cst2 = SDBMConstantExpr::get(ctx(), 2);
  auto var = SDBMSymbolExpr::get(ctx(), 0);
  auto stripe = SDBMStripeExpr::get(var, cst2);

  // We can create sum expressions and query them.
  auto expr = SDBMDiffExpr::get(var, stripe);
  EXPECT_EQ(expr.getLHS(), var);
  EXPECT_EQ(expr.getRHS(), stripe);
  auto expr2 = SDBMDiffExpr::get(stripe, var);
  EXPECT_EQ(expr2.getLHS(), stripe);
  EXPECT_EQ(expr2.getRHS(), var);

  // Sum expressions are trivially comparable.
  EXPECT_EQ(expr, SDBMDiffExpr::get(var, stripe));

  // Hierarchy is okay.
  auto generic = static_cast<SDBMExpr>(expr);
  EXPECT_TRUE(generic.isa<SDBMDiffExpr>());
  EXPECT_TRUE(generic.isa<SDBMVaryingExpr>());
}

TEST(SDBMExpr, AffineRoundTrip) {
  // Build an expression (s0 - s0 # 2)
  auto cst2 = SDBMConstantExpr::get(ctx(), 2);
  auto var = SDBMSymbolExpr::get(ctx(), 0);
  auto stripe = SDBMStripeExpr::get(var, cst2);
  auto expr = SDBMDiffExpr::get(var, stripe);

  // Check that it can be converted to AffineExpr and back, i.e. stripe
  // detection works correctly.
  Optional<SDBMExpr> roundtripped =
      SDBMExpr::tryConvertAffineExpr(expr.getAsAffineExpr());
  ASSERT_TRUE(roundtripped.hasValue());
  EXPECT_EQ(roundtripped, static_cast<SDBMExpr>(expr));

  // Check that (s0 # 2 # 5) can be converted to AffineExpr, i.e. stripe
  // detection supports nested expressions.
  auto cst5 = SDBMConstantExpr::get(ctx(), 5);
  auto outerStripe = SDBMStripeExpr::get(stripe, cst5);
  roundtripped = SDBMExpr::tryConvertAffineExpr(outerStripe.getAsAffineExpr());
  ASSERT_TRUE(roundtripped.hasValue());
  EXPECT_EQ(roundtripped, static_cast<SDBMExpr>(outerStripe));

  // Check that (s0 # 2 # 5 - s0 # 2) + 2 can be converted as an example of a
  // deeper expression tree.
  auto diff = SDBMDiffExpr::get(outerStripe, stripe);
  auto sum = SDBMSumExpr::get(diff, cst2);
  roundtripped = SDBMExpr::tryConvertAffineExpr(sum.getAsAffineExpr());
  ASSERT_TRUE(roundtripped.hasValue());
  EXPECT_EQ(roundtripped, static_cast<SDBMExpr>(sum));
}

TEST(SDBMExpr, MatchStripeMulPattern) {
  // Make sure conversion from AffineExpr recognizes multiplicative stripe
  // pattern (x floordiv B) * B == x # B.
  auto cst = getAffineConstantExpr(42, ctx());
  auto dim = getAffineDimExpr(0, ctx());
  auto floor = getAffineBinaryOpExpr(AffineExprKind::FloorDiv, dim, cst);
  auto mul = getAffineBinaryOpExpr(AffineExprKind::Mul, cst, floor);
  Optional<SDBMExpr> converted = SDBMStripeExpr::tryConvertAffineExpr(mul);
  ASSERT_TRUE(converted.hasValue());
  EXPECT_TRUE(converted->isa<SDBMStripeExpr>());
}

TEST(SDBMExpr, NonSDBM) {
  auto d0 = getAffineDimExpr(0, ctx());
  auto d1 = getAffineDimExpr(1, ctx());
  auto sum = getAffineBinaryOpExpr(AffineExprKind::Add, d0, d1);
  auto c2 = getAffineConstantExpr(2, ctx());
  auto prod = getAffineBinaryOpExpr(AffineExprKind::Mul, d0, c2);
  auto ceildiv = getAffineBinaryOpExpr(AffineExprKind::CeilDiv, d1, c2);

  // The following are not valid SDBM expressions:
  // - a sum of two variables
  EXPECT_FALSE(SDBMExpr::tryConvertAffineExpr(sum).hasValue());
  // - a variable with coefficient other than 1 or -1
  EXPECT_FALSE(SDBMExpr::tryConvertAffineExpr(prod).hasValue());
  // - a ceildiv expression
  EXPECT_FALSE(SDBMExpr::tryConvertAffineExpr(ceildiv).hasValue());
}

} // end namespace

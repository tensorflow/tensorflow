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

#include "mlir/IR/SDBM.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SDBMExpr.h"
#include "gtest/gtest.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;

static MLIRContext *ctx() {
  static thread_local MLIRContext context;
  return &context;
}

static SDBMExpr dim(unsigned pos) { return SDBMDimExpr::get(ctx(), pos); }

static SDBMExpr symb(unsigned pos) { return SDBMSymbolExpr::get(ctx(), pos); }

namespace {

using namespace mlir::ops_assertions;

TEST(SDBMOperators, Add) {
  auto expr = dim(0) + 42;
  auto sumExpr = expr.dyn_cast<SDBMSumExpr>();
  ASSERT_TRUE(sumExpr);
  EXPECT_EQ(sumExpr.getLHS(), dim(0));
  EXPECT_EQ(sumExpr.getRHS().getValue(), 42);
}

TEST(SDBMOperators, AddFolding) {
  auto constant = SDBMConstantExpr::get(ctx(), 2) + 42;
  auto constantExpr = constant.dyn_cast<SDBMConstantExpr>();
  ASSERT_TRUE(constantExpr);
  EXPECT_EQ(constantExpr.getValue(), 44);

  auto expr = (dim(0) + 10) + 32;
  auto sumExpr = expr.dyn_cast<SDBMSumExpr>();
  ASSERT_TRUE(sumExpr);
  EXPECT_EQ(sumExpr.getRHS().getValue(), 42);

  expr = dim(0) + SDBMNegExpr::get(SDBMDimExpr::get(ctx(), 1));
  auto diffExpr = expr.dyn_cast<SDBMDiffExpr>();
  ASSERT_TRUE(diffExpr);
  EXPECT_EQ(diffExpr.getLHS(), dim(0));
  EXPECT_EQ(diffExpr.getRHS(), dim(1));

  auto inverted = SDBMNegExpr::get(SDBMDimExpr::get(ctx(), 1)) + dim(0);
  EXPECT_EQ(inverted, expr);
}

TEST(SDBMOperators, Diff) {
  auto expr = dim(0) - dim(1);
  auto diffExpr = expr.dyn_cast<SDBMDiffExpr>();
  ASSERT_TRUE(diffExpr);
  EXPECT_EQ(diffExpr.getLHS(), dim(0));
  EXPECT_EQ(diffExpr.getRHS(), dim(1));
}

TEST(SDBMOperators, DiffFolding) {
  auto constant = SDBMConstantExpr::get(ctx(), 10) - 3;
  auto constantExpr = constant.dyn_cast<SDBMConstantExpr>();
  ASSERT_TRUE(constantExpr);
  EXPECT_EQ(constantExpr.getValue(), 7);

  auto expr = dim(0) - 3;
  auto sumExpr = expr.dyn_cast<SDBMSumExpr>();
  ASSERT_TRUE(sumExpr);
  EXPECT_EQ(sumExpr.getRHS().getValue(), -3);

  auto zero = dim(0) - dim(0);
  constantExpr = zero.dyn_cast<SDBMConstantExpr>();
  ASSERT_TRUE(constantExpr);
  EXPECT_EQ(constantExpr.getValue(), 0);
}

TEST(SDBMOperators, Stripe) {
  auto expr = stripe(dim(0), 3);
  auto stripeExpr = expr.dyn_cast<SDBMStripeExpr>();
  ASSERT_TRUE(stripeExpr);
  EXPECT_EQ(stripeExpr.getVar(), dim(0));
  EXPECT_EQ(stripeExpr.getStripeFactor().getValue(), 3);
}

TEST(SDBM, SingleConstraint) {
  // Build an SDBM defined by
  //
  //   d0 - 3 <= 0  <=>  d0 <= 3.
  //
  //         cst   d0
  //   cst   inf    3
  //   d0    inf  inf
  auto sdbm = SDBM::get(dim(0) - 3, llvm::None);
  EXPECT_EQ(sdbm(0, 1), 3);
}

TEST(SDBM, Equality) {
  // Build an SDBM defined by
  //
  //   d0 - d1 - 3 = 0  <=> {d0 - d1 - 3 <= 0 and d0 - d1 - 3 >= 0} <=>
  //   <=> {d0 - d1 <= 3 and d1 - d0 <= -3}.
  //
  //         cst   d0   d1
  //   cst   inf  inf  inf
  //   d0    inf  inf   -3
  //   d1    inf    3  inf
  auto sdbm = SDBM::get(llvm::None, dim(0) - dim(1) - 3);
  EXPECT_EQ(sdbm(1, 2), -3);
  EXPECT_EQ(sdbm(2, 1), 3);
}

TEST(SDBM, TrivialSimplification) {
  // Build an SDBM defined by
  //
  //   d0 - 3 <= 0  <=>  d0 <= 3
  //   d0 - 5 <= 0  <=>  d0 <= 5
  //
  // which should get simplifed on construction to only the former.
  //
  //         cst   d0
  //   cst   inf    3
  //   d0    inf  inf;
  auto sdbm = SDBM::get({dim(0) - 3, dim(0) - 5}, llvm::None);
  EXPECT_EQ(sdbm(0, 1), 3);
}

TEST(SDBM, StripeInducedIneqs) {
  // Build an SDBM defined by d1 = d0 # 3, which induces the constraints
  //
  //   d1 - d0 <= 0
  //   d0 - d1 <= 3 - 1 = 2
  //
  //       cst   d0   d1
  // cst   inf  inf  inf
  // d0    inf  inf    2
  // d1    inf    0    0
  auto sdbm = SDBM::get(llvm::None, dim(1) - stripe(dim(0), 3));
  EXPECT_EQ(sdbm(1, 2), 2);
  EXPECT_EQ(sdbm(2, 1), 0);
}

TEST(SDBM, StripeTemporaries) {
  // Build an SDBM defined by d0 # 3 <= 0, which creates a temporary
  // t0 = d0 # 3 leading to a constraint t0 <= 0 and the stripe-induced
  // constraints
  //
  //   t0 - d0 <= 0
  //   d0 - t0 <= 3 - 1 = 2
  //
  //       cst   d0   t0
  // cst   inf  inf    0
  // d0    inf  inf    2
  // t0    inf    0  inf
  auto sdbm = SDBM::get(stripe(dim(0), 3), llvm::None);
  EXPECT_EQ(sdbm(0, 2), 0);
  EXPECT_EQ(sdbm(1, 2), 2);
  EXPECT_EQ(sdbm(2, 1), 0);
}

TEST(SDBM, RoundTripEqs) {
  // Build and SDBM defined by
  //
  //   d0 = s0 # 3 # 5
  //   s0 # 3 # 5 - d1 + 42 = 0
  //
  // and perform a double round-trip between the "list of equalities" and SDBM
  // representation.  After the first round-trip, the equalities may be
  // different due to simplification or equivalent substitutions (e.g., the
  // second equality may become d0 - d1 + 42 = 0).  However, there should not
  // be any further simplification after the second round-trip,

  // Build the SDBM from a pair of equalities and extract back the lists of
  // inequalities and equalities.  Check that all equalities are properly
  // detected and none of them decayed into inequalities.
  auto s = stripe(stripe(symb(0), 3), 5);
  auto sdbm = SDBM::get(llvm::None, {s - dim(0), s - dim(1) + 42});
  SmallVector<SDBMExpr, 4> eqs, ineqs;
  sdbm.getSDBMExpressions(ctx(), ineqs, eqs);
  ASSERT_TRUE(ineqs.empty());

  // Do the second round-trip.
  auto sdbm2 = SDBM::get(llvm::None, eqs);
  SmallVector<SDBMExpr, 4> eqs2, ineqs2;
  sdbm2.getSDBMExpressions(ctx(), ineqs2, eqs2);
  ASSERT_EQ(eqs.size(), eqs2.size());

  // Convert that the sets of equalities are equal, their order is not relevant.
  llvm::DenseSet<SDBMExpr> eqSet, eq2Set;
  eqSet.insert(eqs.begin(), eqs.end());
  eq2Set.insert(eqs2.begin(), eqs2.end());
  EXPECT_EQ(eqSet, eq2Set);
}

TEST(SDBM, StripeTightening) {
  // Build an SDBM defined by
  //
  //   d0 = s0 # 3 # 5
  //   s0 # 3 # 5 - d1 + 42 = 0
  //   d0 - s0 # 3 <= 2
  //
  // where the last inequality is tighter than that induced by the first stripe
  // equality (d0 - s0 # 3 <= 5 - 1 = 4).  Check that the conversion from SDBM
  // back to the lists of constraints conserves both the stripe equality and the
  // tighter inequality.
  auto s = stripe(stripe(symb(0), 3), 5);
  auto tight = dim(0) - stripe(symb(0), 3) - 2;
  auto sdbm = SDBM::get({tight}, {s - dim(0), s - dim(1) + 42});

  SmallVector<SDBMExpr, 4> eqs, ineqs;
  sdbm.getSDBMExpressions(ctx(), ineqs, eqs);
  ASSERT_EQ(ineqs.size(), 1u);
  EXPECT_EQ(ineqs[0], tight);
  EXPECT_EQ(eqs.size(), 2u);
}

TEST(SDBM, StripeTransitive) {
  // Build an SDBM defined by
  //
  //   d0 = d1 # 3
  //   d0 = d2 # 7
  //
  // where the same dimension is declared equal to two stripe expressions over
  // different variables.  This is practically handled by introducing a
  // temporary variable for the second stripe expression and adding an equality
  // constraint between this variable and the original dimension variable.
  //
  //       cst   d0   d1   d2   t0
  // cst   inf  inf  inf  inf  inf
  // d0    inf    0    0  inf    0
  // d1    inf    2  inf  inf  inf
  // d2    inf  inf  inf  inf    6
  // t0    inf    0  inf    0  inf
  auto sdbm = SDBM::get(
      llvm::None, {stripe(dim(1), 3) - dim(0), stripe(dim(2), 7) - dim(0)});
  // Induced by d0 = d1 # 3.
  EXPECT_EQ(sdbm(1, 2), 0);
  EXPECT_EQ(sdbm(2, 1), 2);
  // Equality d0 = t0.
  EXPECT_EQ(sdbm(1, 4), 0);
  EXPECT_EQ(sdbm(4, 1), 0);
  // Induced by t0 = d2 # 7.
  EXPECT_EQ(sdbm(3, 4), 6);
  EXPECT_EQ(sdbm(4, 3), 0);
}

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

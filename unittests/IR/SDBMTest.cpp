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

} // end namespace

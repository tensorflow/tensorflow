//===- sdbm-api-test.cpp - Tests for SDBM expression APIs -----------------===//
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

// RUN: %p/sdbm-api-test | FileCheck %s

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SDBM.h"
#include "mlir/IR/SDBMExpr.h"

#include "llvm/Support/raw_ostream.h"

#include "APITest.h"

using namespace mlir;

static MLIRContext *ctx() {
  static thread_local MLIRContext context;
  return &context;
}

static SDBMExpr dim(unsigned pos) { return SDBMDimExpr::get(ctx(), pos); }

static SDBMExpr symb(unsigned pos) { return SDBMSymbolExpr::get(ctx(), pos); }

namespace {

using namespace mlir::ops_assertions;

TEST_FUNC(SDBM_SingleConstraint) {
  // Build an SDBM defined by
  //   d0 - 3 <= 0  <=>  d0 <= 3.
  auto sdbm = SDBM::get(dim(0) - 3, llvm::None);

  //      CHECK:       cst   d0
  // CHECK-NEXT: cst   inf    3
  // CHECK-NEXT: d0    inf  inf
  sdbm.print(llvm::outs());
}

TEST_FUNC(SDBM_Equality) {
  // Build an SDBM defined by
  //
  //   d0 - d1 - 3 = 0
  //     <=> {d0 - d1 - 3 <= 0 and d0 - d1 - 3 >= 0}
  //     <=> {d0 - d1 <= 3 and d1 - d0 <= -3}.
  auto sdbm = SDBM::get(llvm::None, dim(0) - dim(1) - 3);

  //      CHECK:       cst   d0   d1
  // CHECK-NEXT: cst   inf  inf  inf
  // CHECK-NEXT: d0    inf  inf   -3
  // CHECK-NEXT: d1    inf    3  inf
  sdbm.print(llvm::outs());
}

TEST_FUNC(SDBM_TrivialSimplification) {
  // Build an SDBM defined by
  //
  //   d0 - 3 <= 0  <=>  d0 <= 3
  //   d0 - 5 <= 0  <=>  d0 <= 5
  //
  // which should get simplifed on construction to only the former.
  auto sdbm = SDBM::get({dim(0) - 3, dim(0) - 5}, llvm::None);

  //      CHECK:       cst   d0
  // CHECK-NEXT: cst   inf    3
  // CHECK-NEXT: d0    inf  inf
  sdbm.print(llvm::outs());
}

TEST_FUNC(SDBM_StripeInducedIneqs) {
  // Build an SDBM defined by d1 = d0 # 3, which induces the constraints
  //
  //   d1 - d0 <= 0
  //   d0 - d1 <= 3 - 1 = 2
  auto sdbm = SDBM::get(llvm::None, dim(1) - stripe(dim(0), 3));

  //      CHECK:       cst   d0   d1
  // CHECK-NEXT: cst   inf  inf  inf
  // CHECK-NEXT: d0    inf  inf    2
  // CHECK-NEXT: d1    inf    0    0
  // CHECK-NEXT: d1 = d0 # 3
  sdbm.print(llvm::outs());
}

TEST_FUNC(SDBM_StripeTemporaries) {
  // Build an SDBM defined by d0 # 3 <= 0, which creates a temporary
  // t0 = d0 # 3 leading to a constraint t0 <= 0 and the stripe-induced
  // constraints
  //
  //   t0 - d0 <= 0
  //   d0 - t0 <= 3 - 1 = 2
  auto sdbm = SDBM::get(stripe(dim(0), 3), llvm::None);

  //      CHECK:       cst   d0   t0
  // CHECK-NEXT: cst   inf  inf    0
  // CHECK-NEXT: d0    inf  inf    2
  // CHECK-NEXT: t0    inf    0  inf
  // CHECK-NEXT: t0 = d0 # 3
  sdbm.print(llvm::outs());
}

TEST_FUNC(SDBM_StripeTightening) {
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
  // CHECK-DAG: d0 - s0 # 3 + -2
  // CHECK-DAG: d1 - d0 + -42
  // CHEKC-DAG: d0 - s0 # 3 # 5
  for (auto ineq : ineqs)
    ineq.print(llvm::outs());
  for (auto eq : eqs)
    eq.print(llvm::outs());
}

TEST_FUNC(SDBM_StripeTransitive) {
  // Build an SDBM defined by
  //
  //   d0 = d1 # 3
  //   d0 = d2 # 7
  //
  // where the same dimension is declared equal to two stripe expressions over
  // different variables.  This is practically handled by introducing a
  // temporary variable for the second stripe expression and adding an equality
  // constraint between this variable and the original dimension variable.
  auto sdbm = SDBM::get(
      llvm::None, {stripe(dim(1), 3) - dim(0), stripe(dim(2), 7) - dim(0)});

  //      CHECK:       cst   d0   d1   d2   t0
  // CHECK-NEXT: cst   inf  inf  inf  inf  inf
  // CHECK-NEXT: d0    inf    0    0  inf    0
  // CHECK-NEXT: d1    inf    2  inf  inf  inf
  // CHECK-NEXT: d2    inf  inf  inf  inf    6
  // CHECK-NEXT: t0    inf    0  inf    0  inf
  // CHECK-NEXT: t0 = d2 # 7
  // CHECK-NEXT: d0 = d1 # 3
  sdbm.print(llvm::outs());
}

} // end namespace

int main() {
  RUN_TESTS();
  return 0;
}

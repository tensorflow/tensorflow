//===- AffineMap.cpp - MLIR Affine Map Classes ----------------------------===//
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

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

AffineMap::AffineMap(unsigned numDims, unsigned numSymbols, unsigned numResults,
                     AffineExpr *const *results)
    : numDims(numDims), numSymbols(numSymbols), numResults(numResults),
      results(results) {}

/// Fold to a constant when possible. Canonicalize so that only the RHS is a
/// constant. (4 + d0 becomes d0 + 4). If only one of them is a symbolic
/// expressions, make it the RHS. Return nullptr if it can't be simplified.
AffineExpr *AffineBinaryOpExpr::simplifyAdd(AffineExpr *lhs, AffineExpr *rhs,
                                            MLIRContext *context) {
  if (auto *l = dyn_cast<AffineConstantExpr>(lhs))
    if (auto *r = dyn_cast<AffineConstantExpr>(rhs))
      return AffineConstantExpr::get(l->getValue() + r->getValue(), context);

  if (isa<AffineConstantExpr>(lhs) || (lhs->isSymbolic() && !rhs->isSymbolic()))
    return AffineAddExpr::get(rhs, lhs, context);

  return nullptr;
  // TODO(someone): implement more simplification like x + 0 -> x; (x + 2) + 4
  // -> (x + 6). Do this in a systematic way in conjunction with other
  // simplifications as opposed to incremental hacks.
}

AffineExpr *AffineBinaryOpExpr::simplifySub(AffineExpr *lhs, AffineExpr *rhs,
                                            MLIRContext *context) {
  if (auto *l = dyn_cast<AffineConstantExpr>(lhs))
    if (auto *r = dyn_cast<AffineConstantExpr>(rhs))
      return AffineConstantExpr::get(l->getValue() - r->getValue(), context);

  return nullptr;
  // TODO(someone): implement more simplification like mentioned for add.
}

/// Simplify a multiply expression. Fold it to a constant when possible, and
/// make the symbolic/constant operand the RHS.
AffineExpr *AffineBinaryOpExpr::simplifyMul(AffineExpr *lhs, AffineExpr *rhs,
                                            MLIRContext *context) {
  if (auto *l = dyn_cast<AffineConstantExpr>(lhs))
    if (auto *r = dyn_cast<AffineConstantExpr>(rhs))
      return AffineConstantExpr::get(l->getValue() * r->getValue(), context);

  assert(lhs->isSymbolic() || rhs->isSymbolic());

  // Canonicalize the mul expression so that the constant/symbolic term is the
  // RHS. If both the lhs and rhs are symbolic, swap them if the lhs is a
  // constant. (Note that a constant is trivially symbolic).
  if (!rhs->isSymbolic() || isa<AffineConstantExpr>(lhs)) {
    // At least one of them has to be symbolic.
    return AffineMulExpr::get(rhs, lhs, context);
  }

  return nullptr;
  // TODO(someone): implement some more simplification/canonicalization such as
  // 1*x is same as x, and in general, move it in the form d_i*expr where d_i is
  // a dimensional identifier. So, 2*(d0 + 4) + s0*d0 becomes (2 + s0)*d0 + 8.
}

AffineExpr *AffineBinaryOpExpr::simplifyFloorDiv(AffineExpr *lhs,
                                                 AffineExpr *rhs,
                                                 MLIRContext *context) {
  if (auto *l = dyn_cast<AffineConstantExpr>(lhs))
    if (auto *r = dyn_cast<AffineConstantExpr>(rhs))
      return AffineConstantExpr::get(l->getValue() / r->getValue(), context);

  return nullptr;
  // TODO(someone): implement more simplification along the lines described in
  // simplifyMod TODO. For eg: 128*N floordiv 128 is N.
}

AffineExpr *AffineBinaryOpExpr::simplifyCeilDiv(AffineExpr *lhs,
                                                AffineExpr *rhs,
                                                MLIRContext *context) {
  if (auto *l = dyn_cast<AffineConstantExpr>(lhs))
    if (auto *r = dyn_cast<AffineConstantExpr>(rhs))
      return AffineConstantExpr::get(
          (int64_t)llvm::divideCeil((uint64_t)l->getValue(),
                                    (uint64_t)r->getValue()),
          context);

  return nullptr;
  // TODO(someone): implement more simplification along the lines described in
  // simplifyMod TODO. For eg: 128*N ceildiv 128 is N.
}

AffineExpr *AffineBinaryOpExpr::simplifyMod(AffineExpr *lhs, AffineExpr *rhs,
                                            MLIRContext *context) {
  if (auto *l = dyn_cast<AffineConstantExpr>(lhs))
    if (auto *r = dyn_cast<AffineConstantExpr>(rhs))
      return AffineConstantExpr::get(l->getValue() % r->getValue(), context);

  return nullptr;
  // TODO(someone): implement more simplification; for eg: 2*x mod 2 is 0; (2*x
  // + 1) mod 2 is 1. In general, this can be simplified by using the GCD test
  // iteratively if the RHS of the mod is a small number, or in general using
  // quantifier elimination (add two new variables q and r, and eliminate all
  // variables from the linear system other than r.
}

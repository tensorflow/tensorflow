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
#include "mlir/IR/Attributes.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {

// AffineExprConstantFolder evaluates an affine expression using constant
// operands passed in 'operandConsts'. Returns a pointer to an IntegerAttr
// attribute representing the constant value of the affine expression
// evaluated on constant 'operandConsts'.
class AffineExprConstantFolder {
public:
  AffineExprConstantFolder(unsigned numDims,
                           ArrayRef<Attribute *> operandConsts)
      : numDims(numDims), operandConsts(operandConsts) {}

  /// Attempt to constant fold the specified affine expr, or return null on
  /// failure.
  IntegerAttr *constantFold(AffineExprRef expr) {
    switch (expr->getKind()) {
    case AffineExpr::Kind::Add:
      return constantFoldBinExpr(
          expr, [](int64_t lhs, int64_t rhs) { return lhs + rhs; });
    case AffineExpr::Kind::Mul:
      return constantFoldBinExpr(
          expr, [](int64_t lhs, int64_t rhs) { return lhs * rhs; });
    case AffineExpr::Kind::Mod:
      return constantFoldBinExpr(
          expr, [](int64_t lhs, uint64_t rhs) { return mod(lhs, rhs); });
    case AffineExpr::Kind::FloorDiv:
      return constantFoldBinExpr(
          expr, [](int64_t lhs, uint64_t rhs) { return floorDiv(lhs, rhs); });
    case AffineExpr::Kind::CeilDiv:
      return constantFoldBinExpr(
          expr, [](int64_t lhs, uint64_t rhs) { return ceilDiv(lhs, rhs); });
    case AffineExpr::Kind::Constant:
      return IntegerAttr::get(expr.cast<AffineConstantExprRef>()->getValue(),
                              expr->getContext());
    case AffineExpr::Kind::DimId:
      return dyn_cast_or_null<IntegerAttr>(
          operandConsts[expr.cast<AffineDimExprRef>()->getPosition()]);
    case AffineExpr::Kind::SymbolId:
      return dyn_cast_or_null<IntegerAttr>(
          operandConsts[numDims +
                        expr.cast<AffineSymbolExprRef>()->getPosition()]);
    }
  }

private:
  IntegerAttr *
  constantFoldBinExpr(AffineExprRef expr,
                      std::function<uint64_t(int64_t, uint64_t)> op) {
    auto binOpExpr = expr.cast<AffineBinaryOpExprRef>();
    auto *lhs = constantFold(binOpExpr->getLHS());
    auto *rhs = constantFold(binOpExpr->getRHS());
    if (!lhs || !rhs)
      return nullptr;
    return IntegerAttr::get(op(lhs->getValue(), rhs->getValue()),
                            expr->getContext());
  }

  // The number of dimension operands in AffineMap containing this expression.
  unsigned numDims;
  // The constant valued operands used to evaluate this AffineExpr.
  ArrayRef<Attribute *> operandConsts;
};

} // end anonymous namespace

AffineMap::AffineMap(unsigned numDims, unsigned numSymbols, unsigned numResults,
                     ArrayRef<AffineExprRef> results,
                     ArrayRef<AffineExprRef> rangeSizes)
    : numDims(numDims), numSymbols(numSymbols), numResults(numResults),
      results(results), rangeSizes(rangeSizes) {}

/// Returns a single constant result affine map.
AffineMap *AffineMap::getConstantMap(int64_t val, MLIRContext *context) {
  return get(/*dimCount=*/0, /*symbolCount=*/0,
             {AffineConstantExpr::get(val, context)}, {}, context);
}

bool AffineMap::isIdentity() {
  if (getNumDims() != getNumResults())
    return false;
  ArrayRef<AffineExprRef> results = getResults();
  for (unsigned i = 0, numDims = getNumDims(); i < numDims; ++i) {
    auto expr = results[i].dyn_cast<AffineDimExprRef>();
    if (!expr || expr->getPosition() != i)
      return false;
  }
  return true;
}

bool AffineMap::isSingleConstant() {
  return getNumResults() == 1 && getResult(0).isa<AffineConstantExprRef>();
}

int64_t AffineMap::getSingleConstantResult() {
  assert(isSingleConstant() && "map must have a single constant result");
  return getResult(0).cast<AffineConstantExprRef>()->getValue();
}

AffineExprRef AffineMap::getResult(unsigned idx) { return results[idx]; }

/// Simplify add expression. Return nullptr if it can't be simplified.
AffineExprRef AffineBinaryOpExpr::simplifyAdd(AffineExprRef lhs,
                                              AffineExprRef rhs,
                                              MLIRContext *context) {
  auto lhsConst = lhs.dyn_cast<AffineConstantExprRef>();
  auto rhsConst = rhs.dyn_cast<AffineConstantExprRef>();

  // Fold if both LHS, RHS are a constant.
  if (lhsConst && rhsConst)
    return AffineConstantExpr::get(lhsConst->getValue() + rhsConst->getValue(),
                                   context);

  // Canonicalize so that only the RHS is a constant. (4 + d0 becomes d0 + 4).
  // If only one of them is a symbolic expressions, make it the RHS.
  if (lhs.isa<AffineConstantExprRef>() ||
      (lhs->isSymbolicOrConstant() && !rhs->isSymbolicOrConstant())) {
    return AffineBinaryOpExpr::getAdd(rhs, lhs, context);
  }

  // At this point, if there was a constant, it would be on the right.

  // Addition with a zero is a noop, return the other input.
  if (rhsConst) {
    if (rhsConst->getValue() == 0)
      return lhs;
  }
  // Fold successive additions like (d0 + 2) + 3 into d0 + 5.
  auto lBin = lhs.dyn_cast<AffineBinaryOpExprRef>();
  if (lBin && rhsConst && lBin->getKind() == Kind::Add) {
    if (auto lrhs = lBin->getRHS().dyn_cast<AffineConstantExprRef>())
      return lBin->getLHS() + (lrhs->getValue() + rhsConst->getValue());
  }

  // When doing successive additions, bring constant to the right: turn (d0 + 2)
  // + d1 into (d0 + d1) + 2.
  if (lBin && lBin->getKind() == Kind::Add) {
    if (auto lrhs = lBin->getRHS().dyn_cast<AffineConstantExprRef>()) {
      return lBin->getLHS() + rhs + lrhs;
    }
  }

  return nullptr;
}

/// Simplify a multiply expression. Return nullptr if it can't be simplified.
AffineExprRef AffineBinaryOpExpr::simplifyMul(AffineExprRef lhs,
                                              AffineExprRef rhs,
                                              MLIRContext *context) {
  auto lhsConst = lhs.dyn_cast<AffineConstantExprRef>();
  auto rhsConst = rhs.dyn_cast<AffineConstantExprRef>();

  if (lhsConst && rhsConst)
    return AffineConstantExpr::get(lhsConst->getValue() * rhsConst->getValue(),
                                   context);

  assert(lhs->isSymbolicOrConstant() || rhs->isSymbolicOrConstant());

  // Canonicalize the mul expression so that the constant/symbolic term is the
  // RHS. If both the lhs and rhs are symbolic, swap them if the lhs is a
  // constant. (Note that a constant is trivially symbolic).
  if (!rhs->isSymbolicOrConstant() || lhs.isa<AffineConstantExprRef>()) {
    // At least one of them has to be symbolic.
    return AffineBinaryOpExpr::getMul(rhs, lhs, context);
  }

  // At this point, if there was a constant, it would be on the right.

  // Multiplication with a one is a noop, return the other input.
  if (rhsConst) {
    if (rhsConst->getValue() == 1)
      return lhs;
    // Multiplication with zero.
    if (rhsConst->getValue() == 0)
      return rhsConst;
  }

  // Fold successive multiplications: eg: (d0 * 2) * 3 into d0 * 6.
  auto lBin = lhs.dyn_cast<AffineBinaryOpExprRef>();
  if (lBin && rhsConst && lBin->getKind() == Kind::Mul) {
    if (auto lrhs = lBin->getRHS().dyn_cast<AffineConstantExprRef>())
      return lBin->getLHS() * (lrhs->getValue() * rhsConst->getValue());
  }

  // When doing successive multiplication, bring constant to the right: turn (d0
  // * 2) * d1 into (d0 * d1) * 2.
  if (lBin && lBin->getKind() == Kind::Mul) {
    if (auto lrhs = lBin->getRHS().dyn_cast<AffineConstantExprRef>()) {
      return (lBin->getLHS() * rhs) * lrhs;
    }
  }

  return nullptr;
}

AffineExprRef AffineBinaryOpExpr::simplifyFloorDiv(AffineExprRef lhs,
                                                   AffineExprRef rhs,
                                                   MLIRContext *context) {
  auto lhsConst = lhs.dyn_cast<AffineConstantExprRef>();
  auto rhsConst = rhs.dyn_cast<AffineConstantExprRef>();

  if (lhsConst && rhsConst)
    return AffineConstantExpr::get(
        floorDiv(lhsConst->getValue(), rhsConst->getValue()), context);

  // Fold floordiv of a multiply with a constant that is a multiple of the
  // divisor. Eg: (i * 128) floordiv 64 = i * 2.
  if (rhsConst) {
    if (rhsConst->getValue() == 1)
      return lhs;

    auto lBin = lhs.dyn_cast<AffineBinaryOpExprRef>();
    if (lBin && lBin->getKind() == Kind::Mul) {
      if (auto lrhs = lBin->getRHS().dyn_cast<AffineConstantExprRef>()) {
        // rhsConst is known to be positive if a constant.
        if (lrhs->getValue() % rhsConst->getValue() == 0)
          return lBin->getLHS() * (lrhs->getValue() / rhsConst->getValue());
      }
    }
  }

  return nullptr;
}

AffineExprRef AffineBinaryOpExpr::simplifyCeilDiv(AffineExprRef lhs,
                                                  AffineExprRef rhs,
                                                  MLIRContext *context) {
  auto lhsConst = lhs.dyn_cast<AffineConstantExprRef>();
  auto rhsConst = rhs.dyn_cast<AffineConstantExprRef>();

  if (lhsConst && rhsConst)
    return AffineConstantExpr::get(
        ceilDiv(lhsConst->getValue(), rhsConst->getValue()), context);

  // Fold ceildiv of a multiply with a constant that is a multiple of the
  // divisor. Eg: (i * 128) ceildiv 64 = i * 2.
  if (rhsConst) {
    if (rhsConst->getValue() == 1)
      return lhs;

    auto lBin = lhs.dyn_cast<AffineBinaryOpExprRef>();
    if (lBin && lBin->getKind() == Kind::Mul) {
      if (auto lrhs = lBin->getRHS().dyn_cast<AffineConstantExprRef>()) {
        // rhsConst is known to be positive if a constant.
        if (lrhs->getValue() % rhsConst->getValue() == 0)
          return lBin->getLHS() * (lrhs->getValue() / rhsConst->getValue());
      }
    }
  }

  return nullptr;
}

AffineExprRef AffineBinaryOpExpr::simplifyMod(AffineExprRef lhs,
                                              AffineExprRef rhs,
                                              MLIRContext *context) {
  auto lhsConst = lhs.dyn_cast<AffineConstantExprRef>();
  auto rhsConst = rhs.dyn_cast<AffineConstantExprRef>();

  if (lhsConst && rhsConst)
    return AffineConstantExpr::get(
        mod(lhsConst->getValue(), rhsConst->getValue()), context);

  // Fold modulo of an expression that is known to be a multiple of a constant
  // to zero if that constant is a multiple of the modulo factor. Eg: (i * 128)
  // mod 64 is folded to 0, and less trivially, (i*(j*4*(k*32))) mod 128 = 0.
  if (rhsConst) {
    // rhsConst is known to be positive if a constant.
    if (lhs->getLargestKnownDivisor() % rhsConst->getValue() == 0)
      return AffineConstantExpr::get(0, context);
  }

  return nullptr;
  // TODO(bondhugula): In general, this can be simplified more by using the GCD
  // test, or in general using quantifier elimination (add two new variables q
  // and r, and eliminate all variables from the linear system other than r. All
  // of this can be done through mlir/Analysis/'s FlatAffineConstraints.
}

/// Folds the results of the application of an affine map on the provided
/// operands to a constant if possible. Returns false if the folding happens,
/// true otherwise.
bool AffineMap::constantFold(ArrayRef<Attribute *> operandConstants,
                             SmallVectorImpl<Attribute *> &results) {
  assert(getNumInputs() == operandConstants.size());

  // Fold each of the result expressions.
  AffineExprConstantFolder exprFolder(getNumDims(), operandConstants);
  // Constant fold each AffineExpr in AffineMap and add to 'results'.
  for (auto expr : getResults()) {
    auto *folded = exprFolder.constantFold(expr);
    // If we didn't fold to a constant, then folding fails.
    if (!folded)
      return true;

    results.push_back(folded);
  }
  assert(results.size() == getNumResults() &&
         "constant folding produced the wrong number of results");
  return false;
}

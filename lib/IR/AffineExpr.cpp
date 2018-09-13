//===- AffineExpr.cpp - MLIR Affine Expr Classes --------------------------===//
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
#include "mlir/Support/STLExtras.h"
#include "third_party/llvm/llvm/include/llvm/ADT/STLExtras.h"

using namespace mlir;

AffineBinaryOpExpr::AffineBinaryOpExpr(Kind kind, AffineExpr *lhs,
                                       AffineExpr *rhs)
    : AffineExpr(kind), lhs(lhs), rhs(rhs) {
  // We verify affine op expr forms at construction time.
  switch (kind) {
  case Kind::Add:
    assert(!isa<AffineConstantExpr>(lhs));
    break;
  case Kind::Mul:
    assert(!isa<AffineConstantExpr>(lhs));
    assert(rhs->isSymbolicOrConstant());
    break;
  case Kind::FloorDiv:
    assert(rhs->isSymbolicOrConstant());
    break;
  case Kind::CeilDiv:
    assert(rhs->isSymbolicOrConstant());
    break;
  case Kind::Mod:
    assert(rhs->isSymbolicOrConstant());
    break;
  default:
    llvm_unreachable("unexpected binary affine expr");
  }
}

AffineExpr *AffineBinaryOpExpr::getSub(AffineExpr *lhs, AffineExpr *rhs,
                                       MLIRContext *context) {
  return getAdd(lhs, getMul(rhs, AffineConstantExpr::get(-1, context), context),
                context);
}

AffineExpr *AffineBinaryOpExpr::getAdd(AffineExpr *expr, int64_t rhs,
                                       MLIRContext *context) {
  return get(AffineExpr::Kind::Add, expr, AffineConstantExpr::get(rhs, context),
             context);
}

AffineExpr *AffineBinaryOpExpr::getMul(AffineExpr *expr, int64_t rhs,
                                       MLIRContext *context) {
  return get(AffineExpr::Kind::Mul, expr, AffineConstantExpr::get(rhs, context),
             context);
}

AffineExpr *AffineBinaryOpExpr::getCeilDiv(AffineExpr *lhs, uint64_t rhs,
                                           MLIRContext *context) {
  return get(AffineExpr::Kind::CeilDiv, lhs,
             AffineConstantExpr::get(rhs, context), context);
}

/// Returns true if this expression is made out of only symbols and
/// constants (no dimensional identifiers).
bool AffineExpr::isSymbolicOrConstant() const {
  switch (getKind()) {
  case Kind::Constant:
    return true;
  case Kind::DimId:
    return false;
  case Kind::SymbolId:
    return true;

  case Kind::Add:
  case Kind::Mul:
  case Kind::FloorDiv:
  case Kind::CeilDiv:
  case Kind::Mod: {
    auto expr = cast<AffineBinaryOpExpr>(this);
    return expr->getLHS()->isSymbolicOrConstant() &&
           expr->getRHS()->isSymbolicOrConstant();
  }
  }
}

/// Returns true if this is a pure affine expression, i.e., multiplication,
/// floordiv, ceildiv, and mod is only allowed w.r.t constants.
bool AffineExpr::isPureAffine() const {
  switch (getKind()) {
  case Kind::SymbolId:
  case Kind::DimId:
  case Kind::Constant:
    return true;
  case Kind::Add: {
    auto *op = cast<AffineBinaryOpExpr>(this);
    return op->getLHS()->isPureAffine() && op->getRHS()->isPureAffine();
  }

  case Kind::Mul: {
    // TODO: Canonicalize the constants in binary operators to the RHS when
    // possible, allowing this to merge into the next case.
    auto *op = cast<AffineBinaryOpExpr>(this);
    return op->getLHS()->isPureAffine() && op->getRHS()->isPureAffine() &&
           (isa<AffineConstantExpr>(op->getLHS()) ||
            isa<AffineConstantExpr>(op->getRHS()));
  }
  case Kind::FloorDiv:
  case Kind::CeilDiv:
  case Kind::Mod: {
    auto *op = cast<AffineBinaryOpExpr>(this);
    return op->getLHS()->isPureAffine() &&
           isa<AffineConstantExpr>(op->getRHS());
  }
  }
}

/// Returns the greatest known integral divisor of this affine expression.
uint64_t AffineExpr::getLargestKnownDivisor() const {
  AffineBinaryOpExpr *binExpr = nullptr;
  switch (kind) {
  case Kind::SymbolId:
    LLVM_FALLTHROUGH;
  case Kind::DimId:
    return 1;
  case Kind::Constant:
    return std::abs(cast<AffineConstantExpr>(this)->getValue());
  case Kind::Mul:
    binExpr = cast<AffineBinaryOpExpr>(const_cast<AffineExpr *>(this));
    return binExpr->getLHS()->getLargestKnownDivisor() *
           binExpr->getRHS()->getLargestKnownDivisor();
  case Kind::Add:
    LLVM_FALLTHROUGH;
  case Kind::FloorDiv:
  case Kind::CeilDiv:
  case Kind::Mod:
    binExpr = cast<AffineBinaryOpExpr>(const_cast<AffineExpr *>(this));
    return llvm::GreatestCommonDivisor64(
        binExpr->getLHS()->getLargestKnownDivisor(),
        binExpr->getRHS()->getLargestKnownDivisor());
  }
}

bool AffineExpr::isMultipleOf(int64_t factor) const {
  AffineBinaryOpExpr *binExpr = nullptr;
  uint64_t l, u;
  switch (kind) {
  case Kind::SymbolId:
    LLVM_FALLTHROUGH;
  case Kind::DimId:
    return factor * factor == 1;
  case Kind::Constant:
    return cast<AffineConstantExpr>(this)->getValue() % factor == 0;
  case Kind::Mul:
    binExpr = cast<AffineBinaryOpExpr>(const_cast<AffineExpr *>(this));
    // It's probably not worth optimizing this further (to not traverse the
    // whole sub-tree under - it that would require a version of isMultipleOf
    // that on a 'false' return also returns the largest known divisor).
    return (l = binExpr->getLHS()->getLargestKnownDivisor()) % factor == 0 ||
           (u = binExpr->getRHS()->getLargestKnownDivisor()) % factor == 0 ||
           (l * u) % factor == 0;
  case Kind::Add:
  case Kind::FloorDiv:
  case Kind::CeilDiv:
  case Kind::Mod:
    binExpr = cast<AffineBinaryOpExpr>(const_cast<AffineExpr *>(this));
    return llvm::GreatestCommonDivisor64(
               binExpr->getLHS()->getLargestKnownDivisor(),
               binExpr->getRHS()->getLargestKnownDivisor()) %
               factor ==
           0;
  }
}

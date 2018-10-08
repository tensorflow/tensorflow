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
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::detail;

/// Returns true if this expression is made out of only symbols and
/// constants (no dimensional identifiers).
bool AffineExpr::isSymbolicOrConstant() {
  switch (getKind()) {
  case AffineExprKind::Constant:
    return true;
  case AffineExprKind::DimId:
    return false;
  case AffineExprKind::SymbolId:
    return true;

  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    auto *expr = cast<AffineBinaryOpExpr>(this);
    return expr->getLHS()->isSymbolicOrConstant() &&
           expr->getRHS()->isSymbolicOrConstant();
  }
  }
}

////////////////////////////////// Details /////////////////////////////////////

AffineBinaryOpExpr::AffineBinaryOpExpr(AffineExprKind kind, AffineExprRef lhs,
                                       AffineExprRef rhs, MLIRContext *context)
    : AffineExpr(kind, context), lhs(lhs), rhs(rhs) {
  // We verify affine op expr forms at construction time.
  switch (kind) {
  case AffineExprKind::Add:
    assert(!lhs.isa<AffineConstantExprRef>());
    break;
  case AffineExprKind::Mul:
    assert(!lhs.isa<AffineConstantExprRef>());
    assert(AffineExprRef(rhs)->isSymbolicOrConstant());
    break;
  case AffineExprKind::FloorDiv:
    assert(AffineExprRef(rhs)->isSymbolicOrConstant());
    break;
  case AffineExprKind::CeilDiv:
    assert(AffineExprRef(rhs)->isSymbolicOrConstant());
    break;
  case AffineExprKind::Mod:
    assert(AffineExprRef(rhs)->isSymbolicOrConstant());
    break;
  default:
    llvm_unreachable("unexpected binary affine expr");
  }
}

AffineExprRef AffineBinaryOpExpr::getSub(AffineExprRef lhs, AffineExprRef rhs,
                                         MLIRContext *context) {
  return getAdd(lhs, getMul(rhs, getAffineConstantExpr(-1, context), context),
                context);
}

AffineExprRef AffineBinaryOpExpr::getAdd(AffineExprRef expr, int64_t rhs,
                                         MLIRContext *context) {
  return get(AffineExprKind::Add, expr, getAffineConstantExpr(rhs, context),
             context);
}

AffineExprRef AffineBinaryOpExpr::getMul(AffineExprRef expr, int64_t rhs,
                                         MLIRContext *context) {
  return get(AffineExprKind::Mul, expr, getAffineConstantExpr(rhs, context),
             context);
}

AffineExprRef AffineBinaryOpExpr::getFloorDiv(AffineExprRef lhs, uint64_t rhs,
                                              MLIRContext *context) {
  return get(AffineExprKind::FloorDiv, lhs, getAffineConstantExpr(rhs, context),
             context);
}

AffineExprRef AffineBinaryOpExpr::getCeilDiv(AffineExprRef lhs, uint64_t rhs,
                                             MLIRContext *context) {
  return get(AffineExprKind::CeilDiv, lhs, getAffineConstantExpr(rhs, context),
             context);
}

AffineExprRef AffineBinaryOpExpr::getMod(AffineExprRef lhs, uint64_t rhs,
                                         MLIRContext *context) {
  return get(AffineExprKind::Mod, lhs, getAffineConstantExpr(rhs, context),
             context);
}

/// Returns true if this is a pure affine expression, i.e., multiplication,
/// floordiv, ceildiv, and mod is only allowed w.r.t constants.
bool AffineExpr::isPureAffine() {
  switch (getKind()) {
  case AffineExprKind::SymbolId:
  case AffineExprKind::DimId:
  case AffineExprKind::Constant:
    return true;
  case AffineExprKind::Add: {
    auto *op = cast<AffineBinaryOpExpr>(this);
    return op->getLHS()->isPureAffine() && op->getRHS()->isPureAffine();
  }

  case AffineExprKind::Mul: {
    // TODO: Canonicalize the constants in binary operators to the RHS when
    // possible, allowing this to merge into the next case.
    auto *op = cast<AffineBinaryOpExpr>(this);
    return op->getLHS()->isPureAffine() && op->getRHS()->isPureAffine() &&
           (op->getLHS().isa<AffineConstantExprRef>() ||
            op->getRHS().isa<AffineConstantExprRef>());
  }
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    auto *op = cast<AffineBinaryOpExpr>(this);
    return op->getLHS()->isPureAffine() &&
           op->getRHS().isa<AffineConstantExprRef>();
  }
  }
}

/// Returns the greatest known integral divisor of this affine expression.
uint64_t AffineExpr::getLargestKnownDivisor() {
  AffineBinaryOpExprRef binExpr;
  switch (getKind()) {
  case AffineExprKind::SymbolId:
    LLVM_FALLTHROUGH;
  case AffineExprKind::DimId:
    return 1;
  case AffineExprKind::Constant:
    return std::abs(cast<AffineConstantExpr>(this)->getValue());
  case AffineExprKind::Mul: {
    binExpr = cast<AffineBinaryOpExpr>(this);
    return binExpr->getLHS()->getLargestKnownDivisor() *
           binExpr->getRHS()->getLargestKnownDivisor();
  }
  case AffineExprKind::Add:
    LLVM_FALLTHROUGH;
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    binExpr = cast<AffineBinaryOpExpr>(this);
    return llvm::GreatestCommonDivisor64(
        binExpr->getLHS()->getLargestKnownDivisor(),
        binExpr->getRHS()->getLargestKnownDivisor());
  }
  }
}

bool AffineExpr::isMultipleOf(int64_t factor) {
  AffineBinaryOpExpr *binExpr;
  uint64_t l, u;
  switch (getKind()) {
  case AffineExprKind::SymbolId:
    LLVM_FALLTHROUGH;
  case AffineExprKind::DimId:
    return factor * factor == 1;
  case AffineExprKind::Constant:
    return cast<AffineConstantExpr>(this)->getValue() % factor == 0;
  case AffineExprKind::Mul: {
    binExpr = cast<AffineBinaryOpExpr>(this);
    // It's probably not worth optimizing this further (to not traverse the
    // whole sub-tree under - it that would require a version of isMultipleOf
    // that on a 'false' return also returns the largest known divisor).
    return (l = binExpr->getLHS()->getLargestKnownDivisor()) % factor == 0 ||
           (u = binExpr->getRHS()->getLargestKnownDivisor()) % factor == 0 ||
           (l * u) % factor == 0;
  }
  case AffineExprKind::Add:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    binExpr = cast<AffineBinaryOpExpr>(this);
    return llvm::GreatestCommonDivisor64(
               binExpr->getLHS()->getLargestKnownDivisor(),
               binExpr->getRHS()->getLargestKnownDivisor()) %
               factor ==
           0;
  }
  }
}

MLIRContext *AffineExpr::getContext() { return context; }

///////////////////////////// Done with details ///////////////////////////////

template <> AffineExprRef AffineExprRef::operator+(int64_t v) const {
  return AffineBinaryOpExpr::getAdd(expr, v, expr->getContext());
}
template <> AffineExprRef AffineExprRef::operator+(AffineExprRef other) const {
  return AffineBinaryOpExpr::getAdd(expr, other.expr, expr->getContext());
}
template <> AffineExprRef AffineExprRef::operator*(int64_t v) const {
  return AffineBinaryOpExpr::getMul(expr, v, expr->getContext());
}
template <> AffineExprRef AffineExprRef::operator*(AffineExprRef other) const {
  return AffineBinaryOpExpr::getMul(expr, other.expr, expr->getContext());
}
// Unary minus, delegate to operator*.
template <> AffineExprRef AffineExprRef::operator-() const {
  return AffineBinaryOpExpr::getMul(expr, -1, expr->getContext());
}
// Delegate to operator+.
template <> AffineExprRef AffineExprRef::operator-(int64_t v) const {
  return *this + (-v);
}
template <> AffineExprRef AffineExprRef::operator-(AffineExprRef other) const {
  return *this + (-other);
}
template <> AffineExprRef AffineExprRef::floorDiv(uint64_t v) const {
  return AffineBinaryOpExpr::getFloorDiv(expr, v, expr->getContext());
}
template <> AffineExprRef AffineExprRef::floorDiv(AffineExprRef other) const {
  return AffineBinaryOpExpr::getFloorDiv(expr, other.expr, expr->getContext());
}
template <> AffineExprRef AffineExprRef::ceilDiv(uint64_t v) const {
  return AffineBinaryOpExpr::getCeilDiv(expr, v, expr->getContext());
}
template <> AffineExprRef AffineExprRef::ceilDiv(AffineExprRef other) const {
  return AffineBinaryOpExpr::getCeilDiv(expr, other.expr, expr->getContext());
}
template <> AffineExprRef AffineExprRef::operator%(uint64_t v) const {
  return AffineBinaryOpExpr::getMod(expr, v, expr->getContext());
}
template <> AffineExprRef AffineExprRef::operator%(AffineExprRef other) const {
  return AffineBinaryOpExpr::getMod(expr, other.expr, expr->getContext());
}

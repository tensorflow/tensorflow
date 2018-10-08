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
bool AffineExprClass::isSymbolicOrConstant() {
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
    auto *expr = cast<AffineBinaryOpExprClass>(this);
    return expr->getLHS()->isSymbolicOrConstant() &&
           expr->getRHS()->isSymbolicOrConstant();
  }
  }
}

////////////////////////////////// Details /////////////////////////////////////

AffineBinaryOpExprClass::AffineBinaryOpExprClass(AffineExprKind kind,
                                                 AffineExpr lhs, AffineExpr rhs)
    : AffineExprClass(kind, lhs->getContext()), lhs(lhs), rhs(rhs) {
  // We verify affine op expr forms at construction time.
  switch (kind) {
  case AffineExprKind::Add:
    assert(!lhs.isa<AffineConstantExpr>());
    break;
  case AffineExprKind::Mul:
    assert(!lhs.isa<AffineConstantExpr>());
    assert(AffineExpr(rhs)->isSymbolicOrConstant());
    break;
  case AffineExprKind::FloorDiv:
    assert(AffineExpr(rhs)->isSymbolicOrConstant());
    break;
  case AffineExprKind::CeilDiv:
    assert(AffineExpr(rhs)->isSymbolicOrConstant());
    break;
  case AffineExprKind::Mod:
    assert(AffineExpr(rhs)->isSymbolicOrConstant());
    break;
  default:
    llvm_unreachable("unexpected binary affine expr");
  }
}

AffineExpr AffineBinaryOpExprClass::getSub(AffineExpr lhs, AffineExpr rhs) {
  return getAdd(lhs, getMul(rhs, getAffineConstantExpr(-1, lhs->getContext())));
}

AffineExpr AffineBinaryOpExprClass::getAdd(AffineExpr expr, int64_t rhs) {
  return get(AffineExprKind::Add, expr,
             getAffineConstantExpr(rhs, expr->getContext()));
}

AffineExpr AffineBinaryOpExprClass::getMul(AffineExpr expr, int64_t rhs) {
  return get(AffineExprKind::Mul, expr,
             getAffineConstantExpr(rhs, expr->getContext()));
}

AffineExpr AffineBinaryOpExprClass::getFloorDiv(AffineExpr lhs, uint64_t rhs) {
  return get(AffineExprKind::FloorDiv, lhs,
             getAffineConstantExpr(rhs, lhs->getContext()));
}

AffineExpr AffineBinaryOpExprClass::getCeilDiv(AffineExpr lhs, uint64_t rhs) {
  return get(AffineExprKind::CeilDiv, lhs,
             getAffineConstantExpr(rhs, lhs->getContext()));
}

AffineExpr AffineBinaryOpExprClass::getMod(AffineExpr lhs, uint64_t rhs) {
  return get(AffineExprKind::Mod, lhs,
             getAffineConstantExpr(rhs, lhs->getContext()));
}

/// Returns true if this is a pure affine expression, i.e., multiplication,
/// floordiv, ceildiv, and mod is only allowed w.r.t constants.
bool AffineExprClass::isPureAffine() {
  switch (getKind()) {
  case AffineExprKind::SymbolId:
  case AffineExprKind::DimId:
  case AffineExprKind::Constant:
    return true;
  case AffineExprKind::Add: {
    auto *op = cast<AffineBinaryOpExprClass>(this);
    return op->getLHS()->isPureAffine() && op->getRHS()->isPureAffine();
  }

  case AffineExprKind::Mul: {
    // TODO: Canonicalize the constants in binary operators to the RHS when
    // possible, allowing this to merge into the next case.
    auto *op = cast<AffineBinaryOpExprClass>(this);
    return op->getLHS()->isPureAffine() && op->getRHS()->isPureAffine() &&
           (op->getLHS().isa<AffineConstantExpr>() ||
            op->getRHS().isa<AffineConstantExpr>());
  }
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    auto *op = cast<AffineBinaryOpExprClass>(this);
    return op->getLHS()->isPureAffine() &&
           op->getRHS().isa<AffineConstantExpr>();
  }
  }
}

/// Returns the greatest known integral divisor of this affine expression.
uint64_t AffineExprClass::getLargestKnownDivisor() {
  AffineBinaryOpExpr binExpr;
  switch (getKind()) {
  case AffineExprKind::SymbolId:
    LLVM_FALLTHROUGH;
  case AffineExprKind::DimId:
    return 1;
  case AffineExprKind::Constant:
    return std::abs(cast<AffineConstantExprClass>(this)->getValue());
  case AffineExprKind::Mul: {
    binExpr = cast<AffineBinaryOpExprClass>(this);
    return binExpr->getLHS()->getLargestKnownDivisor() *
           binExpr->getRHS()->getLargestKnownDivisor();
  }
  case AffineExprKind::Add:
    LLVM_FALLTHROUGH;
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    binExpr = cast<AffineBinaryOpExprClass>(this);
    return llvm::GreatestCommonDivisor64(
        binExpr->getLHS()->getLargestKnownDivisor(),
        binExpr->getRHS()->getLargestKnownDivisor());
  }
  }
}

bool AffineExprClass::isMultipleOf(int64_t factor) {
  AffineBinaryOpExprClass *binExpr;
  uint64_t l, u;
  switch (getKind()) {
  case AffineExprKind::SymbolId:
    LLVM_FALLTHROUGH;
  case AffineExprKind::DimId:
    return factor * factor == 1;
  case AffineExprKind::Constant:
    return cast<AffineConstantExprClass>(this)->getValue() % factor == 0;
  case AffineExprKind::Mul: {
    binExpr = cast<AffineBinaryOpExprClass>(this);
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
    binExpr = cast<AffineBinaryOpExprClass>(this);
    return llvm::GreatestCommonDivisor64(
               binExpr->getLHS()->getLargestKnownDivisor(),
               binExpr->getRHS()->getLargestKnownDivisor()) %
               factor ==
           0;
  }
  }
}

MLIRContext *AffineExprClass::getContext() { return context; }

///////////////////////////// Done with details ///////////////////////////////

template <> AffineExpr AffineExpr::operator+(int64_t v) const {
  return AffineBinaryOpExprClass::getAdd(expr, v);
}
template <> AffineExpr AffineExpr::operator+(AffineExpr other) const {
  return AffineBinaryOpExprClass::getAdd(expr, other.expr);
}
template <> AffineExpr AffineExpr::operator*(int64_t v) const {
  return AffineBinaryOpExprClass::getMul(expr, v);
}
template <> AffineExpr AffineExpr::operator*(AffineExpr other) const {
  return AffineBinaryOpExprClass::getMul(expr, other.expr);
}
// Unary minus, delegate to operator*.
template <> AffineExpr AffineExpr::operator-() const {
  return AffineBinaryOpExprClass::getMul(expr, -1);
}
// Delegate to operator+.
template <> AffineExpr AffineExpr::operator-(int64_t v) const {
  return *this + (-v);
}
template <> AffineExpr AffineExpr::operator-(AffineExpr other) const {
  return *this + (-other);
}
template <> AffineExpr AffineExpr::floorDiv(uint64_t v) const {
  return AffineBinaryOpExprClass::getFloorDiv(expr, v);
}
template <> AffineExpr AffineExpr::floorDiv(AffineExpr other) const {
  return AffineBinaryOpExprClass::getFloorDiv(expr, other.expr);
}
template <> AffineExpr AffineExpr::ceilDiv(uint64_t v) const {
  return AffineBinaryOpExprClass::getCeilDiv(expr, v);
}
template <> AffineExpr AffineExpr::ceilDiv(AffineExpr other) const {
  return AffineBinaryOpExprClass::getCeilDiv(expr, other.expr);
}
template <> AffineExpr AffineExpr::operator%(uint64_t v) const {
  return AffineBinaryOpExprClass::getMod(expr, v);
}
template <> AffineExpr AffineExpr::operator%(AffineExpr other) const {
  return AffineBinaryOpExprClass::getMod(expr, other.expr);
}

AffineExpr operator+(int64_t val, AffineExpr expr) {
  return expr + val; // AffineBinaryOpExpr asserts !lhs.isa<AffineConstantExpr>
}
AffineExpr operator-(int64_t val, AffineExpr expr) { return expr * (-1) + val; }
AffineExpr operator*(int64_t val, AffineExpr expr) {
  return expr * val; // AffineBinaryOpExpr asserts !lhs.isa<AffineConstantExpr>
}

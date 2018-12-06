//===- AffineExpr.h - MLIR Affine Expr Class --------------------*- C++ -*-===//
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
//
// An affine expression is an affine combination of dimension identifiers and
// symbols, including ceildiv/floordiv/mod by a constant integer.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_AFFINE_EXPR_H
#define MLIR_IR_AFFINE_EXPR_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/Casting.h"
#include <type_traits>

namespace mlir {

class MLIRContext;

namespace detail {

class AffineExprStorage;
class AffineBinaryOpExprStorage;
class AffineDimExprStorage;
class AffineSymbolExprStorage;
class AffineConstantExprStorage;

} // namespace detail

enum class AffineExprKind {
  Add,
  /// RHS of mul is always a constant or a symbolic expression.
  Mul,
  /// RHS of mod is always a constant or a symbolic expression.
  Mod,
  /// RHS of floordiv is always a constant or a symbolic expression.
  FloorDiv,
  /// RHS of ceildiv is always a constant or a symbolic expression.
  CeilDiv,

  /// This is a marker for the last affine binary op. The range of binary
  /// op's is expected to be this element and earlier.
  LAST_AFFINE_BINARY_OP = CeilDiv,

  /// Constant integer.
  Constant,
  /// Dimensional identifier.
  DimId,
  /// Symbolic identifier.
  SymbolId,
};

/// Base type for affine expression.
/// AffineExpr's are immutable value types with intuitive operators to
/// operate on chainable, lightweight compositions.
/// An AffineExpr is a POD interface to the underlying storage type pointer.
class AffineExpr {
public:
  using ImplType = detail::AffineExprStorage;

  AffineExpr() : expr(nullptr) {}
  /* implicit */ AffineExpr(const ImplType *expr)
      : expr(const_cast<ImplType *>(expr)) {}

  AffineExpr(const AffineExpr &other) : expr(other.expr) {}
  AffineExpr &operator=(AffineExpr other) {
    expr = other.expr;
    return *this;
  }

  bool operator==(AffineExpr other) const { return expr == other.expr; }
  bool operator!=(AffineExpr other) const { return !(*this == other); }
  explicit operator bool() const { return expr; }

  bool operator!() const { return expr == nullptr; }

  template <typename U> bool isa() const;
  template <typename U> U dyn_cast() const;
  template <typename U> U cast() const;

  MLIRContext *getContext() const;

  /// Return the classification for this type.
  AffineExprKind getKind() const;

  void print(raw_ostream &os) const;
  void dump() const;

  /// Returns true if this expression is made out of only symbols and
  /// constants, i.e., it does not involve dimensional identifiers.
  bool isSymbolicOrConstant() const;

  /// Returns true if this is a pure affine expression, i.e., multiplication,
  /// floordiv, ceildiv, and mod is only allowed w.r.t constants.
  bool isPureAffine() const;

  /// Returns the greatest known integral divisor of this affine expression.
  uint64_t getLargestKnownDivisor() const;

  /// Return true if the affine expression is a multiple of 'factor'.
  bool isMultipleOf(int64_t factor) const;

  /// Return true if the affine expression involves AffineDimExpr `position`.
  bool isFunctionOfDim(unsigned position) const;

  AffineExpr operator+(int64_t v) const;
  AffineExpr operator+(AffineExpr other) const;
  AffineExpr operator-() const;
  AffineExpr operator-(int64_t v) const;
  AffineExpr operator-(AffineExpr other) const;
  AffineExpr operator*(int64_t v) const;
  AffineExpr operator*(AffineExpr other) const;
  AffineExpr floorDiv(uint64_t v) const;
  AffineExpr floorDiv(AffineExpr other) const;
  AffineExpr ceilDiv(uint64_t v) const;
  AffineExpr ceilDiv(AffineExpr other) const;
  AffineExpr operator%(uint64_t v) const;
  AffineExpr operator%(AffineExpr other) const;

  friend ::llvm::hash_code hash_value(AffineExpr arg);

protected:
  ImplType *expr;
};

/// Affine binary operation expression. An affine binary operation could be an
/// add, mul, floordiv, ceildiv, or a modulo operation. (Subtraction is
/// represented through a multiply by -1 and add.) These expressions are always
/// constructed in a simplified form. For eg., the LHS and RHS operands can't
/// both be constants. There are additional canonicalizing rules depending on
/// the op type: see checks in the constructor.
class AffineBinaryOpExpr : public AffineExpr {
public:
  using ImplType = detail::AffineBinaryOpExprStorage;
  /* implicit */ AffineBinaryOpExpr(AffineExpr::ImplType *ptr);
  AffineExpr getLHS() const;
  AffineExpr getRHS() const;
};

/// A dimensional identifier appearing in an affine expression.
class AffineDimExpr : public AffineExpr {
public:
  using ImplType = detail::AffineDimExprStorage;
  /* implicit */ AffineDimExpr(AffineExpr::ImplType *ptr);
  unsigned getPosition() const;
};

/// A symbolic identifier appearing in an affine expression.
class AffineSymbolExpr : public AffineExpr {
public:
  using ImplType = detail::AffineSymbolExprStorage;
  /* implicit */ AffineSymbolExpr(AffineExpr::ImplType *ptr);
  unsigned getPosition() const;
};

/// An integer constant appearing in affine expression.
class AffineConstantExpr : public AffineExpr {
public:
  using ImplType = detail::AffineConstantExprStorage;
  /* implicit */ AffineConstantExpr(AffineExpr::ImplType *ptr);
  int64_t getValue() const;
};

/// Make AffineExpr hashable.
inline ::llvm::hash_code hash_value(AffineExpr arg) {
  return ::llvm::hash_value(arg.expr);
}

inline AffineExpr operator+(int64_t val, AffineExpr expr) { return expr + val; }
inline AffineExpr operator*(int64_t val, AffineExpr expr) { return expr * val; }
inline AffineExpr operator-(int64_t val, AffineExpr expr) {
  return expr * (-1) + val;
}

/// These free functions allow clients of the API to not use classes in detail.
AffineExpr getAffineDimExpr(unsigned position, MLIRContext *context);
AffineExpr getAffineSymbolExpr(unsigned position, MLIRContext *context);
AffineExpr getAffineConstantExpr(int64_t constant, MLIRContext *context);

/// This auxiliary free function allows conveniently capturing the LHS, RHS and
/// AffineExprBinaryOp in an AffineBinaryOpExpr.
/// In particular it is used to elegantly write compositions as such:
/// ```c++
/// AffineMap g = /* Some affine map */;
/// if (auto binExpr = e.template dyn_cast<AffineBinaryOpExpr>()) {
///   AffineExpr lhs, rhs;
///   AffineExprBinaryOp binOp;
///   std::tie(lhs, rhs, binOp) = matchBinaryOpExpr(binExpr);
///   return binOp(compose(lhs, g), compose(rhs, g));
/// }
/// ```
using AffineExprBinaryOp = std::function<AffineExpr(AffineExpr, AffineExpr)>;
std::tuple<AffineExpr, AffineExpr, AffineExprBinaryOp>
matchBinaryOpExpr(AffineBinaryOpExpr e);

raw_ostream &operator<<(raw_ostream &os, AffineExpr &expr);

template <typename U> bool AffineExpr::isa() const {
  if (std::is_same<U, AffineBinaryOpExpr>::value) {
    return getKind() <= AffineExprKind::LAST_AFFINE_BINARY_OP;
  }
  if (std::is_same<U, AffineDimExpr>::value) {
    return getKind() == AffineExprKind::DimId;
  }
  if (std::is_same<U, AffineSymbolExpr>::value) {
    return getKind() == AffineExprKind::SymbolId;
  }
  if (std::is_same<U, AffineConstantExpr>::value) {
    return getKind() == AffineExprKind::Constant;
  }
}
template <typename U> U AffineExpr::dyn_cast() const {
  if (isa<U>()) {
    return U(expr);
  }
  return U(nullptr);
}
template <typename U> U AffineExpr::cast() const {
  assert(isa<U>());
  return U(expr);
}

} // namespace mlir

namespace llvm {

// AffineExpr hash just like pointers
template <> struct DenseMapInfo<mlir::AffineExpr> {
  static mlir::AffineExpr getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::AffineExpr(static_cast<mlir::AffineExpr::ImplType *>(pointer));
  }
  static mlir::AffineExpr getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::AffineExpr(static_cast<mlir::AffineExpr::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::AffineExpr val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::AffineExpr LHS, mlir::AffineExpr RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // MLIR_IR_AFFINE_EXPR_H

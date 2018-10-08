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

class AffineExprClass;
class AffineBinaryOpExprClass;
class AffineDimExprClass;
class AffineSymbolExprClass;
class AffineConstantExprClass;

} // namespace detail

enum class AffineExprKind {
  Add,
  // RHS of mul is always a constant or a symbolic expression.
  Mul,
  // RHS of mod is always a constant or a symbolic expression.
  Mod,
  // RHS of floordiv is always a constant or a symbolic expression.
  FloorDiv,
  // RHS of ceildiv is always a constant or a symbolic expression.
  CeilDiv,

  /// This is a marker for the last affine binary op. The range of binary
  /// op's is expected to be this element and earlier.
  LAST_AFFINE_BINARY_OP = CeilDiv,

  // Constant integer.
  Constant,
  // Dimensional identifier.
  DimId,
  // Symbolic identifier.
  SymbolId,
};

/// Helper structure to build AffineExprClass with intuitive operators in order
/// to operate on chainable, lightweight, immutable value types instead of
/// pointer types.
/// TODO(ntv): Add extra out-of-class operators for int op AffineExprBase
/// TODO(ntv): pointer pair
template <typename AffineExprType> class AffineExprBase {
public:
  typedef AffineExprBase TemplateType;
  typedef AffineExprType ImplType;

  AffineExprBase() : expr(nullptr) {}
  /* implicit */ AffineExprBase(const AffineExprType *expr)
      : expr(const_cast<AffineExprType *>(expr)) {}

  AffineExprBase(const AffineExprBase &other) : expr(other.expr) {}
  AffineExprBase &operator=(AffineExprBase other) {
    expr = other.expr;
    return *this;
  }

  bool operator==(AffineExprBase other) const { return expr == other.expr; }

  explicit operator AffineExprType *() const {
    return const_cast<AffineExprType *>(expr);
  }
  /* implicit */ operator AffineExprBase<detail::AffineExprClass>() const {
    return const_cast<detail::AffineExprClass *>(
        static_cast<const detail::AffineExprClass *>(expr));
  }
  explicit operator bool() const { return expr; }

  bool operator!() const { return expr == nullptr; }
  AffineExprType *operator->() const { return expr; }

  template <typename U> bool isa() const {
    using PtrType = typename U::ImplType;
    return llvm::isa<PtrType>(const_cast<AffineExprType *>(this->expr));
  }
  template <typename U> U dyn_cast() const {
    using PtrType = typename U::ImplType;
    return U(llvm::dyn_cast<PtrType>(const_cast<AffineExprType *>(this->expr)));
  }
  template <typename U> U cast() const {
    using PtrType = typename U::ImplType;
    return U(llvm::cast<PtrType>(const_cast<AffineExprType *>(this->expr)));
  }

  AffineExprBase operator+(int64_t v) const;
  AffineExprBase operator+(AffineExprBase other) const;
  AffineExprBase operator-() const;
  AffineExprBase operator-(int64_t v) const;
  AffineExprBase operator-(AffineExprBase other) const;
  AffineExprBase operator*(int64_t v) const;
  AffineExprBase operator*(AffineExprBase other) const;
  AffineExprBase floorDiv(uint64_t v) const;
  AffineExprBase floorDiv(AffineExprBase other) const;
  AffineExprBase ceilDiv(uint64_t v) const;
  AffineExprBase ceilDiv(AffineExprBase other) const;
  AffineExprBase operator%(uint64_t v) const;
  AffineExprBase operator%(AffineExprBase other) const;

  friend ::llvm::hash_code hash_value(AffineExprBase arg);

private:
  AffineExprType *expr;
};

using AffineExpr = AffineExprBase<detail::AffineExprClass>;
using AffineBinaryOpExpr = AffineExprBase<detail::AffineBinaryOpExprClass>;
using AffineDimExpr = AffineExprBase<detail::AffineDimExprClass>;
using AffineSymbolExpr = AffineExprBase<detail::AffineSymbolExprClass>;
using AffineConstantExpr = AffineExprBase<detail::AffineConstantExprClass>;

AffineExpr operator+(int64_t val, AffineExpr expr);
AffineExpr operator-(int64_t val, AffineExpr expr);
AffineExpr operator*(int64_t val, AffineExpr expr);

// Make AffineExpr hashable.
inline ::llvm::hash_code hash_value(AffineExpr arg) {
  return ::llvm::hash_value(static_cast<detail::AffineExprClass *>(arg.expr));
}

// These free functions allow clients of the API to not use classes in detail.
AffineExpr getAffineDimExpr(unsigned position, MLIRContext *context);
AffineExpr getAffineSymbolExpr(unsigned position, MLIRContext *context);
AffineExpr getAffineConstantExpr(int64_t constant, MLIRContext *context);

namespace detail {

/// A one-dimensional affine expression.
/// AffineExpression's are immutable (like Type's)
class AffineExprClass {
public:
  /// Return the classification for this type.
  AffineExprKind getKind() { return kind; }

  void print(raw_ostream &os);
  void dump();

  /// Returns true if this expression is made out of only symbols and
  /// constants, i.e., it does not involve dimensional identifiers.
  bool isSymbolicOrConstant();

  /// Returns true if this is a pure affine expression, i.e., multiplication,
  /// floordiv, ceildiv, and mod is only allowed w.r.t constants.
  bool isPureAffine();

  /// Returns the greatest known integral divisor of this affine expression.
  uint64_t getLargestKnownDivisor();

  /// Return true if the affine expression is a multiple of 'factor'.
  bool isMultipleOf(int64_t factor);

  MLIRContext *getContext();

protected:
  explicit AffineExprClass(AffineExprKind kind, MLIRContext *context)
      : kind(kind), context(context) {}
  ~AffineExprClass() {}

private:
  AffineExprClass(const AffineExprClass &) = delete;
  void operator=(const AffineExprClass &) = delete;

  /// Classification of the subclass
  const AffineExprKind kind;
  MLIRContext *context;
};

inline raw_ostream &operator<<(raw_ostream &os, AffineExpr &expr) {
  expr->print(os);
  return os;
}

/// Affine binary operation expression. An affine binary operation could be an
/// add, mul, floordiv, ceildiv, or a modulo operation. (Subtraction is
/// represented through a multiply by -1 and add.) These expressions are always
/// constructed in a simplified form. For eg., the LHS and RHS operands can't
/// both be constants. There are additional canonicalizing rules depending on
/// the op type: see checks in the constructor.
class AffineBinaryOpExprClass : public AffineExprClass {
public:
  static AffineExpr get(AffineExprKind kind, AffineExpr lhs, AffineExpr rhs);
  static AffineExpr getAdd(AffineExpr lhs, AffineExpr rhs) {
    return get(AffineExprKind::Add, lhs, rhs);
  }
  static AffineExpr getAdd(AffineExpr expr, int64_t rhs);
  static AffineExpr getSub(AffineExpr lhs, AffineExpr rhs);

  static AffineExpr getMul(AffineExpr lhs, AffineExpr rhs) {
    return get(AffineExprKind::Mul, lhs, rhs);
  }
  static AffineExpr getMul(AffineExpr expr, int64_t rhs);
  static AffineExpr getFloorDiv(AffineExpr lhs, AffineExpr rhs) {
    return get(AffineExprKind::FloorDiv, lhs, rhs);
  }
  static AffineExpr getFloorDiv(AffineExpr lhs, uint64_t rhs);
  static AffineExpr getCeilDiv(AffineExpr lhs, AffineExpr rhs) {
    return get(AffineExprKind::CeilDiv, lhs, rhs);
  }
  static AffineExpr getCeilDiv(AffineExpr lhs, uint64_t rhs);
  static AffineExpr getMod(AffineExpr lhs, AffineExpr rhs) {
    return get(AffineExprKind::Mod, lhs, rhs);
  }
  static AffineExpr getMod(AffineExpr lhs, uint64_t rhs);

  AffineExpr getLHS() { return lhs; }
  AffineExpr getRHS() { return rhs; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExprClass *expr) {
    return const_cast<AffineExprClass *>(expr)->getKind() <=
           AffineExprKind::LAST_AFFINE_BINARY_OP;
  }

protected:
  explicit AffineBinaryOpExprClass(AffineExprKind kind, AffineExpr lhs,
                                   AffineExpr rhs);

  const AffineExpr lhs;
  const AffineExpr rhs;

private:
  ~AffineBinaryOpExprClass() = delete;
};

/// A dimensional identifier appearing in an affine expression.
///
/// This is a POD type of int size; so it should be passed around by
/// value.  The underlying data is owned by MLIRContext and is thus immortal for
/// almost all clients.
class AffineDimExprClass : public AffineExprClass {
public:
  static AffineExprBase<AffineExprClass> get(unsigned position,
                                             MLIRContext *context);

  unsigned getPosition() { return position; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExprClass *expr) {
    return const_cast<AffineExprClass *>(expr)->getKind() ==
           AffineExprKind::DimId;
  }

  friend AffineExpr mlir::getAffineDimExpr(unsigned position,
                                           MLIRContext *context);

private:
  ~AffineDimExprClass() = delete;
  explicit AffineDimExprClass(unsigned position, MLIRContext *context)
      : AffineExprClass(AffineExprKind::DimId, context), position(position) {}

  /// Position of this identifier in the argument list.
  unsigned position;
};

/// A symbolic identifier appearing in an affine expression.
//
/// This is a POD type of int size, so it should be passed around by
/// value.  The underlying data is owned by MLIRContext and is thus immortal for
/// almost all clients.
class AffineSymbolExprClass : public AffineExprClass {
public:
  static AffineExprBase<AffineExprClass> get(unsigned position,
                                             MLIRContext *context);

  unsigned getPosition() { return position; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExprClass *expr) {
    return const_cast<AffineExprClass *>(expr)->getKind() ==
           AffineExprKind::SymbolId;
  }

  friend AffineExpr mlir::getAffineSymbolExpr(unsigned position,
                                              MLIRContext *context);

private:
  ~AffineSymbolExprClass() = delete;
  explicit AffineSymbolExprClass(unsigned position, MLIRContext *context)
      : AffineExprClass(AffineExprKind::SymbolId, context), position(position) {
  }

  /// Position of this identifier in the symbol list.
  unsigned position;
};

/// An integer constant appearing in affine expression.
class AffineConstantExprClass : public AffineExprClass {
public:
  static AffineExprBase<AffineExprClass> get(int64_t constant,
                                             MLIRContext *context);

  int64_t getValue() { return constant; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const AffineExprClass *expr) {
    return const_cast<AffineExprClass *>(expr)->getKind() ==
           AffineExprKind::Constant;
  }

  friend AffineExpr mlir::getAffineConstantExpr(int64_t constant,
                                                MLIRContext *context);

private:
  ~AffineConstantExprClass() = delete;
  explicit AffineConstantExprClass(int64_t constant, MLIRContext *context)
      : AffineExprClass(AffineExprKind::Constant, context), constant(constant) {
  }

  // The constant.
  int64_t constant;
};

} // end namespace detail
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

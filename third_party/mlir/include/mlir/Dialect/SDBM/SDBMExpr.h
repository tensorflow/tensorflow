//===- SDBMExpr.h - MLIR SDBM Expression ------------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A striped difference-bound matrix (SDBM) expression is a constant expression,
// an identifier, a binary expression with constant RHS and +, stripe operators
// or a difference expression between two identifiers.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SDBM_SDBMEXPR_H
#define MLIR_DIALECT_SDBM_SDBMEXPR_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace mlir {

class AffineExpr;
class MLIRContext;

enum class SDBMExprKind { Add, Stripe, Diff, Constant, DimId, SymbolId, Neg };

namespace detail {
struct SDBMExprStorage;
struct SDBMBinaryExprStorage;
struct SDBMDiffExprStorage;
struct SDBMTermExprStorage;
struct SDBMConstantExprStorage;
struct SDBMNegExprStorage;
} // namespace detail

class SDBMConstantExpr;
class SDBMDialect;
class SDBMDimExpr;
class SDBMSymbolExpr;
class SDBMTermExpr;

/// Striped Difference-Bounded Matrix (SDBM) expression is a base left-hand side
/// expression for the SDBM framework.  SDBM expressions are a subset of affine
/// expressions supporting low-complexity algorithms for the operations used in
/// loop transformations.  In particular, are supported:
///   - constant expressions;
///   - single variables (dimensions and symbols) with +1 or -1 coefficient;
///   - stripe expressions: "x # C", where "x" is a single variable or another
///     stripe expression, "#" is the stripe operator, and "C" is a constant
///     expression; "#" is defined as x - x mod C.
///   - sum expressions between single variable/stripe expressions and constant
///     expressions;
///   - difference expressions between single variable/stripe expressions.
/// `SDBMExpr` class hierarchy provides a type-safe interface to constructing
/// and operating on SDBM expressions.  For example, it requires the LHS of a
/// sum expression to be a single variable or a stripe expression.  These
/// restrictions are intended to force the caller to perform the necessary
/// simplifications to stay within the SDBM domain, because SDBM expressions do
/// not combine in more cases than they do.  This choice may be reconsidered in
/// the future.
///
/// SDBM expressions are grouped into the following structure
/// - expression
///   - varying
///     - direct
///       - sum <- (term, constant)
///       - term
///         - symbol
///         - dimension
///         - stripe <- (direct, constant)
///     - negation <- (direct)
///     - difference <- (direct, term)
///   - constant
/// The notation <- (...) denotes the types of subexpressions a compound
/// expression can combine.  The tree of subexpressions essentially imposes the
/// following canonicalization rules:
///   - constants are always folded;
///   - constants can only appear on the RHS of an expression;
///   - double negation must be elided;
///   - an additive constant term is only allowed in a sum expression, and
///     should be sunk into the nearest such expression in the tree;
///   - zero constant expression can only appear at the top level.
///
/// `SDBMExpr` and derived classes are thin wrappers around a pointer owned by
/// an MLIRContext, and should be used by-value.  They are uniqued in the
/// MLIRContext and immortal.
class SDBMExpr {
public:
  using ImplType = detail::SDBMExprStorage;
  SDBMExpr() : impl(nullptr) {}
  /* implicit */ SDBMExpr(ImplType *expr) : impl(expr) {}

  /// SDBM expressions are thin wrappers around a unique'ed immutable pointer,
  /// which makes them trivially assignable and trivially copyable.
  SDBMExpr(const SDBMExpr &) = default;
  SDBMExpr &operator=(const SDBMExpr &) = default;

  /// SDBM expressions can be compared straight-forwardly.
  bool operator==(const SDBMExpr &other) const { return impl == other.impl; }
  bool operator!=(const SDBMExpr &other) const { return !(*this == other); }

  /// SDBM expressions are convertible to `bool`: null expressions are converted
  /// to false, non-null expressions are converted to true.
  explicit operator bool() const { return impl != nullptr; }
  bool operator!() const { return !static_cast<bool>(*this); }

  /// Negate the given SDBM expression.
  SDBMExpr operator-();

  /// Prints the SDBM expression.
  void print(raw_ostream &os) const;
  void dump() const;

  /// LLVM-style casts.
  template <typename U> bool isa() const { return U::isClassFor(*this); }
  template <typename U> U dyn_cast() const {
    if (!isa<U>())
      return {};
    return U(const_cast<SDBMExpr *>(this)->impl);
  }
  template <typename U> U cast() const {
    assert(isa<U>() && "cast to incorrect subtype");
    return U(const_cast<SDBMExpr *>(this)->impl);
  }

  /// Support for LLVM hashing.
  ::llvm::hash_code hash_value() const { return ::llvm::hash_value(impl); }

  /// Returns the kind of the SDBM expression.
  SDBMExprKind getKind() const;

  /// Returns the MLIR context in which this expression lives.
  MLIRContext *getContext() const;

  /// Returns the SDBM dialect instance.
  SDBMDialect *getDialect() const;

  /// Convert the SDBM expression into an Affine expression.  This always
  /// succeeds because SDBM are a subset of affine.
  AffineExpr getAsAffineExpr() const;

  /// Try constructing an SDBM expression from the given affine expression.
  /// This may fail if the affine expression is not representable as SDBM, in
  /// which case llvm::None is returned.  The conversion procedure recognizes
  /// (nested) multiplicative ((x floordiv B) * B) and additive (x - x mod B)
  /// patterns for the stripe expression.
  static Optional<SDBMExpr> tryConvertAffineExpr(AffineExpr affine);

protected:
  ImplType *impl;
};

/// SDBM constant expression, wraps a 64-bit integer.
class SDBMConstantExpr : public SDBMExpr {
public:
  using ImplType = detail::SDBMConstantExprStorage;

  using SDBMExpr::SDBMExpr;

  /// Obtain or create a constant expression unique'ed in the given dialect
  /// (which belongs to a context).
  static SDBMConstantExpr get(SDBMDialect *dialect, int64_t value);

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::Constant;
  }

  int64_t getValue() const;
};

/// SDBM varying expression can be one of:
///   - input variable expression;
///   - stripe expression;
///   - negation (product with -1) of either of the above.
///   - sum of a varying and a constant expression
///   - difference between varying expressions
class SDBMVaryingExpr : public SDBMExpr {
public:
  using ImplType = detail::SDBMExprStorage;
  using SDBMExpr::SDBMExpr;

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::DimId ||
           expr.getKind() == SDBMExprKind::SymbolId ||
           expr.getKind() == SDBMExprKind::Neg ||
           expr.getKind() == SDBMExprKind::Stripe ||
           expr.getKind() == SDBMExprKind::Add ||
           expr.getKind() == SDBMExprKind::Diff;
  }
};

/// SDBM direct expression includes exactly one variable (symbol or dimension),
/// which is not negated in the expression.  It can be one of:
///   - term expression;
///   - sum expression.
class SDBMDirectExpr : public SDBMVaryingExpr {
public:
  using SDBMVaryingExpr::SDBMVaryingExpr;

  /// If this is a sum expression, return its variable part, otherwise return
  /// self.
  SDBMTermExpr getTerm();

  /// If this is a sum expression, return its constant part, otherwise return 0.
  int64_t getConstant();

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::DimId ||
           expr.getKind() == SDBMExprKind::SymbolId ||
           expr.getKind() == SDBMExprKind::Stripe ||
           expr.getKind() == SDBMExprKind::Add;
  }
};

/// SDBM term expression can be one of:
///  - single variable expression;
///  - stripe expression.
/// Stripe expressions are treated as terms since, in the SDBM domain, they are
/// attached to temporary variables and can appear anywhere a variable can.
class SDBMTermExpr : public SDBMDirectExpr {
public:
  using SDBMDirectExpr::SDBMDirectExpr;

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::DimId ||
           expr.getKind() == SDBMExprKind::SymbolId ||
           expr.getKind() == SDBMExprKind::Stripe;
  }
};

/// SDBM sum expression.  LHS is a term expression and RHS is a constant.
class SDBMSumExpr : public SDBMDirectExpr {
public:
  using ImplType = detail::SDBMBinaryExprStorage;
  using SDBMDirectExpr::SDBMDirectExpr;

  /// Obtain or create a sum expression unique'ed in the given context.
  static SDBMSumExpr get(SDBMTermExpr lhs, SDBMConstantExpr rhs);

  static bool isClassFor(const SDBMExpr &expr) {
    SDBMExprKind kind = expr.getKind();
    return kind == SDBMExprKind::Add;
  }

  SDBMTermExpr getLHS() const;
  SDBMConstantExpr getRHS() const;
};

/// SDBM difference expression.  LHS is a direct expression, i.e. it may be a
/// sum of a term and a constant.  RHS is a term expression.  Thus the
/// expression (t1 - t2 + C) with term expressions t1,t2 is represented as
///   diff(sum(t1, C), t2)
/// and it is possible to extract the constant factor without negating it.
class SDBMDiffExpr : public SDBMVaryingExpr {
public:
  using ImplType = detail::SDBMDiffExprStorage;
  using SDBMVaryingExpr::SDBMVaryingExpr;

  /// Obtain or create a difference expression unique'ed in the given context.
  static SDBMDiffExpr get(SDBMDirectExpr lhs, SDBMTermExpr rhs);

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::Diff;
  }

  SDBMDirectExpr getLHS() const;
  SDBMTermExpr getRHS() const;
};

/// SDBM stripe expression "x # C" where "x" is a term expression, "C" is a
/// constant expression and "#" is the stripe operator defined as:
///   x # C = x - x mod C.
class SDBMStripeExpr : public SDBMTermExpr {
public:
  using ImplType = detail::SDBMBinaryExprStorage;
  using SDBMTermExpr::SDBMTermExpr;

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::Stripe;
  }

  static SDBMStripeExpr get(SDBMDirectExpr var, SDBMConstantExpr stripeFactor);

  SDBMDirectExpr getLHS() const;
  SDBMConstantExpr getStripeFactor() const;
};

/// SDBM "input" variable expression can be either a dimension identifier or
/// a symbol identifier.  When used to define SDBM functions, dimensions are
/// interpreted as function arguments while symbols are treated as unknown but
/// constant values, hence the name.
class SDBMInputExpr : public SDBMTermExpr {
public:
  using ImplType = detail::SDBMTermExprStorage;
  using SDBMTermExpr::SDBMTermExpr;

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::DimId ||
           expr.getKind() == SDBMExprKind::SymbolId;
  }

  unsigned getPosition() const;
};

/// SDBM dimension expression.  Dimensions correspond to function arguments
/// when defining functions using SDBM expressions.
class SDBMDimExpr : public SDBMInputExpr {
public:
  using ImplType = detail::SDBMTermExprStorage;
  using SDBMInputExpr::SDBMInputExpr;

  /// Obtain or create a dimension expression unique'ed in the given dialect
  /// (which belongs to a context).
  static SDBMDimExpr get(SDBMDialect *dialect, unsigned position);

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::DimId;
  }
};

/// SDBM symbol expression.  Symbols correspond to symbolic constants when
/// defining functions using SDBM expressions.
class SDBMSymbolExpr : public SDBMInputExpr {
public:
  using ImplType = detail::SDBMTermExprStorage;
  using SDBMInputExpr::SDBMInputExpr;

  /// Obtain or create a symbol expression unique'ed in the given dialect (which
  /// belongs to a context).
  static SDBMSymbolExpr get(SDBMDialect *dialect, unsigned position);

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::SymbolId;
  }
};

/// Negation of an SDBM variable expression.  Equivalent to multiplying the
/// expression with -1 (SDBM does not support other coefficients that 1 and -1).
class SDBMNegExpr : public SDBMVaryingExpr {
public:
  using ImplType = detail::SDBMNegExprStorage;
  using SDBMVaryingExpr::SDBMVaryingExpr;

  /// Obtain or create a negation expression unique'ed in the given context.
  static SDBMNegExpr get(SDBMDirectExpr var);

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::Neg;
  }

  SDBMDirectExpr getVar() const;
};

/// A visitor class for SDBM expressions.  Calls the kind-specific function
/// depending on the kind of expression it visits.
template <typename Derived, typename Result = void> class SDBMVisitor {
public:
  /// Visit the given SDBM expression, dispatching to kind-specific functions.
  Result visit(SDBMExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    switch (expr.getKind()) {
    case SDBMExprKind::Add:
    case SDBMExprKind::Diff:
    case SDBMExprKind::DimId:
    case SDBMExprKind::SymbolId:
    case SDBMExprKind::Neg:
    case SDBMExprKind::Stripe:
      return derived->visitVarying(expr.cast<SDBMVaryingExpr>());
    case SDBMExprKind::Constant:
      return derived->visitConstant(expr.cast<SDBMConstantExpr>());
    }

    llvm_unreachable("unsupported SDBM expression kind");
  }

  /// Traverse the SDBM expression tree calling `visit` on each node
  /// in depth-first preorder.
  void walkPreorder(SDBMExpr expr) { return walk</*isPreorder=*/true>(expr); }

  /// Traverse the SDBM expression tree calling `visit` on each node in
  /// depth-first postorder.
  void walkPostorder(SDBMExpr expr) { return walk</*isPreorder=*/false>(expr); }

protected:
  /// Default visitors do nothing.
  void visitSum(SDBMSumExpr) {}
  void visitDiff(SDBMDiffExpr) {}
  void visitStripe(SDBMStripeExpr) {}
  void visitDim(SDBMDimExpr) {}
  void visitSymbol(SDBMSymbolExpr) {}
  void visitNeg(SDBMNegExpr) {}
  void visitConstant(SDBMConstantExpr) {}

  /// Default implementation of visitDirect dispatches to the dedicated for sums
  /// or delegates to visitTerm for the other expression kinds.  Concrete
  /// visitors can overload it.
  Result visitDirect(SDBMDirectExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    if (auto sum = expr.dyn_cast<SDBMSumExpr>())
      return derived->visitSum(sum);
    else
      return derived->visitTerm(expr.cast<SDBMTermExpr>());
  }

  /// Default implementation of visitTerm dispatches to the special functions
  /// for stripes and other variables.  Concrete visitors can override it.
  Result visitTerm(SDBMTermExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    if (expr.getKind() == SDBMExprKind::Stripe)
      return derived->visitStripe(expr.cast<SDBMStripeExpr>());
    else
      return derived->visitInput(expr.cast<SDBMInputExpr>());
  }

  /// Default implementation of visitInput dispatches to the special
  /// functions for dimensions or symbols.  Concrete visitors can override it to
  /// visit all variables instead.
  Result visitInput(SDBMInputExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    if (expr.getKind() == SDBMExprKind::DimId)
      return derived->visitDim(expr.cast<SDBMDimExpr>());
    else
      return derived->visitSymbol(expr.cast<SDBMSymbolExpr>());
  }

  /// Default implementation of visitVarying dispatches to the special
  /// functions for variables and negations thereof.  Concrete visitors can
  /// override it to visit all variables and negations instead.
  Result visitVarying(SDBMVaryingExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    if (auto var = expr.dyn_cast<SDBMDirectExpr>())
      return derived->visitDirect(var);
    else if (auto neg = expr.dyn_cast<SDBMNegExpr>())
      return derived->visitNeg(neg);
    else if (auto diff = expr.dyn_cast<SDBMDiffExpr>())
      return derived->visitDiff(diff);

    llvm_unreachable("unhandled subtype of varying SDBM expression");
  }

  template <bool isPreorder> void walk(SDBMExpr expr) {
    if (isPreorder)
      visit(expr);
    if (auto sumExpr = expr.dyn_cast<SDBMSumExpr>()) {
      walk<isPreorder>(sumExpr.getLHS());
      walk<isPreorder>(sumExpr.getRHS());
    } else if (auto diffExpr = expr.dyn_cast<SDBMDiffExpr>()) {
      walk<isPreorder>(diffExpr.getLHS());
      walk<isPreorder>(diffExpr.getRHS());
    } else if (auto stripeExpr = expr.dyn_cast<SDBMStripeExpr>()) {
      walk<isPreorder>(stripeExpr.getLHS());
      walk<isPreorder>(stripeExpr.getStripeFactor());
    } else if (auto negExpr = expr.dyn_cast<SDBMNegExpr>()) {
      walk<isPreorder>(negExpr.getVar());
    }
    if (!isPreorder)
      visit(expr);
  }
};

/// Overloaded arithmetic operators for SDBM expressions asserting that their
/// arguments have the proper SDBM expression subtype.  Perform canonicalization
/// and constant folding on these expressions.
namespace ops_assertions {

/// Add two SDBM expressions.  At least one of the expressions must be a
/// constant or a negation, but both expressions cannot be negations
/// simultaneously.
SDBMExpr operator+(SDBMExpr lhs, SDBMExpr rhs);
inline SDBMExpr operator+(SDBMExpr lhs, int64_t rhs) {
  return lhs + SDBMConstantExpr::get(lhs.getDialect(), rhs);
}
inline SDBMExpr operator+(int64_t lhs, SDBMExpr rhs) {
  return SDBMConstantExpr::get(rhs.getDialect(), lhs) + rhs;
}

/// Subtract an SDBM expression from another SDBM expression.  Both expressions
/// must not be difference expressions.
SDBMExpr operator-(SDBMExpr lhs, SDBMExpr rhs);
inline SDBMExpr operator-(SDBMExpr lhs, int64_t rhs) {
  return lhs - SDBMConstantExpr::get(lhs.getDialect(), rhs);
}
inline SDBMExpr operator-(int64_t lhs, SDBMExpr rhs) {
  return SDBMConstantExpr::get(rhs.getDialect(), lhs) - rhs;
}

/// Construct a stripe expression from a positive expression and a positive
/// constant stripe factor.
SDBMExpr stripe(SDBMExpr expr, SDBMExpr factor);
inline SDBMExpr stripe(SDBMExpr expr, int64_t factor) {
  return stripe(expr, SDBMConstantExpr::get(expr.getDialect(), factor));
}
} // namespace ops_assertions

} // end namespace mlir

namespace llvm {
// SDBMExpr hash just like pointers.
template <> struct DenseMapInfo<mlir::SDBMExpr> {
  static mlir::SDBMExpr getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::SDBMExpr(static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static mlir::SDBMExpr getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::SDBMExpr(static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::SDBMExpr expr) {
    return expr.hash_value();
  }
  static bool isEqual(mlir::SDBMExpr lhs, mlir::SDBMExpr rhs) {
    return lhs == rhs;
  }
};

// SDBMDirectExpr hash just like pointers.
template <> struct DenseMapInfo<mlir::SDBMDirectExpr> {
  static mlir::SDBMDirectExpr getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::SDBMDirectExpr(
        static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static mlir::SDBMDirectExpr getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::SDBMDirectExpr(
        static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::SDBMDirectExpr expr) {
    return expr.hash_value();
  }
  static bool isEqual(mlir::SDBMDirectExpr lhs, mlir::SDBMDirectExpr rhs) {
    return lhs == rhs;
  }
};

// SDBMTermExpr hash just like pointers.
template <> struct DenseMapInfo<mlir::SDBMTermExpr> {
  static mlir::SDBMTermExpr getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::SDBMTermExpr(static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static mlir::SDBMTermExpr getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::SDBMTermExpr(static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::SDBMTermExpr expr) {
    return expr.hash_value();
  }
  static bool isEqual(mlir::SDBMTermExpr lhs, mlir::SDBMTermExpr rhs) {
    return lhs == rhs;
  }
};

// SDBMConstantExpr hash just like pointers.
template <> struct DenseMapInfo<mlir::SDBMConstantExpr> {
  static mlir::SDBMConstantExpr getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::SDBMConstantExpr(
        static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static mlir::SDBMConstantExpr getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::SDBMConstantExpr(
        static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::SDBMConstantExpr expr) {
    return expr.hash_value();
  }
  static bool isEqual(mlir::SDBMConstantExpr lhs, mlir::SDBMConstantExpr rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // MLIR_DIALECT_SDBM_SDBMEXPR_H

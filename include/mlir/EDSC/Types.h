//===- Types.h - MLIR EDSC Type System --------------------------*- C++ -*-===//
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
// Provides a simple value-based type system to implement an EDSC that
// simplifies emitting MLIR and future MLIR dialects. Most of this should be
// auto-generated in the future.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_EDSC_TYPES_H_
#define MLIR_LIB_EDSC_TYPES_H_

#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"

namespace mlir {

class MLIRContext;

namespace edsc {
namespace detail {

struct ExprStorage;
struct BindableStorage;
struct UnaryExprStorage;
struct BinaryExprStorage;
struct TernaryExprStorage;
struct VariadicExprStorage;

struct StmtStorage;

} // namespace detail

/// EDSC Types closely mirror the core MLIR and uses an abstraction similar to
/// AffineExpr:
///   1. a set of composable structs;
///   2. with by-value semantics everywhere and operator overloading
///   3. with an underlying pointer to impl as payload.
/// The vast majority of this code should be TableGen'd in the future which
/// would allow us to automatically emit an EDSC for any IR dialect we are
/// interested in. In turn this makes any IR dialect fully programmable in a
/// declarative fashion.
///
/// The main differences with the AffineExpr design are as follows:
/// 1. this type-system is an empty shell to which we can lazily bind Value*
///    at the moment of emitting MLIR;
/// 2. the data structures are BumpPointer allocated in a global
///    `ScopedEDSCContext` with scoped lifetime. This allows avoiding to
///    pass and store an extra Context pointer around and keeps users honest:
///    *this is absolutely not meant to escape a local scope*.
///
/// The decision of slicing the underlying IR types into Bindable and
/// NonBindable types is flexible and influences programmability.
enum class ExprKind {
  FIRST_BINDABLE_EXPR = 100,
  Unbound = FIRST_BINDABLE_EXPR,
  LAST_BINDABLE_EXPR = Unbound,
  FIRST_NON_BINDABLE_EXPR = 200,
  FIRST_UNARY_EXPR = FIRST_NON_BINDABLE_EXPR,
  Dealloc = FIRST_UNARY_EXPR,
  Negate,
  LAST_UNARY_EXPR = Negate,
  FIRST_BINARY_EXPR = 300,
  Add = FIRST_BINARY_EXPR,
  Sub,
  Mul,
  Div,
  AddEQ,
  SubEQ,
  MulEQ,
  DivEQ,
  GE,
  GT,
  LE,
  LT,
  EQ,
  NE,
  And,
  Or,
  LAST_BINARY_EXPR = Or,
  FIRST_TERNARY_EXPR = 400,
  Select = FIRST_TERNARY_EXPR,
  IfThenElse,
  LAST_TERNARY_EXPR = IfThenElse,
  FIRST_VARIADIC_EXPR = 500,
  Alloc = FIRST_VARIADIC_EXPR, // Variadic because takes multiple dynamic shape
                               // values.
  Load,
  Store,
  VectorTypeCast, // Variadic because takes a type and anything taking a type
                  // is variadic for now.
  LAST_VARIADIC_EXPR = VectorTypeCast,
  FIRST_STMT_BLOCK_LIKE_EXPR = 600,
  Block = FIRST_STMT_BLOCK_LIKE_EXPR,
  For,
  LAST_STMT_BLOCK_LIKE_EXPR = For,
  LAST_NON_BINDABLE_EXPR = LAST_STMT_BLOCK_LIKE_EXPR,
};

/// Scoped context holding a BumpPtrAllocator.
/// Creating such an object injects a new allocator in Expr::globalAllocator.
/// At the moment we can have only have one such context.
///
/// Usage:
///
/// ```c++
///    MLFunctionBuilder *b = ...;
///    Location someLocation = ...;
///    Value *zeroValue = ...;
///    Value *oneValue = ...;
///
///    ScopedEDSCContext raiiContext;
///    Constant zero, one;
///    Value *val =  MLIREmitter(b)
///         .bind(zero, zeroValue)
///         .bind(one, oneValue)
///         .emit(someLocation, zero + one);
/// ```
///
/// will emit MLIR resembling:
///
/// ```mlir
///    %2 = add(%c0, %c1) : index
/// ```
///
/// The point of the EDSC is to synthesize arbitrarily more complex patterns in
/// a declarative fashion. For example, clipping for guaranteed in-bounds access
/// can be written:
///
/// ```c++
///    auto expr = select(expr < 0, 0, select(expr < size, expr, size - 1));
///    Value *val =  MLIREmitter(b).bind(...).emit(loc, expr);
/// ```
struct ScopedEDSCContext {
  ScopedEDSCContext();
  ~ScopedEDSCContext();
  llvm::BumpPtrAllocator allocator;
};

struct Expr {
public:
  using ImplType = detail::ExprStorage;

  /// Returns the scoped BumpPtrAllocator. This must be done in the context of a
  /// unique `ScopedEDSCContext` declared in an RAII fashion in some enclosing
  /// scope.
  static llvm::BumpPtrAllocator *&globalAllocator() {
    static thread_local llvm::BumpPtrAllocator *allocator = nullptr;
    return allocator;
  }

  Expr() : storage(nullptr) {}
  /* implicit */ Expr(ImplType *storage) : storage(storage) {}

  Expr(const Expr &other) : storage(other.storage) {}
  Expr &operator=(Expr other) {
    storage = other.storage;
    return *this;
  }

  explicit operator bool() { return storage; }
  bool operator!() { return storage == nullptr; }

  template <typename U> bool isa() const;
  template <typename U> U dyn_cast() const;
  template <typename U> U cast() const;

  MLIRContext *getContext() const;

  /// Returns the classification for this type.
  ExprKind getKind() const;

  void print(raw_ostream &os) const;
  void dump() const;

  /// Creates the BinaryExpr corresponding to the operator.
  Expr operator+(Expr other) const;
  Expr operator-(Expr other) const;
  Expr operator*(Expr other) const;
  /// In particular operator==, operator!= return a new Expr and *not* a bool.
  Expr operator==(Expr other) const;
  Expr operator!=(Expr other) const;
  Expr operator<(Expr other) const;
  Expr operator<=(Expr other) const;
  Expr operator>(Expr other) const;
  Expr operator>=(Expr other) const;
  Expr operator&&(Expr other) const;
  Expr operator||(Expr other) const;

  /// For debugging purposes.
  const void *getStoragePtr() const { return storage; }

  friend ::llvm::hash_code hash_value(Expr arg);

protected:
  ImplType *storage;
};

struct Bindable : public Expr {
  using ImplType = detail::BindableStorage;
  friend class Expr;
  Bindable(ExprKind kind = ExprKind::Unbound);
  unsigned getId() const;

protected:
  Bindable(Expr::ImplType *ptr) : Expr(ptr) {}

private:
  static unsigned &newId();
};

struct UnaryExpr : public Expr {
  using ImplType = detail::UnaryExprStorage;
  friend class Expr;

  UnaryExpr(ExprKind kind, Expr expr);
  Expr getExpr() const;

protected:
  UnaryExpr(Expr::ImplType *ptr) : Expr(ptr) {}
};

struct BinaryExpr : public Expr {
  using ImplType = detail::BinaryExprStorage;
  friend class Expr;
  BinaryExpr(ExprKind kind, Expr lhs, Expr rhs);
  Expr getLHS() const;
  Expr getRHS() const;

protected:
  BinaryExpr(Expr::ImplType *ptr) : Expr(ptr) {}
};

struct TernaryExpr : public Expr {
  using ImplType = detail::TernaryExprStorage;
  friend class Expr;
  TernaryExpr(ExprKind kind, Expr cond, Expr lhs, Expr rhs);
  Expr getCond() const;
  Expr getLHS() const;
  Expr getRHS() const;

protected:
  TernaryExpr(Expr::ImplType *ptr) : Expr(ptr) {}
};

struct VariadicExpr : public Expr {
  using ImplType = detail::VariadicExprStorage;
  friend class Expr;
  VariadicExpr(ExprKind kind, llvm::ArrayRef<Expr> exprs,
               llvm::ArrayRef<Type> types = {});
  llvm::ArrayRef<Expr> getExprs() const;
  llvm::ArrayRef<Type> getTypes() const;

protected:
  VariadicExpr(Expr::ImplType *ptr) : Expr(ptr) {}
};

struct StmtBlockLikeExpr : public VariadicExpr {
  using ImplType = detail::VariadicExprStorage;
  friend class Expr;
  StmtBlockLikeExpr(ExprKind kind, llvm::ArrayRef<Expr> exprs,
                    llvm::ArrayRef<Type> types = {})
      : VariadicExpr(kind, exprs, types) {}

protected:
  StmtBlockLikeExpr(Expr::ImplType *ptr) : VariadicExpr(ptr) {}
};

/// A Stmt represent a unit of liaison betweeb a Bindable `lhs`, an Expr `rhs`
/// and a list of `enclosingStmts`. This essentially allows giving a name and a
/// scoping to objects of type `Expr` so they can be reused once bound to an
/// Value*. This enables writing generators such as:
///
/// ```mlir
///    Stmt scalarValue, vectorValue, tmpAlloc, tmpDealloc, vectorView;
///    Stmt block = Block({
///      tmpAlloc = alloc(tmpMemRefType),
///      vectorView = vector_type_cast(tmpAlloc, vectorMemRefType),
///      ForNest(ivs, lbs, ubs, steps, {
///        scalarValue = load(scalarMemRef,
///        accessInfo.clippedScalarAccessExprs), store(scalarValue, tmpAlloc,
///        accessInfo.tmpAccessExprs),
///      }),
///      vectorValue = load(vectorView, zero),
///      tmpDealloc = dealloc(tmpAlloc.getLHS())});
///    emitter.emitStmt(block);
/// ```
///
/// A Stmt can be declared with either:
/// 1. default initialization (e.g. `Stmt foo;`) in which case all of its `lhs`,
///    `rhs` and `enclosingStmts` are unbound;
/// 2. initialization from an Expr without a Bindable `lhs`
///    (e.g. store(scalarValue, tmpAlloc, accessInfo.tmpAccessExprs)), in which
///    case the `lhs` is unbound;
/// 3. an assignment operator to a `lhs` Stmt that is bound implicitly:
///    (e.g. vectorValue = load(vectorView, zero)).
///
/// Only ExprKind::StmtBlockLikeExpr have `enclosedStmts`, these comprise:
/// 1. `For`-loops for which the `lhs` binds to the induction variable, `rhs`
///   binds to an Expr of kind `ExprKind::For` with lower-bound, upper-bound and
///   step respectively;
/// 2. `Block` with an Expr of kind `ExprKind::Block` and which has no `rhs` but
///   only `enclosingStmts`.
struct Stmt {
  using ImplType = detail::StmtStorage;
  friend class Expr;
  Stmt() : storage(nullptr) {}
  Stmt(const Stmt &other) : storage(other.storage) {}
  Stmt operator=(const Stmt &other) {
    this->storage = other.storage; // NBD if &other == this
    return *this;
  }
  Stmt(const Expr &rhs, llvm::ArrayRef<Stmt> stmts = llvm::ArrayRef<Stmt>());
  Stmt(const Bindable &lhs, const Expr &rhs,
       llvm::ArrayRef<Stmt> stmts = llvm::ArrayRef<Stmt>());
  Stmt &operator=(const Expr &expr);

  operator Expr() const { return getLHS(); }

  /// For debugging purposes.
  const void *getStoragePtr() const { return storage; }

  void print(raw_ostream &os, llvm::Twine indent = "") const;
  void dump() const;

  Bindable getLHS() const;
  Expr getRHS() const;
  llvm::ArrayRef<Stmt> getEnclosedStmts() const;

  Expr operator+(Stmt other) const { return getLHS() + other.getLHS(); }
  Expr operator-(Stmt other) const { return getLHS() - other.getLHS(); }
  Expr operator*(Stmt other) const { return getLHS() * other.getLHS(); }

  Expr operator<(Stmt other) const { return getLHS() + other.getLHS(); }
  Expr operator<=(Stmt other) const { return getLHS() + other.getLHS(); }
  Expr operator>(Stmt other) const { return getLHS() + other.getLHS(); }
  Expr operator>=(Stmt other) const { return getLHS() + other.getLHS(); }
  Expr operator&&(Stmt other) const { return getLHS() + other.getLHS(); }
  Expr operator||(Stmt other) const { return getLHS() + other.getLHS(); }

protected:
  ImplType *storage;
};

template <typename U> bool Expr::isa() const {
  auto kind = getKind();
  if (std::is_same<U, Bindable>::value) {
    return kind >= ExprKind::FIRST_BINDABLE_EXPR &&
           kind <= ExprKind::LAST_BINDABLE_EXPR;
  }
  if (std::is_same<U, UnaryExpr>::value) {
    return kind >= ExprKind::FIRST_UNARY_EXPR &&
           kind <= ExprKind::LAST_UNARY_EXPR;
  }
  if (std::is_same<U, BinaryExpr>::value) {
    return kind >= ExprKind::FIRST_BINARY_EXPR &&
           kind <= ExprKind::LAST_BINARY_EXPR;
  }
  if (std::is_same<U, TernaryExpr>::value) {
    return kind >= ExprKind::FIRST_TERNARY_EXPR &&
           kind <= ExprKind::LAST_TERNARY_EXPR;
  }
  if (std::is_same<U, VariadicExpr>::value) {
    return kind >= ExprKind::FIRST_VARIADIC_EXPR &&
           kind <= ExprKind::LAST_VARIADIC_EXPR;
  }
  if (std::is_same<U, StmtBlockLikeExpr>::value) {
    return kind >= ExprKind::FIRST_STMT_BLOCK_LIKE_EXPR &&
           kind <= ExprKind::LAST_STMT_BLOCK_LIKE_EXPR;
  }
  return false;
}

template <typename U> U Expr::dyn_cast() const {
  if (isa<U>()) {
    return U(storage);
  }
  return U(nullptr);
}
template <typename U> U Expr::cast() const {
  assert(isa<U>());
  return U(storage);
}

/// Make Expr hashable.
inline ::llvm::hash_code hash_value(Expr arg) {
  return ::llvm::hash_value(arg.storage);
}

raw_ostream &operator<<(raw_ostream &os, const Expr &expr);
raw_ostream &operator<<(raw_ostream &os, const Stmt &stmt);

} // namespace edsc
} // namespace mlir

namespace llvm {

// Expr hash just like pointers
template <> struct DenseMapInfo<mlir::edsc::Expr> {
  static mlir::edsc::Expr getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::edsc::Expr(static_cast<mlir::edsc::Expr::ImplType *>(pointer));
  }
  static mlir::edsc::Expr getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::edsc::Expr(static_cast<mlir::edsc::Expr::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::edsc::Expr val) {
    return mlir::edsc::hash_value(val);
  }
  static bool isEqual(mlir::edsc::Expr LHS, mlir::edsc::Expr RHS) {
    return LHS.getStoragePtr() == RHS.getStoragePtr();
  }
};

} // namespace llvm

namespace mlir {
namespace edsc {

/// Free function sugar.
///
/// Since bindings are hashed by the underlying pointer address, we need to be
/// sure to construct new elements in a vector. We cannot just use
/// `llvm::SmallVector<Bindable, 8> dims(n);` directly because a single
/// `Bindable` will be default constructed and copied everywhere in the vector.
/// Hilarity ensues when trying to bind structs that are already bound.
llvm::SmallVector<Bindable, 8> makeBindables(unsigned n);
llvm::SmallVector<Expr, 8> makeExprs(unsigned n);
llvm::SmallVector<Expr, 8> makeExprs(ArrayRef<Bindable> bindables);
template <typename IterTy>
llvm::SmallVector<Expr, 8> makeExprs(IterTy begin, IterTy end) {
  return llvm::SmallVector<Expr, 8>(begin, end);
}

Expr alloc(llvm::ArrayRef<Expr> sizes, Type memrefType);
inline Expr alloc(Type memrefType) { return alloc({}, memrefType); }
Expr dealloc(Expr memref);
Expr load(Expr m, Expr index);
Expr load(Expr m, Bindable index);
Expr load(Expr m, llvm::ArrayRef<Expr> indices);
Expr load(Expr m, const llvm::SmallVectorImpl<Bindable> &indices);
Expr store(Expr val, Expr m, Expr index);
Expr store(Expr val, Expr m, Bindable index);
Expr store(Expr val, Expr m, llvm::ArrayRef<Expr> indices);
Expr store(Expr val, Expr m, const llvm::SmallVectorImpl<Bindable> &indices);
Expr select(Expr cond, Expr lhs, Expr rhs);
Expr vector_type_cast(Expr memrefExpr, Type memrefType);

Stmt Block(llvm::ArrayRef<Stmt> stmts);
Stmt For(Expr lb, Expr ub, Expr step, llvm::ArrayRef<Stmt> enclosedStmts);
Stmt For(const Bindable &idx, Expr lb, Expr ub, Expr step,
         llvm::ArrayRef<Stmt> enclosedStmts);
Stmt ForNest(llvm::MutableArrayRef<Bindable> indices, llvm::ArrayRef<Expr> lbs,
             llvm::ArrayRef<Expr> ubs, llvm::ArrayRef<Expr> steps,
             llvm::ArrayRef<Stmt> enclosedStmts);
Stmt ForNest(llvm::MutableArrayRef<Bindable> indices,
             llvm::ArrayRef<Bindable> lbs, llvm::ArrayRef<Bindable> ubs,
             llvm::ArrayRef<Bindable> steps,
             llvm::ArrayRef<Stmt> enclosedStmts);

/// This helper class exists purely for sugaring purposes and allows writing
/// expressions such as:
///
/// ```mlir
///    Indexed A(...), B(...), C(...);
///    ForNest(ivs, zeros, shapeA, ones, {
///      C[ivs] = A[ivs] + B[ivs]
///    });
/// ```
struct Indexed {
  Indexed(Bindable m) : base(m), indices() {}

  /// Returns a new `Indexed`. As a consequence, an Indexed with attached
  /// indices can never be reused unless it is captured (e.g. via a Stmt).
  /// This is consistent with SSA behavior in MLIR but also allows for some
  /// minimal state and sugaring.
  Indexed operator[](llvm::ArrayRef<Expr> indices) const;
  Indexed operator[](llvm::ArrayRef<Bindable> indices) const;

  /// Returns a new `Stmt`.
  /// Emits a `store` and clears the attached indices.
  Stmt operator=(Expr expr); // NOLINT: unconventional-assing-operator

  /// Implicit conversion.
  /// Emits a `load` and clears indices.
  operator Expr() const {
    assert(!indices.empty() && "Expected attached indices to Indexed");
    return load(base, indices);
  }

  /// Operator overloadings.
  Expr operator+(Expr e) const { return static_cast<Expr>(*this) + e; }
  Expr operator-(Expr e) const { return static_cast<Expr>(*this) - e; }
  Expr operator*(Expr e) const { return static_cast<Expr>(*this) * e; }

private:
  Bindable base;
  llvm::SmallVector<Expr, 4> indices;
};

} // namespace edsc
} // namespace mlir

#endif // MLIR_LIB_EDSC_TYPES_H_

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

#ifndef MLIR_EDSC_TYPES_H_
#define MLIR_EDSC_TYPES_H_

#include "mlir-c/Core.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"

namespace mlir {

class MLIRContext;
class FuncBuilder;

namespace edsc {
namespace detail {

struct ExprStorage;
struct StmtStorage;
struct StmtBlockStorage;

} // namespace detail

class StmtBlock;

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
  Unary = FIRST_NON_BINDABLE_EXPR,
  Binary,
  Ternary,
  Variadic,
  FIRST_STMT_BLOCK_LIKE_EXPR = 600,
  For = FIRST_STMT_BLOCK_LIKE_EXPR,
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

  explicit Expr(Type type);
  /* implicit */ Expr(ImplType *storage) : storage(storage) {}
  explicit Expr(edsc_expr_t expr)
      : storage(reinterpret_cast<ImplType *>(expr)) {}
  operator edsc_expr_t() { return edsc_expr_t{storage}; }

  Expr(const Expr &other) = default;
  Expr &operator=(const Expr &other) = default;
  Expr(StringRef name, Type resultType, ArrayRef<Expr> operands,
       ArrayRef<NamedAttribute> atts = {});

  template <typename U> bool isa() const;
  template <typename U> U dyn_cast() const;
  template <typename U> U cast() const;

  /// Returns `true` if this expression builds the MLIR operation specified as
  /// the template argument.  Unlike `isa`, this does not imply we can cast
  /// this Expr to the given type.
  template <typename U> bool is_op() const;

  /// Returns the classification for this type.
  ExprKind getKind() const;
  unsigned getId() const;
  StringRef getName() const;

  /// Returns the types of the values this expression produces.
  ArrayRef<Type> getResultTypes() const;

  /// Returns the list of expressions used as arguments of this expression.
  ArrayRef<Expr> getProperArguments() const;

  /// Returns the list of lists of expressions used as arguments of successors
  /// of this expression (i.e., arguments passed to destination basic blocks in
  /// terminator statements).
  SmallVector<ArrayRef<Expr>, 4> getSuccessorArguments() const;

  /// Returns the list of expressions used as arguments of the `index`-th
  /// successor of this expression.
  ArrayRef<Expr> getSuccessorArguments(int index) const;

  /// Returns the list of argument groups (includes the proper argument group,
  /// followed by successor/block argument groups).
  SmallVector<ArrayRef<Expr>, 4> getAllArgumentGroups() const;

  /// Returns the list of attributes of this expression.
  ArrayRef<NamedAttribute> getAttributes() const;

  /// Returns the attribute with the given name, if any.
  Attribute getAttribute(StringRef name) const;

  /// Returns the list of successors (StmtBlocks) of this expression.
  ArrayRef<StmtBlock> getSuccessors() const;

  /// Build the IR corresponding to this expression.
  SmallVector<Value *, 4>
  build(FuncBuilder &b, const llvm::DenseMap<Expr, Value *> &ssaBindings,
        const llvm::DenseMap<StmtBlock, Block *> &blockBindings) const;

  void print(raw_ostream &os) const;
  void dump() const;
  std::string str() const;

  /// For debugging purposes.
  const void *getStoragePtr() const { return storage; }

  /// Explicit conversion to bool.  Useful in conjunction with dyn_cast.
  explicit operator bool() const { return storage != nullptr; }

  friend ::llvm::hash_code hash_value(Expr arg);

protected:
  friend struct detail::ExprStorage;
  ImplType *storage;

  static void resetIds() { newId() = 0; }
  static unsigned &newId();
};

struct Bindable : public Expr {
  Bindable() = delete;
  Bindable(Expr expr) : Expr(expr) {
    assert(expr.isa<Bindable>() && "expected Bindable");
  }
  Bindable(const Bindable &) = default;
  Bindable &operator=(const Bindable &) = default;
  explicit Bindable(const edsc_expr_t &expr) : Expr(expr) {}
  operator edsc_expr_t() { return edsc_expr_t{storage}; }

private:
  friend class Expr;
  friend struct ScopedEDSCContext;
};

struct UnaryExpr : public Expr {
  friend class Expr;

  UnaryExpr(StringRef name, Expr expr);
  Expr getExpr() const;

  template <typename T> static UnaryExpr make(Expr expr) {
    return UnaryExpr(T::getOperationName(), expr);
  }

protected:
  UnaryExpr(Expr::ImplType *ptr) : Expr(ptr) {
    assert(!ptr || isa<UnaryExpr>() && "expected UnaryExpr");
  }
};

struct BinaryExpr : public Expr {
  friend class Expr;
  BinaryExpr(StringRef name, Type result, Expr lhs, Expr rhs,
             ArrayRef<NamedAttribute> attrs = {});
  Expr getLHS() const;
  Expr getRHS() const;

  template <typename T>
  static BinaryExpr make(Type result, Expr lhs, Expr rhs,
                         ArrayRef<NamedAttribute> attrs = {}) {
    return BinaryExpr(T::getOperationName(), result, lhs, rhs, attrs);
  }

protected:
  BinaryExpr(Expr::ImplType *ptr) : Expr(ptr) {
    assert(!ptr || isa<BinaryExpr>() && "expected BinaryExpr");
  }
};

struct TernaryExpr : public Expr {
  friend class Expr;
  TernaryExpr(StringRef name, Expr cond, Expr lhs, Expr rhs);
  Expr getCond() const;
  Expr getLHS() const;
  Expr getRHS() const;

  template <typename T> static TernaryExpr make(Expr cond, Expr lhs, Expr rhs) {
    return TernaryExpr(T::getOperationName(), cond, lhs, rhs);
  }

protected:
  TernaryExpr(Expr::ImplType *ptr) : Expr(ptr) {
    assert(!ptr || isa<TernaryExpr>() && "expected TernaryExpr");
  }
};

struct VariadicExpr : public Expr {
  friend class Expr;
  VariadicExpr(StringRef name, llvm::ArrayRef<Expr> exprs,
               llvm::ArrayRef<Type> types = {},
               llvm::ArrayRef<NamedAttribute> attrs = {},
               llvm::ArrayRef<StmtBlock> succ = {});
  llvm::ArrayRef<Expr> getExprs() const;
  llvm::ArrayRef<Type> getTypes() const;
  llvm::ArrayRef<StmtBlock> getSuccessors() const;

  template <typename T>
  static VariadicExpr make(llvm::ArrayRef<Expr> exprs,
                           llvm::ArrayRef<Type> types = {},
                           llvm::ArrayRef<NamedAttribute> attrs = {},
                           llvm::ArrayRef<StmtBlock> succ = {}) {
    return VariadicExpr(T::getOperationName(), exprs, types, attrs, succ);
  }

protected:
  VariadicExpr(Expr::ImplType *ptr) : Expr(ptr) {
    assert(!ptr || isa<VariadicExpr>() && "expected VariadicExpr");
  }
};

struct StmtBlockLikeExpr : public Expr {
  friend class Expr;
  StmtBlockLikeExpr(ExprKind kind, llvm::ArrayRef<Expr> exprs,
                    llvm::ArrayRef<Type> types = {});

protected:
  StmtBlockLikeExpr(Expr::ImplType *ptr) : Expr(ptr) {
    assert(!ptr || isa<StmtBlockLikeExpr>() && "expected StmtBlockLikeExpr");
  }
};

/// A Stmt represent a unit of liaison betweeb a Bindable `lhs`, an Expr `rhs`
/// and a list of `enclosingStmts`. This essentially allows giving a name and a
/// scoping to objects of type `Expr` so they can be reused once bound to an
/// Value*. This enables writing generators such as:
///
/// ```mlir
///    Stmt scalarValue, vectorValue, tmpAlloc, tmpDealloc, vectorView;
///    tmpAlloc = alloc(tmpMemRefType);
///    vectorView = vector.type_cast(tmpAlloc, vectorMemRefType),
///    vectorValue = load(vectorView, zero),
///    tmpDealloc = dealloc(tmpAlloc)});
///    emitter.emitStmts({tmpAlloc, vectorView, vectorValue, tmpDealloc});
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
/// 1. `affine.for`-loops for which the `lhs` binds to the induction variable,
/// `rhs`
///   binds to an Expr of kind `ExprKind::For` with lower-bound, upper-bound and
///   step respectively.
// TODO(zinenko): this StmtBlockLikeExpr should be retired in favor of Expr
// that can have a list of Blocks they contain, similarly to the core MLIR
struct Stmt {
  using ImplType = detail::StmtStorage;
  friend class Expr;
  Stmt() : storage(nullptr) {}
  explicit Stmt(ImplType *storage) : storage(storage) {}
  Stmt(const Stmt &other) = default;
  Stmt(const Expr &rhs, llvm::ArrayRef<Stmt> stmts = llvm::ArrayRef<Stmt>());
  Stmt(const Bindable &lhs, const Expr &rhs,
       llvm::ArrayRef<Stmt> stmts = llvm::ArrayRef<Stmt>());

  explicit operator Expr() const { return getLHS(); }
  Stmt &operator=(const Expr &expr);
  Stmt &set(const Stmt &other) {
    this->storage = other.storage;
    return *this;
  }
  Stmt &operator=(const Stmt &other) = delete;
  explicit Stmt(edsc_stmt_t stmt)
      : storage(reinterpret_cast<ImplType *>(stmt)) {}
  operator edsc_stmt_t() { return edsc_stmt_t{storage}; }

  /// For debugging purposes.
  const ImplType *getStoragePtr() const { return storage; }

  void print(raw_ostream &os, llvm::Twine indent = "") const;
  void dump() const;
  std::string str() const;

  Expr getLHS() const;
  Expr getRHS() const;
  llvm::ArrayRef<Stmt> getEnclosedStmts() const;

protected:
  ImplType *storage;
};

/// StmtBlock is a an addressable list of statements.
///
/// This enables writing complex generators such as:
///
/// ```mlir
///    Stmt scalarValue, vectorValue, tmpAlloc, tmpDealloc, vectorView;
///    Stmt block = Block({
///      tmpAlloc = alloc(tmpMemRefType),
///      vectorView = vector.type_cast(tmpAlloc, vectorMemRefType),
///      For(ivs, lbs, ubs, steps, {
///        scalarValue = load(scalarMemRef,
///        accessInfo.clippedScalarAccessExprs), store(scalarValue, tmpAlloc,
///        accessInfo.tmpAccessExprs),
///      }),
///      vectorValue = load(vectorView, zero),
///      tmpDealloc = dealloc(tmpAlloc.getLHS())});
///    emitter.emitBlock(block);
/// ```
struct StmtBlock {
public:
  using ImplType = detail::StmtBlockStorage;

  StmtBlock() : storage(nullptr) {}
  explicit StmtBlock(ImplType *st) : storage(st) {}
  explicit StmtBlock(edsc_block_t st)
      : storage(reinterpret_cast<ImplType *>(st)) {}
  StmtBlock(const StmtBlock &other) = default;
  StmtBlock(llvm::ArrayRef<Stmt> stmts);
  StmtBlock(llvm::ArrayRef<Bindable> args, llvm::ArrayRef<Stmt> stmts = {});

  llvm::ArrayRef<Bindable> getArguments() const;
  llvm::ArrayRef<Type> getArgumentTypes() const;
  llvm::ArrayRef<Stmt> getBody() const;
  uint64_t getId() const;

  void print(llvm::raw_ostream &os, Twine indent) const;
  std::string str() const;

  operator edsc_block_t() { return edsc_block_t{storage}; }

  /// Reset the body of this block with the given list of statements.
  StmtBlock &operator=(llvm::ArrayRef<Stmt> stmts);
  void set(llvm::ArrayRef<Stmt> stmts) { *this = stmts; }

  ImplType *getStoragePtr() const { return storage; }

private:
  ImplType *storage;
};

/// These operator build new expressions from the given expressions. Some of
/// them are unconventional, which mandated extracting them to a separate
/// namespace.  The indended use is as follows.
///
///    using namespace edsc;
///    Expr e1, e2, condition
///    {
///      using namespace edsc::op;
///      condition = !(e1 && e2);  // this is a negation expression
///    }
///    if (!condition)             // this is a nullity check
///      reportError();
///
namespace op {
/// Creates the BinaryExpr corresponding to the operator.
Expr operator+(Expr lhs, Expr rhs);
Expr operator-(Expr lhs, Expr rhs);
Expr operator*(Expr lhs, Expr rhs);
Expr operator/(Expr lhs, Expr rhs);
Expr operator%(Expr lhs, Expr rhs);
/// In particular operator==, operator!= return a new Expr and *not* a bool.
Expr operator==(Expr lhs, Expr rhs);
Expr operator!=(Expr lhs, Expr rhs);
Expr operator<(Expr lhs, Expr rhs);
Expr operator<=(Expr lhs, Expr rhs);
Expr operator>(Expr lhs, Expr rhs);
Expr operator>=(Expr lhs, Expr rhs);
/// NB: Unlike boolean && and || these do not short-circuit.
Expr operator&&(Expr lhs, Expr rhs);
Expr operator||(Expr lhs, Expr rhs);
Expr operator!(Expr expr);

inline Expr operator+(Stmt lhs, Stmt rhs) {
  return lhs.getLHS() + rhs.getLHS();
}
inline Expr operator-(Stmt lhs, Stmt rhs) {
  return lhs.getLHS() - rhs.getLHS();
}
inline Expr operator*(Stmt lhs, Stmt rhs) {
  return lhs.getLHS() * rhs.getLHS();
}

inline Expr operator<(Stmt lhs, Stmt rhs) {
  return lhs.getLHS() < rhs.getLHS();
}
inline Expr operator<=(Stmt lhs, Stmt rhs) {
  return lhs.getLHS() <= rhs.getLHS();
}
inline Expr operator>(Stmt lhs, Stmt rhs) {
  return lhs.getLHS() > rhs.getLHS();
}
inline Expr operator>=(Stmt lhs, Stmt rhs) {
  return lhs.getLHS() >= rhs.getLHS();
}
inline Expr operator&&(Stmt lhs, Stmt rhs) {
  return lhs.getLHS() && rhs.getLHS();
}
inline Expr operator||(Stmt lhs, Stmt rhs) {
  return lhs.getLHS() || rhs.getLHS();
}
inline Expr operator!(Stmt stmt) { return !stmt.getLHS(); }
} // end namespace op

Expr floorDiv(Expr lhs, Expr rhs);
Expr ceilDiv(Expr lhs, Expr rhs);

template <typename U> bool Expr::isa() const {
  auto kind = getKind();
  if (std::is_same<U, Bindable>::value) {
    return kind >= ExprKind::FIRST_BINDABLE_EXPR &&
           kind <= ExprKind::LAST_BINDABLE_EXPR;
  }
  if (std::is_same<U, UnaryExpr>::value) {
    return kind == ExprKind::Unary;
  }
  if (std::is_same<U, BinaryExpr>::value) {
    return kind == ExprKind::Binary;
  }
  if (std::is_same<U, TernaryExpr>::value) {
    return kind == ExprKind::Ternary;
  }
  if (std::is_same<U, VariadicExpr>::value) {
    return kind == ExprKind::Variadic;
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
  return U((Expr::ImplType *)(nullptr));
}
template <typename U> U Expr::cast() const {
  assert(isa<U>());
  return U(storage);
}

template <typename U> bool Expr::is_op() const {
  return U::getOperationName() == getName();
}

/// Make Expr hashable.
inline ::llvm::hash_code hash_value(Expr arg) {
  return ::llvm::hash_value(arg.storage);
}

inline ::llvm::hash_code hash_value(StmtBlock arg) {
  return ::llvm::hash_value(arg.getStoragePtr());
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

// StmtBlock hash just like pointers
template <> struct DenseMapInfo<mlir::edsc::StmtBlock> {
  static mlir::edsc::StmtBlock getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::edsc::StmtBlock(
        static_cast<mlir::edsc::StmtBlock::ImplType *>(pointer));
  }
  static mlir::edsc::StmtBlock getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::edsc::StmtBlock(
        static_cast<mlir::edsc::StmtBlock::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::edsc::StmtBlock val) {
    return mlir::edsc::hash_value(val);
  }
  static bool isEqual(mlir::edsc::StmtBlock LHS, mlir::edsc::StmtBlock RHS) {
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
/// `llvm::SmallVector<Expr, 8> dims(n);` directly because a single
/// `Expr` will be default constructed and copied everywhere in the vector.
/// Hilarity ensues when trying to bind `Expr` multiple times.
llvm::SmallVector<Expr, 8> makeNewExprs(unsigned n, Type type);
template <typename IterTy>
llvm::SmallVector<Expr, 8> copyExprs(IterTy begin, IterTy end) {
  return llvm::SmallVector<Expr, 8>(begin, end);
}
inline llvm::SmallVector<Expr, 8> copyExprs(llvm::ArrayRef<Expr> exprs) {
  return llvm::SmallVector<Expr, 8>(exprs.begin(), exprs.end());
}

Expr alloc(llvm::ArrayRef<Expr> sizes, Type memrefType);
inline Expr alloc(Type memrefType) { return alloc({}, memrefType); }
Expr dealloc(Expr memref);

Expr load(Expr m, llvm::ArrayRef<Expr> indices = {});
inline Expr load(Stmt m, llvm::ArrayRef<Expr> indices = {}) {
  return load(m.getLHS(), indices);
}
Expr store(Expr val, Expr m, llvm::ArrayRef<Expr> indices = {});
inline Expr store(Stmt val, Expr m, llvm::ArrayRef<Expr> indices = {}) {
  return store(val.getLHS(), m, indices);
}
Expr select(Expr cond, Expr lhs, Expr rhs);
Expr vector_type_cast(Expr memrefExpr, Type memrefType);
Expr constantInteger(Type t, int64_t value);
Expr call(Expr func, Type result, llvm::ArrayRef<Expr> args);
Expr call(Expr func, llvm::ArrayRef<Expr> args);

Stmt Return(ArrayRef<Expr> values = {});
Stmt Branch(StmtBlock destination, ArrayRef<Expr> args = {});
Stmt CondBranch(Expr condition, StmtBlock trueDestination,
                ArrayRef<Expr> trueArgs, StmtBlock falseDestination,
                ArrayRef<Expr> falseArgs);
Stmt CondBranch(Expr condition, StmtBlock trueDestination,
                StmtBlock falseDestination);

Stmt For(Expr lb, Expr ub, Expr step, llvm::ArrayRef<Stmt> enclosedStmts);
Stmt For(const Bindable &idx, Expr lb, Expr ub, Expr step,
         llvm::ArrayRef<Stmt> enclosedStmts);
Stmt For(llvm::ArrayRef<Expr> indices, llvm::ArrayRef<Expr> lbs,
         llvm::ArrayRef<Expr> ubs, llvm::ArrayRef<Expr> steps,
         llvm::ArrayRef<Stmt> enclosedStmts);

/// Define a 'affine.for' loop from with multi-valued bounds.
///
///    for max(lbs...) to min(ubs...) {}
///
Stmt MaxMinFor(const Bindable &idx, ArrayRef<Expr> lbs, ArrayRef<Expr> ubs,
               Expr step, ArrayRef<Stmt> enclosedStmts);

/// Define an MLIR Block and bind its arguments to `args`.  The types of block
/// arguments are those of `args`, each of which must have exactly one result
/// type.  The body of the block may be empty and can be reset later.
StmtBlock block(llvm::ArrayRef<Bindable> args, llvm::ArrayRef<Stmt> stmts);
/// Define an MLIR Block without arguments.  The body of the block can be empty
/// and can be reset later.
inline StmtBlock block(llvm::ArrayRef<Stmt> stmts) { return block({}, stmts); }

/// This helper class exists purely for sugaring purposes and allows writing
/// expressions such as:
///
/// ```mlir
///    Indexed A(...), B(...), C(...);
///    For(ivs, zeros, shapeA, ones, {
///      C[ivs] = A[ivs] + B[ivs]
///    });
/// ```
struct Indexed {
  Indexed(Expr e) : base(e), indices() {}

  /// Returns a new `Indexed`. As a consequence, an Indexed with attached
  /// indices can never be reused unless it is captured (e.g. via a Stmt).
  /// This is consistent with SSA behavior in MLIR but also allows for some
  /// minimal state and sugaring.
  Indexed operator()(llvm::ArrayRef<Expr> indices = {});

  /// Returns a new `Stmt`.
  /// Emits a `store` and clears the attached indices.
  Stmt operator=(Expr expr); // NOLINT: unconventional-assing-operator

  /// Implicit conversion.
  /// Emits a `load`.
  operator Expr() { return load(base, indices); }

  /// Operator overloadings.
  Expr operator+(Expr e) {
    using op::operator+;
    return load(base, indices) + e;
  }
  Expr operator-(Expr e) {
    using op::operator-;
    return load(base, indices) - e;
  }
  Expr operator*(Expr e) {
    using op::operator*;
    return load(base, indices) * e;
  }

private:
  Expr base;
  llvm::SmallVector<Expr, 8> indices;
};

struct MaxExpr {
public:
  explicit MaxExpr(llvm::ArrayRef<Expr> arguments);
  explicit MaxExpr(edsc_max_expr_t st)
      : storage(reinterpret_cast<detail::ExprStorage *>(st)) {}
  llvm::ArrayRef<Expr> getArguments() const;

  operator edsc_max_expr_t() { return storage; }

private:
  detail::ExprStorage *storage;
};

struct MinExpr {
public:
  explicit MinExpr(llvm::ArrayRef<Expr> arguments);
  explicit MinExpr(edsc_min_expr_t st)
      : storage(reinterpret_cast<detail::ExprStorage *>(st)) {}
  llvm::ArrayRef<Expr> getArguments() const;

  operator edsc_min_expr_t() { return storage; }

private:
  detail::ExprStorage *storage;
};

Stmt For(const Bindable &idx, MaxExpr lb, MinExpr ub, Expr step,
         llvm::ArrayRef<Stmt> enclosedStmts);
Stmt For(llvm::ArrayRef<Expr> idxs, llvm::ArrayRef<MaxExpr> lbs,
         llvm::ArrayRef<MinExpr> ubs, llvm::ArrayRef<Expr> steps,
         llvm::ArrayRef<Stmt> enclosedStmts);

inline MaxExpr Max(llvm::ArrayRef<Expr> args) { return MaxExpr(args); }
inline MinExpr Min(llvm::ArrayRef<Expr> args) { return MinExpr(args); }

} // namespace edsc
} // namespace mlir

#endif // MLIR_EDSC_TYPES_H_

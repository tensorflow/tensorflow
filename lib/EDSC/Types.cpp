//===- Types.h - MLIR EDSC Type System Implementation -----------*- C++ -*-===//
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

#include "mlir/EDSC/Types.h"
#include "mlir/Support/STLExtras.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using llvm::errs;
using llvm::Twine;

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::detail;

namespace mlir {
namespace edsc {
namespace detail {

struct ExprStorage {
  ExprStorage(ExprKind kind) : kind(kind) {}
  ExprKind kind;
};

struct UnaryExprStorage : public ExprStorage {
  UnaryExprStorage(ExprKind k, Expr expr) : ExprStorage(k), expr(expr) {}
  Expr expr;
};

struct BinaryExprStorage : public ExprStorage {
  BinaryExprStorage(ExprKind k, Expr lhs, Expr rhs)
      : ExprStorage(k), lhs(lhs), rhs(rhs) {}
  Expr lhs, rhs;
};

struct TernaryExprStorage : public ExprStorage {
  TernaryExprStorage(ExprKind k, Expr cond, Expr lhs, Expr rhs)
      : ExprStorage(k), cond(cond), lhs(lhs), rhs(rhs) {}
  Expr cond, lhs, rhs;
};

struct VariadicExprStorage : public ExprStorage {
  VariadicExprStorage(ExprKind k, ArrayRef<Expr> exprs, ArrayRef<Type> types)
      : ExprStorage(k), exprs(exprs.begin(), exprs.end()),
        types(types.begin(), types.end()) {}
  ArrayRef<Expr> exprs;
  ArrayRef<Type> types;
};

struct StmtStorage {
  StmtStorage(Bindable lhs, Expr rhs, llvm::ArrayRef<Stmt> enclosedStmts)
      : lhs(lhs), rhs(rhs), enclosedStmts(enclosedStmts) {}
  Bindable lhs;
  Expr rhs;
  ArrayRef<Stmt> enclosedStmts;
};

} // namespace detail

ScopedEDSCContext::ScopedEDSCContext() { Expr::globalAllocator() = &allocator; }

ScopedEDSCContext::~ScopedEDSCContext() { Expr::globalAllocator() = nullptr; }

ExprKind Expr::getKind() const { return storage->kind; }

Expr Expr::operator+(Expr other) const {
  return BinaryExpr(ExprKind::Add, *this, other);
}
Expr Expr::operator-(Expr other) const {
  return BinaryExpr(ExprKind::Sub, *this, other);
}

Expr Expr::operator==(Expr other) const {
  return BinaryExpr(ExprKind::EQ, *this, other);
}
Expr Expr::operator!=(Expr other) const {
  return BinaryExpr(ExprKind::NE, *this, other);
}
Expr Expr::operator<(Expr other) const {
  return BinaryExpr(ExprKind::LT, *this, other);
}
Expr Expr::operator<=(Expr other) const {
  return BinaryExpr(ExprKind::LE, *this, other);
}
Expr Expr::operator>(Expr other) const {
  return BinaryExpr(ExprKind::GT, *this, other);
}
Expr Expr::operator>=(Expr other) const {
  return BinaryExpr(ExprKind::GE, *this, other);
}
Expr Expr::operator&&(Expr other) const {
  return BinaryExpr(ExprKind::And, *this, other);
}
Expr Expr::operator||(Expr other) const {
  return BinaryExpr(ExprKind::Or, *this, other);
}

// Free functions.
llvm::SmallVector<Bindable, 8> makeBindables(unsigned n) {
  llvm::SmallVector<Bindable, 8> res;
  res.reserve(n);
  for (auto i = 0; i < n; ++i) {
    res.push_back(Bindable());
  }
  return res;
}

llvm::SmallVector<Expr, 8> makeExprs(unsigned n) {
  llvm::SmallVector<Expr, 8> res;
  res.reserve(n);
  for (auto i = 0; i < n; ++i) {
    res.push_back(Expr());
  }
  return res;
}

llvm::SmallVector<Expr, 8> makeExprs(ArrayRef<Bindable> bindables) {
  llvm::SmallVector<Expr, 8> res;
  res.reserve(bindables.size());
  for (auto b : bindables) {
    res.push_back(b);
  }
  return res;
}

Expr alloc(llvm::ArrayRef<Expr> sizes, Type memrefType) {
  return VariadicExpr(ExprKind::Alloc, sizes, memrefType);
}

Stmt Block(ArrayRef<Stmt> stmts) {
  return Stmt(StmtBlockLikeExpr(ExprKind::Block, {}), stmts);
}

Expr dealloc(Expr memref) { return UnaryExpr(ExprKind::Dealloc, memref); }

Stmt For(Expr lb, Expr ub, Expr step, ArrayRef<Stmt> stmts) {
  Bindable idx;
  return For(idx, lb, ub, step, stmts);
}

Stmt For(const Bindable &idx, Expr lb, Expr ub, Expr step,
         ArrayRef<Stmt> stmts) {
  return Stmt(idx, StmtBlockLikeExpr(ExprKind::For, {lb, ub, step}), stmts);
}

Stmt ForNest(MutableArrayRef<Bindable> indices, ArrayRef<Expr> lbs,
             ArrayRef<Expr> ubs, ArrayRef<Expr> steps,
             ArrayRef<Stmt> enclosedStmts) {
  assert(!indices.empty());
  assert(indices.size() == lbs.size());
  assert(indices.size() == ubs.size());
  assert(indices.size() == steps.size());
  Stmt curStmt =
      For(indices.back(), lbs.back(), ubs.back(), steps.back(), enclosedStmts);
  for (int64_t i = indices.size() - 2; i >= 0; --i) {
    Stmt nextStmt = For(indices[i], lbs[i], ubs[i], steps[i], {curStmt});
    curStmt = nextStmt;
  }
  return curStmt;
}

Expr load(Expr m, llvm::ArrayRef<Expr> indices) {
  SmallVector<Expr, 8> exprs;
  exprs.push_back(m);
  exprs.insert(exprs.end(), indices.begin(), indices.end());
  return VariadicExpr(ExprKind::Load, exprs);
}

Expr store(Expr val, Expr m, llvm::ArrayRef<Expr> indices) {
  SmallVector<Expr, 8> exprs;
  exprs.push_back(val);
  exprs.push_back(m);
  exprs.insert(exprs.end(), indices.begin(), indices.end());
  return VariadicExpr(ExprKind::Store, exprs);
}

Expr select(Expr cond, Expr lhs, Expr rhs) {
  return TernaryExpr(ExprKind::Select, cond, lhs, rhs);
}

Expr vector_type_cast(Expr memrefExpr, Type memrefType) {
  return VariadicExpr(ExprKind::VectorTypeCast, {memrefExpr}, {memrefType});
}

void Expr::print(raw_ostream &os) const {
  if (auto unbound = this->dyn_cast<Bindable>()) {
    os << "bindable";
    return;
  } else if (auto bin = this->dyn_cast<BinaryExpr>()) {
    os << bin.getLHS();
    switch (bin.getKind()) {
    case ExprKind::Add:
      os << " + ";
      break;
    case ExprKind::Sub:
      os << " - ";
      break;
    case ExprKind::LT:
      os << " < ";
      break;
    case ExprKind::LE:
      os << " <= ";
      break;
    case ExprKind::GT:
      os << " > ";
      break;
    case ExprKind::GE:
      os << " >= ";
      break;
    default: {
    }
    }
    os << bin.getRHS();
  } else if (auto ter = this->dyn_cast<TernaryExpr>()) {
    switch (ter.getKind()) {
    case ExprKind::Select:
      os << "select(" << ter.getCond() << ", " << ter.getLHS() << ", "
         << ter.getRHS() << ")";
      return;
    default: {
    }
    }
  } else if (auto nar = this->dyn_cast<VariadicExpr>()) {
    switch (nar.getKind()) {
    case ExprKind::Load:
      os << "load( ... )";
      return;
    default: {
    }
    }
  } else if (auto stmtLikeExpr = this->dyn_cast<StmtBlockLikeExpr>()) {
    auto exprs = stmtLikeExpr.getExprs();
    assert(exprs.size() == 3 && "For StmtBlockLikeExpr expected 3 exprs");
    switch (stmtLikeExpr.getKind()) {
    // We only print the lb, ub and step here, which are the StmtBlockLike
    // part of the `for` StmtBlockLikeExpr.
    case ExprKind::For:
      os << exprs[0] << " to " << exprs[1] << " step " << exprs[2];
      return;
    default: {
    }
    }
  }
  os << "unknown_kind(" << static_cast<int>(getKind()) << ")";
}

void Expr::dump() const { this->print(llvm::errs()); }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Expr &expr) {
  expr.print(os);
  return os;
}

Bindable::Bindable(ExprKind kind)
    : Expr(Expr::globalAllocator()->Allocate<detail::ExprStorage>()) {
  // Initialize with placement new.
  new (storage) detail::ExprStorage{kind};
}

UnaryExpr::UnaryExpr(ExprKind kind, Expr expr)
    : Expr(Expr::globalAllocator()->Allocate<detail::UnaryExprStorage>()) {
  // Initialize with placement new.
  new (storage) detail::UnaryExprStorage{kind, expr};
}
Expr UnaryExpr::getExpr() const {
  return static_cast<ImplType *>(storage)->expr;
}

BinaryExpr::BinaryExpr(ExprKind kind, Expr lhs, Expr rhs)
    : Expr(Expr::globalAllocator()->Allocate<detail::BinaryExprStorage>()) {
  // Initialize with placement new.
  new (storage) detail::BinaryExprStorage{kind, lhs, rhs};
}
Expr BinaryExpr::getLHS() const {
  return static_cast<ImplType *>(storage)->lhs;
}
Expr BinaryExpr::getRHS() const {
  return static_cast<ImplType *>(storage)->rhs;
}

TernaryExpr::TernaryExpr(ExprKind kind, Expr cond, Expr lhs, Expr rhs)
    : Expr(Expr::globalAllocator()->Allocate<detail::TernaryExprStorage>()) {
  // Initialize with placement new.
  new (storage) detail::TernaryExprStorage{kind, cond, lhs, rhs};
}
Expr TernaryExpr::getCond() const {
  return static_cast<ImplType *>(storage)->cond;
}
Expr TernaryExpr::getLHS() const {
  return static_cast<ImplType *>(storage)->lhs;
}
Expr TernaryExpr::getRHS() const {
  return static_cast<ImplType *>(storage)->rhs;
}

VariadicExpr::VariadicExpr(ExprKind kind, ArrayRef<Expr> exprs,
                           ArrayRef<Type> types)
    : Expr(Expr::globalAllocator()->Allocate<detail::VariadicExprStorage>()) {
  // Initialize with placement new.
  auto exprStorage = Expr::globalAllocator()->Allocate<Expr>(exprs.size());
  std::uninitialized_copy(exprs.begin(), exprs.end(), exprStorage);
  auto typeStorage = Expr::globalAllocator()->Allocate<Type>(types.size());
  std::uninitialized_copy(types.begin(), types.end(), typeStorage);
  new (storage) detail::VariadicExprStorage{
      kind, ArrayRef<Expr>(exprStorage, exprs.size()),
      ArrayRef<Type>(typeStorage, types.size())};
}
ArrayRef<Expr> VariadicExpr::getExprs() const {
  return static_cast<ImplType *>(storage)->exprs;
}
ArrayRef<Type> VariadicExpr::getTypes() const {
  return static_cast<ImplType *>(storage)->types;
}

Stmt::Stmt(const Bindable &lhs, const Expr &rhs,
           llvm::ArrayRef<Stmt> enclosedStmts) {
  storage = Expr::globalAllocator()->Allocate<detail::StmtStorage>();
  // Initialize with placement new.
  auto enclosedStmtStorage =
      Expr::globalAllocator()->Allocate<Stmt>(enclosedStmts.size());
  std::uninitialized_copy(enclosedStmts.begin(), enclosedStmts.end(),
                          enclosedStmtStorage);
  new (storage) detail::StmtStorage{
      lhs, rhs, ArrayRef<Stmt>(enclosedStmtStorage, enclosedStmts.size())};
}

Stmt::Stmt(const Expr &rhs, llvm::ArrayRef<Stmt> enclosedStmts)
    : Stmt(Bindable(), rhs, enclosedStmts) {}

Stmt &Stmt::operator=(const Expr &expr) {
  Stmt res(Bindable(), expr, {});
  std::swap(res.storage, this->storage);
  return *this;
}

Bindable Stmt::getLHS() const { return static_cast<ImplType *>(storage)->lhs; }

Expr Stmt::getRHS() const { return static_cast<ImplType *>(storage)->rhs; }

llvm::ArrayRef<Stmt> Stmt::getEnclosedStmts() const {
  return storage->enclosedStmts;
}

void Stmt::print(raw_ostream &os, Twine indent) const {
  assert(storage && "Unexpected null storage,stmt must be bound to print");
  auto lhs = getLHS();
  auto rhs = getRHS();

  if (auto stmt = rhs.dyn_cast<StmtBlockLikeExpr>()) {
    switch (stmt.getKind()) {
    case ExprKind::For:
      os << indent << "for(idx(" << lhs << ")=" << rhs << ") {";
      os << " // @" << storage;
      os << "\n";
      for (const auto &s : getEnclosedStmts()) {
        if (!s.getRHS().isa<StmtBlockLikeExpr>()) {
          os << indent << "  ";
        }
        s.print(os, indent + "  ");
        os << ";\n";
      }
      os << indent << "}";
      return;
    default: {
      // TODO(ntv): print more statement cases.
      os << "TODO";
    }
    }
  } else {
    os << "lhs(" << lhs << ") = " << rhs;
  }
}

void Stmt::dump() const { this->print(llvm::errs()); }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Stmt &stmt) {
  stmt.print(os);
  return os;
}

} // namespace edsc
} // namespace mlir

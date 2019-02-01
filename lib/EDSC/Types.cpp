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
#include "mlir-c/Core.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
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
  ExprStorage(ExprKind kind, unsigned id = Expr::newId())
      : kind(kind), id(id) {}
  ExprKind kind;
  unsigned id;
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
} // namespace edsc
} // namespace mlir

mlir::edsc::ScopedEDSCContext::ScopedEDSCContext() {
  Expr::globalAllocator() = &allocator;
  Bindable::resetIds();
}

mlir::edsc::ScopedEDSCContext::~ScopedEDSCContext() {
  Expr::globalAllocator() = nullptr;
}

mlir::edsc::Expr::Expr() {
  // Initialize with placement new.
  storage = Expr::globalAllocator()->Allocate<detail::ExprStorage>();
  new (storage) detail::ExprStorage(ExprKind::Unbound);
}

ExprKind mlir::edsc::Expr::getKind() const { return storage->kind; }

unsigned mlir::edsc::Expr::getId() const {
  return static_cast<ImplType *>(storage)->id;
}

unsigned &mlir::edsc::Expr::newId() {
  static thread_local unsigned id = 0;
  return ++id;
}

Expr mlir::edsc::Expr::operator+(Expr other) const {
  return BinaryExpr(ExprKind::Add, *this, other);
}
Expr mlir::edsc::Expr::operator-(Expr other) const {
  return BinaryExpr(ExprKind::Sub, *this, other);
}
Expr mlir::edsc::Expr::operator*(Expr other) const {
  return BinaryExpr(ExprKind::Mul, *this, other);
}

Expr mlir::edsc::Expr::operator==(Expr other) const {
  return BinaryExpr(ExprKind::EQ, *this, other);
}
Expr mlir::edsc::Expr::operator!=(Expr other) const {
  return BinaryExpr(ExprKind::NE, *this, other);
}
Expr mlir::edsc::Expr::operator<(Expr other) const {
  return BinaryExpr(ExprKind::LT, *this, other);
}
Expr mlir::edsc::Expr::operator<=(Expr other) const {
  return BinaryExpr(ExprKind::LE, *this, other);
}
Expr mlir::edsc::Expr::operator>(Expr other) const {
  return BinaryExpr(ExprKind::GT, *this, other);
}
Expr mlir::edsc::Expr::operator>=(Expr other) const {
  return BinaryExpr(ExprKind::GE, *this, other);
}
Expr mlir::edsc::Expr::operator&&(Expr other) const {
  return BinaryExpr(ExprKind::And, *this, other);
}
Expr mlir::edsc::Expr::operator||(Expr other) const {
  return BinaryExpr(ExprKind::Or, *this, other);
}

// Free functions.
llvm::SmallVector<Expr, 8> mlir::edsc::makeNewExprs(unsigned n) {
  llvm::SmallVector<Expr, 8> res;
  res.reserve(n);
  for (auto i = 0; i < n; ++i) {
    res.push_back(Expr());
  }
  return res;
}

static llvm::SmallVector<Expr, 8> makeExprs(edsc_expr_list_t exprList) {
  llvm::SmallVector<Expr, 8> exprs;
  exprs.reserve(exprList.n);
  for (unsigned i = 0; i < exprList.n; ++i) {
    exprs.push_back(Expr(exprList.exprs[i]));
  }
  return exprs;
}

static void fillStmts(edsc_stmt_list_t enclosedStmts,
                      llvm::SmallVector<Stmt, 8> *stmts) {
  stmts->reserve(enclosedStmts.n);
  for (unsigned i = 0; i < enclosedStmts.n; ++i) {
    stmts->push_back(Stmt(enclosedStmts.stmts[i]));
  }
}

Expr mlir::edsc::alloc(llvm::ArrayRef<Expr> sizes, Type memrefType) {
  return VariadicExpr(ExprKind::Alloc, sizes, memrefType);
}

Stmt mlir::edsc::StmtList(ArrayRef<Stmt> stmts) {
  return Stmt(StmtBlockLikeExpr(ExprKind::StmtList, {}), stmts);
}

edsc_stmt_t StmtList(edsc_stmt_list_t enclosedStmts) {
  llvm::SmallVector<Stmt, 8> stmts;
  fillStmts(enclosedStmts, &stmts);
  return Stmt(mlir::edsc::StmtList(stmts));
}

Expr mlir::edsc::dealloc(Expr memref) {
  return UnaryExpr(ExprKind::Dealloc, memref);
}

Stmt mlir::edsc::For(Expr lb, Expr ub, Expr step, ArrayRef<Stmt> stmts) {
  Expr idx;
  return For(Bindable(idx), lb, ub, step, stmts);
}

Stmt mlir::edsc::For(const Bindable &idx, Expr lb, Expr ub, Expr step,
                     ArrayRef<Stmt> stmts) {
  return Stmt(idx, StmtBlockLikeExpr(ExprKind::For, {lb, ub, step}), stmts);
}

Stmt mlir::edsc::For(ArrayRef<Expr> indices, ArrayRef<Expr> lbs,
                     ArrayRef<Expr> ubs, ArrayRef<Expr> steps,
                     ArrayRef<Stmt> enclosedStmts) {
  assert(!indices.empty());
  assert(indices.size() == lbs.size());
  assert(indices.size() == ubs.size());
  assert(indices.size() == steps.size());
  Expr iv = indices.back();
  Stmt curStmt =
      For(Bindable(iv), lbs.back(), ubs.back(), steps.back(), enclosedStmts);
  for (int64_t i = indices.size() - 2; i >= 0; --i) {
    Expr iiv = indices[i];
    curStmt.set(For(Bindable(iiv), lbs[i], ubs[i], steps[i],
                    llvm::ArrayRef<Stmt>{&curStmt, 1}));
  }
  return curStmt;
}

edsc_stmt_t For(edsc_expr_t iv, edsc_expr_t lb, edsc_expr_t ub,
                edsc_expr_t step, edsc_stmt_list_t enclosedStmts) {
  llvm::SmallVector<Stmt, 8> stmts;
  fillStmts(enclosedStmts, &stmts);
  return Stmt(
      For(Expr(iv).cast<Bindable>(), Expr(lb), Expr(ub), Expr(step), stmts));
}

edsc_stmt_t ForNest(edsc_expr_list_t ivs, edsc_expr_list_t lbs,
                    edsc_expr_list_t ubs, edsc_expr_list_t steps,
                    edsc_stmt_list_t enclosedStmts) {
  llvm::SmallVector<Stmt, 8> stmts;
  fillStmts(enclosedStmts, &stmts);
  return Stmt(For(makeExprs(ivs), makeExprs(lbs), makeExprs(ubs),
                  makeExprs(steps), stmts));
}

Expr mlir::edsc::load(Expr m, ArrayRef<Expr> indices) {
  SmallVector<Expr, 8> exprs;
  exprs.push_back(m);
  exprs.append(indices.begin(), indices.end());
  return VariadicExpr(ExprKind::Load, exprs);
}

edsc_expr_t Load(edsc_indexed_t indexed, edsc_expr_list_t indices) {
  Indexed i(Expr(indexed.base).cast<Bindable>());
  auto exprs = makeExprs(indices);
  Expr res = i(exprs);
  return res;
}

Expr mlir::edsc::store(Expr val, Expr m, ArrayRef<Expr> indices) {
  SmallVector<Expr, 8> exprs;
  exprs.push_back(val);
  exprs.push_back(m);
  exprs.append(indices.begin(), indices.end());
  return VariadicExpr(ExprKind::Store, exprs);
}

edsc_stmt_t Store(edsc_expr_t value, edsc_indexed_t indexed,
                  edsc_expr_list_t indices) {
  Indexed i(Expr(indexed.base).cast<Bindable>());
  auto exprs = makeExprs(indices);
  Indexed loc = i(exprs);
  return Stmt(loc = Expr(value));
}

Expr mlir::edsc::select(Expr cond, Expr lhs, Expr rhs) {
  return TernaryExpr(ExprKind::Select, cond, lhs, rhs);
}

edsc_expr_t Select(edsc_expr_t cond, edsc_expr_t lhs, edsc_expr_t rhs) {
  return select(Expr(cond), Expr(lhs), Expr(rhs));
}

Expr mlir::edsc::vector_type_cast(Expr memrefExpr, Type memrefType) {
  return VariadicExpr(ExprKind::VectorTypeCast, {memrefExpr}, {memrefType});
}

Stmt mlir::edsc::Return(ArrayRef<Expr> values) {
  return VariadicExpr(ExprKind::Return, values);
}

edsc_stmt_t Return(edsc_expr_list_t values) {
  return Stmt(Return(makeExprs(values)));
}

void mlir::edsc::Expr::print(raw_ostream &os) const {
  if (auto unbound = this->dyn_cast<Bindable>()) {
    os << "$" << unbound.getId();
    return;
  } else if (auto un = this->dyn_cast<UnaryExpr>()) {
    os << "unknown_unary";
  } else if (auto bin = this->dyn_cast<BinaryExpr>()) {
    os << "(" << bin.getLHS();
    switch (bin.getKind()) {
    case ExprKind::Add:
      os << " + ";
      break;
    case ExprKind::Sub:
      os << " - ";
      break;
    case ExprKind::Mul:
      os << " * ";
      break;
    case ExprKind::Div:
      os << " / ";
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
      os << "unknown_binary";
    }
    }
    os << bin.getRHS() << ")";
    return;
  } else if (auto ter = this->dyn_cast<TernaryExpr>()) {
    switch (ter.getKind()) {
    case ExprKind::Select:
      os << "select(" << ter.getCond() << ", " << ter.getLHS() << ", "
         << ter.getRHS() << ")";
      return;
    default: {
      os << "unknown_ternary";
    }
    }
  } else if (auto nar = this->dyn_cast<VariadicExpr>()) {
    auto exprs = nar.getExprs();
    switch (nar.getKind()) {
    case ExprKind::Load:
      os << "load(" << exprs[0] << "[";
      interleaveComma(ArrayRef<Expr>(exprs.begin() + 1, exprs.size() - 1), os);
      os << "])";
      return;
    case ExprKind::Store:
      os << "store(" << exprs[0] << ", " << exprs[1] << "[";
      interleaveComma(ArrayRef<Expr>(exprs.begin() + 2, exprs.size() - 2), os);
      os << "])";
      return;
    case ExprKind::Return:
      interleaveComma(exprs, os);
      return;
    default: {
      os << "unknown_variadic";
    }
    }
  } else if (auto stmtLikeExpr = this->dyn_cast<StmtBlockLikeExpr>()) {
    auto exprs = stmtLikeExpr.getExprs();
    switch (stmtLikeExpr.getKind()) {
    // We only print the lb, ub and step here, which are the StmtBlockLike
    // part of the `for` StmtBlockLikeExpr.
    case ExprKind::For:
      assert(exprs.size() == 3 && "For StmtBlockLikeExpr expected 3 exprs");
      os << exprs[0] << " to " << exprs[1] << " step " << exprs[2];
      return;
    default: {
      os << "unknown_stmt";
    }
    }
  }
  os << "unknown_kind(" << static_cast<int>(getKind()) << ")";
}

void mlir::edsc::Expr::dump() const { this->print(llvm::errs()); }

std::string mlir::edsc::Expr::str() const {
  std::string res;
  llvm::raw_string_ostream os(res);
  this->print(os);
  return res;
}

llvm::raw_ostream &mlir::edsc::operator<<(llvm::raw_ostream &os,
                                          const Expr &expr) {
  expr.print(os);
  return os;
}

edsc_expr_t makeBindable() { return Bindable(Expr()); }

mlir::edsc::UnaryExpr::UnaryExpr(ExprKind kind, Expr expr)
    : Expr(Expr::globalAllocator()->Allocate<detail::UnaryExprStorage>()) {
  // Initialize with placement new.
  new (storage) detail::UnaryExprStorage{kind, expr};
}
Expr mlir::edsc::UnaryExpr::getExpr() const {
  return static_cast<ImplType *>(storage)->expr;
}

mlir::edsc::BinaryExpr::BinaryExpr(ExprKind kind, Expr lhs, Expr rhs)
    : Expr(Expr::globalAllocator()->Allocate<detail::BinaryExprStorage>()) {
  // Initialize with placement new.
  new (storage) detail::BinaryExprStorage{kind, lhs, rhs};
}
Expr mlir::edsc::BinaryExpr::getLHS() const {
  return static_cast<ImplType *>(storage)->lhs;
}
Expr mlir::edsc::BinaryExpr::getRHS() const {
  return static_cast<ImplType *>(storage)->rhs;
}

mlir::edsc::TernaryExpr::TernaryExpr(ExprKind kind, Expr cond, Expr lhs,
                                     Expr rhs)
    : Expr(Expr::globalAllocator()->Allocate<detail::TernaryExprStorage>()) {
  // Initialize with placement new.
  new (storage) detail::TernaryExprStorage{kind, cond, lhs, rhs};
}
Expr mlir::edsc::TernaryExpr::getCond() const {
  return static_cast<ImplType *>(storage)->cond;
}
Expr mlir::edsc::TernaryExpr::getLHS() const {
  return static_cast<ImplType *>(storage)->lhs;
}
Expr mlir::edsc::TernaryExpr::getRHS() const {
  return static_cast<ImplType *>(storage)->rhs;
}

mlir::edsc::VariadicExpr::VariadicExpr(ExprKind kind, ArrayRef<Expr> exprs,
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
ArrayRef<Expr> mlir::edsc::VariadicExpr::getExprs() const {
  return static_cast<ImplType *>(storage)->exprs;
}
ArrayRef<Type> mlir::edsc::VariadicExpr::getTypes() const {
  return static_cast<ImplType *>(storage)->types;
}

mlir::edsc::StmtBlockLikeExpr::StmtBlockLikeExpr(ExprKind kind,
                                                 ArrayRef<Expr> exprs,
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
ArrayRef<Expr> mlir::edsc::StmtBlockLikeExpr::getExprs() const {
  return static_cast<ImplType *>(storage)->exprs;
}

mlir::edsc::Stmt::Stmt(const Bindable &lhs, const Expr &rhs,
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

mlir::edsc::Stmt::Stmt(const Expr &rhs, llvm::ArrayRef<Stmt> enclosedStmts)
    : Stmt(Bindable(Expr()), rhs, enclosedStmts) {}

edsc_stmt_t makeStmt(edsc_expr_t e) {
  assert(e && "unexpected empty expression");
  return Stmt(Expr(e));
}

Stmt &mlir::edsc::Stmt::operator=(const Expr &expr) {
  Stmt res(Bindable(Expr()), expr, {});
  std::swap(res.storage, this->storage);
  return *this;
}

Expr mlir::edsc::Stmt::getLHS() const {
  return static_cast<ImplType *>(storage)->lhs;
}

Expr mlir::edsc::Stmt::getRHS() const {
  return static_cast<ImplType *>(storage)->rhs;
}

llvm::ArrayRef<Stmt> mlir::edsc::Stmt::getEnclosedStmts() const {
  return storage->enclosedStmts;
}

void mlir::edsc::Stmt::print(raw_ostream &os, Twine indent) const {
  if (!storage) {
    os << "null_storage";
    return;
  }
  auto lhs = getLHS();
  auto rhs = getRHS();

  if (auto stmtExpr = rhs.dyn_cast<StmtBlockLikeExpr>()) {
    switch (stmtExpr.getKind()) {
    case ExprKind::For:
      os << indent << "for(" << lhs << " = " << stmtExpr << ") {";
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
    case ExprKind::StmtList:
      os << indent << "stmt_list {";
      for (auto &s : getEnclosedStmts()) {
        os << "\n";
        s.print(os, indent + "  ");
      }
      os << "\n" << indent << "}";
      return;
    default: {
      // TODO(ntv): print more statement cases.
      os << "TODO";
    }
    }
  } else {
    os << lhs << " = " << rhs;
  }
}

void mlir::edsc::Stmt::dump() const { this->print(llvm::errs()); }

std::string mlir::edsc::Stmt::str() const {
  std::string res;
  llvm::raw_string_ostream os(res);
  this->print(os);
  return res;
}

llvm::raw_ostream &mlir::edsc::operator<<(llvm::raw_ostream &os,
                                          const Stmt &stmt) {
  stmt.print(os);
  return os;
}

Indexed mlir::edsc::Indexed::operator()(llvm::ArrayRef<Expr> indices) {
  Indexed res(base);
  res.indices = llvm::SmallVector<Expr, 4>(indices.begin(), indices.end());
  return res;
}

// NOLINTNEXTLINE: unconventional-assign-operator
Stmt mlir::edsc::Indexed::operator=(Expr expr) {
  return Stmt(store(expr, base, indices));
}

edsc_indexed_t makeIndexed(edsc_expr_t expr) {
  return edsc_indexed_t{expr, edsc_expr_list_t{nullptr, 0}};
}

edsc_indexed_t index(edsc_indexed_t indexed, edsc_expr_list_t indices) {
  return edsc_indexed_t{indexed.base, indices};
}

mlir_type_t makeScalarType(mlir_context_t context, const char *name,
                           unsigned bitwidth) {
  mlir::MLIRContext *c = reinterpret_cast<mlir::MLIRContext *>(context);
  mlir_type_t res =
      llvm::StringSwitch<mlir_type_t>(name)
          .Case("bf16",
                mlir_type_t{mlir::Type::getBF16(c).getAsOpaquePointer()})
          .Case("f16", mlir_type_t{mlir::Type::getF16(c).getAsOpaquePointer()})
          .Case("f32", mlir_type_t{mlir::Type::getF32(c).getAsOpaquePointer()})
          .Case("f64", mlir_type_t{mlir::Type::getF64(c).getAsOpaquePointer()})
          .Case("index",
                mlir_type_t{mlir::Type::getIndex(c).getAsOpaquePointer()})
          .Case("i",
                mlir_type_t{
                    mlir::Type::getInteger(bitwidth, c).getAsOpaquePointer()})
          .Default(mlir_type_t{nullptr});
  if (!res) {
    llvm_unreachable("Invalid type specifier");
  }
  return res;
}

mlir_type_t makeMemRefType(mlir_context_t context, mlir_type_t elemType,
                           int64_list_t sizes) {
  auto t = mlir::MemRefType::get(
      llvm::ArrayRef<int64_t>(sizes.values, sizes.n),
      mlir::Type::getFromOpaquePointer(elemType),
      {mlir::AffineMap::getMultiDimIdentityMap(
          sizes.n, reinterpret_cast<mlir::MLIRContext *>(context))},
      0);
  return mlir_type_t{t.getAsOpaquePointer()};
}

mlir_type_t makeFunctionType(mlir_context_t context, mlir_type_list_t inputs,
                             mlir_type_list_t outputs) {
  llvm::SmallVector<mlir::Type, 8> ins(inputs.n), outs(outputs.n);
  for (unsigned i = 0; i < inputs.n; ++i) {
    ins[i] = mlir::Type::getFromOpaquePointer(inputs.types[i]);
  }
  for (unsigned i = 0; i < outputs.n; ++i) {
    ins[i] = mlir::Type::getFromOpaquePointer(outputs.types[i]);
  }
  auto ft = mlir::FunctionType::get(
      ins, outs, reinterpret_cast<mlir::MLIRContext *>(context));
  return mlir_type_t{ft.getAsOpaquePointer()};
}

unsigned getFunctionArity(mlir_func_t function) {
  auto *f = reinterpret_cast<mlir::Function *>(function);
  return f->getNumArguments();
}

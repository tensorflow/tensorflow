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
#include "mlir/AffineOps/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/SuperVectorOps/SuperVectorOps.h"
#include "mlir/Support/STLExtras.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using llvm::errs;
using llvm::Twine;

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::detail;

namespace mlir {
namespace edsc {
namespace detail {

template <typename T> ArrayRef<T> copyIntoExprAllocator(ArrayRef<T> elements) {
  if (elements.empty()) {
    return {};
  }
  auto storage = Expr::globalAllocator()->Allocate<T>(elements.size());
  std::uninitialized_copy(elements.begin(), elements.end(), storage);
  return llvm::makeArrayRef(storage, elements.size());
}

struct ExprStorage {
  // Note: this structure is similar to OperationState, but stores lists in a
  // EDSC bump allocator.
  ExprKind kind;
  unsigned id;

  StringRef opName;
  ArrayRef<Expr> operands;
  ArrayRef<Type> resultTypes;
  ArrayRef<NamedAttribute> attributes;

  ExprStorage(ExprKind kind, StringRef name, ArrayRef<Type> results,
              ArrayRef<Expr> children, ArrayRef<NamedAttribute> attrs,
              StringRef descr = "", unsigned exprId = Expr::newId())
      : kind(kind), id(exprId) {
    operands = copyIntoExprAllocator(children);
    resultTypes = copyIntoExprAllocator(results);
    attributes = copyIntoExprAllocator(attrs);
    if (!name.empty()) {
      auto nameStorage = Expr::globalAllocator()->Allocate<char>(name.size());
      std::uninitialized_copy(name.begin(), name.end(), nameStorage);
      opName = StringRef(nameStorage, name.size());
    }
  }
};

struct StmtStorage {
  StmtStorage(Bindable lhs, Expr rhs, llvm::ArrayRef<Stmt> enclosedStmts)
      : lhs(lhs), rhs(rhs), enclosedStmts(enclosedStmts) {}
  Bindable lhs;
  Expr rhs;
  ArrayRef<Stmt> enclosedStmts;
};

struct StmtBlockStorage {
  StmtBlockStorage(ArrayRef<Bindable> args, ArrayRef<Type> argTypes,
                   ArrayRef<Stmt> stmts) {
    arguments = copyIntoExprAllocator(args);
    argumentTypes = copyIntoExprAllocator(argTypes);
    statements = copyIntoExprAllocator(stmts);
  }

  ArrayRef<Bindable> arguments;
  ArrayRef<Type> argumentTypes;
  ArrayRef<Stmt> statements;
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

mlir::edsc::Expr::Expr(Type type) {
  // Initialize with placement new.
  storage = Expr::globalAllocator()->Allocate<detail::ExprStorage>();
  new (storage) detail::ExprStorage(ExprKind::Unbound, "", {type}, {}, {});
}

ExprKind mlir::edsc::Expr::getKind() const { return storage->kind; }

unsigned mlir::edsc::Expr::getId() const {
  return static_cast<ImplType *>(storage)->id;
}

unsigned &mlir::edsc::Expr::newId() {
  static thread_local unsigned id = 0;
  return ++id;
}

ArrayRef<Type> mlir::edsc::Expr::getResultTypes() const {
  return storage->resultTypes;
}

ArrayRef<Expr> mlir::edsc::Expr::getChildExpressions() const {
  return storage->operands;
}

ArrayRef<NamedAttribute> mlir::edsc::Expr::getAttributes() const {
  return storage->attributes;
}

StringRef mlir::edsc::Expr::getName() const {
  return static_cast<ImplType *>(storage)->opName;
}

SmallVector<Value *, 4>
Expr::build(FuncBuilder &b,
            const llvm::DenseMap<Expr, Value *> &ssaBindings) const {
  auto it = ssaBindings.find(*this);
  if (it != ssaBindings.end())
    return {it->second};

  auto *impl = static_cast<ImplType *>(storage);
  auto state = OperationState(b.getContext(), b.getUnknownLoc(), impl->opName);
  SmallVector<Value *, 4> operandValues;
  operandValues.reserve(impl->operands.size());
  for (auto child : impl->operands) {
    auto subResults = child.build(b, ssaBindings);
    assert(subResults.size() == 1 &&
           "expected single-result expression as operand");
    operandValues.push_back(subResults.front());
  }
  state.addOperands(operandValues);
  state.addTypes(impl->resultTypes);
  for (const auto &attr : impl->attributes)
    state.addAttribute(attr.first, attr.second);

  Instruction *inst = b.createOperation(state);
  return llvm::to_vector<4>(inst->getResults());
}

// Create a binary expression between the two arguments emitting `IOp` if
// arguments are integers or vectors/tensors thereof, `FOp` if arguments are
// floating-point or vectors/tensors thereof, and `AffineApplyOp` with an
// expression produced by `affCombiner` if arguments are of the index type.
// Die on unsupported types.
template <typename IOp, typename FOp>
static Expr createBinaryExpr(
    Expr lhs, Expr rhs,
    std::function<AffineExpr(AffineExpr, AffineExpr)> affCombiner) {
  assert(lhs.getResultTypes().size() == 1 && rhs.getResultTypes().size() == 1 &&
         "only single-result exprs are supported in operators");
  auto thisType = lhs.getResultTypes().front();
  auto thatType = rhs.getResultTypes().front();
  assert(thisType == thatType && "cannot mix types in operators");
  if (thisType.isIndex()) {
    MLIRContext *context = thisType.getContext();
    auto d0 = getAffineDimExpr(0, context);
    auto d1 = getAffineDimExpr(1, context);
    auto map = AffineMap::get(2, 0, {affCombiner(d0, d1)}, {});
    auto attr = AffineMapAttr::get(map);
    auto attrId = Identifier::get("map", context);
    auto namedAttr = NamedAttribute{attrId, attr};
    return VariadicExpr("affine.apply", {lhs, rhs}, {IndexType::get(context)},
                        {namedAttr});
  } else if (thisType.isa<IntegerType>()) {
    return BinaryExpr::make<IOp>(thisType, lhs, rhs);
  } else if (thisType.isa<FloatType>()) {
    return BinaryExpr::make<FOp>(thisType, lhs, rhs);
  } else if (auto aggregateType = thisType.dyn_cast<VectorOrTensorType>()) {
    if (aggregateType.getElementType().isa<IntegerType>())
      return BinaryExpr::make<IOp>(thisType, lhs, rhs);
    else if (aggregateType.getElementType().isa<FloatType>())
      return BinaryExpr::make<FOp>(thisType, lhs, rhs);
  }

  llvm_unreachable("failed to create an Expr");
}

Expr mlir::edsc::op::operator+(Expr lhs, Expr rhs) {
  return createBinaryExpr<AddIOp, AddFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 + d1; });
}
Expr mlir::edsc::op::operator-(Expr lhs, Expr rhs) {
  return createBinaryExpr<SubIOp, SubFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 - d1; });
}
Expr mlir::edsc::op::operator*(Expr lhs, Expr rhs) {
  return createBinaryExpr<MulIOp, MulFOp>(
      lhs, rhs, [](AffineExpr d0, AffineExpr d1) { return d0 * d1; });
}

static Expr createComparisonExpr(CmpIPredicate predicate, Expr lhs, Expr rhs) {
  assert(lhs.getResultTypes().size() == 1 && rhs.getResultTypes().size() == 1 &&
         "only single-result exprs are supported in operators");
  auto lhsType = lhs.getResultTypes().front();
  auto rhsType = rhs.getResultTypes().front();
  assert(lhsType == rhsType && "cannot mix types in operators");
  assert((lhsType.isa<IndexType>() || lhsType.isa<IntegerType>()) &&
         "only integer comparisons are supported");

  MLIRContext *context = lhsType.getContext();
  auto attr = IntegerAttr::get(IndexType::get(context),
                               static_cast<int64_t>(predicate));
  auto attrId = Identifier::get(CmpIOp::getPredicateAttrName(), context);
  auto namedAttr = NamedAttribute{attrId, attr};

  return BinaryExpr::make<CmpIOp>(IntegerType::get(1, context), lhs, rhs,
                                  {namedAttr});
}

Expr mlir::edsc::op::operator==(Expr lhs, Expr rhs) {
  return createComparisonExpr(CmpIPredicate::EQ, lhs, rhs);
}
Expr mlir::edsc::op::operator!=(Expr lhs, Expr rhs) {
  return createComparisonExpr(CmpIPredicate::NE, lhs, rhs);
}
Expr mlir::edsc::op::operator<(Expr lhs, Expr rhs) {
  // TODO(ntv,zinenko): signed by default, how about unsigned?
  return createComparisonExpr(CmpIPredicate::SLT, lhs, rhs);
}
Expr mlir::edsc::op::operator<=(Expr lhs, Expr rhs) {
  return createComparisonExpr(CmpIPredicate::SLE, lhs, rhs);
}
Expr mlir::edsc::op::operator>(Expr lhs, Expr rhs) {
  return createComparisonExpr(CmpIPredicate::SGT, lhs, rhs);
}
Expr mlir::edsc::op::operator>=(Expr lhs, Expr rhs) {
  return createComparisonExpr(CmpIPredicate::SGE, lhs, rhs);
}

Expr mlir::edsc::op::operator&&(Expr lhs, Expr rhs) {
  assert(lhs.getResultTypes().size() == 1 && rhs.getResultTypes().size() == 1 &&
         "expected single-result exprs");
  auto thisType = lhs.getResultTypes().front();
  auto thatType = rhs.getResultTypes().front();
  assert(thisType.isInteger(1) && thatType.isInteger(1) &&
         "logical And expects i1");
  return BinaryExpr::make<MulIOp>(thisType, lhs, rhs);
}
Expr mlir::edsc::op::operator||(Expr lhs, Expr rhs) {
  // There is not support for bitwise operations, so we emulate logical 'or'
  //   lhs || rhs
  // as
  //   !(!lhs && !rhs).
  using namespace edsc::op;
  return !(!lhs && !rhs);
}
Expr mlir::edsc::op::operator!(Expr expr) {
  assert(expr.getResultTypes().size() == 1 && "expected single-result exprs");
  auto thisType = expr.getResultTypes().front();
  assert(thisType.isInteger(1) && "logical Not expects i1");
  MLIRContext *context = thisType.getContext();

  // Create constant 1 expression.s
  auto attr = IntegerAttr::get(thisType, 1);
  auto attrId = Identifier::get("value", context);
  auto namedAttr = NamedAttribute{attrId, attr};
  auto cstOne = VariadicExpr("constant", {}, thisType, {namedAttr});

  // Emulate negation as (1 - x) : i1
  return cstOne - expr;
}

llvm::SmallVector<Expr, 8> mlir::edsc::makeNewExprs(unsigned n, Type type) {
  llvm::SmallVector<Expr, 8> res;
  res.reserve(n);
  for (auto i = 0; i < n; ++i) {
    res.push_back(Expr(type));
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
  return VariadicExpr::make<AllocOp>(sizes, memrefType);
}

Expr mlir::edsc::dealloc(Expr memref) {
  return UnaryExpr::make<DeallocOp>(memref);
}

Stmt mlir::edsc::For(Expr lb, Expr ub, Expr step, ArrayRef<Stmt> stmts) {
  assert(lb.getResultTypes().size() == 1 && "expected single-result bounds");
  auto type = lb.getResultTypes().front();
  Expr idx(type);
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

StmtBlock mlir::edsc::block(ArrayRef<Bindable> args, ArrayRef<Type> argTypes,
                            ArrayRef<Stmt> stmts) {
  assert(args.size() == argTypes.size() &&
         "mismatching number of arguments and argument types");
  return StmtBlock(args, argTypes, stmts);
}

edsc_block_t Block(edsc_stmt_list_t enclosedStmts) {
  llvm::SmallVector<Stmt, 8> stmts;
  fillStmts(enclosedStmts, &stmts);
  return StmtBlock(stmts);
}

Expr mlir::edsc::load(Expr m, ArrayRef<Expr> indices) {
  assert(m.getResultTypes().size() == 1 && "expected single-result expr");
  auto type = m.getResultTypes().front().dyn_cast<MemRefType>();
  assert(type && "expected memref type");

  SmallVector<Expr, 8> exprs;
  exprs.push_back(m);
  exprs.append(indices.begin(), indices.end());
  return VariadicExpr::make<LoadOp>(exprs, {type.getElementType()});
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
  return VariadicExpr::make<StoreOp>(exprs);
}

edsc_stmt_t Store(edsc_expr_t value, edsc_indexed_t indexed,
                  edsc_expr_list_t indices) {
  Indexed i(Expr(indexed.base).cast<Bindable>());
  auto exprs = makeExprs(indices);
  Indexed loc = i(exprs);
  return Stmt(loc = Expr(value));
}

Expr mlir::edsc::select(Expr cond, Expr lhs, Expr rhs) {
  return TernaryExpr::make<SelectOp>(cond, lhs, rhs);
}

edsc_expr_t Select(edsc_expr_t cond, edsc_expr_t lhs, edsc_expr_t rhs) {
  return select(Expr(cond), Expr(lhs), Expr(rhs));
}

Expr mlir::edsc::vector_type_cast(Expr memrefExpr, Type memrefType) {
  return VariadicExpr::make<VectorTypeCastOp>({memrefExpr}, {memrefType});
}

Stmt mlir::edsc::Return(ArrayRef<Expr> values) {
  return VariadicExpr::make<ReturnOp>(values);
}

edsc_stmt_t Return(edsc_expr_list_t values) {
  return Stmt(Return(makeExprs(values)));
}

static raw_ostream &printBinaryExpr(raw_ostream &os, BinaryExpr e,
                                    StringRef infix) {
  os << '(' << e.getLHS() << ' ' << infix << ' ' << e.getRHS() << ')';
  return os;
}

// Get the operator spelling for pretty-printing the infix form of a
// comparison operator.
static StringRef getCmpIPredicateInfix(const mlir::edsc::Expr &e) {
  Attribute predicate;
  for (const auto &namedAttr : e.getAttributes()) {
    if (namedAttr.first.is(CmpIOp::getPredicateAttrName())) {
      predicate = namedAttr.second;
      break;
    }
  }
  assert(predicate && "expected a predicate in a comparison expr");

  switch (static_cast<CmpIPredicate>(
      predicate.cast<IntegerAttr>().getValue().getSExtValue())) {
  case CmpIPredicate::EQ:
    return "==";
  case CmpIPredicate::NE:
    return "!=";
  case CmpIPredicate::SGT:
  case CmpIPredicate::UGT:
    return ">";
  case CmpIPredicate::SLT:
  case CmpIPredicate::ULT:
    return "<";
  case CmpIPredicate::SGE:
  case CmpIPredicate::UGE:
    return ">=";
  case CmpIPredicate::SLE:
  case CmpIPredicate::ULE:
    return "<=";
  default:
    llvm_unreachable("unknown predicate");
  }
  return "";
}

static void printAffineExpr(raw_ostream &os, AffineExpr expr,
                            ArrayRef<Expr> dims, ArrayRef<Expr> symbols) {
  struct Visitor : public AffineExprVisitor<Visitor> {
    Visitor(raw_ostream &ostream, ArrayRef<Expr> dimExprs,
            ArrayRef<Expr> symExprs)
        : os(ostream), dims(dimExprs), symbols(symExprs) {}
    raw_ostream &os;
    ArrayRef<Expr> dims;
    ArrayRef<Expr> symbols;

    void visitDimExpr(AffineDimExpr dimExpr) {
      os << dims[dimExpr.getPosition()];
    }

    void visitSymbolExpr(AffineSymbolExpr symbolExpr) {
      os << symbols[symbolExpr.getPosition()];
    }

    void visitConstantExpr(AffineConstantExpr constExpr) {
      os << constExpr.getValue();
    }

    void visitBinaryExpr(AffineBinaryOpExpr expr, StringRef infix) {
      visit(expr.getLHS());
      os << infix;
      visit(expr.getRHS());
    }

    void visitAddExpr(AffineBinaryOpExpr binOp) {
      visitBinaryExpr(binOp, " + ");
    }

    void visitMulExpr(AffineBinaryOpExpr binOp) {
      visitBinaryExpr(binOp, " * ");
    }

    void visitModExpr(AffineBinaryOpExpr binOp) {
      visitBinaryExpr(binOp, " % ");
    }

    void visitCeilDivExpr(AffineBinaryOpExpr binOp) {
      visitBinaryExpr(binOp, " ceildiv ");
    }

    void visitFloorDivExpr(AffineBinaryOpExpr binOp) {
      visitBinaryExpr(binOp, " floordiv ");
    }
  };

  Visitor(os, dims, symbols).visit(expr);
}

static void printAffineMap(raw_ostream &os, AffineMap map,
                           ArrayRef<Expr> operands) {
  auto dims = operands.take_front(map.getNumDims());
  auto symbols = operands.drop_front(map.getNumDims());
  assert(map.getNumResults() == 1 &&
         "only 1-result maps are currently supported");
  printAffineExpr(os, map.getResult(0), dims, symbols);
}

void printAffineApply(raw_ostream &os, mlir::edsc::Expr e) {
  Attribute mapAttr;
  for (const auto &namedAttr : e.getAttributes()) {
    if (namedAttr.first.is("map")) {
      mapAttr = namedAttr.second;
      break;
    }
  }
  assert(mapAttr && "expected a map in an affine apply expr");

  printAffineMap(os, mapAttr.cast<AffineMapAttr>().getValue(),
                 e.getChildExpressions());
}

void mlir::edsc::Expr::print(raw_ostream &os) const {
  if (auto unbound = this->dyn_cast<Bindable>()) {
    os << "$" << unbound.getId();
    return;
  }

  // Handle known binary ops with pretty infix forms.
  if (auto binExpr = this->dyn_cast<BinaryExpr>()) {
    StringRef infix;
    if (binExpr.is_op<AddIOp>() || binExpr.is_op<AddFOp>())
      infix = "+";
    else if (binExpr.is_op<SubIOp>() || binExpr.is_op<SubFOp>())
      infix = "-";
    else if (binExpr.is_op<MulIOp>() || binExpr.is_op<MulFOp>())
      infix = binExpr.getResultTypes().front().isInteger(1) ? "&&" : "*";
    else if (binExpr.is_op<DivISOp>() || binExpr.is_op<DivIUOp>())
      infix = "/";
    else if (binExpr.is_op<RemISOp>() || binExpr.is_op<RemIUOp>())
      infix = "%";
    else if (binExpr.is_op<CmpIOp>())
      infix = getCmpIPredicateInfix(*this);

    if (!infix.empty()) {
      printBinaryExpr(os, binExpr, infix);
      return;
    }
  }

  // Handle known variadic ops with pretty forms.
  if (auto narExpr = this->dyn_cast<VariadicExpr>()) {
    if (narExpr.is_op<LoadOp>()) {
      os << narExpr.getName() << '(' << getChildExpressions().front() << '[';
      interleaveComma(getChildExpressions().drop_front(), os);
      os << "])";
      return;
    }
    if (narExpr.is_op<StoreOp>()) {
      os << narExpr.getName() << '(' << getChildExpressions().front() << ", "
         << getChildExpressions()[1] << '[';
      interleaveComma(getChildExpressions().drop_front(2), os);
      os << "])";
      return;
    }
    if (narExpr.is_op<AffineApplyOp>()) {
      os << '(';
      printAffineApply(os, *this);
      os << ')';
      return;
    }
  }

  // Handle all other types of ops with a more generic printing form.
  if (this->isa<UnaryExpr>() || this->isa<BinaryExpr>() ||
      this->isa<TernaryExpr>() || this->isa<VariadicExpr>()) {
    os << (getName().empty() ? "##unknown##" : getName()) << '(';
    interleaveComma(getChildExpressions(), os);
    auto attrs = getAttributes();
    if (!attrs.empty()) {
      os << '{';
      interleave(
          attrs,
          [&os](const NamedAttribute &attr) {
            os << attr.first.strref() << ": " << attr.second;
          },
          [&os]() { os << ", "; });
      os << '}';
    }
    os << ')';
    return;
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

edsc_expr_t makeBindable(mlir_type_t type) {
  return Bindable(Expr(Type(reinterpret_cast<const Type::ImplType *>(type))));
}

mlir::edsc::UnaryExpr::UnaryExpr(StringRef name, Expr expr)
    : Expr(Expr::globalAllocator()->Allocate<detail::ExprStorage>()) {
  // Initialize with placement new.
  new (storage) detail::ExprStorage(ExprKind::Unary, name, {}, {expr}, {});
}
Expr mlir::edsc::UnaryExpr::getExpr() const {
  return static_cast<ImplType *>(storage)->operands.front();
}

mlir::edsc::BinaryExpr::BinaryExpr(StringRef name, Type result, Expr lhs,
                                   Expr rhs, ArrayRef<NamedAttribute> attrs)
    : Expr(Expr::globalAllocator()->Allocate<detail::ExprStorage>()) {
  // Initialize with placement new.
  new (storage)
      detail::ExprStorage(ExprKind::Binary, name, {result}, {lhs, rhs}, attrs);
}
Expr mlir::edsc::BinaryExpr::getLHS() const {
  return static_cast<ImplType *>(storage)->operands.front();
}
Expr mlir::edsc::BinaryExpr::getRHS() const {
  return static_cast<ImplType *>(storage)->operands.back();
}

mlir::edsc::TernaryExpr::TernaryExpr(StringRef name, Expr cond, Expr lhs,
                                     Expr rhs)
    : Expr(Expr::globalAllocator()->Allocate<detail::ExprStorage>()) {
  // Initialize with placement new.
  assert(lhs.getResultTypes().size() == 1 && "expected single-result expr");
  assert(rhs.getResultTypes().size() == 1 && "expected single-result expr");
  new (storage)
      detail::ExprStorage(ExprKind::Ternary, name,
                          {lhs.getResultTypes().front()}, {cond, lhs, rhs}, {});
}
Expr mlir::edsc::TernaryExpr::getCond() const {
  return static_cast<ImplType *>(storage)->operands[0];
}
Expr mlir::edsc::TernaryExpr::getLHS() const {
  return static_cast<ImplType *>(storage)->operands[1];
}
Expr mlir::edsc::TernaryExpr::getRHS() const {
  return static_cast<ImplType *>(storage)->operands[2];
}

mlir::edsc::VariadicExpr::VariadicExpr(StringRef name, ArrayRef<Expr> exprs,
                                       ArrayRef<Type> types,
                                       ArrayRef<NamedAttribute> attrs)
    : Expr(Expr::globalAllocator()->Allocate<detail::ExprStorage>()) {
  // Initialize with placement new.
  new (storage)
      detail::ExprStorage(ExprKind::Variadic, name, types, exprs, attrs);
}
ArrayRef<Expr> mlir::edsc::VariadicExpr::getExprs() const {
  return static_cast<ImplType *>(storage)->operands;
}
ArrayRef<Type> mlir::edsc::VariadicExpr::getTypes() const {
  return static_cast<ImplType *>(storage)->resultTypes;
}

mlir::edsc::StmtBlockLikeExpr::StmtBlockLikeExpr(ExprKind kind,
                                                 ArrayRef<Expr> exprs,
                                                 ArrayRef<Type> types)
    : Expr(Expr::globalAllocator()->Allocate<detail::ExprStorage>()) {
  // Initialize with placement new.
  new (storage) detail::ExprStorage(kind, "", types, exprs, {});
}
ArrayRef<Expr> mlir::edsc::StmtBlockLikeExpr::getExprs() const {
  return static_cast<ImplType *>(storage)->operands;
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

// Statement with enclosed statements does not have a LHS.
mlir::edsc::Stmt::Stmt(const Expr &rhs, llvm::ArrayRef<Stmt> enclosedStmts)
    : Stmt(Bindable(Expr(Type())), rhs, enclosedStmts) {}

edsc_stmt_t makeStmt(edsc_expr_t e) {
  assert(e && "unexpected empty expression");
  return Stmt(Expr(e));
}

Stmt &mlir::edsc::Stmt::operator=(const Expr &expr) {
  auto types = expr.getResultTypes();
  assert(types.size() == 1 && "single result Expr expected in Stmt::operator=");
  Stmt res(Bindable(Expr(types.front())), expr, {});
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

mlir::edsc::StmtBlock::StmtBlock(llvm::ArrayRef<Stmt> stmts)
    : StmtBlock({}, {}, stmts) {}

mlir::edsc::StmtBlock::StmtBlock(llvm::ArrayRef<Bindable> args,
                                 llvm::ArrayRef<Type> argTypes,
                                 llvm::ArrayRef<Stmt> stmts) {
  storage = Expr::globalAllocator()->Allocate<detail::StmtBlockStorage>();
  new (storage) detail::StmtBlockStorage(args, argTypes, stmts);
}

ArrayRef<mlir::edsc::Bindable> mlir::edsc::StmtBlock::getArguments() const {
  return storage->arguments;
}

ArrayRef<Type> mlir::edsc::StmtBlock::getArgumentTypes() const {
  return storage->argumentTypes;
}

ArrayRef<mlir::edsc::Stmt> mlir::edsc::StmtBlock::getBody() const {
  return storage->statements;
}

void mlir::edsc::StmtBlock::print(llvm::raw_ostream &os, Twine indent) const {
  os << indent << "^bb";
  if (!getArgumentTypes().empty())
    os << '(';
  interleaveComma(getArguments(), os);
  if (!getArgumentTypes().empty())
    os << ')';
  os << ":\n";

  for (auto stmt : getBody())
    stmt.print(os, indent + "  ");
}

std::string mlir::edsc::StmtBlock::str() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  print(os, "");
  return result;
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
                mlir_type_t{mlir::FloatType::getBF16(c).getAsOpaquePointer()})
          .Case("f16",
                mlir_type_t{mlir::FloatType::getF16(c).getAsOpaquePointer()})
          .Case("f32",
                mlir_type_t{mlir::FloatType::getF32(c).getAsOpaquePointer()})
          .Case("f64",
                mlir_type_t{mlir::FloatType::getF64(c).getAsOpaquePointer()})
          .Case("index",
                mlir_type_t{mlir::IndexType::get(c).getAsOpaquePointer()})
          .Case("i",
                mlir_type_t{
                    mlir::IntegerType::get(bitwidth, c).getAsOpaquePointer()})
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
    outs[i] = mlir::Type::getFromOpaquePointer(outputs.types[i]);
  }
  auto ft = mlir::FunctionType::get(
      ins, outs, reinterpret_cast<mlir::MLIRContext *>(context));
  return mlir_type_t{ft.getAsOpaquePointer()};
}

mlir_type_t makeIndexType(mlir_context_t context) {
  auto *ctx = reinterpret_cast<mlir::MLIRContext *>(context);
  auto type = mlir::IndexType::get(ctx);
  return mlir_type_t{type.getAsOpaquePointer()};
}

unsigned getFunctionArity(mlir_func_t function) {
  auto *f = reinterpret_cast<mlir::Function *>(function);
  return f->getNumArguments();
}

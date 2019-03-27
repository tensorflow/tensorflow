//===- MLIREmitter.cpp - MLIR EDSC Emitter Class Implementation -*- C++ -*-===//
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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir-c/Core.h"
#include "mlir/AffineOps/AffineOps.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/EDSC/MLIREmitter.h"
#include "mlir/EDSC/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/SuperVectorOps/SuperVectorOps.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/STLExtras.h"

using llvm::dbgs;
using llvm::errs;

#define DEBUG_TYPE "edsc"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::detail;

static void printDefininingStatement(llvm::raw_ostream &os, Value &v) {
  auto *inst = v.getDefiningOp();
  if (inst) {
    inst->print(os);
    return;
  }
  if (auto forInst = getForInductionVarOwner(&v)) {
    forInst.getOperation()->print(os);
  } else if (auto *bbArg = dyn_cast<BlockArgument>(&v)) {
    os << "block_argument";
  } else {
    os << "unknown_ssa_value";
  }
}

mlir::edsc::MLIREmitter::MLIREmitter(FuncBuilder *builder, Location location)
    : builder(builder), location(location), zeroIndex(builder->getIndexType()),
      oneIndex(builder->getIndexType()) {
  // Build the ubiquitous zero and one at the top of the function.
  bindConstant<ConstantIndexOp>(Bindable(zeroIndex), 0);
  bindConstant<ConstantIndexOp>(Bindable(oneIndex), 1);
}

MLIREmitter &mlir::edsc::MLIREmitter::bind(Bindable e, Value *v) {
  LLVM_DEBUG(printDefininingStatement(llvm::dbgs() << "\nBinding " << e << " @"
                                                   << e.getStoragePtr() << ": ",
                                      *v));
  auto it = ssaBindings.insert(std::make_pair(e, v));
  if (!it.second) {
    printDefininingStatement(llvm::errs() << "\nRebinding " << e << " @"
                                          << e.getStoragePtr() << " ",
                             *v);
    llvm_unreachable("Double binding!");
  }
  return *this;
}

static void checkAffineProvenance(ArrayRef<Value *> values) {
  for (Value *v : values) {
    auto *def = v->getDefiningOp();
    (void)def;
    // There may be no defining instruction if the value is a function
    // argument.  We accept such values.
    assert((!def || def->isa<ConstantIndexOp>() || def->isa<AffineApplyOp>() ||
            def->isa<AffineForOp>() || def->isa<DimOp>()) &&
           "loop bound expression must have affine provenance");
  }
}

static AffineForOp emitStaticFor(FuncBuilder &builder, Location loc,
                                 ArrayRef<Value *> lbs, ArrayRef<Value *> ubs,
                                 uint64_t step) {
  if (lbs.size() != 1 || ubs.size() != 1)
    return AffineForOp();

  auto *lbDef = lbs.front()->getDefiningOp();
  auto *ubDef = ubs.front()->getDefiningOp();
  if (!lbDef || !ubDef)
    return AffineForOp();

  auto lbConst = lbDef->dyn_cast<ConstantIndexOp>();
  auto ubConst = ubDef->dyn_cast<ConstantIndexOp>();
  if (!lbConst || !ubConst)
    return AffineForOp();

  return builder.create<AffineForOp>(loc, lbConst.getValue(),
                                     ubConst.getValue(), step);
}

Value *mlir::edsc::MLIREmitter::emitExpr(Expr e) {
  // It is still necessary in case we try to emit a bindable directly
  // FIXME: make sure isa<Bindable> works and use it below to delegate emission
  // to Expr::build and remove this, now duplicate, check.
  auto it = ssaBindings.find(e);
  if (it != ssaBindings.end()) {
    return it->second;
  }

  Value *res = nullptr;
  bool expectedEmpty = false;
  if (e.isa<UnaryExpr>() || e.isa<BinaryExpr>() || e.isa<TernaryExpr>() ||
      e.isa<VariadicExpr>()) {
    // Emit any successors before the instruction with successors.  At this
    // point, all values defined by the current block must have been bound, the
    // current instruction with successors cannot define new values, so the
    // successor can use those values.
    assert(e.getSuccessors().empty() || e.getResultTypes().empty() &&
                                            "an operation with successors must "
                                            "not have results and vice versa");
    for (StmtBlock block : e.getSuccessors())
      emitBlock(block);
    auto results = e.build(*builder, ssaBindings, blockBindings);
    assert(results.size() <= 1 && "2+-result exprs are not supported");
    expectedEmpty = results.empty();
    if (!results.empty())
      res = results.front();
  }

  if (auto expr = e.dyn_cast<StmtBlockLikeExpr>()) {
    if (expr.getKind() == ExprKind::For) {
      auto exprGroups = expr.getAllArgumentGroups();
      assert(exprGroups.size() == 3 &&
             "expected 3 expr groups in `affine.for`");
      assert(!exprGroups[0].empty() && "expected at least one lower bound");
      assert(!exprGroups[1].empty() && "expected at least one upper bound");
      assert(exprGroups[2].size() == 1 &&
             "the third group (step) must have one element");

      auto lbs = emitExprs(exprGroups[0]);
      auto ubs = emitExprs(exprGroups[1]);
      auto stepExpr = emitExpr(exprGroups[2][0]);

      if (llvm::any_of(lbs, [](Value *v) { return !v; }) ||
          llvm::any_of(ubs, [](Value *v) { return !v; }) || !stepExpr)
        return nullptr;

      checkAffineProvenance(lbs);
      checkAffineProvenance(ubs);

      // Step must be a static constant.
      auto step = stepExpr->getDefiningOp()->cast<ConstantIndexOp>().getValue();

      // Special case with more concise emitted code for static bounds.
      AffineForOp forOp = emitStaticFor(*builder, location, lbs, ubs, step);

      // General case.
      if (!forOp)
        forOp = builder->create<AffineForOp>(
            location, lbs, builder->getMultiDimIdentityMap(lbs.size()), ubs,
            builder->getMultiDimIdentityMap(ubs.size()), step);
      res = forOp.getInductionVar();
    }
  }

  if (!res && !expectedEmpty) {
    // If we hit here it must mean that the Bindables have not all been bound
    // properly. Because EDSCs are currently dynamically typed, it becomes a
    // runtime error.
    e.print(llvm::errs() << "\nError @" << e.getStoragePtr() << ": ");
    auto it = ssaBindings.find(e);
    if (it != ssaBindings.end()) {
      it->second->print(llvm::errs() << "\nError on value: ");
    } else {
      llvm::errs() << "\nUnbound";
    }
    return nullptr;
  }

  auto resIter = ssaBindings.insert(std::make_pair(e, res));
  (void)resIter;
  assert(resIter.second && "insertion failed");
  return res;
}

SmallVector<Value *, 8>
mlir::edsc::MLIREmitter::emitExprs(ArrayRef<Expr> exprs) {
  SmallVector<Value *, 8> res;
  res.reserve(exprs.size());
  for (auto e : exprs) {
    res.push_back(this->emitExpr(e));
    LLVM_DEBUG(
        printDefininingStatement(llvm::dbgs() << "\nEmitted: ", *res.back()));
  }
  return res;
}

mlir::edsc::MLIREmitter &mlir::edsc::MLIREmitter::emitStmt(const Stmt &stmt) {
  auto *block = builder->getBlock();
  auto ip = builder->getInsertionPoint();
  auto *val = emitExpr(stmt.getRHS());
  if (!val) {
    assert((stmt.getRHS().is_op<DeallocOp>() ||
            stmt.getRHS().is_op<StoreOp>() || stmt.getRHS().is_op<ReturnOp>() ||
            stmt.getRHS().is_op<CallIndirectOp>() ||
            stmt.getRHS().is_op<BranchOp>() ||
            stmt.getRHS().is_op<CondBranchOp>()) &&
           "dealloc, store, return, br, cond_br or call_indirect expected as "
           "the only 0-result ops");
    if (stmt.getRHS().is_op<CallIndirectOp>()) {
      assert(
          stmt.getRHS().cast<VariadicExpr>().getTypes().empty() &&
          "function call produced 0 results from a non-zero-result function");
    }
    return *this;
  }
  // Force create a bindable from stmt.lhs and bind it.
  bind(Bindable(stmt.getLHS()), val);
  if (stmt.getRHS().getKind() == ExprKind::For) {
    // Step into the loop.
    builder->setInsertionPointToStart(getForInductionVarOwner(val).getBody());
  }
  emitStmts(stmt.getEnclosedStmts());
  builder->setInsertionPoint(block, ip);

  return *this;
}

void mlir::edsc::MLIREmitter::emitStmts(ArrayRef<Stmt> stmts) {
  for (auto &stmt : stmts) {
    emitStmt(stmt);
  }
}

mlir::edsc::MLIREmitter &
mlir::edsc::MLIREmitter::emitBlock(const StmtBlock &block) {
  // If we have already emitted this block, do nothing.
  if (blockBindings.count(block) != 0)
    return *this;

  // Otherwise, save the current insertion point.
  auto previousBlock = builder->getInsertionBlock();
  auto previousInstr = builder->getInsertionPoint();

  // Create a new IR block and emit the enclosed statements in that block.  Bind
  // the block argument expressions to the arguments of the emitted IR block.
  auto irBlock = builder->createBlock();
  blockBindings.insert({block, irBlock});
  for (const auto &kvp :
       llvm::zip(block.getArguments(), block.getArgumentTypes())) {
    Bindable expr = std::get<0>(kvp);
    assert(expr.getKind() == ExprKind::Unbound &&
           "cannot use bound expressions as block arguments");
    Type type = std::get<1>(kvp);
    bind(expr, irBlock->addArgument(type));
  }
  emitStmts(block.getBody());

  // And finally restore the original insertion point.
  builder->setInsertionPoint(previousBlock, previousInstr);
  return *this;
}

static bool isDynamicSize(int size) { return size < 0; }

/// This function emits the proper Value* at the place of insertion of b,
/// where each value is the proper ConstantOp or DimOp. Returns a vector with
/// these Value*. Note this function does not concern itself with hoisting of
/// constants and will produce redundant IR. Subsequent MLIR simplification
/// passes like LICM and CSE are expected to clean this up.
///
/// More specifically, a MemRefType has a shape vector in which:
///   - constant ranks are embedded explicitly with their value;
///   - symbolic ranks are represented implicitly by -1 and need to be recovered
///     with a DimOp operation.
///
/// Example:
/// When called on:
///
/// ```mlir
///    memref<?x3x4x?x5xf32>
/// ```
///
/// This emits MLIR similar to:
///
/// ```mlir
///    %d0 = dim %0, 0 : memref<?x3x4x?x5xf32>
///    %c3 = constant 3 : index
///    %c4 = constant 4 : index
///    %d3 = dim %0, 3 : memref<?x3x4x?x5xf32>
///    %c5 = constant 5 : index
/// ```
///
/// and returns the vector with {%d0, %c3, %c4, %d3, %c5}.
static SmallVector<Value *, 8> getMemRefSizes(FuncBuilder *b, Location loc,
                                              Value *memRef) {
  assert(memRef->getType().isa<MemRefType>() && "Expected a MemRef value");
  MemRefType memRefType = memRef->getType().cast<MemRefType>();
  SmallVector<Value *, 8> res;
  res.reserve(memRefType.getShape().size());
  const auto &shape = memRefType.getShape();
  for (unsigned idx = 0, n = shape.size(); idx < n; ++idx) {
    if (isDynamicSize(shape[idx])) {
      res.push_back(b->create<DimOp>(loc, memRef, idx));
    } else {
      res.push_back(b->create<ConstantIndexOp>(loc, shape[idx]));
    }
  }
  return res;
}

SmallVector<edsc::Expr, 8>
mlir::edsc::MLIREmitter::makeBoundFunctionArguments(mlir::Function *function) {
  SmallVector<edsc::Expr, 8> res;
  for (unsigned pos = 0, npos = function->getNumArguments(); pos < npos;
       ++pos) {
    auto *arg = function->getArgument(pos);
    Expr b(arg->getType());
    bind(Bindable(b), arg);
    res.push_back(Expr(b));
  }
  return res;
}

SmallVector<edsc::Expr, 8>
mlir::edsc::MLIREmitter::makeBoundMemRefShape(Value *memRef) {
  assert(memRef->getType().isa<MemRefType>() && "Expected a MemRef value");
  MemRefType memRefType = memRef->getType().cast<MemRefType>();
  auto memRefSizes =
      edsc::makeNewExprs(memRefType.getShape().size(), builder->getIndexType());
  auto memrefSizeValues = getMemRefSizes(getBuilder(), getLocation(), memRef);
  assert(memrefSizeValues.size() == memRefSizes.size());
  bindZipRange(llvm::zip(memRefSizes, memrefSizeValues));
  SmallVector<edsc::Expr, 8> res(memRefSizes.begin(), memRefSizes.end());
  return res;
}

mlir::edsc::MLIREmitter::BoundMemRefView
mlir::edsc::MLIREmitter::makeBoundMemRefView(Value *memRef) {
  auto memRefType = memRef->getType().cast<mlir::MemRefType>();
  auto rank = memRefType.getRank();

  SmallVector<edsc::Expr, 8> lbs;
  lbs.reserve(rank);
  Expr zero(builder->getIndexType());
  bindConstant<mlir::ConstantIndexOp>(Bindable(zero), 0);
  for (unsigned i = 0; i < rank; ++i) {
    lbs.push_back(zero);
  }

  auto ubs = makeBoundMemRefShape(memRef);

  SmallVector<edsc::Expr, 8> steps;
  lbs.reserve(rank);
  Expr one(builder->getIndexType());
  bindConstant<mlir::ConstantIndexOp>(Bindable(one), 1);
  for (unsigned i = 0; i < rank; ++i) {
    steps.push_back(one);
  }

  return BoundMemRefView{lbs, ubs, steps};
}

mlir::edsc::MLIREmitter::BoundMemRefView
mlir::edsc::MLIREmitter::makeBoundMemRefView(Expr boundMemRef) {
  auto *v = getValue(mlir::edsc::Expr(boundMemRef));
  assert(v && "Expected a bound Expr");
  return makeBoundMemRefView(v);
}

AffineForOp mlir::edsc::MLIREmitter::getAffineForOp(Expr e) {
  auto *value = ssaBindings.lookup(e);
  assert(value && "Expr not bound");
  return getForInductionVarOwner(value);
}

edsc_expr_t bindConstantBF16(edsc_mlir_emitter_t emitter, double value) {
  auto *e = reinterpret_cast<mlir::edsc::MLIREmitter *>(emitter);
  Expr b(e->getBuilder()->getBF16Type());
  e->bindConstant<mlir::ConstantFloatOp>(Bindable(b), mlir::APFloat(value),
                                         e->getBuilder()->getBF16Type());
  return b;
}

edsc_expr_t bindConstantF16(edsc_mlir_emitter_t emitter, float value) {
  auto *e = reinterpret_cast<mlir::edsc::MLIREmitter *>(emitter);
  Expr b(e->getBuilder()->getBF16Type());
  bool unused;
  mlir::APFloat val(value);
  val.convert(e->getBuilder()->getF16Type().getFloatSemantics(),
              mlir::APFloat::rmNearestTiesToEven, &unused);
  e->bindConstant<mlir::ConstantFloatOp>(Bindable(b), val,
                                         e->getBuilder()->getF16Type());
  return b;
}

edsc_expr_t bindConstantF32(edsc_mlir_emitter_t emitter, float value) {
  auto *e = reinterpret_cast<mlir::edsc::MLIREmitter *>(emitter);
  Expr b(e->getBuilder()->getF32Type());
  e->bindConstant<mlir::ConstantFloatOp>(Bindable(b), mlir::APFloat(value),
                                         e->getBuilder()->getF32Type());
  return b;
}

edsc_expr_t bindConstantF64(edsc_mlir_emitter_t emitter, double value) {
  auto *e = reinterpret_cast<mlir::edsc::MLIREmitter *>(emitter);
  Expr b(e->getBuilder()->getF64Type());
  e->bindConstant<mlir::ConstantFloatOp>(Bindable(b), mlir::APFloat(value),
                                         e->getBuilder()->getF64Type());
  return b;
}

edsc_expr_t bindConstantInt(edsc_mlir_emitter_t emitter, int64_t value,
                            unsigned bitwidth) {
  auto *e = reinterpret_cast<mlir::edsc::MLIREmitter *>(emitter);
  Expr b(e->getBuilder()->getIntegerType(bitwidth));
  e->bindConstant<mlir::ConstantIntOp>(
      b, value, e->getBuilder()->getIntegerType(bitwidth));
  return b;
}

edsc_expr_t bindConstantIndex(edsc_mlir_emitter_t emitter, int64_t value) {
  auto *e = reinterpret_cast<mlir::edsc::MLIREmitter *>(emitter);
  Expr b(e->getBuilder()->getIndexType());
  e->bindConstant<mlir::ConstantIndexOp>(Bindable(b), value);
  return b;
}

edsc_expr_t bindConstantFunction(edsc_mlir_emitter_t emitter,
                                 mlir_func_t function) {
  auto *e = reinterpret_cast<mlir::edsc::MLIREmitter *>(emitter);
  auto *f = reinterpret_cast<mlir::Function *>(function);
  Expr b(f->getType());
  e->bindConstant<mlir::ConstantOp>(Bindable(b),
                                    e->getBuilder()->getFunctionAttr(f));
  return b;
}

unsigned getRankOfFunctionArgument(mlir_func_t function, unsigned pos) {
  auto *f = reinterpret_cast<mlir::Function *>(function);
  assert(pos < f->getNumArguments());
  auto *arg = *(f->getArguments().begin() + pos);
  if (auto memRefType = arg->getType().dyn_cast<mlir::MemRefType>()) {
    return memRefType.getRank();
  }
  return 0;
}

mlir_type_t getTypeOfFunctionArgument(mlir_func_t function, unsigned pos) {
  auto *f = reinterpret_cast<mlir::Function *>(function);
  assert(pos < f->getNumArguments());
  auto *arg = *(f->getArguments().begin() + pos);
  return mlir_type_t{arg->getType().getAsOpaquePointer()};
}

edsc_expr_t bindFunctionArgument(edsc_mlir_emitter_t emitter,
                                 mlir_func_t function, unsigned pos) {
  auto *e = reinterpret_cast<mlir::edsc::MLIREmitter *>(emitter);
  auto *f = reinterpret_cast<mlir::Function *>(function);
  assert(pos < f->getNumArguments());
  auto *arg = *(f->getArguments().begin() + pos);
  Expr b(arg->getType());
  e->bind(Bindable(b), arg);
  return Expr(b);
}

void bindFunctionArguments(edsc_mlir_emitter_t emitter, mlir_func_t function,
                           edsc_expr_list_t *result) {
  auto *e = reinterpret_cast<mlir::edsc::MLIREmitter *>(emitter);
  auto *f = reinterpret_cast<mlir::Function *>(function);
  assert(result->n == f->getNumArguments());
  for (unsigned pos = 0; pos < result->n; ++pos) {
    auto *arg = *(f->getArguments().begin() + pos);
    Expr b(arg->getType());
    e->bind(Bindable(b), arg);
    result->exprs[pos] = Expr(b);
  }
}

unsigned getBoundMemRefRank(edsc_mlir_emitter_t emitter,
                            edsc_expr_t boundMemRef) {
  auto *e = reinterpret_cast<mlir::edsc::MLIREmitter *>(emitter);
  auto *v = e->getValue(mlir::edsc::Expr(boundMemRef));
  assert(v && "Expected a bound Expr");
  auto memRefType = v->getType().cast<mlir::MemRefType>();
  return memRefType.getRank();
}

void bindMemRefShape(edsc_mlir_emitter_t emitter, edsc_expr_t boundMemRef,
                     edsc_expr_list_t *result) {
  auto *e = reinterpret_cast<mlir::edsc::MLIREmitter *>(emitter);
  auto *v = e->getValue(mlir::edsc::Expr(boundMemRef));
  assert(v && "Expected a bound Expr");
  auto memRefType = v->getType().cast<mlir::MemRefType>();
  auto rank = memRefType.getRank();
  assert(result->n == rank && "Unexpected memref shape binding results count");
  auto bindables = e->makeBoundMemRefShape(v);
  for (unsigned i = 0; i < rank; ++i) {
    result->exprs[i] = bindables[i];
  }
}

void bindMemRefView(edsc_mlir_emitter_t emitter, edsc_expr_t boundMemRef,
                    edsc_expr_list_t *resultLbs, edsc_expr_list_t *resultUbs,
                    edsc_expr_list_t *resultSteps) {
  auto *e = reinterpret_cast<mlir::edsc::MLIREmitter *>(emitter);
  auto *v = e->getValue(mlir::edsc::Expr(boundMemRef));
  auto memRefType = v->getType().cast<mlir::MemRefType>();
  auto rank = memRefType.getRank();
  assert(resultLbs->n == rank && "Unexpected memref binding results count");
  assert(resultUbs->n == rank && "Unexpected memref binding results count");
  assert(resultSteps->n == rank && "Unexpected memref binding results count");
  auto bindables = e->makeBoundMemRefShape(v);
  Expr zero(e->getBuilder()->getIndexType());
  e->bindConstant<mlir::ConstantIndexOp>(zero, 0);
  Expr one(e->getBuilder()->getIndexType());
  e->bindConstant<mlir::ConstantIndexOp>(one, 1);
  for (unsigned i = 0; i < rank; ++i) {
    resultLbs->exprs[i] = zero;
    resultUbs->exprs[i] = bindables[i];
    resultSteps->exprs[i] = one;
  }
}

#define DEFINE_EDSL_BINARY_OP(FUN_NAME, OP_SYMBOL)                             \
  edsc_expr_t FUN_NAME(edsc_expr_t e1, edsc_expr_t e2) {                       \
    using edsc::op::operator OP_SYMBOL;                                        \
    return Expr(e1) OP_SYMBOL Expr(e2);                                        \
  }

DEFINE_EDSL_BINARY_OP(Add, +);
DEFINE_EDSL_BINARY_OP(Sub, -);
DEFINE_EDSL_BINARY_OP(Mul, *);
DEFINE_EDSL_BINARY_OP(Div, /);
DEFINE_EDSL_BINARY_OP(Rem, %);
DEFINE_EDSL_BINARY_OP(LT, <);
DEFINE_EDSL_BINARY_OP(LE, <=);
DEFINE_EDSL_BINARY_OP(GT, >);
DEFINE_EDSL_BINARY_OP(GE, >=);
DEFINE_EDSL_BINARY_OP(EQ, ==);
DEFINE_EDSL_BINARY_OP(NE, !=);
DEFINE_EDSL_BINARY_OP(And, &&);
DEFINE_EDSL_BINARY_OP(Or, ||);

#undef DEFINE_EDSL_BINARY_OP

edsc_expr_t FloorDiv(edsc_expr_t e1, edsc_expr_t e2) {
  return edsc::floorDiv(Expr(e1), Expr(e2));
}

edsc_expr_t CeilDiv(edsc_expr_t e1, edsc_expr_t e2) {
  return edsc::ceilDiv(Expr(e1), Expr(e2));
}

#define DEFINE_EDSL_UNARY_OP(FUN_NAME, OP_SYMBOL)                              \
  edsc_expr_t FUN_NAME(edsc_expr_t e) {                                        \
    using edsc::op::operator OP_SYMBOL;                                        \
    return (OP_SYMBOL(Expr(e)));                                               \
  }

DEFINE_EDSL_UNARY_OP(Negate, !);

#undef DEFINE_EDSL_UNARY_OP

edsc_expr_t Call0(edsc_expr_t callee, edsc_expr_list_t args) {
  SmallVector<Expr, 8> exprArgs;
  exprArgs.reserve(args.n);
  for (int i = 0; i < args.n; ++i) {
    exprArgs.push_back(Expr(args.exprs[i]));
  }
  return edsc::call(Expr(callee), exprArgs);
}

edsc_expr_t Call1(edsc_expr_t callee, mlir_type_t result,
                  edsc_expr_list_t args) {
  SmallVector<Expr, 8> exprArgs;
  exprArgs.reserve(args.n);
  for (int i = 0; i < args.n; ++i) {
    exprArgs.push_back(Expr(args.exprs[i]));
  }
  return edsc::call(
      Expr(callee),
      Type::getFromOpaquePointer(reinterpret_cast<const void *>(result)),
      exprArgs);
}

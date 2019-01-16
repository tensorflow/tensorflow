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

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/EDSC/MLIREmitter.h"
#include "mlir/EDSC/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Instructions.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/SuperVectorOps/SuperVectorOps.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/STLExtras.h"

using llvm::dbgs;
using llvm::errs;

#define DEBUG_TYPE "edsc"

namespace mlir {
namespace edsc {

// Factors out the boilerplate that is needed to build and answer the
// following simple question:
//   Given a set of Value* `values`, how do I get the resulting op(`values`)
//
// This is a very loaded question and generally cannot be answered properly.
// For instance, an LLVM operation has many attributes that may not fit within
// this simplistic framing (e.g. overflow behavior etc).
//
// Still, MLIR is a higher-level IR and the Halide experience shows it is
// possible to build useful EDSCs with the right amount of sugar.
//
// To build EDSCs we need to be able to conveniently support simple operations
// such as `add` on the type system. This captures the possible behaviors. In
// the future, this should be automatically constructed from an abstraction
// that is common to the IR verifier, but for now we need to get off the ground
// manually.
//
// This is expected to be a "dialect-specific" functionality: certain dialects
// will not have a simple definition. Two such cases that come to mind are:
//   1. what does it mean to have an operator* on an opaque tensor dialect
//      (dot, vector, hadamard, kronecker ?)-product;
//   2. LLVM add with attributes like overflow.
// This is all left for future consideration; in the meantime let's separate
// concerns and implement useful infrastructure without solving all problems at
// once.

/// Returns the element type if the type is VectorType or MemRefType; returns
/// getType if the type is scalar.
static Type getElementType(const Value &v) {
  if (auto vec = v.getType().dyn_cast<mlir::VectorType>()) {
    return vec.getElementType();
  }
  if (auto mem = v.getType().dyn_cast<mlir::MemRefType>()) {
    return mem.getElementType();
  }
  return v.getType();
}

static bool isIndexElement(const Value &v) {
  return getElementType(v).isIndex();
}
static bool isIntElement(const Value &v) {
  return getElementType(v).isa<IntegerType>();
}
static bool isFloatElement(const Value &v) {
  return getElementType(v).isa<FloatType>();
}

Value *add(FuncBuilder *builder, Location location, Value *a, Value *b) {
  if (isIndexElement(*a)) {
    auto *context = builder->getContext();
    auto d0 = getAffineDimExpr(0, context);
    auto d1 = getAffineDimExpr(1, context);
    auto map = AffineMap::get(2, 0, {d0 + d1}, {});
    return makeSingleValueFromComposedAffineApply(builder, location, map,
                                                  {a, b});
  } else if (isIntElement(*a)) {
    return builder->create<AddIOp>(location, a, b)->getResult();
  }
  assert(isFloatElement(*a) && "Expected float element");
  return builder->create<AddFOp>(location, a, b)->getResult();
}

Value *sub(FuncBuilder *builder, Location location, Value *a, Value *b) {
  if (isIndexElement(*a)) {
    auto *context = builder->getContext();
    auto d0 = getAffineDimExpr(0, context);
    auto d1 = getAffineDimExpr(1, context);
    auto map = AffineMap::get(2, 0, {d0 - d1}, {});
    return makeSingleValueFromComposedAffineApply(builder, location, map,
                                                  {a, b});
  } else if (isIntElement(*a)) {
    return builder->create<SubIOp>(location, a, b)->getResult();
  }
  assert(isFloatElement(*a) && "Expected float element");
  return builder->create<SubFOp>(location, a, b)->getResult();
}

Value *mul(FuncBuilder *builder, Location location, Value *a, Value *b) {
  if (!isFloatElement(*a)) {
    return builder->create<MulIOp>(location, a, b)->getResult();
  }
  assert(isFloatElement(*a) && "Expected float element");
  return builder->create<MulFOp>(location, a, b)->getResult();
}

static void printDefininingStatement(llvm::raw_ostream &os, const Value &v) {
  const auto *inst = v.getDefiningInst();
  if (inst) {
    inst->print(os);
    return;
  }
  // &v is required here otherwise we get:
  //  non-pointer operand type 'const mlir::ForInst' incompatible with nullptr
  if (auto *forInst = dyn_cast<ForInst>(&v)) {
    forInst->print(os);
  } else {
    os << "unknown_ssa_value";
  }
}

MLIREmitter &MLIREmitter::bind(Bindable e, Value *v) {
  LLVM_DEBUG(printDefininingStatement(llvm::dbgs() << "\nBinding " << e << " @"
                                                   << e.getStoragePtr() << ": ",
                                      *v));
  auto it = ssaBindings.insert(std::make_pair(e, v));
  if (!it.second) {
    printDefininingStatement(
        llvm::errs() << "\nRebinding " << e << " @" << e.getStoragePtr(), *v);
    llvm_unreachable("Double binding!");
  }
  return *this;
}

Value *MLIREmitter::emit(Expr e) {
  auto it = ssaBindings.find(e);
  if (it != ssaBindings.end()) {
    return it->second;
  }

  // Skip bindables, they must have been found already.
  Value *res = nullptr;
  if (auto un = e.dyn_cast<UnaryExpr>()) {
    if (un.getKind() == ExprKind::Dealloc) {
      builder->create<DeallocOp>(location, emit(un.getExpr()));
      return nullptr;
    }
  } else if (auto bin = e.dyn_cast<BinaryExpr>()) {
    auto *a = emit(bin.getLHS());
    auto *b = emit(bin.getRHS());
    if (!a || !b) {
      return nullptr;
    }
    if (bin.getKind() == ExprKind::Add) {
      res = add(builder, location, a, b);
    } else if (bin.getKind() == ExprKind::Sub) {
      res = sub(builder, location, a, b);
    } else if (bin.getKind() == ExprKind::Mul) {
      res = mul(builder, location, a, b);
    }
    // Vanilla comparisons operators.
    // else if (bin.getKind() == ExprKind::And) {
    //   // impl i1
    //   res = add(builder, location, a, b); // MulIOp on i1
    // }
    // else if (bin.getKind() == ExprKind::Not) {
    //   res = ...; // 1 - cast<i1>()
    // }
    // else if (bin.getKind() == ExprKind::Or) {
    //   res = ...; // not(not(a) and not(b))
    // }

    // TODO(ntv): signed vs unsiged ??
    // TODO(ntv): integer vs not ??
    // TODO(ntv): float cmp
    else if (bin.getKind() == ExprKind::EQ) {
      res = builder->create<CmpIOp>(location, mlir::CmpIPredicate::EQ, a, b);
    } else if (bin.getKind() == ExprKind::NE) {
      res = builder->create<CmpIOp>(location, mlir::CmpIPredicate::NE, a, b);
    } else if (bin.getKind() == ExprKind::LT) {
      res = builder->create<CmpIOp>(location, mlir::CmpIPredicate::SLT, a, b);
    } else if (bin.getKind() == ExprKind::LE) {
      res = builder->create<CmpIOp>(location, mlir::CmpIPredicate::SLE, a, b);
    } else if (bin.getKind() == ExprKind::GT) {
      res = builder->create<CmpIOp>(location, mlir::CmpIPredicate::SGT, a, b);
    } else if (bin.getKind() == ExprKind::GE) {
      res = builder->create<CmpIOp>(location, mlir::CmpIPredicate::SGE, a, b);
    }

    // TODO(ntv): do we want this?
    //   if (res && ((a->type().is_uint() && !b->type().is_uint()) ||
    //               (!a->type().is_uint() && b->type().is_uint()))) {
    //     std::stringstream ss;
    //     ss << "a: " << *a << "\t b: " << *b;
    //     res->getDefiningOperation()->emitWarning(
    //         "Mixing signed and unsigned integers: " + ss.str());
    //   }
    // }
  }

  if (auto ter = e.dyn_cast<TernaryExpr>()) {
    if (ter.getKind() == ExprKind::Select) {
      auto *cond = emit(ter.getCond());
      auto *lhs = emit(ter.getLHS());
      auto *rhs = emit(ter.getRHS());
      if (!cond || !rhs || !lhs) {
        return nullptr;
      }
      res = builder->create<SelectOp>(location, cond, lhs, rhs)->getResult();
    }
  }

  if (auto nar = e.dyn_cast<VariadicExpr>()) {
    if (nar.getKind() == ExprKind::Alloc) {
      auto exprs = emit(nar.getExprs());
      if (llvm::any_of(exprs, [](Value *v) { return !v; })) {
        return nullptr;
      }
      auto types = nar.getTypes();
      assert(types.size() == 1 && "Expected 1 type");
      res =
          builder->create<AllocOp>(location, types[0].cast<MemRefType>(), exprs)
              ->getResult();
    } else if (nar.getKind() == ExprKind::Load) {
      auto exprs = emit(nar.getExprs());
      if (llvm::any_of(exprs, [](Value *v) { return !v; })) {
        return nullptr;
      }
      assert(exprs.size() > 1 && "Expected > 1 expr");
      assert(nar.getTypes().empty() && "Expected no type");
      SmallVector<Value *, 8> vals(exprs.begin() + 1, exprs.end());
      res = builder->create<LoadOp>(location, exprs[0], vals)->getResult();
    } else if (nar.getKind() == ExprKind::Store) {
      auto exprs = emit(nar.getExprs());
      if (llvm::any_of(exprs, [](Value *v) { return !v; })) {
        return nullptr;
      }
      assert(exprs.size() > 2 && "Expected > 2 expr");
      assert(nar.getTypes().empty() && "Expected no type");
      SmallVector<Value *, 8> vals(exprs.begin() + 2, exprs.end());
      builder->create<StoreOp>(location, exprs[0], exprs[1], vals);
      return nullptr;
    } else if (nar.getKind() == ExprKind::VectorTypeCast) {
      auto exprs = emit(nar.getExprs());
      if (llvm::any_of(exprs, [](Value *v) { return !v; })) {
        return nullptr;
      }
      assert(exprs.size() == 1 && "Expected 1 expr");
      auto types = nar.getTypes();
      assert(types.size() == 1 && "Expected 1 type");
      res = builder
                ->create<VectorTypeCastOp>(location, exprs[0],
                                           types[0].cast<MemRefType>())
                ->getResult();
    }
  }

  if (auto expr = e.dyn_cast<StmtBlockLikeExpr>()) {
    if (expr.getKind() == ExprKind::For) {
      auto exprs = emit(expr.getExprs());
      if (llvm::any_of(exprs, [](Value *v) { return !v; })) {
        return nullptr;
      }
      assert(exprs.size() == 3 && "Expected 3 exprs");
      assert(expr.getTypes().empty() && "Expected no type");
      auto lb =
          exprs[0]->getDefiningInst()->cast<ConstantIndexOp>()->getValue();
      auto ub =
          exprs[1]->getDefiningInst()->cast<ConstantIndexOp>()->getValue();
      auto step =
          exprs[2]->getDefiningInst()->cast<ConstantIndexOp>()->getValue();
      res = builder->createFor(location, lb, ub, step);
    }
  }

  if (!res) {
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

SmallVector<Value *, 8> MLIREmitter::emit(ArrayRef<Expr> exprs) {
  return mlir::functional::map(
      [this](Expr e) {
        auto *res = this->emit(e);
        LLVM_DEBUG(
            printDefininingStatement(llvm::dbgs() << "\nEmitted: ", *res));
        return res;
      },
      exprs);
}

void MLIREmitter::emitStmt(const Stmt &stmt) {
  auto *block = builder->getBlock();
  auto ip = builder->getInsertionPoint();
  // Blocks are just a containing abstraction, they do not emit their RHS.
  if (stmt.getRHS().getKind() != ExprKind::Block) {
    auto *val = emit(stmt.getRHS());
    if (!val) {
      assert((stmt.getRHS().getKind() == ExprKind::Dealloc ||
              stmt.getRHS().getKind() == ExprKind::Store) &&
             "dealloc or store expected as the only 0-result ops");
      return;
    }
    bind(stmt.getLHS(), val);
    if (stmt.getRHS().getKind() == ExprKind::For) {
      // Step into the loop.
      builder->setInsertionPointToStart(cast<ForInst>(val)->getBody());
    }
  }
  emitStmts(stmt.getEnclosedStmts());
  builder->setInsertionPoint(block, ip);
}

void MLIREmitter::emitStmts(ArrayRef<Stmt> stmts) {
  for (auto &stmt : stmts) {
    emitStmt(stmt);
  }
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
  auto memRefType = memRef->getType().template cast<MemRefType>();
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

SmallVector<edsc::Bindable, 8> MLIREmitter::makeBoundSizes(Value *memRef) {
  assert(memRef->getType().isa<MemRefType>() && "Expected a MemRef value");
  MemRefType memRefType = memRef->getType().cast<MemRefType>();
  auto memRefSizes = edsc::makeBindables(memRefType.getShape().size());
  auto memrefSizeValues = getMemRefSizes(getBuilder(), getLocation(), memRef);
  assert(memrefSizeValues.size() == memRefSizes.size());
  bindZipRange(llvm::zip(memRefSizes, memrefSizeValues));
  return memRefSizes;
}

} // namespace edsc
} // namespace mlir

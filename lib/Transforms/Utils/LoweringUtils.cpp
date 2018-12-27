//===- LoweringUtils.cpp - Utilities for Lowering Passes ------------------===//
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
// This file implements utility functions for lowering passes, for example
// lowering affine_apply operations to individual components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/LoweringUtils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace {
// Visit affine expressions recursively and build the sequence of instructions
// that correspond to it.  Visitation functions return an Value of the
// expression subtree they visited or `nullptr` on error.
class AffineApplyExpander
    : public AffineExprVisitor<AffineApplyExpander, Value *> {
public:
  // This internal clsas expects arguments to be non-null, checks must be
  // performed at the call site.
  AffineApplyExpander(FuncBuilder *builder, AffineApplyOp *op)
      : builder(*builder), applyOp(*op), loc(op->getLoc()) {}

  template <typename OpTy> Value *buildBinaryExpr(AffineBinaryOpExpr expr) {
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    if (!lhs || !rhs)
      return nullptr;
    auto op = builder.create<OpTy>(loc, lhs, rhs);
    return op->getResult();
  }

  Value *visitAddExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<AddIOp>(expr);
  }

  Value *visitMulExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<MulIOp>(expr);
  }

  // TODO(zinenko): implement when the standard operators are made available.
  Value *visitModExpr(AffineBinaryOpExpr) {
    builder.getContext()->emitError(loc, "unsupported binary operator: mod");
    return nullptr;
  }

  Value *visitFloorDivExpr(AffineBinaryOpExpr) {
    builder.getContext()->emitError(loc,
                                    "unsupported binary operator: floor_div");
    return nullptr;
  }

  Value *visitCeilDivExpr(AffineBinaryOpExpr) {
    builder.getContext()->emitError(loc,
                                    "unsupported binary operator: ceil_div");
    return nullptr;
  }

  Value *visitConstantExpr(AffineConstantExpr expr) {
    auto valueAttr =
        builder.getIntegerAttr(builder.getIndexType(), expr.getValue());
    auto op =
        builder.create<ConstantOp>(loc, valueAttr, builder.getIndexType());
    return op->getResult();
  }

  Value *visitDimExpr(AffineDimExpr expr) {
    assert(expr.getPosition() < applyOp.getNumOperands() &&
           "affine dim position out of range");
    // FIXME: this assumes a certain order of AffineApplyOp operands, the
    // cleaner interface would be to separate them at the op level.
    return applyOp.getOperand(expr.getPosition());
  }

  Value *visitSymbolExpr(AffineSymbolExpr expr) {
    // FIXME: this assumes a certain order of AffineApplyOp operands, the
    // cleaner interface would be to separate them at the op level.
    assert(expr.getPosition() + applyOp.getAffineMap().getNumDims() <
               applyOp.getNumOperands() &&
           "symbol dim position out of range");
    return applyOp.getOperand(expr.getPosition() +
                              applyOp.getAffineMap().getNumDims());
  }

private:
  FuncBuilder &builder;
  AffineApplyOp &applyOp;

  Location loc;
};
} // namespace

// Given an affine expression `expr` extracted from `op`, build the sequence of
// primitive instructions that correspond to the affine expression in the
// `builder`.
static mlir::Value *expandAffineExpr(FuncBuilder *builder, AffineExpr expr,
                                     AffineApplyOp *op) {
  auto expander = AffineApplyExpander(builder, op);
  return expander.visit(expr);
}

bool mlir::expandAffineApply(AffineApplyOp *op) {
  if (!op)
    return true;

  FuncBuilder builder(cast<OperationStmt>(op->getOperation()));
  auto affineMap = op->getAffineMap();
  for (auto numberedExpr : llvm::enumerate(affineMap.getResults())) {
    Value *expanded = expandAffineExpr(&builder, numberedExpr.value(), op);
    if (!expanded)
      return true;
    op->getResult(numberedExpr.index())->replaceAllUsesWith(expanded);
  }
  op->erase();
  return false;
}

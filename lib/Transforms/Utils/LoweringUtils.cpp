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
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace {
// Visit affine expressions recursively and build the sequence of instructions
// that correspond to it.  Visitation functions return an Value of the
// expression subtree they visited or `nullptr` on error.
class AffineApplyExpander
    : public AffineExprVisitor<AffineApplyExpander, Value *> {
public:
  // This internal class expects arguments to be non-null, checks must be
  // performed at the call site.
  AffineApplyExpander(FuncBuilder *builder, ArrayRef<Value *> dimValues,
                      ArrayRef<Value *> symbolValues, Location loc)
      : builder(*builder), dimValues(dimValues), symbolValues(symbolValues),
        loc(loc) {}

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
    assert(expr.getPosition() < dimValues.size() &&
           "affine dim position out of range");
    return dimValues[expr.getPosition()];
  }

  Value *visitSymbolExpr(AffineSymbolExpr expr) {
    assert(expr.getPosition() < symbolValues.size() &&
           "symbol dim position out of range");
    return symbolValues[expr.getPosition()];
  }

private:
  FuncBuilder &builder;
  ArrayRef<Value *> dimValues;
  ArrayRef<Value *> symbolValues;

  Location loc;
};
} // namespace

// Create a sequence of instructions that implement the `expr` applied to the
// given dimension and symbol values.
mlir::Value *mlir::expandAffineExpr(FuncBuilder *builder, Location loc,
                                    AffineExpr expr,
                                    ArrayRef<Value *> dimValues,
                                    ArrayRef<Value *> symbolValues) {
  return AffineApplyExpander(builder, dimValues, symbolValues, loc).visit(expr);
}

// Create a sequence of instructions that implement the `affineMap` applied to
// the given `operands` (as it it were an AffineApplyOp).
Optional<SmallVector<Value *, 8>>
mlir::expandAffineMap(FuncBuilder *builder, Location loc, AffineMap affineMap,
                      ArrayRef<Value *> operands) {
  auto numDims = affineMap.getNumDims();
  auto expanded = functional::map(
      [numDims, builder, loc, operands](AffineExpr expr) {
        return expandAffineExpr(builder, loc, expr,
                                operands.take_front(numDims),
                                operands.drop_front(numDims));
      },
      affineMap.getResults());
  if (llvm::all_of(expanded, [](Value *v) { return v; }))
    return expanded;
  return None;
}

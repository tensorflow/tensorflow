//===- Utils.cpp - Utilities to support the Linalg dialect ----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Intrinsics.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/FoldUtils.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::linalg::intrinsics;
using namespace mlir::loop;

mlir::edsc::LoopRangeBuilder::LoopRangeBuilder(ValueHandle *iv,
                                               ValueHandle range) {
  assert(range.getType() && "expected !linalg.range type");
  assert(range.getValue()->getDefiningOp() &&
         "need operations to extract range parts");
  auto rangeOp = cast<RangeOp>(range.getValue()->getDefiningOp());
  auto lb = rangeOp.min();
  auto ub = rangeOp.max();
  auto step = rangeOp.step();
  auto forOp = OperationHandle::createOp<ForOp>(lb, ub, step);
  *iv = ValueHandle(forOp.getInductionVar());
  auto *body = forOp.getBody();
  enter(body, /*prev=*/1);
}

mlir::edsc::LoopRangeBuilder::LoopRangeBuilder(ValueHandle *iv,
                                               SubViewOp::Range range) {
  auto forOp =
      OperationHandle::createOp<ForOp>(range.offset, range.size, range.stride);
  *iv = ValueHandle(forOp.getInductionVar());
  auto *body = forOp.getBody();
  enter(body, /*prev=*/1);
}

ValueHandle
mlir::edsc::LoopRangeBuilder::operator()(std::function<void(void)> fun) {
  if (fun)
    fun();
  exit();
  return ValueHandle::null();
}

mlir::edsc::LoopNestRangeBuilder::LoopNestRangeBuilder(
    ArrayRef<ValueHandle *> ivs, ArrayRef<SubViewOp::Range> ranges) {
  loops.reserve(ranges.size());
  for (unsigned i = 0, e = ranges.size(); i < e; ++i) {
    loops.emplace_back(ivs[i], ranges[i]);
  }
  assert(loops.size() == ivs.size() && "Mismatch loops vs ivs size");
}

mlir::edsc::LoopNestRangeBuilder::LoopNestRangeBuilder(
    ArrayRef<ValueHandle *> ivs, ArrayRef<ValueHandle> ranges) {
  loops.reserve(ranges.size());
  for (unsigned i = 0, e = ranges.size(); i < e; ++i) {
    loops.emplace_back(ivs[i], ranges[i]);
  }
  assert(loops.size() == ivs.size() && "Mismatch loops vs ivs size");
}

mlir::edsc::LoopNestRangeBuilder::LoopNestRangeBuilder(
    ArrayRef<ValueHandle *> ivs, ArrayRef<Value> ranges)
    : LoopNestRangeBuilder(
          ivs, SmallVector<ValueHandle, 4>(ranges.begin(), ranges.end())) {}

ValueHandle LoopNestRangeBuilder::LoopNestRangeBuilder::operator()(
    std::function<void(void)> fun) {
  if (fun)
    fun();
  for (auto &lit : reverse(loops)) {
    lit({});
  }
  return ValueHandle::null();
}

static Value emitOrFoldComposedAffineApply(OpBuilder &b, Location loc,
                                           AffineMap map,
                                           ArrayRef<Value> operandsRef,
                                           OperationFolder *folder) {
  SmallVector<Value, 4> operands(operandsRef.begin(), operandsRef.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  canonicalizeMapAndOperands(&map, &operands);
  return folder ? folder->create<AffineApplyOp>(b, loc, map, operands)
                : b.create<AffineApplyOp>(loc, map, operands);
}

SmallVector<Value, 4> mlir::linalg::applyMapToValues(OpBuilder &b, Location loc,
                                                     AffineMap map,
                                                     ArrayRef<Value> values,
                                                     OperationFolder *folder) {
  SmallVector<Value, 4> res;
  res.reserve(map.getNumResults());
  unsigned numDims = map.getNumDims();
  // For each `expr` in `map`, applies the `expr` to the values extracted from
  // ranges. If the resulting application can be folded into a Value, the
  // folding occurs eagerly. Otherwise, an affine.apply operation is emitted.
  for (auto expr : map.getResults()) {
    AffineMap map = AffineMap::get(numDims, 0, expr);
    res.push_back(emitOrFoldComposedAffineApply(b, loc, map, values, folder));
  }
  return res;
}

/// Returns all the operands of `linalgOp` that are not views.
/// Asserts that these operands are value types to allow transformations like
/// tiling to just use the values when cloning `linalgOp`.
SmallVector<Value, 4>
mlir::linalg::getAssumedNonViewOperands(LinalgOp linalgOp) {
  auto *op = linalgOp.getOperation();
  unsigned numViews = linalgOp.getNumInputsAndOutputs();
  unsigned nOperands = op->getNumOperands() - numViews;
  SmallVector<Value, 4> res;
  res.reserve(nOperands);
  for (unsigned i = 0; i < nOperands; ++i) {
    res.push_back(op->getOperand(numViews + i));
    auto t = res.back()->getType();
    (void)t;
    assert((t.isIntOrIndexOrFloat() || t.isa<VectorType>()) &&
           "expected scalar or vector type");
  }
  return res;
}

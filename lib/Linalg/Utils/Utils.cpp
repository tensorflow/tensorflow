//===- Utils.cpp - Utilities to support the Linalg dialect ----------------===//
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
// This file implements utilities for the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Linalg/Utils/Utils.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Linalg/IR/LinalgOps.h"
#include "mlir/Linalg/IR/LinalgTypes.h"
#include "mlir/Linalg/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/STLExtras.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace llvm;

mlir::edsc::LoopNestRangeBuilder::LoopNestRangeBuilder(
    ArrayRef<ValueHandle *> ivs, ArrayRef<ValueHandle> ranges) {
  for (unsigned i = 0, e = ranges.size(); i < e; ++i) {
    assert(ranges[i].getType() && "expected !linalg.range type");
    assert(ranges[i].getValue()->getDefiningOp() &&
           "need operations to extract range parts");
    auto rangeOp = ranges[i].getValue()->getDefiningOp()->cast<RangeOp>();
    auto lb = rangeOp.min();
    auto ub = rangeOp.max();
    // This must be a constexpr index until we relax the affine.for constraint
    auto step =
        rangeOp.step()->getDefiningOp()->cast<ConstantIndexOp>().getValue();
    loops.emplace_back(ivs[i], ValueHandle(lb), ValueHandle(ub), step);
  }
  assert(loops.size() == ivs.size() && "Mismatch loops vs ivs size");
}

mlir::edsc::LoopNestRangeBuilder::LoopNestRangeBuilder(
    ArrayRef<ValueHandle *> ivs, ArrayRef<Value *> ranges)
    : LoopNestRangeBuilder(
          ivs, SmallVector<ValueHandle, 4>(ranges.begin(), ranges.end())) {}

ValueHandle LoopNestRangeBuilder::LoopNestRangeBuilder::operator()(
    ArrayRef<CapturableHandle> stmts) {
  for (auto &lit : reverse(loops)) {
    lit({});
  }
  return ValueHandle::null();
}

SmallVector<Value *, 8> mlir::getRanges(Operation *op) {
  SmallVector<Value *, 8> res;
  if (auto view = op->dyn_cast<ViewOp>()) {
    res.append(view.getIndexings().begin(), view.getIndexings().end());
  } else if (auto slice = op->dyn_cast<SliceOp>()) {
    for (auto *i : slice.getIndexings())
      if (i->getType().isa<RangeType>())
        res.push_back(i);
  } else {
    for (auto *v : op->getOperands()) {
      if (v->getType().isa<ViewType>()) {
        if (auto *vOp = v->getDefiningOp()) {
          auto tmp = getRanges(vOp);
          res.append(tmp.begin(), tmp.end());
        } else {
          llvm_unreachable("Needs an operation to extract ranges from a view");
        }
      }
    }
  }
  return res;
}

// Implementation details:
//   1. Checks whether `ranges` define a new View by performing an equality
//      check between the range ssa-values and the operands of
//      `viewDefiningOp`.
//   2. If all ranges happen to be equal, op creation is elided and the
//      original result is returned instead.
//   3. Otherwise, creates a SliceOp with the new `ranges`.
// This is used to abstract away the creation of a SliceOp.
Value *mlir::createOrReturnView(FuncBuilder *b, Location loc,
                                Operation *viewDefiningOp,
                                ArrayRef<Value *> ranges) {
  if (auto view = viewDefiningOp->dyn_cast<ViewOp>()) {
    auto indexings = view.getIndexings();
    if (std::equal(indexings.begin(), indexings.end(), ranges.begin()))
      return view.getResult();
    return b->create<SliceOp>(loc, view.getResult(), ranges);
  }
  auto slice = viewDefiningOp->cast<SliceOp>();
  unsigned idxRange = 0;
  SmallVector<Value *, 4> newIndexings;
  bool elide = true;
  for (auto indexing : slice.getIndexings()) {
    if (indexing->getType().isa<RangeType>()) {
      elide &= (indexing != ranges[idxRange]);
      newIndexings.push_back(ranges[idxRange++]);
    } else
      newIndexings.push_back(indexing);
  }
  if (elide)
    return slice.getResult();
  return b->create<SliceOp>(loc, slice.getBaseView(), newIndexings);
}

Value *mlir::extractRangePart(Value *range, RangePart part) {
  assert(range->getType().isa<RangeType>() && "expected range type");
  if (range->getDefiningOp()) {
    if (auto r = dyn_cast_or_null<RangeOp>(range->getDefiningOp())) {
      switch (part) {
      case RangePart::Min:
        return r.min();
      case RangePart::Max:
        return r.max();
      case RangePart::Step:
        return r.step();
      }
    }
  }
  llvm_unreachable("need operations to extract range parts");
}

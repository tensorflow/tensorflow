//===- LowerToLoops.cpp - conversion from Linalg library ops to loops------===//
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

#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Linalg/IR/LinalgOps.h"
#include "mlir/Linalg/IR/LinalgTypes.h"
#include "mlir/Linalg/Passes.h"
#include "mlir/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/FoldUtils.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

// Creates a number of ranges equal to the number of results in `map`.
// The returned ranges correspond to the loop ranges, in the proper order, for
// which new loops will be created.
static SmallVector<Value *, 4> emitLoopRanges(OpBuilder &b, Location loc,
                                              AffineMap map,
                                              ArrayRef<Value *> allViewSizes,
                                              OperationFolder &state) {
  // Apply `map` to get view sizes in loop order.
  auto sizes = applyMapToValues(b, loc, map, allViewSizes, state);
  // Create a new range with the applied tile sizes.
  SmallVector<Value *, 4> res;
  for (unsigned idx = 0, e = map.getNumResults(); idx < e; ++idx) {
    res.push_back(b.create<RangeOp>(
        loc, state.create<ConstantIndexOp>(b, loc, 0), sizes[idx],
        state.create<ConstantIndexOp>(b, loc, 1)));
  }
  return res;
}

static void emitLinalgOpAsLoops(LinalgOp &linalgOp, OperationFolder &state) {
  OpBuilder b(linalgOp.getOperation());
  ScopedContext scope(b, linalgOp.getOperation()->getLoc());
  // The flattened loopToOperandRangesMaps is expected to be an invertible
  // permutation map (which is asserted in the inverse calculation).
  auto invertedMap =
      inversePermutation(concatAffineMaps(loopToOperandRangesMaps(linalgOp)));
  if (!invertedMap) {
    mlir::linalg::emitScalarImplementation({}, {}, {}, linalgOp, state);
    return;
  }

  auto loopRanges = emitLoopRanges(scope.getBuilder(), scope.getLocation(),
                                   invertedMap, getViewSizes(linalgOp), state);

  SmallVector<IndexHandle, 4> parallelIvs(linalgOp.getNumParallelLoops());
  SmallVector<IndexHandle, 4> reductionIvs(linalgOp.getNumReductionLoops());
  SmallVector<IndexHandle, 4> windowIvs(linalgOp.getNumWindowLoops());
  auto pivs = IndexHandle::makeIndexHandlePointers(parallelIvs);
  auto rivs = IndexHandle::makeIndexHandlePointers(reductionIvs);
  auto wivs = IndexHandle::makeIndexHandlePointers(windowIvs);
  assert(loopRanges.size() == pivs.size() + rivs.size() + wivs.size());

  // clang-format off
  ArrayRef<Value *> ranges(loopRanges);
  LoopNestRangeBuilder(pivs, ranges.take_front(pivs.size()))([&] {
    LoopNestRangeBuilder(
        rivs, ranges.drop_back(wivs.size()).take_back(rivs.size()))([&] {
      LoopNestRangeBuilder(wivs, ranges.take_back(wivs.size()))(
        [&linalgOp, &parallelIvs, &reductionIvs, &windowIvs, &state] {
        SmallVector<mlir::Value *, 4> parallel(
            parallelIvs.begin(), parallelIvs.end());
        SmallVector<mlir::Value *, 4> reduction(
            reductionIvs.begin(), reductionIvs.end());
        SmallVector<mlir::Value *, 4> window(
            windowIvs.begin(), windowIvs.end());
        mlir::linalg::emitScalarImplementation(
            parallel, reduction, window, linalgOp, state);
      });
    });
  });
  // clang-format on
}

namespace {
struct LowerLinalgToLoopsPass : public FunctionPass<LowerLinalgToLoopsPass> {
  void runOnFunction();
};
} // namespace

void LowerLinalgToLoopsPass::runOnFunction() {
  auto &f = getFunction();
  OperationFolder state;
  f.walk<LinalgOp>([&state](LinalgOp linalgOp) {
    emitLinalgOpAsLoops(linalgOp, state);
    linalgOp.getOperation()->erase();
  });
}

FunctionPassBase *mlir::linalg::createLowerLinalgToLoopsPass() {
  return new LowerLinalgToLoopsPass();
}

static PassRegistration<LowerLinalgToLoopsPass>
    pass("linalg-lower-to-loops",
         "Lower the operations from the linalg dialect into loops");

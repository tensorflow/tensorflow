//===- Transforms.cpp - Implementation of the linalg Transformations ------===//
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
// This file implements analyses and transformations for the linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "linalg4/Transforms.h"
#include "linalg3/Intrinsics.h"
#include "linalg3/TensorOps.h"

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/LoopUtils.h"

using llvm::ArrayRef;
using llvm::SmallVector;
using namespace mlir;
using namespace mlir::edsc;
using namespace linalg;
using namespace linalg::intrinsics;

llvm::Optional<SmallVector<mlir::AffineForOp, 8>>
linalg::writeAsTiledLoops(Operation *op, ArrayRef<uint64_t> tileSizes) {
  auto loops = writeAsLoops(op);
  if (loops.hasValue())
    return mlir::tile(*loops, tileSizes, loops->back());
  return llvm::None;
}

void linalg::lowerToTiledLoops(mlir::Function *f,
                               ArrayRef<uint64_t> tileSizes) {
  f->walk([tileSizes](Operation *op) {
    if (writeAsTiledLoops(op, tileSizes).hasValue())
      op->erase();
  });
}

template <class ConcreteOp>
static Operation::operand_range
getInputsAndOutputs(TensorContractionBase<ConcreteOp> &contraction) {
  auto *inst = static_cast<ConcreteOp *>(&contraction)->getOperation();
  auto begin = inst->operand_begin();
  auto end = inst->operand_begin() + contraction.getNumInputs() +
             contraction.getNumOutputs();
  return {begin, end};
}

static bool isZeroIndex(Value *v) {
  return v->getDefiningOp() && v->getDefiningOp()->isa<ConstantIndexOp>() &&
         v->getDefiningOp()->dyn_cast<ConstantIndexOp>().getValue() == 0;
}

template <typename ConcreteOp>
static llvm::SmallVector<Value *, 4>
makeTiledRanges(TensorContractionBase<ConcreteOp> &contraction,
                ArrayRef<Value *> allRanges, llvm::ArrayRef<Value *> ivs,
                llvm::ArrayRef<Value *> tileSizes) {
  assert(ivs.size() == tileSizes.size());
  if (ivs.empty())
    return RangeParts(allRanges).makeRanges();

  auto *op = static_cast<ConcreteOp *>(&contraction);
  RangeParts result(allRanges.size());
  RangeParts rangeParts(allRanges);

  for (auto map : op->loopsToOperandRangeMaps()) {
    // 1. Take the first ivs results of the map, the other ones are not composed
    // but merely copied over.
    assert(map.getNumSymbols() == 0);
    assert(map.getRangeSizes().empty());
    MLIRContext *context = ScopedContext::getContext();
    unsigned numParallel = op->getNumParallelDims();
    unsigned numReduction = op->getNumReductionDims();
    if (ivs.size() < numParallel + numReduction) {
      // Inject zeros in positions that are not tiled.
      SmallVector<AffineExpr, 4> dimReplacements(numParallel + numReduction);
      for (unsigned i = 0, e = numParallel + numReduction; i < e; ++i) {
        dimReplacements[i] = (i < ivs.size())
                                 ? getAffineDimExpr(i, context)
                                 : getAffineConstantExpr(0, context);
      }
      map = map.replaceDimsAndSymbols(dimReplacements, {}, ivs.size(), 0);
    }

    // 2. Apply the rewritten map to the ranges.
    unsigned numDims = map.getNumDims();
    for (auto en : llvm::enumerate(map.getResults())) {
      auto index = en.index();
      auto expr = en.value();
      AffineMap exprMap = AffineMap::get(numDims, 0, expr, {});
      ValueHandle offset(makeFoldedComposedAffineApply(exprMap, ivs));
      // Offset is normally a function of loop induction variables.
      // If it is 0, it must come from a dimension that was not tiled.
      if (isZeroIndex(offset)) {
        result.mins.push_back(rangeParts.mins[index]);
        result.maxes.push_back(rangeParts.maxes[index]);
        continue;
      }

      ValueHandle step(makeFoldedComposedAffineApply(exprMap, tileSizes));
      ValueHandle min(rangeParts.mins[index]);
      using edsc::op::operator+;
      result.mins.push_back(min + offset);
      // Ideally this should be:
      //   `min(rangeParts.max, rangeParts.min + offset + step)`
      // but that breaks the current limitations of the affine dialect.
      result.maxes.push_back(min + offset + step);
    }
  }
  // Note that for the purpose of tiled ranges and views, the steps do not
  // change in our representation.
  result.steps = rangeParts.steps;

  return result.makeRanges();
}

template <class ConcreteOp>
static SmallVector<Value *, 4>
makeTiledViews(linalg::TensorContractionBase<ConcreteOp> &contraction,
               ArrayRef<Value *> ivs, ArrayRef<Value *> tileSizes) {
  auto tiledRanges =
      makeTiledRanges(contraction, getRanges(contraction), ivs, tileSizes);
  SmallVector<Value *, 4> res;
  unsigned currentRange = 0;
  for (auto *in : getInputsAndOutputs(contraction)) {
    unsigned runningSliceDim = 0;
    auto *runningSlice = in;
    assert(runningSlice->getType().template isa<ViewType>());
    for (unsigned d = 0, e = getViewRank(runningSlice); d < e; ++d) {
      auto *r = tiledRanges[currentRange++];
      runningSlice = slice(runningSlice, r, runningSliceDim++).getValue();
    }
    res.push_back(runningSlice);
  }
  return res;
}

template <class ConcreteOp>
static SmallVector<mlir::AffineForOp, 8>
writeContractionAsTiledViews(TensorContractionBase<ConcreteOp> &contraction,
                             ArrayRef<Value *> tileSizes) {
  assert(tileSizes.size() <=
         contraction.getNumParallelDims() + contraction.getNumReductionDims());

  auto *op = static_cast<ConcreteOp *>(&contraction);
  ScopedContext scope(mlir::FuncBuilder(op->getOperation()), op->getLoc());
  SmallVector<IndexHandle, 4> ivs(tileSizes.size());
  auto pivs = IndexHandle::makeIndexHandlePointers(ivs);

  // clang-format off
  using linalg::common::LoopNestRangeBuilder;
  auto ranges = makeGenericLoopRanges(operandRangesToLoopsMap(contraction),
                                      getRanges(contraction), tileSizes);
  linalg::common::LoopNestRangeBuilder(pivs, ranges)({
    [&contraction, &tileSizes, &ivs]() {
      SmallVector<Value *, 4> ivValues(ivs.begin(), ivs.end());
      auto views = makeTiledViews(contraction, ivValues, tileSizes);
      ScopedContext::getBuilder()->create<ConcreteOp>(
          ScopedContext::getLocation(), views);
      /// NestedBuilders expect handles, we thus return an IndexHandle.
      return IndexHandle();
    }()
  });
  // clang-format on

  SmallVector<mlir::AffineForOp, 8> res;
  res.reserve(ivs.size());
  for (auto iv : ivs)
    res.push_back(getForInductionVarOwner(iv.getValue()));
  return res;
}

llvm::Optional<SmallVector<mlir::AffineForOp, 8>>
linalg::writeAsTiledViews(Operation *op, ArrayRef<Value *> tileSizes) {
  if (auto matmulOp = op->dyn_cast<linalg::MatmulOp>()) {
    return writeContractionAsTiledViews(matmulOp, tileSizes);
  } else if (auto matvecOp = op->dyn_cast<linalg::MatvecOp>()) {
    return writeContractionAsTiledViews(matvecOp, tileSizes);
  } else if (auto dotOp = op->dyn_cast<linalg::DotOp>()) {
    return writeContractionAsTiledViews(dotOp, tileSizes);
  }
  return llvm::None;
}

void linalg::lowerToTiledViews(mlir::Function *f, ArrayRef<Value *> tileSizes) {
  f->walk([tileSizes](Operation *op) {
    if (auto matmulOp = op->dyn_cast<linalg::MatmulOp>()) {
      writeAsTiledViews(matmulOp, tileSizes);
    } else if (auto matvecOp = op->dyn_cast<linalg::MatvecOp>()) {
      writeAsTiledViews(matvecOp, tileSizes);
    } else if (auto dotOp = op->dyn_cast<linalg::DotOp>()) {
      writeAsTiledViews(dotOp, tileSizes);
    } else {
      return;
    }
    op->erase();
  });
}

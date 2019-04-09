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

#include "linalg3/Transforms.h"
#include "linalg2/Intrinsics.h"
#include "linalg3/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace linalg;
using namespace linalg::intrinsics;

void linalg::composeSliceOps(mlir::Function *f) {
  f->walk<SliceOp>([](SliceOp sliceOp) {
    auto *sliceResult = sliceOp.getResult();
    auto viewOp = emitAndReturnFullyComposedView(sliceResult);
    sliceResult->replaceAllUsesWith(viewOp.getResult());
    sliceOp.erase();
  });
}

void linalg::lowerToFinerGrainedTensorContraction(mlir::Function *f) {
  f->walk([](Operation *op) {
    if (auto matmulOp = op->dyn_cast<linalg::MatmulOp>()) {
      matmulOp.writeAsFinerGrainTensorContraction();
    } else if (auto matvecOp = op->dyn_cast<linalg::MatvecOp>()) {
      matvecOp.writeAsFinerGrainTensorContraction();
    } else {
      return;
    }
    op->erase();
  });
}

// Folding eagerly is necessary to abide by affine.for static step requirement.
// Returns nullptr if folding is not trivially feasible.
static Value *tryFold(AffineMap map, SmallVector<Value *, 4> operands) {
  assert(map.getNumResults() == 1 && "single result map expected");
  auto expr = map.getResult(0);
  if (auto dim = expr.dyn_cast<AffineDimExpr>())
    return operands[dim.getPosition()];
  if (auto sym = expr.dyn_cast<AffineSymbolExpr>())
    return operands[map.getNumDims() + sym.getPosition()];
  if (auto cst = expr.dyn_cast<AffineConstantExpr>())
    return constant_index(cst.getValue());
  return nullptr;
}

Value *linalg::makeFoldedComposedAffineApply(AffineMap map,
                                             ArrayRef<Value *> operandsRef) {
  SmallVector<Value *, 4> operands(operandsRef.begin(), operandsRef.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  if (auto *v = tryFold(map, operands)) {
    return v;
  }
  auto *b = ScopedContext::getBuilder();
  auto loc = ScopedContext::getLocation();
  return b->create<AffineApplyOp>(loc, map, operands).getResult();
}

linalg::RangeParts::RangeParts(unsigned reserved) {
  mins.reserve(reserved);
  maxes.reserve(reserved);
  steps.reserve(reserved);
}

static SmallVector<Value *, 4>
extractFromRanges(ArrayRef<Value *> ranges,
                  std::function<Value *(RangeOp)> extract) {
  SmallVector<Value *, 4> res;
  res.reserve(ranges.size());
  for (auto *v : ranges) {
    auto r = v->getDefiningOp()->cast<RangeOp>();
    res.push_back(extract(r));
  }
  return res;
}

linalg::RangeParts::RangeParts(ArrayRef<Value *> ranges)
    : mins(extractFromRanges(ranges, [](RangeOp r) { return r.getMin(); })),
      maxes(extractFromRanges(ranges, [](RangeOp r) { return r.getMax(); })),
      steps(extractFromRanges(ranges, [](RangeOp r) { return r.getStep(); })) {}

SmallVector<Value *, 4> linalg::RangeParts::makeRanges() {
  SmallVector<Value *, 4> res;
  res.reserve(mins.size());
  for (auto z : llvm::zip(mins, maxes, steps)) {
    res.push_back(range(std::get<0>(z), std::get<1>(z), std::get<2>(z)));
  }
  return res;
}

static RangeParts makeGenericRangeParts(AffineMap map,
                                        ArrayRef<Value *> ranges) {
  assert(map.getNumInputs() == ranges.size());
  unsigned numDims = map.getNumDims();
  assert(map.getNumSymbols() == 0);
  assert(map.getRangeSizes().empty());

  RangeParts res(map.getNumResults());
  RangeParts rangeParts(ranges);
  for (auto expr : map.getResults()) {
    AffineMap map = AffineMap::get(numDims, 0, expr, {});
    res.mins.push_back(makeFoldedComposedAffineApply(map, rangeParts.mins));
    res.maxes.push_back(makeFoldedComposedAffineApply(map, rangeParts.maxes));
    res.steps.push_back(makeFoldedComposedAffineApply(map, rangeParts.steps));
  }
  return res;
}

SmallVector<Value *, 4> makeGenericRanges(AffineMap map,
                                          ArrayRef<Value *> ranges) {
  return makeGenericRangeParts(map, ranges).makeRanges();
}

SmallVector<Value *, 4>
linalg::makeGenericLoopRanges(AffineMap operandRangesToLoopMaps,
                              ArrayRef<Value *> ranges,
                              ArrayRef<Value *> tileSizes) {
  RangeParts res = makeGenericRangeParts(operandRangesToLoopMaps, ranges);
  if (tileSizes.empty())
    return res.makeRanges();
  SmallVector<Value *, 4> tiledSteps;
  for (auto z : llvm::zip(res.steps, tileSizes)) {
    auto *step = std::get<0>(z);
    auto tileSize = std::get<1>(z);
    auto stepValue = step->getDefiningOp()->cast<ConstantIndexOp>().getValue();
    auto tileSizeValue =
        tileSize->getDefiningOp()->cast<ConstantIndexOp>().getValue();
    assert(stepValue > 0);
    tiledSteps.push_back(constant_index(stepValue * tileSizeValue));
  }
  res.steps = tiledSteps;
  return res.makeRanges();
}

template <class ContractionOp>
static SmallVector<mlir::AffineForOp, 4>
writeContractionAsLoops(ContractionOp contraction) {
  ScopedContext scope(FuncBuilder(contraction.getOperation()),
                      contraction.getLoc());
  auto allRanges = getRanges(contraction);
  auto loopRanges =
      makeGenericLoopRanges(operandRangesToLoopsMap(contraction), allRanges);

  SmallVector<IndexHandle, 4> parallelIvs(contraction.getNumParallelDims());
  SmallVector<IndexHandle, 4> reductionIvs(contraction.getNumReductionDims());
  auto pivs = IndexHandle::makeIndexHandlePointers(parallelIvs);
  auto rivs = IndexHandle::makeIndexHandlePointers(reductionIvs);
  assert(loopRanges.size() == pivs.size() + rivs.size());

  // clang-format off
  using linalg::common::LoopNestRangeBuilder;
  ArrayRef<Value *> ranges(loopRanges);
  LoopNestRangeBuilder(pivs, ranges.take_front(pivs.size()))({
    LoopNestRangeBuilder(rivs, ranges.take_back(rivs.size()))({
      [&contraction, &parallelIvs, &reductionIvs]() {
        SmallVector<mlir::Value *, 4> parallel(
            parallelIvs.begin(), parallelIvs.end());
        SmallVector<mlir::Value *, 4> reduction(
            reductionIvs.begin(), reductionIvs.end());
        contraction.emitScalarImplementation(parallel, reduction);
        /// NestedBuilders expect handles, we thus return an IndexHandle.
        return IndexHandle();
      }()
    })
  });
  // clang-format on

  // Return the AffineForOp for better compositionality (e.g. tiling).
  SmallVector<mlir::AffineForOp, 4> loops;
  loops.reserve(pivs.size() + rivs.size());
  for (auto iv : parallelIvs)
    loops.push_back(getForInductionVarOwner(iv.getValue()));
  for (auto iv : reductionIvs)
    loops.push_back(getForInductionVarOwner(iv.getValue()));

  return loops;
}

llvm::Optional<SmallVector<mlir::AffineForOp, 4>>
linalg::writeAsLoops(Operation *op) {
  if (auto matmulOp = op->dyn_cast<linalg::MatmulOp>()) {
    return writeContractionAsLoops(matmulOp);
  } else if (auto matvecOp = op->dyn_cast<linalg::MatvecOp>()) {
    return writeContractionAsLoops(matvecOp);
  } else if (auto dotOp = op->dyn_cast<linalg::DotOp>()) {
    return writeContractionAsLoops(dotOp);
  }
  return llvm::None;
}

void linalg::lowerToLoops(mlir::Function *f) {
  f->walk([](Operation *op) {
    if (writeAsLoops(op))
      op->erase();
  });
}

namespace {

/// Rewriting linalg::LoadOp and linalg::StoreOp to mlir::LoadOp and
/// mlir::StoreOp requires finding the proper indexing in the supporting MemRef.
/// This is most easily achieved by calling emitAndReturnFullyComposedView to
/// fold away all the SliceOp.
template <typename LoadOrStoreOpTy> struct Rewriter : public RewritePattern {
  explicit Rewriter(MLIRContext *context)
      : RewritePattern(LoadOrStoreOpTy::getOperationName(), 1, context) {}

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override;
};

struct LowerLinalgLoadStorePass
    : public FunctionPass<LowerLinalgLoadStorePass> {
  void runOnFunction() {
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.push_back(llvm::make_unique<Rewriter<linalg::LoadOp>>(context));
    patterns.push_back(llvm::make_unique<Rewriter<linalg::StoreOp>>(context));
    applyPatternsGreedily(getFunction(), std::move(patterns));
  }
};
} // namespace

/// Emits and returns the standard load and store ops from the view indexings.
/// If the indexing is of index type, use it as an index to the load/store.
/// If the indexing is a range, use range.min + indexing as an index to the
/// load/store.
template <typename LoadOrStoreOp>
static SmallVector<Value *, 8>
emitAndReturnLoadStoreOperands(LoadOrStoreOp loadOrStoreOp, ViewOp viewOp) {
  unsigned storeDim = 0;
  SmallVector<Value *, 8> operands;
  for (auto *indexing : viewOp.getIndexings()) {
    if (indexing->getType().isa<IndexType>()) {
      operands.push_back(indexing);
      continue;
    }
    RangeOp range = indexing->getDefiningOp()->cast<RangeOp>();
    ValueHandle min(range.getMin());
    Value *storeIndex = *(loadOrStoreOp.getIndices().begin() + storeDim++);
    using edsc::op::operator+;
    operands.push_back(min + ValueHandle(storeIndex));
  }
  return operands;
}

template <>
PatternMatchResult
Rewriter<linalg::LoadOp>::matchAndRewrite(Operation *op,
                                          PatternRewriter &rewriter) const {
  auto load = op->cast<linalg::LoadOp>();
  SliceOp slice = load.getView()->getDefiningOp()->dyn_cast<SliceOp>();
  ViewOp view = slice ? emitAndReturnFullyComposedView(slice.getResult())
                      : load.getView()->getDefiningOp()->cast<ViewOp>();
  ScopedContext scope(FuncBuilder(load), load.getLoc());
  auto *memRef = view.getSupportingMemRef();
  auto operands = emitAndReturnLoadStoreOperands(load, view);
  rewriter.replaceOpWithNewOp<mlir::LoadOp>(op, memRef, operands);
  return matchSuccess();
}

template <>
PatternMatchResult
Rewriter<linalg::StoreOp>::matchAndRewrite(Operation *op,
                                           PatternRewriter &rewriter) const {
  auto store = op->cast<linalg::StoreOp>();
  SliceOp slice = store.getView()->getDefiningOp()->dyn_cast<SliceOp>();
  ViewOp view = slice ? emitAndReturnFullyComposedView(slice.getResult())
                      : store.getView()->getDefiningOp()->cast<ViewOp>();
  ScopedContext scope(FuncBuilder(store), store.getLoc());
  auto *valueToStore = store.getValueToStore();
  auto *memRef = view.getSupportingMemRef();
  auto operands = emitAndReturnLoadStoreOperands(store, view);
  rewriter.replaceOpWithNewOp<mlir::StoreOp>(op, valueToStore, memRef,
                                             operands);
  return matchSuccess();
}

FunctionPassBase *linalg::createLowerLinalgLoadStorePass() {
  return new LowerLinalgLoadStorePass();
}

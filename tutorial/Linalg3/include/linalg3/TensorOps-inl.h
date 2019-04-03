//===- TensorOps-inl.h - Linalg dialect TensorOps operation implementation ===//
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

/// The TensorOp-inl.h inclusion pattern is chosen to allow gradual extension of
/// TensorOps by adding implementations as they are needed in the appropriate
/// step in the tutorial.
#ifndef LINALG3_TENSOROPS_INL_H_
#define LINALG3_TENSOROPS_INL_H_

#include "linalg1/Common.h"
#include "linalg1/Utils.h"
#include "linalg2/TensorOps.h"
#include "linalg3/Analysis.h"
#include "linalg3/Ops.h"

template <class ConcreteOp>
mlir::Value *
linalg::TensorContractionBase<ConcreteOp>::getInputView(unsigned i) {
  return *(getInputs().begin() + i);
}

template <class ConcreteOp>
mlir::Value *
linalg::TensorContractionBase<ConcreteOp>::getOutputView(unsigned i) {
  return *(getOutputs().begin() + i);
}

template <class ConcreteOp>
mlir::AffineMap
linalg::TensorContractionBase<ConcreteOp>::loopsToOperandRangesMap() {
  return static_cast<ConcreteOp *>(this)->loopsToOperandRangesMap();
}

template <class ConcreteOp>
void linalg::TensorContractionBase<ConcreteOp>::emitScalarImplementation(
    llvm::ArrayRef<mlir::Value *> parallelIvs,
    llvm::ArrayRef<mlir::Value *> reductionIvs) {
  static_cast<ConcreteOp *>(this)->emitScalarImplementation(parallelIvs,
                                                            reductionIvs);
}

template <class ConcreteOp>
mlir::AffineMap linalg::operandRangesToLoopsMap(
    linalg::TensorContractionBase<ConcreteOp> &tensorContraction) {
  return inverseSubMap(tensorContraction.loopsToOperandRangesMap());
}

// Extract the ranges from a given ViewOp or SliceOp.
//
// In the case of a ViewOp, things are simple: just traverse the indexings and
// get all the ranges (i.e. drop the indices).
//
// In the case of a SliceOp, things are trickier because we need to handle a
// potential rank-reduction:
//   1. Examine the indexing to determine if it is rank-reducing.
//   2. If it is rank-reducing, an offset of 1 is added to the dimensions such
//      that `d >= slicingDim`. This is to account for the rank reduction.
// `getRootIndex` is then called on the **parent** view
static llvm::SmallVector<mlir::Value *, 8>
extractRangesFromViewOrSliceOp(mlir::Value *view) {
  // This expects a viewType which must come from either ViewOp or SliceOp.
  assert(view->getType().isa<linalg::ViewType>() && "expected ViewType");
  if (auto viewOp = view->getDefiningOp()->dyn_cast<linalg::ViewOp>())
    return viewOp.getRanges();

  auto sliceOp = view->getDefiningOp()->cast<linalg::SliceOp>();
  unsigned slicingDim = sliceOp.getSlicingDim();
  auto *indexing = *(sliceOp.getIndexings().begin());
  bool isRankReducing = indexing->getType().isa<mlir::IndexType>();
  unsigned offset = 0;
  llvm::SmallVector<mlir::Value *, 8> res;
  res.reserve(sliceOp.getRank());
  for (unsigned d = 0, e = sliceOp.getRank(); d < e; ++d) {
    if (d == slicingDim && isRankReducing)
      offset = 1;
    auto *parentView = sliceOp.getParentView();
    auto indexingPosPair = linalg::getViewRootIndexing(parentView, d + offset);
    res.push_back(indexingPosPair.first);
  }
  return res;
}

template <class ConcreteOp>
static llvm::SmallVector<mlir::Value *, 8>
getInputRanges(linalg::TensorContractionBase<ConcreteOp> &tensorContraction) {
  llvm::SmallVector<mlir::Value *, 8> res;
  for (auto *in : tensorContraction.getInputs()) {
    auto subres = extractRangesFromViewOrSliceOp(in);
    res.append(subres.begin(), subres.end());
  }
  return res;
}

template <class ConcreteOp>
static llvm::SmallVector<mlir::Value *, 8>
getOutputRanges(linalg::TensorContractionBase<ConcreteOp> &tensorContraction) {
  llvm::SmallVector<mlir::Value *, 8> res;
  for (auto *out : tensorContraction.getOutputs()) {
    auto subres = extractRangesFromViewOrSliceOp(out);
    res.append(subres.begin(), subres.end());
  }
  return res;
}

template <class ConcreteOp>
llvm::SmallVector<mlir::Value *, 8> linalg::getRanges(
    linalg::TensorContractionBase<ConcreteOp> &tensorContraction) {
  llvm::SmallVector<mlir::Value *, 8> res = getInputRanges(tensorContraction);
  llvm::SmallVector<mlir::Value *, 8> tmp = getOutputRanges(tensorContraction);
  res.append(tmp.begin(), tmp.end());
  return res;
}

#endif // LINALG3_TENSOROPS_INL_H_

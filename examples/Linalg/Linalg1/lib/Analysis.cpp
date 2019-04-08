//===- Analysis.cpp - Implementation of analysis functions for Linalg -----===//
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
// This file implements a simple IR operation to create a new RangeType in the
// linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "linalg1/Analysis.h"
#include "linalg1/Ops.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace linalg;

ViewOp linalg::getViewBaseViewOp(Value *view) {
  auto viewType = view->getType().dyn_cast<ViewType>();
  (void)viewType;
  assert(viewType.isa<ViewType>() && "expected a ViewType");
  while (auto slice = view->getDefiningOp()->dyn_cast<SliceOp>()) {
    view = slice.getParentView();
    assert(viewType.isa<ViewType>() && "expected a ViewType");
  }
  return view->getDefiningOp()->cast<ViewOp>();
}

Value *linalg::getViewSupportingMemRef(Value *view) {
  return getViewBaseViewOp(view).getSupportingMemRef();
}

std::pair<mlir::Value *, unsigned> linalg::getViewRootIndexing(Value *view,
                                                               unsigned dim) {
  auto viewType = view->getType().dyn_cast<ViewType>();
  (void)viewType;
  assert(viewType.isa<ViewType>() && "expected a ViewType");
  assert(dim < viewType.getRank() && "dim exceeds rank");
  if (auto viewOp = view->getDefiningOp()->dyn_cast<ViewOp>())
    return std::make_pair(viewOp.getIndexing(dim), dim);

  auto sliceOp = view->getDefiningOp()->cast<SliceOp>();
  auto *parentView = sliceOp.getParentView();
  unsigned sliceDim = sliceOp.getSlicingDim();
  auto *indexing = sliceOp.getIndexing();
  if (indexing->getDefiningOp()) {
    if (auto rangeOp = indexing->getDefiningOp()->cast<RangeOp>()) {
      // If I sliced with a range and I sliced at this dim, then I'm it.
      if (dim == sliceDim) {
        return std::make_pair(rangeOp.getResult(), dim);
      }
      // Otherwise, I did not change the rank, just go look for `dim` into my
      // parent.
      return getViewRootIndexing(parentView, dim);
    }
  }
  assert(indexing->getType().isa<IndexType>());
  // If I get here, I indexed and reduced along the dim `sliceDim` from my
  // parent. I need to query my parent for `dim` or `dim + 1` depending on
  // whether dim > sliceDim or not.
  unsigned parentDim = dim > sliceDim ? dim + 1 : dim;
  return getViewRootIndexing(parentView, parentDim);
}

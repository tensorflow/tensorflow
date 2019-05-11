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

#include "linalg2/Transforms.h"
#include "linalg2/Analysis.h"
#include "linalg2/Intrinsics.h"
#include "linalg2/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

using llvm::ArrayRef;
using llvm::SmallVector;
using mlir::FuncBuilder;
using mlir::MemRefType;
using mlir::Value;
using mlir::edsc::ScopedContext;
using mlir::edsc::ValueHandle;
using mlir::edsc::intrinsics::constant_index;

using namespace linalg;
using namespace linalg::intrinsics;

// We need to traverse the slice chain from the original ViewOp for various
// analyses. This builds the chain.
static SmallVector<Value *, 8> getViewChain(mlir::Value *v) {
  assert(v->getType().isa<ViewType>() && "ViewType expected");
  if (v->getDefiningOp()->isa<ViewOp>()) {
    return SmallVector<mlir::Value *, 8>{v};
  }

  SmallVector<mlir::Value *, 8> tmp;
  do {
    auto sliceOp = v->getDefiningOp()->cast<SliceOp>(); // must be a slice op
    tmp.push_back(v);
    v = sliceOp.getParentView();
  } while (!v->getType().isa<ViewType>());
  assert(v->getDefiningOp()->isa<ViewOp>() && "must be a ViewOp");
  tmp.push_back(v);
  return SmallVector<mlir::Value *, 8>(tmp.rbegin(), tmp.rend());
}

static mlir::Value *createFullyComposedIndexing(unsigned dim,
                                                ArrayRef<Value *> chain) {
  using namespace mlir::edsc::op;
  assert(chain.front()->getType().isa<ViewType>() && "must be a ViewType");
  auto viewOp = chain.front()->getDefiningOp()->cast<ViewOp>();
  auto *indexing = viewOp.getIndexing(dim);
  if (!indexing->getType().isa<RangeType>())
    return indexing;
  auto rangeOp = indexing->getDefiningOp()->cast<RangeOp>();
  Value *min = rangeOp.getMin(), *max = rangeOp.getMax(),
        *step = rangeOp.getStep();
  for (auto *v : chain.drop_front(1)) {
    auto slice = v->getDefiningOp()->cast<SliceOp>();
    if (slice.getRank() != slice.getParentRank()) {
      // Rank-reducing slice.
      if (slice.getSlicingDim() == dim) {
        // Slice a single element across dim -> done.
        return ValueHandle(min) +
               ValueHandle(slice.getIndexing()) * ValueHandle(step);
      }
      // Adjust the dim to account for the slice.
      dim = (slice.getSlicingDim() < dim) ? dim - 1 : dim;
    } else { // not a rank-reducing slice.
      if (slice.getSlicingDim() == dim) {
        auto range = slice.getIndexing()->getDefiningOp()->cast<RangeOp>();
        auto oldMin = min;
        min = ValueHandle(min) + ValueHandle(range.getMin());
        // ideally: max = min(oldMin + ValueHandle(range.getMax()), oldMax);
        // but we cannot represent min/max with index and have it compose with
        // affine.map atm.
        max = ValueHandle(oldMin) + ValueHandle(range.getMax());
        // ideally: parametric steps.
        // but we cannot represent parametric steps with index atm.
        step = ValueHandle(step) * ValueHandle(range.getStep());
      }
    }
  }
  return linalg::intrinsics::range(min, max, step).getValue();
}

ViewOp linalg::emitAndReturnFullyComposedView(Value *v) {
  ScopedContext scope(FuncBuilder(v->getDefiningOp()),
                      v->getDefiningOp()->getLoc());
  assert(v->getType().isa<ViewType>() && "must be a ViewType");
  auto *memRef = getViewSupportingMemRef(v);
  auto chain = getViewChain(v);
  unsigned rank = memRef->getType().cast<MemRefType>().getRank();
  SmallVector<Value *, 8> ranges;
  ranges.reserve(rank);
  for (unsigned idx = 0; idx < rank; ++idx) {
    ranges.push_back(createFullyComposedIndexing(idx, chain));
  }
  return view(memRef, ranges).getOperation()->cast<ViewOp>();
}

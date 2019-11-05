//===- Promotion.cpp - Implementation of linalg Promotion -----------------===//
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
// This file implements the linalg dialect Promotion pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Intrinsics.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/FoldUtils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::linalg::intrinsics;
using namespace mlir::loop;

using llvm::SetVector;

#define DEBUG_TYPE "linalg-promotion"

static AffineMap getAffineDifferenceMap(MLIRContext *context) {
  AffineExpr d0(getAffineDimExpr(0, context)), d1(getAffineDimExpr(1, context));
  return AffineMap::get(2, 0, {d0 - d1});
}

// TODO(ntv): replace this with 1-D memref alloc once there is an std.view op.
static Value *allocBuffer(Type elementType, Value *size) {
  if (auto cst = dyn_cast_or_null<ConstantIndexOp>(size->getDefiningOp()))
    return buffer_alloc(
        BufferType::get(size->getContext(), elementType, cst.getValue()));
  return buffer_alloc(BufferType::get(size->getContext(), elementType), size);
}

// Performs promotion of a `subView` into a local buffer of the size of the
// *ranges* of the `subView`. This produces a buffer whose size may be bigger
// than the actual size of the `subView` at the boundaries.
// This is related to the full/partial tile problem.
// Returns a PromotionInfo containing a `buffer`, `fullLocalView` and
// `partialLocalView` such that:
//   * `buffer` is always the size of the full tile.
//   * `fullLocalView` is a dense contiguous view into that buffer.
//   * `partialLocalView` is a dense non-contiguous slice of `fullLocalView`
//     that corresponds to the size of `subView` and accounting for boundary
//     effects.
// The point of the full tile buffer is that constant static tile sizes are
// folded and result in a buffer type with statically known size and alignment
// properties.
// To account for general boundary effects, padding must be performed on the
// boundary tiles. For now this is done with an unconditional `fill` op followed
// by a partial `copy` op.
static PromotionInfo promoteFullTileBuffer(OpBuilder &b, Location loc,
                                           SubViewOp subView,
                                           OperationFolder *folder) {
  auto zero = constant_index(folder, 0);
  auto one = constant_index(folder, 1);

  auto viewType = subView.getViewType();
  auto rank = viewType.getRank();
  Value *allocSize = one;
  SmallVector<Value *, 8> fullRanges, partialRanges;
  fullRanges.reserve(rank);
  partialRanges.reserve(rank);
  for (auto en : llvm::enumerate(subView.getRanges())) {
    auto rank = en.index();
    auto rangeValue = en.value();
    Value *d =
        isa<DimOp>(rangeValue.max->getDefiningOp())
            ? rangeValue.max
            : applyMapToValues(b, loc, getAffineDifferenceMap(b.getContext()),
                               {rangeValue.max, rangeValue.min}, folder)
                  .front();
    allocSize = muli(folder, allocSize, d).getValue();
    fullRanges.push_back(range(folder, zero, d, one));
    partialRanges.push_back(range(folder, zero, dim(subView, rank), one));
  }
  auto *buffer = allocBuffer(viewType.getElementType(), allocSize);
  auto fullLocalView = view(buffer, fullRanges);
  auto partialLocalView = slice(fullLocalView, partialRanges);
  return PromotionInfo{buffer, fullLocalView, partialLocalView};
}

SmallVector<PromotionInfo, 8>
mlir::linalg::promoteSubViews(OpBuilder &b, Location loc,
                              ArrayRef<Value *> subViews,
                              OperationFolder *folder) {
  if (subViews.empty())
    return {};

  ScopedContext scope(b, loc);
  SmallVector<PromotionInfo, 8> res;
  res.reserve(subViews.size());
  DenseMap<Value *, PromotionInfo> promotionInfoMap;
  for (auto *v : subViews) {
    SubViewOp subView = cast<SubViewOp>(v->getDefiningOp());
    auto viewType = subView.getViewType();
    // TODO(ntv): support more cases than just float.
    if (!viewType.getElementType().isa<FloatType>())
      continue;
    auto promotionInfo = promoteFullTileBuffer(b, loc, subView, folder);
    promotionInfoMap.insert(std::make_pair(subView.getResult(), promotionInfo));
    res.push_back(promotionInfo);
  }

  for (auto *v : subViews) {
    SubViewOp subView = cast<SubViewOp>(v->getDefiningOp());
    auto info = promotionInfoMap.find(v);
    if (info == promotionInfoMap.end())
      continue;
    // TODO(ntv): value to fill with should be related to the operation.
    // For now, just use APFloat(0.0f).
    auto t = subView.getViewType().getElementType().cast<FloatType>();
    Value *fillVal = constant_float(folder, APFloat(0.0f), t);
    // TODO(ntv): fill is only necessary if `promotionInfo` has a full local
    // view that is different from the partial local view and we are on the
    // boundary.
    fill(info->second.fullLocalView, fillVal);
  }

  for (auto *v : subViews) {
    auto info = promotionInfoMap.find(v);
    if (info == promotionInfoMap.end())
      continue;
    copy(cast<SubViewOp>(v->getDefiningOp()), info->second.partialLocalView);
  }
  return res;
}

static void promoteSubViewOperands(LinalgOp op, SetVector<Value *> subViews,
                                   OperationFolder *folder) {
  // 1. Promote the specified views and use them in the new op.
  OpBuilder b(op);
  ScopedContext scope(b, op.getLoc());
  auto promotedBufferAndViews =
      promoteSubViews(b, op.getLoc(), subViews.getArrayRef(), folder);
  SmallVector<Value *, 8> opViews;
  opViews.reserve(op.getNumInputsAndOutputs());
  SmallVector<std::pair<Value *, Value *>, 8> writebackViews;
  writebackViews.reserve(subViews.size());
  unsigned promotedIdx = 0;
  for (auto *view : op.getInputsAndOutputs()) {
    if (subViews.count(view) != 0) {
      opViews.push_back(promotedBufferAndViews[promotedIdx].fullLocalView);
      writebackViews.emplace_back(std::make_pair(
          view, promotedBufferAndViews[promotedIdx].partialLocalView));
      promotedIdx++;
    } else {
      opViews.push_back(view);
    }
  }

  // 2. Append all other operands as they appear, this enforces that such
  // operands are not views. This is to support cases such as FillOp taking
  // extra scalars etc.
  auto operands = getAssumedNonViewOperands(op);
  opViews.append(operands.begin(), operands.end());
  op.clone(b, op.getLoc(), opViews);

  // 3. Emit write-back for the promoted output views: copy the partial view.
  for (auto viewAndPartialLocalView : writebackViews) {
    // Note: use the old op to determine whether the operand view is an output.
    bool isOutput =
        op.getIndexOfOutput(viewAndPartialLocalView.first).hasValue();
    if (isOutput)
      copy(viewAndPartialLocalView.second, viewAndPartialLocalView.first);
  }

  // 4. Dealloc local buffers.
  for (const auto &pi : promotedBufferAndViews)
    buffer_dealloc(pi.buffer);
}

static void promoteSubViews(FuncOp f) {
  SmallVector<LinalgOp, 8> toErase;
  OperationFolder folder(f.getContext());
  f.walk([&folder, &toErase](LinalgOp op) {
    // TODO(ntv) some heuristic here to decide what to promote. Atm it is all or
    // nothing.
    SetVector<Value *> subViews;
    for (auto it : op.getInputsAndOutputs())
      if (auto sv = dyn_cast_or_null<SubViewOp>(it->getDefiningOp()))
        subViews.insert(sv);
    if (!subViews.empty()) {
      promoteSubViewOperands(op, subViews, &folder);
      toErase.push_back(op);
    }
  });
  for (auto op : toErase)
    op.erase();
}

namespace {
struct LinalgPromotionPass : public FunctionPass<LinalgPromotionPass> {
  void runOnFunction() override { promoteSubViews(getFunction()); }
};
} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::linalg::createLinalgPromotionPass() {
  return std::make_unique<LinalgPromotionPass>();
}

static PassRegistration<LinalgPromotionPass>
    pass("linalg-promote-subviews", "promote subview ops to local buffers");

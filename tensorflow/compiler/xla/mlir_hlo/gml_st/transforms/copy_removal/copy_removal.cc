/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <utility>

#include "gml_st/transforms/passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_NAIVECOPYREMOVALPASS
#include "gml_st/transforms/passes.h.inc"

/// Remove memref::CopyOp whose target (can be either a memref::SubViewOp or
/// memref::AllocOp) has no other users.
LogicalResult removeCopy(memref::CopyOp op, PatternRewriter &rewriter) {
  auto valueIt = op.getTarget();
  Operation *onlyNonStoreLikeUser = op;
  for (auto subviewOp = valueIt.getDefiningOp<memref::SubViewOp>(); subviewOp;
       onlyNonStoreLikeUser = subviewOp, valueIt = subviewOp.getSource(),
            subviewOp = valueIt.getDefiningOp<memref::SubViewOp>()) {
    // TODO(vuson) simplify if other uses are also memref.copy writing to
    // subview
    //    %alloc_4 = memref.alloc()
    //    %subview_5 = memref.subview %alloc_4
    //    %subview_6 = memref.subview %alloc_4
    //    memref.copy %arg0, %subview_6
    //    memref.copy %arg1, %subview_5
    if (!subviewOp->hasOneUse()) return failure();
  }

  auto hasOnlyStoreLikeUsers = [&](Value alloc) {
    return !llvm::any_of(alloc.getUsers(), [&](Operation *op) {
      if (op == onlyNonStoreLikeUser) return false;
      // TODO(vuson) remove this exception when MemoryEffectOpInterface gets
      // corrected for linalg::FillOp. Right now it has MemoryEffects::Read
      // while the only thing it ever reads is metadata such as dynamic sizes.
      if (isa<linalg::FillOp>(op)) return false;
      if (auto effect = dyn_cast<MemoryEffectOpInterface>(op)) {
        return effect.getEffectOnValue<MemoryEffects::Read>(alloc)
                   .has_value() ||
               !effect.getEffectOnValue<MemoryEffects::Write>(alloc)
                    .has_value();
      }
      return true;
    });
  };
  if (!valueIt.getDefiningOp<memref::AllocOp>() ||
      !hasOnlyStoreLikeUsers(valueIt))
    return failure();

  rewriter.eraseOp(op);
  return success();
}

struct NaiveCopyRemovalPass
    : public impl::NaiveCopyRemovalPassBase<NaiveCopyRemovalPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = func.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add(removeCopy);
    memref::AllocOp::getCanonicalizationPatterns(patterns, ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createNaiveCopyRemovalPass() {
  return std::make_unique<NaiveCopyRemovalPass>();
}

}  // namespace mlir::gml_st

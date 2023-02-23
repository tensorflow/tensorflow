/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <optional>
#include <utility>
#include <vector>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "gml_st/utils/vector_utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_OPTIMIZEVECTORTRANSFERPASS
#include "gml_st/transforms/passes.h.inc"

// Return the uses of op if they all are either StoreOp, TransferWriteOp, or
// SubviewOp with only StoreOp/TransferWriteOp users.
std::optional<llvm::SmallVector<Operation*>> getUsesIfAllStores(Operation* op) {
  llvm::SmallVector<Operation*> opUses;
  for (OpOperand& use : op->getUses()) {
    Operation* useOp = use.getOwner();
    if (isa<vector::TransferWriteOp, memref::StoreOp>(useOp)) {
      opUses.push_back(useOp);
      continue;
    }
    if (isa<memref::SubViewOp>(useOp)) {
      if (auto subviewUses = getUsesIfAllStores(useOp)) {
        opUses.insert(opUses.end(), subviewUses->begin(), subviewUses->end());
        opUses.push_back(useOp);
        continue;
      }
    }
    return std::nullopt;
  }
  return opUses;
}

// Track temporary allocations that are never read from. If this is the case
// it means both the allocations and associated stores can be removed.
void eraseDeadAllocAndStores(func::FuncOp func) {
  std::vector<Operation*> opToErase;
  func.walk([&](memref::AllocOp op) {
    if (auto uses = getUsesIfAllStores(op)) {
      // Insert the uses first,
      opToErase.insert(opToErase.end(), uses->begin(), uses->end());
      // then the op itself, since we will be erasing from opToErase's start.
      opToErase.push_back(op.getOperation());
    }
  });
  for (Operation* op : opToErase) {
    op->erase();
  }
}

// Pattern to canonialize tranpose where only one dimension is not unit
// dimension. In this case the transpose is a no-op and should be simplified
// before getting to the conversion to llvm/spirv.
class TransposeUnitDimToShapeCast
    : public OpRewritePattern<vector::TransposeOp> {
 public:
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter& rewriter) const override {
    unsigned numNonUnitSrcDim =
        llvm::count_if(op.getSourceVectorType().getShape(),
                       [](int64_t dim) { return dim != 1; });
    if (numNonUnitSrcDim != 1) return failure();
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        op, op.getResultVectorType(), op.getVector());
    return success();
  }
};

// TODO(vuson) unify this with its counterpart in IREE whenever possible.
struct OptimizeVectorTransferPass
    : public impl::OptimizeVectorTransferPassBase<OptimizeVectorTransferPass> {
  OptimizeVectorTransferPass() = default;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto* ctx = func.getContext();

    // Generate vector.shape_cast for dropping leading one dimensions in vector
    // ops. This increases the chance that we can forward more transfer writes
    // to transfer reads.
    {
      RewritePatternSet patterns(ctx);
      mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
      patterns.add<TransposeUnitDimToShapeCast>(ctx);
      mlir::vector::
          populateVectorTransferCollapseInnerMostContiguousDimsPatterns(
              patterns);
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Move bitcast inwards from loop region boundaries to increase chances to
    // cancel them.
    {
      RewritePatternSet patterns(ctx);
      vector::populateBubbleVectorBitCastOpPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Third stage of patterns to flatten transfer ops.
    {
      RewritePatternSet patterns(ctx);
      mlir::vector::populateVectorTransferDropUnitDimsPatterns(patterns);
      mlir::vector::populateFlattenVectorTransferPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
    // Delete potential dead alloc and associated ops after store to load
    // forwarding.
    eraseDeadAllocAndStores(func);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createOptimizeVectorTransferPass() {
  return std::make_unique<OptimizeVectorTransferPass>();
}
}  // namespace gml_st
}  // namespace mlir

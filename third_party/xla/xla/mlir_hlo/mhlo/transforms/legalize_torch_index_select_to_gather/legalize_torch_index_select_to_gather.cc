/* Copyright 2023 The OpenXLA Authors.

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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_LEGALIZETORCHINDEXSELECTTOGATHERPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

struct TorchIndexSelectIsGather : public OpRewritePattern<TorchIndexSelectOp> {
  using OpRewritePattern<TorchIndexSelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TorchIndexSelectOp op,
                                PatternRewriter &rewriter) const override {
    auto operand = op.getOperand();
    auto operandTy = operand.getType();
    if (!operandTy.hasRank()) {
      return rewriter.notifyMatchFailure(op, "unranked operand");
    }

    auto index = op.getIndex();
    if (!operand.getType().hasStaticShape() ||
        !index.getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "operand and index must have static shapes");
    }

    int64_t dim = static_cast<int64_t>(op.getDim());
    int64_t batchDims = op.getBatchDims();
    if (dim < batchDims) {
      return rewriter.notifyMatchFailure(
          op, "dim must be greater than or equal to the number of batch dims");
    }

    int64_t indexVectorDim = index.getType().getRank();
    auto indexTy = index.getType();
    auto indexElementTy = indexTy.getElementType().dyn_cast<IntegerType>();
    if (!indexElementTy) {
      return rewriter.notifyMatchFailure(
          op, "index must have integer element type");
    }

    if (index.getType().getElementType().getIntOrFloatBitWidth() == 64 &&
        operandTy.getShape()[dim] < std::numeric_limits<uint32_t>::max()) {
      index = rewriter.create<ConvertOp>(
          op.getLoc(), index, rewriter.getIntegerType(32, /*isSigned=*/false));
    }

    if (batchDims > 0) {
      llvm::SmallVector<int64_t> newIndexShape(indexTy.getShape());
      newIndexShape.push_back(1);
      auto newIndexType = RankedTensorType::get(
          newIndexShape, index.getType().getElementType());

      llvm::SmallVector<Value> toConcat;
      for (auto batchDim = 0; batchDim < batchDims; ++batchDim) {
        toConcat.push_back(
            rewriter.create<IotaOp>(op.getLoc(), newIndexType, batchDim));
      }
      toConcat.push_back(
          rewriter.create<ReshapeOp>(op.getLoc(), newIndexType, index));
      index = rewriter.create<ConcatenateOp>(op.getLoc(), ValueRange(toConcat),
                                             indexVectorDim);
    }

    llvm::SmallVector<int64_t> offsetDims;
    llvm::SmallVector<int64_t> collapsedSliceDims;
    llvm::SmallVector<int64_t> startIndexMap;
    llvm::SmallVector<int64_t> sliceSizes(operandTy.getShape());
    for (auto i = 0; i < operandTy.getRank(); ++i) {
      if (i < batchDims || i == dim) {
        sliceSizes[i] = std::min(sliceSizes[i], static_cast<int64_t>(1));
        collapsedSliceDims.push_back(i);
        startIndexMap.push_back(i);
      } else {
        if (i < dim) {
          offsetDims.push_back(i);
        } else {
          offsetDims.push_back(i + indexVectorDim - (1 + batchDims));
        }
      }
    }

    auto gatherDimensionNumbersAttr = GatherDimensionNumbersAttr::get(
        rewriter.getContext(), offsetDims, collapsedSliceDims, startIndexMap,
        indexVectorDim);

    auto sliceSizesAttr = rewriter.getI64TensorAttr(sliceSizes);

    auto gatherOp =
        rewriter.create<GatherOp>(op.getLoc(), operand, index,
                                  gatherDimensionNumbersAttr, sliceSizesAttr);
    rewriter.replaceOp(op, gatherOp);
    return success();
  }
};

struct LegalizeTorchIndexSelectToGatherPass
    : public impl::LegalizeTorchIndexSelectToGatherPassBase<
          LegalizeTorchIndexSelectToGatherPass> {
  /// Perform the lowering of standard dialect operations to approximations.
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateTorchIndexSelectToGatherPatterns(&getContext(), &patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
}  // namespace

void populateTorchIndexSelectToGatherPatterns(mlir::MLIRContext *context,
                                              RewritePatternSet *patterns) {
  patterns->add<TorchIndexSelectIsGather>(context);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeTorchIndexSelectToGatherPass() {
  return std::make_unique<LegalizeTorchIndexSelectToGatherPass>();
}

}  // namespace mhlo
}  // namespace mlir

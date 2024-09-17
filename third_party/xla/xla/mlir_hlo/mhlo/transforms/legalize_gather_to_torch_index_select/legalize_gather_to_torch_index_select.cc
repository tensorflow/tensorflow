/* Copyright 2019 The OpenXLA Authors.

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
#include <memory>
#include <utility>

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_LEGALIZEGATHERTOTORCHINDEXSELECTPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

struct GatherIsTorchIndexSelect : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp gather,
                                PatternRewriter &rewriter) const override {
    auto startIndices = gather.getStartIndices();
    auto startIndicesTy = mlir::cast<ShapedType>(startIndices.getType());
    if (!startIndicesTy.hasRank()) {
      return rewriter.notifyMatchFailure(gather, "unranked start_indices");
    }

    auto operand = gather.getOperand();
    auto operandTy = mlir::cast<ShapedType>(operand.getType());
    if (!operandTy.hasRank()) {
      return rewriter.notifyMatchFailure(gather, "unranked operand");
    }

    int64_t indexVectorDim = std::max<int64_t>(0, startIndicesTy.getRank() - 1);

    // We can use torch_index_select if the last dimension represents the
    // gather indices.
    auto dimensionNumbers = gather.getDimensionNumbers();
    if (dimensionNumbers.getIndexVectorDim() != indexVectorDim) {
      return rewriter.notifyMatchFailure(
          gather, "index_vector_dim not last dimension of start_indices");
    }

    // Index select only works across a single dimension.
    if (!startIndicesTy.getShape().empty() &&
        startIndicesTy.getShape().back() != 1) {
      return rewriter.notifyMatchFailure(
          gather, "start_indices index vector dimension not 1");
    }

    // Only support the default case for start_index_map.
    if (dimensionNumbers.getStartIndexMap().size() != 1 ||
        dimensionNumbers.getStartIndexMap()[0] != 0) {
      return rewriter.notifyMatchFailure(gather, "start_index_map != [0]");
    }

    auto resultTy =
        mlir::dyn_cast<RankedTensorType>(gather.getResult().getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(gather, "unranked result");
    }

    // Offset dimensions should be the defaults.
    if (static_cast<int64_t>(dimensionNumbers.getOffsetDims().size()) !=
        resultTy.getRank() - indexVectorDim) {
      return rewriter.notifyMatchFailure(
          gather, "offset_dims.size not operand rank minus index_vector_dim");
    }

    for (const auto &it : llvm::enumerate(dimensionNumbers.getOffsetDims())) {
      if (static_cast<int64_t>(it.index() + indexVectorDim) != it.value()) {
        return rewriter.notifyMatchFailure(
            gather, "offset_dims != [index_vector_dim, result.rank)");
      }
    }

    for (const auto &it :
         llvm::enumerate(gather.getSliceSizes().getValues<APInt>())) {
      // First shape value must be 1.
      if (it.index() == 0) {
        if (it.value().getSExtValue() != 1) {
          return rewriter.notifyMatchFailure(gather, "slice_size[0] != 1");
        }
        continue;
      }

      // The gather needs to index the entire slice for each other dimension.
      if (it.value().getSExtValue() != operandTy.getDimSize(it.index())) {
        return rewriter.notifyMatchFailure(
            gather, "slice_size doesn't match operand dimension");
      }
    }

    llvm::SmallVector<int64_t, 4> indexSelectShape =
        llvm::to_vector<4>(startIndicesTy.getShape());

    for (auto dim : operandTy.getShape().drop_front()) {
      indexSelectShape.push_back(dim);
    }

    if (dimensionNumbers.getCollapsedSliceDims().size() != 1 ||
        dimensionNumbers.getCollapsedSliceDims()[0] != 0) {
      return rewriter.notifyMatchFailure(gather, "collapsed_slice_dims != [0]");
    }

    auto torchIndexSelect = rewriter.create<TorchIndexSelectOp>(
        gather.getLoc(),
        RankedTensorType::get(indexSelectShape, operandTy.getElementType()),
        operand, gather.getStartIndices(), rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(0));

    rewriter.replaceOpWithNewOp<ReshapeOp>(gather, gather.getType(),
                                           torchIndexSelect);

    return success();
  }
};

struct LegalizeGatherToTorchIndexSelectPass
    : public impl::LegalizeGatherToTorchIndexSelectPassBase<
          LegalizeGatherToTorchIndexSelectPass> {
  /// Perform the lowering of standard dialect operations to approximations.
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGatherToTorchIndexSelectPatterns(&getContext(), &patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
}  // namespace

void populateGatherToTorchIndexSelectPatterns(mlir::MLIRContext *context,
                                              RewritePatternSet *patterns) {
  patterns->add<GatherIsTorchIndexSelect>(context);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeGatherToTorchIndexSelectPass() {
  return std::make_unique<LegalizeGatherToTorchIndexSelectPass>();
}

}  // namespace mhlo
}  // namespace mlir

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

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMSORTFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

using mlir::arith::ConstantIndexOp;
using mlir::thlo::SortOp;

constexpr llvm::StringRef kSortTransformedLabel = "__sort_transformed_label__";

struct TileSortPattern : public OpRewritePattern<SortOp> {
  TileSortPattern(MLIRContext *context, TilingOptions options,
                  PatternBenefit benefit = 1)
      : OpRewritePattern<thlo::SortOp>(context, benefit),
        options(std::move(options)) {}

  LogicalResult matchAndRewrite(SortOp op,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(op, kSortTransformedLabel)) return failure();

    if (isa<gml_st::ParallelOp, scf::ForOp>(op->getParentOp())) {
      return rewriter.notifyMatchFailure(
          op, "has already been tiled by another pass.");
    }

    auto tilingResult = tileUsingGmlSt(
        options, rewriter, cast<TilingInterface>(op.getOperation()));
    if (failed(tilingResult)) return failure();

    // If we did not tile (e.g. when all tile sizes are 0), do not replace
    // original op and just mark it as transformed then return.
    if (tilingResult->loop != nullptr) {
      rewriter.replaceOp(op, tilingResult->loop->getResults());
    }
    setLabel(tilingResult->tiledOps.front(), kSortTransformedLabel);
    return success();
  }

 private:
  TilingOptions options;
};

struct TransformSortForCpuPass
    : public impl::TransformSortForCpuPassBase<TransformSortForCpuPass> {
  TransformSortForCpuPass() = default;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<GmlStDialect, arith::ArithDialect, tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    auto getTileSize = [&](mlir::OpBuilder b, Operation *op) {
      // use tile sizes 1 by default
      auto sortOp = llvm::cast<SortOp>(op);
      auto size = sortOp.getLoopIteratorTypes().size();
      return SmallVector<Value>(size,
                                b.create<ConstantIndexOp>(op->getLoc(), 1));
    };

    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    TilingOptions tilingOptions;
    tilingOptions.tileSizeComputationFn = getTileSize;

    RewritePatternSet patterns(ctx);
    patterns.add<TileSortPattern>(ctx, tilingOptions);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Ensure we drop the marker in the end.
    f.walk([](thlo::SortOp sortOp) {
      removeLabel(sortOp, kSortTransformedLabel);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformSortForCpuPass() {
  return std::make_unique<TransformSortForCpuPass>();
}

}  // namespace mlir::gml_st

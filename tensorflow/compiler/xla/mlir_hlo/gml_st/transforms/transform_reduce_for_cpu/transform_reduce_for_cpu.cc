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

#include <cstdint>
#include <memory>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/interfaces/tiling_interface_impl.h"
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMREDUCEFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

static constexpr llvm::StringRef kReduceTransformedLabel =
    "__reduce_transformed_label__";

FailureOr<TilingResult> tileReduce(PatternRewriter &rewriter,
                                   linalg::ReduceOp reduceOp,
                                   ArrayRef<int64_t> tileSizes,
                                   bool distribute) {
  TilingOptions opts;
  opts.setTileSizeComputationFn(tileSizes);
  opts.distribute = distribute;
  return tile(opts, rewriter, cast<TilingInterface>(reduceOp.getOperation()));
}

SmallVector<int64_t> getParallelDimTileSizes(int64_t reductionDim,
                                             int64_t parallelDimTileSize) {
  return reductionDim ? SmallVector<int64_t>{parallelDimTileSize, 0}
                      : SmallVector<int64_t>{0, parallelDimTileSize};
}

SmallVector<int64_t> getReductionDimTileSizes(int64_t reductionDim,
                                              int64_t reductionDimTileSize) {
  return reductionDim ? SmallVector<int64_t>{0, reductionDimTileSize}
                      : SmallVector<int64_t>{reductionDimTileSize, 0};
}

LogicalResult validateOp(linalg::ReduceOp reduceOp, PatternRewriter &rewriter) {
  ArrayRef<int64_t> reduceDimensions = reduceOp.getDimensions();
  if (reduceDimensions.size() != 1)
    return rewriter.notifyMatchFailure(
        reduceOp, "expects 1 reduction dimension element. 0 or > 1 received.");
  OpOperandVector operands = reduceOp.getDpsInputOperands();
  if (operands.size() != 1)
    return rewriter.notifyMatchFailure(reduceOp,
                                       "expects 1 operand. 0 or > 1 received.");
  if (cast<RankedTensorType>(operands[0]->get().getType()).getRank() != 2)
    return rewriter.notifyMatchFailure(reduceOp, "expects rank 2.");

  return success();
}

/// Pattern to tile `linalg.reduce` and fuse `linalg.fill` into generated
/// `gml_st.parallel`.
struct ReduceTransformPattern : public OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  explicit ReduceTransformPattern(MLIRContext *context,
                                  int64_t parallelDimTileSize = 4,
                                  int64_t reductionDimTileSize = 2,
                                  PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::ReduceOp>(context, benefit),
        parallelDimTileSize(parallelDimTileSize),
        reductionDimTileSize(reductionDimTileSize) {}

  LogicalResult matchAndRewrite(linalg::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(reduceOp, kReduceTransformedLabel))
      return rewriter.notifyMatchFailure(reduceOp,
                                         "has already been transformed.");

    if (failed(validateOp(reduceOp, rewriter))) return failure();

    // First level tiling: parallel dimension.
    auto tilingParallelDimsResult =
        tileReduce(rewriter, reduceOp,
                   getParallelDimTileSizes(reduceOp.getDimensions()[0],
                                           parallelDimTileSize),
                   /*distribute=*/true);
    if (failed(tilingParallelDimsResult)) return failure();

    // Update the results if tiling occurred.
    if (tilingParallelDimsResult->loop != nullptr) {
      rewriter.replaceOp(reduceOp,
                         tilingParallelDimsResult->loop->getResults());
      reduceOp = cast<linalg::ReduceOp>(tilingParallelDimsResult->tiledOp);
      // Fuse linalg.map ops into the loop.
      fuseGreedily(rewriter, *reduceOp->getBlock(),
                   [](Operation *op) { return isa<linalg::MapOp>(op); });
    }

    // Fusion into the output.
    OpOperand *reduceOutput = reduceOp.getDpsInitOperand(0);
    auto materialize = reduceOutput->get().getDefiningOp<MaterializeOp>();
    if (!materialize) {
      return rewriter.notifyMatchFailure(
          reduceOp,
          "has failed to 'materialize' output during 'linalg.fill' fusion.");
    }
    if (materialize.getSource().getDefiningOp<linalg::FillOp>()) {
      if (failed(fuse(rewriter, materialize))) return failure();
    }

    // Second level tiling: reduction dimension.
    auto tilingReductionDimsResult =
        tileReduce(rewriter, reduceOp,
                   getReductionDimTileSizes(reduceOp.getDimensions()[0],
                                            reductionDimTileSize),
                   /*distribute=*/false);
    if (failed(tilingReductionDimsResult)) return failure();

    // Update the results if tiling occurred.
    if (tilingReductionDimsResult->loop != nullptr) {
      rewriter.replaceOp(reduceOp,
                         tilingReductionDimsResult->loop->getResults());
      reduceOp = cast<linalg::ReduceOp>(tilingReductionDimsResult->tiledOp);
      // Fuse linalg.map ops into the loop.
      fuseGreedily(rewriter, *reduceOp->getBlock(),
                   [](Operation *op) { return isa<linalg::MapOp>(op); });
    }

    setLabel(reduceOp, kReduceTransformedLabel);

    // Peel parallel loops.
    if (auto loop =
            dyn_cast_or_null<ParallelOp>(tilingParallelDimsResult->loop)) {
      auto peelingResult = peelAllLoops(loop, rewriter);
      // Mark all for loops inside remainder parallel loops as peeled to prevent
      // downstream peeling pass from peeling them.
      for (auto *remParLoop : peelingResult) {
        remParLoop->walk([&](Operation *childOp) {
          if (isa<ForOp>(childOp)) {
            setLabel(childOp, kPeelingAppliedLabel);
          }
        });
      }
    }

    // Peel reduction loop inside the main parallel loop.
    if (auto loop = dyn_cast_or_null<ForOp>(tilingReductionDimsResult->loop)) {
      peelAllLoops(loop, rewriter);
    }

    return success();
  }

 private:
  int64_t parallelDimTileSize;
  int64_t reductionDimTileSize;
};

struct TransformReduceForCpuPass
    : public impl::TransformReduceForCpuPassBase<TransformReduceForCpuPass> {
  TransformReduceForCpuPass() = default;

  explicit TransformReduceForCpuPass(llvm::ArrayRef<int64_t> reduceTileSizes) {
    tileSizes = reduceTileSizes;
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<mlir::gml_st::GmlStDialect, arith::ArithDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
    mlir::gml_st::registerGmlStTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    if (tileSizes.empty()) {
      tileSizes = {4, 2};
    }

    assert(tileSizes.size() == 2 &&
           "Tiling sizes for Reduce should have 2 element.");

    RewritePatternSet patterns(ctx);
    patterns.add<ReduceTransformPattern>(ctx, tileSizes[0], tileSizes[1]);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Ensure we drop the marker in the end.
    f.walk([](linalg::ReduceOp reduceOp) {
      removeLabel(reduceOp, kReduceTransformedLabel);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformReduceForCpuPass(llvm::ArrayRef<int64_t> reduceTileSizes) {
  return std::make_unique<mlir::gml_st::TransformReduceForCpuPass>(
      reduceTileSizes);
}

}  // namespace mlir::gml_st

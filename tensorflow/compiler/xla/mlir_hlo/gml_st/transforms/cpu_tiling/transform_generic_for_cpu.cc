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

#include <cstdint>
#include <memory>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMGENERICFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

constexpr llvm::StringRef kGenericTransformedLabel =
    "__generic_transformed_label__";

using utils::IteratorType;

// Returns a vector of tile sizes, where all elements are ones if the
// corresponding dimensions have the given iteration type. The rest of the
// elements are zeros.
SmallVector<int64_t> getSizeOneTileSizesForIterType(TilingInterface op,
                                                    IteratorType type) {
  return llvm::to_vector(llvm::map_range(op.getLoopIteratorTypes(),
                                         [&](IteratorType iterType) -> int64_t {
                                           return iterType == type ? 1 : 0;
                                         }));
}

/// Pattern to tile parallel dims of `linalg.generic`, fuse and then tile the
/// reduction dims and fuse again.
struct GenericTransformPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(genericOp, kGenericTransformedLabel)) {
      return rewriter.notifyMatchFailure(genericOp,
                                         "has already been transformed.");
    }

    if (isa<gml_st::ParallelOp, scf::ForOp>(genericOp->getParentOp())) {
      return rewriter.notifyMatchFailure(
          genericOp, "has already been tiled by another pass.");
    }

    // Find fusion cluster.
    auto producerFilterFn = [](Operation *op) {
      return isa<linalg::BroadcastOp, linalg::FillOp, linalg::MapOp,
                 linalg::TransposeOp, tensor::CastOp>(op);
    };
    auto cluster = getFusionCluster(genericOp, producerFilterFn);
    auto *tilingRoot = cluster.root;
    if (!isa<linalg::MapOp>(tilingRoot) &&
        !isa<linalg::GenericOp>(tilingRoot)) {
      return rewriter.notifyMatchFailure(
          tilingRoot,
          "Expected MapOp or GenericOp as a root of fusion cluster.");
    }

    // First level tiling: parallel dimensions.
    auto tilingParallelDimsResult = tileParallelDims(rewriter, tilingRoot);
    if (failed(tilingParallelDimsResult)) return failure();

    // Update the results if tiling occurred.
    rewriter.replaceOp(tilingRoot,
                       tilingParallelDimsResult->loop->getResults());
    assert(tilingParallelDimsResult->tiledOps.size() == 1 &&
           "Expected only one tiled op generated");

    tilingRoot = (tilingParallelDimsResult->tiledOps.front());

    // Fuse greedily into root op.
    fuseGreedily(rewriter, *tilingRoot->getBlock(), [&](Operation *op) {
      return cluster.operations.contains(op);
    });

    (void)fuseFillOpsIntoParallelOp(rewriter, tilingParallelDimsResult->loop);

    // Second level of tiling: reduction dimensions.
    for (auto tiledGenericOp :
         llvm::to_vector(tilingRoot->getBlock()->getOps<linalg::GenericOp>())) {
      FailureOr<scf::SCFTilingResult> reductionDimTilingResult =
          tileUsingSCFForOpAndFuseGreedily(
              rewriter, tiledGenericOp,
              getSCFTilingOptions(getSizeOneTileSizesForIterType(
                  tiledGenericOp.getOperation(), IteratorType::reduction)),
              kGenericTransformedLabel, producerFilterFn);
      if (failed(reductionDimTilingResult)) return failure();
    }

    return success();
  }

 private:
  // Find a cluster of operations that can be tiled and fused together around
  // the root op.
  FusionCluster getFusionCluster(
      linalg::GenericOp genericOp,
      llvm::function_ref<bool(Operation *)> filterFn) const {
    // Find a chain of MapOp users and use the last one as a root of cluster.
    SetVector<Operation *> resultOps;
    Operation *rootOp = genericOp.getOperation();

    while (rootOp->hasOneUse() && isa<linalg::MapOp>(*rootOp->user_begin())) {
      resultOps.insert(rootOp);
      rootOp = *rootOp->user_begin();
    }

    // Run DFS to find all MapOps, TransposeOps, BroadcastOps that can be
    // fused in the root op.
    SmallVector<Operation *> remainingProducers;
    remainingProducers.reserve(genericOp.getDpsInputOperands().size());
    resultOps.insert(genericOp.getOperation());
    for (Value operand : genericOp.getOperands())
      remainingProducers.push_back(operand.getDefiningOp());

    while (!remainingProducers.empty()) {
      Operation *curOp = remainingProducers.pop_back_val();
      if (!curOp || resultOps.contains(curOp)) continue;
      if (!isa<linalg::GenericOp>(curOp) && filterFn(curOp)) {
        resultOps.insert(curOp);
        for (Value operand : genericOp.getOperands())
          remainingProducers.push_back(operand.getDefiningOp());
      }
    }
    return {resultOps, rootOp};
  }

  FailureOr<TilingResult> tileParallelDims(PatternRewriter &rewriter,
                                           Operation *tilingRoot) const {
    auto tileableOp = cast<TilingInterface>(tilingRoot);
    TilingOptions opts;
    opts.setTileSizeComputationFn(
        getSizeOneTileSizesForIterType(tileableOp, IteratorType::parallel));

    return tileUsingGmlSt(opts, rewriter, tileableOp);
  }
};

struct TransformGenericForCpuPass
    : public impl::TransformGenericForCpuPassBase<TransformGenericForCpuPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<mlir::gml_st::GmlStDialect, arith::ArithDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<GenericTransformPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Ensure we drop the marker in the end.
    f.walk([](linalg::GenericOp genericOp) {
      removeLabel(genericOp, kGenericTransformedLabel);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformGenericForCpuPass() {
  return std::make_unique<mlir::gml_st::TransformGenericForCpuPass>();
}

}  // namespace mlir::gml_st

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
#include <string>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/interfaces/tiling_interface.h"
#include "gml_st/interfaces/tiling_interface_impl.h"
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "gml_st/utils/linalg_utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_GREEDYTILINGANDFUSIONPASS
#include "gml_st/transforms/passes.h.inc"

namespace {

class FuseTensorExtractPattern : public OpRewritePattern<tensor::ExtractOp> {
 public:
  explicit FuseTensorExtractPattern(MLIRContext *context)
      : OpRewritePattern<tensor::ExtractOp>(context) {}

  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    if (extractOp->getParentOfType<ParallelOp>())
      return rewriter.notifyMatchFailure(extractOp, "already fused");

    if (extractOp->getUsers().empty())
      return rewriter.notifyMatchFailure(extractOp, "op is trivially dead");

    ParallelOp outerMostParallelOp;
    for (Operation *user : extractOp->getUsers()) {
      ParallelOp parallelOp = user->getParentOfType<gml_st::ParallelOp>();
      while (parallelOp && parallelOp->getParentOfType<gml_st::ParallelOp>())
        parallelOp = parallelOp->getParentOfType<gml_st::ParallelOp>();

      if (!parallelOp)
        return rewriter.notifyMatchFailure(extractOp, "consumer is not fused");

      if (!outerMostParallelOp)
        outerMostParallelOp = parallelOp;
      else if (outerMostParallelOp != parallelOp)
        return rewriter.notifyMatchFailure(
            extractOp,
            "consumers are not all nested under the same ParallelOp");
    }

    rewriter.setInsertionPointToStart(outerMostParallelOp.getBody());
    Value newExtractOp = rewriter.create<tensor::ExtractOp>(
        extractOp.getLoc(), extractOp.getTensor(), extractOp.getIndices());
    rewriter.replaceAllUsesWith(extractOp, newExtractOp);

    return success();
  }
};

}  // namespace

struct GreedyTilingAndFusionPass
    : public impl::GreedyTilingAndFusionPassBase<GreedyTilingAndFusionPass> {
  GreedyTilingAndFusionPass() = default;
  GreedyTilingAndFusionPass(bool distr, ArrayRef<int64_t> ts, StringRef dl) {
    this->distribute = distr;
    this->tileSizes = ts;
    this->distributionLabel = dl.str();
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry
        .insert<GmlStDialect, linalg::LinalgDialect, tensor::TensorDialect>();
    registerGmlStTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    TilingOptions opts;
    opts.distribute = distribute;
    opts.distributionLabel = distributionLabel;
    SmallVector<int64_t> ts(tileSizes.begin(), tileSizes.end());
    opts.tileSizeComputationFn = [ts](OpBuilder &b, Operation *op) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(
          &op->getParentOfType<func::FuncOp>().getBody().front());
      return llvm::to_vector(llvm::map_range(ts, [&](int64_t s) {
        Value v = b.create<arith::ConstantIndexOp>(op->getLoc(), s);
        return v;
      }));
    };

    auto tilingFilterFn = [&](TilingInterface op) {
      return success(llvm::none_of(op->getUsers(), [](Operation *user) {
        return llvm::isa<MaterializeOp>(user) ||
               llvm::isa<gml_st::TilingInterface>(user);
      }));
    };

    {
      RewritePatternSet patterns(ctx);
      populateTilingPatterns(ctx, tilingFilterFn, opts, &patterns);

      auto fusionFilterFn = [](MaterializeOp) { return success(); };
      populateFusionPatterns(ctx, fusionFilterFn, &patterns);

      if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
        return signalPassFailure();
    }

    RewritePatternSet patterns(ctx);

    patterns.add<FuseTensorExtractPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();

    // Clean up by removing temporary attributes.
    removeTilingLabels(f);
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createGreedyTilingAndFusionPass() {
  return std::make_unique<GreedyTilingAndFusionPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createGreedyTilingAndFusionPass(
    bool distribute, ArrayRef<int64_t> tileSizes, StringRef distributionLabel) {
  return std::make_unique<GreedyTilingAndFusionPass>(distribute, tileSizes,
                                                     distributionLabel);
}

}  // namespace gml_st
}  // namespace mlir

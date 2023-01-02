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

#include <algorithm>
#include <memory>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/interfaces/tiling_interface_impl.h"
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "gml_st/utils/linalg_utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_TILINGCWISEPASS
#include "gml_st/transforms/passes.h.inc"

bool isRootOfCwiseExpr(Operation *op) {
  return isCwiseGenericOp(op) &&
         llvm::none_of(op->getUsers(), [](Operation *user) {
           return isCwiseGenericOp(user) || llvm::isa<MaterializeOp>(user);
         });
}

struct TilingCwisePass : public impl::TilingCwisePassBase<TilingCwisePass> {
  TilingCwisePass() = default;
  TilingCwisePass(bool distribute, ArrayRef<int64_t> tileSizes,
                  StringRef distributionLabel) {
    distribute_ = distribute;
    tileSizes_ = tileSizes;
    distributionLabel_ = distributionLabel.str();
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry
        .insert<GmlStDialect, linalg::LinalgDialect, tensor::TensorDialect>();
    registerGmlStTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    // Populate tiling options.
    TilingOptions opts;
    opts.distribute = distribute_;
    opts.tileSizeComputationFn = [&](OpBuilder &b, Operation *op) {
      Location loc = op->getLoc();
      int64_t rank =
          op->getResultTypes().front().dyn_cast<RankedTensorType>().getRank();
      SmallVector<Value> tileSizesValues(
          rank, b.create<arith::ConstantIndexOp>(loc, 1));

      // Materialize the given right-aligned tile sizes as constants.
      int64_t i = rank - 1;
      for (int64_t t : llvm::reverse(tileSizes_)) {
        tileSizesValues[i--] = b.create<arith::ConstantIndexOp>(loc, t);
        if (i < 0) break;
      }

      return tileSizesValues;
    };
    opts.distributionLabel = distributionLabel_;

    // Tile the roots of cwise expressions and fuse all cwise operands greedily.
    auto tileRootOfCwiseExprFn = [](TilingInterface op) {
      return success(isRootOfCwiseExpr(op));
    };
    auto fuseCwiseOperandsGreedilyFn = [](MaterializeOp op) {
      return success(isCwiseGenericOp(op.getSource().getDefiningOp()));
    };

    // Populate tiling and fusion patterns.
    RewritePatternSet patterns(ctx);
    populateTilingPatterns(ctx, tileRootOfCwiseExprFn, opts, &patterns);
    populateFusionPatterns(ctx, fuseCwiseOperandsGreedilyFn, &patterns);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Clean up by removing temporary attributes.
    removeTilingLabels(f);
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createTilingCwisePass() {
  return std::make_unique<TilingCwisePass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createTilingCwisePass(
    bool distribute, ArrayRef<int64_t> tileSizes, StringRef distributionLabel) {
  return std::make_unique<TilingCwisePass>(distribute, tileSizes,
                                           distributionLabel);
}

}  // namespace gml_st
}  // namespace mlir

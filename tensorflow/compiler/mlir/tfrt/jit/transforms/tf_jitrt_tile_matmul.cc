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
#include <vector>

#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_DEF_TILEMATMUL
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using mlir::failure;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::success;
using mlir::linalg::LinalgOp;
using mlir::linalg::LinalgTilingOptions;
using mlir::linalg::MatmulOp;

// Tiles a linalg.matmul op.
struct MatmulTilingPattern : public OpRewritePattern<MatmulOp> {
  MatmulTilingPattern(const LinalgTilingOptions &options, MLIRContext *context,
                      mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<MatmulOp>(context, benefit), options(options) {}

  LogicalResult matchAndRewrite(MatmulOp linalg_op,
                                PatternRewriter &rewriter) const override {
    if (hasTransformationAttr(linalg_op)) return failure();

    auto tiled_op = mlir::gml_st::tileLinalgOp(rewriter, linalg_op, options);

    if (failed(tiled_op)) return failure();

    // If we did not tile (e.g. when all tile sizes are 0), just mark the matmul
    // op as transformed and return.
    if (tiled_op->loops.empty()) {
      setTransformationAttr(rewriter, linalg_op);
      return success();
    }

    tiled_op->loops.front()->walk(
        [&](LinalgOp tOp) { setTransformationAttr(rewriter, tOp); });

    rewriter.replaceOp(linalg_op, tiled_op->tensorResults);
    return success();
  }

 private:
  LinalgTilingOptions options;
};

// TODO(vuson): Handle batch_*matmul + matmul_unsigned + quantized_*matmul
struct TileMatmulPass : public impl::TileMatmulBase<TileMatmulPass> {
  TileMatmulPass() = default;
  explicit TileMatmulPass(llvm::ArrayRef<int64_t> matmul_tiles) {
    matmul_tile_sizes = matmul_tiles;
  }
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    assert(matmul_tile_sizes.size() <= 3 &&
           "Tiling sizes for MatMul should only have at most 3 elements");

    // If we have at least two tile sizes, generate a parallel gml_st.loop for
    // the parallel dimensions.
    if (matmul_tile_sizes.size() & 0b10) {
      llvm::SmallVector<int64_t, 3> parTileSizeVector(
          matmul_tile_sizes.begin(), matmul_tile_sizes.begin() + 2);
      parTileSizeVector.push_back(0);
      auto parPatterns =
          mlir::linalg::getLinalgTilingCanonicalizationPatterns(context);
      parPatterns.add<MatmulTilingPattern>(
          // LinalgTilingOptions{}.setTileSizes(matmul_tile_sizes),
          LinalgTilingOptions{}.setTileSizes(parTileSizeVector),
          parPatterns.getContext());
      if (failed(mlir::applyPatternsAndFoldGreedily(func,
                                                    std::move(parPatterns)))) {
        return signalPassFailure();
      }
      // Ensure we drop the marker in the end.
      func.walk([](LinalgOp op) { removeTransformationAttr(op); });
    }

    // If we have an odd number of tile sizes, generate a sequential gml_st.loop
    // for the reduction dimension.
    if (matmul_tile_sizes.size() & 0b1) {
      llvm::SmallVector<int64_t, 3> seqTileSizeVector({0, 0});
      seqTileSizeVector.push_back((*matmul_tile_sizes).back());
      auto seqPatterns =
          mlir::linalg::getLinalgTilingCanonicalizationPatterns(context);
      seqPatterns.add<MatmulTilingPattern>(
          LinalgTilingOptions{}.setTileSizes(seqTileSizeVector),
          seqPatterns.getContext());
      if (failed(mlir::applyPatternsAndFoldGreedily(func,
                                                    std::move(seqPatterns)))) {
        return signalPassFailure();
      }

      // Ensure we drop the marker in the end.
      func.walk([](LinalgOp op) { removeTransformationAttr(op); });
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTileMatmulPass() {
  return std::make_unique<TileMatmulPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateTileMatmulPass(
    llvm::ArrayRef<int64_t> matmul_tile_sizes) {
  return std::make_unique<TileMatmulPass>(matmul_tile_sizes);
}

}  // namespace tensorflow

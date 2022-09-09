/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_DEF_TILEFILL
#define GEN_PASS_DEF_TILECWISE
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using mlir::failure;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::PatternRewriter;
using mlir::SmallVector;
using mlir::success;
using mlir::Value;
using mlir::arith::ConstantIndexOp;
using mlir::gml_st::LoopOp;
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::linalg::LinalgOp;
using mlir::linalg::LinalgTilingOptions;

struct TileCWisePattern : public mlir::OpInterfaceRewritePattern<LinalgOp> {
  TileCWisePattern(LinalgTilingOptions options, MLIRContext *context,
                   llvm::function_ref<bool(Operation *)> match_fn,
                   mlir::PatternBenefit benefit = 1)
      : mlir::OpInterfaceRewritePattern<LinalgOp>(context, benefit),
        match_fn(match_fn),
        options(options) {}

  LogicalResult matchAndRewrite(LinalgOp linalg_op,
                                PatternRewriter &rewriter) const override {
    if (hasTransformationAttr(linalg_op)) return failure();
    if (!match_fn(linalg_op)) return failure();

    auto tiled_linalg_op =
        mlir::gml_st::tileLinalgOp(rewriter, linalg_op, options);
    if (failed(tiled_linalg_op) || tiled_linalg_op.getValue().loops.empty())
      return failure();

    LoopOp tiled_loop =
        mlir::dyn_cast<LoopOp>(*tiled_linalg_op.getValue().loops.front());
    if (!tiled_loop) return failure();

    tiled_loop->walk(
        [&](LinalgOp tiledOp) { setTransformationAttr(rewriter, tiledOp); });

    rewriter.replaceOp(linalg_op, tiled_loop->getResults());
    return success();
  }

 private:
  llvm::function_ref<bool(Operation *)> match_fn;
  LinalgTilingOptions options;
};

// Return true if the generic has only parallel iterations. This disallows
// windowed and reduction iteration.
bool isNonTiledCwiseGeneric(Operation *op) {
  if (op->getParentOfType<LoopOp>()) return false;
  auto linalg_op = mlir::dyn_cast<GenericOp>(op);
  if (linalg_op) {
    if (!linalg_op.hasTensorSemantics()) return false;
    return llvm::all_of(linalg_op.iterator_types(), [](auto type) {
      return mlir::isParallelIterator(type);
    });
  }
  if (auto fill_op = mlir::dyn_cast<FillOp>(op)) {
    return fill_op.hasTensorSemantics();
  }
  return false;
}

// Return true if the generic has only parallel iterations. This disallows
// windowed and reduction iteration.
bool isNonTiledFill(Operation *op) {
  if (op->getParentOfType<LoopOp>()) return false;
  if (auto fill_op = mlir::dyn_cast<FillOp>(op)) {
    return fill_op.hasTensorSemantics();
  }
  return false;
}

void Tile(mlir::func::FuncOp func, int64_t tile_size,
          llvm::function_ref<bool(Operation *)> match_fn) {
  LinalgTilingOptions tiling_options;
  // Tile the innermost dimension by `tile_size` for vectorization and scalarize
  // the other dimensions.
  tiling_options.setTileSizeComputationFunction(
      [&](OpBuilder b, Operation *op) {
        auto num_loops = llvm::cast<LinalgOp>(op).getNumLoops();
        SmallVector<Value> tiles(num_loops,
                                 b.create<ConstantIndexOp>(op->getLoc(), 1));
        if (!tiles.empty())
          tiles.back() = b.create<ConstantIndexOp>(op->getLoc(), tile_size);
        return tiles;
      });

  mlir::RewritePatternSet patterns(func.getContext());
  patterns.add<TileCWisePattern>(tiling_options, patterns.getContext(),
                                 match_fn);
  (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));

  // Ensure we drop the marker in the end.
  func.walk([](LinalgOp op) { removeTransformationAttr(op); });
}

struct TileCWisePass : public impl::TileCWiseBase<TileCWisePass> {
  TileCWisePass() = default;
  explicit TileCWisePass(int64_t tile_size) { cwise_tile_size = tile_size; }

  void runOnOperation() override {
    auto func = getOperation();
    Tile(func, cwise_tile_size, isNonTiledCwiseGeneric);
  }
};

struct TileFillPass : public impl::TileFillBase<TileFillPass> {
  TileFillPass() = default;
  explicit TileFillPass(int64_t tile_size) { cwise_tile_size = tile_size; }

  void runOnOperation() override {
    auto func = getOperation();
    Tile(func, cwise_tile_size, isNonTiledFill);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateTileCWisePass() {
  return std::make_unique<TileCWisePass>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateTileCWisePass(
    int64_t cwise_tile_size) {
  return std::make_unique<TileCWisePass>(cwise_tile_size);
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateTileFillPass() {
  return std::make_unique<TileFillPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> CreateTileFillPass(
    int64_t cwise_tile_size) {
  return std::make_unique<TileFillPass>(cwise_tile_size);
}

}  // namespace tensorflow

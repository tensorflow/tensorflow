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

#include <utility>

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h.inc"

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
using mlir::linalg::GenericOp;
using mlir::linalg::LinalgOp;
using mlir::linalg::LinalgTilingOptions;
using mlir::linalg::LinalgTransformationFilter;
using mlir::linalg::TiledLoopOp;

struct TileCWisePattern : public mlir::OpInterfaceRewritePattern<LinalgOp> {
  /// MatchAnyOpTag-based constructor with a mandatory `filter`.
  TileCWisePattern(LinalgTilingOptions options,
                   LinalgTransformationFilter filter, MLIRContext *context,
                   mlir::PatternBenefit benefit = 1)
      : mlir::OpInterfaceRewritePattern<LinalgOp>(context, benefit),
        filter(filter),
        options(options) {}

  LogicalResult matchAndRewrite(LinalgOp linalg_op,
                                PatternRewriter &rewriter) const override {
    // Check if it is cwise on tensors.
    if (failed(filter.checkAndNotify(rewriter, linalg_op))) return failure();

    auto tiled_linalg_op = tileLinalgOp(rewriter, linalg_op, options);
    if (failed(tiled_linalg_op) || tiled_linalg_op.getValue().loops.empty())
      return failure();

    TiledLoopOp tiled_loop =
        mlir::dyn_cast<TiledLoopOp>(*tiled_linalg_op.getValue().loops.front());
    if (!tiled_loop) return failure();

    tiled_loop->walk([&](LinalgOp tiledOp) {
      filter.replaceLinalgTransformationFilter(rewriter, tiledOp);
    });

    rewriter.replaceOp(linalg_op, tiled_loop->getResults());
    return success();
  }

 private:
  LinalgTransformationFilter filter;
  LinalgTilingOptions options;
};

// Return true if the generic has only parallel iterations. This disallows
// windowed and reduction iteration.
bool isNonTiledCwise(Operation *op) {
  if (op->getParentOfType<TiledLoopOp>()) return false;
  auto linalg_op = mlir::dyn_cast<GenericOp>(op);
  if (!linalg_op || !linalg_op.hasTensorSemantics()) return false;
  return llvm::all_of(linalg_op.iterator_types(),
                      [](auto type) { return mlir::isParallelIterator(type); });
}

struct CodegenForCWisePass : public CodegenCWiseBase<CodegenForCWisePass> {
  CodegenForCWisePass() = default;
  explicit CodegenForCWisePass(int64_t tile_size) {
    cwise_tile_size = tile_size;
  }
  void runOnFunction() override {
    constexpr llvm::StringRef kTiledId = "tiled";
    auto func = getFunction();

    LinalgTilingOptions tiling_options;
    // Tile the innermost dimension by 8 for vectorization and scalarize the
    // other dimensions.
    tiling_options.setTileSizeComputationFunction([&](OpBuilder b,
                                                      Operation *op) {
      auto num_loops = llvm::cast<LinalgOp>(op).getNumLoops();
      SmallVector<Value> tiles(num_loops,
                               b.create<ConstantIndexOp>(op->getLoc(), 1));
      if (!tiles.empty())
        tiles.back() = b.create<ConstantIndexOp>(op->getLoc(), cwise_tile_size);
      return tiles;
    });
    tiling_options.setLoopType(mlir::linalg::LinalgTilingLoopType::TiledLoops);

    auto filter = LinalgTransformationFilter(
                      llvm::ArrayRef<mlir::Identifier>{},
                      {mlir::Identifier::get(kTiledId, func.getContext())})
                      .addFilter([](Operation *op) {
                        return success(isNonTiledCwise(op));
                      });

    mlir::OwningRewritePatternList patterns(func.getContext());
    patterns.insert<TileCWisePattern>(tiling_options, filter,
                                      patterns.getContext());
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));

    // Ensure we drop the marker in the end.
    func.walk([](LinalgOp op) {
      op->removeAttr(mlir::linalg::LinalgTransforms::kLinalgTransformMarker);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::FunctionPass> CreateCodegenStrategyForCWisePass() {
  return std::make_unique<CodegenForCWisePass>();
}

std::unique_ptr<mlir::FunctionPass> CreateCodegenStrategyForCWisePass(
    int64_t cwise_tile_size) {
  return std::make_unique<CodegenForCWisePass>(cwise_tile_size);
}

}  // namespace tensorflow

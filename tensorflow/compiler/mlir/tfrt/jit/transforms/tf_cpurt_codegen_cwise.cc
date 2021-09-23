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

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h.inc"

using mlir::ConstantIndexOp;
using mlir::failure;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::PatternRewriter;
using mlir::SmallVector;
using mlir::success;
using mlir::Value;
using mlir::linalg::GenericOp;
using mlir::linalg::LinalgOp;
using mlir::linalg::LinalgTilingOptions;
using mlir::linalg::LinalgTransformationFilter;
using mlir::linalg::TiledLoopOp;

struct TileCWisePattern : public mlir::OpInterfaceRewritePattern<LinalgOp> {
  /// MatchAnyOpTag-based constructor with a mandatory `filter`.
  TileCWisePattern(LinalgTilingOptions options,
                   LinalgTransformationFilter filter,
                   mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : mlir::OpInterfaceRewritePattern<LinalgOp>(context, benefit),
        filter(filter),
        options(options) {}

  LogicalResult matchAndRewrite(LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    // Check if it is cwise on tensors.
    if (failed(filter.checkAndNotify(rewriter, linalgOp))) return failure();

    auto tiledLinalgOp = tileLinalgOp(rewriter, linalgOp, options);
    if (!tiledLinalgOp) return failure();

    TiledLoopOp tiledLoopOp =
        mlir::dyn_cast<TiledLoopOp>(*tiledLinalgOp.getValue().loops.front());
    if (!tiledLoopOp) return failure();

    tiledLoopOp->walk([&](LinalgOp tiledOp) {
      filter.replaceLinalgTransformationFilter(rewriter, tiledOp);
    });

    rewriter.replaceOp(linalgOp, tiledLoopOp->getResults());
    return success();
  }

 private:
  LinalgTransformationFilter filter;
  LinalgTilingOptions options;
};

// Return true if the generic has only parallel iterations. This disallows
// windowed and reduction iteration.
bool isaCwise(mlir::Operation *op) {
  auto linalgOp = mlir::dyn_cast<mlir::linalg::GenericOp>(op);
  if (!linalgOp || !linalgOp.hasTensorSemantics()) return false;
  return llvm::all_of(linalgOp.iterator_types(),
                      [](auto type) { return mlir::isParallelIterator(type); });
}

struct CodegenForCWisePass : public CodegenCWiseBase<CodegenForCWisePass> {
  void runOnFunction() override {
    constexpr llvm::StringRef kTiledId = "tiled";
    auto func = getFunction();

    mlir::linalg::LinalgTilingOptions tiling_options;
    // Tile the innermost dimension by 8 for vectorization and scalarize the
    // other dimensions.
    tiling_options.setTileSizeComputationFunction(
        [](OpBuilder b, Operation *op) {
          auto num_loops = llvm::cast<mlir::linalg::LinalgOp>(op).getNumLoops();
          SmallVector<Value> tiles(num_loops,
                                   b.create<ConstantIndexOp>(op->getLoc(), 1));
          tiles.back() = b.create<ConstantIndexOp>(op->getLoc(), 8);
          return tiles;
        });
    tiling_options.setLoopType(mlir::linalg::LinalgTilingLoopType::TiledLoops);

    auto filter = mlir::linalg::LinalgTransformationFilter(
                      llvm::ArrayRef<mlir::Identifier>{},
                      {mlir::Identifier::get(kTiledId, func.getContext())})
                      .addFilter([](mlir::Operation *op) {
                        return mlir::success(isaCwise(op));
                      });

    mlir::OwningRewritePatternList patterns(func.getContext());
    patterns.insert<TileCWisePattern>(tiling_options, filter,
                                      patterns.getContext());
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));

    // Ensure we drop the marker in the end.
    func.walk([](mlir::linalg::LinalgOp op) {
      op->removeAttr(mlir::linalg::LinalgTransforms::kLinalgTransformMarker);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::FunctionPass> CreateCodegenStrategyForCWisePass() {
  return std::make_unique<CodegenForCWisePass>();
}

}  // namespace tensorflow

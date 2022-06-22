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

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/pass_detail.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

using linalg::GenericOp;
using linalg::LinalgTransformationFilter;

// Tiles a GenericOp that models a 2D row or column reduction.
struct LinalgGenericTilingPattern : public OpRewritePattern<GenericOp> {
  LinalgGenericTilingPattern(ArrayRef<int64_t> tileSizes,
                             const LinalgTransformationFilter &filter,
                             MLIRContext *context,
                             mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit),
        filter(filter),
        tileSizes(tileSizes.begin(), tileSizes.end()) {}

  LogicalResult matchAndRewrite(GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, linalgOp))) return failure();

    if (linalgOp.getNumLoops() != tileSizes.size()) return failure();

    FailureOr<Operation *> tilingResult;
    if (llvm::all_of(tileSizes, [](int64_t size) { return size == 1; })) {
      tilingResult = tileToPoints(rewriter, linalgOp);
    } else {
      tilingResult = tileToSlices(rewriter, linalgOp, tileSizes);
    }

    if (failed(tilingResult)) return failure();

    rewriter.replaceOp(linalgOp, (*tilingResult)->getResults());
    return success();
  }

 private:
  LinalgTransformationFilter filter;
  SmallVector<int64_t> tileSizes;
};

bool isCwiseGeneric(Operation *op) {
  auto linalgOp = mlir::dyn_cast<GenericOp>(op);
  if (!linalgOp) return false;
  if (!linalgOp.hasTensorSemantics()) return false;

  if (llvm::any_of(linalgOp.iterator_types(),
                   [](auto type) { return !mlir::isParallelIterator(type); }))
    return false;

  return llvm::all_of(linalgOp.indexing_maps(), [](Attribute attr) {
    return attr.cast<AffineMapAttr>().isIdentity();
  });
}

struct TilingPass : public TilingPassBase<TilingPass> {
  TilingPass() = default;
  explicit TilingPass(llvm::ArrayRef<int64_t> sizes) { tileSizes = sizes; }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<GmlStDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = func.getContext();

    auto filter = LinalgTransformationFilter(
                      llvm::None, {mlir::StringAttr::get(context, "tiled")})
                      .addFilter([](Operation *op) {
                        return success(isCwiseGeneric(op));
                      });

    RewritePatternSet patterns(context);
    patterns.add<LinalgGenericTilingPattern>(tileSizes, filter,
                                             patterns.getContext());

    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns));

    // Ensure we drop the marker in the end.
    func.walk([](GenericOp op) {
      op->removeAttr(mlir::linalg::LinalgTransforms::kLinalgTransformMarker);
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createTilingPass(
    ArrayRef<int64_t> tileSizes) {
  return std::make_unique<TilingPass>(tileSizes);
}

}  // namespace gml_st
}  // namespace mlir

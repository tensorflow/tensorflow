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

#include <memory>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_TILEBYONEPASS
#include "gml_st/transforms/passes.h.inc"

static constexpr llvm::StringRef kTileByOneLabel = "__tile_by_one_label__";

SmallVector<Value> unitTileSizeComputationFunction(OpBuilder &b,
                                                   Operation *op) {
  // Determine rank.
  auto iface = cast<TilingInterface>(op);
  int64_t rank = iface.getLoopIteratorTypes().size();

  // Build unit tile sizes.
  auto one = b.create<arith::ConstantIndexOp>(op->getLoc(), 1);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(
      &op->getParentOfType<func::FuncOp>().getBody().front());
  SmallVector<Value> tileSize(rank, one);

  return tileSize;
}

template <typename OpTy>
struct TileByOnePattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Skip ops that are already tiled.
    if (hasLabel(op, kTileByOneLabel)) return failure();

    // Skip if iteration domain is statically known to be of size 1.
    auto iface = llvm::cast<TilingInterface>(op.getOperation());
    // TODO(frgossen): Avoid creating the IR for these ranges. Instead, the
    // tiling interface should allow to access statically known iteration
    // domains.
    SmallVector<Range> iterationDomain = iface.getIterationDomain(rewriter);
    auto isRangeSizeOne = [](Range range) {
      if (!range.size.is<Attribute>()) return false;
      auto intAttr = range.size.get<Attribute>().dyn_cast<IntegerAttr>();
      if (!intAttr) return false;
      return intAttr.getInt() == 1;
    };
    if (llvm::all_of(iterationDomain, isRangeSizeOne)) return failure();

    // Tile.
    scf::SCFTilingOptions opts;
    opts.setTileSizeComputationFunction(unitTileSizeComputationFunction);
    FailureOr<scf::SCFTilingResult> tilingResult =
        tileUsingSCFForOp(rewriter, iface, opts);
    if (failed(tilingResult))
      return rewriter.notifyMatchFailure(op, "tiling to scf.for failed");

    // Mark resulting tiled ops.
    for (Operation *tiled : tilingResult->tiledOps) {
      setLabel(tiled, kTileByOneLabel);
    }

    rewriter.replaceOp(op, tilingResult->replacements);
    return success();
  }
};

struct TileByOnePass : public impl::TileByOnePassBase<TileByOnePass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<GmlStDialect, arith::ArithDialect, tensor::TensorDialect,
                    scf::SCFDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    // Populate patterns.
    RewritePatternSet patterns(ctx);
    // clang-format off
    patterns.add<
        TileByOnePattern<thlo::ConcatenateOp>,
        TileByOnePattern<thlo::GatherOp>,
        TileByOnePattern<thlo::ReverseOp>,
        TileByOnePattern<thlo::ScatterOp>,
        TileByOnePattern<thlo::SortOp>,
        TileByOnePattern<linalg::MapOp>>(ctx);
    // clang-format on

    // Apply patterns.
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Clean up by removing temporary attributes.
    f->walk([](Operation *op) { removeLabel(op, kTileByOneLabel); });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createTileByOnePass() {
  return std::make_unique<TileByOnePass>();
}

}  // namespace gml_st
}  // namespace mlir

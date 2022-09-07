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

#include <iterator>
#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_UNCOLLAPSEMATERIALIZEOPSPASS
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h.inc"

// Uncollapse materialize operations with nested tile chains t1, t2, ..., tn. A
// materialize op of the form ...
//   `materialize(t1(t2(...(tn(sn)))), arg)`
// ... is expanded into ...
//   `materialize(t1(s1), materialize(t2(...(tn(sn))), arg))`.
struct UncollapseMaterializePattern : public OpRewritePattern<MaterializeOp> {
  using OpRewritePattern<MaterializeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MaterializeOp op,
                                PatternRewriter &rewriter) const override {
    // Find head of the tile chain.
    auto tileDef = op.set().getDefiningOp<TileOp>();
    if (!tileDef) return failure();

    // Find tail of the tile chain.
    auto superTile = tileDef.superset();
    auto superTileDef = superTile.getDefiningOp<TileOp>();
    if (!superTileDef) return failure();

    // Create independent head tile and tail tile chain.
    auto loc = op.getLoc();
    auto newTileSpace = rewriter.create<SpaceOp>(loc, superTileDef.getType(),
                                                 superTileDef.sizes(),
                                                 superTileDef.static_sizes());
    auto newTile = rewriter.create<TileOp>(
        loc, newTileSpace, tileDef.offsets(), tileDef.sizes(),
        tileDef.strides(), tileDef.static_offsets(), tileDef.static_sizes(),
        tileDef.static_strides());
    auto newInnerMaterialize =
        rewriter.create<MaterializeOp>(loc, op.source(), superTile);

    // Create expanded materialize op.
    rewriter.replaceOpWithNewOp<MaterializeOp>(op, newInnerMaterialize,
                                               newTile);
    return success();
  }
};

struct UncollapseMaterializeOpsPass
    : public impl::UncollapseMaterializeOpsPassBase<
          UncollapseMaterializeOpsPass> {
  void runOnOperation() final {
    MLIRContext *ctx = &getContext();

    // Populate collapse or uncollapse pattern.
    RewritePatternSet patterns(ctx);
    patterns.add<UncollapseMaterializePattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createUncollapseMaterializeOpsPass() {
  return std::make_unique<UncollapseMaterializeOpsPass>();
}

}  // namespace gml_st
}  // namespace mlir

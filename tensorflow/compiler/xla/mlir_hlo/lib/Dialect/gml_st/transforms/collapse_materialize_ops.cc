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
#include "mlir-hlo/Dialect/gml_st/transforms/rewriters.h"
#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {

// Collapse materialize operations with nested tile chains t1, t2, ..., tn, and
// u1, u2, ..., un. A materialize op of the form ...
//   `materialize(t1(t2(...(tn(sn)))), materialize(u1(u2(...(un(sn')))), arg))`
// ... is collapsed as ...
//   `materialize(t1(t2(...(tn(u1(u2(...(un(sn'))))))), arg)`.
FailureOr<MaterializeOp> collapseMaterializeOp(OpBuilder &b, MaterializeOp op) {
  // Find inner materialize op.
  auto innerMaterialize = op.source().getDefiningOp<MaterializeOp>();
  if (!innerMaterialize) return failure();

  // Find outer tile chain to replace its root space op.
  llvm::SmallVector<TileOp> tileChain;
  Operation *tileDef = op.set().getDefiningOp();
  while (tileDef && !llvm::isa<SpaceOp>(tileDef)) {
    auto tileOp = llvm::dyn_cast<TileOp>(tileDef);
    if (!tileOp) return failure();
    tileChain.push_back(tileOp);
    tileDef = tileOp.superset().getDefiningOp();
  }

  // Create new tile chain, starting with its tail.
  auto loc = op.getLoc();
  Value newTileChain = innerMaterialize.set();
  while (!tileChain.empty()) {
    TileOp tileOp = tileChain.pop_back_val();
    newTileChain =
        b.create<TileOp>(loc, newTileChain, tileOp.offsets(), tileOp.sizes(),
                         tileOp.strides(), tileOp.static_offsets(),
                         tileOp.static_sizes(), tileOp.static_strides());
  }

  // Create collapsed materialize op.
  return b.create<MaterializeOp>(loc, innerMaterialize.source(), newTileChain);
}

// Uncollapse materialize operations with nested tile chains t1, t2, ..., tn. A
// materialize op of the form ...
//   `materialize(t1(t2(...(tn(sn)))), arg)`
// ... is expanded into ...
//   `materialize(t1(s1), materialize(t2(...(tn(sn))), arg))`.
FailureOr<MaterializeOp> uncollapseMaterializeOp(OpBuilder &b,
                                                 MaterializeOp op) {
  // Find head of the tile chain.
  auto tileDef = op.set().getDefiningOp<TileOp>();
  if (!tileDef) return failure();

  // Find tail of the tile chain.
  auto superTile = tileDef.superset();
  auto superTileDef = superTile.getDefiningOp<TileOp>();
  if (!superTileDef) return failure();

  // Create independent head tile and tail tile chain.
  Location loc = op.getLoc();
  auto newTileSpace =
      b.create<SpaceOp>(loc, superTileDef.getType(), superTileDef.sizes(),
                        superTileDef.static_sizes());
  auto newTile =
      b.create<TileOp>(loc, newTileSpace, tileDef.offsets(), tileDef.sizes(),
                       tileDef.strides(), tileDef.static_offsets(),
                       tileDef.static_sizes(), tileDef.static_strides());
  auto newInnerMaterialize =
      b.create<MaterializeOp>(loc, op.source(), superTile);

  // Create expanded materialize op.
  return b.create<MaterializeOp>(loc, newInnerMaterialize, newTile);
}

namespace {

#define GEN_PASS_DEF_COLLAPSEMATERIALIZEOPSPASS
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h.inc"

struct CollapseMaterializeOpPattern : public OpRewritePattern<MaterializeOp> {
  using OpRewritePattern<MaterializeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaterializeOp op,
                                PatternRewriter &rewriter) const override {
    auto collapsed = collapseMaterializeOp(rewriter, op);
    if (failed(collapsed)) return failure();
    rewriter.replaceOp(op, {*collapsed});
    return success();
  }
};

struct UncollapseMaterializeOpPattern : public OpRewritePattern<MaterializeOp> {
  using OpRewritePattern<MaterializeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MaterializeOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<MaterializeOp> uncollapsed =
        uncollapseMaterializeOp(rewriter, op);
    if (failed(uncollapsed)) return failure();
    rewriter.replaceOp(op, {*uncollapsed});
    return success();
  }
};

struct CollapseMaterializeOpsPass
    : public impl::CollapseMaterializeOpsPassBase<CollapseMaterializeOpsPass> {
  explicit CollapseMaterializeOpsPass(bool reverse)
      : CollapseMaterializeOpsPassBase() {
    reverse_ = reverse;
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateCollapseMaterializeOpsPatterns(ctx, reverse_, &patterns);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void populateCollapseMaterializeOpsPatterns(MLIRContext *ctx, bool reverse,
                                            RewritePatternSet *patterns) {
  if (!reverse) {
    patterns->add<CollapseMaterializeOpPattern>(ctx);
  } else {
    patterns->add<UncollapseMaterializeOpPattern>(ctx);
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> createCollapseMaterializeOpsPass(
    bool reverse) {
  return std::make_unique<CollapseMaterializeOpsPass>(reverse);
}

}  // namespace gml_st
}  // namespace mlir

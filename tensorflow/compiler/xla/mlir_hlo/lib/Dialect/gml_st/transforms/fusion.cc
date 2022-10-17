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

#include "mlir-hlo/Dialect/gml_st/transforms/fusion.h"

#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir-hlo/Dialect/gml_st/transforms/rewriters.h"
#include "mlir-hlo/Dialect/gml_st/transforms/tiling_interface.h"
#include "mlir-hlo/Dialect/gml_st/transforms/tiling_interface_impl.h"
#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"
#include "mlir-hlo/Dialect/thlo/IR/thlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_FUSIONPASS
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h.inc"

// TODO(frgossen): Move this to the shape reification pass.
struct DimOpFissionPattern : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp extract,
                                PatternRewriter& rewriter) const override {
    auto shapeDef = llvm::dyn_cast_or_null<shape::ShapeOfOp>(
        extract.getTensor().getDefiningOp());
    if (!shapeDef || extract.getIndices().size() != 1) return failure();
    rewriter.replaceOpWithNewOp<tensor::DimOp>(extract, shapeDef.getArg(),
                                               extract.getIndices().front());
    return success();
  }
};

// TODO(frgossen): Implement this through the shape reification interface and
// move this pattern to the shape reification pass.
struct DimOpReificationPattern : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter& rewriter) const override {
    Operation* def = op.getSource().getDefiningOp();
    if (!def) return failure();

    // Case MaterializeOp.
    if (auto materializeOp = llvm::dyn_cast<MaterializeOp>(def)) {
      assert(materializeOp->getNumResults() == 1 && "assume single result");
      Value set = materializeOp.getSet();
      if (!set.getType().isa<TileType>()) return failure();
      rewriter.replaceOpWithNewOp<gml_st::SizeOp>(op, set, op.getIndex());
      return success();
    }

    // Case GenericOp.
    if (auto genericOp = llvm::dyn_cast<linalg::GenericOp>(def)) {
      if (genericOp.getNumResults() != 1 || !genericOp.hasTensorSemantics()) {
        return failure();
      }
      Value outputOperand = genericOp.getOutputOperand(0)->get();
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, outputOperand,
                                                 op.getIndex());
      return success();
    }

    // Case EmptyOp.
    if (auto emptyTensorOp = llvm::dyn_cast<tensor::EmptyOp>(def)) {
      if (auto indexConstantOp = llvm::dyn_cast_or_null<arith::ConstantOp>(
              op.getIndex().getDefiningOp())) {
        int64_t idx =
            indexConstantOp.getValue().dyn_cast<IntegerAttr>().getInt();
        OpFoldResult dim = emptyTensorOp.getMixedSizes()[idx];
        Value dimValue;
        if (dim.is<Value>()) {
          dimValue = dim.get<Value>();
        } else {
          assert(dim.is<Attribute>() && "expected Value or Attribute");
          int64_t dimInt = dim.get<Attribute>().cast<IntegerAttr>().getInt();
          dimValue =
              rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dimInt);
        }
        assert(dimValue);
        rewriter.replaceOp(op, ValueRange{dimValue});
        return success();
      }
    }

    // Case ConcatenateOp.
    if (auto concat = llvm::dyn_cast<thlo::ConcatenateOp>(def)) {
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, concat.getInit(),
                                                 op.getIndex());
      return success();
    }

    // Case DynamicBroadcastInDimOp.
    if (auto bcast = llvm::dyn_cast<thlo::DynamicBroadcastInDimOp>(def)) {
      rewriter.replaceOpWithNewOp<tensor::DimOp>(op, bcast.getInit(),
                                                 op.getIndex());
      return success();
    }

    return failure();
  }
};

// Helper function to extract indices from the subset-based representation in.
// This is to adapt to the tiling interface.
void getOrMaterializeMixedOffsetsAndSizes(OpBuilder& b, Location loc,
                                          Value tile,
                                          SmallVector<OpFoldResult>& offsets,
                                          SmallVector<OpFoldResult>& sizes) {
  // If the tile is not nested, we can extract the indices from the op.
  if (auto tileOp = tile.getDefiningOp<TileOp>()) {
    if (tileOp.getSuperset().getDefiningOp<SpaceOp>()) {
      offsets = tileOp.getMixedOffsets();
      sizes = tileOp.getMixedSizes();
      return;
    }
  }

  // Otherwise, we have to materialize ops to extract the needed offstes and
  // sizes.
  int64_t rank = tile.getType().cast<TileType>().getRank();
  offsets.clear();
  offsets.reserve(rank);
  sizes.clear();
  sizes.reserve(rank);
  for (int64_t i = 0; i < rank; i++) {
    auto iCst = b.create<arith::ConstantIndexOp>(loc, i);
    Value offset = b.create<OffsetOp>(loc, tile, iCst);
    offsets.push_back(offset);
    Value size = b.create<SizeOp>(loc, tile, iCst);
    sizes.push_back(size);
  }
}

class FusionPattern : public OpRewritePattern<MaterializeOp> {
 public:
  FusionPattern(MLIRContext* context,
                function_ref<LogicalResult(Operation*)> filterFn,
                mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<MaterializeOp>(context, benefit), filterFn(filterFn) {}

  LogicalResult matchAndRewrite(MaterializeOp materializeOp,
                                PatternRewriter& rewriter) const override {
    assert(filterFn && "expect filter function");
    if (failed(filterFn(materializeOp))) return failure();

    Location loc = materializeOp.getLoc();
    FailureOr<Value> fused = createFusedOp(rewriter, loc, materializeOp);
    if (failed(fused)) return failure();

    // Insert cast if needed.
    if (fused->getType() != materializeOp.getType()) {
      fused =
          rewriter.create<tensor::CastOp>(loc, materializeOp.getType(), *fused)
              .getResult();
    }

    rewriter.replaceOp(materializeOp, *fused);
    return success();
  }

 private:
  function_ref<LogicalResult(Operation*)> filterFn;
};

struct FusionPass : public impl::FusionPassBase<FusionPass> {
  FusionPass(StringRef producer, StringRef consumer) {
    this->producerLabel = producer.str();
    this->consumerLabel = consumer.str();
  }

  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<GmlStDialect, tensor::TensorDialect>();
    registerGmlStTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() final {
    MLIRContext* ctx = &getContext();

    auto filterFn = [&](Operation* op) {
      auto materializeOp = cast<MaterializeOp>(op);
      Operation* producerOp = materializeOp.getSource().getDefiningOp();
      if (!producerOp || (!producerLabel.empty() &&
                          !hasMatchingLabel(producerOp, producerLabel))) {
        return failure();
      }

      Operation* consumerOp = nullptr;
      if (!consumerLabel.empty()) {
        for (Operation* user : materializeOp.getResult().getUsers()) {
          if (hasMatchingLabel(user, consumerLabel)) {
            consumerOp = user;
            break;
          }
        }
        return success(consumerOp != nullptr);
      }

      return success();
    };

    // Populate patterns.
    RewritePatternSet patterns(ctx);
    populateFusionPatterns(ctx, filterFn, &patterns);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

FailureOr<Value> createFusedOp(OpBuilder& b, Location loc,
                               MaterializeOp materializeOp) {
  auto tileableOp = materializeOp.getSource().getDefiningOp<TilingInterface>();
  if (!tileableOp) return failure();

  Value tile = materializeOp.getSet();
  if (!tile.getType().isa<TileType>()) return failure();

  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  getOrMaterializeMixedOffsetsAndSizes(b, loc, tile, offsets, sizes);

  // Tile the producer.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(materializeOp);
  FailureOr<Value> tiledProducer =
      tileableOp.generateResultTileValue(b, /*resultNumber=*/0, offsets, sizes);
  if (failed(tiledProducer)) return failure();

  return tiledProducer;
}

void populateFusionPatterns(MLIRContext* ctx,
                            function_ref<LogicalResult(Operation*)> filterFn,
                            RewritePatternSet* patterns) {
  patterns->insert<FusionPattern>(ctx, filterFn);
  // clang-format off
  patterns->insert<
      DimOpFissionPattern,
      DimOpReificationPattern>(ctx);
  // clang-format on
}

std::unique_ptr<OperationPass<func::FuncOp>> createFusionPass(
    StringRef producer, StringRef consumer) {
  return std::make_unique<FusionPass>(producer, consumer);
}

}  // namespace gml_st
}  // namespace mlir

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

#include <algorithm>
#include <array>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMPACKFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

FailureOr<Operation *> tileUsingSCFForAndReplace(
    PatternRewriter &rewriter, Operation *op,
    const scf::SCFTilingOptions &tilingOptions) {
  if (hasLabel(op, kTransformedLabel)) return failure();

  auto tilingResult = scf::tileUsingSCFForOp(rewriter, op, tilingOptions);
  if (failed(tilingResult) || tilingResult->loops.empty()) return failure();

  for (Operation *tiledOp : tilingResult->tiledOps)
    setLabel(tiledOp, kTransformedLabel);
  rewriter.replaceOp(op, tilingResult->replacements);
  return tilingResult->tiledOps.front();
}

LogicalResult tilePackOp(tensor::PackOp packOp, PatternRewriter &rewriter) {
  // Tile tensor.pack ops.
  auto packTilingOptions =
      scf::SCFTilingOptions().setTileSizeComputationFunction(
          [&](OpBuilder b, Operation *op) {
            auto numLoops =
                cast<mlir::TilingInterface>(op).getLoopIteratorTypes().size();
            SmallVector<Value> tiles(
                numLoops, b.create<arith::ConstantIndexOp>(op->getLoc(), 1));
            return tiles;
          });

  return tileUsingSCFForAndReplace(rewriter, packOp, packTilingOptions);
}

LogicalResult tileUnpackOp(tensor::UnPackOp unpackOp,
                           PatternRewriter &rewriter) {
  // Tile tensor.unpack op.
  auto unpackTilingOptions =
      scf::SCFTilingOptions().setTileSizeComputationFunction(
          [](OpBuilder &builder, Operation *op) {
            Location loc = op->getLoc();
            auto unpackOp = cast<tensor::UnPackOp>(op);
            auto numLoops = unpackOp.getDestRank();
            auto dimAndTileMapping = unpackOp.getDimAndTileMapping();
            SmallVector<Value> tileSizes;
            for (size_t i = 0; i < numLoops; ++i) {
              if (dimAndTileMapping.count(i)) {
                tileSizes.push_back(getValueOrCreateConstantIndexOp(
                    builder, loc, dimAndTileMapping[i]));
              } else {
                tileSizes.push_back(
                    builder.create<memref::DimOp>(loc, unpackOp.getDest(), i));
              }
            }
            return tileSizes;
          });

  return tileUsingSCFForAndReplace(rewriter, unpackOp, unpackTilingOptions);
}

struct TransformPackForCpuPass
    : public impl::TransformPackForCpuPassBase<TransformPackForCpuPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry
        .insert<arith::ArithDialect, scf::SCFDialect, tensor::TensorDialect>();
    tensor::registerTilingInterfaceExternalModels(registry);
    tensor::registerInferTypeOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    {
      RewritePatternSet patterns(ctx);
      patterns.add(tilePackOp);
      patterns.add(tileUnpackOp);
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
        return signalPassFailure();
    }

    // Expanding pack and unpack ops to other primitive tensor/linalg ops and
    // canonicalize tiled ops.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<linalg::GeneralizeOuterUnitDimsPackOpPattern,
                   linalg::GeneralizeOuterUnitDimsUnPackOpPattern>(ctx);
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
        return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformPackForCpuPass() {
  return std::make_unique<TransformPackForCpuPass>();
}

}  // namespace mlir::gml_st

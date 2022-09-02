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

#include "mlir-hlo/Dialect/gml_st/transforms/tiling_interface_impl.h"

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/tiling_interface.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"

namespace mlir {
namespace gml_st {
namespace {

struct ExternalLinalgGenericTilingInterface
    : public TilingInterface::ExternalModel<
          ExternalLinalgGenericTilingInterface, linalg::GenericOp> {
  /// Return the destination operands.
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &) const {
    return cast<linalg::DestinationStyleOpInterface>(op).getOutputOperands();
  }

  /// Return the loop iterator type.
  SmallVector<StringRef> getLoopIteratorTypes(Operation *op) const {
    linalg::GenericOp genericOp = cast<linalg::GenericOp>(op);
    return llvm::to_vector(
        llvm::map_range(genericOp.iterator_types(), [](Attribute strAttr) {
          return strAttr.cast<StringAttr>().getValue();
        }));
  }

  /// Return the iteration domain range.
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);
    Location loc = op->getLoc();
    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
    SmallVector<OpFoldResult> allShapesSizes =
        linalgOp.createFlatListOfOperandDims(b, loc);
    AffineMap map = linalgOp.getShapesToLoopsMap();

    IRRewriter rewriter(b);
    return llvm::to_vector(
        llvm::map_range(map.getResults(), [&](AffineExpr loopExpr) {
          OpFoldResult ofr = makeComposedFoldedAffineApply(
              rewriter, loc, loopExpr, allShapesSizes);
          return Range{b.getIndexAttr(0), ofr, b.getIndexAttr(1)};
        }));
  }

  // Instantiate the tiled implementation of the operation.
  TilingInterface getTiledImplementation(Operation *op, OpBuilder &b,
                                         ValueRange /*dest*/,
                                         ArrayRef<OpFoldResult> offsets,
                                         ArrayRef<OpFoldResult> sizes,
                                         bool /*tileDestOperands*/) const {
    Location loc = op->getLoc();
    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
    SmallVector<Value> valuesToTile = linalgOp.getInputAndOutputOperands();
    SmallVector<Optional<linalg::SliceParameters>> allSliceParams =
        linalg::computeAllSliceParameters(b, loc, linalgOp, valuesToTile,
                                          offsets, sizes, {}, true);

    SmallVector<Value> tiledOperands;
    for (auto item : llvm::zip(valuesToTile, allSliceParams)) {
      Value valueToTile = std::get<0>(item);
      auto valueToTileTy = valueToTile.getType().cast<RankedTensorType>();
      const Optional<linalg::SliceParameters> &sliceParams = std::get<1>(item);

      SmallVector<Value> dynamicSizes =
          tensor::createDynamicDimValues(b, loc, valueToTile);
      auto staticSizes = b.getI64ArrayAttr(valueToTileTy.getShape());
      Value set = b.create<SpaceOp>(loc, dynamicSizes, staticSizes);
      if (sliceParams.has_value()) {
        set = b.create<TileOp>(loc, set, sliceParams->offsets,
                               sliceParams->sizes, sliceParams->strides);
      }
      Value materializedTile = b.create<MaterializeOp>(loc, valueToTile, set);
      tiledOperands.push_back(materializedTile);
    }

    SmallVector<Type> resultTensorTypes = llvm::to_vector(llvm::map_range(
        linalgOp.getOutputTensorOperands(), [&](OpOperand *opOperand) {
          return tiledOperands[opOperand->getOperandNumber()].getType();
        }));

    Operation *tiledOp =
        linalgOp.clone(b, loc, resultTensorTypes, tiledOperands);
    offsetIndices(b, cast<linalg::LinalgOp>(tiledOp), offsets);

    return {tiledOp};
  }

  FailureOr<Value> generateResultTileValue(Operation *op, OpBuilder &b,
                                           unsigned resultNumber,
                                           ValueRange dest,
                                           ArrayRef<OpFoldResult> offsets,
                                           ArrayRef<OpFoldResult> sizes,
                                           bool tileDestOperands) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);

    // Check that the indexing map used for the output is a projected
    // permutation. This could be relaxed with a more general approach that can
    // map the offsets and sizes from the result to iteration space tiles
    // (filling in full extent for dimensions not used to access the result).
    AffineMap indexingMap =
        linalgOp.getTiedIndexingMapForResult(op->getResult(resultNumber));
    if (!indexingMap.isProjectedPermutation()) {
      return op->emitOpError(
          "unhandled tiled implementation generation when result is not "
          "accessed using a permuted projection");
    }

    auto numLoops = linalgOp.getNumLoops();
    auto tilingInterfaceOp = cast<TilingInterface>(op);
    SmallVector<OpFoldResult> iterationTileOffsets(numLoops),
        iterationTileSizes(numLoops);
    if (!indexingMap.isPermutation()) {
      SmallVector<Range> iterationDomain =
          tilingInterfaceOp.getIterationDomain(b);
      for (const auto &range : llvm::enumerate(iterationDomain)) {
        iterationTileOffsets[range.index()] = range.value().offset;
        iterationTileSizes[range.index()] = range.value().size;
      }
    }
    for (const auto &resultExpr : llvm::enumerate(indexingMap.getResults())) {
      unsigned dimPosition =
          resultExpr.value().cast<AffineDimExpr>().getPosition();
      iterationTileOffsets[dimPosition] = offsets[resultExpr.index()];
      iterationTileSizes[dimPosition] = sizes[resultExpr.index()];
    }

    TilingInterface tiledOp = tilingInterfaceOp.getTiledImplementation(
        b, dest, iterationTileOffsets, iterationTileSizes, tileDestOperands);

    return tiledOp->getResult(resultNumber);
  }
};

}  // namespace

void registerGmlStTilingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *) {
    linalg::GenericOp::attachInterface<ExternalLinalgGenericTilingInterface>(
        *ctx);
  });
}

}  // namespace gml_st
}  // namespace mlir

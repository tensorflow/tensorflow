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

#include "gml_st/interfaces/tiling_interface_impl.h"

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/interfaces/tiling_interface.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

namespace mlir {
namespace gml_st {
namespace {

template <typename LinalgOpTy>
struct ExternalLinalgOpTilingInterface
    : public TilingInterface::ExternalModel<
          ExternalLinalgOpTilingInterface<LinalgOpTy>, LinalgOpTy> {
  /// Return the destination operands.
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &) const {
    return cast<DestinationStyleOpInterface>(op).getDpsInitOperands();
  }

  /// Return the loop iterator type.
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    return linalgOp.getIteratorTypesArray();
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
                                         ArrayRef<OpFoldResult> offsets,
                                         ArrayRef<OpFoldResult> sizes,
                                         bool useExtractSlice) const {
    Location loc = op->getLoc();
    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
    OperandRange valuesToTile = linalgOp->getOperands();
    SmallVector<Optional<linalg::SliceParameters>> allSliceParams =
        linalg::computeAllSliceParameters(b, loc, linalgOp, valuesToTile,
                                          offsets, sizes, {}, true);

    SmallVector<Value> tiledOperands;
    for (const auto &[valueToTile, sliceParams] :
         llvm::zip(valuesToTile, allSliceParams)) {
      // Use the original operand if it is not a ranked tensor. This could be a
      // scalar, e.g. for `linalg.fill`.
      auto valueToTileTy =
          valueToTile.getType().template dyn_cast<RankedTensorType>();
      if (!valueToTileTy) {
        tiledOperands.push_back(valueToTile);
        continue;
      }
      tiledOperands.push_back(
          sliceParams.has_value()
              ? materializeSlice(b, loc, valueToTile, sliceParams->offsets,
                                 sliceParams->sizes, sliceParams->strides,
                                 useExtractSlice)
              : materializeIdentitySlice(b, loc, valueToTile, useExtractSlice));
    }

    SmallVector<Type> resultTensorTypes = llvm::to_vector(llvm::map_range(
        linalgOp.getDpsInitOperands(), [&](OpOperand *opOperand) {
          return tiledOperands[opOperand->getOperandNumber()].getType();
        }));

    Operation *tiledOp = clone(b, linalgOp, resultTensorTypes, tiledOperands);
    offsetIndices(b, cast<linalg::LinalgOp>(tiledOp), offsets);

    return {tiledOp};
  }

  FailureOr<Value> generateResultTileValue(Operation *op, OpBuilder &b,
                                           unsigned resultNumber,
                                           ArrayRef<OpFoldResult> offsets,
                                           ArrayRef<OpFoldResult> sizes) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);

    // Check that the indexing map used for the output is a projected
    // permutation. This could be relaxed with a more general approach that can
    // map the offsets and sizes from the result to iteration space tiles
    // (filling in full extent for dimensions not used to access the result).
    AffineMap indexingMap =
        linalgOp.getIndexingMapMatchingResult(op->getResult(resultNumber));
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
        b, iterationTileOffsets, iterationTileSizes, /*useExtractSlice=*/false);

    return tiledOp->getResult(resultNumber);
  }
};

}  // namespace

void registerGmlStTilingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *) {
    linalg::BroadcastOp::attachInterface<
        ExternalLinalgOpTilingInterface<linalg::BroadcastOp>>(*ctx);
    linalg::FillOp::attachInterface<
        ExternalLinalgOpTilingInterface<linalg::FillOp>>(*ctx);
    linalg::GenericOp::attachInterface<
        ExternalLinalgOpTilingInterface<linalg::GenericOp>>(*ctx);
    linalg::MapOp::attachInterface<
        ExternalLinalgOpTilingInterface<linalg::MapOp>>(*ctx);
    linalg::MatmulOp::attachInterface<
        ExternalLinalgOpTilingInterface<linalg::MatmulOp>>(*ctx);
    linalg::Mmt4DOp::attachInterface<
        ExternalLinalgOpTilingInterface<linalg::Mmt4DOp>>(*ctx);
    linalg::ReduceOp::attachInterface<
        ExternalLinalgOpTilingInterface<linalg::ReduceOp>>(*ctx);
    linalg::TransposeOp::attachInterface<
        ExternalLinalgOpTilingInterface<linalg::TransposeOp>>(*ctx);
  });
}

}  // namespace gml_st
}  // namespace mlir

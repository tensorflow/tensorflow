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

#include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface_impl.h"

#include <functional>
#include <tuple>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeRange.h"

namespace mlir {
namespace gml_st {

namespace {

// Whether the operand needs the materialization of a point, given the output
// subset.
// This is the case if a) the outputput subset is a point and b) there are
// no reductions.
bool operandMaterializesToPoint(
    Value operand, const SmallVector<int64_t>& operandDimsToOutputDims,
    Value subset) {
  if (!subset.getType().isa<PointType>()) return false;

  const auto& operandShape =
      operand.getType().cast<RankedTensorType>().getShape();
  return llvm::all_of(llvm::zip(operandDimsToOutputDims, operandShape),
                      [](const auto& e) {
                        auto outputDimIndex = std::get<0>(e);
                        auto operandDimSize = std::get<1>(e);
                        // If the operand dimension maps to an output dim (the
                        // output is already known to be a point), or the
                        // operand's dimensions has a static size of 1, the
                        // operand can materialize as a point.
                        return (outputDimIndex >= 0) || (operandDimSize == 1);
                      });
}

Value buildPointOp(Location loc, OpBuilder& builder, Value operand,
                   Value subset,
                   const SmallVector<int64_t>& operandDimsToOutputDims) {
  auto operandShape = operand.getType().cast<RankedTensorType>().getShape();

  SmallVector<int64_t> staticOffsets(operandShape.size(),
                                     ShapedType::kDynamicStrideOrOffset);
  SmallVector<Value> dynamicOffsets;
  for (int i = 0; i < operandShape.size(); ++i) {
    if (int outputDim = operandDimsToOutputDims[i]; outputDim >= 0) {
      auto index = builder.create<arith::ConstantIndexOp>(loc, outputDim);
      dynamicOffsets.push_back(builder.create<OffsetOp>(loc, subset, index));
    } else {
      staticOffsets[i] = 0;
    }
  }

  SmallVector<int64_t> staticSizes(operandShape.size(), 1);
  auto staticSizesAttr = builder.getI64ArrayAttr(staticSizes);
  SpaceOp spaceOp =
      builder.create<SpaceOp>(loc, builder.getType<TileType>(staticSizes),
                              ValueRange{}, staticSizesAttr);

  auto staticOffsetsAttr = builder.getI64ArrayAttr(staticOffsets);
  return builder.create<PointOp>(loc, builder.getType<PointType>(), spaceOp,
                                 dynamicOffsets, staticOffsetsAttr);
}

Value buildTileOp(Location loc, OpBuilder& builder, Value operand, Value subset,
                  const SmallVector<int64_t>& operandDimsToOutputDims) {
  auto operandRank = operand.getType().cast<RankedTensorType>().getRank();

  SmallVector<int64_t> staticSizes(operandRank, ShapedType::kDynamicSize);
  SmallVector<int64_t> staticStrides(operandRank,
                                     ShapedType::kDynamicStrideOrOffset);
  SmallVector<int64_t> staticOffsets(operandRank,
                                     ShapedType::kDynamicStrideOrOffset);

  SmallVector<Value> dynamicSizes;
  SmallVector<Value> dynamicStrides;
  SmallVector<Value> dynamicOffsets;
  for (int i = 0; i < operandRank; ++i) {
    if (int outputDim = operandDimsToOutputDims[i]; outputDim >= 0) {
      auto index = builder.create<arith::ConstantIndexOp>(loc, outputDim);
      dynamicOffsets.push_back(builder.create<OffsetOp>(loc, subset, index));
      if (subset.getType().isa<PointType>()) {
        staticSizes[i] = 1;
        staticStrides[i] = 1;
      } else {
        dynamicStrides.push_back(builder.create<StrideOp>(loc, subset, index));
        dynamicSizes.push_back(builder.create<SizeOp>(loc, subset, index));
      }
    } else {
      staticOffsets[i] = 0;
      staticStrides[i] = 1;
      dynamicSizes.push_back(builder.create<tensor::DimOp>(loc, operand, i));
    }
  }

  auto staticSizesAttr = builder.getI64ArrayAttr(staticSizes);
  auto tileType = builder.getType<TileType>(staticSizes);
  SpaceOp spaceOp =
      builder.create<SpaceOp>(loc, tileType, dynamicSizes, staticSizesAttr);

  auto staticOffsetsAttr = builder.getI64ArrayAttr(staticOffsets);
  auto staticStridesAttr = builder.getI64ArrayAttr(staticStrides);
  return builder.create<TileOp>(loc, tileType, spaceOp, dynamicOffsets,
                                dynamicSizes, dynamicStrides, staticOffsetsAttr,
                                staticSizesAttr, staticStridesAttr);
}

// For each iterator, returns the dimension in the output affine map where it
// occurs (unless it's a reduction iterator).
Optional<SmallVector<Optional<int32_t>>> mapIteratorsToOutputs(
    AffineMap outputMap) {
  SmallVector<Optional<int32_t>> result(outputMap.getNumInputs());
  for (uint32_t i = 0; i < outputMap.getNumResults(); ++i) {
    auto dim = outputMap.getResult(i).dyn_cast<AffineDimExpr>();
    if (!dim) return {};
    if (result[dim.getPosition()]) return {};
    result[dim.getPosition()] = i;
  }
  return result;
}

struct LinalgGenericFusionInterface
    : public FusionInterface::ExternalModel<LinalgGenericFusionInterface,
                                            linalg::GenericOp> {
  // Supports linalg.generics with a single output, if all output dimensions in
  // all affine maps are affine dimensions (e.g.., (a,b,c) -> (a,b), but not
  // (a,b,c) -> (a, 0)).
  // See the test file tiling_and_fusion.mlir for examples.
  Value fuse(Operation* op, Location loc, Value subset,
             OpBuilder& builder) const {
    auto genericOp = llvm::cast<linalg::GenericOp>(op);
    if (genericOp.getNumOutputs() != 1) return {};
    Value output = genericOp.outputs().front();
    auto outputRank = output.getType().cast<RankedTensorType>().getRank();

    auto indexingMaps =
        to_vector(genericOp.indexing_maps().getAsValueRange<AffineMapAttr>());
    auto maybeIteratorsToOutputs = mapIteratorsToOutputs(indexingMaps.back());
    if (!maybeIteratorsToOutputs) return {};
    const SmallVector<Optional<int32_t>>& iteratorsToOutputs =
        *maybeIteratorsToOutputs;

    SmallVector<Value> materializedOperands;
    SmallVector<bool> operandsArePoints;
    for (const auto&& [operand, operandMap] :
         llvm::zip(genericOp.inputs(), indexingMaps)) {
      // Mapping from an operand dimension to an output dimension, or -1 if it
      // doesn't occur in the output.
      SmallVector<int64_t> operandDimsToOutputDims;

      // Whether the composition of the inverse of the operand's affine map and
      // the output's affine map is the identity function (i.e., a given output
      // coordinate maps to the same coordinate in the input).
      bool isIdentity = operandMap.getResults().size() == outputRank;
      SmallVector<bool> containsDim(outputRank);
      for (const AffineExpr& expression : operandMap.getResults()) {
        auto dim = expression.dyn_cast<AffineDimExpr>();
        if (!dim) return {};
        auto output = iteratorsToOutputs[dim.getPosition()];
        operandDimsToOutputDims.push_back(output.value_or(-1));
        if (output) containsDim[*output] = true;
        isIdentity &= output.value_or(-1) == operandDimsToOutputDims.size() - 1;
      }

      Value operandSubset;
      if (isIdentity) {
        operandSubset = subset;
        operandsArePoints.push_back(subset.getType().isa<PointType>());
      } else if (operandDimsToOutputDims.size() == outputRank &&
                 !llvm::is_contained(containsDim, false)) {
        operandSubset = builder.create<TransposeDimsOp>(
            loc, subset,
            DenseI64ArrayAttr::get(builder.getContext(),
                                   operandDimsToOutputDims));
        operandsArePoints.push_back(subset.getType().isa<PointType>());
      } else if (operandMaterializesToPoint(operand, operandDimsToOutputDims,
                                            subset)) {
        operandSubset = buildPointOp(loc, builder, operand, subset,
                                     operandDimsToOutputDims);
        operandsArePoints.push_back(true);
      } else {
        operandSubset =
            buildTileOp(loc, builder, operand, subset, operandDimsToOutputDims);
        operandsArePoints.push_back(false);
      }

      materializedOperands.push_back(
          builder.create<MaterializeOp>(loc, operand, operandSubset));
    }

    materializedOperands.push_back(
        builder.create<MaterializeOp>(loc, output, subset));
    if (!llvm::is_contained(operandsArePoints, false)) {
      // Create scalar computation by copying from the `linalg.generic`
      // body.
      BlockAndValueMapping bvm;
      Block* block = genericOp.getBody();
      assert(block->getArguments().size() == materializedOperands.size() &&
             "block argument count and sub operand count should be equal");
      for (const auto&& [arg, materialized] :
           llvm::zip(block->getArguments(), materializedOperands)) {
        bvm.map(arg, materialized);
      }
      for (auto& it : block->without_terminator()) builder.clone(it, bvm);
      auto innerResults = block->getTerminator()->getOperands();
      assert(innerResults.size() == 1 && "expect unique inner result");
      return bvm.lookup(innerResults.front());
    }

    // Materialize tiled `linalg.generic` op.
    auto outputTy = output.getType().cast<RankedTensorType>();
    RankedTensorType subResultTy;
    if (subset.getType().isa<TileType>()) {
      subResultTy =
          RankedTensorType::get(subset.getType().cast<TileType>().getShape(),
                                outputTy.getElementType());
    } else {
      // Replace the materialized operand: it must be a tensor.
      subResultTy = RankedTensorType::get(
          SmallVector<int64_t>(outputTy.getShape().size(), 1),
          outputTy.getElementType());
      materializedOperands.back() = builder.create<tensor::FromElementsOp>(
          loc, subResultTy, materializedOperands.back());
    }

    linalg::LinalgOp linalgOp = genericOp;
    auto outputOp = cast<linalg::GenericOp>(
        *linalgOp.clone(builder, loc, subResultTy, materializedOperands));

    // If any operands are points...
    if (llvm::is_contained(operandsArePoints, true)) {
      SmallVector<AffineMap> newIndexingMaps;
      for (const auto&& [isPoint, indexingMap] :
           llvm::zip(operandsArePoints, indexingMaps)) {
        if (isPoint) {
          // Replace the affine map for the input with (...) -> () - the input
          // to the linalg.generic is a scalar.
          newIndexingMaps.push_back(AffineMap::get(indexingMap.getNumInputs(),
                                                   indexingMap.getNumSymbols(),
                                                   indexingMap.getContext()));
        } else {
          newIndexingMaps.push_back(indexingMap);
        }
      }
      newIndexingMaps.push_back(indexingMaps.back());
      outputOp.setIndexingMapsAttr(
          builder.getAffineMapArrayAttr(newIndexingMaps));
    }

    Value result = outputOp.getResults().front();
    if (subset.getType().isa<PointType>()) {
      result = builder.create<tensor::ExtractOp>(
          loc, result,
          SmallVector<Value>(outputTy.getShape().size(),
                             builder.create<arith::ConstantIndexOp>(loc, 0)));
    }
    return result;
  }
};

}  // namespace

void registerFusionInterfaceExternalModels(DialectRegistry& registry) {
  registry.insert<linalg::LinalgDialect>();
  registry.addExtension(+[](MLIRContext* ctx, linalg::LinalgDialect*) {
    linalg::GenericOp::attachInterface<LinalgGenericFusionInterface>(*ctx);
  });
}

}  // namespace gml_st
}  // namespace mlir

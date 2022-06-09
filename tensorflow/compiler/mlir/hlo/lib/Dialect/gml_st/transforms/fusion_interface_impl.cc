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

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace gml_st {

namespace {

Value materializeSpaceFromTensor(Value operand, const SmallVector<Value>& dims,
                                 OpBuilder& builder) {
  auto loc = operand.getLoc();
  auto ty = operand.getType().cast<RankedTensorType>();

  // Collect dimension info and materialize `dim` ops for dynamic dimensions.
  SmallVector<int64_t> staticDims;
  SmallVector<Value> dynamicDims;
  for (const auto& it : llvm::enumerate(ty.getShape())) {
    int64_t d = it.value();
    if (d != ShapedType::kDynamicSize) {
      staticDims.push_back(d);
    } else {
      staticDims.push_back(ShapedType::kDynamicSize);
      dynamicDims.push_back(dims[it.index()]);
    }
  }

  // Materialize `space` op.
  auto spaceTy = builder.getType<TileType>(ty.getShape());
  auto staticDimsAttr = builder.getI64ArrayAttr(staticDims);
  return builder.create<SpaceOp>(loc, spaceTy, dynamicDims, staticDimsAttr);
}

struct DynamicBroadcastInDimFusionIterface
    : public FusionIterface::ExternalModel<DynamicBroadcastInDimFusionIterface,
                                           DynamicBroadcastInDimOp> {
  Value fuse(Operation* op, MaterializeOp materializeOp,
             OpBuilder& builder) const {
    // Supports tile subsets.
    Value tile = materializeOp.subset();
    Type subsetTy = tile.getType();
    if (!subsetTy.isa<TileType>()) return {};

    // Create the needed constants only once.
    DenseMap<uint64_t, Value> localIndexConstants;
    auto bcastOp = llvm::dyn_cast<DynamicBroadcastInDimOp>(op);
    auto loc = bcastOp.getLoc();
    auto getIndexConstant = [&](uint64_t c) -> Value {
      auto it = localIndexConstants.find(c);
      if (it != localIndexConstants.end()) return it->second;
      auto cst = builder.create<arith::ConstantIndexOp>(loc, c);
      localIndexConstants[c] = cst;
      return cst;
    };

    Value operand = bcastOp.operand();
    auto operandTy = operand.getType().cast<RankedTensorType>();
    auto tileTy = tile.getType().cast<TileType>();
    auto resultTy = bcastOp.getType().cast<RankedTensorType>();

    // Materiaize operand dimensions.
    SmallVector<Value> operandDims;
    operandDims.reserve(operandTy.getRank());
    for (const auto& it : llvm::enumerate(operandTy.getShape())) {
      int64_t d = it.value();
      Value dim =
          d == ShapedType::kDynamicSize
              ? builder.create<tensor::DimOp>(loc, operand, it.index())
                    .getResult()
              : builder.create<arith::ConstantIndexOp>(loc, d).getResult();
      operandDims.push_back(dim);
    }

    // Materialize operand and result space.
    Value operandSpace =
        materializeSpaceFromTensor(operand, operandDims, builder);

    // Materialize offsets and sizes for operand tile.
    auto collapsedTile = builder.create<CollapseTileOp>(
        loc, tile, bcastOp.broadcast_dimensions());
    SmallVector<Value> argTileOffsets;
    SmallVector<Value> argTileSizes;
    for (const auto& it : llvm::enumerate(bcastOp.broadcast_dimensions())) {
      Value argIdx = getIndexConstant(it.index());
      Value resultIdx = getIndexConstant(it.value().getLimitedValue());

      // If corresponding operand and result dimensions are different, the
      // dimension is expanding.
      auto argDim = operandDims[it.index()];
      auto resultDim =
          builder.create<tensor::DimOp>(loc, bcastOp.init(), resultIdx);
      auto isExpanding = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, argDim, resultDim);

      // Copy offset for non-expanding dimensions and index at 0, otrherwise.
      auto tileOffset = builder.create<OffsetOp>(loc, collapsedTile, argIdx);
      auto tileSize = builder.create<SizeOp>(loc, collapsedTile, argIdx);
      argTileOffsets.push_back(builder.create<arith::SelectOp>(
          loc, isExpanding, getIndexConstant(0), tileOffset));
      argTileSizes.push_back(builder.create<arith::SelectOp>(
          loc, isExpanding, getIndexConstant(1), tileSize));
    }

    // Materialize operand tile.
    int64_t rank = operandTy.getRank();
    auto staticOffsets = builder.getI64ArrayAttr(
        SmallVector<int64_t>(rank, ShapedType::kDynamicStrideOrOffset));
    auto staticSizes = builder.getI64ArrayAttr(
        SmallVector<int64_t>(rank, ShapedType::kDynamicSize));
    auto staticStrides = builder.getI64ArrayAttr(SmallVector<int64_t>(rank, 1));
    auto operandTileTy = builder.getType<TileType>(
        SmallVector<int64_t>(rank, ShapedType::kDynamicSize));
    auto operandTile = builder.create<TileOp>(
        loc, operandTileTy, operandSpace, argTileOffsets, argTileSizes,
        ValueRange{}, staticOffsets, staticSizes, staticStrides);

    // Materialize operands' subsets.
    Value tiledInit = builder.create<MaterializeOp>(loc, bcastOp.init(), tile);
    Value tiledOperand =
        builder.create<MaterializeOp>(loc, operand, operandTile);

    // Finally, materialize tiled broadcast.
    auto tiledResultTy =
        RankedTensorType::get(tileTy.getShape(), resultTy.getElementType());
    return builder.create<DynamicBroadcastInDimOp>(
        loc, tiledResultTy, tiledInit, tiledOperand,
        bcastOp.broadcast_dimensions(),
        bcastOp.known_expanding_dimensionsAttr(),
        bcastOp.known_nonexpanding_dimensionsAttr());
  }
};

template <typename OpTy>
struct ElementwiseFusionInterface
    : public FusionIterface::ExternalModel<ElementwiseFusionInterface<OpTy>,
                                           OpTy> {
  Value fuse(Operation* op, MaterializeOp materializeOp,
             OpBuilder& builder) const {
    // Supports tile and point subsets.
    Value subset = materializeOp.subset();
    Type subsetTy = subset.getType();
    if (!subsetTy.isa<PointType, TileType>()) return {};

    // Materialize subsets for all arguments.
    auto ewiseOp = cast<OpTy>(op);
    Location loc = materializeOp.getLoc();
    auto subsetArgs = llvm::to_vector(
        llvm::map_range(ewiseOp->getOperands(), [&](const auto& arg) -> Value {
          return builder.create<MaterializeOp>(loc, arg, subset);
        }));

    // Materialize elementwise op for subset.
    return llvm::TypeSwitch<Type, Value>(subsetTy)
        .Case([&](TileType) -> Value {
          return builder.create<OpTy>(loc, subsetArgs);
        })
        .Case([&](PointType) -> Value {
          return mhlo::MhloOpToStdScalarOp::mapOp(
              ewiseOp, materializeOp.getType(), subsetArgs, &builder);
        })
        .Default([](Type) -> Value { return {}; });
  }
};

}  // namespace

void registerFusionInterfaceExternalModels(DialectRegistry& registry) {
  registry.insert<mhlo::MhloDialect>();
  registry.addExtension(+[](MLIRContext* ctx, mhlo::MhloDialect*) {
    mhlo::AddOp::attachInterface<ElementwiseFusionInterface<mhlo::AddOp>>(*ctx);
    mhlo::SubOp::attachInterface<ElementwiseFusionInterface<mhlo::SubOp>>(*ctx);
    mhlo::CosOp::attachInterface<ElementwiseFusionInterface<mhlo::CosOp>>(*ctx);
    mhlo::TanhOp::attachInterface<ElementwiseFusionInterface<mhlo::TanhOp>>(
        *ctx);

    // TODO(frgossen): Implement the interface directly on these extension ops
    // when cyclic dependencies are resolved.
    DynamicBroadcastInDimOp::attachInterface<
        DynamicBroadcastInDimFusionIterface>(*ctx);
  });
}

}  // namespace gml_st
}  // namespace mlir

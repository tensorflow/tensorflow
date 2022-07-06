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

#include <tuple>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"

namespace mlir {
namespace gml_st {

namespace {

bool isElementwiseOrTranspose(linalg::GenericOp genericOp) {
  if (!genericOp.hasTensorSemantics()) return false;
  if (genericOp.outputs().size() != 1) return false;
  if (!llvm::all_of(genericOp.iterator_types(), [](Attribute attr) {
        return mlir::isParallelIterator(attr);
      })) {
    return false;
  }
  if (!llvm::all_of(genericOp.indexing_maps(), [](Attribute attr) {
        return attr.cast<AffineMapAttr>().getAffineMap().isPermutation();
      })) {
    return false;
  }
  return true;
}

struct LinalgGenericFusionInterface
    : public FusionInterface::ExternalModel<LinalgGenericFusionInterface,
                                            linalg::GenericOp> {
  Value fuse(Operation* op, Location loc, Value subset,
             OpBuilder& builder) const {
    auto genericOp = llvm::cast<linalg::GenericOp>(op);

    // Supports only tile subsets.
    auto tileTy = subset.getType().dyn_cast<TileType>();
    if (!tileTy) return {};

    // Supports only element-wise `linalg.generic` ops.
    if (!isElementwiseOrTranspose(genericOp)) return {};

    // Create tiled op.
    Value output = genericOp.outputs().front();
    auto outputTy = output.getType().cast<RankedTensorType>();
    auto subResultTy =
        RankedTensorType::get(tileTy.getShape(), outputTy.getElementType());
    SmallVector<Value> subOperands;
    subOperands.reserve(genericOp.getNumInputs());
    for (const auto& inputAndMap :
         llvm::zip(genericOp.inputs(), genericOp.getIndexingMaps())) {
      Value input;
      AffineMap map;
      std::tie(input, map) = inputAndMap;
      if (map.isIdentity()) {
        subOperands.push_back(
            builder.create<MaterializeOp>(loc, input, subset));
        continue;
      }
      assert(map.isPermutation());
      // Materialize a space for the input.
      auto inputTy = input.getType().cast<RankedTensorType>();
      auto spaceTy = builder.getType<TileType>(inputTy.getShape());
      auto dynamicDims = tensor::createDynamicDimValues(builder, loc, input);
      auto staticDims = builder.getI64ArrayAttr(inputTy.getShape());
      Value inputSpace =
          builder.create<SpaceOp>(loc, spaceTy, dynamicDims, staticDims);

      // Create a new tile with permutated operands.
      SmallVector<Value> inputTileOffsets, inputTileSizes, inputTileStrides;
      SmallVector<int64_t> inputStaticOffsets, inputStaticSizes,
          inputStaticStrides;
      // Use original tileOp where possible.
      auto tileOp = subset.getDefiningOp<TileOp>();

      for (unsigned int r = 0, e = map.getNumResults(); r < e; ++r) {
        auto permutedDim = map.getPermutedPosition(r);
        auto permutedDimConstant =
            builder.create<arith::ConstantIndexOp>(loc, permutedDim);
        // TODO(unknown): With a canonicalizer, we could always use values.
        if (!tileOp || tileOp.isDynamicOffset(permutedDim)) {
          inputTileOffsets.push_back(
              builder.createOrFold<OffsetOp>(loc, subset, permutedDimConstant));
          inputStaticOffsets.push_back(ShapedType::kDynamicStrideOrOffset);
        } else {
          inputStaticOffsets.push_back(tileOp.getStaticOffset(permutedDim));
        }
        if (!tileOp || tileOp.isDynamicSize(permutedDim)) {
          inputTileSizes.push_back(
              builder.createOrFold<SizeOp>(loc, subset, permutedDimConstant));
          inputStaticSizes.push_back(ShapedType::kDynamicSize);
        } else {
          inputStaticSizes.push_back(tileOp.getStaticSize(permutedDim));
        }
        if (!tileOp || tileOp.isDynamicStride(permutedDim)) {
          inputTileStrides.push_back(
              builder.createOrFold<StrideOp>(loc, subset, permutedDimConstant));
          inputStaticStrides.push_back(ShapedType::kDynamicStrideOrOffset);
        } else {
          inputStaticStrides.push_back(tileOp.getStaticStride(permutedDim));
        }
      }
      auto operandTileTy =
          TileType::get(subResultTy.getContext(), inputStaticSizes);
      auto permutedSubset = builder.create<TileOp>(
          loc, operandTileTy, inputSpace, inputTileOffsets, inputTileSizes,
          inputTileStrides, builder.getI64ArrayAttr(inputStaticOffsets),
          builder.getI64ArrayAttr(inputStaticSizes),
          builder.getI64ArrayAttr(inputStaticStrides));
      subOperands.push_back(
          builder.create<MaterializeOp>(loc, input, permutedSubset));
    }
    // Materialize the tiled output.
    subOperands.push_back(builder.create<MaterializeOp>(loc, output, subset));
    linalg::LinalgOp linalgOp = genericOp;
    Operation* tiledOp = linalgOp.clone(builder, loc, subResultTy, subOperands);
    return tiledOp->getResults().front();
  }
};

template <typename OpTy>
struct ElementwiseFusionInterface
    : public FusionInterface::ExternalModel<ElementwiseFusionInterface<OpTy>,
                                            OpTy> {
  Value fuse(Operation* op, Location loc, Value subset,
             OpBuilder& builder) const {
    // Supports tile and point subsets.
    Type subsetTy = subset.getType();
    if (!subsetTy.isa<PointType, TileType>()) return {};

    // Expect ranked element-wise op.
    auto cwiseOp = llvm::cast<OpTy>(op);
    auto rankedTy = cwiseOp.getType().template dyn_cast<RankedTensorType>();
    if (!rankedTy) return {};

    // Materialize subsets for all arguments.
    auto subsetArgs = llvm::to_vector(
        llvm::map_range(cwiseOp->getOperands(), [&](const auto& arg) -> Value {
          return builder.create<MaterializeOp>(loc, arg, subset);
        }));

    // Materialize elementwise op for subset.
    return llvm::TypeSwitch<Type, Value>(subsetTy)
        .Case([&](TileType) -> Value {
          return builder.create<OpTy>(loc, subsetArgs);
        })
        .Case([&](PointType) -> Value {
          return mhlo::MhloOpToStdScalarOp::mapOp(
              cwiseOp, rankedTy.getElementType(), subsetArgs, &builder);
        })
        .Default([](Type) -> Value { return {}; });
  }
};

}  // namespace

void registerFusionInterfaceExternalModels(DialectRegistry& registry) {
  registry.insert<linalg::LinalgDialect>();
  registry.addExtension(+[](MLIRContext* ctx, linalg::LinalgDialect*) {
    linalg::GenericOp::attachInterface<LinalgGenericFusionInterface>(*ctx);
  });

  // TODO(frgossen): Update tests and remove these in favor of
  // `linalg.generic`-based fusions.
  registry.insert<mhlo::MhloDialect>();
  registry.addExtension(+[](MLIRContext* ctx, mhlo::MhloDialect*) {
    mhlo::AddOp::attachInterface<ElementwiseFusionInterface<mhlo::AddOp>>(*ctx);
    mhlo::SubOp::attachInterface<ElementwiseFusionInterface<mhlo::SubOp>>(*ctx);
    mhlo::CosOp::attachInterface<ElementwiseFusionInterface<mhlo::CosOp>>(*ctx);
    mhlo::TanhOp::attachInterface<ElementwiseFusionInterface<mhlo::TanhOp>>(
        *ctx);
  });
}

}  // namespace gml_st
}  // namespace mlir

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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace gml_st {

namespace {

enum class LinalgGenericFusionKind {
  FuseAsElementwise,
  FuseAsTranspose,
  None,
};

LinalgGenericFusionKind getLinalgGenericFusionKind(
    linalg::GenericOp genericOp) {
  // Only consider all-parallel `linalg.generic` ops with a unique result and
  // tensor semantics for fusion.
  if (!genericOp.hasTensorSemantics() || genericOp.outputs().size() != 1 ||
      llvm::any_of(genericOp.iterator_types(), [](Attribute attr) {
        return !mlir::isParallelIterator(attr);
      })) {
    return LinalgGenericFusionKind::None;
  }

  // Fuse as element-wise if all maps are identity maps.
  if (llvm::all_of(genericOp.indexing_maps(), [](Attribute attr) {
        return attr.cast<AffineMapAttr>().getAffineMap().isIdentity();
      })) {
    return LinalgGenericFusionKind::FuseAsElementwise;
  }

  // Fuse as transpose if all maps are permutation maps.
  if (llvm::all_of(genericOp.indexing_maps(), [](Attribute attr) {
        return attr.cast<AffineMapAttr>().getAffineMap().isPermutation();
      })) {
    return LinalgGenericFusionKind::FuseAsTranspose;
  }

  return LinalgGenericFusionKind::None;
}

Value fuseAsElementwise(linalg::GenericOp genericOp, Location loc, Value subset,
                        OpBuilder& builder) {
  assert(getLinalgGenericFusionKind(genericOp) ==
             LinalgGenericFusionKind::FuseAsElementwise &&
         "expect element-wise linalg.generic op");
  linalg::LinalgOp linalgOp = genericOp;
  return llvm::TypeSwitch<Type, Value>(subset.getType())
      .Case([&](TileType tileTy) -> Value {
        // Create tiled op.
        Value output = genericOp.outputs().front();
        auto outputTy = output.getType().cast<RankedTensorType>();
        auto subResultTy =
            RankedTensorType::get(tileTy.getShape(), outputTy.getElementType());
        SmallVector<Value> subOperands;
        subOperands.reserve(genericOp.getNumInputs());
        for (auto input : genericOp.inputs()) {
          subOperands.push_back(
              builder.create<MaterializeOp>(loc, input, subset));
        }
        subOperands.push_back(
            builder.create<MaterializeOp>(loc, output, subset));
        Operation* tiledOp =
            linalgOp.clone(builder, loc, subResultTy, subOperands);

        return tiledOp->getResults().front();
      })
      .Case([&](PointType) -> Value {
        // Create scalar computation.
        BlockAndValueMapping bvm;
        Block* block = genericOp.getBody();
        for (auto it : llvm::zip(block->getArguments(), linalgOp.inputs())) {
          bvm.map(std::get<0>(it),
                  builder.create<MaterializeOp>(loc, std::get<1>(it), subset));
        }
        for (auto& it : block->without_terminator()) builder.clone(it, bvm);

        auto innerResults = block->getTerminator()->getOperands();
        assert(innerResults.size() == 1 && "expect unique inner result");
        return bvm.lookup(innerResults.front());
      })
      .Default([](Type) -> Value { return {}; });
}

Value fuseAsTranspose(linalg::GenericOp genericOp, Location loc, Value subset,
                      OpBuilder& builder) {
  assert(getLinalgGenericFusionKind(genericOp) ==
             LinalgGenericFusionKind::FuseAsTranspose &&
         "expect transposing linalg.generic op");

  auto tileTy = subset.getType().dyn_cast<TileType>();
  if (!tileTy) return {};

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
      subOperands.push_back(builder.create<MaterializeOp>(loc, input, subset));
      continue;
    }
    assert(map.isPermutation());
    SmallVector<int64_t> permutation;
    permutation.reserve(map.getNumResults());
    for (unsigned int r = 0, e = map.getNumResults(); r < e; ++r) {
      permutation.push_back(map.getPermutedPosition(r));
    }
    auto transposedTile = builder.create<TransposeTileOp>(
        loc, subset, DenseI64ArrayAttr::get(builder.getContext(), permutation));
    subOperands.push_back(
        builder.create<MaterializeOp>(loc, input, transposedTile));
  }
  // Materialize the tiled output.
  subOperands.push_back(builder.create<MaterializeOp>(loc, output, subset));
  linalg::LinalgOp linalgOp = genericOp;
  Operation* tiledOp = linalgOp.clone(builder, loc, subResultTy, subOperands);
  return tiledOp->getResults().front();
}

struct LinalgGenericFusionInterface
    : public FusionInterface::ExternalModel<LinalgGenericFusionInterface,
                                            linalg::GenericOp> {
  Value fuse(Operation* op, Location loc, Value subset,
             OpBuilder& builder) const {
    auto genericOp = llvm::cast<linalg::GenericOp>(op);
    auto kind = getLinalgGenericFusionKind(genericOp);

    if (kind == LinalgGenericFusionKind::FuseAsElementwise) {
      return fuseAsElementwise(genericOp, loc, subset, builder);
    }

    if (kind == LinalgGenericFusionKind::FuseAsTranspose) {
      return fuseAsTranspose(genericOp, loc, subset, builder);
    }

    return {};
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

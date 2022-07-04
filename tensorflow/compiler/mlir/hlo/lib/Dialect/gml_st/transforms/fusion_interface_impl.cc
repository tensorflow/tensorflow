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
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace gml_st {

namespace {

bool isElementwise(linalg::GenericOp genericOp) {
  if (!genericOp.hasTensorSemantics()) return false;
  if (genericOp.outputs().size() != 1) return false;
  if (!llvm::all_of(genericOp.iterator_types(), [](Attribute attr) {
        return mlir::isParallelIterator(attr);
      })) {
    return false;
  }
  if (!llvm::all_of(genericOp.indexing_maps(), [](Attribute attr) {
        return attr.cast<AffineMapAttr>().isIdentity();
      })) {
    return false;
  }
  return true;
}

struct LingalgGenericFusionInterface
    : public FusionIterface::ExternalModel<LingalgGenericFusionInterface,
                                           linalg::GenericOp> {
  Value fuse(Operation* op, Location loc, Value subset,
             OpBuilder& builder) const {
    auto genericOp = llvm::cast<linalg::GenericOp>(op);

    // Supports only tile subsets.
    auto tileTy = subset.getType().dyn_cast<TileType>();
    if (!tileTy.isa<TileType>()) return {};

    // Supports only element-wise `linalg.generic` ops.
    if (!isElementwise(genericOp)) return {};

    // Create tiled op.
    Value output = genericOp.outputs().front();
    auto outputTy = output.getType().cast<RankedTensorType>();
    auto subResultTy =
        RankedTensorType::get(tileTy.getShape(), outputTy.getElementType());
    SmallVector<Value> subOperands;
    subOperands.reserve(genericOp.getNumInputs());
    for (auto input : genericOp.inputs()) {
      subOperands.push_back(builder.create<MaterializeOp>(loc, input, subset));
    }
    subOperands.push_back(builder.create<MaterializeOp>(loc, output, subset));
    linalg::LinalgOp linalgOp = genericOp;
    Operation* tiledOp = linalgOp.clone(builder, loc, subResultTy, subOperands);
    return tiledOp->getResults().front();
  }
};

template <typename OpTy>
struct ElementwiseFusionInterface
    : public FusionIterface::ExternalModel<ElementwiseFusionInterface<OpTy>,
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
    linalg::GenericOp::attachInterface<LingalgGenericFusionInterface>(*ctx);
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

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

#include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface.cc.inc"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace gml_st {

namespace {

template <typename OpTy>
struct BinaryElementwiseFusionInterface
    : public FusionIterface::ExternalModel<
          BinaryElementwiseFusionInterface<OpTy>, OpTy> {
  Value fuse(Operation* op, MaterializeOp materializeOp,
             OpBuilder& builder) const {
    auto binaryElementwiseOp = cast<OpTy>(op);
    Value subset = materializeOp.subset();
    Location loc = materializeOp.getLoc();

    return llvm::TypeSwitch<Type, Value>(subset.getType())
        .Case([&](PointType) -> Value {
          auto lhs = builder.create<MaterializeOp>(
              loc, binaryElementwiseOp.lhs(), subset);
          auto rhs = builder.create<MaterializeOp>(
              loc, binaryElementwiseOp.rhs(), subset);
          return mhlo::MhloOpToStdScalarOp::map<OpTy>(
              binaryElementwiseOp, materializeOp.getType(),
              llvm::ArrayRef<Value>{lhs, rhs}, &builder);
        })
        .Case([&](TileType) -> Value {
          auto lhs = builder.create<MaterializeOp>(
              loc, binaryElementwiseOp.lhs(), subset);
          auto rhs = builder.create<MaterializeOp>(
              loc, binaryElementwiseOp.rhs(), subset);
          return builder.create<OpTy>(loc, lhs, rhs);
        })
        .Default([&](Type) -> Value { return {}; });
  }
};

}  // namespace

void registerFusionInterfaceExternalModels(DialectRegistry& registry) {
  registry.insert<mhlo::MhloDialect>();
  registry.addExtension(+[](MLIRContext* ctx, mhlo::MhloDialect* /*dialect*/) {
    mhlo::AddOp::attachInterface<BinaryElementwiseFusionInterface<mhlo::AddOp>>(
        *ctx);
    mhlo::SubOp::attachInterface<BinaryElementwiseFusionInterface<mhlo::SubOp>>(
        *ctx);
  });
}

}  // namespace gml_st
}  // namespace mlir

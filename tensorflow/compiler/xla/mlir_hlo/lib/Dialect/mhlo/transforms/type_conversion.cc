/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/Dialect/mhlo/transforms/type_conversion.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {

class Value;

namespace mhlo {

namespace {

Type convertInteger(IntegerType intType) {
  return IntegerType::get(intType.getContext(),
                          intType.getIntOrFloatBitWidth());
}

Type convertShapedType(ShapedType shapedType) {
  if (auto intType = shapedType.getElementType().dyn_cast<IntegerType>())
    return shapedType.clone(convertInteger(intType));
  return shapedType;
}

llvm::Optional<Value> materializeCastFromIllegal(OpBuilder& builder, Type type,
                                                 ValueRange inputs,
                                                 Location loc) {
  Type fromType = getElementTypeOrSelf(inputs[0].getType());
  Type toType = getElementTypeOrSelf(type);
  if ((!fromType.isSignedInteger() && !fromType.isUnsignedInteger()) ||
      !toType.isSignlessInteger())
    return llvm::None;
  // Use unrealized conversion casts to do signful->signless conversions.
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

llvm::Optional<Value> materializeCastToIllegal(OpBuilder& builder, Type type,
                                               ValueRange inputs,
                                               Location loc) {
  Type fromType = getElementTypeOrSelf(inputs[0].getType());
  Type toType = getElementTypeOrSelf(type);
  if (!fromType.isSignlessInteger() ||
      (!toType.isSignedInteger() && !toType.isUnsignedInteger()))
    return llvm::None;
  // Use unrealized conversion casts to do signless->signful conversions.
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

llvm::Optional<Value> scalarToTensor(OpBuilder& builder, Type /*type*/,
                                     ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  if (inputs.front().getType().isa<ShapedType>()) {
    return llvm::None;
  }
  return builder
      .create<tensor::FromElementsOp>(
          loc, RankedTensorType::get({}, inputs.front().getType()),
          inputs.front())
      .getResult();
}

}  // namespace

RemoveSignTypeConverter::RemoveSignTypeConverter() {
  addConversion([](Type type) { return type; });

  addConversion(convertInteger);
  addConversion(convertShapedType);

  addArgumentMaterialization(materializeCastFromIllegal);
  addSourceMaterialization(materializeCastToIllegal);
  addTargetMaterialization(materializeCastFromIllegal);
}

LinalgTypeConverter::LinalgTypeConverter() : RemoveSignTypeConverter() {
  addArgumentMaterialization(scalarToTensor);
}

}  // namespace mhlo

namespace stablehlo {

// Our guiding principle is to support all StableHLO functionality in MHLO.
// The inverse is not necessarily true - some MHLO types are missing from
// StableHLO (either deliberately or haven't yet been proposed to StableHLO).
// As a result, these MHLO types will fail here.
HloToStablehloTypeConverter::HloToStablehloTypeConverter() {
  addConversion([](Type hloType) { return hloType; });
  // !mhlo.async_bundle is only used in mhlo.async_start, mhlo.async_update
  // and mhlo.async_done which are private to XLA.
  // This means that these ops are deliberately not part of StableHLO,
  // and as a result this type is not part of StableHLO either.
  addConversion([](mhlo::AsyncBundleType) -> Type { return {}; });
  addConversion([](mhlo::TokenType hloType) -> Type {
    return stablehlo::TokenType::get(hloType.getContext());
  });
  addConversion([](RankedTensorType hloType) -> Type {
    if (auto hloExtensions =
            hloType.getEncoding()
                .dyn_cast_or_null<mhlo::TypeExtensionsAttr>()) {
      auto stablehloExtensions = stablehlo::TypeExtensionsAttr::get(
          hloType.getContext(), hloExtensions.getBounds());
      return RankedTensorType::get(hloType.getShape(), hloType.getElementType(),
                                   stablehloExtensions);
    }
    return hloType;
  });
  addConversion([&](TupleType hloType) -> Type {
    SmallVector<Type> stablehloTypes;
    if (failed(convertTypes(hloType.getTypes(), stablehloTypes))) return {};
    return TupleType::get(hloType.getContext(), stablehloTypes);
  });
};

// Our guiding principle is to support all StableHLO functionality in MHLO.
// This means that the StableHLO => HLO type conversion should always succeed.
StablehloToHloTypeConverter::StablehloToHloTypeConverter() {
  addConversion([](Type stablehloType) { return stablehloType; });
  addConversion([](stablehlo::TokenType stablehloType) -> Type {
    return mhlo::TokenType::get(stablehloType.getContext());
  });
  addConversion([](RankedTensorType stablehloType) -> Type {
    if (auto stablehloExtensions =
            stablehloType.getEncoding()
                .dyn_cast_or_null<stablehlo::TypeExtensionsAttr>()) {
      auto hloExtensions = mhlo::TypeExtensionsAttr::get(
          stablehloType.getContext(), stablehloExtensions.getBounds());
      return RankedTensorType::get(stablehloType.getShape(),
                                   stablehloType.getElementType(),
                                   hloExtensions);
    }
    return stablehloType;
  });
  addConversion([&](TupleType stablehloType) -> Type {
    SmallVector<Type> hloTypes;
    if (failed(convertTypes(stablehloType.getTypes(), hloTypes))) return {};
    return TupleType::get(stablehloType.getContext(), hloTypes);
  });
};

void registerFuncOpsForTypeConversion(ConversionTarget& target,
                                      RewritePatternSet& patterns,
                                      TypeConverter& converter) {
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    return converter.isSignatureLegal(op.getCalleeType());
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    return converter.isLegal(op.getOperandTypes());
  });
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  populateCallOpTypeConversionPattern(patterns, converter);
  populateReturnOpTypeConversionPattern(patterns, converter);
}

}  // namespace stablehlo

}  // namespace mlir

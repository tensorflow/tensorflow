/* Copyright 2021 The OpenXLA Authors.

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

#include "mhlo/utils/type_conversion.h"

#include <optional>

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
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
  if (auto intType = mlir::dyn_cast<IntegerType>(shapedType.getElementType()))
    return shapedType.clone(convertInteger(intType));
  return shapedType;
}

std::optional<Value> materializeCastFromIllegal(OpBuilder& builder, Type type,
                                                ValueRange inputs,
                                                Location loc) {
  Type fromType = getElementTypeOrSelf(inputs[0].getType());
  Type toType = getElementTypeOrSelf(type);
  if ((!fromType.isSignedInteger() && !fromType.isUnsignedInteger()) ||
      !toType.isSignlessInteger())
    return std::nullopt;
  // Use unrealized conversion casts to do signful->signless conversions.
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

std::optional<Value> materializeCastToIllegal(OpBuilder& builder, Type type,
                                              ValueRange inputs, Location loc) {
  Type fromType = getElementTypeOrSelf(inputs[0].getType());
  Type toType = getElementTypeOrSelf(type);
  if (!fromType.isSignlessInteger() ||
      (!toType.isSignedInteger() && !toType.isUnsignedInteger()))
    return std::nullopt;
  // Use unrealized conversion casts to do signless->signful conversions.
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

std::optional<Value> scalarToTensor(OpBuilder& builder, Type /*type*/,
                                    ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  if (mlir::isa<ShapedType>(inputs.front().getType())) {
    return std::nullopt;
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

HloTypeConverter::HloTypeConverter() {
  addConversion([&](Type type) -> Type {
    // We cannot use an allowlist here because HLO dialects can be embedded
    // into programs with other dialects which can involve other types.
    // However, we restrict the use of types defined in the source dialect.
    // This check is here only for exceptional situations, e.g. when we added
    // a new type and forgot to update the converters in the subclass.
    if (isSourceDialect(type.getDialect())) return {};
    return type;
  });
  addConversion([&](RankedTensorType type) -> Type {
    auto encoding = type.getEncoding();
    if (!encoding) return type;

    // Since this type converter can be used in all sorts of programs,
    // we generally want to allow most of the encodings to pass through,
    // However, we restrict the use of encodings defined in the source dialect.
    if (isSourceDialect(encoding.getDialect())) {
      auto convertedEncoding = convertSourceDialectEncoding(encoding);
      if (!convertedEncoding) return {};
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   convertedEncoding);
    }
    return type;
  });
  addConversion([&](TupleType type) -> Type {
    SmallVector<Type> convertedTypes;
    if (failed(convertTypes(type.getTypes(), convertedTypes))) return {};
    return TupleType::get(type.getContext(), convertedTypes);
  });
}

HloToStablehloTypeConverter::HloToStablehloTypeConverter()
    : HloTypeConverter() {
  // !mhlo.async_bundle is only used in mhlo.async_start, mhlo.async_update
  // and mhlo.async_done which are private to XLA.
  // This means that these ops are deliberately not part of StableHLO,
  // and as a result this type is not part of StableHLO either.
  addConversion([](mhlo::AsyncBundleType) -> Type { return {}; });
  addConversion([](mhlo::TokenType type) -> Type {
    return stablehlo::TokenType::get(type.getContext());
  });
  // Consider implementing stablehlo::CustomType to provide an escape hatch
  // for modelling MHLO types that aren't yet in StableHLO.
  // Proposal: https://github.com/openxla/stablehlo/issues/743.
}

bool HloToStablehloTypeConverter::isSourceDialect(Dialect& dialect) {
  return dialect.getNamespace() == mhlo::MhloDialect::getDialectNamespace();
}

Attribute HloToStablehloTypeConverter::convertSourceDialectEncoding(
    Attribute attr) {
  if (auto hloAttr = mlir::dyn_cast_or_null<mhlo::TypeExtensionsAttr>(attr)) {
    return stablehlo::TypeExtensionsAttr::get(hloAttr.getContext(),
                                              hloAttr.getBounds());
  }
  // Our guiding principle is to support all MHLO encodings in StableHLO.
  // This check is here only for exceptional situations, e.g. when we added
  // a new MHLO encoding and forgot to update the code above.
  return {};
}

StablehloToHloTypeConverter::StablehloToHloTypeConverter()
    : HloTypeConverter() {
  addConversion([](stablehlo::TokenType stablehloType) -> Type {
    return mhlo::TokenType::get(stablehloType.getContext());
  });
}

bool StablehloToHloTypeConverter::isSourceDialect(Dialect& dialect) {
  return dialect.getNamespace() ==
         stablehlo::StablehloDialect::getDialectNamespace();
}

Attribute StablehloToHloTypeConverter::convertSourceDialectEncoding(
    Attribute attr) {
  if (auto stablehloAttr =
          mlir::dyn_cast_or_null<stablehlo::TypeExtensionsAttr>(attr)) {
    return mhlo::TypeExtensionsAttr::get(stablehloAttr.getContext(),
                                         stablehloAttr.getBounds());
  }
  // Our guiding principle is to support all StableHLO encodings in MHLO.
  // This check is here only for exceptional situations, e.g. when we added
  // a new StableHLO encoding and forgot to update the code above.
  return {};
}

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

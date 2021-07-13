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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {

struct Value;

namespace mhlo {

namespace {

Type convertInteger(IntegerType int_type) {
  return IntegerType::get(int_type.getContext(),
                          int_type.getIntOrFloatBitWidth());
}

Type convertShapedType(ShapedType shaped_type) {
  if (auto int_type = shaped_type.getElementType().dyn_cast<IntegerType>())
    return shaped_type.clone(convertInteger(int_type));
  return shaped_type;
}

llvm::Optional<Value> materializeCastFromIllegal(OpBuilder& builder, Type type,
                                                 ValueRange inputs,
                                                 Location loc) {
  Type from_type = getElementTypeOrSelf(inputs[0].getType());
  Type to_type = getElementTypeOrSelf(type);
  if ((!from_type.isSignedInteger() && !from_type.isUnsignedInteger()) ||
      !to_type.isSignlessInteger())
    return llvm::None;
  // Use unrealized conversion casts to do signful->signless conversions.
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

llvm::Optional<Value> materializeCastToIllegal(OpBuilder& builder, Type type,
                                               ValueRange inputs,
                                               Location loc) {
  Type from_type = getElementTypeOrSelf(inputs[0].getType());
  Type to_type = getElementTypeOrSelf(type);
  if (!from_type.isSignlessInteger() ||
      (!to_type.isSignedInteger() && !to_type.isUnsignedInteger()))
    return llvm::None;
  // Use unrealized conversion casts to do signless->signful conversions.
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
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

}  // namespace mhlo
}  // namespace mlir

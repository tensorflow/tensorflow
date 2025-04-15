/* Copyright 2019 The OpenXLA Authors.

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

// This file defines helpers useful when creating or manipulating lhlo/hlo.

#include "utils/convert_op_folder.h"

#include <utility>

#include "llvm/ADT/APSInt.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace hlo {

mlir::ElementsAttr convertElementsAttr(const mlir::ElementsAttr& elements,
                                       mlir::Type newType) {
  auto oldType = getElementTypeOrSelf(elements);
  // TODO(kramerb): Add support when MLIR can represent const complex tensors.
  if (!mlir::isa<mlir::IntegerType, mlir::FloatType>(oldType) ||
      !mlir::isa<mlir::IntegerType, mlir::FloatType>(newType)) {
    return {};
  }

  size_t bitWidth = newType.isBF16() ? 64 : newType.getIntOrFloatBitWidth();
  // Treat signless integers except i1 as signed.
  bool isOldTypeUnsigned = oldType.isInteger(1) || oldType.isUnsignedInteger();
  bool isNewTypeUnsigned = newType.isInteger(1) || newType.isUnsignedInteger();

  if (mlir::isa<mlir::FloatType>(oldType)) {
    if (auto newFloatType = mlir::dyn_cast<mlir::FloatType>(newType)) {
      // Float -> Float
      return mlir::cast<DenseIntOrFPElementsAttr>(elements).mapValues(
          newType, [&](const APFloat& floatVal) -> APInt {
            APFloat convertedFloat = floatVal;
            bool losesInfo = false;
            convertedFloat.convert(newFloatType.getFloatSemantics(),
                                   APFloat::rmNearestTiesToEven, &losesInfo);
            return convertedFloat.bitcastToAPInt();
          });
    }
    // Float -> Int
    return mlir::cast<DenseIntOrFPElementsAttr>(elements).mapValues(
        newType, [&](const APFloat& floatVal) -> APInt {
          bool ignored;
          APSInt intVal(bitWidth, isNewTypeUnsigned);
          floatVal.convertToInteger(intVal, APFloat::rmTowardZero, &ignored);
          return std::move(intVal);
        });
  }

  // old_type is Integer
  if (auto newFloatType = mlir::dyn_cast<mlir::FloatType>(newType)) {
    // Int -> Float
    return mlir::cast<DenseIntOrFPElementsAttr>(elements).mapValues(
        newType, [&](const APInt& intVal) -> APInt {
          APFloat floatVal(newFloatType.getFloatSemantics(),
                           APInt::getZero(newFloatType.getWidth()));
          floatVal.convertFromAPInt(intVal,
                                    /*isSigned=*/!isOldTypeUnsigned,
                                    APFloat::rmNearestTiesToEven);
          return floatVal.bitcastToAPInt();
        });
  }
  // new_type is Integer
  // Int -> Int
  return mlir::cast<DenseIntOrFPElementsAttr>(elements).mapValues(
      newType, [&](const APInt& intVal) -> APInt {
        return APSInt(intVal, isOldTypeUnsigned).extOrTrunc(bitWidth);
      });
}

}  // namespace hlo
}  // namespace mlir

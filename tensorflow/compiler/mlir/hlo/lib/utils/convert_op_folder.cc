/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/utils/convert_op_folder.h"

#include "llvm/ADT/APSInt.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace hlo {

mlir::ElementsAttr ConvertElementsAttr(const mlir::ElementsAttr& elements,
                                       mlir::Type new_type) {
  auto old_type = getElementTypeOrSelf(elements);
  // TODO(kramerb): Add support when MLIR can represent const complex tensors.
  if (old_type.isa<mlir::ComplexType>() || new_type.isa<mlir::ComplexType>()) {
    return {};
  }

  size_t bit_width = new_type.isBF16() ? 64 : new_type.getIntOrFloatBitWidth();
  // Treat signless integers except i1 as signed.
  bool is_old_type_unsigned =
      old_type.isInteger(1) || old_type.isUnsignedInteger();
  bool is_new_type_unsigned =
      new_type.isInteger(1) || new_type.isUnsignedInteger();

  if (old_type.isa<mlir::FloatType>()) {
    if (auto newFloatType = new_type.dyn_cast<mlir::FloatType>()) {
      // Float -> Float
      return elements.cast<DenseIntOrFPElementsAttr>().mapValues(
          new_type, [&](const APFloat& float_val) -> APInt {
            APFloat converted_float = float_val;
            bool loses_info = false;
            converted_float.convert(newFloatType.getFloatSemantics(),
                                    APFloat::rmNearestTiesToEven, &loses_info);
            return converted_float.bitcastToAPInt();
          });
    }
    // Float -> Int
    return elements.cast<DenseIntOrFPElementsAttr>().mapValues(
        new_type, [&](const APFloat& float_val) -> APInt {
          bool ignored;
          APSInt int_val(bit_width, is_new_type_unsigned);
          float_val.convertToInteger(int_val, APFloat::rmTowardZero, &ignored);
          return int_val;
        });
  }

  // old_type is Integer
  if (auto newFloatType = new_type.dyn_cast<mlir::FloatType>()) {
    // Int -> Float
    return elements.cast<DenseIntOrFPElementsAttr>().mapValues(
        new_type, [&](const APInt& int_val) -> APInt {
          APFloat float_val(newFloatType.getFloatSemantics(),
                            APInt::getZero(newFloatType.getWidth()));
          float_val.convertFromAPInt(int_val,
                                     /*isSigned=*/!is_old_type_unsigned,
                                     APFloat::rmNearestTiesToEven);
          return float_val.bitcastToAPInt();
        });
  }
  // new_type is Integer
  // Int -> Int
  return elements.cast<DenseIntOrFPElementsAttr>().mapValues(
      new_type, [&](const APInt& int_val) -> APInt {
        return APSInt(int_val, is_old_type_unsigned).extOrTrunc(bit_width);
      });
}

}  // namespace hlo
}  // namespace mlir

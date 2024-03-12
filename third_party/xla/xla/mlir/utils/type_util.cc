/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/mlir/utils/type_util.h"

#include "absl/status/statusor.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "xla/primitive_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<mlir::Type> ConvertPrimitiveTypeToMlirType(
    xla::PrimitiveType type, mlir::Builder b) {
  switch (type) {
    case xla::PrimitiveType::PRED:
      return b.getI1Type();
    case xla::PrimitiveType::F8E5M2:
      return b.getFloat8E5M2Type();
    case xla::PrimitiveType::F8E4M3FN:
      return b.getFloat8E4M3FNType();
    case xla::PrimitiveType::F8E4M3B11FNUZ:
      return b.getFloat8E4M3B11FNUZType();
    case xla::PrimitiveType::F8E5M2FNUZ:
      return b.getFloat8E5M2FNUZType();
    case xla::PrimitiveType::F8E4M3FNUZ:
      return b.getFloat8E4M3FNUZType();
    case xla::PrimitiveType::F16:
      return b.getF16Type();
    case xla::PrimitiveType::BF16:
      return b.getBF16Type();
    case xla::PrimitiveType::F32:
      return b.getF32Type();
    case xla::PrimitiveType::F64:
      return b.getF64Type();
    // TODO(b/130356985): Support unsigned primitive types.
    default:
      if (xla::primitive_util::IsIntegralType(type)) {
        return mlir::IntegerType::get(
            b.getContext(),
            /*width=*/xla::primitive_util::BitWidth(type),
            /*signed=*/
            xla::primitive_util::IsUnsignedIntegralType(type)
                ? mlir::IntegerType::Unsigned
                : mlir::IntegerType::Signless);
      }
      if (xla::primitive_util::IsComplexType(type)) {
        TF_ASSIGN_OR_RETURN(
            mlir::Type component_type,
            xla::ConvertPrimitiveTypeToMlirType(
                xla::primitive_util::ComplexComponentType(type), b));
        return mlir::ComplexType::get(component_type);
      }
      return xla::Internal("Unsupported type: %s",
                           xla::PrimitiveType_Name(type));
  }
}

xla::PrimitiveType ConvertMlirTypeToPrimitiveType(mlir::Type type) {
  if (type.isFloat8E5M2()) {
    return xla::PrimitiveType::F8E5M2;
  } else if (type.isFloat8E4M3FN()) {
    return xla::PrimitiveType::F8E4M3FN;
  } else if (type.isFloat8E4M3B11FNUZ()) {
    return xla::PrimitiveType::F8E4M3B11FNUZ;
  } else if (type.isFloat8E4M3FNUZ()) {
    return xla::PrimitiveType::F8E4M3FNUZ;
  } else if (type.isFloat8E5M2FNUZ()) {
    return xla::PrimitiveType::F8E5M2FNUZ;
  } else if (type.isBF16()) {
    return xla::PrimitiveType::BF16;
  } else if (type.isF16()) {
    return xla::PrimitiveType::F16;
  } else if (type.isF32()) {
    return xla::PrimitiveType::F32;
  } else if (type.isF64()) {
    return xla::PrimitiveType::F64;
  } else if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    mlir::Type element_ty = complex_type.getElementType();
    return xla::primitive_util::ComplexType(
        ConvertMlirTypeToPrimitiveType(element_ty));
  } else if (auto integer_type = type.dyn_cast<mlir::IntegerType>()) {
    bool is_unsigned = integer_type.isUnsigned();
    if (integer_type.getWidth() == 1) {
      return xla::PrimitiveType::PRED;
    }
    return is_unsigned ? xla::primitive_util::UnsignedIntegralTypeForBitWidth(
                             integer_type.getWidth())
                       : xla::primitive_util::SignedIntegralTypeForBitWidth(
                             integer_type.getWidth());
  }
  return xla::PrimitiveType::PRIMITIVE_TYPE_INVALID;
}
}  // namespace xla

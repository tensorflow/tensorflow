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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/primitive_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

absl::StatusOr<mlir::Type> ConvertPrimitiveTypeToMlirType(
    xla::PrimitiveType type, mlir::Builder b) {
  switch (type) {
    case xla::PrimitiveType::PRED:
      return b.getI1Type();
    case xla::PrimitiveType::F4E2M1FN:
      return b.getType<mlir::Float4E2M1FNType>();
    case xla::PrimitiveType::F8E5M2:
      return b.getType<mlir::Float8E5M2Type>();
    case xla::PrimitiveType::F8E4M3:
      return b.getType<mlir::Float8E4M3Type>();
    case xla::PrimitiveType::F8E4M3FN:
      return b.getType<mlir::Float8E4M3FNType>();
    case xla::PrimitiveType::F8E4M3B11FNUZ:
      return b.getType<mlir::Float8E4M3B11FNUZType>();
    case xla::PrimitiveType::F8E5M2FNUZ:
      return b.getType<mlir::Float8E5M2FNUZType>();
    case xla::PrimitiveType::F8E4M3FNUZ:
      return b.getType<mlir::Float8E4M3FNUZType>();
    case xla::PrimitiveType::F8E3M4:
      return b.getType<mlir::Float8E3M4Type>();
    case xla::PrimitiveType::F8E8M0FNU:
      return b.getType<mlir::Float8E8M0FNUType>();
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
  if (llvm::isa<mlir::Float4E2M1FNType>(type)) {
    return xla::PrimitiveType::F4E2M1FN;
  } else if (llvm::isa<mlir::Float8E5M2Type>(type)) {
    return xla::PrimitiveType::F8E5M2;
  } else if (llvm::isa<mlir::Float8E4M3Type>(type)) {
    return xla::PrimitiveType::F8E4M3;
  } else if (llvm::isa<mlir::Float8E4M3FNType>(type)) {
    return xla::PrimitiveType::F8E4M3FN;
  } else if (llvm::isa<mlir::Float8E4M3B11FNUZType>(type)) {
    return xla::PrimitiveType::F8E4M3B11FNUZ;
  } else if (llvm::isa<mlir::Float8E4M3FNUZType>(type)) {
    return xla::PrimitiveType::F8E4M3FNUZ;
  } else if (llvm::isa<mlir::Float8E5M2FNUZType>(type)) {
    return xla::PrimitiveType::F8E5M2FNUZ;
  } else if (llvm::isa<mlir::Float8E3M4Type>(type)) {
    return xla::PrimitiveType::F8E3M4;
  } else if (llvm::isa<mlir::Float8E8M0FNUType>(type)) {
    return xla::PrimitiveType::F8E8M0FNU;
  } else if (type.isBF16()) {
    return xla::PrimitiveType::BF16;
  } else if (type.isF16()) {
    return xla::PrimitiveType::F16;
  } else if (type.isF32()) {
    return xla::PrimitiveType::F32;
  } else if (type.isF64()) {
    return xla::PrimitiveType::F64;
  } else if (auto complex_type = mlir::dyn_cast<mlir::ComplexType>(type)) {
    mlir::Type element_ty = complex_type.getElementType();
    return xla::primitive_util::ComplexType(
        ConvertMlirTypeToPrimitiveType(element_ty));
  } else if (auto integer_type = mlir::dyn_cast<mlir::IntegerType>(type)) {
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

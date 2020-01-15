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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project

namespace {
// Returns the shape of the given value if it's ranked; returns llvm::None
// otherwise.
llvm::Optional<llvm::ArrayRef<int64_t>> GetShape(mlir::Value value) {
  auto shaped_type = value->getType().cast<mlir::ShapedType>();
  if (shaped_type.hasRank()) return shaped_type.getShape();
  return llvm::None;
}
}  // namespace

namespace mlir {
namespace TF {
//===----------------------------------------------------------------------===//
// Utility iterators
//===----------------------------------------------------------------------===//

OperandShapeIterator::OperandShapeIterator(Operation::operand_iterator it)
    : llvm::mapped_iterator<Operation::operand_iterator,
                            llvm::Optional<ArrayRef<int64_t>> (*)(Value)>(
          it, &GetShape) {}

ResultShapeIterator::ResultShapeIterator(Operation::result_iterator it)
    : llvm::mapped_iterator<Operation::result_iterator,
                            llvm::Optional<ArrayRef<int64_t>> (*)(Value)>(
          it, &GetShape) {}

//===----------------------------------------------------------------------===//
// TF types helper functions
//===----------------------------------------------------------------------===//

TensorFlowType TensorFlowRefType::get(Type type) {
  MLIRContext* ctx = type.getContext();
  switch (getElementTypeOrSelf(type).getKind()) {
    case StandardTypes::F16:
      return HalfRefType::get(ctx);
    case StandardTypes::F32:
      return FloatRefType::get(ctx);
    case StandardTypes::F64:
      return DoubleRefType::get(ctx);
    case StandardTypes::BF16:
      return Bfloat16RefType::get(ctx);
    case StandardTypes::Complex: {
      const auto& etype = type.cast<ComplexType>().getElementType();
      switch (getElementTypeOrSelf(etype).getKind()) {
        case StandardTypes::F32:
          return Complex64RefType::get(ctx);
        case StandardTypes::F64:
          return Complex128RefType::get(ctx);
        default:
          llvm_unreachable("unexpected complex type");
      }
    }
    case StandardTypes::Integer: {
      const auto& itype = type.cast<IntegerType>();
      switch (itype.getWidth()) {
        case 1:
          return BoolRefType::get(ctx);
        case 8:
          return Int8RefType::get(ctx);
        case 16:
          return Int16RefType::get(ctx);
        case 32:
          return Int32RefType::get(ctx);
        case 64:
          return Int64RefType::get(ctx);
        default:
          llvm_unreachable("unexpected integer type");
      }
    }
#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  case TensorFlowTypes::enumerant:              \
    return tftype##RefType::get(ctx);

#define HANDLE_TF_REF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
    default:
      llvm_unreachable("unexpected type kind");
  }
}

Type TensorFlowRefType::RemoveRef() {
  MLIRContext* ctx = getContext();
  switch (getKind()) {
    case TensorFlowTypes::HALF_REF:
      return mlir::FloatType::getF16(ctx);
    case TensorFlowTypes::FLOAT_REF:
      return mlir::FloatType::getF32(ctx);
    case TensorFlowTypes::DOUBLE_REF:
      return mlir::FloatType::getF64(ctx);
    case TensorFlowTypes::BFLOAT16_REF:
      return mlir::FloatType::getBF16(ctx);
    case TensorFlowTypes::BOOL_REF:
      return mlir::IntegerType::get(1, ctx);
    case TensorFlowTypes::INT8_REF:
      return mlir::IntegerType::get(8, ctx);
    case TensorFlowTypes::INT16_REF:
      return mlir::IntegerType::get(16, ctx);
    case TensorFlowTypes::INT32_REF:
      return mlir::IntegerType::get(32, ctx);
    case TensorFlowTypes::INT64_REF:
      return mlir::IntegerType::get(64, ctx);
    case TensorFlowTypes::COMPLEX64_REF:
      return mlir::ComplexType::get(mlir::FloatType::getF32(ctx));
    case TensorFlowTypes::COMPLEX128_REF:
      return mlir::ComplexType::get(mlir::FloatType::getF64(ctx));
#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  case TensorFlowTypes::enumerant##_REF:        \
    return tftype##Type::get(ctx);

#define HANDLE_TF_REF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
    default:
      llvm_unreachable("unexpected tensorflow ref type kind");
  }
}

Type TensorFlowTypeWithSubtype::RemoveSubtypes() {
  MLIRContext* ctx = getContext();
  switch (getKind()) {
    case TensorFlowTypes::VARIANT:
      return VariantType::get(ctx);
    case TensorFlowTypes::RESOURCE:
      return ResourceType::get(ctx);
    default:
      llvm_unreachable("unexpected tensorflow type with subtypes kind");
  }
}

}  // namespace TF
}  // namespace mlir

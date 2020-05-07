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
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project

namespace {
// Returns the shape of the given value if it's ranked; returns llvm::None
// otherwise.
llvm::Optional<llvm::ArrayRef<int64_t>> GetShape(mlir::Value value) {
  auto shaped_type = value.getType().cast<mlir::ShapedType>();
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
          return itype.isUnsigned() ? TensorFlowType(Uint8RefType::get(ctx))
                                    : Int8RefType::get(ctx);
        case 16:
          return itype.isUnsigned() ? TensorFlowType(Uint16RefType::get(ctx))
                                    : Int16RefType::get(ctx);
        case 32:
          return itype.isUnsigned() ? TensorFlowType(Uint32RefType::get(ctx))
                                    : Int32RefType::get(ctx);
        case 64:
          return itype.isUnsigned() ? TensorFlowType(Uint64RefType::get(ctx))
                                    : Int64RefType::get(ctx);
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
    case TensorFlowTypes::UINT8_REF:
      return mlir::IntegerType::get(8, IntegerType::Unsigned, ctx);
    case TensorFlowTypes::UINT16_REF:
      return mlir::IntegerType::get(16, IntegerType::Unsigned, ctx);
    case TensorFlowTypes::UINT32_REF:
      return mlir::IntegerType::get(32, IntegerType::Unsigned, ctx);
    case TensorFlowTypes::UINT64_REF:
      return mlir::IntegerType::get(64, IntegerType::Unsigned, ctx);
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

ArrayRef<TensorType> TensorFlowTypeWithSubtype::GetSubtypes() {
  switch (getKind()) {
    case TensorFlowTypes::VARIANT:
      return this->cast<VariantType>().getSubtypes();
    case TensorFlowTypes::RESOURCE:
      return this->cast<ResourceType>().getSubtypes();
    default:
      llvm_unreachable("unexpected tensorflow type with subtypes kind");
  }
}

// TODO(jpienaar): BroadcastCompatible and HasCompatibleElementTypes have
// similar structure that could be extracted into helper method.
bool BroadcastCompatible(ArrayRef<Type> lhs, ArrayRef<Type> rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (auto types : llvm::zip(lhs, rhs)) {
    auto lhs_type = std::get<0>(types);
    auto rhs_type = std::get<1>(types);

    // This should be true for all TF ops:
    auto lhs_tt = lhs_type.dyn_cast<TensorType>();
    auto rhs_tt = rhs_type.dyn_cast<TensorType>();
    if (!lhs_tt || !rhs_tt) {
      if (lhs_type != rhs_type) return false;
      continue;
    }

    // Verify matching element types. These should be identical, except for
    // variant type where unknown subtype is considered compatible with all
    // subtypes.
    auto lhs_et = lhs_tt.getElementType();
    auto rhs_et = rhs_tt.getElementType();
    if (lhs_et != rhs_et) {
      // If either does not have subtypes, then the element types don't match.
      auto lhs_wst = lhs_et.dyn_cast<TF::TensorFlowTypeWithSubtype>();
      auto rhs_wst = rhs_et.dyn_cast<TF::TensorFlowTypeWithSubtype>();
      if (!lhs_wst || !rhs_wst) return false;

      // Consider the subtype of variant types.
      auto lhs_wst_st = lhs_wst.GetSubtypes();
      auto rhs_wst_st = rhs_wst.GetSubtypes();
      if (!lhs_wst_st.empty() && !rhs_wst_st.empty()) {
        for (auto subtypes : llvm::zip(lhs_wst_st, rhs_wst_st)) {
          if (!BroadcastCompatible(std::get<0>(subtypes),
                                   std::get<1>(subtypes)))
            return false;
        }
      }
    }

    auto lhs_rt = lhs_type.dyn_cast<RankedTensorType>();
    auto rhs_rt = rhs_type.dyn_cast<RankedTensorType>();
    if (!lhs_rt || !rhs_rt) return true;
    SmallVector<int64_t, 4> shape;
    return OpTrait::util::getBroadcastedShape(lhs_rt.getShape(),
                                              rhs_rt.getShape(), shape);
  }
  return true;
}

bool HasCompatibleElementTypes(Type lhs, Type rhs,
                               bool may_ignore_ref_type_lhs) {
  // Fast path if everything is equal.
  if (lhs == rhs) return true;

  // In TF all values are tensors.
  auto lhs_tt = lhs.cast<TensorType>();
  auto rhs_tt = rhs.cast<TensorType>();

  // Verify matching element types. These should be identical dynamically,
  // so this allows for types not yet fully refined.
  auto lhs_et = lhs_tt.getElementType();
  auto rhs_et = rhs_tt.getElementType();
  if (lhs_et == rhs_et) return true;

  // Remove ref types.
  if (may_ignore_ref_type_lhs) {
    if (auto ref_type = lhs_et.dyn_cast<TF::TensorFlowRefType>()) {
      lhs_et = ref_type.RemoveRef();
      if (lhs_et == rhs_et) return true;
    }
  }

  if (lhs_et.getKind() != rhs_et.getKind()) return false;

  // If either is not type that contain subtypes then the element types don't
  // match.
  auto lhs_wst = lhs_et.dyn_cast<TF::TensorFlowTypeWithSubtype>();
  auto rhs_wst = rhs_et.dyn_cast<TF::TensorFlowTypeWithSubtype>();
  if (!lhs_wst || !rhs_wst) return false;

  // Consider the subtype recursively.
  auto lhs_wst_st = lhs_wst.GetSubtypes();
  auto rhs_wst_st = rhs_wst.GetSubtypes();
  if (lhs_wst_st.empty() || rhs_wst_st.empty()) return true;
  if (lhs_wst_st.size() != rhs_wst_st.size()) return false;
  for (auto subtypes : llvm::zip(lhs_wst_st, rhs_wst_st)) {
    if (!HasCompatibleElementTypes(std::get<0>(subtypes),
                                   std::get<1>(subtypes)))
      return false;
  }
  return true;
}

}  // namespace TF
}  // namespace mlir

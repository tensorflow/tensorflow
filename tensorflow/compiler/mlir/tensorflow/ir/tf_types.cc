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

// Merges cast compatible shapes and returns a more refined shape. The two
// shapes are cast compatible if they have the same rank and at each dimension,
// either both have same size or one of them is dynamic. Returns false if the
// given shapes are not cast compatible. The refined shape is same or more
// precise than the two input shapes.
bool GetCastCompatibleShape(llvm::ArrayRef<int64_t> a_shape,
                            llvm::ArrayRef<int64_t> b_shape,
                            llvm::SmallVectorImpl<int64_t>* refined_shape) {
  if (a_shape.size() != b_shape.size()) return false;
  int64_t rank = a_shape.size();
  refined_shape->reserve(rank);
  for (auto dims : llvm::zip(a_shape, b_shape)) {
    int64_t dim1 = std::get<0>(dims);
    int64_t dim2 = std::get<1>(dims);

    if (mlir::ShapedType::isDynamic(dim1)) {
      refined_shape->push_back(dim2);
      continue;
    }
    if (mlir::ShapedType::isDynamic(dim2)) {
      refined_shape->push_back(dim1);
      continue;
    }
    if (dim1 == dim2) {
      refined_shape->push_back(dim1);
      continue;
    }
    return false;
  }
  return true;
}

// Given two types `a` and `b`, returns a refined type which is cast compatible
// with both `a` and `b` and is equal to or more precise than both of them. It
// returns empty Type if the input types are not cast compatible.
//
// The two types are considered cast compatible if they have dynamically equal
// shapes and element type. For element types that do not have subtypes, they
// must be equal. However for TensorFlow types such as Resource and Variant,
// that also have subtypes, we recursively check for subtype compatibilty for
// Resource types and assume all variant types are cast compatible. If either
// one of `a` or `b` have empty subtypes, they are considered cast compatible.
//
// The returned type is same or more precise than the input types. For example,
// if `a` and `b` are cast compatible types tensor<2x?x?xf32> and
// tensor<?x4x?xf32> respectively, the returned type is tensor<2x4x?xf32>.
//
// Provides option to ignore ref types on 'a'. This is useful for TF ops that
// might allow operands to either be same as result type or be a ref type
// corresponding to it.
mlir::Type GetCastCompatibleType(mlir::Type a, mlir::Type b,
                                 bool may_ignore_ref_type_a) {
  // Fast path if everything is equal.
  if (a == b) return b;

  auto a_tt = a.dyn_cast<mlir::TensorType>();
  auto b_tt = b.dyn_cast<mlir::TensorType>();

  // If only one of a or b is a tensor type, they are incompatible.
  if (static_cast<bool>(a_tt) ^ static_cast<bool>(b_tt)) return nullptr;

  // For non-tensor types, we do not need to worry about shape and can return
  // early.
  if (!a_tt && !b_tt) {
    // Remove ref types.
    if (may_ignore_ref_type_a) {
      if (auto ref_type = a.dyn_cast<mlir::TF::TensorFlowRefType>()) {
        a = ref_type.RemoveRef();
        if (a == b) return a;
      }
    }
    if (a.getKind() != b.getKind()) return nullptr;

    // If either is not a type that contain subtypes then the types are not cast
    // compatible.
    auto a_wst = a.dyn_cast<mlir::TF::TensorFlowTypeWithSubtype>();
    auto b_wst = b.dyn_cast<mlir::TF::TensorFlowTypeWithSubtype>();
    if (!a_wst || !b_wst) return nullptr;

    // For Variant types we are more permissive right now and accept all pairs
    // of Variant types. If we are more constrainted and check compatibility of
    // subtypes, we might reject valid graphs.
    // TODO(prakalps): Variant doesn't have a subtype, we assign it
    // one, so we should only assign it one when we know the subtype. Then we
    // can be more constrained and check subtypes for cast compatibility as
    // well.
    if (a.isa<mlir::TF::VariantType>()) return a;

    // For Resource types, we recursively check the subtypes for cast
    // compatibility, if possible. Otherwise treat them as compatible.
    auto a_wst_st = a_wst.GetSubtypes();
    auto b_wst_st = b_wst.GetSubtypes();
    if (a_wst_st.empty() || b_wst_st.empty()) return a;
    if (a_wst_st.size() != b_wst_st.size()) return nullptr;
    llvm::SmallVector<mlir::TensorType, 4> refined_subtypes;
    for (auto subtypes : llvm::zip(a_wst_st, b_wst_st)) {
      mlir::Type refined_st =
          GetCastCompatibleType(std::get<0>(subtypes), std::get<1>(subtypes),
                                /*may_ignore_ref_type_a=*/false);
      if (!refined_st) return nullptr;
      refined_subtypes.push_back(refined_st.cast<mlir::TensorType>());
    }

    return mlir::TF::ResourceType::get(refined_subtypes, a.getContext());
  }

  // For tensor types, check compatibility of both element type and shape.
  mlir::Type refined_element_ty = GetCastCompatibleType(
      a_tt.getElementType(), b_tt.getElementType(), may_ignore_ref_type_a);
  if (!refined_element_ty) return nullptr;

  if (!a_tt.hasRank() && !b_tt.hasRank()) {
    return mlir::UnrankedTensorType::get(refined_element_ty);
  }
  if (!a_tt.hasRank()) {
    return mlir::RankedTensorType::get(b_tt.getShape(), refined_element_ty);
  }
  if (!b_tt.hasRank()) {
    return mlir::RankedTensorType::get(a_tt.getShape(), refined_element_ty);
  }

  llvm::SmallVector<int64_t, 8> refined_shape;
  if (!GetCastCompatibleShape(a_tt.getShape(), b_tt.getShape(), &refined_shape))
    return nullptr;

  return mlir::RankedTensorType::get(refined_shape, refined_element_ty);
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
  return GetCastCompatibleType(lhs, rhs, may_ignore_ref_type_lhs) != nullptr;
}

bool AreCastCompatible(ArrayRef<Type> types) {
  Type common = types.front();
  for (auto type : types.drop_front()) {
    Type refined_type =
        GetCastCompatibleType(common, type, /*may_ignore_ref_type_a=*/false);
    if (!refined_type) return false;
    common = refined_type;
  }
  return true;
}

ShapedType DropTypeSubTypes(ShapedType ty) {
  Type element_ty = ty.getElementType();
  auto subtype_ty = element_ty.dyn_cast<TF::TensorFlowTypeWithSubtype>();
  if (!subtype_ty) return ty;

  Type default_ty = GetDefaultTypeOf(subtype_ty);
  if (ty.hasRank()) return RankedTensorType::get(ty.getShape(), default_ty);

  return UnrankedTensorType::get(default_ty);
}

ShapedType DropRefType(ShapedType ty) {
  Type element_ty = ty.getElementType();
  TF::TensorFlowRefType ref_ty = element_ty.dyn_cast<TF::TensorFlowRefType>();
  if (!ref_ty) return ty;

  Type default_ty = TF::GetDefaultTypeOf(ref_ty);
  if (ty.hasRank()) return RankedTensorType::get(ty.getShape(), default_ty);

  return UnrankedTensorType::get(default_ty);
}

}  // namespace TF
}  // namespace mlir

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
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

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

bool TensorFlowType::classof(Type type) {
  return type.getDialect().getNamespace() == "tf";
}
bool TensorFlowRefType::classof(Type type) {
  return type.isa<
#define HANDLE_TF_TYPE(tftype, enumerant, name)
#define HANDLE_TF_REF_TYPE(tftype, enumerant, name) tftype##Type,
#define HANDLE_LAST_TF_TYPE(tftype, enumerant, name) tftype##Type
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
      >();
}

TensorFlowType TensorFlowRefType::get(Type type) {
  MLIRContext* ctx = type.getContext();
  type = getElementTypeOrSelf(type);
  if (type.isF16()) {
    return HalfRefType::get(ctx);
  } else if (type.isF32()) {
    return FloatRefType::get(ctx);
  } else if (type.isF64()) {
    return DoubleRefType::get(ctx);
  } else if (type.isBF16()) {
    return Bfloat16RefType::get(ctx);
  } else if (auto complex_type = type.dyn_cast<ComplexType>()) {
    Type etype = complex_type.getElementType();
    if (etype.isF32()) {
      return Complex64RefType::get(ctx);
    } else if (etype.isF64()) {
      return Complex128RefType::get(ctx);
    }
    llvm_unreachable("unexpected complex type");
  } else if (auto itype = type.dyn_cast<IntegerType>()) {
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
#define HANDLE_TF_TYPE(tftype, enumerant, name)        \
  if (auto derived_ty = type.dyn_cast<tftype##Type>()) \
    return tftype##RefType::get(ctx);

#define HANDLE_TF_REF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
  llvm_unreachable("unexpected type kind");
}

Type TensorFlowRefType::RemoveRef() {
  MLIRContext* ctx = getContext();
  if (isa<HalfRefType>()) return mlir::FloatType::getF16(ctx);
  if (isa<FloatRefType>()) return mlir::FloatType::getF32(ctx);
  if (isa<DoubleRefType>()) return mlir::FloatType::getF64(ctx);
  if (isa<Bfloat16RefType>()) return mlir::FloatType::getBF16(ctx);
  if (isa<BoolRefType>()) return mlir::IntegerType::get(ctx, 1);
  if (isa<Int8RefType>()) return mlir::IntegerType::get(ctx, 8);
  if (isa<Int16RefType>()) return mlir::IntegerType::get(ctx, 16);
  if (isa<Int32RefType>()) return mlir::IntegerType::get(ctx, 32);
  if (isa<Int64RefType>()) return mlir::IntegerType::get(ctx, 64);
  if (isa<Uint8RefType>())
    return mlir::IntegerType::get(ctx, 8, IntegerType::Unsigned);
  if (isa<Uint16RefType>())
    return mlir::IntegerType::get(ctx, 16, IntegerType::Unsigned);
  if (isa<Uint32RefType>())
    return mlir::IntegerType::get(ctx, 32, IntegerType::Unsigned);
  if (isa<Uint64RefType>())
    return mlir::IntegerType::get(ctx, 64, IntegerType::Unsigned);
  if (isa<Complex64RefType>())
    return mlir::ComplexType::get(mlir::FloatType::getF32(ctx));
  if (isa<Complex128RefType>())
    return mlir::ComplexType::get(mlir::FloatType::getF64(ctx));
#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  if (isa<tftype##RefType>()) return tftype##Type::get(ctx);

#define HANDLE_TF_REF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
  llvm_unreachable("unexpected tensorflow ref type kind");
}

bool TensorFlowTypeWithSubtype::classof(Type type) {
  return type.isa<ResourceType, VariantType>();
}

Type TensorFlowTypeWithSubtype::RemoveSubtypes() {
  MLIRContext* ctx = getContext();
  if (isa<VariantType>()) return VariantType::get(ctx);
  if (isa<ResourceType>()) return ResourceType::get(ctx);
  llvm_unreachable("unexpected tensorflow type with subtypes kind");
}

TensorFlowTypeWithSubtype TensorFlowTypeWithSubtype::clone(
    ArrayRef<TensorType> new_subtypes) {
  MLIRContext* ctx = getContext();
  if (isa<VariantType>())
    return VariantType::get(new_subtypes, ctx)
        .cast<TensorFlowTypeWithSubtype>();
  if (isa<ResourceType>())
    return ResourceType::get(new_subtypes, ctx)
        .cast<TensorFlowTypeWithSubtype>();
  llvm_unreachable("unexpected tensorflow type with subtypes kind");
}

ArrayRef<TensorType> TensorFlowTypeWithSubtype::GetSubtypes() {
  if (auto variant_type = dyn_cast<VariantType>())
    return variant_type.getSubtypes();
  if (auto resource_type = dyn_cast<ResourceType>())
    return resource_type.getSubtypes();
  llvm_unreachable("unexpected tensorflow type with subtypes kind");
}

// TODO(jpienaar): BroadcastCompatible and HasCompatibleElementTypes have
// similar structure that could be extracted into helper method.
bool BroadcastCompatible(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (auto types : llvm::zip(lhs, rhs)) {
    // Drop ref types because they don't affect broadcast compatibility. E.g.,
    // `tensor<!tf.f32ref>` and `tensor<f32>` should be considered broadcast
    // compatible.
    auto lhs_type = DropRefType(std::get<0>(types));
    auto rhs_type = DropRefType(std::get<1>(types));

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
    if (a.getTypeID() != b.getTypeID()) return nullptr;

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

bool HasCompatibleElementTypes(Type lhs, Type rhs,
                               bool may_ignore_ref_type_lhs) {
  return GetCastCompatibleType(lhs, rhs, may_ignore_ref_type_lhs) != nullptr;
}

bool AreCastCompatible(TypeRange types) {
  Type common = types.front();
  for (auto type : types.drop_front()) {
    Type refined_type =
        GetCastCompatibleType(common, type, /*may_ignore_ref_type_a=*/false);
    if (!refined_type) return false;
    common = refined_type;
  }
  return true;
}

bool ArraysAreCastCompatible(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (auto pair : llvm::zip(lhs, rhs)) {
    auto lhs_i = std::get<0>(pair);
    auto rhs_i = std::get<1>(pair);
    if (!AreCastCompatible({lhs_i, rhs_i})) return false;
  }
  return true;
}

// Assumes a function `GetDefaultTypeOf(ComposedType)` that returns the default
// type for a composed type (such as a ref type or a type with subtypes).
template <typename ComposedType>
Type DropTypeHelper(Type ty) {
  Type element_ty = getElementTypeOrSelf(ty);
  auto composed_type = element_ty.dyn_cast<ComposedType>();
  if (!composed_type) return ty;

  Type default_ty = GetDefaultTypeOf(composed_type);
  if (auto ranked_ty = ty.dyn_cast<RankedTensorType>()) {
    return RankedTensorType::get(ranked_ty.getShape(), default_ty);
  } else if (ty.dyn_cast<UnrankedTensorType>()) {
    return UnrankedTensorType::get(default_ty);
  } else {
    return default_ty;
  }
}

Type DropSubTypes(Type ty) {
  return DropTypeHelper<TF::TensorFlowTypeWithSubtype>(ty);
}

Type DropRefType(Type ty) { return DropTypeHelper<TF::TensorFlowRefType>(ty); }

Type DropRefAndSubTypes(Type ty) { return DropRefType(DropSubTypes(ty)); }

void TensorFlowDialect::registerTypes() {
  addTypes<
#define HANDLE_TF_TYPE(tftype, enumerant, name) tftype##Type,
#define HANDLE_LAST_TF_TYPE(tftype, enumerant, name) tftype##Type
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
      >();
}

}  // namespace TF
}  // namespace mlir

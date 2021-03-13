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

// This file defines the types used in the standard MLIR TensorFlow dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TYPES_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TYPES_H_

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project

namespace mlir {
namespace TF {
//===----------------------------------------------------------------------===//
// Utility iterators
//===----------------------------------------------------------------------===//

// An iterator for the tensor shapes of an op's operands of shaped types.
// Returns llvm::None if a operand is unranked; returns ArrayRef<int64_t> as the
// shape otherwise.
class OperandShapeIterator final
    : public llvm::mapped_iterator<Operation::operand_iterator,
                                   llvm::Optional<ArrayRef<int64_t>> (*)(
                                       Value)> {
 public:
  using reference = llvm::Optional<ArrayRef<int64_t>>;

  /// Initializes the operand shape iterator to the specified operand iterator.
  explicit OperandShapeIterator(Operation::operand_iterator it);
};

using OperandShapeRange = iterator_range<OperandShapeIterator>;

// An iterator for the tensor shapes of an op's results of shaped types.
// Returns llvm::None if a result is unranked; returns ArrayRef<int64_t> as the
// shape otherwise.
class ResultShapeIterator final
    : public llvm::mapped_iterator<Operation::result_iterator,
                                   llvm::Optional<ArrayRef<int64_t>> (*)(
                                       Value)> {
 public:
  using reference = llvm::Optional<ArrayRef<int64_t>>;

  /// Initializes the result shape iterator to the specified result iterator.
  explicit ResultShapeIterator(Operation::result_iterator it);
};

using ResultShapeRange = iterator_range<ResultShapeIterator>;

//===----------------------------------------------------------------------===//
// TensorFlow types
//===----------------------------------------------------------------------===//

// The base class in the TensorFlow type hierarchy.
class TensorFlowType : public Type {
 public:
  using Type::Type;

  // Support method to enable LLVM-style type casting.
  static bool classof(Type type);
};

// Returns true if the specified type is a valid TensorFlow element type.
static inline bool IsValidTFElementType(Type type) {
  return type.isa<ComplexType, FloatType, IntegerType, TensorFlowType>();
}

// Returns true if this is a valid TensorFlow tensor type.
static inline bool IsValidTFTensorType(Type type) {
  // TensorFlow types should be tensors of one of the valid TensorFlow element
  // types.
  if (auto tensor_ty = type.dyn_cast<TensorType>())
    return IsValidTFElementType(tensor_ty.getElementType());
  return false;
}

namespace detail {
// Common implementation of TensorFlow types. The template argument indicates
// the concrete derived class per CRTP.
template <typename Derived>
class TensorFlowTypeImpl
    : public Type::TypeBase<Derived, TensorFlowType, TypeStorage> {
 public:
  using Base = typename Type::TypeBase<Derived, TensorFlowType, TypeStorage>;
  using TFBase = TensorFlowTypeImpl<Derived>;
  using Base::Base;
};
}  // namespace detail

// TensorFlowRefType class supports all the ref types in TensorFlow dialect.
class TensorFlowRefType : public TensorFlowType {
 public:
  using TensorFlowType::TensorFlowType;

  // Checks if a type is TensorFlow Ref type.
  static bool classof(Type type);

  // Converts a type to the corresponding TensorFlowRef type.
  static TensorFlowType get(Type type);
  static TensorFlowType getChecked(Type type, MLIRContext* context,
                                   Location loc) {
    if (failed(verify(loc, type))) {
      return TensorFlowRefType();
    }
    return get(type);
  }

  static LogicalResult verify(Location loc, Type type) {
    // type should be a valid TensorFlow type.
    if (!IsValidTFTensorType(type)) {
      return emitError(loc) << "invalid TensorFlow type: " << type;
    }
    return success();
  }

  // Converts a TensorFlowRef type to the corresponding TensorFlow or standard
  // type.
  Type RemoveRef();
};

// Returns the corresponding TensorFlow or standard type from TensorFlowRef
// type.
static inline Type GetDefaultTypeOf(TensorFlowRefType type) {
  return type.RemoveRef();
}

// Returns the element type if `type` is a `ShapedType` and the type itself
// otherwise, converting `TensorFlowRef` type to corresponding `TensorFlow` or
// standard type if necessary.
static inline Type GetElementTypeOrSelfResolveRef(Type type) {
  Type element_type = mlir::getElementTypeOrSelf(type);
  if (auto ref_type = element_type.dyn_cast<mlir::TF::TensorFlowRefType>()) {
    element_type = ref_type.RemoveRef();
  }
  return element_type;
}

#define HANDLE_TF_TYPE(tftype, enumerant, name)                          \
  class tftype##Type : public detail::TensorFlowTypeImpl<tftype##Type> { \
   public:                                                               \
    using TFBase::TFBase;                                                \
  };

// Custom TensorFlow types are defined separately.
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name)

// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"

namespace detail {
// Storage type contains inferred subtypes for TypeWithSubtype.
class TypeWithSubtypeStorage : public TypeStorage {
 public:
  using KeyTy = ArrayRef<TensorType>;

  // NOLINTNEXTLINE
  static TypeWithSubtypeStorage* construct(TypeStorageAllocator& allocator,
                                           const KeyTy& key) {
    ArrayRef<TensorType> subtypes = allocator.copyInto(key);
    return new (allocator.allocate<TypeWithSubtypeStorage>())
        TypeWithSubtypeStorage(subtypes);
  }

  explicit TypeWithSubtypeStorage(const KeyTy& key) : subtypes_(key) {}

  bool operator==(const KeyTy& key) const { return key == subtypes_; }

  static llvm::hash_code hashKey(const KeyTy& key) {
    return llvm::hash_combine_range(key.begin(), key.end());
  }

  KeyTy subtypes_;
};

// Common implementation of TensorFlow types with subtypes. These subtypes are
// opaque and their interpretation depends on the actual underlying type.
// The template argument indicates the concrete derived class per CRTP. Concrete
// classes must implement the following:
//   - `static std::string getTypeName()` that returns the name of the type for
//     verification logging.
template <typename Derived>
class TypeWithSubtypeImpl
    : public Type::TypeBase<Derived, TensorFlowType, TypeWithSubtypeStorage> {
 public:
  using Base = Type::TypeBase<Derived, TensorFlowType, TypeWithSubtypeStorage>;
  using TFBase = TypeWithSubtypeImpl<Derived>;
  using Base::Base;

  static Derived get(ArrayRef<TensorType> subtypes, MLIRContext* context) {
    return Base::get(context, subtypes);
  }

  static Derived getChecked(ArrayRef<TensorType> subtypes, MLIRContext* context,
                            Location loc) {
    return Base::getChecked(loc, subtypes);
  }
  static Derived getChecked(function_ref<InFlightDiagnostic()> emitError,
                            MLIRContext* context,
                            ArrayRef<TensorType> subtypes) {
    return Base::getChecked(emitError, context, subtypes);
  }

  static Derived get(MLIRContext* context) { return get({}, context); }

  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<TensorType> subtypes) {
    // Each of the subtypes should be a valid TensorFlow type.
    for (TensorType subtype : subtypes) {
      if (!IsValidTFTensorType(subtype)) {
        return emitError() << "invalid " << Derived::getTypeName()
                           << " subtype: " << subtype;
      }
    }
    return success();
  }

  ArrayRef<TensorType> getSubtypes() { return Base::getImpl()->subtypes_; }
};
}  // namespace detail

// TensorFlowTypeWithSubtype class supports all the types with subtypes in
// TensorFlow dialect.
class TensorFlowTypeWithSubtype : public TensorFlowType {
 public:
  using TensorFlowType::TensorFlowType;

  // Checks if a type is TensorFlow type with subtypes.
  static bool classof(Type type);

  // Converts a TypeWithSubtype type to the same type but without its subtypes.
  Type RemoveSubtypes();

  // Clone the current Type with new subtypes.
  TensorFlowTypeWithSubtype clone(ArrayRef<TensorType> new_subtypes);

  // Returns the subtypes.
  ArrayRef<TensorType> GetSubtypes();
};

// Returns the corresponding TensorFlow type with subtypes but without its
// subtypes.
static inline Type GetDefaultTypeOf(TensorFlowTypeWithSubtype type) {
  return type.RemoveSubtypes();
}

// TensorFlow resource type is used to support TensorFlow resource variables,
// which represent shared, persistent state manipulated by a TensorFlow program.
// ResourceType stores shape and datatype for subtypes unlike most other data
// types that don't have any associated information.
class ResourceType : public detail::TypeWithSubtypeImpl<ResourceType> {
 public:
  using TFBase::TFBase;
  static std::string getTypeName() { return "ResourceType"; }
};

// TensorFlow variant type is used to support arbitrary custom C++ data types.
// VariantType stores inferred shape and datatype for subtypes unlike most other
// data types that don't have any associated information. For example, variants
// encoding TensorList type stores the common shape and dtype of the list
// elements as the only subtype.
class VariantType : public detail::TypeWithSubtypeImpl<VariantType> {
 public:
  using TFBase::TFBase;
  static std::string getTypeName() { return "VariantType"; }
};

// Given two types `a` and `b`, returns a refined type which is cast compatible
// with both `a` and `b` and is equal to or more precise than both of them. It
// returns empty Type if the input types are not cast compatible.
// Provides option to ignore ref types on 'a'. This is useful for TF ops that
// might allow operands to either be same as result type or be a ref type
// corresponding to it.
mlir::Type GetCastCompatibleType(mlir::Type a, mlir::Type b,
                                 bool may_ignore_ref_type_a);

// Returns whether two arrays of Type are broadcast compatible.
bool BroadcastCompatible(TypeRange lhs, TypeRange rhs);

// Returns whether the two elemental types are compatible. Shapes are compatible
// if:
// - the types are statically equal
// - could be dynamically equal
//   - considering dynamic shapes equal unless contradictory info known;
//   - element types are equivalent, modulo subtypes possible be less exact
//     (e.g., a resource type without subtype is considered compatible with
//      resource type with known subtype).
// Provide option to ignore ref types on 'lhs'.
bool HasCompatibleElementTypes(Type lhs, Type rhs,
                               bool may_ignore_ref_type_lhs = false);

// Returns true if all TensorFlow types can be cast to one
// another. In other words, a single run-time value is legal for both the types.
// For example, tensor<*xf32>, tensor<?xf32> and tensor<3xf32> are cast
// compatible.
bool AreCastCompatible(TypeRange types);

// Returns true if corresponding elements of lhs and rhs AreCastCompatible and
// lhs and rhs are the same length.
bool ArraysAreCastCompatible(TypeRange lhs, TypeRange rhs);

// If `ty` is a tensor type and its element type has subtypes, then returns a
// new type of same shape but dropped subtypes for the element type.
// Otherwise, if `ty` has subtypes, then returns corresponding type with dropped
// subtypes.
// Otherwise, returns the original type `ty`.
Type DropSubTypes(Type ty);

// If `ty` is a tensor type and has elements of a ref type, then returns a new
// type of same shape but corresponding non-ref type as element type.
// Otherwise, if `ty` is a ref type, then returns corresponding non-ref type.
// Otherwise, returns the original type `ty`.
Type DropRefType(Type ty);

// Convenience call for executing both `DropRefType` and `DropSubTypes`.
Type DropRefAndSubTypes(Type ty);

}  // end namespace TF
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TYPES_H_

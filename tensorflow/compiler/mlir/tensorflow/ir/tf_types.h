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

#include "mlir/IR/Diagnostics.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir

namespace mlir {
namespace TF {

namespace TensorFlowTypes {
// List of supported TensorFlowType kinds, necessary for isa/dyn_cast.
enum Kind {
  FIRST_USED_TENSORFLOW_TYPE = Type::FIRST_TENSORFLOW_TYPE,
#define HANDLE_TF_TYPE(tftype, enumerant, name) enumerant,
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
  LAST_USED_TENSORFLOW_TYPE,
};
}  // namespace TensorFlowTypes

// The base class in the tensor flow type hierarchy.
class TensorFlowType : public Type {
 public:
  using Type::Type;

  // Support method to enable LLVM-style type casting.
  static bool classof(Type type) {
    return type.getKind() >= Type::FIRST_TENSORFLOW_TYPE &&
           type.getKind() <= TensorFlowTypes::LAST_USED_TENSORFLOW_TYPE;
  }
};

// Returns true if the specified type is a valid TensorFlow element type.
static inline bool IsValidTFElementType(Type type) {
  return type.isa<FloatType>() || type.isa<IntegerType>() ||
         type.isa<TensorFlowType>();
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
// Common implementation of TensorFlow types.  The template argument indicates
// the concrete derived class per CRTP.  Concrete classes must implement the
// following:
//   - `static unsigned getTypeKind()` that returns the (fixed) kind of the
//     type.
template <typename Derived>
class TensorFlowTypeImpl : public Type::TypeBase<Derived, TensorFlowType> {
 public:
  using Base = typename Type::TypeBase<Derived, TensorFlowType>;
  using TFBase = TensorFlowTypeImpl<Derived>;
  using Base::Base;

  // Get the unique'ed type in the given context.
  static Derived get(MLIRContext *context) {
    return Base::get(context, Derived::getTypeKind());
  }

  // Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) { return kind == Derived::getTypeKind(); }
};
}  // namespace detail

#define HANDLE_TF_TYPE(tftype, enumerant, name)                          \
  class tftype##Type : public detail::TensorFlowTypeImpl<tftype##Type> { \
   public:                                                               \
    using TFBase::TFBase;                                                \
    static unsigned getTypeKind() { return TensorFlowTypes::enumerant; } \
  };

// Custom TensorFlow types are defined separately.
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name)

// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"

// Storage type contains inferred subtypes for VariantType.
class VariantTypeStorage : public TypeStorage {
 public:
  using KeyTy = ArrayRef<TensorType>;

  // NOLINTNEXTLINE
  static VariantTypeStorage* construct(TypeStorageAllocator& allocator,
                                       const KeyTy& key) {
    ArrayRef<TensorType> subtypes = allocator.copyInto(key);
    return new (allocator.allocate<VariantTypeStorage>())
        VariantTypeStorage(subtypes);
  }

  explicit VariantTypeStorage(const KeyTy& key) : subtypes_(key) {}

  bool operator==(const KeyTy& key) const { return key == subtypes_; }

  static llvm::hash_code hashKey(const KeyTy& key) {
    return llvm::hash_combine_range(key.begin(), key.end());
  }

  KeyTy subtypes_;
};

// TensorFlow variant type is used to support arbitrary custom C++ data types.
// VariantType stores inferred shape and datatype for subtypes unlike most other
// data types don't have any associated information. These subtypes are opaque
// and their interpretation depends on the actual underlying type. For example,
// variants encoding TensorList type stores the common shape and dtype of the
// list elements as the only subtype.
class VariantType
    : public Type::TypeBase<VariantType, TensorFlowType, VariantTypeStorage> {
 public:
  using Base::Base;

  static VariantType get(ArrayRef<TensorType> subtypes, MLIRContext* context) {
    return Base::get(context, TensorFlowTypes::VARIANT, subtypes);
  }

  static VariantType getChecked(ArrayRef<TensorType> subtypes,
                                MLIRContext* context, Location loc) {
    return Base::getChecked(loc, context, TensorFlowTypes::VARIANT, subtypes);
  }

  static VariantType get(MLIRContext* context) { return get({}, context); }

  static bool kindof(unsigned kind) { return kind == TensorFlowTypes::VARIANT; }

  static LogicalResult verifyConstructionInvariants(
      llvm::Optional<Location> loc, MLIRContext* context,
      ArrayRef<TensorType> subtypes) {
    // Each of the subtypes should be a valid TensorFlow type.
    for (TensorType subtype : subtypes) {
      if (!IsValidTFTensorType(subtype)) {
        if (loc) {
          emitError(*loc) << "invalid VariantType subtype: " << subtype;
        }
        return failure();
      }
    }
    return success();
  }

  ArrayRef<TensorType> getSubtypes() { return getImpl()->subtypes_; }
};

}  // end namespace TF
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TYPES_H_

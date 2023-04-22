/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TFR_IR_TFR_TYPES_H_
#define TENSORFLOW_COMPILER_MLIR_TFR_IR_TFR_TYPES_H_

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project

namespace mlir {
namespace TFR {

class TFRType : public Type {
 public:
  using Type::Type;

  static bool classof(Type type);
};

namespace detail {

struct TFRTypeStorage final
    : public TypeStorage,
      public llvm::TrailingObjects<TFRTypeStorage, StringAttr> {
  using KeyTy = ArrayRef<StringAttr>;

  explicit TFRTypeStorage(unsigned num_attrs) : num_attrs(num_attrs) {}

  static TFRTypeStorage* construct(TypeStorageAllocator& allocator, KeyTy key) {
    // Allocate a new storage instance.
    auto byteSize = TFRTypeStorage::totalSizeToAlloc<StringAttr>(key.size());
    auto rawMem = allocator.allocate(byteSize, alignof(TFRTypeStorage));
    auto result = ::new (rawMem) TFRTypeStorage(key.size());

    // Copy in the string attributes into the trailing storage.
    std::uninitialized_copy(key.begin(), key.end(),
                            result->getTrailingObjects<StringAttr>());
    return result;
  }

  bool operator==(const KeyTy& attrs) const { return attrs == GetAttrs(); }

  KeyTy GetAttrs() const {
    return {getTrailingObjects<StringAttr>(), num_attrs};
  }

  unsigned num_attrs;
};

template <typename Derived>
class TFRTypeImpl : public Type::TypeBase<Derived, TFRType, TFRTypeStorage> {
 public:
  using Base = Type::TypeBase<Derived, TFRType, TFRTypeStorage>;
  using TFRBase = TFRTypeImpl<Derived>;
  using Base::Base;

  static Derived get(ArrayRef<StringAttr> attrs, MLIRContext* context) {
    return Base::get(context, attrs);
  }

  static Derived getChecked(ArrayRef<StringAttr> attrs, Location loc) {
    return Base::getChecked(loc, loc.getContext(), attrs);
  }
  static Derived getChecked(function_ref<InFlightDiagnostic()> emitError,
                            MLIRContext* context, ArrayRef<StringAttr> attrs) {
    return Base::getChecked(emitError, context, attrs);
  }

  static Derived get(MLIRContext* context) { return get({}, context); }

  // TODO(fengliuai): fix the implementation
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<StringAttr> attrs) {
    return success();
  }

  ArrayRef<StringAttr> getAttrKeys() { return Base::getImpl()->GetAttrs(); }
};
}  // namespace detail

class TFRTensorType : public detail::TFRTypeImpl<TFRTensorType> {
 public:
  using TFRBase::TFRBase;
  static std::string getTypeName() { return "TFRTensorType"; }
};

class TFRTensorListType : public detail::TFRTypeImpl<TFRTensorListType> {
 public:
  using TFRBase::TFRBase;
  static std::string getTypeName() { return "TFRTensorListType"; }
};

class TFRAttrType : public Type::TypeBase<TFRAttrType, TFRType, TypeStorage> {
 public:
  using Base::Base;
  static std::string getTypeName() { return "TFRAttrType"; }
};

}  // namespace TFR
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TFR_IR_TFR_TYPES_H_

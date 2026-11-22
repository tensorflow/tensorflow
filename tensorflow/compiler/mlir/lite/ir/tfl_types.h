/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Element types owned by the TFLite dialect: `!tfl.string`, `!tfl.quint8`,
// `!tfl.resource` and `!tfl.variant`.
//
// The TFLite op type constraints accept both these types and the matching
// `tf_type` types. The `tf_type` types are matched by their registered name, so
// the TFLite dialect neither includes nor links the `tf_type` dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_TYPES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_TYPES_H_

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/TypeSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TFL {

// A variable-length string. Corresponds to `tflite::TensorType_STRING`.
//
// Syntax: `!tfl.string`
class StringType : public Type::TypeBase<StringType, Type, TypeStorage> {
 public:
  using Base::Base;
  static constexpr StringLiteral name = "tfl.string";
};

// An 8-bit quantized unsigned integer without quantization parameters.
// Corresponds to `tflite::TensorType_UINT8`.
//
// Syntax: `!tfl.quint8`
class Quint8Type : public Type::TypeBase<Quint8Type, Type, TypeStorage> {
 public:
  using Base::Base;
  static constexpr StringLiteral name = "tfl.quint8";
};

namespace detail {

// Storage of a type with optional tensor subtypes.
struct TypeWithSubtypesStorage : public TypeStorage {
  using KeyTy = ArrayRef<TensorType>;

  explicit TypeWithSubtypesStorage(ArrayRef<TensorType> subtypes)
      : subtypes(subtypes) {}

  bool operator==(const KeyTy& key) const { return key == subtypes; }

  static llvm::hash_code hashKey(const KeyTy& key) {
    return llvm::hash_combine_range(key.begin(), key.end());
  }

  static TypeWithSubtypesStorage* construct(TypeStorageAllocator& allocator,
                                            const KeyTy& key) {
    return new (allocator.allocate<TypeWithSubtypesStorage>())
        TypeWithSubtypesStorage(allocator.copyInto(key));
  }

  ArrayRef<TensorType> subtypes;
};

// Common implementation of the types with optional tensor subtypes. The
// subtypes are opaque; their meaning depends on the concrete type.
template <typename Derived>
class TypeWithSubtypesImpl
    : public Type::TypeBase<Derived, Type, TypeWithSubtypesStorage> {
 public:
  using Base = Type::TypeBase<Derived, Type, TypeWithSubtypesStorage>;
  using Base::Base;

  static Derived get(MLIRContext* context, ArrayRef<TensorType> subtypes = {}) {
    return Base::get(context, subtypes);
  }

  ArrayRef<TensorType> getSubtypes() const { return Base::getImpl()->subtypes; }
};

}  // namespace detail

// A handle to a resource, such as a variable or a hash table. The optional
// subtypes describe the resource, e.g. the type of a variable. Corresponds to
// `tflite::TensorType_RESOURCE`.
//
// Syntax: `!tfl.resource` or `!tfl.resource<tensor<...>, ...>`
class ResourceType : public detail::TypeWithSubtypesImpl<ResourceType> {
 public:
  using TypeWithSubtypesImpl::TypeWithSubtypesImpl;
  static constexpr StringLiteral name = "tfl.resource";
};

// An opaque value, such as a tensor list. The optional subtypes describe the
// value, e.g. the element type of a tensor list. Corresponds to
// `tflite::TensorType_VARIANT`.
//
// Syntax: `!tfl.variant` or `!tfl.variant<tensor<...>, ...>`
class VariantType : public detail::TypeWithSubtypesImpl<VariantType> {
 public:
  using TypeWithSubtypesImpl::TypeWithSubtypesImpl;
  static constexpr StringLiteral name = "tfl.variant";
};

//===----------------------------------------------------------------------===//
// Matching of the TFLite types and the corresponding `tf_type` types.
//===----------------------------------------------------------------------===//

// Returns true if `type` is the `tf_type` type with the given mnemonic (e.g.
// "string" for `!tf_type.string`). Matches by the registered type name, so it
// needs neither the `tf_type` headers nor the `tf_type` library.
bool IsTfTypeWithMnemonic(Type type, StringRef mnemonic);

// Returns true if `type` is `!tfl.string` or `!tf_type.string`.
bool IsStringType(Type type);

// Returns true if `type` is `!tfl.quint8` or `!tf_type.quint8`.
bool IsQuint8Type(Type type);

// Returns true if `type` is a quantized unsigned integer type of `width` bits
// without quantization parameters: `!tfl.quint8`, `!tf_type.quint8` or
// `!tf_type.quint16`.
bool IsQuintType(Type type, unsigned width);

// Returns true if `type` is `!tfl.resource` or `!tf_type.resource`, with or
// without subtypes.
bool IsResourceType(Type type);

// Returns true if `type` is `!tfl.variant` or `!tf_type.variant`, with or
// without subtypes.
bool IsVariantType(Type type);

//===----------------------------------------------------------------------===//
// Cast compatibility.
//===----------------------------------------------------------------------===//

// Returns true if a single run-time value is legal for all `types`: the shapes
// are compatible (e.g. `tensor<*xf32>`, `tensor<?xf32>` and `tensor<3xf32>`)
// and the element types are equal. Resource (and variant) types of the same
// kind are compatible regardless of their subtypes. A TFLite type and the
// matching `tf_type` type (e.g. `!tfl.resource` and `!tf_type.resource`) are
// not cast compatible. Reference types are not resolved.
bool AreCastCompatible(TypeRange types);

// Returns true if `lhs` and `rhs` have the same length and their corresponding
// elements are cast compatible.
bool ArraysAreCastCompatible(TypeRange lhs, TypeRange rhs);

//===----------------------------------------------------------------------===//
// Parsing and printing, used by the TFLite dialect.
//===----------------------------------------------------------------------===//

// Parses the rest of the TFLite type named `mnemonic`, after the mnemonic has
// been parsed. Returns std::nullopt if `mnemonic` doesn't name one of the types
// declared in this file, and a null type (after emitting a diagnostic) on a
// parse error.
std::optional<Type> ParseTflType(MLIRContext* context, StringRef mnemonic,
                                 DialectAsmParser& parser);

// Prints `type` and returns true if it is one of the types declared in this
// file. Returns false otherwise.
bool PrintTflType(Type type, DialectAsmPrinter& printer);

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_TYPES_H_

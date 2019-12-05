//===- Types.h - MLIR Type Classes ------------------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef MLIR_IR_TYPES_H
#define MLIR_IR_TYPES_H

#include "mlir/IR/TypeSupport.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace mlir {
class FloatType;
class Identifier;
class IndexType;
class IntegerType;
class MLIRContext;
class TypeStorage;

namespace detail {
struct FunctionTypeStorage;
struct OpaqueTypeStorage;
} // namespace detail

/// Instances of the Type class are immutable and uniqued.  They wrap a pointer
/// to the storage object owned by MLIRContext.  Therefore, instances of Type
/// are passed around by value.
///
/// Some types are "primitives" meaning they do not have any parameters, for
/// example the Index type.  Parametric types have additional information that
/// differentiates the types of the same kind between them, for example the
/// Integer type has bitwidth, making i8 and i16 belong to the same kind by be
/// different instances of the IntegerType.
///
/// Types are constructed and uniqued via the 'detail::TypeUniquer' class.
///
/// Derived type classes are expected to implement several required
/// implementation hooks:
///  * Required:
///    - static bool kindof(unsigned kind);
///      * Returns if the provided type kind corresponds to an instance of the
///        current type. Used for isa/dyn_cast casting functionality.
///
///  * Optional:
///    - static LogicalResult verifyConstructionInvariants(
///                                               Optional<Location> loc,
///                                               MLIRContext *context,
///                                               Args... args)
///      * This method is invoked when calling the 'TypeBase::get/getChecked'
///        methods to ensure that the arguments passed in are valid to construct
///        a type instance with.
///      * This method is expected to return failure if a type cannot be
///        constructed with 'args', success otherwise.
///      * 'args' must correspond with the arguments passed into the
///        'TypeBase::get' call after the type kind.
///
///
/// Type storage objects inherit from TypeStorage and contain the following:
///    - The type kind (for LLVM-style RTTI).
///    - The dialect that defined the type.
///    - Any parameters of the type.
/// For non-parametric types, a convenience DefaultTypeStorage is provided.
/// Parametric storage types must derive TypeStorage and respect the following:
///    - Define a type alias, KeyTy, to a type that uniquely identifies the
///      instance of the type within its kind.
///      * The key type must be constructible from the values passed into the
///        detail::TypeUniquer::get call after the type kind.
///      * If the KeyTy does not have an llvm::DenseMapInfo specialization, the
///        storage class must define a hashing method:
///         'static unsigned hashKey(const KeyTy &)'
///
///    - Provide a method, 'bool operator==(const KeyTy &) const', to
///      compare the storage instance against an instance of the key type.
///
///    - Provide a construction method:
///        'DerivedStorage *construct(TypeStorageAllocator &, const KeyTy &key)'
///      that builds a unique instance of the derived storage. The arguments to
///      this function are an allocator to store any uniqued data within the
///      context and the key type for this storage.
class Type {
public:
  /// Integer identifier for all the concrete type kinds.
  /// Note: This is not an enum class as each dialect will likely define a
  /// separate enumeration for the specific types that they define. Not being an
  /// enum class also simplifies the handling of type kinds by not requiring
  /// casts for each use.
  enum Kind {
    // Builtin types.
    Function,
    Opaque,
    LAST_BUILTIN_TYPE = Opaque,

  // Reserve type kinds for dialect specific type system extensions.
#define DEFINE_SYM_KIND_RANGE(Dialect)                                         \
  FIRST_##Dialect##_TYPE, LAST_##Dialect##_TYPE = FIRST_##Dialect##_TYPE + 0xff,
#include "DialectSymbolRegistry.def"
  };

  /// Utility class for implementing types.
  template <typename ConcreteType, typename BaseType,
            typename StorageType = DefaultTypeStorage>
  using TypeBase = detail::StorageUserBase<ConcreteType, BaseType, StorageType,
                                           detail::TypeUniquer>;

  using ImplType = TypeStorage;

  Type() : impl(nullptr) {}
  /* implicit */ Type(const ImplType *impl)
      : impl(const_cast<ImplType *>(impl)) {}

  Type(const Type &other) : impl(other.impl) {}
  Type &operator=(Type other) {
    impl = other.impl;
    return *this;
  }

  bool operator==(Type other) const { return impl == other.impl; }
  bool operator!=(Type other) const { return !(*this == other); }
  explicit operator bool() const { return impl; }

  bool operator!() const { return impl == nullptr; }

  template <typename U> bool isa() const;
  template <typename U> U dyn_cast() const;
  template <typename U> U dyn_cast_or_null() const;
  template <typename U> U cast() const;

  // Support type casting Type to itself.
  static bool classof(Type) { return true; }

  /// Return the classification for this type.
  unsigned getKind() const;

  /// Return the LLVMContext in which this type was uniqued.
  MLIRContext *getContext() const;

  /// Get the dialect this type is registered to.
  Dialect &getDialect() const;

  // Convenience predicates.  This is only for floating point types,
  // derived types should use isa/dyn_cast.
  bool isIndex();
  bool isBF16();
  bool isF16();
  bool isF32();
  bool isF64();

  /// Return true if this is an integer type with the specified width.
  bool isInteger(unsigned width);

  /// Return the bit width of an integer or a float type, assert failure on
  /// other types.
  unsigned getIntOrFloatBitWidth();

  /// Return true if this is an integer or index type.
  bool isIntOrIndex();
  /// Return true if this is an integer, index, or float type.
  bool isIntOrIndexOrFloat();
  /// Return true of this is an integer or a float type.
  bool isIntOrFloat();

  /// Print the current type.
  void print(raw_ostream &os);
  void dump();

  friend ::llvm::hash_code hash_value(Type arg);

  unsigned getSubclassData() const;
  void setSubclassData(unsigned val);

  /// Methods for supporting PointerLikeTypeTraits.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(impl);
  }
  static Type getFromOpaquePointer(const void *pointer) {
    return Type(reinterpret_cast<ImplType *>(const_cast<void *>(pointer)));
  }

protected:
  ImplType *impl;
};

inline raw_ostream &operator<<(raw_ostream &os, Type type) {
  type.print(os);
  return os;
}

/// Function types map from a list of inputs to a list of results.
class FunctionType
    : public Type::TypeBase<FunctionType, Type, detail::FunctionTypeStorage> {
public:
  using Base::Base;

  static FunctionType get(ArrayRef<Type> inputs, ArrayRef<Type> results,
                          MLIRContext *context);

  // Input types.
  unsigned getNumInputs() const { return getSubclassData(); }

  Type getInput(unsigned i) const { return getInputs()[i]; }

  ArrayRef<Type> getInputs() const;

  // Result types.
  unsigned getNumResults() const;

  Type getResult(unsigned i) const { return getResults()[i]; }

  ArrayRef<Type> getResults() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) { return kind == Kind::Function; }
};

/// Opaque types represent types of non-registered dialects. These are types
/// represented in their raw string form, and can only usefully be tested for
/// type equality.
class OpaqueType
    : public Type::TypeBase<OpaqueType, Type, detail::OpaqueTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new OpaqueType with the provided dialect and string data.
  static OpaqueType get(Identifier dialect, StringRef typeData,
                        MLIRContext *context);

  /// Get or create a new OpaqueType with the provided dialect and string data.
  /// If the given identifier is not a valid namespace for a dialect, then a
  /// null type is returned.
  static OpaqueType getChecked(Identifier dialect, StringRef typeData,
                               MLIRContext *context, Location location);

  /// Returns the dialect namespace of the opaque type.
  Identifier getDialectNamespace() const;

  /// Returns the raw type data of the opaque type.
  StringRef getTypeData() const;

  /// Verify the construction of an opaque type.
  static LogicalResult verifyConstructionInvariants(Optional<Location> loc,
                                                    MLIRContext *context,
                                                    Identifier dialect,
                                                    StringRef typeData);

  static bool kindof(unsigned kind) { return kind == Kind::Opaque; }
};

// Make Type hashable.
inline ::llvm::hash_code hash_value(Type arg) {
  return ::llvm::hash_value(arg.impl);
}

template <typename U> bool Type::isa() const {
  assert(impl && "isa<> used on a null type.");
  return U::classof(*this);
}
template <typename U> U Type::dyn_cast() const {
  return isa<U>() ? U(impl) : U(nullptr);
}
template <typename U> U Type::dyn_cast_or_null() const {
  return (impl && isa<U>()) ? U(impl) : U(nullptr);
}
template <typename U> U Type::cast() const {
  assert(isa<U>());
  return U(impl);
}

} // end namespace mlir

namespace llvm {

// Type hash just like pointers.
template <> struct DenseMapInfo<mlir::Type> {
  static mlir::Type getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::Type(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static mlir::Type getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::Type(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::Type val) { return mlir::hash_value(val); }
  static bool isEqual(mlir::Type LHS, mlir::Type RHS) { return LHS == RHS; }
};

/// We align TypeStorage by 8, so allow LLVM to steal the low bits.
template <> struct PointerLikeTypeTraits<mlir::Type> {
public:
  static inline void *getAsVoidPointer(mlir::Type I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::Type getFromVoidPointer(void *P) {
    return mlir::Type::getFromOpaquePointer(P);
  }
  enum { NumLowBitsAvailable = 3 };
};

} // namespace llvm

#endif // MLIR_IR_TYPES_H

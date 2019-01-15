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

#include "mlir/IR/Location.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace mlir {
class FloatType;
class IndexType;
class IntegerType;
class MLIRContext;

namespace detail {
struct FunctionTypeStorage;
struct TypeStorage;
struct UnknownTypeStorage;
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
/// implementaiton hooks:
///  * Required:
///    - static char typeID;
///      * A unique identifier for this type used during registration.
///
///    - static bool kindof(unsigned kind);
///      * Returns if the provided type kind corresponds to an instance of the
///        current type. Used for isa/dyn_cast casting functionality.
///
///  * Optional:
///    - static bool verifyConstructionInvariants(llvm::Optional<Location> loc,
///                                               MLIRContext *context,
///                                               Args... args)
///      * This method is invoked when calling the 'TypeBase::get/getChecked'
///        methods to ensure that the arguments passed in are valid to construct
///        a type instance with.
///      * This method is expected to return true if a type cannot be
///        constructed with 'args'.
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
///      * The key type must have a llvm::DenseMapInfo specialization for
///        hashing.
///
///    - Provide a method, 'KeyTy getKey() const', to construct the key type
///      from an existing storage instance.
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
    Unknown,

    // TODO(riverriddle) Index shouldn't really be a builtin.
    // Target pointer sized integer, used (e.g.) in affine mappings.
    Index,
    LAST_BUILTIN_TYPE = Index,

  // Reserve type kinds for dialect specific type system extensions.
#define DEFINE_TYPE_KIND_RANGE(Dialect)                                        \
  FIRST_##Dialect##_TYPE, LAST_##Dialect##_TYPE = FIRST_##Dialect##_TYPE + 0xff,
#include "DialectTypeRegistry.def"
  };

  /// Utility class for implementing types. Clients are not expected to interact
  /// with this class directly. The template arguments to this class are defined
  /// as follows:
  ///   - ConcreteType
  ///     * The top level derived class type.
  ///
  ///   - BaseType
  ///     * The base type class that this utility should derive from, e.g Type,
  ///       TensorType, TensorOrVectorType.
  ///
  ///   - StorageType
  ///     * The type storage object containing the necessary instance
  ///       information for the ConcreteType.
  template <typename ConcreteType, typename BaseType,
            typename StorageType = detail::DefaultTypeStorage>
  class TypeBase : public BaseType {
  public:
    using BaseType::BaseType;

    /// Utility declarations for the concrete type class.
    using Base = TypeBase<ConcreteType, BaseType, StorageType>;
    using ImplType = StorageType;

  protected:
    /// Get or create a new ConcreteType instance within the context. This
    /// function is guaranteed to return a non null type and will assert if the
    /// arguments provided are invalid.
    template <typename... Args>
    static ConcreteType get(MLIRContext *context, unsigned kind, Args... args) {
      // Ensure that the invariants are correct for type construction.
      assert(!ConcreteType::verifyConstructionInvariants(llvm::None, context,
                                                         args...));
      return detail::TypeUniquer(context).get<ConcreteType>(kind, args...);
    }

    /// Get or create a new ConcreteType instance within the context, defined at
    /// the given, potentially unknown, location. If the arguments provided are
    /// invalid then emit errors and return a null type.
    template <typename... Args>
    static ConcreteType getChecked(Location loc, MLIRContext *context,
                                   unsigned kind, Args... args) {
      // If the construction invariants fail then we return a null type.
      if (ConcreteType::verifyConstructionInvariants(loc, context, args...))
        return ConcreteType();
      return detail::TypeUniquer(context).get<ConcreteType>(kind, args...);
    }

    /// Default implementation that just returns false for success.
    template <typename... Args>
    static bool verifyConstructionInvariants(llvm::Optional<Location> loc,
                                             MLIRContext *context,
                                             Args... args) {
      return false;
    }
  };

  using ImplType = detail::TypeStorage;

  Type() : type(nullptr) {}
  /* implicit */ Type(const ImplType *type)
      : type(const_cast<ImplType *>(type)) {}

  Type(const Type &other) : type(other.type) {}
  Type &operator=(Type other) {
    type = other.type;
    return *this;
  }

  bool operator==(Type other) const { return type == other.type; }
  bool operator!=(Type other) const { return !(*this == other); }
  explicit operator bool() const { return type; }

  bool operator!() const { return type == nullptr; }

  template <typename U> bool isa() const;
  template <typename U> U dyn_cast() const;
  template <typename U> U dyn_cast_or_null() const;
  template <typename U> U cast() const;

  /// Return the classification for this type.
  unsigned getKind() const;

  /// Return the LLVMContext in which this type was uniqued.
  MLIRContext *getContext() const;

  /// Get the dialect this type is registered to.
  const Dialect &getDialect() const;

  // Convenience predicates.  This is only for floating point types,
  // derived types should use isa/dyn_cast.
  bool isIndex() const;
  bool isBF16() const;
  bool isF16() const;
  bool isF32() const;
  bool isF64() const;

  /// Return true if this is an integer type with the specified width.
  bool isInteger(unsigned width) const;

  /// Return the bit width of an integer or a float type, assert failure on
  /// other types.
  unsigned getIntOrFloatBitWidth() const;

  /// Return true if this is an integer or index type.
  bool isIntOrIndex() const;
  /// Return true if this is an integer, index, or float type.
  bool isIntOrIndexOrFloat() const;
  /// Return true of this is an integer or a float type.
  bool isIntOrFloat() const;

  // Convenience factories.
  static IndexType getIndex(MLIRContext *ctx);
  static IntegerType getInteger(unsigned width, MLIRContext *ctx);
  static FloatType getBF16(MLIRContext *ctx);
  static FloatType getF16(MLIRContext *ctx);
  static FloatType getF32(MLIRContext *ctx);
  static FloatType getF64(MLIRContext *ctx);

  /// Print the current type.
  void print(raw_ostream &os) const;
  void dump() const;

  friend ::llvm::hash_code hash_value(Type arg);

  unsigned getSubclassData() const;
  void setSubclassData(unsigned val);

  /// Methods for supporting PointerLikeTypeTraits.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(type);
  }
  static Type getFromOpaquePointer(const void *pointer) {
    return Type((ImplType *)(pointer));
  }

protected:
  ImplType *type;
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

  /// Unique identifier for this type class.
  static char typeID;
};

inline bool Type::isIndex() const { return getKind() == Kind::Index; }

/// Index is special integer-like type with unknown platform-dependent bit width
/// used in subscripts and loop induction variables.
class IndexType : public Type::TypeBase<IndexType, Type> {
public:
  using Base::Base;

  /// Crete an IndexType instance, unique in the given context.
  static IndexType get(MLIRContext *context) {
    return Base::get(context, Kind::Index);
  }

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) { return kind == Kind::Index; }

  /// Unique identifier for this type class.
  static char typeID;
};

/// Unknown types represent types of non-registered dialects. These are types
/// represented in their raw string form, and can only usefully be tested for
/// type equality.
class UnknownType
    : public Type::TypeBase<UnknownType, Type, detail::UnknownTypeStorage> {
public:
  using Base::Base;

  /// Get or create a new UnknownType with the provided dialect and string data.
  static UnknownType get(Identifier dialect, StringRef typeData,
                         MLIRContext *context);

  /// Returns the dialect namespace of the unknown type.
  Identifier getDialectNamespace() const;

  /// Returns the raw type data of the unknown type.
  StringRef getTypeData() const;

  static bool kindof(unsigned kind) { return kind == Kind::Unknown; }

  /// Unique identifier for this type class.
  static char typeID;
};

inline IndexType Type::getIndex(MLIRContext *ctx) {
  return IndexType::get(ctx);
}

// Make Type hashable.
inline ::llvm::hash_code hash_value(Type arg) {
  return ::llvm::hash_value(arg.type);
}

template <typename U> bool Type::isa() const {
  assert(type && "isa<> used on a null type.");
  return U::kindof(getKind());
}
template <typename U> U Type::dyn_cast() const {
  return isa<U>() ? U(type) : U(nullptr);
}
template <typename U> U Type::dyn_cast_or_null() const {
  return (type && isa<U>()) ? U(type) : U(nullptr);
}
template <typename U> U Type::cast() const {
  assert(isa<U>());
  return U(type);
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

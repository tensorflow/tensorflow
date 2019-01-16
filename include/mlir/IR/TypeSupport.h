//===- TypeSupport.h --------------------------------------------*- C++ -*-===//
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
//
// This file defines support types for registering dialect extended types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_TYPE_SUPPORT_H
#define MLIR_IR_TYPE_SUPPORT_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {
class Dialect;
class MLIRContext;

//===----------------------------------------------------------------------===//
// TypeStorage
//===----------------------------------------------------------------------===//

namespace detail {

class TypeUniquer;

/// Base storage class appearing in a Type.
struct alignas(8) TypeStorage {
  friend TypeUniquer;

protected:
  /// This constructor is used by derived classes as part of the TypeUniquer.
  /// When using this constructor, the initializeTypeInfo function must be
  /// invoked afterwards for the storage to be valid.
  TypeStorage(unsigned subclassData = 0)
      : dialect(nullptr), kind(0), subclassData(subclassData) {}

public:
  /// Get the dialect that this type is registered to.
  const Dialect &getDialect() const {
    assert(dialect && "Malformed type storage object.");
    return *dialect;
  }

  /// Get the kind classification of this type.
  unsigned getKind() const { return kind; }

  /// Get the subclass data.
  unsigned getSubclassData() const { return subclassData; }

  /// Set the subclass data.
  void setSubclassData(unsigned val) { subclassData = val; }

private:
  // Constructor used for simple type storage that have no subclass data. This
  // constructor should not be used by derived storage classes.
  TypeStorage(const Dialect &dialect, unsigned kind)
      : dialect(&dialect), kind(kind), subclassData(0) {}

  // Initialize an existing type storage with a kind and a context. This is used
  // by the TypeUniquer when initializing a newly constructed derived type
  // storage object.
  void initializeTypeInfo(const Dialect &newDialect, unsigned newKind) {
    dialect = &newDialect;
    kind = newKind;
  }

  /// The registered information for the current type.
  const Dialect *dialect;

  /// Classification of the subclass, used for type checking.
  unsigned kind;

  /// Space for subclasses to store data.
  unsigned subclassData;
};

/// Default storage type for types that require no additional initialization or
/// storage.
using DefaultTypeStorage = TypeStorage;

} // end namespace detail

//===----------------------------------------------------------------------===//
// TypeStorageAllocator
//===----------------------------------------------------------------------===//

// This is a utility allocator used to allocate memory for derived types that
// will be tied to the current MLIRContext.
class TypeStorageAllocator {
public:
  TypeStorageAllocator(MLIRContext *ctx) : ctx(ctx) {}

  /// Copy the specified array of elements into memory managed by our bump
  /// pointer allocator.  This assumes the elements are all PODs.
  template <typename T> ArrayRef<T> copyInto(ArrayRef<T> elements) {
    if (elements.empty())
      return llvm::None;
    auto result = getAllocator().Allocate<T>(elements.size());
    std::uninitialized_copy(elements.begin(), elements.end(), result);
    return ArrayRef<T>(result, elements.size());
  }

  /// Copy the provided string into memory managed by our bump pointer
  /// allocator.
  StringRef copyInto(StringRef str) {
    auto result = copyInto(ArrayRef<char>(str.data(), str.size()));
    return StringRef(result.data(), str.size());
  }

  // Allocate an instance of the provided type.
  template <typename T> T *allocate() { return getAllocator().Allocate<T>(); }

private:
  /// Get a reference to the internal allocator.
  llvm::BumpPtrAllocator &getAllocator();

  MLIRContext *ctx;
};

//===----------------------------------------------------------------------===//
// TypeUniquer
//===----------------------------------------------------------------------===//
namespace detail {
// A utility class to get, or create, unique instances of types within an
// MLIRContext. This class manages all creation and uniquing of types.
class TypeUniquer {
public:
  template <typename T, typename... Args>
  static T get(MLIRContext *ctx, Args &&... args) {
    TypeUniquer &instance = ctx->getTypeUniquer();
    return instance.getImpl<T>(ctx, args...);
  }

private:
  /// A utility wrapper object representing a hashed storage object. This class
  /// contains a storage object and an existing computed hash value.
  struct HashedStorageType {
    unsigned hashValue;
    TypeStorage *storage;
  };

  /// A lookup key for derived instances of TypeStorage objects.
  struct TypeLookupKey {
    /// The known derived kind for the storage.
    unsigned kind;

    /// The known hash value of the key.
    unsigned hashValue;

    /// An equality function for comparing with an existing storage instance.
    llvm::function_ref<bool(const TypeStorage *)> isEqual;
  };

  /// Get an uniqued instance of a type T. This overload is used for derived
  /// types that have complex storage or uniquing constraints.
  template <typename T, typename... Args>
  typename std::enable_if<
      !std::is_same<typename T::ImplType, DefaultTypeStorage>::value, T>::type
  getImpl(MLIRContext *ctx, unsigned kind, Args &&... args) {
    using ImplType = typename T::ImplType;
    using KeyTy = typename ImplType::KeyTy;

    // Construct a value of the derived key type.
    auto derivedKey = getKey<ImplType>(args...);

    // Create a hash of the kind and the derived key.
    unsigned hashValue = llvm::hash_combine(
        kind, llvm::DenseMapInfo<KeyTy>::getHashValue(derivedKey));

    // Generate an equality function for the derived storage.
    std::function<bool(const TypeStorage *)> isEqual =
        [&derivedKey](const TypeStorage *existing) {
          return static_cast<const ImplType &>(*existing) == derivedKey;
        };

    // Look to see if the type has been created already.
    auto existing =
        storageTypes.insert_as({}, TypeLookupKey{kind, hashValue, isEqual});

    // If it has been created, return it.
    if (!existing.second)
      return T(existing.first->storage);

    // Otherwise, construct and initialize the derived storage for this type
    // instance.
    TypeStorageAllocator allocator(ctx);
    TypeStorage *storage = ImplType::construct(allocator, derivedKey);
    storage->initializeTypeInfo(lookupDialectForType<T>(ctx), kind);
    *existing.first = HashedStorageType{hashValue, storage};
    return T(storage);
  }

  /// Get an uniqued instance of a type T. This overload is used for derived
  /// types that use the DefaultTypeStorage and thus need no additional storage
  /// or uniquing.
  template <typename T, typename... Args>
  typename std::enable_if<
      std::is_same<typename T::ImplType, DefaultTypeStorage>::value, T>::type
  getImpl(MLIRContext *ctx, unsigned kind) {
    // Check for an existing instance with this kind.
    auto *&result = simpleTypes[kind];
    if (!result) {
      // Otherwise, allocate and initialize one.
      TypeStorageAllocator allocator(ctx);
      result = new (allocator.allocate<DefaultTypeStorage>())
          DefaultTypeStorage(lookupDialectForType<T>(ctx), kind);
    }
    return T(result);
  }

  /// Get the dialect that the type 'T' was registered with.
  template <typename T>
  static const Dialect &lookupDialectForType(MLIRContext *ctx) {
    return lookupDialectForType(ctx, &T::typeID);
  }

  /// Get the dialect that registered the type with the provided typeid.
  static const Dialect &lookupDialectForType(MLIRContext *ctx,
                                             const void *const typeID);

  //===--------------------------------------------------------------------===//
  // Key Construction
  //===--------------------------------------------------------------------===//

  /// Utilities for generating a derived storage key.
  /// Overload for if the key can be directly constructed from the provided
  /// arguments.
  template <typename ImplTy, typename... Args>
  static typename std::enable_if<
      std::is_constructible<typename ImplTy::KeyTy, Args...>::value,
      typename ImplTy::KeyTy>::type
  getKey(Args &&... args) {
    return typename ImplTy::KeyTy(args...);
  }
  // If the key cannot be directly constructed, query the derived storage for a
  // construction function.
  template <typename ImplTy, typename... Args>
  static typename std::enable_if<
      !std::is_constructible<typename ImplTy::KeyTy, Args...>::value,
      typename ImplTy::KeyTy>::type
  getKey(Args &&... args) {
    return ImplTy::getKey(args...);
  }

  //===--------------------------------------------------------------------===//
  // Instance Storage
  //===--------------------------------------------------------------------===//

  /// Storage info for derived TypeStorage objects.
  struct StorageKeyInfo : DenseMapInfo<HashedStorageType> {
    static HashedStorageType getEmptyKey() {
      return HashedStorageType{0, DenseMapInfo<TypeStorage *>::getEmptyKey()};
    }
    static HashedStorageType getTombstoneKey() {
      return HashedStorageType{0,
                               DenseMapInfo<TypeStorage *>::getTombstoneKey()};
    }

    static unsigned getHashValue(const HashedStorageType &key) {
      return key.hashValue;
    }
    static unsigned getHashValue(TypeLookupKey key) { return key.hashValue; }

    static bool isEqual(const HashedStorageType &lhs,
                        const HashedStorageType &rhs) {
      return lhs.storage == rhs.storage;
    }
    static bool isEqual(const TypeLookupKey &lhs,
                        const HashedStorageType &rhs) {
      if (isEqual(rhs, getEmptyKey()) || isEqual(rhs, getTombstoneKey()))
        return false;
      // If the lookup kind matches the kind of the storage, then invoke the
      // equality function on the lookup key.
      return lhs.kind == rhs.storage->getKind() && lhs.isEqual(rhs.storage);
    }
  };

  // Unique types with specific hashing or storage constraints.
  using StorageTypeSet = llvm::DenseSet<HashedStorageType, StorageKeyInfo>;
  StorageTypeSet storageTypes;

  // Unique types with just the kind.
  DenseMap<unsigned, TypeStorage *> simpleTypes;
};
} // namespace detail

} // end namespace mlir

#endif

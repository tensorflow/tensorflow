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

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfo.h"
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
    auto result = getAllocator().Allocate<T>(elements.size());
    std::uninitialized_copy(elements.begin(), elements.end(), result);
    return ArrayRef<T>(result, elements.size());
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
  /// Lookup key for storage types.
  struct TypeLookupKey {
    /// The known hash value of the key.
    unsigned hashValue;

    /// An equality function for comparing with an existing storage instance.
    llvm::function_ref<bool(const detail::TypeStorage *)> isEqual;
  };

  TypeUniquer(MLIRContext *ctx) : ctx(ctx) {}

  /// Get an uniqued instance of a type T. This overload is used for derived
  /// types that have complex storage or uniquing constraints.
  template <typename T, typename... Args>
  typename std::enable_if<
      !std::is_same<typename T::ImplType, detail::DefaultTypeStorage>::value,
      T>::type
  get(unsigned kind, Args... args) {
    using ImplType = typename T::ImplType;
    using KeyTy = typename ImplType::KeyTy;

    // Construct a value of the derived key type.
    auto derivedKey = getKey<ImplType>(args...);

    // Create a hash of the kind and the derived key.
    unsigned hashValue = llvm::hash_combine(
        kind, llvm::DenseMapInfo<KeyTy>::getHashValue(derivedKey));

    // Generate an equality function for the derived storage.
    std::function<bool(const detail::TypeStorage *)> isEqual =
        [kind, &derivedKey](const detail::TypeStorage *existing) {
          // Check that these type storages have the same kind.
          if (kind != existing->getKind())
            return false;
          // Generate a key from the derived storage and compare it to the
          // current key.
          auto *derivedStorage = static_cast<const ImplType *>(existing);
          return derivedStorage->getKey() == derivedKey;
        };

    // Lookup an existing type with the given key.
    detail::TypeStorage *storage = lookup(TypeLookupKey{hashValue, isEqual});
    if (storage)
      return T(storage);

    // Get the dialect this type was registered to.
    auto &dialect = lookupDialectForType<T>();

    // Otherwise, construct and initialize the derived storage for this type
    // instance.
    TypeStorageAllocator allocator(ctx);
    storage = ImplType::construct(allocator, args...);
    storage->initializeTypeInfo(dialect, kind);

    // Insert the new type storage instance into the context.
    insert(hashValue, storage);
    return T(storage);
  }

  /// Get an uniqued instance of a type T. This overload is used for derived
  /// types that use the DefaultTypeStorage and thus need no additional storage
  /// or uniquing.
  template <typename T, typename... Args>
  typename std::enable_if<
      std::is_same<typename T::ImplType, detail::DefaultTypeStorage>::value,
      T>::type
  get(unsigned kind) {
    auto &dialect = lookupDialectForType<T>();
    return T(getSimple(dialect, kind));
  }

private:
  /// Get the dialect that the type 'T' was registered with.
  template <typename T> const Dialect &lookupDialectForType() {
    return lookupDialectForType(&T::typeID);
  }

  /// Get the dialect that registered the type with the provided typeid.
  const Dialect &lookupDialectForType(const void *const typeID);

  /// Get or create a uniqued type by its kind. This overload is used for
  /// simple types that are only uniqued by kind.
  detail::TypeStorage *getSimple(const Dialect &dialect, unsigned kind);

  /// Utilities for generating a derived storage key.
  /// Overload for if the key can be directly constructed from the provided
  /// arguments.
  template <typename ImplTy, typename... Args>
  static typename std::enable_if<
      std::is_constructible<typename ImplTy::KeyTy, Args...>::value,
      typename ImplTy::KeyTy>::type
  getKey(Args... args) {
    return typename ImplTy::KeyTy(args...);
  }
  // If the key cannot be directly constructed, query the derived storage for a
  // construction function.
  template <typename ImplTy, typename... Args>
  static typename std::enable_if<
      !std::is_constructible<typename ImplTy::KeyTy, Args...>::value,
      typename ImplTy::KeyTy>::type
  getKey(Args... args) {
    return ImplTy::getKey(args...);
  }

  /// Look up a uniqued type with a lookup key. This is used if the type defines
  /// a storage key.
  detail::TypeStorage *lookup(const TypeLookupKey &key);

  /// Insert a new type storage into the context.
  void insert(unsigned hashValue, detail::TypeStorage *storage);

  /// The current context that a type is being uniqued from.
  MLIRContext *ctx;
};
} // namespace detail

} // end namespace mlir

#endif

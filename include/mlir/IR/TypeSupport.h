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

/// TypeID is used to provide a unique address identifier for derived Type
/// classes.
struct TypeID {
  template <typename T> static TypeID *getID() {
    static TypeID id;
    return &id;
  }
};

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

// This is a utility allocator used to allocate memory for instances of derived
// Types.
class TypeStorageAllocator {
public:
  /// Copy the specified array of elements into memory managed by our bump
  /// pointer allocator.  This assumes the elements are all PODs.
  template <typename T> ArrayRef<T> copyInto(ArrayRef<T> elements) {
    if (elements.empty())
      return llvm::None;
    auto result = allocator.Allocate<T>(elements.size());
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
  template <typename T> T *allocate() { return allocator.Allocate<T>(); }

  /// Allocate 'size' bytes of 'alignment' aligned memory.
  void *allocate(size_t size, size_t alignment) {
    return allocator.Allocate(size, alignment);
  }

private:
  /// The raw allocator for type storage objects.
  llvm::BumpPtrAllocator allocator;
};

//===----------------------------------------------------------------------===//
// TypeUniquer
//===----------------------------------------------------------------------===//
namespace detail {
// A utility class to get, or create, unique instances of types within an
// MLIRContext. This class manages all creation and uniquing of types.
class TypeUniquer {
public:
  /// Get an uniqued instance of a type T. This overload is used for derived
  /// types that have complex storage or uniquing constraints.
  template <typename T, typename... Args>
  static typename std::enable_if<
      !std::is_same<typename T::ImplType, DefaultTypeStorage>::value, T>::type
  get(MLIRContext *ctx, unsigned kind, Args &&... args) {
    using ImplType = typename T::ImplType;

    // Construct a value of the derived key type.
    auto derivedKey = getKey<ImplType>(args...);

    // Create a hash of the kind and the derived key.
    unsigned hashValue = getHash<ImplType>(kind, derivedKey);

    // Generate an equality function for the derived storage.
    std::function<bool(const TypeStorage *)> isEqual =
        [&derivedKey](const TypeStorage *existing) {
          return static_cast<const ImplType &>(*existing) == derivedKey;
        };

    // Generate a constructor function for the derived storage.
    std::function<TypeStorage *(TypeStorageAllocator &)> constructorFn =
        [&](TypeStorageAllocator &allocator) {
          TypeStorage *storage = ImplType::construct(allocator, derivedKey);
          storage->initializeTypeInfo(lookupDialectForType<T>(ctx), kind);
          return storage;
        };

    // Get an instance for the derived storage.
    return T(getImpl(ctx, kind, hashValue, isEqual, constructorFn));
  }

  /// Get an uniqued instance of a type T. This overload is used for derived
  /// types that use the DefaultTypeStorage and thus need no additional storage
  /// or uniquing.
  template <typename T, typename... Args>
  static typename std::enable_if<
      std::is_same<typename T::ImplType, DefaultTypeStorage>::value, T>::type
  get(MLIRContext *ctx, unsigned kind) {
    auto constructorFn = [=](TypeStorageAllocator &allocator) {
      return new (allocator.allocate<DefaultTypeStorage>())
          DefaultTypeStorage(lookupDialectForType<T>(ctx), kind);
    };
    return T(getImpl(ctx, kind, constructorFn));
  }

private:
  /// Implementation for getting/creating an instance of a derived type with
  /// complex storage.
  static TypeStorage *
  getImpl(MLIRContext *ctx, unsigned kind, unsigned hashValue,
          llvm::function_ref<bool(const TypeStorage *)> isEqual,
          std::function<TypeStorage *(TypeStorageAllocator &)> constructorFn);

  /// Implementation for getting/creating an instance of a derived type with
  /// default storage.
  static TypeStorage *
  getImpl(MLIRContext *ctx, unsigned kind,
          std::function<TypeStorage *(TypeStorageAllocator &)> constructorFn);

  /// Get the dialect that the type 'T' was registered with.
  template <typename T>
  static const Dialect &lookupDialectForType(MLIRContext *ctx) {
    return lookupDialectForType(ctx, T::getTypeID());
  }

  /// Get the dialect that registered the type with the provided typeid.
  static const Dialect &lookupDialectForType(MLIRContext *ctx,
                                             const TypeID *const typeID);

  //===--------------------------------------------------------------------===//
  // Util
  //===--------------------------------------------------------------------===//

  /// Utilities for detecting if specific traits hold for a given type 'T'.
  template <typename...> using void_t = void;
  template <class, template <class...> class Op, class... Args>
  struct detector {
    using value_t = std::false_type;
  };
  template <template <class...> class Op, class... Args>
  struct detector<void_t<Op<Args...>>, Op, Args...> {
    using value_t = std::true_type;
  };
  template <template <class...> class Op, class... Args>
  using is_detected = typename detector<void, Op, Args...>::value_t;

  //===--------------------------------------------------------------------===//
  // Key Construction
  //===--------------------------------------------------------------------===//

  /// Trait to check if ImplTy provides a 'getKey' method with types 'Args'.
  template <typename ImplTy, typename... Args>
  using has_impltype_getkey_t =
      decltype(ImplTy::getKey(std::declval<Args>()...));

  /// Used to construct an instance of 'ImplType::KeyTy' if there is an
  /// 'ImplTy::getKey' function for the provided arguments.
  template <typename ImplTy, typename... Args>
  static typename std::enable_if<
      is_detected<has_impltype_getkey_t, ImplTy, Args...>::value,
      typename ImplTy::KeyTy>::type
  getKey(Args &&... args) {
    return ImplTy::getKey(args...);
  }
  /// If there is no 'ImplTy::getKey' method, then we try to directly construct
  /// the 'ImplTy::KeyTy' with the provided arguments.
  template <typename ImplTy, typename... Args>
  static typename std::enable_if<
      !is_detected<has_impltype_getkey_t, ImplTy, Args...>::value,
      typename ImplTy::KeyTy>::type
  getKey(Args &&... args) {
    return typename ImplTy::KeyTy(args...);
  }

  //===--------------------------------------------------------------------===//
  // Key and Kind Hashing
  //===--------------------------------------------------------------------===//

  /// Trait to check if ImplType provides a 'hashKey' method for 'T'.
  template <typename ImplType, typename T>
  using has_impltype_hash_t = decltype(ImplType::hashKey(std::declval<T>()));

  /// Used to generate a hash for the 'ImplTy::KeyTy' and kind of a storage
  /// instance if there is an 'ImplTy::hashKey' overload for 'DerivedKey'.
  template <typename ImplTy, typename DerivedKey>
  static typename std::enable_if<
      is_detected<has_impltype_hash_t, ImplTy, DerivedKey>::value,
      ::llvm::hash_code>::type
  getHash(unsigned kind, const DerivedKey &derivedKey) {
    return llvm::hash_combine(kind, ImplTy::hashKey(derivedKey));
  }
  /// If there is no 'ImplTy::hashKey' default to using the
  /// 'llvm::DenseMapInfo' definition for 'DerivedKey' for generating a hash.
  template <typename ImplTy, typename DerivedKey>
  static typename std::enable_if<
      !is_detected<has_impltype_hash_t, ImplTy, DerivedKey>::value,
      ::llvm::hash_code>::type
  getHash(unsigned kind, const DerivedKey &derivedKey) {
    return llvm::hash_combine(
        kind, llvm::DenseMapInfo<DerivedKey>::getHashValue(derivedKey));
  }
};
} // namespace detail

} // end namespace mlir

#endif

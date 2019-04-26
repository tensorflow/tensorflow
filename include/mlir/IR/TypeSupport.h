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
#include "mlir/Support/StorageUniquer.h"
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
} // end namespace detail

/// Base storage class appearing in a Type.
class TypeStorage : public StorageUniquer::BaseStorage {
  friend detail::TypeUniquer;
  friend StorageUniquer;

protected:
  /// This constructor is used by derived classes as part of the TypeUniquer.
  /// When using this constructor, the initializeTypeInfo function must be
  /// invoked afterwards for the storage to be valid.
  TypeStorage(unsigned subclassData = 0)
      : dialect(nullptr), subclassData(subclassData) {}

public:
  /// Get the dialect that this type is registered to.
  const Dialect &getDialect() const {
    assert(dialect && "Malformed type storage object.");
    return *dialect;
  }
  /// Get the subclass data.
  unsigned getSubclassData() const { return subclassData; }

  /// Set the subclass data.
  void setSubclassData(unsigned val) { subclassData = val; }

private:
  // Set the dialect for this storage instance. This is used by the TypeUniquer
  // when initializing a newly constructed type storage object.
  void initializeDialect(const Dialect &newDialect) { dialect = &newDialect; }

  /// The registered information for the current type.
  const Dialect *dialect;

  /// Space for subclasses to store data.
  unsigned subclassData;
};

/// Default storage type for types that require no additional initialization or
/// storage.
using DefaultTypeStorage = TypeStorage;

//===----------------------------------------------------------------------===//
// TypeStorageAllocator
//===----------------------------------------------------------------------===//

// This is a utility allocator used to allocate memory for instances of derived
// Types.
using TypeStorageAllocator = StorageUniquer::StorageAllocator;

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
    // Lookup an instance of this complex storage type.
    using ImplType = typename T::ImplType;
    return ctx->getTypeUniquer().getComplex<ImplType>(
        [&](ImplType *storage) {
          storage->initializeDialect(lookupDialectForType<T>(ctx));
        },
        kind, std::forward<Args>(args)...);
  }

  /// Get an uniqued instance of a type T. This overload is used for derived
  /// types that use the DefaultTypeStorage and thus need no additional storage
  /// or uniquing.
  template <typename T, typename... Args>
  static typename std::enable_if<
      std::is_same<typename T::ImplType, DefaultTypeStorage>::value, T>::type
  get(MLIRContext *ctx, unsigned kind) {
    // Lookup an instance of this simple storage type.
    return ctx->getTypeUniquer().getSimple<TypeStorage>(
        [&](TypeStorage *storage) {
          storage->initializeDialect(lookupDialectForType<T>(ctx));
        },
        kind);
  }

private:
  /// Get the dialect that the type 'T' was registered with.
  template <typename T>
  static const Dialect &lookupDialectForType(MLIRContext *ctx) {
    return lookupDialectForType(ctx, T::getTypeID());
  }

  /// Get the dialect that registered the type with the provided typeid.
  static const Dialect &lookupDialectForType(MLIRContext *ctx,
                                             const TypeID *const typeID);
};
} // namespace detail

} // end namespace mlir

#endif

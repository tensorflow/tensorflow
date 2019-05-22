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
#include "mlir/IR/StorageUniquerSupport.h"

namespace mlir {
struct ClassID;
class Dialect;
class MLIRContext;

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
  Dialect &getDialect() {
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
  void initializeDialect(Dialect &newDialect) { dialect = &newDialect; }

  /// The dialect for this type.
  Dialect *dialect;

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
  /// Get an uniqued instance of a type T.
  template <typename T, typename... Args>
  static T get(MLIRContext *ctx, unsigned kind, Args &&... args) {
    return ctx->getTypeUniquer().get<typename T::ImplType>(
        [&](TypeStorage *storage) {
          storage->initializeDialect(lookupDialectForType<T>(ctx));
        },
        kind, std::forward<Args>(args)...);
  }

private:
  /// Get the dialect that the type 'T' was registered with.
  template <typename T> static Dialect &lookupDialectForType(MLIRContext *ctx) {
    return lookupDialectForType(ctx, T::getClassID());
  }

  /// Get the dialect that registered the type with the provided typeid.
  static Dialect &lookupDialectForType(MLIRContext *ctx,
                                       const ClassID *const typeID);
};
} // namespace detail

} // end namespace mlir

#endif

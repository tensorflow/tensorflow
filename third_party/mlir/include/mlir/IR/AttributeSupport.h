//===- AttributeSupport.h ---------------------------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines support types for registering dialect extended attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_ATTRIBUTESUPPORT_H
#define MLIR_IR_ATTRIBUTESUPPORT_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "llvm/ADT/PointerIntPair.h"

namespace mlir {
class MLIRContext;
class Type;

//===----------------------------------------------------------------------===//
// AttributeStorage
//===----------------------------------------------------------------------===//

namespace detail {
class AttributeUniquer;
} // end namespace detail

/// Base storage class appearing in an attribute. Derived storage classes should
/// only be constructed within the context of the AttributeUniquer.
class AttributeStorage : public StorageUniquer::BaseStorage {
  friend detail::AttributeUniquer;
  friend StorageUniquer;

public:
  /// Get the type of this attribute.
  Type getType() const;

  /// Get the dialect of this attribute.
  Dialect &getDialect() const {
    assert(dialect && "Malformed attribute storage object.");
    return const_cast<Dialect &>(*dialect);
  }

protected:
  /// Construct a new attribute storage instance with the given type.
  /// Note: All attributes require a valid type. If no type is provided here,
  ///       the type of the attribute will automatically default to NoneType
  ///       upon initialization in the uniquer.
  AttributeStorage(Type type);
  AttributeStorage();

  /// Set the type of this attribute.
  void setType(Type type);

  // Set the dialect for this storage instance. This is used by the
  // AttributeUniquer when initializing a newly constructed storage object.
  void initializeDialect(Dialect &newDialect) { dialect = &newDialect; }

private:
  /// The dialect for this attribute.
  Dialect *dialect;

  /// The opaque type of the attribute value.
  const void *type;
};

/// Default storage type for attributes that require no additional
/// initialization or storage.
using DefaultAttributeStorage = AttributeStorage;

//===----------------------------------------------------------------------===//
// AttributeStorageAllocator
//===----------------------------------------------------------------------===//

// This is a utility allocator used to allocate memory for instances of derived
// Attributes.
using AttributeStorageAllocator = StorageUniquer::StorageAllocator;

//===----------------------------------------------------------------------===//
// AttributeUniquer
//===----------------------------------------------------------------------===//
namespace detail {
// A utility class to get, or create, unique instances of attributes within an
// MLIRContext. This class manages all creation and uniquing of attributes.
class AttributeUniquer {
public:
  /// Get an uniqued instance of attribute T.
  template <typename T, typename... Args>
  static T get(MLIRContext *ctx, unsigned kind, Args &&... args) {
    return ctx->getAttributeUniquer().get<typename T::ImplType>(
        getInitFn(ctx, T::getClassID()), kind, std::forward<Args>(args)...);
  }

private:
  /// Returns a functor used to initialize new attribute storage instances.
  static std::function<void(AttributeStorage *)>
  getInitFn(MLIRContext *ctx, const ClassID *const attrID);
};
} // namespace detail

} // end namespace mlir

#endif

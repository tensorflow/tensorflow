//===- DialectInterface.h - IR Dialect Interfaces ---------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_DIALECTINTERFACE_H
#define MLIR_IR_DIALECTINTERFACE_H

#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
class Dialect;
class MLIRContext;
class Operation;

//===----------------------------------------------------------------------===//
// DialectInterface
//===----------------------------------------------------------------------===//
namespace detail {
/// The base class used for all derived interface types. This class provides
/// utilities necessary for registration.
template <typename ConcreteType, typename BaseT>
class DialectInterfaceBase : public BaseT {
public:
  using Base = DialectInterfaceBase<ConcreteType, BaseT>;

  /// Get a unique id for the derived interface type.
  static ClassID *getInterfaceID() { return ClassID::getID<ConcreteType>(); }

protected:
  DialectInterfaceBase(Dialect *dialect) : BaseT(dialect, getInterfaceID()) {}
};
} // end namespace detail

/// This class represents an interface overridden for a single dialect.
class DialectInterface {
public:
  virtual ~DialectInterface();

  /// The base class used for all derived interface types. This class provides
  /// utilities necessary for registration.
  template <typename ConcreteType>
  using Base = detail::DialectInterfaceBase<ConcreteType, DialectInterface>;

  /// Return the dialect that this interface represents.
  Dialect *getDialect() const { return dialect; }

  /// Return the derived interface id.
  ClassID *getID() const { return interfaceID; }

protected:
  DialectInterface(Dialect *dialect, ClassID *id)
      : dialect(dialect), interfaceID(id) {}

private:
  /// The dialect that represents this interface.
  Dialect *dialect;

  /// The unique identifier for the derived interface type.
  ClassID *interfaceID;
};

//===----------------------------------------------------------------------===//
// DialectInterfaceCollection
//===----------------------------------------------------------------------===//

namespace detail {
/// This class is the base class for a collection of instances for a specific
/// interface kind.
class DialectInterfaceCollectionBase {
  /// DenseMap info for dialect interfaces that allows lookup by the dialect.
  struct InterfaceKeyInfo : public DenseMapInfo<const DialectInterface *> {
    using DenseMapInfo<const DialectInterface *>::isEqual;

    static unsigned getHashValue(Dialect *key) { return llvm::hash_value(key); }
    static unsigned getHashValue(const DialectInterface *key) {
      return getHashValue(key->getDialect());
    }

    static bool isEqual(Dialect *lhs, const DialectInterface *rhs) {
      if (rhs == getEmptyKey() || rhs == getTombstoneKey())
        return false;
      return lhs == rhs->getDialect();
    }
  };

  /// A set of registered dialect interface instances.
  using InterfaceSetT = DenseSet<const DialectInterface *, InterfaceKeyInfo>;
  using InterfaceVectorT = std::vector<const DialectInterface *>;

public:
  DialectInterfaceCollectionBase(MLIRContext *ctx, ClassID *interfaceKind);
  virtual ~DialectInterfaceCollectionBase();

protected:
  /// Get the interface for the dialect of given operation, or null if one
  /// is not registered.
  const DialectInterface *getInterfaceFor(Operation *op) const;

  /// Get the interface for the given dialect.
  const DialectInterface *getInterfaceFor(Dialect *dialect) const {
    auto it = interfaces.find_as(dialect);
    return it == interfaces.end() ? nullptr : *it;
  }

  /// An iterator class that iterates the held interface objects of the given
  /// derived interface type.
  template <typename InterfaceT>
  class iterator : public llvm::mapped_iterator<
                       InterfaceVectorT::const_iterator,
                       const InterfaceT &(*)(const DialectInterface *)> {
    static const InterfaceT &remapIt(const DialectInterface *interface) {
      return *static_cast<const InterfaceT *>(interface);
    }

    iterator(InterfaceVectorT::const_iterator it)
        : llvm::mapped_iterator<
              InterfaceVectorT::const_iterator,
              const InterfaceT &(*)(const DialectInterface *)>(it, &remapIt) {}

    /// Allow access to the constructor.
    friend DialectInterfaceCollectionBase;
  };

  /// Iterator access to the held interfaces.
  template <typename InterfaceT> iterator<InterfaceT> interface_begin() const {
    return iterator<InterfaceT>(orderedInterfaces.begin());
  }
  template <typename InterfaceT> iterator<InterfaceT> interface_end() const {
    return iterator<InterfaceT>(orderedInterfaces.end());
  }

private:
  /// A set of registered dialect interface instances.
  InterfaceSetT interfaces;
  /// An ordered list of the registered interface instances, necessary for
  /// deterministic iteration.
  // NOTE: SetVector does not provide find access, so it can't be used here.
  InterfaceVectorT orderedInterfaces;
};
} // namespace detail

/// A collection of dialect interfaces within a context, for a given concrete
/// interface type.
template <typename InterfaceType>
class DialectInterfaceCollection
    : public detail::DialectInterfaceCollectionBase {
public:
  using Base = DialectInterfaceCollection<InterfaceType>;

  /// Collect the registered dialect interfaces within the provided context.
  DialectInterfaceCollection(MLIRContext *ctx)
      : detail::DialectInterfaceCollectionBase(
            ctx, InterfaceType::getInterfaceID()) {}

  /// Get the interface for a given object, or null if one is not registered.
  /// The object may be a dialect or an operation instance.
  template <typename Object>
  const InterfaceType *getInterfaceFor(Object *obj) const {
    return static_cast<const InterfaceType *>(
        detail::DialectInterfaceCollectionBase::getInterfaceFor(obj));
  }

  /// Iterator access to the held interfaces.
  using iterator =
      detail::DialectInterfaceCollectionBase::iterator<InterfaceType>;
  iterator begin() const { return interface_begin<InterfaceType>(); }
  iterator end() const { return interface_end<InterfaceType>(); }

private:
  using detail::DialectInterfaceCollectionBase::interface_begin;
  using detail::DialectInterfaceCollectionBase::interface_end;
};

} // namespace mlir

#endif

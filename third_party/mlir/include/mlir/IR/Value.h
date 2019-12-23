//===- Value.h - Base of the SSA Value hierarchy ----------------*- C++ -*-===//
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
// This file defines generic Value type and manipulation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_VALUE_H
#define MLIR_IR_VALUE_H

#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Block;
class BlockArgument;
class Operation;
class OpResult;
class Region;
class Value;

namespace detail {
/// The internal implementation of a Value.
class ValueImpl : public IRObjectWithUseList {
protected:
  /// This enumerates all of the SSA value kinds.
  enum class Kind {
    BlockArgument,
    OpResult,
  };

  ValueImpl(Kind kind, Type type) : typeAndKind(type, kind) {}

private:
  /// The type of the value and its kind.
  llvm::PointerIntPair<Type, 1, Kind> typeAndKind;

  /// Allow access to 'typeAndKind'.
  friend Value;
};

/// The internal implementation of a BlockArgument.
class BlockArgumentImpl : public ValueImpl {
  BlockArgumentImpl(Type type, Block *owner)
      : ValueImpl(Kind::BlockArgument, type), owner(owner) {}

  /// The owner of this argument.
  Block *owner;

  /// Allow access to owner and constructor.
  friend BlockArgument;
};

class OpResultImpl : public ValueImpl {
  OpResultImpl(Type type, Operation *owner)
      : ValueImpl(Kind::OpResult, type), owner(owner) {}

  /// The owner of this result.
  Operation *owner;

  /// Allow access to owner and the constructor.
  friend OpResult;
};
} // end namespace detail

/// This class represents an instance of an SSA value in the MLIR system,
/// representing a computable value that has a type and a set of users. An SSA
/// value is either a BlockArgument or the result of an operation. Note: This
/// class has value-type semantics and is just a simple wrapper around a
/// ValueImpl that is either owner by a block(in the case of a BlockArgument) or
/// an Operation(in the case of an OpResult).
///
class Value {
public:
  /// This enumerates all of the SSA value kinds in the MLIR system.
  enum class Kind {
    BlockArgument,
    OpResult,
  };

  Value(std::nullptr_t) : impl(nullptr) {}
  Value(detail::ValueImpl *impl = nullptr) : impl(impl) {}
  Value(const Value &) = default;
  Value &operator=(const Value &) = default;
  ~Value() {}

  template <typename U> bool isa() const {
    assert(impl && "isa<> used on a null type.");
    return U::classof(*this);
  }
  template <typename U> U dyn_cast() const {
    return isa<U>() ? U(impl) : U(nullptr);
  }
  template <typename U> U dyn_cast_or_null() const {
    return (impl && isa<U>()) ? U(impl) : U(nullptr);
  }
  template <typename U> U cast() const {
    assert(isa<U>());
    return U(impl);
  }

  /// Temporary methods to enable transition of Value to being used as a
  /// value-type.
  /// TODO(riverriddle) Remove these when all usages have been removed.
  Value operator*() const { return *this; }
  Value *operator->() const { return (Value *)this; }

  operator bool() const { return impl; }
  bool operator==(const Value &other) const { return impl == other.impl; }
  bool operator!=(const Value &other) const { return !(*this == other); }

  /// Return the kind of this value.
  Kind getKind() const { return (Kind)impl->typeAndKind.getInt(); }

  /// Return the type of this value.
  Type getType() const { return impl->typeAndKind.getPointer(); }

  /// Utility to get the associated MLIRContext that this value is defined in.
  MLIRContext *getContext() const { return getType().getContext(); }

  /// Mutate the type of this Value to be of the specified type.
  ///
  /// Note that this is an extremely dangerous operation which can create
  /// completely invalid IR very easily.  It is strongly recommended that you
  /// recreate IR objects with the right types instead of mutating them in
  /// place.
  void setType(Type newType) { impl->typeAndKind.setPointer(newType); }

  /// Replace all uses of 'this' value with the new value, updating anything in
  /// the IR that uses 'this' to use the other value instead.  When this returns
  /// there are zero uses of 'this'.
  void replaceAllUsesWith(Value newValue) const {
    impl->replaceAllUsesWith(newValue.impl);
  }

  /// If this value is the result of an operation, return the operation that
  /// defines it.
  Operation *getDefiningOp() const;

  /// If this value is the result of an operation, use it as a location,
  /// otherwise return an unknown location.
  Location getLoc();

  /// Return the Region in which this Value is defined.
  Region *getParentRegion();

  using use_iterator = ValueUseIterator<OpOperand>;
  using use_range = iterator_range<use_iterator>;

  inline use_iterator use_begin();
  inline use_iterator use_end();

  /// Returns a range of all uses, which is useful for iterating over all uses.
  inline use_range getUses();

  using user_iterator = ValueUserIterator<IROperand>;
  using user_range = iterator_range<user_iterator>;

  user_iterator user_begin() const { return impl->user_begin(); }
  user_iterator user_end() const { return impl->user_end(); }

  /// Returns a range of all users.
  user_range getUsers() const { return impl->getUsers(); }

  /// Returns true if this value has no uses.
  bool use_empty() const { return impl->use_empty(); }

  /// Returns true if this value has exactly one use.
  bool hasOneUse() const { return impl->hasOneUse(); }

  /// Drop all uses of this object from their respective owners.
  void dropAllUses() const { impl->dropAllUses(); }

  void print(raw_ostream &os);
  void dump();

  /// Methods for supporting PointerLikeTypeTraits.
  void *getAsOpaquePointer() const { return static_cast<void *>(impl); }
  static Value getFromOpaquePointer(const void *pointer) {
    return reinterpret_cast<detail::ValueImpl *>(const_cast<void *>(pointer));
  }

  friend ::llvm::hash_code hash_value(Value arg);

protected:
  /// The internal implementation of this value.
  mutable detail::ValueImpl *impl;

  /// Allow access to 'impl'.
  friend OpOperand;
};

inline raw_ostream &operator<<(raw_ostream &os, Value value) {
  value.print(os);
  return os;
}

// Utility functions for iterating through Value uses.
inline auto Value::use_begin() -> use_iterator {
  return use_iterator((OpOperand *)impl->getFirstUse());
}

inline auto Value::use_end() -> use_iterator { return use_iterator(nullptr); }

inline auto Value::getUses() -> iterator_range<use_iterator> {
  return {use_begin(), use_end()};
}

/// Block arguments are values.
class BlockArgument : public Value {
public:
  using Value::Value;

  /// Temporary methods to enable transition of Value to being used as a
  /// value-type.
  /// TODO(riverriddle) Remove this when all usages have been removed.
  BlockArgument *operator->() { return this; }

  static bool classof(Value value) {
    return value.getKind() == Kind::BlockArgument;
  }

  /// Returns the block that owns this argument.
  Block *getOwner() const { return getImpl()->owner; }

  /// Returns the number of this argument.
  unsigned getArgNumber() const;

private:
  /// Allocate a new argument with the given type and owner.
  static BlockArgument create(Type type, Block *owner) {
    return new detail::BlockArgumentImpl(type, owner);
  }

  /// Destroy and deallocate this argument.
  void destroy() { delete getImpl(); }

  /// Get a raw pointer to the internal implementation.
  detail::BlockArgumentImpl *getImpl() const {
    return reinterpret_cast<detail::BlockArgumentImpl *>(impl);
  }

  /// Allow access to `create` and `destroy`.
  friend Block;
};

/// This is a value defined by a result of an operation.
class OpResult : public Value {
public:
  using Value::Value;

  /// Temporary methods to enable transition of Value to being used as a
  /// value-type.
  /// TODO(riverriddle) Remove these when all usages have been removed.
  OpResult *operator*() { return this; }
  OpResult *operator->() { return this; }

  static bool classof(Value value) { return value.getKind() == Kind::OpResult; }

  /// Returns the operation that owns this result.
  Operation *getOwner() const { return getImpl()->owner; }

  /// Returns the number of this result.
  unsigned getResultNumber() const;

private:
  /// Allocate a new result with the given type and owner.
  static OpResult create(Type type, Operation *owner) {
    return new detail::OpResultImpl(type, owner);
  }

  /// Destroy and deallocate this result.
  void destroy() { delete getImpl(); }

  /// Get a raw pointer to the internal implementation.
  detail::OpResultImpl *getImpl() const {
    return reinterpret_cast<detail::OpResultImpl *>(impl);
  }

  /// Allow access to `create` and `destroy`.
  friend Operation;
};

/// Make Value hashable.
inline ::llvm::hash_code hash_value(Value arg) {
  return ::llvm::hash_value(arg.impl);
}

} // namespace mlir

namespace llvm {

template <> struct DenseMapInfo<mlir::Value> {
  static mlir::Value getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::Value(static_cast<mlir::detail::ValueImpl *>(pointer));
  }
  static mlir::Value getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::Value(static_cast<mlir::detail::ValueImpl *>(pointer));
  }
  static unsigned getHashValue(mlir::Value val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::Value LHS, mlir::Value RHS) { return LHS == RHS; }
};

/// Allow stealing the low bits of a value.
template <> struct PointerLikeTypeTraits<mlir::Value> {
public:
  static inline void *getAsVoidPointer(mlir::Value I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::Value getFromVoidPointer(void *P) {
    return mlir::Value::getFromOpaquePointer(P);
  }
  enum {
    NumLowBitsAvailable =
        PointerLikeTypeTraits<mlir::detail::ValueImpl *>::NumLowBitsAvailable
  };
};

template <> struct DenseMapInfo<mlir::BlockArgument> {
  static mlir::BlockArgument getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::BlockArgument(static_cast<mlir::detail::ValueImpl *>(pointer));
  }
  static mlir::BlockArgument getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::BlockArgument(static_cast<mlir::detail::ValueImpl *>(pointer));
  }
  static unsigned getHashValue(mlir::BlockArgument val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::BlockArgument LHS, mlir::BlockArgument RHS) {
    return LHS == RHS;
  }
};

/// Allow stealing the low bits of a value.
template <> struct PointerLikeTypeTraits<mlir::BlockArgument> {
public:
  static inline void *getAsVoidPointer(mlir::Value I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::BlockArgument getFromVoidPointer(void *P) {
    return mlir::Value::getFromOpaquePointer(P).cast<mlir::BlockArgument>();
  }
  enum {
    NumLowBitsAvailable =
        PointerLikeTypeTraits<mlir::detail::ValueImpl *>::NumLowBitsAvailable
  };
};
} // end namespace llvm

#endif

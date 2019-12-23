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

/// Using directives that simplify the transition of Value to being value typed.
using BlockArgumentPtr = BlockArgument *;
using OpResultPtr = OpResult *;
using ValueRef = Value &;
using ValuePtr = Value *;

/// Operands contain a Value.
using OpOperand = IROperandImpl<Value>;

/// This is the common base class for all SSA values in the MLIR system,
/// representing a computable value that has a type and a set of users.
///
class Value : public IRObjectWithUseList {
public:
  /// This enumerates all of the SSA value kinds in the MLIR system.
  enum class Kind {
    BlockArgument, // block argument
    OpResult,      // operation result
  };

  ~Value() {}

  template <typename U> bool isa() const { return U::classof(this); }
  template <typename U> U *dyn_cast() const {
    return isa<U>() ? (U *)this : nullptr;
  }
  template <typename U> U *cast() const {
    assert(isa<U>());
    return (U *)this;
  }

  Kind getKind() const { return typeAndKind.getInt(); }

  Type getType() const { return typeAndKind.getPointer(); }

  /// Utility to get the associated MLIRContext that this value is defined in.
  MLIRContext *getContext() const { return getType().getContext(); }

  /// Mutate the type of this Value to be of the specified type.
  ///
  /// Note that this is an extremely dangerous operation which can create
  /// completely invalid IR very easily.  It is strongly recommended that you
  /// recreate IR objects with the right types instead of mutating them in
  /// place.
  void setType(Type newType) { typeAndKind.setPointer(newType); }

  /// Replace all uses of 'this' value with the new value, updating anything in
  /// the IR that uses 'this' to use the other value instead.  When this returns
  /// there are zero uses of 'this'.
  void replaceAllUsesWith(ValuePtr newValue) {
    IRObjectWithUseList::replaceAllUsesWith(newValue);
  }

  /// If this value is the result of an operation, return the operation that
  /// defines it.
  Operation *getDefiningOp();

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

  void print(raw_ostream &os);
  void dump();

protected:
  Value(Kind kind, Type type) : typeAndKind(type, kind) {}

private:
  llvm::PointerIntPair<Type, 1, Kind> typeAndKind;
};

inline raw_ostream &operator<<(raw_ostream &os, ValueRef value) {
  value.print(os);
  return os;
}

// Utility functions for iterating through Value uses.
inline auto Value::use_begin() -> use_iterator {
  return use_iterator((OpOperand *)getFirstUse());
}

inline auto Value::use_end() -> use_iterator { return use_iterator(nullptr); }

inline auto Value::getUses() -> iterator_range<use_iterator> {
  return {use_begin(), use_end()};
}

/// Block arguments are values.
class BlockArgument : public Value {
public:
  static bool classof(const Value *value) {
    return const_cast<Value *>(value)->getKind() == Kind::BlockArgument;
  }

  Block *getOwner() { return owner; }

  /// Returns the number of this argument.
  unsigned getArgNumber();

private:
  friend class Block; // For access to private constructor.
  BlockArgument(Type type, Block *owner)
      : Value(Value::Kind::BlockArgument, type), owner(owner) {}

  /// The owner of this operand.
  /// TODO: can encode this more efficiently to avoid the space hit of this
  /// through bitpacking shenanigans.
  Block *const owner;
};

/// This is a value defined by a result of an operation.
class OpResult : public Value {
public:
  OpResult(Type type, Operation *owner)
      : Value(Value::Kind::OpResult, type), owner(owner) {}

  static bool classof(const Value *value) {
    return const_cast<Value *>(value)->getKind() == Kind::OpResult;
  }

  Operation *getOwner() { return owner; }

  /// Returns the number of this result.
  unsigned getResultNumber();

private:
  /// The owner of this operand.
  /// TODO: can encode this more efficiently to avoid the space hit of this
  /// through bitpacking shenanigans.
  Operation *const owner;
};
} // namespace mlir

#endif

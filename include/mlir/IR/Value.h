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
class Function;
class Operation;
class Value;

/// Operands contain a Value.
using InstOperand = IROperandImpl<Value>;

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

  Kind getKind() { return typeAndKind.getInt(); }

  Type getType() { return typeAndKind.getPointer(); }

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
  void replaceAllUsesWith(Value *newValue) {
    IRObjectWithUseList::replaceAllUsesWith(newValue);
  }

  /// Return the function that this Value is defined in.
  Function *getFunction();

  /// If this value is the result of an operation, return the operation that
  /// defines it.
  Operation *getDefiningOp();

  using use_iterator = ValueUseIterator<InstOperand>;
  using use_range = llvm::iterator_range<use_iterator>;

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

inline raw_ostream &operator<<(raw_ostream &os, Value &value) {
  value.print(os);
  return os;
}

// Utility functions for iterating through Value uses.
inline auto Value::use_begin() -> use_iterator {
  return use_iterator((InstOperand *)getFirstUse());
}

inline auto Value::use_end() -> use_iterator { return use_iterator(nullptr); }

inline auto Value::getUses() -> llvm::iterator_range<use_iterator> {
  return {use_begin(), use_end()};
}

/// Block arguments are values.
class BlockArgument : public Value {
public:
  static bool classof(const Value *value) {
    return const_cast<Value *>(value)->getKind() == Kind::BlockArgument;
  }

  /// Return the function that this argument is defined in.
  Function *getFunction();

  Block *getOwner() { return owner; }

  /// Returns the number of this argument.
  unsigned getArgNumber();

  /// Returns if the current argument is a function argument.
  bool isFunctionArgument();

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

/// This is a helper template used to implement an iterator that contains a
/// pointer to some object and an index into it.  The iterator moves the
/// index but keeps the object constant.
template <typename ConcreteType, typename ObjectType, typename ElementType>
class IndexedAccessorIterator
    : public llvm::iterator_facade_base<
          ConcreteType, std::random_access_iterator_tag, ElementType *,
          std::ptrdiff_t, ElementType *, ElementType *> {
public:
  ptrdiff_t operator-(const IndexedAccessorIterator &rhs) const {
    assert(object == rhs.object && "incompatible iterators");
    return index - rhs.index;
  }
  bool operator==(const IndexedAccessorIterator &rhs) const {
    return object == rhs.object && index == rhs.index;
  }
  bool operator<(const IndexedAccessorIterator &rhs) const {
    assert(object == rhs.object && "incompatible iterators");
    return index < rhs.index;
  }

  ConcreteType &operator+=(ptrdiff_t offset) {
    this->index += offset;
    return static_cast<ConcreteType &>(*this);
  }
  ConcreteType &operator-=(ptrdiff_t offset) {
    this->index -= offset;
    return static_cast<ConcreteType &>(*this);
  }

protected:
  IndexedAccessorIterator(ObjectType *object, unsigned index)
      : object(object), index(index) {}
  ObjectType *object;
  unsigned index;
};

} // namespace mlir

#endif

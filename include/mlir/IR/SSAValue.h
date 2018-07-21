//===- SSAValue.h - Base of the value hierarchy -----------------*- C++ -*-===//
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
// This file defines generic SSAValue type and manipulation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_SSAVALUE_H
#define MLIR_IR_SSAVALUE_H

#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/iterator_range.h"

namespace mlir {
class OperationInst;
class SSAOperand;
template <typename OperandType, typename OwnerType> class SSAValueUseIterator;

/// This enumerates all of the SSA value kinds in the MLIR system.
enum class SSAValueKind {
  // TODO: BBArg,
  InstResult,

  // FnArg
  // StmtResult
  // ForStmt
};

/// This is the common base class for all values in the MLIR system,
/// representing a computable value that has a type and a set of users.
///
class SSAValue {
public:
  ~SSAValue() {
    assert(use_empty() && "Cannot destroy a value that still has uses!");
  }

  SSAValueKind getKind() const { return typeAndKind.getInt(); }

  Type *getType() const { return typeAndKind.getPointer(); }

  /// Returns true if this value has no uses.
  bool use_empty() const { return firstUse == nullptr; }

  /// Returns true if this value has exactly one use.
  inline bool hasOneUse() const;

  using use_iterator = SSAValueUseIterator<SSAOperand, void>;
  using use_range = llvm::iterator_range<use_iterator>;

  inline use_iterator use_begin() const;
  inline use_iterator use_end() const;

  /// Returns a range of all uses, which is useful for iterating over all uses.
  inline use_range getUses() const;

  /// Replace all uses of 'this' value with the new value, updating anything in
  /// the IR that uses 'this' to use the other value instead.  When this returns
  /// there are zero uses of 'this'.
  void replaceAllUsesWith(SSAValue *newValue);

  /// If this value is the result of an OperationInst, return the instruction
  /// that defines it.
  OperationInst *getDefiningInst();
  const OperationInst *getDefiningInst() const {
    return const_cast<SSAValue *>(this)->getDefiningInst();
  }

protected:
  SSAValue(SSAValueKind kind, Type *type) : typeAndKind(type, kind) {}

private:
  friend class SSAOperand;
  const llvm::PointerIntPair<Type *, 3, SSAValueKind> typeAndKind;
  SSAOperand *firstUse = nullptr;
};

/// This template unifies the implementation logic for CFGValue and StmtValue
/// while providing more type-specific APIs when walking use lists etc.
///
/// SSAOperandTy is the concrete instance of SSAOperand to use (including
/// substituted template arguments) and KindTy is the enum 'kind' discriminator
/// that subclasses want to use.
///
template <typename SSAOperandTy, typename KindTy>
class SSAValueImpl : public SSAValue {
public:
  // Provide more specific implementations of the base class functionality.
  KindTy getKind() const { return (KindTy)SSAValue::getKind(); }

  // TODO: using use_iterator = SSAValueUseIterator<SSAOperandTy>;
  // TODO: using use_range = llvm::iterator_range<use_iterator>;

  // TODO: inline use_iterator use_begin() const;
  // TODO: inline use_iterator use_end() const;

  /// Returns a range of all uses, which is useful for iterating over all uses.
  // TODO: inline use_range getUses() const;

protected:
  SSAValueImpl(KindTy kind, Type *type) : SSAValue((SSAValueKind)kind, type) {}
};

// FIXME: Implement SSAValueUseIterator here.

/// An iterator over all uses of a ValueBase.
template <typename OperandType, typename OwnerType>
class SSAValueUseIterator
    : public std::iterator<std::forward_iterator_tag, SSAOperand> {
public:
  SSAValueUseIterator() = default;
  explicit SSAValueUseIterator(SSAOperand *current) : current(current) {}
  OperandType *operator->() const { return current; }
  OperandType &operator*() const { return current; }

  template<typename SFINAE_Owner = OwnerType>
  typename std::enable_if<!std::is_void<OwnerType>::value, SFINAE_Owner>::type
  getUser() const {
    return current->getOwner();
  }

  SSAValueUseIterator &operator++() {
    assert(current && "incrementing past end()!");
    current = (OperandType *)current->getNextOperandUsingThisValue();
    return *this;
  }

  SSAValueUseIterator operator++(int unused) {
    SSAValueUseIterator copy = *this;
    ++*this;
    return copy;
  }

  friend bool operator==(SSAValueUseIterator lhs, SSAValueUseIterator rhs) {
    return lhs.current == rhs.current;
  }

  friend bool operator!=(SSAValueUseIterator lhs, SSAValueUseIterator rhs) {
    return !(lhs == rhs);
  }

private:
  OperandType *current;
};

} // namespace mlir

#endif

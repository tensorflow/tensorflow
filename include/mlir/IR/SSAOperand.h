//===- SSAOperand.h ---------------------------------------------*- C++ -*-===//
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
// This file defines generic SSA operand and manipulation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_SSAOPERAND_H
#define MLIR_IR_SSAOPERAND_H

#include "mlir/IR/SSAValue.h"

namespace mlir {
class SSAValue;

/// A reference to a value, suitable for use as an operand of an instruction,
/// statement, etc.
class IROperand {
public:
  IROperand() {}
  IROperand(IRObjectWithUseList *value) : value(value) { insertIntoCurrent(); }

  /// Return the current value being used by this operand.
  IRObjectWithUseList *get() const { return value; }

  /// Set the current value being used by this operand.
  void set(IRObjectWithUseList *newValue) {
    // It isn't worth optimizing for the case of switching operands on a single
    // value.
    removeFromCurrent();
    value = newValue;
    insertIntoCurrent();
  }

  /// \brief Remove this use of the operand.
  void drop() {
    removeFromCurrent();
    value = nullptr;
    nextUse = nullptr;
    back = nullptr;
  }

  ~IROperand() { removeFromCurrent(); }

  /// Return the next operand on the use-list of the value we are referring to.
  /// This should generally only be used by the internal implementation details
  /// of the SSA machinery.
  IROperand *getNextOperandUsingThisValue() { return nextUse; }

  /// We support a move constructor so IROperand's can be in vectors, but this
  /// shouldn't be used by general clients.
  IROperand(IROperand &&other) {
    other.removeFromCurrent();
    value = other.value;
    other.value = nullptr;
    nextUse = nullptr;
    back = nullptr;
    insertIntoCurrent();
  }

private:
  /// The value used as this operand.  This can be null when in a
  /// "dropAllUses" state.
  IRObjectWithUseList *value = nullptr;

  /// The next operand in the use-chain.
  IROperand *nextUse = nullptr;

  /// This points to the previous link in the use-chain.
  IROperand **back = nullptr;

  /// Operands are not copyable or assignable.
  IROperand(const IROperand &use) = delete;
  IROperand &operator=(const IROperand &use) = delete;

  void removeFromCurrent() {
    if (!back)
      return;
    *back = nextUse;
    if (nextUse)
      nextUse->back = back;
  }

  void insertIntoCurrent() {
    back = &value->firstUse;
    nextUse = value->firstUse;
    if (nextUse)
      nextUse->back = &nextUse;
    value->firstUse = this;
  }
};

/// A reference to a value, suitable for use as an operand of an instruction,
/// statement, etc.  IRValueTy is the root type to use for values this tracks,
/// and SSAUserTy is the type that will contain operands.
template <typename IRValueTy, typename IROwnerTy>
class IROperandImpl : public IROperand {
public:
  IROperandImpl(IROwnerTy *owner) : owner(owner) {}
  IROperandImpl(IROwnerTy *owner, IRValueTy *value)
      : IROperand(value), owner(owner) {}

  /// Return the current value being used by this operand.
  IRValueTy *get() const { return (IRValueTy *)IROperand::get(); }

  /// Set the current value being used by this operand.
  void set(IRValueTy *newValue) { IROperand::set(newValue); }

  /// Return the user that owns this use.
  IROwnerTy *getOwner() { return owner; }
  const IROwnerTy *getOwner() const { return owner; }

  /// Return which operand this is in the operand list of the User.
  // TODO:  unsigned getOperandNumber() const;

  /// We support a move constructor so IROperand's can be in vectors, but this
  /// shouldn't be used by general clients.
  IROperandImpl(IROperandImpl &&other)
      : IROperand(std::move(other)), owner(other.owner) {}

private:
  /// The owner of this operand.
  IROwnerTy *const owner;
};

inline auto IRObjectWithUseList::use_begin() const -> use_iterator {
  return use_iterator(firstUse);
}

inline auto IRObjectWithUseList::use_end() const -> use_iterator {
  return use_iterator(nullptr);
}

inline auto IRObjectWithUseList::getUses() const
    -> llvm::iterator_range<use_iterator> {
  return {use_begin(), use_end()};
}

/// Returns true if this value has exactly one use.
inline bool IRObjectWithUseList::hasOneUse() const {
  return firstUse && firstUse->getNextOperandUsingThisValue() == nullptr;
}

} // namespace mlir

#endif

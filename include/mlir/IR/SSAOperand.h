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
class SSAOperand {
public:
  SSAOperand() {}
  SSAOperand(SSAValue *value) : value(value) { insertIntoCurrent(); }

  /// Return the current value being used by this operand.
  SSAValue *get() const { return value; }

  /// Set the current value being used by this operand.
  void set(SSAValue *newValue) {
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

  ~SSAOperand() { removeFromCurrent(); }

  /// Return the next operand on the use-list of the value we are referring to.
  /// This should generally only be used by the internal implementation details
  /// of the SSA machinery.
  SSAOperand *getNextOperandUsingThisValue() { return nextUse; }

private:
  /// The value used as this operand.  This can be null when in a
  /// "dropAllUses" state.
  SSAValue *value = nullptr;

  /// The next operand in the use-chain.
  SSAOperand *nextUse = nullptr;

  /// This points to the previous link in the use-chain.
  SSAOperand **back = nullptr;

  /// Operands are not copyable or assignable.
  SSAOperand(const SSAOperand &use) = delete;
  SSAOperand &operator=(const SSAOperand &use) = delete;

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
/// statement, etc.  SSAValueTy is the root type to use for values this tracks,
/// and SSAUserTy is the type that will contain operands.
template <typename SSAValueTy, typename SSAOwnerTy>
class SSAOperandImpl : public SSAOperand {
public:
  SSAOperandImpl(SSAOwnerTy *owner) : owner(owner) {}
  SSAOperandImpl(SSAOwnerTy *owner, SSAValueTy *value)
      : SSAOperand(value), owner(owner) {}

  /// Return the current value being used by this operand.
  SSAValueTy *get() const { return (SSAValueTy *)SSAOperand::get(); }

  /// Set the current value being used by this operand.
  void set(SSAValueTy *newValue) { SSAOperand::set(newValue); }

  /// Return the user that owns this use.
  SSAOwnerTy *getOwner() { return owner; }
  const SSAOwnerTy *getOwner() const { return owner; }

  /// Return which operand this is in the operand list of the User.
  // TODO:  unsigned getOperandNumber() const;

private:
  /// The owner of this operand.
  SSAOwnerTy *const owner;
};

inline auto SSAValue::use_begin() const -> use_iterator {
  return SSAValue::use_iterator(firstUse);
}

inline auto SSAValue::use_end() const -> use_iterator {
  return SSAValue::use_iterator(nullptr);
}

inline auto SSAValue::getUses() const -> llvm::iterator_range<use_iterator> {
  return {use_begin(), use_end()};
}

/// Returns true if this value has exactly one use.
inline bool SSAValue::hasOneUse() const {
  return firstUse && firstUse->getNextOperandUsingThisValue() == nullptr;
}

} // namespace mlir

#endif

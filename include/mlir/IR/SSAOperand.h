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

namespace mlir {

/// A reference to a value, suitable for use as an operand of an instruction,
/// statement, etc.  SSAValueTy is the root type to use for values this tracks,
/// and SSAUserTy is the type that will contain operands.
template <typename SSAValueTy, typename SSAUserTy>
class SSAOperand {
public:
  SSAOperand(SSAUserTy *user) : user(user) {}
  SSAOperand(SSAUserTy *user, SSAValueTy *value) : value(value), user(user) {
    insertIntoCurrent();
  }

  /// Return the current value being used by this operand.
  SSAValueTy *get() const { return value; }

  /// Set the current value being used by this operand.
  void set(SSAValueTy *newValue) {
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

  /// Return the user that owns this use.
  SSAUserTy *getUser() { return user; }
  const SSAUserTy *getUser() const { return user; }

  /// Return which operand this is in the operand list of the User.
  // TODO:  unsigned getOperandNumber() const;

private:
  /// The value used as this operand.  This can be transiently null when in a
  /// "dropAllUses" state.
  SSAValueTy *value = nullptr;

  /// The next operand in the use-chain.
  SSAOperand *nextUse = nullptr;

  /// This points to the previous link in the use-chain.
  SSAOperand **back = nullptr;

  /// The user of this operand.
  SSAUserTy *const user;

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

} // namespace mlir

#endif

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
#include "mlir/IR/UseDefLists.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Function;
class Instruction;
class OperationStmt;
class Operation;

/// This enumerates all of the SSA value kinds in the MLIR system.
enum class SSAValueKind {
  BBArgument,     // basic block argument
  InstResult,     // instruction result
  BlockArgument,  // Block argument
  StmtResult,     // statement result
  ForStmt,        // for statement induction variable
};

/// This is the common base class for all values in the MLIR system,
/// representing a computable value that has a type and a set of users.
///
class SSAValue : public IRObjectWithUseList {
public:
  ~SSAValue() {}

  SSAValueKind getKind() const { return typeAndKind.getInt(); }

  Type getType() const { return typeAndKind.getPointer(); }

  /// Replace all uses of 'this' value with the new value, updating anything in
  /// the IR that uses 'this' to use the other value instead.  When this returns
  /// there are zero uses of 'this'.
  void replaceAllUsesWith(SSAValue *newValue) {
    IRObjectWithUseList::replaceAllUsesWith(newValue);
  }

  /// Return the function that this SSAValue is defined in.
  Function *getFunction();

  /// Return the function that this SSAValue is defined in.
  const Function *getFunction() const {
    return const_cast<SSAValue *>(this)->getFunction();
  }

  /// If this value is the result of an Instruction, return the instruction
  /// that defines it.
  Instruction *getDefiningInst();
  const Instruction *getDefiningInst() const {
    return const_cast<SSAValue *>(this)->getDefiningInst();
  }

  /// If this value is the result of an OperationStmt, return the statement
  /// that defines it.
  OperationStmt *getDefiningStmt();
  const OperationStmt *getDefiningStmt() const {
    return const_cast<SSAValue *>(this)->getDefiningStmt();
  }

  /// If this value is the result of an Operation, return the operation that
  /// defines it.
  Operation *getDefiningOperation();
  const Operation *getDefiningOperation() const {
    return const_cast<SSAValue *>(this)->getDefiningOperation();
  }

  void print(raw_ostream &os) const;
  void dump() const;

protected:
  SSAValue(SSAValueKind kind, Type type) : typeAndKind(type, kind) {}

private:
  const llvm::PointerIntPair<Type, 3, SSAValueKind> typeAndKind;
};

inline raw_ostream &operator<<(raw_ostream &os, const SSAValue &value) {
  value.print(os);
  return os;
}

/// This template unifies the implementation logic for CFGValue and MLValue
/// while providing more type-specific APIs when walking use lists etc.
///
/// IROperandTy is the concrete instance of IROperand to use (including
/// substituted template arguments).
/// IROwnerTy is the type of the owner of an IROperandTy type.
/// KindTy is the enum 'kind' discriminator that subclasses want to use.
///
template <typename IROperandTy, typename IROwnerTy, typename KindTy>
class SSAValueImpl : public SSAValue {
public:
  // Provide more specific implementations of the base class functionality.
  KindTy getKind() const { return (KindTy)SSAValue::getKind(); }

  using use_iterator = SSAValueUseIterator<IROperandTy, IROwnerTy>;
  using use_range = llvm::iterator_range<use_iterator>;

  inline use_iterator use_begin() const;
  inline use_iterator use_end() const;

  /// Returns a range of all uses, which is useful for iterating over all uses.
  inline use_range getUses() const;

protected:
  SSAValueImpl(KindTy kind, Type type) : SSAValue((SSAValueKind)kind, type) {}
};

// Utility functions for iterating through SSAValue uses.
template <typename IROperandTy, typename IROwnerTy, typename KindTy>
inline auto SSAValueImpl<IROperandTy, IROwnerTy, KindTy>::use_begin() const
    -> use_iterator {
  return use_iterator((IROperandTy *)getFirstUse());
}

template <typename IROperandTy, typename IROwnerTy, typename KindTy>
inline auto SSAValueImpl<IROperandTy, IROwnerTy, KindTy>::use_end() const
    -> use_iterator {
  return use_iterator(nullptr);
}

template <typename IROperandTy, typename IROwnerTy, typename KindTy>
inline auto SSAValueImpl<IROperandTy, IROwnerTy, KindTy>::getUses() const
    -> llvm::iterator_range<use_iterator> {
  return {use_begin(), use_end()};
}

} // namespace mlir

#endif

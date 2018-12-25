//===- CFGValue.h - CFGValue base class and SSA type decls ------*- C++ -*-===//
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
// This file defines SSA manipulation implementations for CFG functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_CFGVALUE_H
#define MLIR_IR_CFGVALUE_H

#include "mlir/IR/SSAValue.h"

namespace mlir {
class BasicBlock;
class CFGValue;
class CFGFunction;
class Instruction;

/// This enum contains all of the SSA value kinds that are valid in a CFG
/// function.  This should be kept as a proper subtype of SSAValueKind,
/// including having all of the values of the enumerators align.
enum class CFGValueKind {
  BBArgument = (int)SSAValueKind::BBArgument,
  InstResult = (int)SSAValueKind::InstResult,
};

/// The operand of a CFG Instruction contains a CFGValue.
using InstOperand = IROperandImpl<CFGValue, Instruction>;

/// CFGValue is the base class for SSA values in CFG functions.
class CFGValue : public SSAValueImpl<InstOperand, Instruction, CFGValueKind> {
public:
  static bool classof(const SSAValue *value) {
    switch (value->getKind()) {
    case SSAValueKind::BBArgument:
    case SSAValueKind::InstResult:
      return true;

    case SSAValueKind::MLFuncArgument:
    case SSAValueKind::BlockArgument:
    case SSAValueKind::StmtResult:
    case SSAValueKind::ForStmt:
      return false;
    }
  }

  /// Return the function that this CFGValue is defined in.
  CFGFunction *getFunction();

  /// Return the function that this CFGValue is defined in.
  const CFGFunction *getFunction() const {
    return const_cast<CFGValue *>(this)->getFunction();
  }

protected:
  CFGValue(CFGValueKind kind, Type type) : SSAValueImpl(kind, type) {}
};

/// Basic block arguments are CFG Values.
class BBArgument : public CFGValue {
public:
  static bool classof(const SSAValue *value) {
    return value->getKind() == SSAValueKind::BBArgument;
  }

  /// Return the function that this argument is defined in.
  CFGFunction *getFunction();
  const CFGFunction *getFunction() const {
    return const_cast<BBArgument *>(this)->getFunction();
  }

  BasicBlock *getOwner() { return owner; }
  const BasicBlock *getOwner() const { return owner; }

private:
  friend class BasicBlock; // For access to private constructor.
  BBArgument(Type type, BasicBlock *owner)
      : CFGValue(CFGValueKind::BBArgument, type), owner(owner) {}

  /// The owner of this operand.
  /// TODO: can encode this more efficiently to avoid the space hit of this
  /// through bitpacking shenanigans.
  BasicBlock *const owner;
};

/// Instruction results are CFG Values.
class InstResult : public CFGValue {
public:
  InstResult(Type type, Instruction *owner)
      : CFGValue(CFGValueKind::InstResult, type), owner(owner) {}

  static bool classof(const SSAValue *value) {
    return value->getKind() == SSAValueKind::InstResult;
  }

  Instruction *getOwner() { return owner; }
  const Instruction *getOwner() const { return owner; }

  /// Return the number of this result.
  unsigned getResultNumber() const;

private:
  /// The owner of this operand.
  /// TODO: can encode this more efficiently to avoid the space hit of this
  /// through bitpacking shenanigans.
  Instruction *const owner;
};

} // namespace mlir

#endif

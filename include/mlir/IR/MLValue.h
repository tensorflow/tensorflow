//===- MLValue.h - MLValue base class and SSA type decls ------*- C++ -*-===//
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
// This file defines SSA manipulation implementations for ML functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_MLVALUE_H
#define MLIR_IR_MLVALUE_H

#include "mlir/IR/SSAValue.h"

namespace mlir {
class ForStmt;
class MLValue;
using MLFunction = Function;
class Statement;
class StmtBlock;

/// This enum contains all of the SSA value kinds that are valid in an ML
/// function.  This should be kept as a proper subtype of SSAValueKind,
/// including having all of the values of the enumerators align.
enum class MLValueKind {
  BlockArgument = (int)SSAValueKind::BlockArgument,
  StmtResult = (int)SSAValueKind::StmtResult,
  ForStmt = (int)SSAValueKind::ForStmt,
};

/// The operand of ML function statement contains an MLValue.
using StmtOperand = IROperandImpl<MLValue, Statement>;

/// MLValue is the base class for SSA values in ML functions.
class MLValue : public SSAValueImpl<StmtOperand, Statement, MLValueKind> {
public:
  /// Returns true if the given MLValue can be used as a dimension id.
  bool isValidDim() const;

  /// Returns true if the given MLValue can be used as a symbol.
  bool isValidSymbol() const;

  static bool classof(const SSAValue *value) {
    switch (value->getKind()) {
    case SSAValueKind::BlockArgument:
    case SSAValueKind::StmtResult:
    case SSAValueKind::ForStmt:
      return true;
    }
  }

  /// Return the function that this MLValue is defined in.
  MLFunction *getFunction();

  /// Return the function that this MLValue is defined in.
  const MLFunction *getFunction() const {
    return const_cast<MLValue *>(this)->getFunction();
  }

protected:
  MLValue(MLValueKind kind, Type type) : SSAValueImpl(kind, type) {}
};

/// Block arguments are ML Values.
class BlockArgument : public MLValue {
public:
  static bool classof(const SSAValue *value) {
    return value->getKind() == SSAValueKind::BlockArgument;
  }

  /// Return the function that this argument is defined in.
  MLFunction *getFunction();
  const MLFunction *getFunction() const {
    return const_cast<BlockArgument *>(this)->getFunction();
  }

  StmtBlock *getOwner() { return owner; }
  const StmtBlock *getOwner() const { return owner; }

private:
  friend class StmtBlock; // For access to private constructor.
  BlockArgument(Type type, StmtBlock *owner)
      : MLValue(MLValueKind::BlockArgument, type), owner(owner) {}

  /// The owner of this operand.
  /// TODO: can encode this more efficiently to avoid the space hit of this
  /// through bitpacking shenanigans.
  StmtBlock *const owner;
};

/// This is a value defined by a result of an operation instruction.
class StmtResult : public MLValue {
public:
  StmtResult(Type type, OperationStmt *owner)
      : MLValue(MLValueKind::StmtResult, type), owner(owner) {}

  static bool classof(const SSAValue *value) {
    return value->getKind() == SSAValueKind::StmtResult;
  }

  OperationStmt *getOwner() { return owner; }
  const OperationStmt *getOwner() const { return owner; }

  /// Returns the number of this result.
  unsigned getResultNumber() const;

private:
  /// The owner of this operand.
  /// TODO: can encode this more efficiently to avoid the space hit of this
  /// through bitpacking shenanigans.
  OperationStmt *const owner;
};

// TODO(clattner) clean all this up.
using CFGValue = MLValue;
using BBArgument = BlockArgument;
using InstResult = StmtResult;

} // namespace mlir

#endif

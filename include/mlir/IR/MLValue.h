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
class MLValue;
class Statement;
class MLFunction;
class ForStmt;

/// This enum contains all of the SSA value kinds that are valid in an ML
/// function.  This should be kept as a proper subtype of SSAValueKind,
/// including having all of the values of the enumerators align.
enum class MLValueKind {
  FnArgument = (int)SSAValueKind::FnArgument,
  StmtResult = (int)SSAValueKind::StmtResult,
  InductionVar = (int)SSAValueKind::InductionVar,
};

/// The operand of ML function statement contains an MLValue.
using StmtOperand = IROperandImpl<MLValue, Statement>;

/// MLValue is the base class for SSA values in ML functions.
class MLValue : public SSAValueImpl<StmtOperand, MLValueKind> {
public:
  static bool classof(const SSAValue *value) {
    switch (value->getKind()) {
    case SSAValueKind::FnArgument:
    case SSAValueKind::StmtResult:
    case SSAValueKind::InductionVar:
      return true;

    case SSAValueKind::BBArgument:
    case SSAValueKind::InstResult:
      return false;
    }
  }

protected:
  MLValue(MLValueKind kind, Type *type) : SSAValueImpl(kind, type) {}
};

/// This is the value defined by an argument of an ML function.
class FnArgument : public MLValue {
public:
  static bool classof(const SSAValue *value) {
    return value->getKind() == SSAValueKind::FnArgument;
  }

  MLFunction *getOwner() { return owner; }
  const MLFunction *getOwner() const { return owner; }

private:
  friend class MLFunction; // For access to private constructor.
  FnArgument(Type *type, MLFunction *owner)
      : MLValue(MLValueKind::FnArgument, type), owner(owner) {}

  /// The owner of this operand.
  /// TODO: can encode this more efficiently to avoid the space hit of this
  /// through bitpacking shenanigans.
  MLFunction *const owner;
};

/// This is a value defined by a result of an operation instruction.
class StmtResult : public MLValue {
public:
  StmtResult(Type *type, OperationStmt *owner)
      : MLValue(MLValueKind::StmtResult, type), owner(owner) {}

  static bool classof(const SSAValue *value) {
    return value->getKind() == SSAValueKind::StmtResult;
  }

  OperationStmt *getOwner() { return owner; }
  const OperationStmt *getOwner() const { return owner; }

  /// Return the number of this result.
  unsigned getResultNumber() const;

private:
  /// The owner of this operand.
  /// TODO: can encode this more efficiently to avoid the space hit of this
  /// through bitpacking shenanigans.
  OperationStmt *const owner;
};

/// This is a value defined by a loop induction variable.
class InductionVar : public MLValue {
public:
  InductionVar(Type *type, ForStmt *owner)
      : MLValue(MLValueKind::InductionVar, type), owner(owner) {}

  static bool classof(const SSAValue *value) {
    return value->getKind() == SSAValueKind::InductionVar;
  }

  ForStmt *getOwner() { return owner; }
  const ForStmt *getOwner() const { return owner; }

private:
  /// The owner of this operand.
  /// TODO: can encode this more efficiently to avoid the space hit of this
  /// through bitpacking shenanigans.
  ForStmt *const owner;
};

} // namespace mlir

#endif

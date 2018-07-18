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

#ifndef MLIR_IR_CFGOPERAND_H
#define MLIR_IR_CFGOPERAND_H

#include "mlir/IR/SSAOperand.h"
#include "mlir/IR/SSAValue.h"

namespace mlir {
class CFGValue;
class Instruction;

enum class CFGValueKind {
  // TODO: Constants (uses attrs as representation)
  // TODO: BBArg,
  InstResult,
};

/// The operand of a CFG Instruction contains a CFGValue.
using InstOperand = SSAOperand<CFGValue, Instruction>;

/// CFGValue is the base class for CFG value types.
class CFGValue : public SSAValue<InstOperand, CFGValueKind> {
protected:
  CFGValue(CFGValueKind kind, Type *type) : SSAValue(kind, type) {}
};

/// Instruction results are CFG Values.
class InstResult : public CFGValue {
public:
  InstResult(Type *type, Instruction *owner)
      : CFGValue(CFGValueKind::InstResult, type), owner(owner) {}

private:
  /// The owner of this operand.
  /// TODO: can encode this more efficiently to avoid the space hit of this
  /// through bitpacking shenanigans.
  Instruction *const owner;
};

} // namespace mlir

#endif

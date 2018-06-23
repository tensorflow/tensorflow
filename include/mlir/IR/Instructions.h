//===- Instructions.h - MLIR CFG Instruction Classes ------------*- C++ -*-===//
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
// This file defines the classes for CFGFunction instructions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_INSTRUCTIONS_H
#define MLIR_IR_INSTRUCTIONS_H

#include "mlir/Support/LLVM.h"

namespace mlir {
  class BasicBlock;
  class CFGFunction;


/// Terminator instructions are the last part of a basic block, used to
/// represent control flow and returns.
class TerminatorInst {
public:
  enum class Kind {
    Return
  };

  Kind getKind() const { return kind; }

  /// Return the BasicBlock that contains this terminator instruction.
  BasicBlock *getBlock() const {
    return block;
  }
  CFGFunction *getFunction() const;

  void print(raw_ostream &os) const;
  void dump() const;

protected:
  TerminatorInst(Kind kind, BasicBlock *block) : kind(kind), block(block) {}

private:
  Kind kind;
  BasicBlock *block;
};

class ReturnInst : public TerminatorInst {
public:
  explicit ReturnInst(BasicBlock *block);
  // TODO: Flesh this out.


  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const TerminatorInst *inst) {
    return inst->getKind() == Kind::Return;
  }
};

} // end namespace mlir

#endif  // MLIR_IR_INSTRUCTIONS_H

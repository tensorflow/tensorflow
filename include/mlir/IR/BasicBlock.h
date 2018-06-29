//===- BasicBlock.h - MLIR BasicBlock Class ---------------------*- C++ -*-===//
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

#ifndef MLIR_IR_BASICBLOCK_H
#define MLIR_IR_BASICBLOCK_H

#include "mlir/IR/Instructions.h"

namespace mlir {

/// Each basic block in a CFG function contains a list of basic block arguments,
/// normal instructions, and a terminator instruction.
///
/// Basic blocks form a graph (the CFG) which can be traversed through
/// predecessor and successor edges.
class BasicBlock {
public:
  explicit BasicBlock(CFGFunction *function);

  /// Return the function that a BasicBlock is part of.
  CFGFunction *getFunction() const {
    return function;
  }

  // TODO: bb arguments

  // TODO: Wrong representation.
  std::vector<OperationInst*> instList;

  void setTerminator(TerminatorInst *inst) {
    terminator = inst;
  }
  TerminatorInst *getTerminator() const { return terminator; }

  void print(raw_ostream &os) const;
  void dump() const;

private:
  CFGFunction *const function;
  // FIXME: wrong representation and API, leaks memory etc.
  TerminatorInst *terminator = nullptr;
};

} // end namespace mlir

#endif  // MLIR_IR_BASICBLOCK_H

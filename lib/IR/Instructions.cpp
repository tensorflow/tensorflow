//===- Instructions.cpp - MLIR CFGFunction Instruction Classes ------------===//
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

#include "mlir/IR/Instructions.h"
#include "mlir/IR/BasicBlock.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// Instruction
//===----------------------------------------------------------------------===//

CFGFunction *Instruction::getFunction() const {
  return getBlock()->getFunction();
}

//===----------------------------------------------------------------------===//
// OperationInst
//===----------------------------------------------------------------------===//

OperationInst::OperationInst(Identifier name, BasicBlock *block) :
  Instruction(Kind::Operation, block), name(name) {
  getBlock()->instList.push_back(this);
}

//===----------------------------------------------------------------------===//
// Terminators
//===----------------------------------------------------------------------===//

ReturnInst::ReturnInst(BasicBlock *parent)
  : TerminatorInst(Kind::Return, parent) {
  getBlock()->setTerminator(this);
}

BranchInst::BranchInst(BasicBlock *dest, BasicBlock *parent)
  : TerminatorInst(Kind::Branch, parent), dest(dest) {
  getBlock()->setTerminator(this);
}

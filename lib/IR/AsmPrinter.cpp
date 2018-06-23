//===- AsmPrinter.cpp - MLIR Assembly Printer Implementation --------------===//
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
// This file implements the MLIR AsmPrinter class, which is used to implement
// the various print() methods on the core IR objects.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"
using namespace mlir;


//===----------------------------------------------------------------------===//
// Function printing
//===----------------------------------------------------------------------===//

static void printFunctionSignature(const Function *fn, raw_ostream &os) {
  auto type = fn->getType();

  os << "@" << fn->getName() << '(';
  interleave(type->getInputs(),
             [&](Type *eltType) { os << *eltType; },
             [&]() { os << ", "; });
  os << ')';

  switch (type->getResults().size()) {
  case 0: break;
  case 1:
    os << " -> " << *type->getResults()[0];
    break;
  default:
    os << " -> (";
    interleave(type->getResults(),
               [&](Type *eltType) { os << *eltType; },
               [&]() { os << ", "; });
    os << ')';
    break;
  }
}

void ExtFunction::print(raw_ostream &os) const {
  os << "extfunc ";
  printFunctionSignature(this, os);
  os << "\n";
}

//===----------------------------------------------------------------------===//
// CFG Function printing
//===----------------------------------------------------------------------===//

namespace {
class CFGFunctionState {
public:
  CFGFunctionState(const CFGFunction *function, raw_ostream &os);

  const CFGFunction *getFunction() const { return function; }

  void print();
  void print(const BasicBlock *block);
  void print(const TerminatorInst *inst);

  unsigned getBBID(const BasicBlock *block) {
    auto it = basicBlockIDs.find(block);
    assert(it != basicBlockIDs.end() && "Block not in this function?");
    return it->second;
  }

private:
  const CFGFunction *function;
  raw_ostream &os;
  DenseMap<BasicBlock*, unsigned> basicBlockIDs;
};
} // end anonymous namespace

CFGFunctionState::CFGFunctionState(const CFGFunction *function, raw_ostream &os)
  : function(function), os(os) {

  // Each basic block gets a unique ID per function.
  unsigned blockID = 0;
  for (auto *block : function->blockList)
    basicBlockIDs[block] = blockID++;
}

void CFGFunctionState::print() {
  os << "cfgfunc ";
  printFunctionSignature(this->getFunction(), os);
  os << " {\n";

  for (auto *block : function->blockList)
    print(block);
  os << "}\n\n";
}

void CFGFunctionState::print(const BasicBlock *block) {
  os << "bb" << getBBID(block) << ":\n";

  // TODO Print arguments and instructions.

  print(block->getTerminator());
}

void CFGFunctionState::print(const TerminatorInst *inst) {
  switch (inst->getKind()) {
  case TerminatorInst::Kind::Return:
    os << "  return\n";
    break;
  }
}

//===----------------------------------------------------------------------===//
// print and dump methods
//===----------------------------------------------------------------------===//

void TerminatorInst::print(raw_ostream &os) const {
  CFGFunctionState state(getFunction(), os);
  state.print(this);
}

void TerminatorInst::dump() const {
  print(llvm::errs());
}

void BasicBlock::print(raw_ostream &os) const {
  CFGFunctionState state(getFunction(), os);
  state.print();
}

void BasicBlock::dump() const {
  print(llvm::errs());
}

void Function::print(raw_ostream &os) const {
  switch (getKind()) {
  case Kind::ExtFunc: return cast<ExtFunction>(this)->print(os);
  case Kind::CFGFunc: return cast<CFGFunction>(this)->print(os);
  }
}

void Function::dump() const {
  print(llvm::errs());
}

void CFGFunction::print(raw_ostream &os) const {
  CFGFunctionState state(this, os);
  state.print();
}

void Module::print(raw_ostream &os) const {
  for (auto *fn : functionList)
    fn->print(os);
}

void Module::dump() const {
  print(llvm::errs());
}


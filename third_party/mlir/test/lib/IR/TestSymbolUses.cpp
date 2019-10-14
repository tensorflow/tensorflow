//===- TestSymbolUses.cpp - Pass to test symbol uselists ------------------===//
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

#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// This is a symbol test pass that tests the symbol uselist functionality
/// provided by the symbol table.
struct SymbolUsesPass : public ModulePass<SymbolUsesPass> {
  void runOnModule() override {
    auto module = getModule();

    for (FuncOp func : module.getOps<FuncOp>()) {
      // Test computing uses on a non symboltable op.
      Optional<SymbolTable::UseRange> symbolUses =
          SymbolTable::getSymbolUses(func);

      // Test the conservative failure case.
      if (!symbolUses) {
        func.emitRemark() << "function contains an unknown nested operation "
                             "that 'may' define a new symbol table";
        return;
      }
      if (unsigned numUses = llvm::size(*symbolUses))
        func.emitRemark() << "function contains " << numUses
                          << " nested references";

      // Test the functionality of symbolKnownUseEmpty.
      if (SymbolTable::symbolKnownUseEmpty(func.getName(), module)) {
        func.emitRemark() << "function has no uses";
        continue;
      }

      // Test the functionality of getSymbolUses.
      symbolUses = SymbolTable::getSymbolUses(func.getName(), module);
      assert(symbolUses.hasValue() && "expected no unknown operations");
      for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
        symbolUse.getUser()->emitRemark()
            << "found use of function : " << symbolUse.getSymbolRef();
      }
      func.emitRemark() << "function has " << llvm::size(*symbolUses)
                        << " uses";
    }
  }
};
} // end anonymous namespace

static PassRegistration<SymbolUsesPass> pass("test-symbol-uses",
                                             "Test detection of symbol uses");

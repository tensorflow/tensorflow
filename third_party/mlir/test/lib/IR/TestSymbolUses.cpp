//===- TestSymbolUses.cpp - Pass to test symbol uselists ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// This is a symbol test pass that tests the symbol uselist functionality
/// provided by the symbol table along with erasing from the symbol table.
struct SymbolUsesPass : public ModulePass<SymbolUsesPass> {
  void runOnModule() override {
    auto module = getModule();
    std::vector<FuncOp> ops_to_delete;

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
      if (func.symbolKnownUseEmpty(module)) {
        func.emitRemark() << "function has no uses";
        if (func.getBody().empty())
          ops_to_delete.push_back(func);
        continue;
      }

      // Test the functionality of getSymbolUses.
      symbolUses = func.getSymbolUses(module);
      assert(symbolUses.hasValue() && "expected no unknown operations");
      for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
        symbolUse.getUser()->emitRemark()
            << "found use of function : " << symbolUse.getSymbolRef();
      }
      func.emitRemark() << "function has " << llvm::size(*symbolUses)
                        << " uses";
    }

    for (FuncOp func : ops_to_delete) {
      // In order to test the SymbolTable::erase method, also erase completely
      // useless functions.
      SymbolTable table(module);
      auto func_name = func.getName();
      assert(table.lookup(func_name) && "expected no unknown operations");
      table.erase(func);
      assert(!table.lookup(func_name) &&
             "expected erased operation to be unknown now");
      module.emitRemark() << func_name << " function successfully erased";
    }
  }
};

/// This is a symbol test pass that tests the symbol use replacement
/// functionality provided by the symbol table.
struct SymbolReplacementPass : public ModulePass<SymbolReplacementPass> {
  void runOnModule() override {
    auto module = getModule();

    for (FuncOp func : module.getOps<FuncOp>()) {
      StringAttr newName = func.getAttrOfType<StringAttr>("sym.new_name");
      if (!newName)
        continue;
      if (succeeded(func.replaceAllSymbolUses(newName.getValue(), module)))
        func.setName(newName.getValue());
    }
  }
};
} // end anonymous namespace

static PassRegistration<SymbolUsesPass> pass("test-symbol-uses",
                                             "Test detection of symbol uses");

static PassRegistration<SymbolReplacementPass>
    rauwPass("test-symbol-rauw", "Test replacement of symbol uses");

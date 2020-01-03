//===- TestFunctionLike.cpp - Pass to test helpers on FunctionLike --------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// This is a test pass for verifying FuncOp's eraseArgument method.
struct TestFuncEraseArg : public ModulePass<TestFuncEraseArg> {
  void runOnModule() override {
    auto module = getModule();

    for (FuncOp func : module.getOps<FuncOp>()) {
      SmallVector<unsigned, 4> indicesToErase;
      for (auto argIndex : llvm::seq<int>(0, func.getNumArguments())) {
        if (func.getArgAttr(argIndex, "test.erase_this_arg")) {
          // Push back twice to test that duplicate arg indices are handled
          // correctly.
          indicesToErase.push_back(argIndex);
          indicesToErase.push_back(argIndex);
        }
      }
      // Reverse the order to test that unsorted index lists are handled
      // correctly.
      std::reverse(indicesToErase.begin(), indicesToErase.end());
      func.eraseArguments(indicesToErase);
    }
  }
};

/// This is a test pass for verifying FuncOp's setType method.
struct TestFuncSetType : public ModulePass<TestFuncSetType> {
  void runOnModule() override {
    auto module = getModule();
    SymbolTable symbolTable(module);

    for (FuncOp func : module.getOps<FuncOp>()) {
      auto sym = func.getAttrOfType<FlatSymbolRefAttr>("test.set_type_from");
      if (!sym)
        continue;
      func.setType(symbolTable.lookup<FuncOp>(sym.getValue()).getType());
    }
  }
};
} // end anonymous namespace

static PassRegistration<TestFuncEraseArg> pass("test-func-erase-arg",
                                               "Test erasing func args.");

static PassRegistration<TestFuncSetType> pass2("test-func-set-type",
                                               "Test FuncOp::setType.");

//===- TestCallGraph.cpp - Test callgraph construction and iteration ------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for constructing and iterating over a
// callgraph.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct TestCallGraphPass : public ModulePass<TestCallGraphPass> {
  void runOnModule() {
    llvm::errs() << "Testing : " << getModule().getAttr("test.name") << "\n";
    getAnalysis<CallGraph>().print(llvm::errs());
  }
};
} // end anonymous namespace

static PassRegistration<TestCallGraphPass>
    pass("test-print-callgraph",
         "Print the contents of a constructed callgraph.");

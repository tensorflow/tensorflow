//===- TestLiveness.cpp - Test liveness construction and information
//-------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for constructing and resolving liveness
// information.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Liveness.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TestLivenessPass : public FunctionPass<TestLivenessPass> {
  void runOnFunction() override {
    llvm::errs() << "Testing : " << getFunction().getName() << "\n";
    getAnalysis<Liveness>().print(llvm::errs());
  }
};

} // end anonymous namespace

static PassRegistration<TestLivenessPass>
    pass("test-print-liveness",
         "Print the contents of a constructed liveness information.");

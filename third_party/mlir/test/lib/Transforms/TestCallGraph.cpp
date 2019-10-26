//===- TestCallGraph.cpp - Test callgraph construction and iteration ------===//
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

//===- TestLiveness.cpp - Test liveness construction and information
//-------===//
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

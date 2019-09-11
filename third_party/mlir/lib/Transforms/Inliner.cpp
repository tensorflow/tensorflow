//===- Inliner.cpp - Pass to inline function calls ------------------------===//
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

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

// TODO(riverriddle) This pass should currently only be used for basic testing
// of inlining functionality.
namespace {
struct Inliner : public ModulePass<Inliner> {
  void runOnModule() override {
    auto module = getModule();

    // Collect each of the direct function calls within the module.
    SmallVector<CallOp, 16> callOps;
    for (auto &f : module)
      f.walk([&](CallOp callOp) { callOps.push_back(callOp); });

    // Build the inliner interface.
    InlinerInterface interface(&getContext());

    // Try to inline each of the call operations.
    for (auto &call : callOps) {
      if (failed(inlineFunction(
              interface, module.lookupSymbol<FuncOp>(call.getCallee()), call,
              llvm::to_vector<8>(call.getArgOperands()),
              llvm::to_vector<8>(call.getResults()), call.getLoc())))
        continue;

      // If the inlining was successful then erase the call.
      call.erase();
    }
  }
};
} // end anonymous namespace

static PassRegistration<Inliner> pass("inline", "Inline function calls");

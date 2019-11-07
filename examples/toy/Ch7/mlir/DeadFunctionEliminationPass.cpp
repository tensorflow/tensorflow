//===- DeadFunctionEliminationPass.cpp - Eliminate inlined functions ------===//
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
// This file implements a Module level pass performing dead function
// elimination. This is required as a post-processing step after function
// inlining.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace {
/// This is a simple function DCE pass that deletes all non-main functions after
/// inlining.
/// TODO(riverriddle) This is only necessary because MLIR currently does not
/// have generic DCE support for functions.
class DeadFunctionEliminationPass
    : public mlir::ModulePass<DeadFunctionEliminationPass> {
public:
  void runOnModule() override {
    mlir::ModuleOp module = getModule();
    mlir::SymbolTable moduleSymTable(module);

    // Eliminate non-main functions.
    auto mainFn = moduleSymTable.lookup<mlir::FuncOp>("main");
    for (mlir::FuncOp func :
         llvm::make_early_inc_range(module.getOps<mlir::FuncOp>())) {
      if (func != mainFn)
        func.erase();
    }
  }
};
} // end anonymous namespace

/// Create a pass that eliminates inlined functions in toy.
std::unique_ptr<mlir::Pass> mlir::toy::createDeadFunctionEliminationPass() {
  return std::make_unique<DeadFunctionEliminationPass>();
}

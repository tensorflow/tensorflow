/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <queue>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Identifier.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

// The cmd line flag to specify the whitelist of functions. Rest are trimmed
// after this pass is run.
// NOLINTNEXTLINE
static llvm::cl::list<std::string> trim_funcs_whitelist(
    "tfl-trim-funcs-whitelist", llvm::cl::value_desc("list"),
    llvm::cl::desc("comma separated list of whitelisted functions. The first "
                   "function specified will be used as main."),
    llvm::cl::CommaSeparated);

namespace mlir {
namespace TFL {
namespace {

// The pass to trim functions before we legalize to TFL
// dialect using the specified whitelist.
class TrimFunctionsPass
    : public mlir::PassWrapper<TrimFunctionsPass, OperationPass<ModuleOp>> {
 public:
  explicit TrimFunctionsPass() : trim_funcs_whitelist_(trim_funcs_whitelist) {}
  explicit TrimFunctionsPass(llvm::ArrayRef<std::string> trim_funcs_whitelist)
      : trim_funcs_whitelist_(trim_funcs_whitelist) {}

 private:
  void runOnOperation() override;
  bool TrimModule();
  void Verify();

  llvm::ArrayRef<std::string> trim_funcs_whitelist_;
};

void TrimFunctionsPass::runOnOperation() {
  // trim the functions in the module using the trim_funcs_whitelist_
  // by removing functions not in the whitelist.
  if (TrimModule()) {
    // verify the updated module is still valid, if not signal the
    // pass as failed.
    Verify();
  }
}

bool TrimFunctionsPass::TrimModule() {
  // if no trim_funcs_whitelist_ is specified, this pass is a no-op.
  if (trim_funcs_whitelist_.empty()) return false;

  llvm::SmallVector<FuncOp, 4> funcs_to_trim;
  for (auto func : getOperation().getOps<FuncOp>()) {
    if (llvm::is_contained(trim_funcs_whitelist_, func.getName())) {
      // If no main is specified in the whitelist, use the 1st func
      // in trim_funcs_whitelist as the main.
      // TODO(ashwinm): Currently tflite flatbuffer export assumes there is
      // always a main. This is strictly not required for TFlite. We need to
      // remove that restriction once we have support to attribute the main
      // tensorflow function in MLIR TF import using an entry_point attr.
      if (!llvm::is_contained(trim_funcs_whitelist_, "main") &&
          func.getName() == trim_funcs_whitelist_[0]) {
        func.setName("main");
      }
    } else {
      funcs_to_trim.push_back(func);
    }
  }

  // remove all unexported functions from the module.
  for (auto func : funcs_to_trim) {
    func.erase();
  }
  return true;
}

// validate that all reachable functions from the remaining functions are
// also in the whitelist.
void TrimFunctionsPass::Verify() {
  // TODO(ashwinm): Instead, we should make sure that references to all
  // SymbolRefAttrs of all ops are present.
  SymbolTable symbol_table = SymbolTable(getOperation());
  llvm::SetVector<FuncOp> reachable_funcs;
  for (auto func : getOperation().getOps<FuncOp>()) {
    auto walk_result = func.walk([&](CallOp op) -> WalkResult {
      if (!symbol_table.lookup<FuncOp>(op.getCallee()))
        return getOperation().emitError()
               << func.getName() << " is not in the funcs whitelist";
      return WalkResult::advance();
    });
    if (walk_result.wasInterrupted()) return signalPassFailure();
  }
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect TrimFunctions
/// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateTrimFunctionsPass(
    llvm::ArrayRef<std::string> trim_funcs_whitelist) {
  return std::make_unique<TrimFunctionsPass>(trim_funcs_whitelist);
}

static PassRegistration<TrimFunctionsPass> pass(
    "tfl-trim-funcs-tf",
    "Trim functions to restrict them to a specified whitelist prior to "
    "legalization to TensorFlow lite dialect");

}  // namespace TFL
}  // namespace mlir

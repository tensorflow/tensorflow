/* Copyright 2024 The OpenXLA Authors.

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
#include <memory>
#include <queue>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitters/ir/xla_ops.h"

namespace xla {
namespace emitters {

#define GEN_PASS_DEF_ERASEDEADFUNCTIONSPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

namespace {

struct CallInfo {
  xla::PureCallOp call;
  int count;
};

llvm::DenseSet<mlir::func::FuncOp> FindLiveFunctions(mlir::ModuleOp module) {
  std::queue<mlir::func::FuncOp> worklist;
  llvm::DenseSet<mlir::func::FuncOp> live_funcs;
  module.walk([&](mlir::func::FuncOp func) {
    if (!func.isPrivate()) {
      worklist.push(func);
      live_funcs.insert(func);
    }
  });

  mlir::SymbolTableCollection symbol_table;
  while (!worklist.empty()) {
    auto func = worklist.front();
    worklist.pop();
    func.walk([&](mlir::CallOpInterface call) {
      auto callee = mlir::cast<mlir::func::FuncOp>(
          call.resolveCallableInTable(&symbol_table));
      if (live_funcs.insert(callee).second) {
        worklist.push(callee);
      }
    });
  }
  return live_funcs;
}

class EraseDeadFunctionsPass
    : public impl::EraseDeadFunctionsPassBase<EraseDeadFunctionsPass> {
 public:
  void runOnOperation() override {
    // Find live functions and erase dead ones.
    auto live = FindLiveFunctions(getOperation());
    getOperation().walk([&](mlir::func::FuncOp func) {
      if (!live.contains(func)) {
        func.erase();
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateEraseDeadFunctionsPass() {
  return std::make_unique<EraseDeadFunctionsPass>();
}

}  // namespace emitters
}  // namespace xla

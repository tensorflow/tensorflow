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
#include <queue>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_PREINLINERPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

namespace {

struct CallInfo {
  PureCallOp call;
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
      auto callee =
          mlir::cast<mlir::func::FuncOp>(call.resolveCallable(&symbol_table));
      if (live_funcs.insert(callee).second) {
        worklist.push(callee);
      }
    });
  }
  return live_funcs;
}

class PreInlinerPass : public impl::PreInlinerPassBase<PreInlinerPass> {
 public:
  void runOnOperation() override {
    // Find live functions and erase dead ones.
    auto live = FindLiveFunctions(getOperation());
    getOperation().walk([&](mlir::func::FuncOp func) {
      if (!live.contains(func)) {
        func.erase();
      }
    });

    absl::flat_hash_map<std::string, CallInfo> calls;
    getOperation().walk([&](PureCallOp call) {
      auto& info = calls[call.getCallee().str()];
      info.call = call;
      info.count++;
    });

    for (const auto& [_, info] : calls) {
      if (info.count == 1) {
        info.call->setAttr("xla_gpu.always_inline",
                           mlir::UnitAttr::get(&getContext()));
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreatePreInlinerPass() {
  return std::make_unique<PreInlinerPass>();
}

}  // namespace gpu
}  // namespace xla

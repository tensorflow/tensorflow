/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/transforms/func_to_graph/pass.h"

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/transforms/func_to_graph/func_to_graph.h"

namespace mlir {
namespace tfg {
namespace {

#define GEN_PASS_DEF_FUNCTOGRAPH
#include "tensorflow/core/transforms/passes.h.inc"

class FuncToGraphPass : public impl::FuncToGraphBase<FuncToGraphPass> {
 public:
  FuncToGraphPass() = default;

  // This will try to lower the function that has attribute
  // `tfg.lifted_graph_version` to a graph. It replaces all the uses of
  // arguments with related op results. The relation between args and ops is
  // identified by the tfg.name attr. The arg's tfg.name attr will be prefixed
  // with the related op's tfg.name. Besides, The ReturnOp will be dropped
  // directly.
  void runOnOperation() override;
};
}  // namespace

void FuncToGraphPass::runOnOperation() {
  ModuleOp module = getOperation();

  auto *dialect = getContext().getLoadedDialect<TFGraphDialect>();
  StringAttr lifted_graph_func_name =
      dialect->getLiftedGraphFuncNameAttrIdentifier();

  GraphFuncOp lifted_graph_func;
  for (auto func : module.getOps<GraphFuncOp>()) {
    if (func.getSymName() == lifted_graph_func_name) {
      lifted_graph_func = func;
      break;
    }
  }

  if (!lifted_graph_func) return;

  auto status = FuncToGraph(lifted_graph_func);
  if (!status.ok()) {
    emitError(lifted_graph_func.getLoc())
        << "FuncToGraph failed: " << status.message();
    signalPassFailure();
  }
}

std::unique_ptr<Pass> CreateFuncToGraphPass() {
  return std::make_unique<FuncToGraphPass>();
}

}  // namespace tfg
}  // namespace mlir

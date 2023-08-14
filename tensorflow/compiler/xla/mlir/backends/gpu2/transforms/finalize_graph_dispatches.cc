/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/xla_gpu_api.h"
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_dialect.h"  // IWYU pragma: keep
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_ops.h"
#include "tensorflow/compiler/xla/mlir/backends/gpu2/transforms/passes.h"

#define GEN_PASS_DECL_FINALIZEGRAPHDISPATCHES
#include "tensorflow/compiler/xla/mlir/backends/gpu2/transforms/passes.h.inc"

#define GEN_PASS_DEF_FINALIZEGRAPHDISPATCHES
#include "tensorflow/compiler/xla/mlir/backends/gpu2/transforms/passes.h.inc"

namespace xla::gpu {
namespace {

using namespace mlir;  // NOLINT

//===----------------------------------------------------------------------===//
// Outline xla_gpu.graph.dispatch region into a graph construction function
//===----------------------------------------------------------------------===//

struct OutlinedGraphDispatch {
  SmallVector<Value> args;
  func::FuncOp func;
};

// Outlines a graph dispatch operation into a function that returns a graph
// constructed using XLA:GPU runtime APIs.
static FailureOr<OutlinedGraphDispatch> outlineGraphDispatch(
    SymbolTable& sym_table, XlaGpuApi& api, GraphDispatchOp op) {
  ImplicitLocOpBuilder b(op.getLoc(), sym_table.getOp());
  auto module = cast<ModuleOp>(sym_table.getOp());

  // Collect all the values defined above the dispatch region.
  SetVector<Value> args_set;
  getUsedValuesDefinedAbove(op.getBody(), args_set);

  // Check that we captured execution context argument, and make sure it comes
  // first in the argument list.
  SmallVector<Value> args = args_set.takeVector();
  llvm::partition(args, [](Value value) {
    return isa<ExecutionContextType>(value.getType());
  });

  if (args.empty() || !args[0].getType().isa<ExecutionContextType>())
    return op.emitError(
        "graph dispatch regions doesn't capture execution context");

  // Create a function that creates a new graph instance and add node to it by
  // executing graph dispatch region.
  auto func = b.create<func::FuncOp>(
      "__xla_gpu.graph.create",
      b.getFunctionType(TypeRange(args), b.getType<GraphType>()));
  func.setPrivate();
  sym_table.insert(func);

  Block* body = func.addEntryBlock();
  b.setInsertionPointToStart(body);

  // `!xla_gpu.execution_context` argument
  auto ctx = cast<TypedValue<ExecutionContextType>>(func.getArgument(0));

  func::FuncOp create_graph = api.getCreateGraph(b, module);
  Value graph = b.create<func::CallOp>(create_graph.getSymName(),
                                       b.getType<GraphType>(), ctx)
                    .getResult(0);

  // Move operations from the graph dispatch region into the function body.
  body->getOperations().splice(body->end(),
                               op.getBody().front().getOperations());

  // Remap implicit graph operand of the graph dispatch operation.
  op.getGraph().replaceAllUsesWith(graph);

  // Remap all captured values to block arguments inside the function body.
  for (auto tuple : llvm::zip(args, func.getArguments())) {
    std::get<0>(tuple).replaceUsesWithIf(
        std::get<1>(tuple), [&](OpOperand& operand) {
          return operand.getOwner()->getBlock() == body;
        });
  }

  b.create<func::ReturnOp>(TypeRange(), graph);

  return OutlinedGraphDispatch{std::move(args), func};
}

//===----------------------------------------------------------------------===//

TypedValue<ExecutionContextType> getExecutionContext(Operation* op) {
  auto func = op->getParentOfType<func::FuncOp>();
  return func.getArguments().front().cast<TypedValue<ExecutionContextType>>();
}

class FinalizeGraphDispatchesPass
    : public ::impl::FinalizeGraphDispatchesBase<FinalizeGraphDispatchesPass> {
 public:
  void runOnOperation() override {
    XlaGpuApi api;

    SmallVector<GraphDispatchOp> dispatches;
    getOperation().walk([&](GraphDispatchOp op) { dispatches.push_back(op); });

    for (GraphDispatchOp op : dispatches) {
      FailureOr<OutlinedGraphDispatch> outlined =
          outlineGraphDispatch(api.symTable(getOperation()), api, op);
      if (failed(outlined)) return signalPassFailure();

      ImplicitLocOpBuilder b(op.getLoc(), op);

      // Call graph builder function to construct a new instance of a graph.
      Value graph =
          b.create<func::CallOp>(outlined->func.getSymName(),
                                 b.getType<GraphType>(), outlined->args)
              .getResult(0);

      // Execute constructed graph using XLA:GPU API.
      func::FuncOp execute_graph = api.getExecuteGraph(b, getOperation());
      SmallVector<Value> args = {getExecutionContext(op), graph};
      b.create<func::CallOp>(execute_graph.getSymName(), TypeRange(), args);

      // Erase the original graph dispatch operation.
      op->erase();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createFinalizeGraphDispatchesPass() {
  return std::make_unique<FinalizeGraphDispatchesPass>();
}

}  // namespace xla::gpu

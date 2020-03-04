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

// This transformation pass transforms from TF executor dialect to MLIR TF
// control dialect.

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "mlir/Support/LLVM.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/control_flow_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

#define DEBUG_TYPE "tf-executor-to-ctl"

namespace mlir {

namespace {
struct ExecutorToControlDialectConversion
    : public FunctionPass<ExecutorToControlDialectConversion> {
  void runOnFunction() override;
};
}  // end anonymous namespace

static bool HasSingleGraph(FuncOp function) {
  // We expect the function has only one region with one block,
  if (function.getBlocks().size() != 1) return false;
  auto &block = function.front();
  // and the block contains two ops,
  if (std::next(block.begin()) == block.end()) return false;
  // one GraphOp,
  if (!isa<tf_executor::GraphOp>(block.begin())) return false;
  // followed by a terminator.
  if (!std::next(block.begin())->isKnownTerminator()) return false;
  return true;
}

void ExecutorToControlDialectConversion::runOnFunction() {
  if (!HasSingleGraph(getFunction())) {
    LLVM_DEBUG(llvm::dbgs()
               << "Expect a Function with a single block and a single graph op,"
                  " skip tf_executor dialect conversion\n");
    return;
  }
  Type control_type = TFControlFlow::TFControlType::get(&getContext());

  Block &body = getFunction().front();
  auto graph = cast<tf_executor::GraphOp>(body.front());
  OpBuilder builder(&body);
  SmallString<64> new_op_name;
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(graph.GetBody()))) {
    LLVM_DEBUG(llvm::dbgs() << "Process: " << op.getName() << "\n");

    if (auto fetch = dyn_cast<tf_executor::FetchOp>(op)) {
      // Replace all the operands of the fetch op with the uses of the graph
      // results, remove the fetch op afterwards.
      for (auto ops_and_ret_vals :
           llvm::zip(graph.getResults(), fetch.getOperands()))
        std::get<0>(ops_and_ret_vals)
            .replaceAllUsesWith(std::get<1>(ops_and_ret_vals));
      op.erase();
      continue;
    }

    builder.setInsertionPoint(&op);

    if (auto island = dyn_cast<tf_executor::IslandOp>(op)) {
      Value ctl_sequence = nullptr;
      if (island.GetBody().without_terminator().empty() &&
          island.getNumOperands() > 1) {
        // For an empty island with multiple control inputs, we create a no-op
        // inside it which will group all the inputs into one control output.
        // This helps reducing the number of edges when there are multiple
        // islands depending on this one.
        builder.setInsertionPointToStart(&island.GetBody());
        builder.create<TF::NoOp>(op.getLoc(), ArrayRef<Type>{},
                                 ArrayRef<Value>{}, ArrayRef<NamedAttribute>{});
        builder.setInsertionPoint(&op);
      }
      for (Operation &wrapped_op : island.GetBody()) {
        LLVM_DEBUG(llvm::dbgs()
                   << " In island: " << wrapped_op.getName() << "\n");
        if (isa<tf_executor::YieldOp>(wrapped_op)) {
          for (auto ops_and_ret_vals :
               llvm::zip(island.getResults(), wrapped_op.getOperands()))
            std::get<0>(ops_and_ret_vals)
                .replaceAllUsesWith(std::get<1>(ops_and_ret_vals));
          break;
        }
        // Add a leading _ off the name.
        new_op_name = "_";
        new_op_name += wrapped_op.getName().getStringRef();
        OperationState state(wrapped_op.getLoc(), new_op_name);

        // Add an operand for each non-control input we find. Collect control
        // values separately to add them to the island operands
        state.operands.append(wrapped_op.getOperands().begin(),
                              wrapped_op.getOperands().end());

        // Chain operations through a control dependency, except for the first
        // operations in the sequence that carry the control dependencies held
        // by the island itself.
        if (ctl_sequence) {
          state.operands.push_back(ctl_sequence);
        } else {
          for (Value ctl_operand : island.getOperands())
            state.operands.push_back(ctl_operand);
        }

        // Add a result type for each result
        state.types.append(wrapped_op.getResultTypes().begin(),
                           wrapped_op.getResultTypes().end());
        state.types.push_back(control_type);

        // Create the replacement operation.
        auto *replacement = builder.createOperation(state);
        replacement->setAttrs(wrapped_op.getAttrList());

        for (auto ops_and_ret_vals :
             llvm::zip(wrapped_op.getResults(), replacement->getResults()))
          std::get<0>(ops_and_ret_vals)
              .replaceAllUsesWith(std::get<1>(ops_and_ret_vals));

        ctl_sequence = replacement->getResult(replacement->getNumResults() - 1);
      }

      if (ctl_sequence) {
        // If ctl_sequence is non-null, this means at least one operation has
        // been rewritten from ops in island. Last op rewritten must logically
        // carry // all the island control inputs, we can simply use it to
        // replace all uses of island's control output.
        island.control().replaceAllUsesWith(ctl_sequence);
      } else if (island.getNumOperands() > 0) {
        // Getting here means island had an effectively empty body and there is
        // just one control input. In this case, island's control output should
        // be replaced with the control input.
        assert(island.getNumOperands() == 1);
        island.control().replaceAllUsesWith(island.getOperand(0));
      }

      op.erase();
      continue;
    }

    new_op_name.clear();
    if (isa<tf_executor::SwitchOp>(op)) {
      new_op_name = "_tf.Switch";
    } else if (isa<tf_executor::SwitchNOp>(op)) {
      new_op_name = "_tf._SwitchN";
    } else if (isa<tf_executor::MergeOp>(op)) {
      new_op_name = "_tf.Merge";
    } else if (isa<tf_executor::NextIterationSourceOp>(op)) {
      new_op_name = "_tf.NextIteration.source";
    } else if (isa<tf_executor::NextIterationSinkOp>(op)) {
      new_op_name = "_tf.NextIteration.sink";
    } else if (isa<tf_executor::LoopCondOp>(op)) {
      new_op_name = "_tf.LoopCond";
    } else if (isa<tf_executor::EnterOp>(op)) {
      new_op_name = "_tf.Enter";
    } else if (isa<tf_executor::ExitOp>(op)) {
      new_op_name = "_tf.Exit";
    } else if (isa<tf_executor::ControlTriggerOp>(op)) {
      new_op_name = "_tf.ControlTrigger";
    } else {
      op.emitOpError() << "unhandled op in tf_executor to _tf conversion";
      return signalPassFailure();
    }
    OperationState state(op.getLoc(), new_op_name);
    // Drop all TokenType operands since they don't exist in the control
    // dialect.
    auto non_null_operands = llvm::make_filter_range(
        op.getOperands(),
        [](Value v) { return !v.getType().isa<tf_executor::TokenType>(); });
    state.operands.append(non_null_operands.begin(), non_null_operands.end());
    for (Type result_type : op.getResultTypes()) {
      // Filter out TokenType, they don't exist in the control dialect.
      if (result_type.isa<tf_executor::TokenType>()) continue;
      if (!result_type.isa<tf_executor::ControlType>())
        state.types.push_back(result_type);
      else
        state.types.push_back(control_type);
    }
    // The control dialect has a control result for the sink operation.
    if (isa<tf_executor::NextIterationSinkOp>(op))
      state.types.push_back(control_type);

    // Create the replacement operation.
    auto *replacement = builder.createOperation(state);
    replacement->setAttrs(op.getAttrList());

    if (auto next_iteration =
            dyn_cast<tf_executor::NextIterationSourceOp>(op)) {
      next_iteration.output().replaceAllUsesWith(replacement->getResult(0));
      next_iteration.token().dropAllUses();
      next_iteration.control().replaceAllUsesWith(replacement->getResult(1));
    } else {
      for (auto ops_and_ret_vals :
           llvm::zip(op.getResults(), replacement->getResults()))
        std::get<0>(ops_and_ret_vals)
            .replaceAllUsesWith(std::get<1>(ops_and_ret_vals));
    }
    op.erase();
  }

  // Now we have rewritten all ops inside GraphOp to TF Control dialect. We need
  // to move all operations outside of GraphOp and remove it.
  body.getOperations().splice(body.begin(), graph.GetBody().getOperations());
  graph.erase();
}

std::unique_ptr<OpPassBase<FuncOp>>
CreateTFExecutorToControlDialectConversion() {
  return std::make_unique<ExecutorToControlDialectConversion>();
}

}  // namespace mlir

static mlir::PassRegistration<mlir::ExecutorToControlDialectConversion> pass(
    "tf-executor-to-control-conversion",
    "Convert from TF executor dialect to TF control dialect");

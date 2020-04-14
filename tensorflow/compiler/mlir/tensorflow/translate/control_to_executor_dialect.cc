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

// This transformation pass transforms MLIR TF control dialect into a
// combination of the TF and TF executor dialects.
//
// !! This code is only intended for migration purpose and will be deleted when
// !! the importer is updated to directly emit the tf_executor dialect.

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/control_flow_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

#define DEBUG_TYPE "tf-ctl-to-executor"

namespace mlir {

namespace {
// This pass checks if a function contains only operations in the TF control
// dialect and converts it to a mix of the tf_executor and tf dialects.
// Operations that exist in the tf_executor dialects are used directly,
// otherwise _tf operations are wrapped in an island and the _ prefix is
// removed. Control dependencies are moved to be handled by the island itself.
struct ControlToExecutorDialectConversion
    : public PassWrapper<ControlToExecutorDialectConversion, FunctionPass> {
  void runOnFunction() override;

 private:
  tf_executor::IslandOp CreateIslandForOp(Operation *op, OpBuilder *builder);
};
}  // end anonymous namespace

static bool IsUnderscoredTFOp(Operation *op) {
  return op->getName().getStringRef().startswith("_tf.");
}

static bool HasOnlyTFControlOperations(FuncOp function) {
  return llvm::all_of(function, [](Block &block) {
    return llvm::all_of(block, [](Operation &op) {
      return IsUnderscoredTFOp(&op) || isa<ReturnOp>(op);
    });
  });
}

tf_executor::IslandOp ControlToExecutorDialectConversion::CreateIslandForOp(
    Operation *op, OpBuilder *builder) {
  // Create a new region for the tf_executor.island body
  SmallVector<Value, 8> operands;
  for (Value operand : op->getOperands())
    if (operand.getType().isa<tf_executor::ControlType>())
      operands.push_back(operand);
  SmallVector<Type, 8> types;
  for (Type result_type : op->getResultTypes())
    if (!result_type.isa<TFControlFlow::TFControlType>())
      types.push_back(result_type);
  types.push_back(tf_executor::ControlType::get(&getContext()));

  auto island = builder->create<tf_executor::IslandOp>(
      op->getLoc(), types, operands, ArrayRef<NamedAttribute>{});
  island.body().push_back(new Block);

  return island;
}

void ControlToExecutorDialectConversion::runOnFunction() {
  if (!HasOnlyTFControlOperations(getFunction())) {
    LLVM_DEBUG(llvm::dbgs() << "Function has unsupported operation, skip "
                               "tf_executor dialect conversion\n");
    return;
  }
  if (getFunction().getBlocks().size() != 1) {
    LLVM_DEBUG(llvm::dbgs() << "Expect single block function, , skip "
                               "tf_executor dialect conversion\n");
    return;
  }

  Block &body = getFunction().getBody().front();
  OpBuilder builder(&body, body.begin());

  // Create a new tf_executor.graph at the beginning of the function.
  auto graph_op = builder.create<tf_executor::GraphOp>(
      getFunction().getLoc(), getFunction().getType().getResults());
  graph_op.body().push_back(new Block);
  builder.setInsertionPointToEnd(&graph_op.GetBody());
  llvm::StringMap<tf_executor::NextIterationSourceOp> frame_name_to_loop;

  // Loop over operations in the function and move them into the graph region.
  for (Operation &op : llvm::make_early_inc_range(body)) {
    // Skip the just-created tf_executor.graph.
    if (isa<tf_executor::GraphOp>(op)) continue;

    // This is the new operation that will replace the current one in the graph.
    Operation *replacement = nullptr;
    if (op.isKnownTerminator()) {
      // This is the return of the function, we will create a fetch in the graph
      // matching the operands of the returns. The return is then updated to
      // take as operands the results of the tf_executor.graph operation.
      SmallVector<Value, 8> ret_vals;
      for (Value operand : op.getOperands()) ret_vals.push_back(operand);
      for (auto &graph_result : llvm::enumerate(graph_op.getResults()))
        op.setOperand(graph_result.index(), graph_result.value());
      builder.create<tf_executor::FetchOp>(getFunction().getLoc(), ret_vals);
      continue;
    }
    assert(IsUnderscoredTFOp(&op) && "Expected only _tf operations");

    // The operands and types arrays are used to create the tf_executor ops.
    SmallVector<Value, 8> operands;
    operands.append(op.getOperands().begin(), op.getOperands().end());
    SmallVector<Type, 8> types;
    for (Type result_type : op.getResultTypes()) {
      if (result_type.isa<TFControlFlow::TFControlType>())
        types.push_back(tf_executor::ControlType::get(&getContext()));
      else
        types.push_back(result_type);
    }
    auto loc = op.getLoc();

    // Match the specific operation that has a tf_executor equivalent, the
    // others will be wrapped in an island.

    // FIXME: StringSwitch

    if (op.getName().getStringRef() == "_tf.Switch") {
      replacement = builder.create<tf_executor::SwitchOp>(
          loc, types, operands, ArrayRef<NamedAttribute>{});
    } else if (op.getName().getStringRef() == "_tf._SwitchN") {
      replacement = builder.create<tf_executor::SwitchNOp>(
          loc, types, operands, ArrayRef<NamedAttribute>{});
    } else if (op.getName().getStringRef() == "_tf.Merge") {
      replacement = builder.create<tf_executor::MergeOp>(
          loc, types, operands, ArrayRef<NamedAttribute>{});
    } else if (op.getName().getStringRef() == "_tf.NextIteration.source") {
      replacement = builder.create<tf_executor::NextIterationSourceOp>(
          loc, op.getResult(0).getType());
      // Record a mapping of the name to the nextiteration.source so that when
      // we convert the sink we can get the token.
      StringAttr frame = op.getAttrOfType<StringAttr>("name");
      assert(!frame.getValue().empty());
      frame_name_to_loop[frame.getValue()] =
          cast<tf_executor::NextIterationSourceOp>(replacement);
      // Replace the results here since the _tf source does not produce a token
      // there isn't a mapping for the new result #1.
      op.getResult(0).replaceAllUsesWith(replacement->getResult(0));
      for (int i : llvm::seq<int>(1, op.getNumResults()))
        op.getResult(i).replaceAllUsesWith(replacement->getResult(i + 1));
      replacement->setAttrs(op.getAttrList());
      op.erase();
      continue;
    } else if (op.getName().getStringRef() == "_tf.NextIteration.sink") {
      StringAttr frame = op.getAttrOfType<StringAttr>("name");
      assert(!frame.getValue().empty());
      tf_executor::NextIterationSourceOp srcOp =
          frame_name_to_loop[frame.getValue()];
      replacement = builder.create<tf_executor::NextIterationSinkOp>(
          loc, srcOp.token(), operands, ArrayRef<NamedAttribute>{});
      replacement->setAttrs(op.getAttrList());
      op.erase();
      continue;
    } else if (op.getName().getStringRef() == "_tf.LoopCond") {
      replacement = builder.create<tf_executor::LoopCondOp>(
          loc, types, operands, ArrayRef<NamedAttribute>{});
    } else if (op.getName().getStringRef() == "_tf.Enter") {
      replacement = builder.create<tf_executor::EnterOp>(
          loc, types, operands, ArrayRef<NamedAttribute>{});
    } else if (op.getName().getStringRef() == "_tf.Exit") {
      replacement = builder.create<tf_executor::ExitOp>(
          loc, types, operands, ArrayRef<NamedAttribute>{});
    } else if (op.getName().getStringRef() == "_tf.ControlTrigger") {
      replacement =
          builder.create<tf_executor::ControlTriggerOp>(loc, operands);
    } else {
      tf_executor::IslandOp island = CreateIslandForOp(&op, &builder);
      replacement = island.getOperation();

      // General case, drop the leading _ off the name and wrap in an island.
      OperationState result(loc, op.getName().getStringRef().drop_front());

      // Only the non-control operands are carried over, the island is handling
      // the control input.
      for (Value operand : op.getOperands())
        if (!operand.getType().isa<tf_executor::ControlType>())
          result.operands.push_back(operand);

      // Add a result type for each non-control result we find
      bool sawControlResult = false;
      for (Type result_type : op.getResultTypes()) {
        if (result_type.isa<TFControlFlow::TFControlType>()) {
          sawControlResult = true;
          continue;
        }
        // We assume all control inputs are at the end of the result list.
        assert(!sawControlResult && "all control results must be last");
        result.types.push_back(result_type);
      }

      // Create the operation inside the island
      OpBuilder island_builder = OpBuilder::atBlockEnd(&island.GetBody());
      Operation *inner_op = island_builder.createOperation(result);
      inner_op->setAttrs(op.getAttrList());

      // Add the terminator for the island
      SmallVector<Value, 8> ret_vals(inner_op->getResults());
      island_builder.create<tf_executor::YieldOp>(loc, ret_vals);
    }

    // Copy the attributes from the original operation to the replacement and
    // remap the results.
    if (!isa<tf_executor::IslandOp>(replacement))
      replacement->setAttrs(op.getAttrList());
    for (int i : llvm::seq<int>(0, op.getNumResults()))
      op.getResult(i).replaceAllUsesWith(replacement->getResult(i));
    op.erase();
  }
}

OperationPass<FuncOp> *CreateTFControlToExecutorDialectConversion() {
  return new ControlToExecutorDialectConversion();
}

}  // namespace mlir

static mlir::PassRegistration<mlir::ControlToExecutorDialectConversion> pass(
    "tf-control-to-executor-conversion",
    "Transform from TF control dialect to TF executor dialect.");

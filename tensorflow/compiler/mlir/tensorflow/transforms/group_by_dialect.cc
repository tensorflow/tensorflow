/* Copyright 2022 Google Inc. All Rights Reserved.

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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TF {
namespace {

void wrapOpsInFunction(std::vector<Operation*>& ops, int function_id);

class GroupByDialectPass : public GroupByDialectPassBase<GroupByDialectPass> {
 public:
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    int function_id = 0;

    for (Block& block : func.getBody().getBlocks()) {
      StringRef current_dialect("<none>");
      std::vector<Operation*> ops;
      for (Operation& op : block.getOperations()) {
        StringRef dialect = op.getName().getDialectNamespace();
        if (dialect != current_dialect) {
          if (!top_level_dialects_.contains(current_dialect)) {
            wrapOpsInFunction(ops, function_id++);
          }
          ops.clear();
          current_dialect = dialect;
        }
        ops.push_back(&op);
      }
      if (!top_level_dialects_.contains(current_dialect)) {
        wrapOpsInFunction(ops, function_id++);
      }
    }
  }

  llvm::SmallDenseSet<StringRef> top_level_dialects_ = {"glue", "func"};
};

// Compute the set of all values which are inputs to `ops`, but not generated
// by an operation in `ops`, and all outputs which are used outside of `ops.
void computeInputsOutputs(std::vector<Operation*>& ops,
                          std::vector<Value>* inputs,
                          std::vector<Value>* outputs) {
  // All operations.
  llvm::DenseSet<Operation*> all_operations(ops.begin(), ops.end());

  // All results of all ops.
  llvm::DenseSet<Value> all_internal_results;
  for (Operation* op : ops) {
    for (Value result : op->getResults()) {
      all_internal_results.insert(result);
    }
  }

  // All operand values in our set not produced as result by some op in our set.
  llvm::DenseSet<Value> inputs_seen;
  for (Operation* op : ops) {
    for (Value operand : op->getOperands()) {
      if (!all_internal_results.contains(operand)) {
        if (!inputs_seen.contains(operand)) {
          inputs->push_back(operand);
          inputs_seen.insert(operand);
        }
      }
    }
  }

  // All results in our set that have a user outside our set.
  llvm::DenseSet<Value> outputs_seen;
  for (Operation* op : ops) {
    for (Value result : op->getResults()) {
      for (auto& use : result.getUses()) {
        if (!all_operations.contains(use.getOwner())) {
          if (!outputs_seen.contains(result)) {
            outputs->push_back(result);
            outputs_seen.insert(result);
          }
          break;
        }
      }
    }
  }
}

void wrapOpsInFunction(std::vector<Operation*>& ops, int function_id) {
  if (ops.empty()) {
    return;
  }

  std::vector<Value> inputs;
  std::vector<Value> outputs;
  computeInputsOutputs(ops, &inputs, &outputs);

  std::vector<Type> input_types;
  std::vector<Type> output_types;

  input_types.reserve(inputs.size());
  for (Value v : inputs) {
    input_types.push_back(v.getType());
  }
  output_types.reserve(outputs.size());
  for (Value v : outputs) {
    output_types.push_back(v.getType());
  }

  // Create the function.
  MLIRContext* context = ops[0]->getContext();
  StringRef dialect = ops[0]->getName().getDialectNamespace();
  OpBuilder builder(context);
  builder.setInsertionPointToEnd(ops[0]->getParentOp()->getBlock());
  auto func = builder.create<mlir::func::FuncOp>(
      ops[0]->getLoc(), dialect.str() + std::to_string(function_id),
      builder.getFunctionType(input_types, output_types));
  func->setAttr("dialect", builder.getStringAttr(dialect));
  auto block = func.addEntryBlock();

  llvm::DenseSet<Operation*> all_operations(ops.begin(), ops.end());
  for (BlockArgument& arg : block->getArguments()) {
    inputs[arg.getArgNumber()].replaceUsesWithIf(arg, [=](OpOperand& o) {
      // Within the operations we're moving, we need to replace uses of
      // values generated elsewhere.
      return all_operations.contains(o.getOwner());
    });
  }

  // Insert function call.
  builder.setInsertionPoint(ops[0]);
  auto call = builder.create<mlir::func::CallOp>(
      ops[0]->getLoc(), func.getFunctionType().getResults(), func.getSymName(),
      inputs);
  for (auto& v : llvm::enumerate(outputs)) {
    v.value().replaceUsesWithIf(call.getResult(v.index()), [=](OpOperand& o) {
      // Outside of what we're moving, results of our operations need to
      // be replaced by results from the function call.
      return !all_operations.contains(o.getOwner());
    });
  }

  // Move ops inside function & add return.
  builder.setInsertionPointToEnd(block);
  for (Operation* op : ops) {
    op->remove();
    builder.insert(op);
  }
  builder.create<mlir::func::ReturnOp>(ops[0]->getLoc(), outputs);
}

}  // namespace

std::unique_ptr<Pass> CreateGroupByDialectPass() {
  return std::make_unique<GroupByDialectPass>();
}

void RegisterGroupByDialectPass() { registerPass(CreateGroupByDialectPass); }

}  // namespace TF
}  // namespace mlir

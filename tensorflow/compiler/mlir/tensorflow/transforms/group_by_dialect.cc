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

#include <memory>
#include <string>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {
namespace {

void wrapOpsInFunction(std::vector<Operation*>& ops, int function_id,
                       Operation* module);

#define GEN_PASS_DEF_GROUPBYDIALECTPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

class GroupByDialectPass
    : public impl::GroupByDialectPassBase<GroupByDialectPass> {
 public:
  void runOnOperation() override;

 private:
  void processFunction(mlir::func::FuncOp func, int& counter,
                       llvm::SmallDenseSet<StringRef>& dialects,
                       Operation* module);
  void processRegion(mlir::Region& region, int& counter,
                     llvm::SmallDenseSet<StringRef>& dialects,
                     Operation* module);

  llvm::SmallDenseSet<StringRef> top_level_dialects_ = {"ml_program", "glue",
                                                        "func"};
};

// Compute the set of all values which are inputs to `ops`, but not generated
// by an operation in `ops`, and all outputs which are used outside of `ops.
void computeInputsOutputs(std::vector<Operation*>& ops,
                          std::vector<Value>* inputs,
                          std::vector<Value>* outputs) {
  // All operations.
  llvm::DenseSet<Operation*> all_operations;

  // All results of all ops.
  llvm::DenseSet<Value> all_internal_results;
  for (Operation* outer : ops) {
    outer->walk([&](Operation* op) {
      all_operations.insert(op);
      for (Value result : op->getResults()) {
        all_internal_results.insert(result);
      }
      // We treat block arguments of inner blocks as "results", too, in
      // the sense that they're values produced inside this op.
      for (Region& region : op->getRegions()) {
        for (Block& block : region.getBlocks()) {
          for (BlockArgument& arg : block.getArguments()) {
            all_internal_results.insert(arg);
          }
        }
      }
    });
  }

  // All operand values in our set not produced as result by some op in our set.
  llvm::DenseSet<Value> inputs_seen;
  for (Operation* outer : ops) {
    outer->walk([&](Operation* op) {
      for (Value operand : op->getOperands()) {
        if (!all_internal_results.contains(operand)) {
          if (!inputs_seen.contains(operand)) {
            inputs->push_back(operand);
            inputs_seen.insert(operand);
          }
        }
      }
    });
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

void wrapOpsInFunction(std::vector<Operation*>& ops, int function_id,
                       Operation* module) {
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
  // Every ModuleOp has at least one region and one block.
  Block* first_block = &module->getRegion(0).front();
  builder.setInsertionPointToEnd(first_block);
  auto func = builder.create<mlir::func::FuncOp>(
      ops[0]->getLoc(), dialect.str() + std::to_string(function_id),
      builder.getFunctionType(input_types, output_types));
  func->setAttr("dialect", builder.getStringAttr(dialect));
  auto block = func.addEntryBlock();

  llvm::DenseSet<Operation*> all_operations;
  for (Operation* outer : ops) {
    outer->walk([&](Operation* op) { all_operations.insert(op); });
  }

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
  for (const auto& v : llvm::enumerate(outputs)) {
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

void GroupByDialectPass::processFunction(
    mlir::func::FuncOp func, int& counter,
    llvm::SmallDenseSet<StringRef>& dialects, Operation* module) {
  // don't re-process functions we generated
  if (func->getAttr("dialect")) return;
  processRegion(func.getBody(), counter, dialects, module);
}

void GroupByDialectPass::processRegion(mlir::Region& region, int& counter,
                                       llvm::SmallDenseSet<StringRef>& dialects,
                                       Operation* module) {
  for (Block& block : region.getBlocks()) {
    StringRef current_dialect("<none>");
    std::vector<Operation*> ops;
    for (Operation& op : block.getOperations()) {
      StringRef dialect = op.getName().getDialectNamespace();
      for (Region& region : op.getRegions()) {
        // When processing nested operations, move all ops (except for func.*)
        // that aren't of the parent dialect into a function or their own.
        llvm::SmallDenseSet<StringRef> parent_dialect = {dialect, "func"};
        processRegion(region, counter, parent_dialect, module);
      }
      if (dialect != current_dialect) {
        if (!dialects.contains(current_dialect)) {
          wrapOpsInFunction(ops, counter++, module);
        }
        ops.clear();
        current_dialect = dialect;
      }
      ops.push_back(&op);
    }
    if (!dialects.contains(current_dialect)) {
      wrapOpsInFunction(ops, counter++, module);
    }
  }
}

void GroupByDialectPass::runOnOperation() {
  int counter = 0;
  Operation* module = getOperation();
  module->walk([&](func::FuncOp func) {
    processFunction(func, counter, top_level_dialects_, module);
  });
}

std::unique_ptr<Pass> CreateGroupByDialectPass() {
  return std::make_unique<GroupByDialectPass>();
}

void RegisterGroupByDialectPass() { registerPass(CreateGroupByDialectPass); }

}  // namespace TF
}  // namespace mlir

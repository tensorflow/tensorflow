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
#include "tensorflow/compiler/mlir/lite/experimental/common/outline_operations.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/cluster_util.h"

namespace mlir {
namespace TFL {
namespace common {

bool IsConstantOrNone(Operation* op) {
  return (op->getNumResults() == 1 &&
          mlir::isa<NoneType>(op->getResult(0).getType())) ||
         matchPattern(op, m_Constant()) || isa<QConstOp>(op);
}

// Pre-order traverse, adding results and BlockArgs to `been_defined` and
// collecting operands not contained within `been_defined`. If we encounter an
// operand that references a Value that has been defined (and added to
// `been_defined`) it is garuanteed that the Value definition is not contained
// in descedant node of reference, and given that the input DAG is valid, the
// definition is self-contained within `op` so it is not depended upon.
// Otherwise, the operand must have been defined somewhere above the Subgraph,
// so union with other operand dependencies.
llvm::SmallVector<Value> AccumulateOperandsDefinedAbove(
    const llvm::SetVector<Operation*>& partition_ops) {
  // Assuming that all are topologically sorted.
  llvm::SetVector<Value> been_defined;
  llvm::SetVector<Value> results;
  auto update_from_op = [&](Operation* op) {
    been_defined.insert(op->getResults().begin(), op->getResults().end());
    for (Value input : op->getOperands()) {
      if (been_defined.contains(input)) {
        continue;
      }
      results.insert(input);
    }
  };
  for (Operation* op : partition_ops) {
    update_from_op(op);
    op->walk<WalkOrder::PreOrder>([&](Block* nested_block) {
      been_defined.insert(nested_block->getArguments().begin(),
                          nested_block->getArguments().end());
      for (Operation& op : nested_block->getOperations()) update_from_op(&op);
    });
  }
  return SmallVector<Value>(results.getArrayRef());
}

llvm::SmallVector<Value> AccumulateResultsDefinedWithin(
    const llvm::SetVector<Operation*>& partition_ops) {
  llvm::SmallVector<Value> values_for_results;
  for (Operation* op : partition_ops) {
    if (IsConstantOrNone(op)) {
      continue;
    }
    for (Value output : op->getResults()) {
      bool output_consumed_outside_subgraph = false;
      for (Operation* consumer : output.getUsers()) {
        if (llvm::all_of(partition_ops, [&](Operation* op) {
              return !op->isAncestor(consumer);
            })) {
          output_consumed_outside_subgraph = true;
        }
      }
      if (output_consumed_outside_subgraph) {
        values_for_results.push_back(output);
      }
    }
  }
  return values_for_results;
}

// Compute signature for raised func from arugments and outputs of
// Operation partition.
llvm::SmallVector<Type> TypesFromValues(
    const llvm::SmallVector<Value>& values) {
  llvm::SmallVector<Type> types;
  for (auto value : values) {
    types.push_back(value.getType());
  }
  return types;
}

func::FuncOp BuildFuncOp(const Subgraph& subgraph, OpBuilder& builder,
                         ModuleOp& module, OpsAdded& ops_added) {
  // The parameters of the new MLIR function are taken to be the union
  // of all operands referenced by Operations within the subraph.
  // Likewise the results of the function are any Value(s) that are defined
  // within the subgraph and are referenced outside the subgraph.
  llvm::SmallVector<Type> input_types =
      TypesFromValues(subgraph.FuncArguments());
  llvm::SmallVector<Type> return_types =
      TypesFromValues(subgraph.FuncOutputs());

  FunctionType function_type =
      builder.getFunctionType(input_types, return_types);

  std::string function_name = absl::StrCat("func_", subgraph.subgraph_id_);

  func::FuncOp new_func = func::FuncOp::create(builder.getUnknownLoc(),
                                               function_name, function_type);
  new_func.setVisibility(func::FuncOp::Visibility::Private);
  new_func.addEntryBlock();

  // To form the body of the new function we need to clone each
  // Operation along with its respective operands and result Values(s).
  // The semantic of `Operation::clone` is copying given entity *into* this
  // entity. The new FuncOp body is populated by cloning partitioned ops into
  // it. Cloning Operation(s) will create cloned Value(s) for the results of a
  // cloned op, but it needs a reference to the new operand Value(s) which are
  // the result of the cloned ops. The approach is to traverse the subgraph in
  // order, accumulating clones of defined Values into a `IRMapping`
  // and pass that map to calls to clone ops.
  OpBuilder function_builder(new_func.getBody());
  // Prefered data structure for mapping MLIR values.
  IRMapping values_in_scope;
  // Function arguments can appear as operands, so they clone should
  // be aware of them.
  assert(subgraph.FuncArguments().size() == new_func.getNumArguments());
  for (int i = 0; i < subgraph.FuncArguments().size(); ++i) {
    Value original_value = subgraph.FuncArguments()[i];
    Value new_func_arg = new_func.getArgument(i);
    values_in_scope.map(original_value, new_func_arg);
  }

  for (Operation* op : subgraph.partition_ops_) {
    function_builder.clone(*op, values_in_scope);
  }
  SmallVector<Value> return_operands;
  for (Value result : subgraph.FuncOutputs()) {
    Value cloned_output = values_in_scope.lookup(result);
    return_operands.push_back(cloned_output);
  }
  function_builder.create<mlir::func::ReturnOp>(new_func.getLoc(),
                                                return_operands);
  ops_added.func_op = new_func;
  module.push_back(new_func);
  return new_func;
}

void ExtractSubgraphToFunc(const Subgraph& subgraph, OpBuilder& builder,
                           ModuleOp& module, OpsAdded& ops_added) {
  func::FuncOp func = BuildFuncOp(subgraph, builder, module, ops_added);

  // We just use the location of the last ops in the subgraph as the location
  // for the call_op.
  Operation* last_output = subgraph.partition_ops_.back();

  builder.setInsertionPoint(last_output);
  auto call_op = builder.create<func::CallOp>(last_output->getLoc(), func,
                                              subgraph.FuncArguments());
  ops_added.call_op = call_op;
  // FuncOutputs refer to the original `Values` in input module which are now
  // invalid after pulling out the defining ops. The values in
  // `call_ops.getResult` refer to the clones of original `Values` which are now
  // returned by the new `FuncOp`. We can replace each in `FuncOutputs` with
  // clone in `call_op` to fix up.
  for (int i = 0; i < subgraph.FuncOutputs().size(); ++i) {
    Value output = subgraph.FuncOutputs()[i];
    output.replaceAllUsesWith(call_op.getResult(i));
  }

  // Clear the subgraph.
  // Those ops should be removed.
  for (auto* op : subgraph.partition_ops_) {
    if (IsConstantOrNone(op)) {
      continue;
    }
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  }
  // Ensure that users of the call op's results appear after the launch op in
  // order to preserve the dominance property.
  TF::ReorderOpResultUses(call_op);
}

}  // namespace common
}  // namespace TFL
}  // namespace mlir

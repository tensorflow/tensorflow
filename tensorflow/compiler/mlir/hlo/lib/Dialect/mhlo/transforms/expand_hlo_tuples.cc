/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace mhlo {
namespace {

// This pass assumes the function to be expanded has no callees, to be specific,
// the function is more like the main function.
class ExpandHloTuplesPass
    : public ExpandHloTuplesPassBase<ExpandHloTuplesPass> {
 public:
  ExpandHloTuplesPass() = default;
  ExpandHloTuplesPass(const ExpandHloTuplesPass&) {}
  explicit ExpandHloTuplesPass(const std::string& entry_function_name) {
    entry_function_name_ = entry_function_name;
  }

  // Expands the mhlo.tuple used in return op. Also updates function
  // signature accordingly.
  void ExpandTupledTensorInReturnOp(func::FuncOp func) {
    FunctionType old_func_type = func.getFunctionType();
    // Update input signatures.
    // We will flatten the tuples for the function inputs as well.
    // So if an input is tuple, will be flattened and packed as following:
    // func_1(%arg0: tuple<input1, input2>) =>
    //
    // func_1(%arg0: <input1>, %arg1: <input2>) {
    //  %0 = mhlo.tuple(%arg0, %arg1)
    // }
    SmallVector<Type, 4> expanded_input_types;
    SmallVector<BlockArgument, 20> func_arguments(func.getArguments().begin(),
                                                  func.getArguments().end());
    for (auto argument : func_arguments) {
      auto type = argument.getType();
      auto tuple_type = type.dyn_cast_or_null<TupleType>();
      if (!tuple_type) {
        expanded_input_types.push_back(type);
      } else {
        // We need to
        // 1) expand the tuple
        // 2) insert a new tuple
        // 3) rewire the new tuple
        int original_argument_index = argument.getArgNumber();
        int argument_index = original_argument_index;
        SmallVector<Value, 4> flattened_operands;
        // insert the flattened tuples after the original tuple.
        Location loc = func.getBody().getLoc();
        for (auto flattened_type : tuple_type.getTypes()) {
          expanded_input_types.push_back(flattened_type);
          func.insertArgument(++argument_index, flattened_type, {}, loc);
          flattened_operands.push_back(func.getArgument(argument_index));
        }

        // Construct a new tuple and rewire it.
        OpBuilder builder(func.getBody());
        builder.setInsertionPointToStart(&func.getBody().front());
        auto new_tuple =
            builder.create<mhlo::TupleOp>(loc, tuple_type, flattened_operands);
        func.getArgument(original_argument_index).replaceAllUsesWith(new_tuple);

        // Now the original argument has been rewired, we should be able to
        // safely erase it.
        func.eraseArgument(original_argument_index);
      }
    }

    // Update output signatures.
    auto return_op = cast<mlir::func::ReturnOp>(func.getBody().back().back());

    // Expand all tuples in old return operands.
    SmallVector<Value, 4> expanded_return_operands;
    SmallVector<Type, 4> expanded_result_types;
    for (auto value : return_op.getOperands()) {
      auto tuple = dyn_cast_or_null<mhlo::TupleOp>(value.getDefiningOp());
      if (!tuple) {
        expanded_return_operands.push_back(value);
        expanded_result_types.push_back(value.getType());
        continue;
      }

      for (auto tuple_operand : tuple.getOperands()) {
        expanded_return_operands.push_back(tuple_operand);
        expanded_result_types.push_back(tuple_operand.getType());
      }
    }

    if (expanded_return_operands.empty()) return;

    OpBuilder builder(return_op);
    builder.create<mlir::func::ReturnOp>(return_op.getLoc(),
                                         expanded_return_operands);
    return_op.erase();
    auto new_func_type =
        FunctionType::get(old_func_type.getContext(), expanded_input_types,
                          expanded_result_types);
    func.setType(new_func_type);
  }

  void runOnOperation() override {
    auto module = getOperation();
    // Find `main` function.
    auto entry_function =
        module.lookupSymbol<func::FuncOp>(entry_function_name_);
    if (!entry_function) {
      return;
    }

    ExpandTupledTensorInReturnOp(entry_function);
  }
};

}  // end namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateExpandHloTuplesPass(
    const std::string& entry_function_name) {
  return std::make_unique<ExpandHloTuplesPass>(entry_function_name);
}

}  // namespace mhlo
}  // namespace mlir

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

#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
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

#define GEN_PASS_DEF_EXPANDHLOTUPLESPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

// This pass assumes the function to be expanded has no callees, to be specific,
// the function is more like the main function.
class ExpandHloTuplesPass
    : public impl::ExpandHloTuplesPassBase<ExpandHloTuplesPass> {
 public:
  ExpandHloTuplesPass() = default;
  ExpandHloTuplesPass(const ExpandHloTuplesPass&) = default;
  explicit ExpandHloTuplesPass(const std::string& entryFunctionName) {
    entry_function_name_ = entryFunctionName;
  }

  // Expands the mhlo.tuple used in return op. Also updates function
  // signature accordingly.
  void expandTupledTensorInReturnOp(func::FuncOp func) {
    FunctionType oldFuncType = func.getFunctionType();
    // Update input signatures.
    // We will flatten the tuples for the function inputs as well.
    // So if an input is tuple, will be flattened and packed as following:
    // func_1(%arg0: tuple<input1, input2>) =>
    //
    // func_1(%arg0: <input1>, %arg1: <input2>) {
    //  %0 = mhlo.tuple(%arg0, %arg1)
    // }
    SmallVector<Type, 4> expandedInputTypes;
    SmallVector<BlockArgument, 20> funcArguments(func.getArguments().begin(),
                                                 func.getArguments().end());
    for (auto argument : funcArguments) {
      auto type = argument.getType();
      auto tupleType = type.dyn_cast_or_null<TupleType>();
      if (!tupleType) {
        expandedInputTypes.push_back(type);
      } else {
        // We need to
        // 1) expand the tuple
        // 2) insert a new tuple
        // 3) rewire the new tuple
        int originalArgumentIndex = argument.getArgNumber();
        int argumentIndex = originalArgumentIndex;
        SmallVector<Value, 4> flattenedOperands;
        // insert the flattened tuples after the original tuple.
        Location loc = func.getBody().getLoc();
        for (auto flattenedType : tupleType.getTypes()) {
          expandedInputTypes.push_back(flattenedType);
          func.insertArgument(++argumentIndex, flattenedType, {}, loc);
          flattenedOperands.push_back(func.getArgument(argumentIndex));
        }

        // Construct a new tuple and rewire it.
        OpBuilder builder(func.getBody());
        builder.setInsertionPointToStart(&func.getBody().front());
        auto newTuple =
            builder.create<mhlo::TupleOp>(loc, tupleType, flattenedOperands);
        func.getArgument(originalArgumentIndex).replaceAllUsesWith(newTuple);

        // Now the original argument has been rewired, we should be able to
        // safely erase it.
        func.eraseArgument(originalArgumentIndex);
      }
    }

    // Update output signatures.
    auto returnOp = cast<mlir::func::ReturnOp>(func.getBody().back().back());
    OpBuilder builder(returnOp);

    // Expand all tuples in old return operands.
    SmallVector<Value, 4> expandedReturnOperands;
    SmallVector<Type, 4> expandedResultTypes;
    for (auto value : returnOp.getOperands()) {
      if (auto tupleTy = value.getType().dyn_cast<TupleType>()) {
        llvm::copy(tupleTy.getTypes(), std::back_inserter(expandedResultTypes));
        for (auto [index, ty] : llvm::enumerate(tupleTy.getTypes())) {
          expandedReturnOperands.push_back(
              builder.createOrFold<mhlo::GetTupleElementOp>(value.getLoc(), ty,
                                                            value, index));
        }
      } else {
        expandedReturnOperands.push_back(value);
        expandedResultTypes.push_back(value.getType());
      }
    }

    if (returnOp.getOperands() == expandedReturnOperands) return;

    builder.create<mlir::func::ReturnOp>(returnOp.getLoc(),
                                         expandedReturnOperands);
    returnOp.erase();
    auto newFuncType = FunctionType::get(
        oldFuncType.getContext(), expandedInputTypes, expandedResultTypes);
    func.setType(newFuncType);
  }

  void runOnOperation() override {
    auto module = getOperation();
    // Find `main` function.
    auto entryFunction =
        module.lookupSymbol<func::FuncOp>(entry_function_name_);
    if (!entryFunction) {
      return;
    }

    // Recursively expand tuples until all of them are gone.
    while (
        llvm::any_of(llvm::concat<const Type>(entryFunction.getArgumentTypes(),
                                              entryFunction.getResultTypes()),
                     [](Type type) { return type.isa<TupleType>(); })) {
      expandTupledTensorInReturnOp(entryFunction);
    }
  }
};

}  // end namespace

std::unique_ptr<OperationPass<ModuleOp>> createExpandHloTuplesPass(
    const std::string& entryFunctionName) {
  return std::make_unique<ExpandHloTuplesPass>(entryFunctionName);
}

}  // namespace mhlo
}  // namespace mlir

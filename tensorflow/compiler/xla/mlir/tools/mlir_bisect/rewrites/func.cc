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

#include <functional>
#include <iterator>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/tools/mlir_bisect/bisect_lib.h"

namespace mlir {
namespace bisect {
namespace {

void SetReturnValues(func::FuncOp func, ValueRange values) {
  // We only operate on functions without arguments.
  func.setFunctionType(mlir::FunctionType::get(func.getContext(), /*inputs=*/{},
                                               values.getTypes()));
  func.getBody().getBlocks().front().getTerminator()->setOperands(values);
}

SmallVector<OwningOpRef<ModuleOp>> TruncateFunction(BisectState&,
                                                    func::FuncOp func) {
  SmallVector<OwningOpRef<ModuleOp>> result;
  for (auto& ret : func.getBody().getBlocks().front().without_terminator()) {
    if (func.getBody().getBlocks().front().getTerminator()->getOperands() ==
        ret.getResults()) {
      continue;
    }
    auto [module, ret_clone] = CloneModuleFor(&ret);
    SetReturnValues(ret_clone->getParentOfType<func::FuncOp>(),
                    ret_clone->getResults());
    result.push_back(std::move(module));
  }
  return result;
}

SmallVector<OwningOpRef<ModuleOp>> ReturnOperandsOfTerminatorOperands(
    BisectState&, func::FuncOp func) {
  SmallVector<OwningOpRef<ModuleOp>> result;
  auto [module, func_clone] = CloneModuleFor(func);
  auto* terminator = func_clone.getBody().getBlocks().front().getTerminator();
  SmallVector<Value> new_operands;
  for (auto operand : terminator->getOperands()) {
    if (operand.getDefiningOp()) {
      llvm::copy(operand.getDefiningOp()->getOperands(),
                 std::back_inserter(new_operands));
    } else {
      return result;
    }
  }
  SetReturnValues(func_clone, new_operands);
  result.push_back(std::move(module));
  return result;
}

REGISTER_MLIR_REDUCE_STRATEGY(TruncateFunction);
REGISTER_MLIR_REDUCE_STRATEGY(ReturnOperandsOfTerminatorOperands);

}  // namespace
}  // namespace bisect
}  // namespace mlir

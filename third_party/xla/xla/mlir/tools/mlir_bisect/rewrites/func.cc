/* Copyright 2023 The OpenXLA Authors. All Rights Reserved.

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

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir/tools/mlir_bisect/bisect_lib.h"

namespace mlir {
namespace bisect {
namespace {

void SetReturnValues(func::FuncOp func, ValueRange values) {
  // We only operate on functions without arguments.
  func.setFunctionType(mlir::FunctionType::get(func.getContext(), /*inputs=*/{},
                                               values.getTypes()));
  func.getBody().getBlocks().front().getTerminator()->setOperands(values);
}

SmallVector<std::function<OwningOpRef<ModuleOp>()>> TruncateFunction(
    BisectState&, func::FuncOp func) {
  SmallVector<std::function<OwningOpRef<ModuleOp>()>> result;
  for (auto& ret : func.getBody().getBlocks().front().without_terminator()) {
    if (func.getBody().getBlocks().front().getTerminator()->getOperands() ==
        ret.getResults()) {
      continue;
    }
    auto fun = [r = &ret]() -> OwningOpRef<ModuleOp> {
      auto [module, ret_clone] = CloneModuleFor(r);
      SetReturnValues(ret_clone->getParentOfType<func::FuncOp>(),
                      ret_clone->getResults());
      return std::move(module);
    };
    result.push_back(fun);
  }
  return result;
}

SmallVector<std::function<OwningOpRef<ModuleOp>()>>
ReturnOperandsOfTerminatorOperands(BisectState&, func::FuncOp func) {
  SmallVector<std::function<OwningOpRef<ModuleOp>()>> result;
  result.push_back([func]() -> OwningOpRef<ModuleOp> {
    auto [module, func_clone] = CloneModuleFor(func);
    auto* terminator = func_clone.getBody().getBlocks().front().getTerminator();
    SmallVector<Value> new_operands;
    for (auto operand : terminator->getOperands()) {
      if (operand.getDefiningOp()) {
        llvm::copy(operand.getDefiningOp()->getOperands(),
                   std::back_inserter(new_operands));
      } else {
        return nullptr;
      }
    }
    SetReturnValues(func_clone, new_operands);
    return std::move(module);
  });
  return result;
}

REGISTER_MLIR_REDUCE_STRATEGY(TruncateFunction);
REGISTER_MLIR_REDUCE_STRATEGY(ReturnOperandsOfTerminatorOperands);

}  // namespace
}  // namespace bisect
}  // namespace mlir

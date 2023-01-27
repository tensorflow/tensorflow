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
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/tools/mlir_bisect/bisect_lib.h"

namespace mlir {
namespace bisect {
namespace {

SmallVector<OwningOpRef<ModuleOp>> TruncateFunction(BisectState&,
                                                    func::FuncOp func) {
  SmallVector<OwningOpRef<ModuleOp>> result;
  for (auto& ret : func.getBody().getBlocks().front().without_terminator()) {
    if (func.getBody().getBlocks().front().getTerminator()->getOperands() ==
        ret.getResults()) {
      continue;
    }
    auto [module_clone, ret_clone] = CloneModuleFor(&ret);
    auto func_clone = ret_clone->getParentOfType<func::FuncOp>();
    // We only operate on functions without arguments.
    func_clone.setFunctionType(mlir::FunctionType::get(
        func_clone.getContext(), /*inputs=*/{}, ret_clone->getResultTypes()));
    func_clone.getBody().getBlocks().front().getTerminator()->setOperands(
        ret_clone->getResults());
    result.push_back(std::move(module_clone));
  }
  return result;
}

REGISTER_MLIR_REDUCE_STRATEGY(TruncateFunction);

}  // namespace
}  // namespace bisect
}  // namespace mlir

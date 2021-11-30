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
#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback_util.h"

#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback_async.h"

namespace tfrt {
namespace fallback_async {

bool IsArgConsumedByFallback(mlir::FuncOp func, int arg_index) {
  auto arg = func.getArgument(arg_index);

  // Return true if any user is a fallback op. It is more interesting to know
  // whether it is consumed by any fallback op than whether it is only consumed
  // by fallback ops. For example, the conversion from a DenseHostTensor to a
  // fallback tensor is more costly than the conversion from fallback tensor to
  // a DenseHostTensor. So as long as one of the users of a captured variable is
  // a fallback op, we should keep this variable as a fallback tensor.
  for (mlir::Operation *user : arg.getUsers()) {
    if (llvm::isa<FallbackAsyncDialect>(user->getDialect())) return true;
  }
  return false;
}

void ForEachArgConsumedByFallback(
    mlir::FuncOp func, llvm::function_ref<void(int arg_index)> action) {
  for (int arg_index = 0; arg_index < func.getNumArguments(); ++arg_index) {
    if (IsArgConsumedByFallback(func, arg_index)) action(arg_index);
  }
}

void ForEachArgConsumedByFallback(
    mlir::ModuleOp module,
    llvm::function_ref<void(llvm::StringRef func_name, int arg_index)> action) {
  for (auto func : module.getOps<mlir::FuncOp>()) {
    ForEachArgConsumedByFallback(
        func, [func_name = func.getName(), action](int arg_index) {
          action(func_name, arg_index);
        });
  }
}

}  // namespace fallback_async
}  // namespace tfrt

/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/fusions/triton/passes.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"

namespace xla::gpu {

bool ContainsOp(mlir::Operation* op,
                llvm::function_ref<bool(mlir::Operation*)> fn) {
  auto visitor = [&](mlir::Operation* nested_op) {
    return fn(nested_op) ? mlir::WalkResult::interrupt()
                         : mlir::WalkResult::advance();
  };
  return op->walk(visitor).wasInterrupted();
}

}  // namespace xla::gpu

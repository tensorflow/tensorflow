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
#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_PASSES_MLIR_TO_GRAPH_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_PASSES_MLIR_TO_GRAPH_PASSES_H_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace tensorflow {
namespace tf2xla {
namespace internal {

// Verifies that Executor input is of the expected format.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateVerifyInputDialectToExecutorPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_VERIFYINPUTDIALECTTOEXECUTORPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/mlir_to_graph_passes.h.inc"

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_PASSES_MLIR_TO_GRAPH_PASSES_H_

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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_ASSIGN_OP_KEY_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_ASSIGN_OP_KEY_H_
#include <memory>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace tensorflow {
namespace mlrt_compiler {

// Create a pass that assigns an op_key to every fallback OP. The op_key
// provides a uniform key to look up online cost for a specific op.
// This pass is expected to run before parallerization.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateAssignOpKeyPass();

}  // namespace mlrt_compiler
}  // namespace tensorflow
#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_ASSIGN_OP_KEY_H_

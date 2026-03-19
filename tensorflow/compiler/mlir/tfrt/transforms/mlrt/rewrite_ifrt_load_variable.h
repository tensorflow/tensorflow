/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_REWRITE_IFRT_LOAD_VARIABLE_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_REWRITE_IFRT_LOAD_VARIABLE_H_

#include <memory>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace tensorflow {
namespace mlrt_compiler {

// Creates a pass that converts tf.IfrtLoadVariableOp to
// tf_mlrt.TFIfrtLoadVariableOp and inserts tf_mlrt.Await on the returned future
// from tf_mlrt.TFIfrtLoadVariableOp if it is used by CPU ops.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateRewriteIfrtLoadVariablePass();

}  // namespace mlrt_compiler
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_REWRITE_IFRT_LOAD_VARIABLE_H_

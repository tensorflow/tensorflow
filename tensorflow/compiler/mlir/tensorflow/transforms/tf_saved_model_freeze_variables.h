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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_SAVED_MODEL_FREEZE_VARIABLES_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_SAVED_MODEL_FREEZE_VARIABLES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/public/session.h"

namespace mlir {
namespace tf_saved_model {

// Freezes readonly variables in the graph.
LogicalResult FreezeVariables(ModuleOp module, tensorflow::Session* session);

}  // namespace tf_saved_model

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TF_SAVED_MODEL_FREEZE_VARIABLES_H_

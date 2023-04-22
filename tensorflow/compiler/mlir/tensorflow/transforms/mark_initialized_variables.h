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
#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_MARK_INITIALIZED_VARIABLES_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_MARK_INITIALIZED_VARIABLES_H_

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/public/session.h"

namespace mlir {
namespace tf_saved_model {
// Marks all variables in 'function' whether they are initialized
// in 'session' or not by setting an attribute named 'is_initialized'
// on each variable op with value true/false based on variable is initialized
// in the session or not.
// If 'session' is NULL the function is no-op.
// Returns failure in case fetching variables from session failed, success
// otherwise.
LogicalResult MarkInitializedVariablesInFunction(FuncOp function,
                                                 tensorflow::Session* session,
                                                 mlir::MLIRContext* context);

}  // namespace tf_saved_model
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_MARK_INITIALIZED_VARIABLES_H_

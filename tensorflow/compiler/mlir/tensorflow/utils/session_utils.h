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
#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_SESSION_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_SESSION_UTILS_H_

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/public/session.h"

namespace mlir {
namespace tf_saved_model {

// Returns the variable for the provided 'var_handle_op'.
std::string GetVariableName(TF::VarHandleOp var_handle_op);

// Returns pointer to the variable from 'session' that 'var_handle_op'
// refers to which is in 'device_name' device. If failed to fetch the value null
// will be returned.
// Note, caller is responsible for Unref the variable.
tensorflow::Var* GetVariableFromSession(mlir::TF::VarHandleOp var_handle_op,
                                        llvm::StringRef device_name,
                                        const tensorflow::DeviceMgr* mgr);

// Returns resource tensors from session for all variables in 'module'.
absl::StatusOr<std::vector<tensorflow::Tensor>> GetResourcesFromSession(
    llvm::ArrayRef<TF::VarHandleOp> var_handle_ops,
    tensorflow::Session* session);

}  // namespace tf_saved_model
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_SESSION_UTILS_H_

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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_UTILS_EXPORT_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_UTILS_EXPORT_H_


#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/framework/function.pb.h"

namespace tensorflow {

// Exports every function in `module` into `tensorflow.FunctionDef` and calls
// `callback` for each `tensorflow.FunctionDef`. Modifies `module` in place to
// be suitable for FunctionDef export.
absl::Status ExportFunctionDefs(
    mlir::ModuleOp module,
    absl::AnyInvocable<absl::Status(tensorflow::FunctionDef)> callback,
    bool export_tf_original_func_name = true);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_UTILS_EXPORT_H_

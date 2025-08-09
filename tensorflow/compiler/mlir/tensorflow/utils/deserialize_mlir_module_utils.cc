/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/deserialize_mlir_module_utils.h"

#include "absl/status/status.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "xla/status_macros.h"

namespace tensorflow {

absl::Status DeserializeMlirModule(
    llvm::StringRef serialized_mlir_module, mlir::MLIRContext* mlir_context,
    mlir::OwningOpRef<mlir::ModuleOp>* mlir_module) {
  TF_RET_CHECK(!serialized_mlir_module.empty())
      << "unexpected empty serialized MLIR module string";
  TF_RET_CHECK(mlir_module) << "unexpected null MLIR module pointer";

  // Make sure we catch any error reported by MLIR and forward it to the TF
  // error reporting system.
  mlir::StatusScopedDiagnosticHandler error_handler(mlir_context);

  // Parse the module.
  *mlir_module = mlir::parseSourceString<mlir::ModuleOp>(serialized_mlir_module,
                                                         mlir_context);
  if (!*mlir_module)
    return error_handler.Combine(
        absl::InvalidArgumentError("could not parse MLIR module"));

  return absl::OkStatus();
}

}  // namespace tensorflow

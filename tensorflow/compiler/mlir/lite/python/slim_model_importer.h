/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_SLIM_MODEL_IMPORTER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_SLIM_MODEL_IMPORTER_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project

namespace tensorflow {

// Loads a slim model from a directory.
// The directory is expected to contain:
// - weights_metadata.json
// - params.bin
// - one or more .mlirbc files as specified in weights_metadata.json
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> LoadSlimModel(
    absl::string_view model_dir, mlir::MLIRContext* context);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_SLIM_MODEL_IMPORTER_H_

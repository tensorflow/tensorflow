/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_TF_MLIR_TRANSLATE_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_TF_MLIR_TRANSLATE_H_

#include <memory>
#include <string>
#include <unordered_set>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"

namespace tensorflow {

// Converts a TensorFlow SavedModel stored in the directory with the given
// `saved_model_dir` into a MLIR module. Creates MLIR entities into the
// given MLIR `context`.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelObjectGraphToMlirImport(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags,
    absl::Span<std::string> exported_names, mlir::MLIRContext* context,
    bool unconditionally_use_set_output_shapes = false,
    bool import_variables_as_dense_resources = false);

// Converts a TensorFlow V1 SavedModel stored in the directory with the given
// `saved_model_dir` into a MLIR module. Creates MLIR entities into the
// given MLIR `context`.
// 'saved_model_bundle' if not null, will be initialized with the model bundle.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelSignatureDefsToMlirImport(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags,
    absl::Span<std::string> exported_names, mlir::MLIRContext* context,
    MLIRImportOptions options,
    std::unique_ptr<tensorflow::SavedModelBundle>* saved_model_bundle =
        nullptr);

// Converts a TensorFlow V1 SavedModel stored in the directory with the given
// `saved_model_dir` into a MLIR module. Creates MLIR entities into the
// given MLIR `context`. This does not create session internally so it is faster
// and does not perform any graph transformation.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelSignatureDefsToMlirImportLite(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags,
    absl::Span<std::string> exported_names, mlir::MLIRContext* context,
    MLIRImportOptions options);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_TF_MLIR_TRANSLATE_H_

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

#include <string>
#include <unordered_set>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project

namespace tensorflow {
// TODO(antiagainst): Directly manipulating files in library functions is not
// a good idea. We should pass in a string/stream here.

// Converts a TensorFlow GraphDef stored in the file with the given
// `input_filename` into a MLIR module. Creates MLIR entities into the
// given MLIR `context`.
mlir::OwningModuleRef GraphdefToMlirTranslateFunction(
    llvm::StringRef input, absl::string_view debug_info_file,
    absl::string_view input_arrays, absl::string_view input_dtypes,
    absl::string_view input_shapes, absl::string_view output_arrays,
    absl::string_view control_output_arrays, bool prune_unused_nodes,
    bool convert_legacy_fed_inputs, bool graph_as_function, bool upgrade_legacy,
    mlir::MLIRContext* context);

// Similar as the above function, but replaces all constant tensors
// with randomly generated splat values.
mlir::OwningModuleRef GraphdefToSplattedMlirTranslateFunction(
    llvm::StringRef input, absl::string_view debug_info_file,
    absl::string_view input_arrays, absl::string_view input_dtypes,
    absl::string_view input_shapes, absl::string_view output_arrays,
    absl::string_view control_output_arrays, bool prune_unused_nodes,
    bool convert_legacy_fed_inputs, bool graph_as_function, bool upgrade_legacy,
    mlir::MLIRContext* context);

// Converts a TensorFlow SavedModel stored in the directory with the given
// `saved_model_dir` into a MLIR module. Creates MLIR entities into the
// given MLIR `context`.
mlir::OwningModuleRef SavedModelObjectGraphToMlirImport(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags,
    absl::Span<std::string> exported_names, mlir::MLIRContext* context);

// Converts a TensorFlow V1 SavedModel stored in the directory with the given
// `saved_model_dir` into a MLIR module. Creates MLIR entities into the
// given MLIR `context`.
mlir::OwningModuleRef SavedModelSignatureDefsToMlirImport(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags, mlir::MLIRContext* context);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_TF_MLIR_TRANSLATE_H_

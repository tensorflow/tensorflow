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

#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir

namespace tensorflow {
// TODO(antiagainst): Directly manipulating files in library functions is not
// a good idea. We should pass in a string/stream here.

// Converts a TensorFlow GraphDef stored in the file with the given
// `input_filename` into a MLIR module. Creates MLIR entities into the
// given MLIR `context`.
mlir::OwningModuleRef GraphdefToMlirTranslateFunction(
    absl::string_view input_filename, absl::string_view debug_info_file,
    absl::string_view input_arrays, absl::string_view input_dtypes,
    absl::string_view input_shapes, absl::string_view output_arrays,
    absl::string_view inference_type, absl::string_view min_values,
    absl::string_view max_values, bool prune_unused_nodes,
    mlir::MLIRContext* context);

// Similar as the above function, but replaces all constant tensors
// with randomly generated splat values.
mlir::OwningModuleRef GraphdefToSplattedMlirTranslateFunction(
    absl::string_view input_filename, absl::string_view debug_info_file,
    absl::string_view input_arrays, absl::string_view input_dtypes,
    absl::string_view input_shapes, absl::string_view output_arrays,
    absl::string_view inference_type, absl::string_view min_values,
    absl::string_view max_values, bool prune_unused_nodes,
    mlir::MLIRContext* context);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_TF_MLIR_TRANSLATE_H_

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TO_STABLEHLO_TF_TO_STABLEHLO_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TO_STABLEHLO_TF_TO_STABLEHLO_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project

namespace mlir {

// Converts a TensorFlow model (either from a SavedModel or an MLIR module) to a
// StableHLO MLIR module.
//
// Args:
//  input_path: The path to the input TensorFlow SavedModel or MLIR module.
//  context: The MLIR context to use for parsing or creating the MLIR module.
//  exported_model_signatures: List of exported model signatures (strings) to
//    convert.
//  tag_names: List of tag names (strings) used for loading SavedModel.
//    Ignored for MLIR input.
//  input_arg_shapes_str:  A string representation of input argument shapes for
//    'main' entry-point, separating tensors with ':', dimension with ',', and
//    using '?' for unknown sizes. For example, 'input-arg-shapes=1,2::1,?'
//    expresses argument shapes [1,2], [] and [1,?].
//  is_input_mlir_module: If true, `input_path` is treated as an MLIR
//    module instead of a SavedModel.
//
// Returns:
//   An absl::StatusOr containing the converted StableHLO MLIR module on
//   success, or an absl::Status with an error message on failure.
absl::StatusOr<OwningOpRef<ModuleOp>> TfToStablehlo(
    absl::string_view input_path, MLIRContext* context,
    const std::vector<std::string>& exported_model_signatures,
    const std::vector<std::string>& tag_names,
    absl::string_view input_arg_shapes_str, bool is_input_mlir_module);

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TO_STABLEHLO_TF_TO_STABLEHLO_H_

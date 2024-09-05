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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TO_STABLEHLO_PYTHON_PYWRAP_TENSORFLOW_TO_STABLEHLO_LIB_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TO_STABLEHLO_PYTHON_PYWRAP_TENSORFLOW_TO_STABLEHLO_LIB_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace mlir::tensorflow_to_stablehlo::pywrap {

// Converts a TensorFlow SavedModel to a StableHLO MLIR module and serializes it
// to bytecode.
//
// Args:
//   input_path: The path to the SavedModel directory.
//    exported_model_signatures: Comma-separated list of exported model
//   signatures to convert. tag_names: Comma-separated list of tags for loading
//    SavedModel.
//   input_arg_shapes_str: A string representation of input argument
//    shapes for 'main' entry-point, separating tensors with ':', dimension
//    with ',', and using '?' for unknown sizes. For example,
//    'input-arg-shapes=1,2::1,?' expresses argument shapes [1,2], [] and [1,?].
//
// Returns:
//   An absl::StatusOr containing the serialized bytecode of the StableHLO
//   module on success, or an error status on failure.
absl::StatusOr<std::string> PywrapSavedModelToStablehlo(
    absl::string_view input_path,
    const std::vector<std::string>& exported_model_signatures,
    const std::vector<std::string>& tag_names,
    absl::string_view input_arg_shapes_str);

// Converts a TensorFlow MLIR module string to a StableHLO MLIR module and
// serializes it to bytecode.
//
// Args:
//   module_op_str: TensorFlow MLIR module string.
//   input_arg_shapes_str: A string representation of input argument
//    shapes for 'main' entry-point, separating tensors with ':', dimension
//    with ',', and using '?' for unknown sizes. For example,
//    'input-arg-shapes=1,2::1,?' expresses argument shapes [1,2], [] and [1,?].
//
// Returns:
//   An absl::StatusOr containing the serialized bytecode of the StableHLO
//   module on success, or an error status on failure.
absl::StatusOr<std::string> PywrapTfModuleToStablehlo(
    absl::string_view module_op_str, absl::string_view input_arg_shapes_str);

}  // namespace mlir::tensorflow_to_stablehlo::pywrap

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TO_STABLEHLO_PYTHON_PYWRAP_TENSORFLOW_TO_STABLEHLO_LIB_H_

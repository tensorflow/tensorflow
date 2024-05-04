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

absl::StatusOr<std::string> PywrapSavedModelToStablehlo(
    absl::string_view input_path,
    const std::vector<std::string>& exported_model_signatures,
    const std::vector<std::string>& tag_names,
    absl::string_view input_arg_shapes_str);

absl::StatusOr<std::string> PywrapTfModuleToStablehlo(
    absl::string_view module_op_str,
    const std::vector<std::string>& exported_model_signatures,
    const std::vector<std::string>& tag_names,
    absl::string_view input_arg_shapes_str);

}  // namespace mlir::tensorflow_to_stablehlo::pywrap

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TO_STABLEHLO_PYTHON_PYWRAP_TENSORFLOW_TO_STABLEHLO_LIB_H_

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_EXPORT_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_EXPORT_H_

#include <string>

#include "absl/strings/string_view.h"

namespace stablehlo::quantization {

// Suffix string for the module export step. Used for debugging.
constexpr absl::string_view kExportStepSuffix = "_export";

// Options when running passes for exporting an MLIR ModuleOp.
struct ExportOptions {
  // If set to `true`, it runs `DuplicateShapeDeterminingConstantsPass` before
  // lowering to tf_executor dialect.
  bool duplicate_shape_determining_constants = true;

  // If set to `true`, unfreezes constants into variables and saves them to a
  // checkpoint file. Setting this to `true` is an experimental feature that has
  // no stability guarantees.
  bool unfreeze_constants = false;

  // Path to the directory where checkpoint files are saved.
  std::string checkpoint_dir = "";

  // Name used to identify the ModuleOp this is exporting. Only used for
  // debugging and does not modify the behavior of the export.
  std::string debug_name = "stablehlo_quant";
};

}  // namespace stablehlo::quantization

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_EXPORT_H_

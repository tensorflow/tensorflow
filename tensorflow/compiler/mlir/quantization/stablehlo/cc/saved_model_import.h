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
// Functionalities for importing MLIR ModuleOp from TensorFlow SavedModel.

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_SAVED_MODEL_IMPORT_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_SAVED_MODEL_IMPORT_H_

#include <string>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/types.h"

namespace mlir::quant::stablehlo {

// Gets the function aliases from the SavedModel.
absl::StatusOr<absl::flat_hash_map<FunctionName, FunctionAlias>>
GetFunctionAliases(absl::string_view saved_model_path,
                   const std::unordered_set<std::string>& tags);

// Updates the function aliases. `module_op` may have different
// function names from the original model, so it re-associates the aliases
// with the new function names. Both the input `function_aliases` and the
// returned value are function name -> alias mappings. `function_aliases` is
// the function alias mapping of the original function. The original function's
// name is retrieved by looking at the "tf._original_func_name" string attribute
// attached to a `func::FuncOp`.
void UpdateFunctionAliases(
    absl::flat_hash_map<FunctionName, FunctionAlias>& function_aliases,
    ModuleOp module_op);

}  // namespace mlir::quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_SAVED_MODEL_IMPORT_H_

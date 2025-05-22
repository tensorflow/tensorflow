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

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_TF_SAVED_MODEL_IMPORT_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_TF_SAVED_MODEL_IMPORT_H_

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/types.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"

namespace mlir::tf_quant::stablehlo {

// Represents a pair of `mlir::ModuleOp` and `tensorflow::SavedModelBundle`. The
// SavedModelBundle complements the imported ModuleOp by providing access to
// `tensorflow::Session` which may be useful when reading values from resources
// (e.g. `TF::VarHandleOp`s).
using ImportedMlirModuleOp =
    std::pair<OwningOpRef<ModuleOp>,
              std::unique_ptr<::tensorflow::SavedModelBundle>>;
using quant::stablehlo::FunctionAlias;
using quant::stablehlo::FunctionName;

// Loads a SavedModel at `saved_model_path` and converts it to `mlir::ModuleOp`.
//
// `tags` identify the `tensorflow::MetaGraphDef` to load from the SavedModel.
// Similarly, `signature_keys` identify the functions (`SignatureDef`s) to load
// within the `MetaGraphDef`. `ctx` is the `MLIRContext`, which should outlive
// the returned `ModuleOp`, thus marked with the lifetime bound attribute.
// TODO: b/329206105 - Add unit tests after decomposing preprocessing passes.
absl::StatusOr<ImportedMlirModuleOp> SavedModelToMlirModuleOp(
    absl::string_view saved_model_path,
    const std::unordered_set<std::string>& tags,
    const std::vector<std::string>& signature_keys,
    MLIRContext& ctx ABSL_ATTRIBUTE_LIFETIME_BOUND);

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

// Loads a SavedModel to `mlir::ModuleOp` and performs preprocesses including
// shape inference and graph freezing.
// TODO: b/329206105 - Add unit tests after decomposing preprocessing passes.
absl::StatusOr<OwningOpRef<ModuleOp>> ImportSavedModel(
    absl::string_view saved_model_path,
    const std::vector<std::string>& signature_keys,
    const std::unordered_set<std::string>& tags,
    const ::stablehlo::quantization::QuantizationConfig& quantization_config,
    absl::string_view mlir_dump_file_prefix,
    absl::flat_hash_map<FunctionName, FunctionAlias>& function_aliases,
    MLIRContext& ctx ABSL_ATTRIBUTE_LIFETIME_BOUND);

}  // namespace mlir::tf_quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_TF_SAVED_MODEL_IMPORT_H_

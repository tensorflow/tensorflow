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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/saved_model_import.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/types.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/tf_quantize_preprocess.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace mlir::quant::stablehlo {

using ::stablehlo::quantization::QuantizationConfig;
using ::tensorflow::MLIRImportOptions;
using ::tensorflow::SavedModelBundle;
using ::tensorflow::SavedModelSignatureDefsToMlirImport;
using ::tensorflow::tf_quantization::PreprocessAndFreezeGraph;

absl::StatusOr<ImportedMlirModuleOp> SavedModelToMlirModuleOp(
    const absl::string_view saved_model_path,
    const std::unordered_set<std::string>& tags,
    const std::vector<std::string>& signature_keys,
    MLIRContext& ctx ABSL_ATTRIBUTE_LIFETIME_BOUND) {
  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  import_options.lift_variables = false;
  import_options.include_variables_in_initializers = true;

  auto bundle = std::make_unique<SavedModelBundle>();

  // Copy to eliminate the `const` qualifier so that `absl::MakeSpan` can be
  // called on it.
  std::vector<std::string> exported_names = signature_keys;
  absl::StatusOr<OwningOpRef<ModuleOp>> module_op =
      SavedModelSignatureDefsToMlirImport(saved_model_path, tags,
                                          absl::MakeSpan(exported_names), &ctx,
                                          import_options, &bundle);
  if (!module_op.status().ok()) {
    return absl::InternalError(absl::StrCat("Failed to import SavedModel: ",
                                            module_op.status().ToString()));
  }

  return std::make_pair(std::move(*module_op), std::move(bundle));
}

absl::StatusOr<absl::flat_hash_map<FunctionName, FunctionAlias>>
GetFunctionAliases(absl::string_view saved_model_path,
                   const std::unordered_set<std::string>& tags) {
  tensorflow::MetaGraphDef meta_graph;
  TF_RETURN_IF_ERROR(tensorflow::ReadMetaGraphDefFromSavedModel(
      saved_model_path, tags, &meta_graph));

  absl::flat_hash_map<FunctionName, FunctionAlias> function_aliases(
      meta_graph.meta_info_def().function_aliases().begin(),
      meta_graph.meta_info_def().function_aliases().end());
  return function_aliases;
}

void UpdateFunctionAliases(
    absl::flat_hash_map<FunctionName, FunctionAlias>& function_aliases,
    ModuleOp module_op) {
  absl::flat_hash_set<FunctionName> existing_func_names;
  module_op->walk([&](func::FuncOp func_op) {
    FunctionName func_name = func_op.getSymName().str();
    existing_func_names.insert(func_name);
    // We may retrieve the original function's name from the attribute.
    // Functions without this attribute are ignored.
    auto original_func_name =
        func_op->getAttrOfType<StringAttr>("tf._original_func_name");
    if (original_func_name) {
      if (auto alias_itr = function_aliases.find(original_func_name.str());
          alias_itr != function_aliases.end()) {
        const FunctionAlias alias = alias_itr->second;
        function_aliases[func_name] = alias;
      }
    }
  });

  // Remove aliases to function that no-longer exists.
  absl::erase_if(function_aliases, [&existing_func_names](const auto& item) {
    return !existing_func_names.contains(item.first);
  });
}

absl::StatusOr<OwningOpRef<ModuleOp>> ImportSavedModel(
    const absl::string_view saved_model_path,
    const std::vector<std::string>& signature_keys,
    const std::unordered_set<std::string>& tags,
    const QuantizationConfig& quantization_config,
    const absl::string_view mlir_dump_file_prefix,
    absl::flat_hash_map<FunctionName, FunctionAlias>& function_aliases,
    MLIRContext& ctx ABSL_ATTRIBUTE_LIFETIME_BOUND) {
  TF_ASSIGN_OR_RETURN(
      ImportedMlirModuleOp imported_module,
      SavedModelToMlirModuleOp(saved_model_path, tags, signature_keys, ctx));
  auto [module_op, saved_model_bundle] = std::move(imported_module);

  UpdateFunctionAliases(function_aliases, *module_op);

  // Collect the names of the functions that have aliases so that they may not
  // be inlined.
  absl::flat_hash_set<std::string> aliased_function_names;
  absl::c_for_each(function_aliases, [&](const auto& aliases) {
    return aliased_function_names.insert(aliases.first);
  });

  TF_RETURN_IF_ERROR(PreprocessAndFreezeGraph(
      mlir_dump_file_prefix, /*is_inliner_run=*/true,
      /*noinline_functions=*/aliased_function_names, *module_op, &ctx,
      saved_model_bundle == nullptr ? nullptr
                                    : saved_model_bundle->GetSession(),
      /*run_tf_to_stablehlo=*/true, /*deserialize_xla_call_module=*/false));
  return std::move(module_op);
}

}  // namespace mlir::quant::stablehlo

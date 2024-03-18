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

#include <string>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tsl/platform/errors.h"

namespace mlir::quant::stablehlo {

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

}  // namespace mlir::quant::stablehlo

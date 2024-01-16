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
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace mlir::quant::stablehlo {

absl::flat_hash_map<FunctionName, FunctionAlias> UpdateFunctionAliases(
    const absl::flat_hash_map<FunctionName, FunctionAlias> function_aliases,
    ModuleOp module_op) {
  absl::flat_hash_map<FunctionName, FunctionAlias> updated_function_aliases;

  module_op->walk([&](func::FuncOp func_op) {
    // We may retrieve the original function's name from the attribute.
    // Functions without this attribute are ignored.
    auto original_func_name =
        func_op->getAttrOfType<StringAttr>("tf._original_func_name");
    if (original_func_name) {
      if (auto alias_itr = function_aliases.find(original_func_name.str());
          alias_itr != function_aliases.end()) {
        const FunctionAlias alias = alias_itr->second;
        const FunctionName new_func_name = func_op.getSymName().str();

        updated_function_aliases[new_func_name] = alias;
      }
    }
  });

  return updated_function_aliases;
}

}  // namespace mlir::quant::stablehlo

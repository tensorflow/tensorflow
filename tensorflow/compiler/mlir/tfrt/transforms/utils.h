/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_UTILS_H_

#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project

namespace tensorflow {

// Checks if the given `value` is a resource argument.
bool IsResourceArgument(mlir::Value value);

// Checks if an operand is the value of a variable.
bool IsResultVariable(const mlir::Value &original_operand,
                      const mlir::Value &operand);

// Canonicalize the symbol attr to the original TF function name.
std::optional<std::string> CanonicalizeTensorflowFunctionName(
    const mlir::SymbolTable &symbol_table, absl::string_view mlir_func_name,
    bool use_mlir_func_name = false);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_UTILS_H_

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_SERIALIZE_MLIR_MODULE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_SERIALIZE_MLIR_MODULE_UTILS_H_

#include <string>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Prints a MLIR module `module_op` and returns it as a string.
std::string SerializeMlirModule(mlir::ModuleOp module_op);

// Parses a MLIR module from `mlir_module_string` into `mlir_module` with
// context `mlir_context`.
absl::Status DeserializeMlirModule(
    llvm::StringRef serialized_mlir_module, mlir::MLIRContext* mlir_context,
    mlir::OwningOpRef<mlir::ModuleOp>* mlir_module);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_SERIALIZE_MLIR_MODULE_UTILS_H_

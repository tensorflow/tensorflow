/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_SUPPORT_MODULE_PARSING_H_
#define XLA_PYTHON_IFRT_SUPPORT_MODULE_PARSING_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace xla {
namespace ifrt {
namespace support {

// Initializes the given MLIR dialect registry with dialects that are required
// by IFRT IR passes.
void InitializeMlirDialectRegistry(mlir::DialectRegistry& registry);

// Registers all dialects required by IFRT IR modules.
void RegisterMlirDialects(mlir::MLIRContext& context);

// Converts an IFRT IR module string to an mlir::Module.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseMlirModuleString(
    absl::string_view mlir_module_str, mlir::MLIRContext& context);

}  // namespace support
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SUPPORT_MODULE_PARSING_H_

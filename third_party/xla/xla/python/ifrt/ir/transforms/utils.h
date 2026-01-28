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

#ifndef XLA_PYTHON_IFRT_IR_TRANSFORMS_UTILS_H_
#define XLA_PYTHON_IFRT_IR_TRANSFORMS_UTILS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"

namespace xla {
namespace ifrt {

// Used for comparing CallOps without including control dependencies.
struct IfrtCallOpInfo : llvm::DenseMapInfo<xla::ifrt::CallOp> {
  static unsigned getHashValue(xla::ifrt::CallOp call_op);
  static bool isEqual(xla::ifrt::CallOp lhs, xla::ifrt::CallOp rhs);
};

// Retrieves the function named "main" from the given module, if it exists, and
// fails otherwise.
mlir::func::FuncOp GetMainFunction(mlir::ModuleOp module);

// Returns true if transferring between from and to array requires a reshard.
bool IsReshard(xla::ifrt::IfrtArrayType from, xla::ifrt::IfrtArrayType to);

// Updates the FunctionType of the given `func_op` to match the block arguments
// types and return operands types in its region.
void UpdateFunctionType(mlir::func::FuncOp func_op);

// Converts a mlir::Type to a ifrt DType.
absl::StatusOr<DType> ToIfrtDType(mlir::Type type);

// Prints the MLIR operation as a string.
std::string OperationToString(mlir::Operation* op,
                              const mlir::OpPrintingFlags& flags);

// Clones a given mlir::ModuleOp into the given MLIR context.
// Note: This function is loading dialects into the context, and thus it is
// not thread-safe w.r.t. calling it with the same context from multiple
// threads.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CloneModuleIntoContext(
    mlir::ModuleOp module, mlir::MLIRContext& context);

// Clones a given mlir::ModuleOp using a mlir::OpBuilder. This function is used
// to clone a module into a new MLIR context, which was used to construct the
// builder. For other cases, regular mlir::ModuleOp::clone() should be used.
mlir::ModuleOp CloneModuleUsingBuilder(mlir::ModuleOp module,
                                       mlir::OpBuilder& builder);

// Expands a vector of platform names from short format (e.g., tpu:2,host:2) to
// long format with an entry for each platform instance.
absl::StatusOr<std::vector<std::string>> ExpandPlatformNames(
    const mlir::Pass::ListOption<std::string>& platform_names);

// Returns a pretty string representation of the location.
std::string GetPrettyLocation(mlir::Location loc);

// Returns a fingerprint of the provided module.
uint64_t MlirModuleFingerprint(mlir::ModuleOp module);

// Extracts the XLA compile options overrides for the given atom program module.
// Returns std::nullopt if no overrides are found.
absl::StatusOr<std::optional<xla::CompileOptions>> GetModuleXlaCompileOverrides(
    mlir::StringAttr compile_options_key,
    std::shared_ptr<
        absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
        compile_options_overrides);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_TRANSFORMS_UTILS_H_

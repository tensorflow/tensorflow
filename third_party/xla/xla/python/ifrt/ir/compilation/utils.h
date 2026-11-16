/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_COMPILATION_UTILS_H_
#define XLA_PYTHON_IFRT_IR_COMPILATION_UTILS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"

namespace xla {
namespace ifrt {

// Returns true if `platform_name` identifies an XLA-compiled device platform
// (`tpu`, `cuda`, `rocm`, or `cpu`).
bool IsXlaPlatform(absl::string_view platform_name);

// Looks up the platform name for logical `device_id` in `platform_names`,
// emitting an op error on `op` if `device_id` is out of bounds. The returned
// `llvm::StringRef` borrows from the matching element in `platform_names` and
// is only valid while `platform_names` and its elements remain alive.
mlir::FailureOr<llvm::StringRef> GetPlatformNameForDeviceId(
    mlir::Operation* op, int device_id,
    absl::Span<const std::string> platform_names ABSL_ATTRIBUTE_LIFETIME_BOUND);

// Sets the strict memory reservation in the compile options.
absl::Status SetStrictMemoryReservation(absl::string_view program_name,
                                        int64_t device_memory,
                                        xla::CompileOptions& options);

// Expands a vector of platform names from short format (e.g., tpu:2,host:2) to
// long format with an entry for each platform instance.
absl::StatusOr<std::vector<std::string>> ExpandPlatformNames(
    const mlir::Pass::ListOption<std::string>& platform_names);

// Extracts the XlaCompileOptions overrides for the given atom program module.
// Returns nullptr if no overrides are found.
absl::StatusOr<XlaCompileOptions*> GetModuleXlaCompileOverrides(
    mlir::StringAttr compile_options_key,
    std::shared_ptr<
        absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
        compile_options_overrides);

// Extracts the xla::CompileOptions for the given atom program module. Returns
// std::nullopt if no overrides are found.
absl::StatusOr<std::optional<xla::CompileOptions>> GetModuleCompileOverrides(
    mlir::StringAttr compile_options_key,
    std::shared_ptr<
        absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
        compile_options_overrides);

// Returns the default compile options for the given CallOp.
xla::CompileOptions GetDefaultCompileOptions(CallOp call_op,
                                             bool enable_sharding_propagation,
                                             bool enable_parameter_tupling);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_COMPILATION_UTILS_H_

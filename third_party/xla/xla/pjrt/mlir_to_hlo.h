/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_PJRT_MLIR_TO_HLO_H_
#define XLA_PJRT_MLIR_TO_HLO_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "xla/client/xla_computation.h"

namespace xla {

// Converts an MHLO/CHLO module string to an mlir::Module.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseMlirModuleString(
    absl::string_view mlir_module_str, mlir::MLIRContext& context);

// Converts an CHLO/MHLO module to XLA HLO.
absl::Status MlirToXlaComputation(mlir::ModuleOp module,
                                  XlaComputation& xla_computation,
                                  bool use_tuple_args, bool return_tuple);

// Converts an MHLO/CHLO module string to an XLA computation.
absl::Status ParseMlirModuleStringAndConvertToXlaComputation(
    absl::string_view mlir_module_str, XlaComputation& xla_computation,
    bool use_tuple_args, bool return_tuple);

// Serialize using MLIR Bytecode Format which does not guarantee forward or
// backward compatiblity of the dialects used. If passing StableHLO with forward
// or backward compatibility requirements, use SerializeUsingVersionedStablehlo.
absl::StatusOr<std::string> SerializeUsingNativeBytecode(
    mlir::ModuleOp mlir_module, std::optional<int64_t> plugin_version);

// Serializes an MLIR module to a portable artifact with forward and backward
// compatibility. Supports modules using StableHLO/MHLO/CHLO/Func dialects.
// Target parameter is a StableHLO version string ("0.9.0") which can be used
// for forward compatibility to specify the target downgrade version.
// Most commonly should use:
//   `mlir::stablehlo::getCurrentVersion()` for backward compat but not forward.
//   `mlir::stablehlo::getMinimumVersion()` for maximum forward compatibility.
// Ideally should be the `mlir::stablehlo::getCurrentVersion()` of the plugin.
// If program contains dialects that aren't supposed in StableHLO portable
// artifacts, use SerializeUsingNativeBytecode.
absl::StatusOr<std::string> SerializeUsingVersionedStablehlo(
    mlir::ModuleOp mlir_module, absl::string_view target, bool inplace = false);

// Given a module that might be a portable artifact, deserialize and upgrade it
// back to StableHLO.
// If module is not a portable artifact, this method is identity. Only fails
// on portable artifacts that are outside of the compatibility window.
// `ParseMlirModuleString` uses this method, and should be preferred to directly
// calling `UpgradeVersionedStablehlo` where possible.
absl::Status UpgradeVersionedStablehlo(mlir::ModuleOp mlir_module);

}  // namespace xla

#endif  // XLA_PJRT_MLIR_TO_HLO_H_

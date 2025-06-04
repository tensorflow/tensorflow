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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "xla/hlo/builder/xla_computation.h"

namespace xla {

// Registers all MLIR dialects that are necessary for using MLIR HLO modules.
void RegisterAllHloDialects(mlir::DialectRegistry& registry);

// Converts an MHLO/CHLO module string to an mlir::Module.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseMlirModuleString(
    absl::string_view mlir_module_str, mlir::MLIRContext& context);

// Converts an CHLO/MHLO module to XLA HLO.
// TODO(b/345414638): Delete `use_shardy` when we move Shardy as the first pass
// in the XLA pipeline.
absl::Status MlirToXlaComputation(mlir::ModuleOp module,
                                  XlaComputation& xla_computation,
                                  bool use_tuple_args, bool return_tuple,
                                  bool use_shardy);

// Converts an MHLO/CHLO module string to an XLA computation.
absl::Status ParseMlirModuleStringAndConvertToXlaComputation(
    absl::string_view mlir_module_str, XlaComputation& xla_computation,
    bool use_tuple_args, bool return_tuple);

// Export a StableHLO + Shardy module into a pure StableHLO module, to prepare
// for a round trip to HLO, such that the Shardy ops and attributes are
// preserved when going back to MLIR for Shardy propagation.
absl::Status ExportShardyForHloRoundTrip(mlir::ModuleOp module);

// Export a StableHLO + Shardy module into a pure StableHLO module, targeting
// the GSPMD partitioner. No round tripping back to MLIR is needed, since GSPMD
// expected HLO, unlike `ExportShardyForHloRoundTrip`.
// This function should only be used when Shardy is enabled in JAX, but JAX
// export loaded a GSPMD checkpoint.
// TODO(b/420837831): delete this once we don't fall back to GSPMD.
absl::Status ExportShardyForGSPMD(mlir::ModuleOp module);

// Returns a version of StableHLO ~12w old, for forward compatibility with PJRT
// plugins on a quarterly update cycle.
std::string GetDefaultStablehloVersion(
    std::optional<int64_t> plugin_version = std::nullopt);

// Serialize using MLIR Bytecode Format which does not guarantee forward or
// backward compatiblity of the dialects used. If passing StableHLO with forward
// or backward compatibility requirements, use SerializeUsingVersionedStablehlo.
//
// VHLO support was added in PJRT plugin version 41.
//   For plugin_version < 41, returns `SerializeUsingNativeBytecode`.
//   For plugin_version >= 41, returns `SerializeUsingVersionedStablehlo`.
absl::StatusOr<std::string> Serialize(mlir::ModuleOp mlir_module,
                                      absl::string_view target,
                                      bool inplace = false);

// Serializes an MLIR module to a portable artifact with forward and backward
// compatibility. Supports modules using StableHLO/MHLO/CHLO/Func dialects.
// The `requested_target` parameter is a StableHLO version string ("0.9.0")
// which can be used for forward compatibility to specify the target downgrade
// version. Most commonly should use:
//   `mlir::stablehlo::getCurrentVersion()` for backward compat but not forward.
//   `mlir::stablehlo::getMinimumVersion()` for maximum forward compatibility.
// In PJRT, the `requested_target` should be the current version of the PJRT
// plugin. Serialize will use `min(framework_version, plugin_version)` to
// serialize. If program contains dialects that aren't supported in StableHLO
// portable artifacts, use SerializeUsingNativeBytecode.
absl::StatusOr<std::string> SerializeUsingVersionedStablehlo(
    mlir::ModuleOp mlir_module, absl::string_view requested_target,
    bool inplace = false);

// Given a module that might be a portable artifact, deserialize and upgrade it
// back to StableHLO.
// If module is not a portable artifact, this method is identity. Only fails
// on portable artifacts that are outside of the compatibility window.
// `ParseMlirModuleString` uses this method, and should be preferred to directly
// calling `UpgradeVersionedStablehlo` where possible.
absl::Status UpgradeVersionedStablehlo(mlir::ModuleOp mlir_module);

}  // namespace xla

#endif  // XLA_PJRT_MLIR_TO_HLO_H_

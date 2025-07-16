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

#ifndef XLA_HLO_TRANSLATE_STABLEHLO_H_
#define XLA_HLO_TRANSLATE_STABLEHLO_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"

namespace xla {

// Registers dialects necessary for converting MLIR to HLO.
void RegisterMlirToHloDependentDialects(mlir::DialectRegistry& registry);

// Convert HloModule to StableHLO module.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloToStablehlo(
    mlir::MLIRContext& ctx, const xla::HloModule* hlo_module);

// Convert HloModuleProto to StableHLO module.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloToStablehlo(
    mlir::MLIRContext& ctx, const xla::HloModuleProto* hlo_module);

// Convert HloModuleProto to StableHLO module.
// DO NOT USE THIS METHOD WITHOUT A GOOD REASON. Prefer ConvertHloToStablehlo.
// Currently it exists to satisfy TF2XLA compilation APIs where certain behavior
// is dependent on not important all computations. In general we want a single
// conversion path for all HLO, and are working to obsolete this method.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
ConvertHloToStablehloWithOptions(mlir::MLIRContext& ctx,
                                 const xla::HloModuleProto* hlo_module,
                                 bool import_all_computations);

// Convert StableHLO module to HloModule.
absl::StatusOr<std::unique_ptr<xla::HloModule>> ConvertStablehloToHlo(
    mlir::ModuleOp module);

// Convert StableHLO module to HloModuleProto.
absl::Status ConvertStablehloToHloProto(mlir::ModuleOp module,
                                        xla::HloProto* hlo_proto);

// Convert StableHLO module to HloModule.
// DO NOT USE THIS METHOD WITHOUT A GOOD REASON. Prefer ConvertStablehloToHlo.
// Currently it exists to satisfy the PJRT compilation APIs where a framework
// may specify that a computation should use tuples. This is seldom used, the
// main exception being computations with 2k+ parameters targeting TPU.
absl::StatusOr<std::unique_ptr<xla::HloModule>>
ConvertStablehloToHloWithOptions(mlir::ModuleOp module, bool use_tuple_args,
                                 bool return_tuple);

// Convert StableHLO module to HloModuleProto.
// Some platforms run out of memory when the argument list is too long.
// This API wraps the arguments in a tuple (if use_tuple_args = true)
// as a workaround. The long-term solution is to add an HLO pass to do this.
// In general, prefer the other ConvertStablehloToHloProto method.
absl::Status ConvertStablehloWithManyArgsToHloProto(
    mlir::ModuleOp module, xla::HloProto* hlo_proto,
    bool use_tuple_args = false);

}  // namespace xla

#endif  // XLA_HLO_TRANSLATE_STABLEHLO_H_

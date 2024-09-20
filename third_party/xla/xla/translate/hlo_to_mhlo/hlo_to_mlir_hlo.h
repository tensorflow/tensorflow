/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_TRANSLATE_HLO_TO_MHLO_HLO_TO_MLIR_HLO_H_
#define XLA_TRANSLATE_HLO_TO_MHLO_HLO_TO_MLIR_HLO_H_

#include <stdbool.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class ModuleOp;
}  // namespace mlir

namespace xla {
class HloModule;
class HloModuleProto;

// Converts an HLO module proto to a MLIR module in HLO dialect.
//
// If `import_all_computation` is set to true, imports all computations
// irrespective if transitively called from entry computation.
//
// If `flatten_computation_args_result` is set to true, flattens all tuple
// arguments and result of every computation when importing them as func ops.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloToMlirHlo(
    mlir::MLIRContext& ctx, xla::HloModuleProto const* hlo_module,
    bool import_all_computations = false,
    bool flatten_computation_args_result = false);

absl::Status ConvertHloToMlirHlo(mlir::ModuleOp module,
                                 xla::HloModuleProto const* hlo_module,
                                 bool import_all_computations = false,
                                 bool flatten_computation_args_result = false);

// Converts an HLO module to a MLIR module in HLO dialect.
//
// If `import_all_computation` is set to true, imports all computations
// irrespective if transitively called from entry computation.
//
// If `flatten_computation_args_result` is set to true, flattens all tuple
// arguments and result of every computation when importing them as func ops.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloToMlirHlo(
    mlir::MLIRContext& ctx, const xla::HloModule* hlo_module,
    bool import_all_computations = false,
    bool flatten_computation_args_result = false);

absl::Status ConvertHloToMlirHlo(mlir::ModuleOp module,
                                 const xla::HloModule* hlo_module,
                                 bool import_all_computations = false,
                                 bool flatten_computation_args_result = false);

}  // namespace xla

#endif  // XLA_TRANSLATE_HLO_TO_MHLO_HLO_TO_MLIR_HLO_H_

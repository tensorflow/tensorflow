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

#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_module_importer.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloToMlirHlo(
    mlir::MLIRContext& ctx, xla::HloModuleProto const* hlo_module,
    bool import_all_computations, bool flatten_computation_args_result,
    bool emit_stablehlo) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(&ctx));
  TF_RETURN_IF_ERROR(
      ConvertHloToMlirHlo(*module, hlo_module, import_all_computations,
                          flatten_computation_args_result, emit_stablehlo));
  return module;
}

absl::Status ConvertHloToMlirHlo(mlir::ModuleOp module,
                                 xla::HloModuleProto const* hlo_module_proto,
                                 bool import_all_computation,
                                 bool flatten_computation_args_result,
                                 bool emit_stablehlo) {
  mlir::BaseScopedDiagnosticHandler diag_handler(module.getContext());
  return HloModuleImporter(module, import_all_computation,
                           flatten_computation_args_result, emit_stablehlo)
      .Import(*hlo_module_proto);
}

absl::Status ConvertHloToMlirHlo(mlir::ModuleOp module,
                                 const xla::HloModule* hlo_module,
                                 bool import_all_computation,
                                 bool flatten_computation_args_result,
                                 bool emit_stablehlo) {
  mlir::BaseScopedDiagnosticHandler diag_handler(module.getContext());
  return HloModuleImporter(module, import_all_computation,
                           flatten_computation_args_result, emit_stablehlo)
      .Import(*hlo_module);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloToMlirHlo(
    mlir::MLIRContext& ctx, const xla::HloModule* hlo_module,
    bool import_all_computations, bool flatten_computation_args_result,
    bool emit_stablehlo) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(&ctx));
  TF_RETURN_IF_ERROR(
      ConvertHloToMlirHlo(*module, hlo_module, import_all_computations,
                          flatten_computation_args_result, emit_stablehlo));
  return module;
}

}  // namespace xla

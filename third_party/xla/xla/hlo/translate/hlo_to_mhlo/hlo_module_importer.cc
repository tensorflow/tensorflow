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

#include "xla/hlo/translate/hlo_to_mhlo/hlo_module_importer.h"

#include <memory>

#include "absl/status/status.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_function_importer.h"
#include "xla/hlo/translate/hlo_to_mhlo/module_attributes_importer.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

#define DEBUG_TYPE "xla-translate"

namespace xla {

HloModuleImporter::HloModuleImporter(mlir::ModuleOp module,
                                     bool import_all_computation,
                                     bool flatten_computation_args_result,
                                     bool emit_stablehlo)
    : import_all_computation_(import_all_computation),
      flatten_computation_args_result_(flatten_computation_args_result),
      symbol_table_(module),
      emit_stablehlo_(emit_stablehlo),
      builder_(module.getContext()) {
  module.getContext()->loadDialect<mlir::arith::ArithDialect>();
  module.getContext()->loadDialect<mlir::func::FuncDialect>();
  module.getContext()->loadDialect<mlir::mhlo::MhloDialect>();
  module.getContext()->loadDialect<mlir::stablehlo::StablehloDialect>();
  module.getContext()->loadDialect<mlir::quant::QuantDialect>();
}

namespace {
absl::Status ConvertToMhlo(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  if (failed(pm.run(module))) {
    return absl::InternalError("Failed to convert to MHLO");
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status HloModuleImporter::Import(const HloModule& hlo_module) {
  auto module = llvm::cast<mlir::ModuleOp>(symbol_table_.getOp());
  module.setName(hlo_module.name());

  ImportCrossProgramPrefetches(hlo_module, module,
                               flatten_computation_args_result_, builder_);
  ImportFrontendAttributes(hlo_module, module, builder_);
  ImportInputOutputAlias(hlo_module, module, builder_);
  ImportIsDynamic(hlo_module, module, builder_);
  ImportNumPartitions(hlo_module, module, builder_);
  ImportNumReplicas(hlo_module, module, builder_);
  ImportSpmdOutputSharding(hlo_module, module, builder_);
  ImportSpmdParametersShardings(hlo_module, module,
                                flatten_computation_args_result_, builder_);
  ImportUseAutoSpmdPartitioning(hlo_module, module, builder_);

  if (!import_all_computation_) {
    // Only import the entry computation, any reachable one will be imported
    // unless turned into a region operation.
    TF_RETURN_IF_ERROR(HloFunctionImporter::ImportAsFunc(
                           *hlo_module.entry_computation(), symbol_table_,
                           &function_map_, &builder_,
                           /*is_main*/ true, flatten_computation_args_result_)
                           .status());

    // Convert all ops to MHLO
    LLVM_DEBUG(llvm::dbgs() << "Emit StableHLO: " << emit_stablehlo_ << "\n");
    if (!emit_stablehlo_) {
      TF_RETURN_IF_ERROR(ConvertToMhlo(module));
    }
    return absl::OkStatus();
  }

  auto* module_entry_computation = hlo_module.entry_computation();
  for (const auto* computation : hlo_module.computations()) {
    TF_RETURN_IF_ERROR(HloFunctionImporter::ImportAsFunc(
                           *computation, symbol_table_, &function_map_,
                           &builder_,
                           /*is_main*/ computation == module_entry_computation,
                           flatten_computation_args_result_)
                           .status());
  }

  ImportEntryComputationLayoutAndTiles(
      hlo_module, module, flatten_computation_args_result_, builder_);
  TF_RETURN_IF_ERROR(ImportLayoutModes(
      hlo_module, module, flatten_computation_args_result_, builder_));

  // Convert all ops to MHLO
  LLVM_DEBUG(llvm::dbgs() << "Emit StableHLO: " << emit_stablehlo_ << "\n");
  if (!emit_stablehlo_) {
    TF_RETURN_IF_ERROR(ConvertToMhlo(module));
  }
  return absl::OkStatus();
}

absl::Status HloModuleImporter::Import(const HloModuleProto& module_proto) {
  DebugOptions debug_options;
  TF_ASSIGN_OR_RETURN(
      auto module_config,
      HloModule::CreateModuleConfigFromProto(module_proto, debug_options));
  TF_ASSIGN_OR_RETURN(auto module,
                      HloModule::CreateFromProto(module_proto, module_config));

  return Import(*module);
}

}  // namespace xla

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

#include "xla/hlo/translate/stablehlo.h"

#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_module_importer.h"
#include "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/hlo/translate/mhlo_to_hlo/module_attributes_exporter.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

namespace {
absl::Status MhloToStablehlo(mlir::ModuleOp module) {
  auto context = module.getContext();
  mlir::PassManager pm(context);
  mlir::BaseScopedDiagnosticHandler diag_handler(context);
  mlir::mhlo::HloLegalizeToStablehloPassOptions options;
  options.allow_xla_features_ = true;
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass(options));
  if (failed(pm.run(module))) {
    return diag_handler.ConsumeStatus();
  }
  return absl::OkStatus();
}

// TODO(b/385393967) Separate createCanonicalizerPass from StableHLO -> HLO
// Translation
absl::Status StablehloToMhlo(mlir::ModuleOp module, bool run_canonicalizer) {
  mlir::MLIRContext* context = module->getContext();
  mlir::PassManager pm(context);
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createChloLegalizeToHloPass());
  if (run_canonicalizer) {
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  }
  // In order to export to XLA, we must sink constants to control flow
  // regions, since XLA uses functional control flow.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo_ext::createSinkConstantsToControlFlowPass());

  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);
  if (failed(pm.run(module))) {
    VLOG(1) << "MHLO->HLO lowering passes failed. Module:\n" << module;
    return diagnostic_handler.ConsumeStatus();
  }

  VLOG(5) << "MHLO module after lowering, before HLO import, Module:\n"
          << module;

  return absl::OkStatus();
}

}  // namespace

void RegisterMlirToHloDependentDialects(mlir::DialectRegistry& registry) {
  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  registry.insert<mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                  mlir::shape::ShapeDialect, mlir::ub::UBDialect>();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloToStablehlo(
    mlir::MLIRContext& ctx, const xla::HloModule* hlo_module) {
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(&ctx));
  TF_RETURN_IF_ERROR(HloModuleImporter(mlir_module.get(),
                                       /*import_all_computation=*/true,
                                       /*flatten_computation_args_result=*/true)
                         .Import(*hlo_module));
  TF_RETURN_IF_ERROR(MhloToStablehlo(mlir_module.get()));
  return std::move(mlir_module);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertHloToStablehlo(
    mlir::MLIRContext& ctx, const xla::HloModuleProto* hlo_module_proto) {
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(&ctx));
  TF_RETURN_IF_ERROR(HloModuleImporter(mlir_module.get(),
                                       /*import_all_computation=*/true,
                                       /*flatten_computation_args_result=*/true)
                         .Import(*hlo_module_proto));
  TF_RETURN_IF_ERROR(MhloToStablehlo(mlir_module.get()));
  return std::move(mlir_module);
}

namespace {
absl::Status ConvertStablehloToHloProtoInternal(mlir::ModuleOp module,
                                                xla::HloProto* hlo_proto,
                                                bool use_tuple_args,
                                                bool return_tuple,
                                                bool run_canonicalizer) {
  if (!module) return absl::InvalidArgumentError("Module is null");

  TF_RETURN_IF_ERROR(StablehloToMhlo(module, run_canonicalizer));

  mlir::MlirToHloConversionOptions options;
  options.return_tuple = return_tuple;
  options.use_tuple_args = use_tuple_args;
  TF_RETURN_IF_ERROR(mlir::ConvertMlirHloToHlo(module, hlo_proto, options));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<xla::HloModule>> ConvertStablehloToHloInternal(
    mlir::ModuleOp module, bool use_tuple_args, bool return_tuple) {
  xla::HloProto hlo_proto;
  TF_RETURN_IF_ERROR(ConvertStablehloToHloProtoInternal(
      module, &hlo_proto, use_tuple_args, return_tuple,
      /*run_canonicalizer=*/true));

  // Create default config and modify config with values stored
  // in MLIR module attributes
  const xla::HloModuleProto& module_proto = hlo_proto.hlo_module();
  auto config = xla::HloModule::CreateModuleConfigFromProto(
      module_proto, xla::GetDebugOptionsFromFlags());
  if (!config.ok()) {
    return config.status();
  }
  mlir::mhlo::ExportHloModuleConfig(config.value(), module);

  return xla::HloModule::CreateFromProto(module_proto, config.value());
}

}  // namespace

absl::StatusOr<std::unique_ptr<xla::HloModule>> ConvertStablehloToHlo(
    mlir::ModuleOp module) {
  return ConvertStablehloToHloInternal(module,
                                       /*use_tuple_args=*/false,
                                       /*return_tuple=*/false);
}

absl::StatusOr<std::unique_ptr<xla::HloModule>>
ConvertStablehloToHloWithOptions(mlir::ModuleOp module, bool use_tuple_args,
                                 bool return_tuple) {
  return ConvertStablehloToHloInternal(module, use_tuple_args, return_tuple);
}

absl::Status ConvertStablehloToHloProto(mlir::ModuleOp module,
                                        xla::HloProto* hlo_proto) {
  if (!module) return absl::InvalidArgumentError("Module is null");

  TF_RETURN_IF_ERROR(StablehloToMhlo(module, /*run_canonicalizer=*/true));
  return ConvertStablehloToHloProtoInternal(module, hlo_proto,
                                            /*use_tuple_args=*/false,
                                            /*return_tuple=*/false,
                                            /*run_canonicalizer=*/true);
}

absl::Status ConvertStablehloWithManyArgsToHloProto(mlir::ModuleOp module,
                                                    xla::HloProto* hlo_proto,
                                                    bool use_tuple_args) {
  if (!module) return absl::InvalidArgumentError("Module is null");

  // TODO: Why are we removing attributes.
  module->removeAttr("mhlo.xla_entry_computation_parameter_layouts");
  module->removeAttr("mhlo.xla_entry_computation_parameter_tiles");
  return ConvertStablehloToHloProtoInternal(module, hlo_proto, use_tuple_args,
                                            /*return_tuple=*/false,
                                            /*run_canonicalizer=*/false);
}

}  // namespace xla

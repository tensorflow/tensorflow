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

#include "xla/pjrt/mlir_to_hlo.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/api/PortableApi.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/transforms/Passes.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_export.h"
#include "xla/service/spmd/shardy/utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace {
using mlir::mhlo::ChloLegalizeToHighLevelMhloPassOptions;
}

void RegisterAllHloDialects(mlir::DialectRegistry& registry) {
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::ml_program::MLProgramDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  mlir::func::registerAllExtensions(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::sdy::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
}

absl::Status MlirToXlaComputation(
    mlir::ModuleOp module, XlaComputation& xla_computation, bool use_tuple_args,
    bool return_tuple, ExecutableBuildOptions* exec_build_options,
    const ChloLegalizeToHighLevelMhloPassOptions& chlo_opts) {
  mlir::MLIRContext* context = module->getContext();
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);
  {
    mlir::PassManager pm(context);

    // TODO(b/420837831): Remove this once we don't need to fall back to GSPMD.
    if (exec_build_options && exec_build_options->use_shardy_partitioner() &&
        xla::sdy::hasGspmdAttrsOrOps(module)) {
      LOG(WARNING)
          << "Module has GSPMD attrs or ops, but Shardy is enabled. Disabling "
             "Shardy and falling back to using GSPMD propagation.";
      exec_build_options->set_use_shardy_partitioner(false);
      TF_RETURN_IF_ERROR(ExportShardyForGSPMD(module));
    }

    // Export a StableHLO + Shardy module into a pure StableHLO module, to
    // prepare for a round trip to HLO, such that the Shardy ops and attributes
    // are preserved when going back to MLIR for Shardy propagation. This is a
    // no-op if the module is already pure StableHLO.
    // NOTE: we don't use `use_shardy` because it isn't guaranteed to be true if
    // the module has Shardy artifacts.
    bool enable_hlo_sharding_v3 =
        exec_build_options && exec_build_options->has_debug_options() &&
        exec_build_options->debug_options().xla_enable_hlo_sharding_v3();
    xla::sdy::addSdyRoundTripExportPipeline(pm, /*keepMeshesInlined=*/false,
                                            enable_hlo_sharding_v3);

    // CHLO -> MHLO for high level ops (TopK, Erf, RaggedDot, etc.)
    // CHLO -> StableHLO otherwise
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::stablehlo_ext::createChloRecomposeOpsPass());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createChloLegalizeToHighLevelMhloPass(chlo_opts));
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::stablehlo::createChloLegalizeToStablehloPass());

    // Expand stablehlo complex math functions such as log_plus_one, etc.
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::stablehlo::createStablehloComplexMathExpanderPass());

    // In order to export to XLA, we must sink constants to control flow
    // regions, since XLA uses functional control flow.
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::stablehlo_ext::createSinkConstantsToControlFlowPass());

    if (failed(pm.run(module))) {
      VLOG(1) << "MHLO->HLO lowering passes failed.";
      return diagnostic_handler.ConsumeStatus();
    }

    VLOG(5) << "MHLO module after lowering, before HLO import ";
    if (VLOG_IS_ON(5)) {
      module->dump();
    }
  }

  // TODO(b/345414638): Delete when we move Shardy as the first pass in the
  // XLA pipeline.
  if (use_tuple_args && exec_build_options &&
      exec_build_options->use_shardy_partitioner()) {
    // Shardy can't handle tuple args when round-tripping. So delay using
    // tuples until after Shardy is run.
    sdy::setFrontendAttribute(module, sdy::kUseTupleArgs,
                              mlir::StringAttr::get(context, "t"));
    use_tuple_args = false;
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      xla::ConvertStablehloToHloWithOptions(
                          module, use_tuple_args, return_tuple));

  xla_computation = XlaComputation(hlo_module->ToProto());
  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseMlirModuleString(
    absl::string_view mlir_module_str, mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  RegisterAllHloDialects(registry);
  context.appendDialectRegistry(registry);

  mlir::BaseScopedDiagnosticHandler diagnostic_handler(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(mlir_module_str.data(), mlir_module_str.size()),
          mlir::ParserConfig{&context});
  if (!module) {
    mlir::emitError(mlir::UnknownLoc::get(&context))
        << "Failed to parse using StableHLO v"
        << mlir::vhlo::Version::getCurrentVersion() << ", "
        << "this could indicate forward incompatibility, >12w old "
           "unsupported plugin, or a portable artifact that needs to be "
           "further downgraded.";
    return diagnostic_handler.ConsumeStatus();
  }

  TF_RETURN_IF_ERROR(UpgradeVersionedStablehlo(*module));
  return std::move(module);
}

absl::Status ParseMlirModuleStringAndConvertToXlaComputation(
    absl::string_view mlir_module_str, XlaComputation& xla_computation,
    bool use_tuple_args, bool return_tuple) {
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      xla::ParseMlirModuleString(mlir_module_str, context));
  return xla::MlirToXlaComputation(*module, xla_computation, use_tuple_args,
                                   return_tuple,
                                   /*exec_build_options=*/nullptr);
}

absl::Status ExportShardyForHloRoundTrip(mlir::ModuleOp module) {
  mlir::MLIRContext* context = module.getContext();
  mlir::PassManager pm(context);
  xla::sdy::addSdyRoundTripExportPipeline(pm);
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);
  if (!mlir::succeeded(pm.run(module))) {
    const absl::Status status = diagnostic_handler.ConsumeStatus();
    return absl::InvalidArgumentError(
        absl::StrCat("Shardy export for HLO round trip failed;\n\nDetailed "
                     "error from MLIR: ",
                     status.message()));
  }
  return absl::OkStatus();
}

absl::Status ExportShardyForGSPMD(mlir::ModuleOp module) {
  if (!xla::sdy::hasShardyMesh(module)) {
    return absl::OkStatus();
  }
  mlir::MLIRContext* context = module.getContext();
  mlir::PassManager pm(context);
  // Export sharding constraints to StableHLO @Sharding custom calls for GSPMD
  // to handle.
  xla::sdy::StablehloExportPipelineOptions options;
  options.keepHloShardingConstraints = true;
  options.addMissingShardingToControlFlow = false;
  xla::sdy::addStablehloExportPipeline(pm, options);
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);
  if (!mlir::succeeded(pm.run(module))) {
    const absl::Status status = diagnostic_handler.ConsumeStatus();
    return absl::InvalidArgumentError(
        absl::StrCat("Shardy export for GSPMD failed;\n\nDetailed "
                     "error from MLIR: ",
                     status.message()));
  }
  return absl::OkStatus();
}

std::optional<mlir::StringRef> FindPotentiallyUnstableDialects(
    mlir::ModuleOp module) {
  std::optional<mlir::StringRef> unstable_dialect = std::nullopt;
  module->walk([&](mlir::Operation* op) {
    if (!llvm::isa<mlir::ModuleOp>(op) &&
        !llvm::isa<mlir::stablehlo::StablehloDialect, mlir::func::FuncDialect,
                   mlir::chlo::ChloDialect, mlir::sdy::SdyDialect>(
            op->getDialect())) {
      unstable_dialect = op->getDialect()->getNamespace();
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return unstable_dialect;
}

// Helper method to convert a mlir::FailureOr<T> to an absl::StatusOr<T>.
template <typename T>
absl::StatusOr<T> ExpectSuccess(mlir::FailureOr<T> result, std::string msg) {
  if (mlir::failed(result)) {
    return absl::InvalidArgumentError(msg);
  }
  return result.value();
}

absl::StatusOr<std::string> SerializeUsingVersionedStablehlo(
    mlir::ModuleOp mlir_module, absl::string_view requested_target,
    bool inplace, bool allow_mixed_serialization) {
  mlir::MLIRContext* context = mlir_module->getContext();
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);

  // Usually the plugin is older than the framework, but occasionally a plugin's
  // nightly build will use the latest public release of a framework. Serialize
  // using the framework's version in these cases.
  TF_ASSIGN_OR_RETURN(
      std::string target,
      ExpectSuccess(mlir::stablehlo::getSmallerVersion(
                        requested_target, mlir::stablehlo::getCurrentVersion()),
                    "Invalid StableHLO target version requested."));

  // Legalize CHLO -> [StableHLO+Shape] -> StableHLO
  // Preserve higher-level ops with XLA support. To be replaced by composites.
  mlir::PassManager pm(context);
  // Expand stablehlo complex math functions such as log_plus_one, etc.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloComplexMathExpanderPass());

  // Determine whether we need to export non-StableHLO ops.
  // - For shardy, convert Shardy ops to StableHLO ops, and stringify the Shardy
  //   attributes.
  if (!allow_mixed_serialization) {
    xla::sdy::addSdyRoundTripExportPipeline(pm);
  }
  pm.addPass(mlir::stablehlo_ext::createChloPreserveHighLevelOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createChloLegalizeToStablehloPass());
  pm.addPass(
      mlir::stablehlo::createStablehloCompatibilityExpanderPass({target}));
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createChloLegalizeToStablehloPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createShapeLegalizeToStablehloPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());  // not required
  if (!mlir::succeeded(pm.run(mlir_module))) {
    const absl::Status status = diagnostic_handler.ConsumeStatus();
    return absl::InvalidArgumentError(absl::StrCat(
        "CHLO => [StableHLO+Shape] => StableHLO failed;\n\nDetailed "
        "error from MLIR: ",
        status.message()));
  }

  // Avoid mutating the original module if it will be reused elsewhere
  mlir::OwningOpRef<mlir::ModuleOp> cloned;
  if (!inplace) {
    cloned = mlir_module.clone();
    mlir_module = *cloned;
  }

  // Serialize portable artifact
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  // TODO(gleasonk): make `allowOtherDialects` an allow-list of dialects instead
  // of a boolean.
  if (mlir::failed(mlir::stablehlo::serializePortableArtifact(
          mlir_module, target, os,
          /*allowOtherDialects=*/allow_mixed_serialization))) {
    const absl::Status status = diagnostic_handler.ConsumeStatus();
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to serialize StableHLO to plugin version ", target,
                     ";\n\nDetailed error from MLIR: ", status.message()));
  }
  return buffer;
}

// TODO (b/344930098): Delete this method when mixed serialization is supported
// by all plugins in the 12w compat window (Sep 2025, StableHLO v1.11.0).
absl::StatusOr<std::string> LegacySerialize(mlir::ModuleOp module,
                                            mlir::vhlo::Version target,
                                            bool inplace) {
  if (!FindPotentiallyUnstableDialects(module).has_value()) {
    // No unstable dialects, still need to convert SDY to custom calls.
    return SerializeUsingVersionedStablehlo(
        module, target.toString(), inplace,
        /*allow_mixed_serialization=*/false);
  }

  // Use native bytecode, no stability.
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  mlir::BytecodeWriterConfig config;
  auto version = target.getBytecodeVersion();
  if (mlir::failed(version)) {
    return absl::InvalidArgumentError("Failed to get bytecode version");
  }
  config.setDesiredBytecodeVersion(version.value());
  if (mlir::failed(mlir::writeBytecodeToFile(module, os, config))) {
    return absl::InvalidArgumentError("mlir::writeBytecodeToFile failed");
  }
  return bytecode;
}

absl::Status UpgradeVersionedStablehlo(mlir::ModuleOp mlir_module) {
  // Upgrade if VHLO
  mlir::PassManager pm(mlir_module->getContext());
  mlir::stablehlo::createStablehloDeserializePipeline(pm);
  if (!mlir::succeeded(pm.run(mlir_module)))
    return xla::InvalidArgument("Failed to upgrade versioned StableHLO.");
  return absl::OkStatus();
}

std::string GetDefaultStablehloVersion() {
  // This version must be >=12w old.
  return mlir::vhlo::Version::fromCompatibilityRequirement(
             mlir::vhlo::Version::CompatibilityRequirement::WEEK_12)
      .toString();
}

absl::StatusOr<std::string> Serialize(mlir::ModuleOp module,
                                      absl::string_view target,
                                      bool inplace) {
  // Current PJRT users expect 12 weeks forward compat, VHLO provides this
  // compat.
  TF_ASSIGN_OR_RETURN(
      auto version,
      ExpectSuccess(mlir::vhlo::Version::fromString(target),
                    "Invalid StableHLO target version requested."));

  // TODO (b/344930098): Once v1.11.0 is >=12w old, only use mixed serialization
  // ~Sep 2025, can delete legacy path.
  bool supports_mixed_serialization = mlir::vhlo::Version(1, 11, 0) <= version;
  if (!supports_mixed_serialization) {
    return LegacySerialize(module, version, inplace);
  }

  return SerializeUsingVersionedStablehlo(module, target, inplace,
                                          /*allow_mixed_serialization=*/true);
}

}  // namespace xla

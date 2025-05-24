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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
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
#include "xla/debug_options_flags.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/service/spmd/shardy/utils.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {

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

absl::Status MlirToXlaComputation(mlir::ModuleOp module,
                                  XlaComputation& xla_computation,
                                  bool use_tuple_args, bool return_tuple,
                                  bool use_shardy) {
  mlir::MLIRContext* context = module->getContext();
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);
  {
    mlir::PassManager pm(context);

    // CHLO -> MHLO for high level ops (TopK, Erf, RaggedDot, etc.)
    // CHLO -> StableHLO otherwise
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::stablehlo_ext::createChloRecomposeOpsPass());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createChloLegalizeToHighLevelMhloPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::stablehlo::createChloLegalizeToStablehloPass());

    // Expand stablehlo complex math functions such as log_plus_one, etc.
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::stablehlo::createStablehloComplexMathExpanderPass());

    // In order to export to XLA, we must sink constants to control flow
    // regions, since XLA uses functional control flow.
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::stablehlo_ext::createSinkConstantsToControlFlowPass());

    // Export an StableHLO + Shardy module into a pure StableHLO module, to
    // prepare for a round trip to HLO, such that the Shardy ops and attributes
    // are preserved when going back to MLIR for Shardy propagation. This is a
    // no-op if the module is already pure StableHLO.
    // NOTE: we don't use `use_shardy` because it isn't guaranteed to be true if
    // the module has Shardy artifacts.
    xla::sdy::addSdyRoundTripExportPipeline(pm);

    if (failed(pm.run(module))) {
      VLOG(1) << "MHLO->HLO lowering passes failed.";
      module->dump();
      return diagnostic_handler.ConsumeStatus();
    }

    VLOG(5) << "MHLO module after lowering, before HLO import ";
    if (VLOG_IS_ON(5)) {
      module->dump();
    }
  }

  // TODO(b/345414638): Delete when we move Shardy as the first pass in the
  // XLA pipeline.
  if (use_tuple_args && use_shardy) {
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
                                   return_tuple, /*use_shardy=*/false);
}

absl::Status ExportShardyForHloRoundTrip(mlir::ModuleOp module) {
  mlir::MLIRContext* context = module.getContext();
  mlir::PassManager pm(context);
  xla::sdy::addSdyRoundTripExportPipeline(pm);
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);
  if (!mlir::succeeded(pm.run(module))) {
    const absl::Status status = diagnostic_handler.ConsumeStatus();
    return absl::InvalidArgumentError(
        absl::StrCat("Shardy export failed;\n\nDetailed "
                     "error from MLIR: ",
                     status.message()));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> SerializeUsingNativeBytecode(
    mlir::ModuleOp module) {
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  mlir::BytecodeWriterConfig config;
  // Pin bytecode version to 1 until transition to stable.
  // TODO: b/285913864 - Remove post enabling frameworks to set it.
  config.setDesiredBytecodeVersion(1);
  // In
  // https://github.com/google/jax/commit/184e3a88004680dbf34328b05c5fc0d869cc4a93,
  // fields on some ops were changed to use Dense{Bool,I64}ArrayAttr instead of
  // I64DenseElementsAttr (DenseIntElementsAttr). Some clients still expect
  // dense elements, not dense arrays, so convert the arrays to elements before
  // serializing. The elements need to be converted back to arrays when
  // deserializing.
  // TODO: b/320507168 - Remove this conversion code.
  mlir::OwningOpRef<mlir::ModuleOp> cloned = module.clone();
  if (mlir::failed(mlir::writeBytecodeToFile(*cloned, os, config))) {
    return absl::InvalidArgumentError("mlir::writeBytecodeToFile failed");
  }
  return bytecode;
}

absl::StatusOr<std::string> SerializeUsingVersionedStablehlo(
    mlir::ModuleOp mlir_module, absl::string_view requested_target,
    bool inplace) {
  mlir::MLIRContext* context = mlir_module->getContext();
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(context);

  // Usually the plugin is older than the framework, but occasionally a plugin's
  // nightly build will use the latest public release of a framework. Serialize
  // using the framework's version in these cases.
  auto target = mlir::stablehlo::getSmallerVersion(
      requested_target, mlir::stablehlo::getCurrentVersion());
  if (mlir::failed(target)) {
    return absl::InvalidArgumentError(
        "Invalid StableHLO target version requested.");
  }

  // Legalize CHLO -> [StableHLO+Shape] -> StableHLO
  // Preserve higher-level ops with XLA support. To be replaced by composites.
  mlir::PassManager pm(context);
  // Expand stablehlo complex math functions such as log_plus_one, etc.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloComplexMathExpanderPass());

  xla::sdy::addSdyRoundTripExportPipeline(pm);
  pm.addPass(mlir::stablehlo_ext::createChloPreserveHighLevelOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createChloLegalizeToStablehloPass());
  pm.addPass(mlir::stablehlo::createStablehloCompatibilityExpanderPass(
      {target.value()}));
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
  if (mlir::failed(mlir::stablehlo::serializePortableArtifact(
          mlir_module, target.value(), os))) {
    const absl::Status status = diagnostic_handler.ConsumeStatus();
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to serialize StableHLO to plugin version ", target.value(),
        ";\n\nDetailed error from MLIR: ", status.message()));
  }
  return buffer;
}

absl::Status UpgradeVersionedStablehlo(mlir::ModuleOp mlir_module) {
  // Upgrade if VHLO
  mlir::PassManager pm(mlir_module->getContext());
  mlir::stablehlo::createStablehloDeserializePipeline(pm);
  if (!mlir::succeeded(pm.run(mlir_module)))
    return xla::InvalidArgument("Failed to upgrade versioned StableHLO.");
  return absl::OkStatus();
}

std::string GetDefaultStablehloVersion(std::optional<int64_t> plugin_version) {
  // TODO: (b/370803410) Use WEEK_12 in PJRT, some plugins were not up to date,
  // so temporarily using 1.0.0 to allow them time for a new release.
  // PJRT v54 released Jun 10, so most plugins should use WEEK_12 by default.
  if (plugin_version.has_value() && plugin_version.value() < 54) {
    return "0.19.0";
  }

  // This version must be >=12w old.
  return mlir::vhlo::Version::fromCompatibilityRequirement(
             mlir::vhlo::Version::CompatibilityRequirement::WEEK_12)
      .toString();
}

absl::StatusOr<std::string> Serialize(mlir::ModuleOp module,
                                      absl::string_view target, bool inplace) {
  // Current PJRT users expect 12 weeks forward compat, VHLO provides this
  // compat.
  // TODO (b/344930098): Allow VHLO interop and remove the all_stablehlo check
  bool all_stablehlo_or_shardy = true;
  module->walk([&](mlir::Operation* op) {
    if (!llvm::isa<mlir::ModuleOp>(op) &&
        !llvm::isa<mlir::stablehlo::StablehloDialect, mlir::func::FuncDialect,
                   mlir::chlo::ChloDialect, mlir::sdy::SdyDialect>(
            op->getDialect())) {
      all_stablehlo_or_shardy = false;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!all_stablehlo_or_shardy) {
    return SerializeUsingNativeBytecode(module);
  }
  return SerializeUsingVersionedStablehlo(module, target, inplace);
}

}  // namespace xla

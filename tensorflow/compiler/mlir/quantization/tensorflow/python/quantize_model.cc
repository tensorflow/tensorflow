/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/constants.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_preprocess.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_freeze_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/tsl/platform/path.h"

namespace tensorflow {
namespace quantization {
namespace internal {
namespace {

void AddExportPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::quant::CreateInsertMainFunctionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::CreateFunctionalToExecutorDialectConversionPass());
  pm.addPass(mlir::CreateBreakUpIslandsPass());
  pm.addPass(mlir::quant::CreateMergeInitializerFunctionOpsToMainPass());
}

// Returns the name of the initializer node from a set of control return nodes.
// Returns an empty string if no initializer node exists. This assumes that
// there is only one node for initialization.
std::string GetInitNodeName(
    const absl::flat_hash_set<Node *> &control_ret_nodes) {
  for (Node *control_ret_node : control_ret_nodes) {
    if (absl::StrContains(control_ret_node->name(), kInitOpNamePrefix)) {
      VLOG(1) << "Init node found: " << control_ret_node->name();
      return control_ret_node->name();
    }
  }
  return "";
}

// Converts MLIR ModuleOp to ExportedModel. Returns InternalError status
// when the GraphDef conversion fails.
absl::StatusOr<ExportedModel> ConvertMlirModuleToExportedModel(
    const mlir::ModuleOp module_op) {
  const GraphExportConfig config{};
  FunctionLibraryDefinition flib_def{OpRegistry::Global(),
                                     FunctionDefLibrary()};
  std::unique_ptr<Graph> graph;
  absl::flat_hash_set<Node *> control_ret_nodes{};
  if (const auto status = ConvertMlirToGraph(module_op, config, &graph,
                                             &flib_def, &control_ret_nodes);
      !status.ok()) {
    return absl::InternalError("Failed to convert MLIR to GraphDef. " +
                               status.error_message());
  }

  auto graph_def = std::make_unique<GraphDef>();
  graph->ToGraphDef(graph_def.get());

  return ExportedModel{*graph_def, GetInitNodeName(control_ret_nodes)};
}

// Creates a new file to dump the intermediate MLIRs by prefixing the
// `dump_file_name` with the value of the TF_QUANT_MLIR_DUMP_PREFIX env
// variable. Returns absl::FailedPreconditionError if the env variable is not
// set or set to an empty string.
[[nodiscard]] absl::StatusOr<std::unique_ptr<llvm::raw_fd_ostream>>
CreateMlirDumpFile(const absl::string_view dump_file_name) {
  const auto prefix =
      absl::NullSafeStringView(std::getenv("TF_QUANT_MLIR_DUMP_PREFIX"));
  if (prefix.empty()) {
    return absl::FailedPreconditionError(
        "Environment variable not set: TF_QUANT_MLIR_DUMP_PREFIX, "
        "IR dump file for TF quantization is not created.");
  }

  Env *env = Env::Default();
  const Status status = env->RecursivelyCreateDir(std::string(prefix));
  if (!status.ok()) {
    return ToAbslStatus(status);
  }

  std::error_code ec{};  // NOLINT: Required to create llvm::raw_fd_ostream
  const std::string dump_file_path = tsl::io::JoinPath(prefix, dump_file_name);
  auto dump_file = std::make_unique<llvm::raw_fd_ostream>(dump_file_path, ec);
  if (ec) {
    return absl::InternalError(absl::StrFormat(
        "Unable to open file: %s, error: %s", dump_file_path, ec.message()));
  }

  LOG(INFO) << "IR dump file created: " << dump_file_path;
  return dump_file;
}

// If verbosity level >= 1, this will dump intermediate IRs of passes to a file.
// The file path is given by prefixing `name`.mlir with the value of the
// TF_QUANT_MLIR_DUMP_PREFIX env variable. Returns `nullptr` iff the verbosity
// level < 1 or TF_QUANT_MLIR_DUMP_PREFIX is not set or set to an empty string.
// The returned ostream instance should live until the pass run is complete.
[[nodiscard]] absl::StatusOr<std::unique_ptr<llvm::raw_ostream>>
MaybeEnableIrPrinting(mlir::PassManager &pm, const absl::string_view name) {
  if (!VLOG_IS_ON(1)) {
    // Verbosity level is too low to enable IR printing.
    return nullptr;
  }

  absl::StatusOr<std::unique_ptr<llvm::raw_fd_ostream>> dump_file =
      CreateMlirDumpFile(/*dump_file_name=*/absl::StrCat(name, ".mlir"));
  if (absl::IsFailedPrecondition(dump_file.status())) {
    // The env variable TF_QUANT_MLIR_DUMP_PREFIX is not set. IR printing will
    // not be enabled.
    LOG(WARNING) << dump_file.status();
    return nullptr;
  } else if (!dump_file.ok()) {
    return dump_file.status();
  }

  mlir::OpPrintingFlags flag{};
  flag.useLocalScope().elideLargeElementsAttrs().enableDebugInfo();

  // IR printing requires multithreading disabled.
  pm.getContext()->disableMultithreading();

  // The configuration uses the default parameter values for
  // `PassManager::enableIRPrinting`, except for the `printModuleScope`
  // parameter, which is true by default. It is set to false to avoid the dump
  // file size becoming too large when the passes are running on a large model.
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/[](mlir::Pass *,
                                   mlir::Operation *) { return true; },
      /*shouldPrintAfterPass=*/
      [](mlir::Pass *, mlir::Operation *) { return true; },
      /*printModuleScope=*/false, /*printAfterOnlyOnChange=*/true,
      /*printAfterOnlyOnFailure=*/false, **dump_file, flag);

  LOG(INFO) << "IR dump for TensorFlow quantization pipeline enabled. ";
  return dump_file;
}

}  // namespace

absl::StatusOr<ExportedModel> QuantizeQatModel(
    const absl::string_view saved_model_path,
    const absl::string_view exported_names_str, const absl::string_view tags,
    const absl::string_view quant_opts_serialized) {
  const std::unordered_set<std::string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());
  std::vector<std::string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  QuantizationOptions quantization_options;
  if (!quantization_options.ParseFromString(
          // NOLINTNEXTLINE: std::string conversion required.
          std::string(quant_opts_serialized))) {
    return absl::InternalError(
        "Failed to parse QuantizationOptions from string.");
  }

  // Convert the SavedModelBundle to an MLIR module.
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::scf::SCFDialect,
                  mlir::tf_saved_model::TensorFlowSavedModelDialect,
                  mlir::TF::TensorFlowDialect, mlir::shape::ShapeDialect,
                  mlir::quant::QuantizationDialect>();
  mlir::MLIRContext context(registry);

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(saved_model_path, tag_set,
                                          absl::MakeSpan(exported_names),
                                          &context, import_options,
                                          /*lift_variables=*/false, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("Failed to import SavedModel: " +
                               module.status().error_message());
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  const Status status = PreprocessAndFreezeGraph(
      module_ref.get(), &context, bundle ? bundle->GetSession() : nullptr);
  if (!status.ok()) {
    return absl::InternalError("Failed to preprocess graph: " +
                               status.error_message());
  }

  mlir::PassManager pm(&context);
  const absl::StatusOr<std::unique_ptr<llvm::raw_ostream>> out_dump_file =
      MaybeEnableIrPrinting(pm, /*name=*/"tf_quantize_qat");
  if (!out_dump_file.ok()) {
    return absl::InternalError(out_dump_file.status().message());
  }

  AddQuantizeQatPasses(pm, quantization_options);
  AddExportPasses(pm);

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module_ref))) {
    return absl::InternalError(
        "failed to apply the quantization: " +
        diagnostic_handler.ConsumeStatus().error_message());
  }

  return ConvertMlirModuleToExportedModel(*module_ref);
}

absl::StatusOr<ExportedModel> QuantizePtqModelPreCalibration(
    const absl::string_view saved_model_path,
    const absl::string_view exported_names_str, const absl::string_view tags,
    const absl::string_view quant_opts_serialized) {
  const std::unordered_set<std::string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());
  std::vector<std::string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  QuantizationOptions quantization_options;
  if (!quantization_options.ParseFromString(
          // NOLINTNEXTLINE: std::string conversion required.
          std::string(quant_opts_serialized))) {
    return absl::InternalError(
        "Failed to parse QuantizationOptions from string.");
  }

  // Convert the SavedModelBundle to an MLIR module.
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::scf::SCFDialect,
                  mlir::tf_saved_model::TensorFlowSavedModelDialect,
                  mlir::TF::TensorFlowDialect, mlir::shape::ShapeDialect,
                  mlir::quant::QuantizationDialect>();
  mlir::MLIRContext context(registry);

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(saved_model_path, tag_set,
                                          absl::MakeSpan(exported_names),
                                          &context, import_options,
                                          /*lift_variables=*/false, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("Failed to import SavedModel: " +
                               module.status().error_message());
  }
  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  const Status status = PreprocessAndFreezeGraph(
      module_ref.get(), &context, bundle ? bundle->GetSession() : nullptr);
  if (!status.ok()) {
    return absl::InternalError("Failed to preprocess graph: " +
                               status.error_message());
  }

  mlir::PassManager pm(&context);
  const absl::StatusOr<std::unique_ptr<llvm::raw_ostream>> out_dump_file =
      MaybeEnableIrPrinting(pm, /*name=*/"tf_quantize_ptq_pre_calibration");
  if (!out_dump_file.ok()) {
    return absl::InternalError(out_dump_file.status().message());
  }

  AddQuantizePtqPreCalibrationPasses(pm, quantization_options);
  AddExportPasses(pm);

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module_ref))) {
    return absl::InternalError(
        "Failed to apply the quantization at the pre-calibration stage: " +
        diagnostic_handler.ConsumeStatus().error_message());
  }

  return ConvertMlirModuleToExportedModel(*module_ref);
}

absl::StatusOr<ExportedModel> QuantizePtqModelPostCalibration(
    const absl::string_view saved_model_path,
    const absl::string_view exported_names_str, const absl::string_view tags,
    const absl::string_view quant_opts_serialized) {
  const std::unordered_set<std::string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());
  std::vector<std::string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  QuantizationOptions quantization_options;
  if (!quantization_options.ParseFromString(
          // NOLINTNEXTLINE: std::string conversion required.
          std::string(quant_opts_serialized))) {
    return absl::InternalError(
        "Failed to parse QuantizationOptions from string.");
  }

  // Convert the SavedModelBundle to an MLIR module.
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::scf::SCFDialect,
                  mlir::tf_saved_model::TensorFlowSavedModelDialect,
                  mlir::TF::TensorFlowDialect, mlir::shape::ShapeDialect,
                  mlir::quant::QuantizationDialect>();
  mlir::MLIRContext context(registry);

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(saved_model_path, tag_set,
                                          absl::MakeSpan(exported_names),
                                          &context, import_options,
                                          /*lift_variables=*/true, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("Failed to import SavedModel: " +
                               module.status().error_message());
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  mlir::PassManager pm(&context);
  const absl::StatusOr<std::unique_ptr<llvm::raw_ostream>> out_dump_file =
      MaybeEnableIrPrinting(pm, /*name=*/"tf_quantize_ptq_post_calibration");
  if (!out_dump_file.ok()) {
    return absl::InternalError(out_dump_file.status().message());
  }

  AddQuantizePtqPostCalibrationPasses(pm, quantization_options);
  AddExportPasses(pm);

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module_ref))) {
    return absl::InternalError(
        "Failed to apply the quantization at the post-calibation stage: " +
        diagnostic_handler.ConsumeStatus().error_message());
  }

  return ConvertMlirModuleToExportedModel(*module_ref);
}

absl::StatusOr<ExportedModel> QuantizePtqDynamicRange(
    const absl::string_view saved_model_path,
    const absl::string_view exported_names_str, const absl::string_view tags,
    const absl::string_view quant_opts_serialized) {
  const std::unordered_set<std::string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());
  std::vector<std::string> exported_names =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  QuantizationOptions quantization_options;
  if (!quantization_options.ParseFromString(
          // NOLINTNEXTLINE: std::string conversion required.
          std::string(quant_opts_serialized))) {
    return absl::InternalError(
        "Failed to parse QuantizationOptions from string.");
  }

  // Convert the SavedModelBundle to an MLIR module.
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::scf::SCFDialect,
                  mlir::tf_saved_model::TensorFlowSavedModelDialect,
                  mlir::TF::TensorFlowDialect, mlir::shape::ShapeDialect,
                  mlir::quant::QuantizationDialect>();
  mlir::MLIRContext context(registry);

  MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      SavedModelSignatureDefsToMlirImport(saved_model_path, tag_set,
                                          absl::MakeSpan(exported_names),
                                          &context, import_options,
                                          /*lift_variables=*/false, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("Failed to import SavedModel: " +
                               module.status().error_message());
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  const Status status = PreprocessAndFreezeGraph(
      module_ref.get(), &context, bundle ? bundle->GetSession() : nullptr);
  if (!status.ok()) {
    return absl::InternalError("Failed to preprocess graph: " +
                               status.error_message());
  }

  mlir::PassManager pm(&context);
  const absl::StatusOr<std::unique_ptr<llvm::raw_ostream>> out_dump_file =
      MaybeEnableIrPrinting(pm, /*name=*/"tf_quantize_drq");
  if (!out_dump_file.ok()) {
    return absl::InternalError(out_dump_file.status().message());
  }

  AddQuantizePtqDynamicRangePasses(pm, quantization_options);
  AddExportPasses(pm);

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module_ref))) {
    return absl::InternalError(
        "Failed to apply the quantization: " +
        diagnostic_handler.ConsumeStatus().error_message());
  }

  return ConvertMlirModuleToExportedModel(*module_ref);
}

}  // namespace internal
}  // namespace quantization
}  // namespace tensorflow

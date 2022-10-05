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

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
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

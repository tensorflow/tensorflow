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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
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

Status PreprocessAndFreezeGraph(mlir::ModuleOp module,
                                mlir::MLIRContext* context,
                                llvm::Optional<Session*> session) {
  mlir::PassManager pm_before_freezing_variables(context);
  mlir::StatusScopedDiagnosticHandler statusHandler(module.getContext(),
                                                    /*propagate=*/true);

  mlir::TF::StandardPipelineOptions standard_pipeline_options;
  standard_pipeline_options.enable_inliner = false;
  standard_pipeline_options.form_clusters = false;
  mlir::TF::CreateTFStandardPipeline(pm_before_freezing_variables,
                                     standard_pipeline_options);

  pm_before_freezing_variables.addNestedPass<mlir::func::FuncOp>(
      mlir::TFDevice::CreateDecomposeResourceOpsPass());

  mlir::PassManager pm_after_freezing_variables(context);
  pm_after_freezing_variables.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm_after_freezing_variables.addPass(mlir::createCanonicalizerPass());
  pm_after_freezing_variables.addPass(mlir::createInlinerPass());

  if (failed(pm_before_freezing_variables.run(module))) {
    return statusHandler.ConsumeStatus();
  }

  if (session.has_value() && failed(mlir::tf_saved_model::FreezeVariables(
                                 module, session.getValue()))) {
    return statusHandler.ConsumeStatus();
  }

  if (failed(pm_after_freezing_variables.run(module))) {
    return statusHandler.ConsumeStatus();
  }

  return OkStatus();
}

// Converts MLIR ModuleOp to TensorFlow GraphDef. Returns InternalError status
// when the GraphDef conversion fails.
absl::StatusOr<GraphDef> ConvertMlirModuleToGraphDef(
    const mlir::ModuleOp module_op) {
  GraphExportConfig config{};
  StatusOr<std::unique_ptr<GraphDef>> graph =
      ConvertMlirToGraphdef(module_op, config);
  if (!graph.ok()) {
    return absl::InternalError("Failed to convert MLIR to GraphDef: " +
                               graph.status().error_message());
  }
  return *std::move(*graph);
}

}  // namespace

absl::StatusOr<GraphDef> QuantizeQATModel(
    const absl::string_view saved_model_path,
    const absl::string_view exported_names_str, const absl::string_view tags,
    const std::string& quant_opts_serialized) {
  const std::unordered_set<std::string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());
  std::vector<std::string> exported_names_vec =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  QuantizationOptions quantization_options;
  if (!quantization_options.ParseFromString(quant_opts_serialized)) {
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
      SavedModelSignatureDefsToMlirImport(
          saved_model_path, tag_set,
          absl::Span<std::string>(exported_names_vec), &context, import_options,
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

  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateConvertFakeQuantToQdqPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateUnrollBatchMatMulPassPass());
  // TODO(b/229995333): Add PrepareLiftingPass for QAT. In QAT, AffineOps are
  // connected to FakeQuantOp instead of the ConstOp so need to add separate
  // pattern for FakeQuantOp.
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::quant::CreatePrepareLiftingPass());
  pm.addPass(mlir::quant::CreateLiftQuantizableSpotsAsFunctionsPass());
  pm.addPass(mlir::quant::CreateInsertQuantizedFunctionsPass(
      mlir::quant::QuantizationMethod::kQuantizationAwareTraining,
      quantization_options.op_set()));
  pm.addPass(mlir::quant::CreateQuantizeCompositeFunctionsPass(
      mlir::quant::QuantizationMethod::kQuantizationAwareTraining));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::quant::CreateOptimizePass());

  pm.addPass(mlir::quant::CreateInsertMainFunctionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::CreateFunctionalToExecutorDialectConversionPass());
  pm.addPass(mlir::CreateBreakUpIslandsPass());

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module_ref))) {
    return absl::InternalError(
        "failed to apply the quantization: " +
        diagnostic_handler.ConsumeStatus().error_message());
  }

  return ConvertMlirModuleToGraphDef(*module_ref);
}

absl::StatusOr<GraphDef> QuantizePTQModelPreCalibration(
    const absl::string_view saved_model_path,
    const absl::string_view exported_names_str, const absl::string_view tags) {
  const std::unordered_set<std::string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());
  std::vector<std::string> exported_names_vec =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());

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
      SavedModelSignatureDefsToMlirImport(
          saved_model_path, tag_set,
          absl::Span<std::string>(exported_names_vec), &context, import_options,
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
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateUnrollBatchMatMulPassPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::quant::CreatePrepareLiftingPass());
  pm.addPass(mlir::quant::CreateLiftQuantizableSpotsAsFunctionsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateInsertCustomAggregationOpsPass());
  pm.addPass(mlir::quant::CreateIssueIDsOfCustomAggregationOpsPass());
  pm.addPass(mlir::quant::CreateInsertMainFunctionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::CreateFunctionalToExecutorDialectConversionPass());
  pm.addPass(mlir::CreateBreakUpIslandsPass());

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module_ref))) {
    return absl::InternalError(
        "Failed to apply the quantization at the pre-calibration stage: " +
        diagnostic_handler.ConsumeStatus().error_message());
  }

  return ConvertMlirModuleToGraphDef(*module_ref);
}

absl::StatusOr<GraphDef> QuantizePTQModelPostCalibration(
    const absl::string_view saved_model_path,
    const absl::string_view exported_names_str, const absl::string_view tags,
    const std::string& quant_opts_serialized) {
  const std::unordered_set<std::string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());
  std::vector<std::string> exported_names_vec =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  QuantizationOptions quantization_options;
  if (!quantization_options.ParseFromString(quant_opts_serialized)) {
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
      SavedModelSignatureDefsToMlirImport(
          saved_model_path, tag_set,
          absl::Span<std::string>(exported_names_vec), &context, import_options,
          /*lift_variables=*/true, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("Failed to import SavedModel: " +
                               module.status().error_message());
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = std::move(module).value();

  mlir::PassManager pm(&context);

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateConvertCustomAggregationOpToQuantStatsPass());
  pm.addPass(mlir::quant::CreateInsertQuantizedFunctionsPass(
      mlir::quant::QuantizationMethod::kPostTrainingQuantization,
      quantization_options.op_set()));
  pm.addPass(mlir::quant::CreateQuantizeCompositeFunctionsPass(
      mlir::quant::QuantizationMethod::kPostTrainingQuantization));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::quant::CreateOptimizePass());

  pm.addPass(mlir::quant::CreateInsertMainFunctionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::CreateFunctionalToExecutorDialectConversionPass());
  pm.addPass(mlir::CreateBreakUpIslandsPass());

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module_ref))) {
    return absl::InternalError(
        "Failed to apply the quantization at the post-calibation stage: " +
        diagnostic_handler.ConsumeStatus().error_message());
  }

  return ConvertMlirModuleToGraphDef(*module_ref);
}

absl::StatusOr<GraphDef> QuantizePTQDynamicRange(
    const absl::string_view saved_model_path,
    const absl::string_view exported_names_str, const absl::string_view tags,
    const std::string& quant_opts_serialized) {
  const std::unordered_set<std::string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());
  std::vector<std::string> exported_names_vec =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());
  QuantizationOptions quantization_options;
  if (!quantization_options.ParseFromString(quant_opts_serialized)) {
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
      SavedModelSignatureDefsToMlirImport(
          saved_model_path, tag_set,
          absl::Span<std::string>(exported_names_vec), &context, import_options,
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

  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateUnrollBatchMatMulPassPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::quant::CreatePrepareLiftingPass());
  pm.addPass(mlir::quant::CreateLiftQuantizableSpotsAsFunctionsDRQPass());
  pm.addPass(mlir::quant::CreateInsertQuantizedFunctionsPass(
      mlir::quant::QuantizationMethod::kDynamicRangeQuantization,
      quantization_options.op_set()));
  pm.addPass(mlir::quant::CreateQuantizeCompositeFunctionsPass(
      mlir::quant::QuantizationMethod::kDynamicRangeQuantization));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addPass(mlir::quant::CreateInsertMainFunctionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::CreateFunctionalToExecutorDialectConversionPass());
  pm.addPass(mlir::CreateBreakUpIslandsPass());

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module_ref))) {
    return absl::InternalError(
        "Failed to apply the quantization: " +
        diagnostic_handler.ConsumeStatus().error_message());
  }

  return ConvertMlirModuleToGraphDef(*module_ref);
}

}  // namespace internal
}  // namespace quantization
}  // namespace tensorflow

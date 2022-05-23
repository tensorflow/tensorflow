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
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_import_options.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_utils.h"

using tensorflow::GraphDef;

namespace tensorflow {
namespace quantization {
namespace internal {

absl::StatusOr<GraphDef> QuantizeQATModel(absl::string_view saved_model_path,
                                          absl::string_view exported_names_str,
                                          absl::string_view tags) {
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

  tensorflow::MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<tensorflow::SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  tensorflow::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      tensorflow::SavedModelSignatureDefsToMlirImport(
          saved_model_path, tag_set,
          absl::Span<std::string>(exported_names_vec), &context, import_options,
          true, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("failed to import SavedModel: " +
                               module.status().error_message());
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = module.ConsumeValueOrDie();

  mlir::PassManager pm(&context);

  std::string error;
  llvm::raw_string_ostream error_stream(error);

  pm.addPass(mlir::createCanonicalizerPass());
  // Freezes constants so that FakeQuant ops can reference quantization ranges.
  pm.addPass(mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::tf_saved_model::CreateFreezeGlobalTensorsPass());

  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateConvertFakeQuantToQdqPass());

  // TODO(b/229995333): Add PrepareLiftingPass for QAT. In QAT, AffineOps are
  // connected to FakeQuantOp instead of the ConstOp so need to add separate
  // pattern for FakeQuantOp.
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::quant::CreatePrepareLiftingPass());
  pm.addPass(mlir::quant::CreateLiftQuantizableSpotsAsFunctionsPass());
  pm.addPass(mlir::quant::CreateInsertQuantizedFunctionsPass(
      mlir::quant::QuantizationMethod::kQuantizationAwareTraining));
  pm.addPass(mlir::quant::CreateQuantizeCompositeFunctionsPass(
      mlir::quant::QuantizationMethod::kQuantizationAwareTraining));
  pm.addPass(mlir::createSymbolDCEPass());

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

  // Export as GraphDef.
  tensorflow::GraphExportConfig confs;
  stream_executor::port::StatusOr<std::unique_ptr<GraphDef>> graph =
      tensorflow::ConvertMlirToGraphdef(*module_ref, confs);
  if (!graph.ok()) {
    return absl::InternalError("failed to convert MLIR to graphdef: " +
                               graph.status().error_message());
  }

  return *graph.ConsumeValueOrDie();
}

absl::StatusOr<GraphDef> QuantizePTQModelPreCalibration(
    absl::string_view saved_model_path, absl::string_view exported_names_str,
    absl::string_view tags) {
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

  tensorflow::MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<tensorflow::SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  tensorflow::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      tensorflow::SavedModelSignatureDefsToMlirImport(
          saved_model_path, tag_set,
          absl::Span<std::string>(exported_names_vec), &context, import_options,
          true, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("failed to import SavedModel: " +
                               module.status().error_message());
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = module.ConsumeValueOrDie();

  mlir::PassManager pm(&context);

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::quant::CreatePrepareLiftingPass());
  // TODO(b/230426953): Add TFShapeInferencePass to infer shapes for unranked
  // types. pm.addPass(mlir::TF::CreateTFShapeInferencePass());
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
        "failed to apply the quantization at the pre-calibration stage: " +
        diagnostic_handler.ConsumeStatus().error_message());
  }

  // Export as GraphDef.
  tensorflow::GraphExportConfig confs;
  stream_executor::port::StatusOr<std::unique_ptr<GraphDef>> graph =
      tensorflow::ConvertMlirToGraphdef(*module_ref, confs);
  if (!graph.ok()) {
    return absl::InternalError("failed to convert MLIR to graphdef: " +
                               graph.status().error_message());
  }

  return *graph.ConsumeValueOrDie();
}

absl::StatusOr<GraphDef> QuantizePTQModelPostCalibration(
    absl::string_view saved_model_path, absl::string_view exported_names_str,
    absl::string_view tags) {
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

  tensorflow::MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<tensorflow::SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  tensorflow::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module =
      tensorflow::SavedModelSignatureDefsToMlirImport(
          saved_model_path, tag_set,
          absl::Span<std::string>(exported_names_vec), &context, import_options,
          true, &bundle);

  if (!module.status().ok()) {
    return absl::InternalError("failed to import SavedModel: " +
                               module.status().error_message());
  }

  mlir::OwningOpRef<mlir::ModuleOp> module_ref = module.ConsumeValueOrDie();

  mlir::PassManager pm(&context);

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateConvertCustomAggregationOpToQuantStatsPass());
  pm.addPass(mlir::quant::CreateInsertQuantizedFunctionsPass(
      mlir::quant::QuantizationMethod::kPostTrainingQuantization));
  pm.addPass(mlir::quant::CreateQuantizeCompositeFunctionsPass(
      mlir::quant::QuantizationMethod::kPostTrainingQuantization));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::quant::CreateInsertMainFunctionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::CreateFunctionalToExecutorDialectConversionPass());
  pm.addPass(mlir::CreateBreakUpIslandsPass());

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*module_ref))) {
    return absl::InternalError(
        "failed to apply the quantization at the post-calibation stage: " +
        diagnostic_handler.ConsumeStatus().error_message());
  }

  // Export as GraphDef.
  tensorflow::GraphExportConfig confs;
  stream_executor::port::StatusOr<std::unique_ptr<GraphDef>> graph =
      tensorflow::ConvertMlirToGraphdef(*module_ref, confs);
  if (!graph.ok()) {
    return absl::InternalError("failed to convert MLIR to graphdef: " +
                               graph.status().error_message());
  }

  return *graph.ConsumeValueOrDie();
}

}  // namespace internal
}  // namespace quantization
}  // namespace tensorflow

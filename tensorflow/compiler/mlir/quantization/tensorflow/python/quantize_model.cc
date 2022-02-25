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
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
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
#include "tensorflow/core/util/env_var.h"

using tensorflow::FunctionDefLibrary;
using tensorflow::Graph;
using tensorflow::GraphDef;
using tensorflow::ImportGraphDefOptions;
using tensorflow::OpRegistry;

namespace mlir {
namespace quant {

absl::StatusOr<tensorflow::GraphDef> QuantizeQATModel(
    absl::string_view saved_model_path, absl::string_view exported_names_str,
    absl::string_view tags) {
  const std::unordered_set<std::string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());
  std::vector<std::string> exported_names_vec =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());

  // Convert the SavedModelBundle to an MLIR module.
  DialectRegistry registry;
  registry.insert<StandardOpsDialect, scf::SCFDialect,
                  tf_saved_model::TensorFlowSavedModelDialect,
                  TF::TensorFlowDialect, shape::ShapeDialect,
                  QuantizationDialect>();
  MLIRContext context(registry);

  tensorflow::MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<tensorflow::SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  auto module_or = tensorflow::SavedModelSignatureDefsToMlirImport(
      saved_model_path, tag_set, absl::Span<std::string>(exported_names_vec),
      &context, import_options, /*lift_variables=*/true, &bundle);

  if (!module_or.status().ok()) {
    return absl::InternalError("failed to import SavedModel: " +
                               module_or.status().error_message());
  }

  OwningOpRef<mlir::ModuleOp> moduleRef = module_or.ConsumeValueOrDie();

  PassManager pm(&context);

  std::string error;
  llvm::raw_string_ostream error_stream(error);

  pm.addPass(createCanonicalizerPass());
  // Freezes constants so that FakeQuant ops can reference quantization ranges.
  pm.addPass(tf_saved_model::CreateOptimizeGlobalTensorsPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(tf_saved_model::CreateFreezeGlobalTensorsPass());

  pm.addNestedPass<FuncOp>(CreateConvertFakeQuantToQdqPass());
  pm.addNestedPass<FuncOp>(TF::CreateFusedKernelMatcherPass());
  pm.addPass(CreateLiftQuantizableSpotsAsFunctionsPass());
  pm.addPass(CreateInsertQuantizedFunctionsPass());
  pm.addPass(CreateQuantizeCompositeFunctionsPass());
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addPass(CreateInsertMainFunctionPass());
  pm.addNestedPass<FuncOp>(CreateFunctionalToExecutorDialectConversionPass());
  pm.addPass(CreateBreakUpIslandsPass());

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*moduleRef))) {
    return absl::InternalError(
        "failed to apply the quantization: " +
        diagnostic_handler.ConsumeStatus().error_message());
  }

  // Export as GraphDef.
  tensorflow::GraphExportConfig confs;
  auto graph_or = tensorflow::ConvertMlirToGraphdef(*moduleRef, confs);
  if (!graph_or.ok()) {
    return absl::InternalError("failed to convert MLIR to graphdef: " +
                               graph_or.status().error_message());
  }

  return *graph_or.ConsumeValueOrDie();
}

absl::StatusOr<tensorflow::GraphDef> QuantizePTQModelPreCalibration(
    absl::string_view saved_model_path, absl::string_view exported_names_str,
    absl::string_view tags) {
  const std::unordered_set<std::string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());
  std::vector<std::string> exported_names_vec =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());

  // Convert the SavedModelBundle to an MLIR module.
  DialectRegistry registry;
  registry.insert<StandardOpsDialect, scf::SCFDialect,
                  tf_saved_model::TensorFlowSavedModelDialect,
                  TF::TensorFlowDialect, shape::ShapeDialect,
                  QuantizationDialect>();
  MLIRContext context(registry);

  tensorflow::MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<tensorflow::SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  auto module_or = tensorflow::SavedModelSignatureDefsToMlirImport(
      saved_model_path, tag_set, absl::Span<std::string>(exported_names_vec),
      &context, import_options,
      /*lift_variables=*/true, &bundle);

  if (!module_or.status().ok()) {
    return absl::InternalError("failed to import SavedModel: " +
                               module_or.status().error_message());
  }

  OwningOpRef<mlir::ModuleOp> moduleRef = module_or.ConsumeValueOrDie();

  PassManager pm(&context);

  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(TF::CreateFusedKernelMatcherPass());
  pm.addPass(CreateLiftQuantizableSpotsAsFunctionsPass());
  pm.addNestedPass<FuncOp>(CreateInsertCustomAggregationOpsPass());
  pm.addPass(CreateIssueIDsOfCustomAggregationOpsPass());
  pm.addPass(CreateInsertMainFunctionPass());
  pm.addNestedPass<FuncOp>(CreateFunctionalToExecutorDialectConversionPass());
  pm.addPass(CreateBreakUpIslandsPass());

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*moduleRef))) {
    return absl::InternalError(
        "failed to apply the quantization at the pre-calibration stage: " +
        diagnostic_handler.ConsumeStatus().error_message());
  }

  // Export as GraphDef.
  tensorflow::GraphExportConfig confs;
  auto graph_or = tensorflow::ConvertMlirToGraphdef(*moduleRef, confs);
  if (!graph_or.ok()) {
    return absl::InternalError("failed to convert MLIR to graphdef: " +
                               graph_or.status().error_message());
  }

  return *graph_or.ConsumeValueOrDie();
}

absl::StatusOr<tensorflow::GraphDef> QuantizePTQModelPostCalibration(
    absl::string_view saved_model_path, absl::string_view exported_names_str,
    absl::string_view tags) {
  const std::unordered_set<std::string> tag_set =
      absl::StrSplit(tags, ',', absl::SkipEmpty());
  std::vector<std::string> exported_names_vec =
      absl::StrSplit(exported_names_str, ',', absl::SkipEmpty());

  // Convert the SavedModelBundle to an MLIR module.
  DialectRegistry registry;
  registry.insert<StandardOpsDialect, scf::SCFDialect,
                  tf_saved_model::TensorFlowSavedModelDialect,
                  TF::TensorFlowDialect, shape::ShapeDialect,
                  QuantizationDialect>();
  MLIRContext context(registry);

  tensorflow::MLIRImportOptions import_options;
  import_options.upgrade_legacy = true;
  auto bundle = std::make_unique<tensorflow::SavedModelBundle>();

  // TODO(b/213406917): Add support for the object graph based saved model input
  auto module_or = tensorflow::SavedModelSignatureDefsToMlirImport(
      saved_model_path, tag_set, absl::Span<std::string>(exported_names_vec),
      &context, import_options,
      /*lift_variables=*/true, &bundle);

  if (!module_or.status().ok()) {
    return absl::InternalError("failed to import SavedModel: " +
                               module_or.status().error_message());
  }

  OwningOpRef<mlir::ModuleOp> moduleRef = module_or.ConsumeValueOrDie();

  PassManager pm(&context);

  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(CreateConvertCustomAggregationOpToQuantStatsPass());
  pm.addPass(CreateInsertQuantizedFunctionsPass());
  pm.addPass(CreateQuantizeCompositeFunctionsPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(CreateInsertMainFunctionPass());
  pm.addNestedPass<FuncOp>(CreateFunctionalToExecutorDialectConversionPass());
  pm.addPass(CreateBreakUpIslandsPass());

  mlir::StatusScopedDiagnosticHandler diagnostic_handler(&context);
  if (failed(pm.run(*moduleRef))) {
    return absl::InternalError(
        "failed to apply the quantization at the post-calibation stage: " +
        diagnostic_handler.ConsumeStatus().error_message());
  }

  // Export as GraphDef.
  tensorflow::GraphExportConfig confs;
  auto graph_or = tensorflow::ConvertMlirToGraphdef(*moduleRef, confs);
  if (!graph_or.ok()) {
    return absl::InternalError("failed to convert MLIR to graphdef: " +
                               graph_or.status().error_message());
  }

  return *graph_or.ConsumeValueOrDie();
}

}  // namespace quant
}  // namespace mlir

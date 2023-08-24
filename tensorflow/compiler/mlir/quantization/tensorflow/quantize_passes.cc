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
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_passes.h"

#include <optional>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace tensorflow {
namespace quantization {
namespace {

void AddConvertTpuToCpuModelPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::quant::CreateConvertTpuModelToCpuPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::quant::CreateCastBf16OpsToF32Pass());
}
}  // namespace

void AddQuantizeQatPasses(
    mlir::PassManager &pm, const QuantizationOptions &quantization_options,
    std::optional<const absl::string_view> mlir_dump_file_prefix) {
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateConvertFakeQuantToQdqPass());
  if (quantization_options.op_set() == OpSet::UNIFORM_QUANTIZED) {
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::TF::CreateUnrollBatchMatMulPassPass());
  }
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateConvertTfXlaOpToTfOpPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreatePrepareLiftingPass(quantization_options.op_set()));

  pm.addPass(mlir::quant::CreateLiftQuantizableSpotsAsFunctionsPass(
      quantization_options.op_set(),
      quantization_options.enable_two_input_tensors()));
  pm.addPass(mlir::quant::CreateInsertQuantizedFunctionsPass(
      quantization_options.quantization_method().experimental_method(),
      quantization_options.op_set()));
  // TODO(b/260677670): Pass quantization options as pass's inputs where
  // applicable
  pm.addPass(mlir::quant::CreateQuantizeCompositeFunctionsPass(
      quantization_options.quantization_method().experimental_method(),
      quantization_options.op_set(),
      quantization_options.enable_per_channel_quantization(),
      quantization_options.min_num_elements_for_weights(),
      quantization_options.enable_legacy_weight_only(), mlir_dump_file_prefix));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  // TODO(b/264637396): Deprecate TF opset
  if (quantization_options.op_set() != OpSet::TF) {
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::TF::CreateTFShapeInferencePass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    if (quantization_options.op_set() == OpSet::XLA) {
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::quant::CreateReplaceCastHacksWithTFXLAOpsPass());
    }
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  }
  pm.addNestedPass<mlir::func::FuncOp>(mlir::quant::CreateOptimizePass());
}

void AddQuantizePtqDynamicRangePasses(
    mlir::PassManager &pm, const QuantizationOptions &quantization_options,
    std::optional<const absl::string_view> mlir_dump_file_prefix) {
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateUnrollBatchMatMulPassPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  if (quantization_options.experimental_enable_tpu_model_support()) {
    AddConvertTpuToCpuModelPasses(pm);
  }
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateConvertTfXlaOpToTfOpPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreatePrepareLiftingPass(quantization_options.op_set()));
  pm.addPass(mlir::quant::CreateLiftQuantizableSpotsAsFunctionsDRQPass(
      quantization_options.quantization_method().experimental_method(),
      quantization_options.op_set(),
      quantization_options.min_num_elements_for_weights()));
  pm.addPass(mlir::quant::CreateInsertQuantizedFunctionsPass(
      quantization_options.quantization_method().experimental_method(),
      quantization_options.op_set()));
  pm.addPass(mlir::quant::CreateQuantizeCompositeFunctionsPass(
      quantization_options.quantization_method().experimental_method(),
      quantization_options.op_set(),
      quantization_options.enable_per_channel_quantization(),
      quantization_options.min_num_elements_for_weights(),
      quantization_options.enable_legacy_weight_only(), mlir_dump_file_prefix));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  // TODO(b/264637396): Deprecate TF opset
  if (quantization_options.op_set() != OpSet::TF) {
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::TF::CreateTFShapeInferencePass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    if (quantization_options.op_set() == OpSet::XLA) {
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::quant::CreateReplaceCastHacksWithTFXLAOpsPass());
    }
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  }

  pm.addNestedPass<mlir::func::FuncOp>(mlir::quant::CreateOptimizePass());
}

void AddQuantizePtqPreCalibrationPasses(
    mlir::PassManager &pm, const QuantizationOptions &quantization_options) {
  if (quantization_options.op_set() == OpSet::UNIFORM_QUANTIZED) {
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::TF::CreateUnrollBatchMatMulPassPass());
  }
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  if (quantization_options.experimental_enable_tpu_model_support()) {
    AddConvertTpuToCpuModelPasses(pm);
  }
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateConvertTfXlaOpToTfOpPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreatePrepareLiftingPass(quantization_options.op_set()));
  pm.addPass(mlir::quant::CreateLiftQuantizableSpotsAsFunctionsPass(
      quantization_options.op_set(),
      quantization_options.enable_two_input_tensors()));
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateInsertCustomAggregationOpsPass());
  pm.addPass(mlir::quant::CreateIssueIDsOfCustomAggregationOpsPass());
}

void AddQuantizePtqPostCalibrationPasses(
    mlir::PassManager &pm, const QuantizationOptions &quantization_options,
    std::optional<const absl::string_view> mlir_dump_file_prefix) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateConvertCustomAggregationOpToQuantStatsPass());
  pm.addPass(mlir::quant::CreateInsertQuantizedFunctionsPass(
      quantization_options.quantization_method().experimental_method(),
      quantization_options.op_set()));
  pm.addPass(mlir::quant::CreateQuantizeCompositeFunctionsPass(
      quantization_options.quantization_method().experimental_method(),
      quantization_options.op_set(),
      quantization_options.enable_per_channel_quantization(),
      quantization_options.min_num_elements_for_weights(),
      quantization_options.enable_legacy_weight_only(), mlir_dump_file_prefix));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  // TODO(b/264637396): Deprecate TF opset
  if (quantization_options.op_set() != OpSet::TF) {
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::TF::CreateTFShapeInferencePass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    if (quantization_options.op_set() == OpSet::XLA) {
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::quant::CreateReplaceCastHacksWithTFXLAOpsPass());
    }
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  }
  pm.addNestedPass<mlir::func::FuncOp>(mlir::quant::CreateOptimizePass());
}

}  // namespace quantization
}  // namespace tensorflow

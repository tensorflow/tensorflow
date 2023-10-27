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
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"

namespace tensorflow {
namespace quantization {
namespace {

using ::tensorflow::quantization::QuantizationOptions;

// Currently server cannot handle UniformQuantizedTypes. Instead, unpack
// quantized ops to primitive StableHLO ops. We currently go through a
// StableHLO <-> MHLO roundtrip to utilize the MHLOQuantToInt pass.
void AddStablehloQuantToIntPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createInlinerPass());
  // StableHLO -> MHLO legalization.
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::stablehlo::createConvertMHLOQuantToIntPass(
          /*legalize_chlo=*/true));
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // MHLO -> StableHLO legalization.
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
}

void AddStaticRangeQuantizationPass(
    mlir::PassManager &pm, const QuantizationOptions &quantization_options,
    std::optional<const absl::string_view> mlir_dump_file_prefix) {
  // TODO: b/299545840 - Include QuantizeCompositeFunctionsPass as in bug.
}

void AddConvertTpuToCpuModelPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::quant::CreateConvertTpuModelToCpuPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::quant::CreateCastBf16OpsToF32Pass());
}

// NOMUTANTS -- Add tests for individual passes with migration below.
// Serializes the StableHLO module into a tf.XlaCallModuleOp for compatibility
// with passes that expect TF format. This also allows the StableHLO ops to be
// exported as a TF SavedModel.
void AddCallModuleSerializationPasses(mlir::PassManager &pm) {
  pm.addPass(
      mlir::quant::stablehlo::
          createReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass());
  pm.addPass(mlir::TF::CreateXlaCallModuleSerializationPass());
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
  if (quantization_options.experimental_enable_tpu_model_support()) {
    AddConvertTpuToCpuModelPasses(pm);
  }
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateConvertTfXlaOpToTfOpPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreatePrepareLiftingPass(quantization_options.op_set()));

  pm.addPass(mlir::quant::CreateLiftQuantizableSpotsAsFunctionsPass(
      quantization_options));
  pm.addPass(mlir::quant::CreateInsertQuantizedFunctionsPass(
      quantization_options.quantization_method().preset_method(),
      quantization_options.op_set()));
  // TODO: b/260677670 - Pass quantization options as pass's inputs where
  // applicable
  pm.addPass(mlir::quant::CreateQuantizeCompositeFunctionsPass(
      quantization_options.quantization_method().preset_method(),
      quantization_options.op_set(),
      quantization_options.enable_per_channel_quantization(),
      quantization_options.min_num_elements_for_weights(),
      quantization_options.enable_legacy_weight_only(), mlir_dump_file_prefix));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  // TODO: b/264637396 - Deprecate TF opset
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
      quantization_options.quantization_method().preset_method(),
      quantization_options.op_set(),
      quantization_options.min_num_elements_for_weights()));
  pm.addPass(mlir::quant::CreateInsertQuantizedFunctionsPass(
      quantization_options.quantization_method().preset_method(),
      quantization_options.op_set()));
  pm.addPass(mlir::quant::CreateQuantizeCompositeFunctionsPass(
      quantization_options.quantization_method().preset_method(),
      quantization_options.op_set(),
      quantization_options.enable_per_channel_quantization(),
      quantization_options.min_num_elements_for_weights(),
      quantization_options.enable_legacy_weight_only(), mlir_dump_file_prefix));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  // TODO: b/264637396) - Deprecate TF opset
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
      quantization_options));
  // TODO: b/295140328 - Add debugger support for weight only
  if (quantization_options.has_debugger_options()) {
    pm.addPass(mlir::quant::CreateAddDumpTensorOpPass(
        quantization_options.debugger_options().debugger_type(),
        quantization_options.debugger_options().log_dir_path()));
  }
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateInsertCustomAggregationOpsPass(
          quantization_options.calibration_options()));
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
      quantization_options.quantization_method().preset_method(),
      quantization_options.op_set()));
  pm.addPass(mlir::quant::CreateQuantizeCompositeFunctionsPass(
      quantization_options.quantization_method().preset_method(),
      quantization_options.op_set(),
      quantization_options.enable_per_channel_quantization(),
      quantization_options.min_num_elements_for_weights(),
      quantization_options.enable_legacy_weight_only(), mlir_dump_file_prefix));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  // TODO: b/264637396 - Deprecate TF opset
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

// StableHLO Quantization passes that are ran if StableHLO opset is selected.
// TODO: b/298581932 - Add tests for passes below once migration is complete.
void AddQuantizePtqPreCalibrationStablehloPasses(
    mlir::PassManager &pm, const QuantizationOptions &quantization_options) {
  pm.addPass(
      mlir::quant::stablehlo::createLiftQuantizableSpotsAsFunctionsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateInsertCustomAggregationOpsPass(
          quantization_options.calibration_options()));
  pm.addPass(mlir::quant::CreateIssueIDsOfCustomAggregationOpsPass());
  // NOMUTANTS -- Add tests after all passes in function below are migrated.
  // StableHLO Quantizer currently uses TF's calibration passes. Serialize
  // the StableHLO module as tf.XlaCallModule to run calibration.
  AddCallModuleSerializationPasses(pm);
}

// TODO: b/298581932 - Migrate and add passes below.
void AddQuantizePtqPostCalibrationStablehloPasses(
    mlir::PassManager &pm, const QuantizationOptions &quantization_options,
    std::optional<const absl::string_view> mlir_dump_file_prefix) {
  // Deserializes the StableHLO module embedded in tf.XlaCallModule and lifts
  // the StableHLO functions to the top level module. This is needed for
  // StableHLO quantization.
  pm.addPass(mlir::TF::CreateXlaCallModuleDeserializationPass());
  pm.addPass(mlir::quant::stablehlo::createRestoreFunctionNamePass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateConvertCustomAggregationOpToQuantStatsPass());
  AddStaticRangeQuantizationPass(pm, quantization_options,
                                 mlir_dump_file_prefix);
  AddStablehloQuantToIntPasses(pm);
  AddCallModuleSerializationPasses(pm);
}

void AddQuantizeWeightOnlyPasses(
    mlir::PassManager &pm, const QuantizationOptions &quantization_options,
    std::optional<const absl::string_view> mlir_dump_file_prefix) {
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  // Add PrepareLiftingPass to utilize its functionalities like folding batch
  // normalization ops and removing training related ops.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreatePrepareLiftingPass(quantization_options.op_set()));
  pm.addPass(mlir::quant::CreateQuantizeWeightsPass(quantization_options));
  pm.addPass(mlir::quant::CreatePropagateQuantizeTypePass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateReplaceCastHacksWithTFXLAOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  // Use optimize pass to remove double casts that are inserted when inlining
  // functions.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::quant::CreateOptimizePass());
}

}  // namespace quantization
}  // namespace tensorflow

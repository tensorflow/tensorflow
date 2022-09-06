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

void AddQuantizeQatPasses(mlir::PassManager &pm,
                          const QuantizationOptions &quantization_options) {
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
      mlir::quant::QuantizationMethod::kQuantizationAwareTraining,
      quantization_options.op_set()));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  // For XLA opset, the graph is inlined to take benefit of constant folding
  // and the TF Conv/Matmul ops with cast-hack are converted to XLA ops.
  if (quantization_options.op_set() == OpSet::XLA) {
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::TF::CreateTFShapeInferencePass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::quant::CreateReplaceCastHacksWithTFXLAOpsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  }

  pm.addNestedPass<mlir::func::FuncOp>(mlir::quant::CreateOptimizePass());
}

void AddQuantizePtqDynamicRangePasses(
    mlir::PassManager &pm, const QuantizationOptions &quantization_options) {
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateUnrollBatchMatMulPassPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::quant::CreatePrepareLiftingPass());
  pm.addPass(mlir::quant::CreateLiftQuantizableSpotsAsFunctionsDRQPass(
      quantization_options.min_num_elements_for_weights()));
  pm.addPass(mlir::quant::CreateInsertQuantizedFunctionsPass(
      mlir::quant::QuantizationMethod::kDynamicRangeQuantization,
      quantization_options.op_set()));
  pm.addPass(mlir::quant::CreateQuantizeCompositeFunctionsPass(
      mlir::quant::QuantizationMethod::kDynamicRangeQuantization,
      quantization_options.op_set()));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
}

void AddQuantizePtqPreCalibrationPasses(mlir::PassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateUnrollBatchMatMulPassPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::quant::CreatePrepareLiftingPass());
  pm.addPass(mlir::quant::CreateLiftQuantizableSpotsAsFunctionsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateInsertCustomAggregationOpsPass());
  pm.addPass(mlir::quant::CreateIssueIDsOfCustomAggregationOpsPass());
}

void AddQuantizePtqPostCalibrationPasses(
    mlir::PassManager &pm, const QuantizationOptions &quantization_options) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::quant::CreateConvertCustomAggregationOpToQuantStatsPass());
  pm.addPass(mlir::quant::CreateInsertQuantizedFunctionsPass(
      mlir::quant::QuantizationMethod::kPostTrainingQuantization,
      quantization_options.op_set()));
  pm.addPass(mlir::quant::CreateQuantizeCompositeFunctionsPass(
      mlir::quant::QuantizationMethod::kPostTrainingQuantization,
      quantization_options.op_set()));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  // For XLA opset, the graph is inlined to take benefit of constant folding
  // and the TF Conv/Matmul ops with cast-hack are converted to XLA ops.
  if (quantization_options.op_set() == OpSet::XLA) {
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::TF::CreateTFShapeInferencePass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::quant::CreateReplaceCastHacksWithTFXLAOpsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  }

  pm.addNestedPass<mlir::func::FuncOp>(mlir::quant::CreateOptimizePass());
}

}  // namespace quantization
}  // namespace tensorflow

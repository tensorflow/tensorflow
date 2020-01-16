/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"

#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassManager.h"  // TF:llvm-project
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/decode_constant.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"

namespace mlir {
/// Create a pass to convert from the TFExecutor to the TF control dialect.
std::unique_ptr<OpPassBase<FuncOp>>
CreateTFExecutorToControlDialectConversion();
}  // namespace mlir

namespace tensorflow {

void AddQuantizationPasses(const mlir::TFL::QuantizationSpecs& quant_specs,
                           mlir::OpPassManager* pass_manager) {
  pass_manager->addPass(mlir::TFL::CreatePrepareQuantizePass(quant_specs));
  pass_manager->addPass(mlir::TFL::CreateQuantizePass());
  bool emit_quant_adaptor_ops =
      quant_specs.inference_type != quant_specs.inference_input_type;
  pass_manager->addPass(
      mlir::TFL::CreatePostQuantizePass(emit_quant_adaptor_ops));

  if (quant_specs.default_ranges.first.hasValue() ||
      quant_specs.default_ranges.second.hasValue()) {
    pass_manager->addPass(mlir::TFL::CreateDefaultQuantParamsPass(
        quant_specs.default_ranges.first.getValueOr(0.0),
        quant_specs.default_ranges.second.getValueOr(0.0)));
    pass_manager->addPass(mlir::TFL::CreateQuantizePass());
    pass_manager->addPass(
        mlir::TFL::CreatePostQuantizePass(emit_quant_adaptor_ops));
  }
}

void AddTFToTFLConversionPasses(const mlir::TFL::PassConfig& pass_config,
                                mlir::OpPassManager* pass_manager) {
  pass_manager->addPass(mlir::tf_executor::CreateSwitchFoldPass());
  if (pass_config.skip_control_dialect) {
    // Merge islands.
    pass_manager->addPass(
        mlir::tf_executor::CreateTFExecutorIslandCoarseningPass());
    // Assuming island coarsening above results in a graph with a single island,
    // a canonicalization can be ran to hoist the ops of the single island out.
    pass_manager->addPass(mlir::createCanonicalizerPass());

    if (pass_config.form_clusters)
      pass_manager->addPass(mlir::TFDevice::CreateClusterFormationPass());
  } else {
    pass_manager->addPass(mlir::CreateTFExecutorToControlDialectConversion());
    pass_manager->addPass(mlir::TFControlFlow::CreateRaiseTFControlFlowPass());
  }

  if (!pass_config.quant_specs.serialized_quant_stats.empty()) {
    pass_manager->addPass(
        mlir::quant::CreateImportQuantStatsPassForTFControlDialect(
            pass_config.quant_specs.serialized_quant_stats));
  }

  if (pass_config.lower_tensor_list_ops) {
    // Execute this pass before `CanonicalizerPass` in case some TensorList
    // ops are constant folded into variant types.
    // TODO(b/137125056): Move this pass after `CanonicalizerPass` after we
    // handle constant ops that produce `TensorList`.
    // TODO(haoliang): Add this pass by default.
    pass_manager->addPass(mlir::TFL::CreateLowerStaticTensorListPass());
  }

  // Enable fusing composite ops that can be lowered to built-in TFLite ops.
  if (pass_config.emit_builtin_tflite_ops) {
    pass_manager->addPass(mlir::TFL::CreatePrepareCompositeFunctionsPass());
  }

  // The ophint extractions happen before lots of other passes:
  // The assumption of ophint-extraction is each ophinted region is a black-box
  // and nodes within this black-box is NOT connected to the nodes OUTSIDE the
  // black-box.
  // Some passes may merge nodes together (such as const nodes), however, this
  // will break the ophint-extraction assumption. (The nodes within the black
  // box is not isolated anymore).
  // So ophint extraction and legalization needs to happen before
  // the canonicalization pass.
  if (pass_config.emit_builtin_tflite_ops) {
    pass_manager->addPass(mlir::TFL::CreateExtractOphintPass());
    // Convert composite op pass will happen after ophint extraction pass.
    pass_manager->addPass(mlir::TFL::CreateLegalizeOphintFuncOpPass());
  }

  // TODO(jpienaar): Revise post dialect constants.
  pass_manager->addPass(mlir::TF::CreateDecodeConstantPass());
  // Canonicalization includes const folding, which is utilized here to optimize
  // away ops that can't get constant folded after PrepareTF pass. For example,
  // tf.Conv2D is split into tf.Transpose and tfl.Conv2D.
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCSEPass());

  if (pass_config.inline_functions) {
    pass_manager->addPass(mlir::createInlinerPass());
  }

  // The below passes only make sense if Builtin TFLite ops are enabled
  // for emission.
  if (pass_config.emit_builtin_tflite_ops) {
    // Prepare for TFLite dialect, rerun canonicalization, and then legalize to
    // the TFLite dialect.
    pass_manager->addPass(mlir::TFL::CreatePrepareTFPass());
    pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
    pass_manager->addPass(mlir::TFL::CreateLegalizeTFPass());
    pass_manager->addPass(mlir::TFL::CreateOptimizePass());
    // This pass operates on TensorFlow ops but is triggered after legalization
    // so that it can target constants introduced once TensorFlow Identity ops
    // are removed during legalization.
    pass_manager->addPass(mlir::TFL::CreateOptimizeFunctionalOpsPass());
    pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
    pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
    // This pass should be always at the end of the floating point model
    // conversion. Some TFL ops like unidirectional
    // sequence lstm will have stateful operands and some optimization passes
    // will merge those operands if they have identical values & types. However,
    // it's not desired by TFL. This pass serves as a "fix" pass to split the
    // merged inputs until we have 1st class variable support or reuse
    // tf.variable to model this.
    pass_manager->addPass(mlir::TFL::CreateSplitMergedOperandsPass());

    // Run quantization after all the floating point model conversion is
    // completed.
    if (pass_config.quant_specs.RunPropagationAndRewriteQuantizationPasses()) {
      AddQuantizationPasses(pass_config.quant_specs, pass_manager);
    }
  }
}

}  // namespace tensorflow

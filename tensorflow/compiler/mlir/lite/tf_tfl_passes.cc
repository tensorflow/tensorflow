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
    // TODO(haoliang): Add this pass by default.
    pass_manager->addPass(mlir::TFL::CreateLowerStaticTensorListPass());
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

  // This decomposes resource ops like ResourceGather into read-variable op
  // followed by gather. This is used when the saved model import path is used
  // during which resources dont get frozen in the python layer.
  pass_manager->addNestedPass<mlir::FuncOp>(
      mlir::TFDevice::CreateDecomposeResourceOpsPass());

  // This pass does resource analysis of saved model global tensors and marks
  // those deemed read-only as immutable.
  pass_manager->addPass(
      mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());
  // This pass marks non-exported functions as symbol visibility 'private'
  // those deemed read-only as immutable.
  pass_manager->addPass(
      mlir::tf_saved_model::
          CreateMarkFunctionVisibilityUsingSavedModelLinkagePass());

  // Enable fusing composite ops that can be lowered to built-in TFLite ops.
  if (pass_config.emit_builtin_tflite_ops) {
    pass_manager->addPass(mlir::TFL::CreatePrepareCompositeFunctionsPass());
  }

  // Legalize while early to allow further constant folding.
  // TODO(jpienaar): This may not actually matter as we do canonicalization
  // after the legalize below, for now it needs to be below the above passes
  // that work on TF dialect and before inliner so that the function calls in
  // body and cond are inlined for optimization.
  if (pass_config.legalize_tf_while) {
    pass_manager->addNestedPass<mlir::FuncOp>(
        mlir::TFL::CreateLegalizeTFWhilePass());
  }

  // Add function inlining pass. Both TF and TFLite dialects are opted into
  // function inliner interface.
  pass_manager->addPass(mlir::createInlinerPass());

  // TODO(jpienaar): Revise post dialect constants.
  pass_manager->addPass(mlir::TF::CreateDecodeConstantPass());
  // Canonicalization includes const folding, which is utilized here to optimize
  // away ops that can't get constant folded after PrepareTF pass. For example,
  // tf.Conv2D is split into tf.Transpose and tfl.Conv2D.
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pass_manager->addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
  // This pass does dead code elimination based on symbol visibility.
  pass_manager->addPass(mlir::createSymbolDCEPass());
  // This pass 'freezes' immutable global tensors and inlines them as tf
  // constant ops.
  pass_manager->addPass(mlir::tf_saved_model::CreateFreezeGlobalTensorsPass());

  if (pass_config.shape_inference) {
    // Add a shape inference pass to optimize away the unnecessary casts.
    pass_manager->addPass(mlir::TF::CreateTFShapeInferencePass());
  }

  // The below passes only make sense if Builtin TFLite ops are enabled
  // for emission.
  if (pass_config.emit_builtin_tflite_ops) {
    // Prepare for TFLite dialect, rerun canonicalization, and then legalize to
    // the TFLite dialect.
    pass_manager->addPass(
        mlir::TFL::CreatePrepareTFPass(pass_config.unfold_batch_matmul));
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

namespace mlir {
namespace TFL {

struct StandardPipelineOptions
    : public PassPipelineOptions<StandardPipelineOptions> {
  // TODO(b/150915052): All the tf_tfl_translate_cl flags should
  // move inside this.
};

// NOLINTNEXTLINE
// This creates the standard pass pipeline for TF->TFLite. This
// represents a std configuration for TFLite, for use with APIs like
// tensorflow/python/pywrap_mlir.py::experimental_run_pass_pipeline
// This does not yet include quantization passes.
void CreateTFLStandardPipeline(OpPassManager& pm,
                               const StandardPipelineOptions& options) {
  OpPassManager& func_pm = pm.nest<FuncOp>();

  // tf_executor dialect passes - Cleaning up the IR.
  func_pm.addPass(tf_executor::CreateSwitchFoldPass());
  func_pm.addPass(tf_executor::CreateTFExecutorGraphPruningPass());
  func_pm.addPass(tf_executor::CreateTFExecutorIslandCoarseningPass());

  // more cleanup of executor dialect and raise to control flow.
  pm.addPass(mlir::CreateTFExecutorToControlDialectConversion());
  pm.addPass(mlir::TFControlFlow::CreateRaiseTFControlFlowPass());

  // This is needed for control flow support with TF TensorList.
  pm.addPass(mlir::TFL::CreateLowerStaticTensorListPass());

  // Saved model pass to mark global tensors immutable.
  pm.addPass(mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());
  // Used to mark non-exported functions in saved model private.
  pm.addPass(mlir::tf_saved_model::
                 CreateMarkFunctionVisibilityUsingSavedModelLinkagePass());
  // Op fusion pass.
  pm.addPass(mlir::TFL::CreatePrepareCompositeFunctionsPass());

  pm.addNestedPass<mlir::FuncOp>(mlir::TFL::CreateLegalizeTFWhilePass());

  pm.addPass(mlir::createInlinerPass());

  // Canonicalize, CSE etc.
  pm.addPass(mlir::TF::CreateDecodeConstantPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
  // DCE for private symbols.
  pm.addPass(mlir::createSymbolDCEPass());

  // freeze global tensors.
  pm.addPass(mlir::tf_saved_model::CreateFreezeGlobalTensorsPass());

  // TFLite dialect passes.
  pm.addPass(mlir::TFL::CreatePrepareTFPass(true));
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::TFL::CreateLegalizeTFPass());
  pm.addPass(mlir::TFL::CreateOptimizePass());
  pm.addPass(mlir::TFL::CreateOptimizeFunctionalOpsPass());

  // Canonicalize, CSE etc.
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCSEPass());

  // Pass for stateful operands like LSTM.
  pm.addPass(mlir::TFL::CreateSplitMergedOperandsPass());

  pm.addPass(mlir::TFL::CreateWhileOutlinePass());

  pm.addPass(mlir::TFL::CreateRuntimeTypeVerifyPass());
}

// Registers a pass pipeline for the standard TFL passes.
static mlir::PassPipelineRegistration<StandardPipelineOptions> pipeline(
    "tfl-standard-pipeline",
    "Run the standard passes involved in transforming/optimizing the TF "
    "program to TFLite after "
    "importing into MLIR.",
    CreateTFLStandardPipeline);

}  // namespace TFL
}  // namespace mlir

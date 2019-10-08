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

#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Transforms/Passes.h"  // TF:local_config_mlir
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

void AddTFToTFLConversionPasses(const mlir::TFL::PassConfig& pass_config,
                                mlir::PassManager* pass_manager) {
  pass_manager->addPass(mlir::tf_executor::CreateSwitchFoldPass());
  pass_manager->addPass(mlir::CreateTFExecutorToControlDialectConversion());
  if (!pass_config.quant_specs.serialized_quant_stats.empty()) {
    pass_manager->addPass(
        mlir::quant::CreateImportQuantStatsPassForTFControlDialect(
            pass_config.quant_specs.serialized_quant_stats));
  }
  pass_manager->addPass(mlir::TFControlFlow::CreateRaiseTFControlFlowPass());
  if (pass_config.lower_tensor_list_ops) {
    // Execute this pass before `CanonicalizerPass` in case some TensorList
    // ops are constant folded into variant types.
    // TODO(b/137125056): Move this pass after `CanonicalizerPass` after we
    // handle constant ops that produce `TensorList`.
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

  // TODO(jpienaar): Revise post dialect constants.
  pass_manager->addPass(mlir::TF::CreateDecodeConstantPass());
  // Canonicalization includes const folding, which is utilized here to optimize
  // away ops that can't get constant folded after PrepareTF pass. For example,
  // tf.Conv2D is split into tf.Transpose and tfl.Conv2D.
  pass_manager->addPass(mlir::createCanonicalizerPass());

  // The below passes only make sense if Builtin TFLite ops are enabled
  // for emission.
  if (pass_config.emit_builtin_tflite_ops) {
    // Prepare for TFLite dialect, rerun canonicalization, and then legalize to
    // the TFLite dialect.
    pass_manager->addPass(mlir::TFL::CreatePrepareTFPass());
    pass_manager->addPass(mlir::createCanonicalizerPass());
    pass_manager->addPass(mlir::TFL::CreateLegalizeTFPass());
    pass_manager->addPass(mlir::TFL::CreateOptimizePass());
    if (pass_config.quant_specs.RunPropagationAndRewriteQuantizationPasses()) {
      pass_manager->addPass(
          mlir::TFL::CreatePrepareQuantizePass(pass_config.quant_specs));
      pass_manager->addPass(mlir::TFL::CreateQuantizePass());
      pass_manager->addPass(mlir::TFL::CreatePostQuantizePass(
          pass_config.emit_quant_adaptor_ops));
    }
    pass_manager->addPass(mlir::createCanonicalizerPass());
    pass_manager->addPass(mlir::createCSEPass());
    // This pass should be always at the end. Some TFL ops like unidirectional
    // sequence lstm will have stateful operands and some optimization passes
    // will merge those operands if they have identical values & types. However,
    // it's not desired by TFL. This pass serves as a "fix" pass to split the
    // merged inputs until we have 1st class variable support or reuse
    // tf.ariable to model this.
    pass_manager->addPass(mlir::TFL::CreateSplitMergedOperandsPass());
  }
}

}  // namespace tensorflow

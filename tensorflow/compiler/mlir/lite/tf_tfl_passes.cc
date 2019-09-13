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
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/decode_constant.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"

namespace mlir {
/// Create a pass to convert from the TFExecutor to the TF control dialect.
std::unique_ptr<FunctionPassBase> CreateTFExecutorToControlDialectConversion();
}  // namespace mlir

namespace tensorflow {

bool ShouldRunQuantizePasses(mlir::ModuleOp m) {
  if (mlir::FuncOp main_fn = m.lookupSymbol<mlir::FuncOp>("main")) {
    return main_fn.getAttrOfType<mlir::UnitAttr>("tf.quantize") !=
           mlir::Attribute();
  }
  return false;
}

void AddTFToTFLConversionPasses(const mlir::TFL::PassConfig& pass_config,
                                mlir::PassManager* pass_manager) {
  pass_manager->addPass(mlir::tf_executor::CreateSwitchFoldPass());
  pass_manager->addPass(mlir::CreateTFExecutorToControlDialectConversion());
  pass_manager->addPass(mlir::TFControlFlow::CreateRaiseTFControlFlowPass());
  // Ophint extraction will happen after island extraction pass.
  pass_manager->addPass(mlir::TFL::CreateExtractOphintPass());
  // Convert composite op pass will happen after ophint extraction pass.
  pass_manager->addPass(mlir::TFL::CreateLegalizeOphintFuncOpPass());

  if (pass_config.lower_tensor_list_ops) {
    // Execute this pass before `CanonicalizerPass` in case some TensorList
    // ops are constant folded into variant types.
    // TODO(b/137125056): Move this pass after `CanonicalizerPass` after we
    // handle constant ops that produce `TensorList`.
    // TODO(haoliang): Add this pass by default.
    pass_manager->addPass(mlir::TFL::CreateLowerStaticTensorListPass());
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
    if (pass_config.run_quantize) {
      pass_manager->addPass(mlir::TFL::CreatePrepareQuantizePass(
          /*quantize_sign=*/false));
      pass_manager->addPass(mlir::TFL::CreateQuantizePass());
      pass_manager->addPass(mlir::TFL::CreatePostQuantizePass(
          pass_config.emit_quant_adaptor_ops));
    }
    pass_manager->addPass(mlir::createCanonicalizerPass());
    pass_manager->addPass(mlir::createCSEPass());
  }
}

}  // namespace tensorflow

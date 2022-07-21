/* Copyright 2019 Google Inc. All Rights Reserved.

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

#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/InitAllPasses.h"  // from @llvm-project
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "tensorflow//compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/transforms/register_passes.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/register.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/register_passes.h"
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/test_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_graph_optimization_pass.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tosa/tf_passes.h"
#include "tensorflow/compiler/mlir/tosa/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/tosa/tfl_passes.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_passes.h"
#include "tensorflow/core/platform/init_main.h"

int main(int argc, char **argv) {
  tensorflow::InitMlir y(&argc, &argv);

  mlir::registerAllPasses();
  mlir::registerTensorFlowPasses();
  mlir::TFDevice::registerTensorFlowDevicePasses();
  mlir::tf_saved_model::registerTensorFlowSavedModelPasses();
  mlir::TFL::registerTensorFlowLitePasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  // These are in compiler/mlir/xla and not part of the above MHLO passes.
  mlir::mhlo::registerTfXlaPasses();
  mlir::mhlo::registerXlaPasses();
  mlir::mhlo::registerLegalizeTFPass();
  mlir::mhlo::registerLegalizeTFControlFlowPass();
  mlir::mhlo::registerLegalizeTfTypesPassPass();
  mlir::tosa::registerLegalizeTosaPasses();
  mlir::tosa::registerTFtoTOSALegalizationPipeline();
  mlir::tosa::registerTFLtoTOSALegalizationPipeline();
  mlir::tosa::registerTFTFLtoTOSALegalizationPipeline();
  mlir::tf_test::registerTensorFlowTestPasses();
  tensorflow::RegisterConvertMlirToXlaHloPipelineWithDefaults();
  tensorflow::RegisterGraphOptimizationPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  registry.insert<mlir::shape::ShapeDialect>();
  registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  registry.insert<mlir::kernel_gen::tf_framework::TFFrameworkDialect>();
  return failed(
      mlir::MlirOptMain(argc, argv, "TensorFlow pass driver\n", registry));
}

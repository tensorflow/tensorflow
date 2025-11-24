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

#include "mlir/InitAllPasses.h"               // from @llvm-project
#include "mlir/Support/LogicalResult.h"       // from @llvm-project
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"           // from @llvm-project
#include "tensorflow//compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "tensorflow/compiler/mlir/register_common_dialects.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/host_runtime/lower_cluster_to_runtime_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/host_runtime/runtime_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/sparsecore/sparsecore_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/test_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_graph_optimization_pass.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mlprogram_util.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/mlir_to_graph_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/tosa/tf_passes.h"
#include "tensorflow/compiler/mlir/tosa/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/tosa/tfl_passes.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"
#include "xla/mlir/framework/transforms/passes.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"

int main(int argc, char** argv) {
  tensorflow::InitMlir y(&argc, &argv);

  mlir::registerAllPasses();
  mlir::registerTransformsPasses();
  mlir::registerTensorFlowPasses();
  mlir::TFDevice::registerTensorFlowDevicePasses();
  mlir::tf_saved_model::registerTensorFlowSavedModelPasses();
  mlir::TFL::registerTensorFlowLitePasses();
  mlir::mhlo::registerAllMhloPasses();

  // These are in compiler/mlir/tf2xla and not part of the above MHLO passes.
  mlir::mhlo::registerLegalizeTfPasses();
  mlir::mhlo::registerTfXlaPasses();
  mlir::quant::stablehlo::registerBridgePasses();
  tensorflow::tf2xla::internal::registerTFXLABridgeClusteringPasses();
  tensorflow::tf2xla::internal::registerTFXLABridgeMlirToGraphPasses();
  mlir::tf_test::registerTensorFlowTestPasses();
  mlir::xla_framework::registerXlaFrameworkPasses();
  tensorflow::RegisterConvertMlirToXlaHloPipelineWithDefaults();
  tensorflow::RegisterGraphOptimizationPasses();
  tensorflow::RegisterMlProgramPasses();
  mlir::TFTPU::registerRuntimeLoweringPasses();
  mlir::TFDevice::registerSparseCorePasses();
  mlir::tosa::registerLegalizeTosaPasses();
  mlir::tosa::registerTFtoTOSALegalizationPipeline();
  mlir::tosa::registerTFLtoTOSALegalizationPipeline();
  mlir::tosa::registerTFTFLtoTOSALegalizationPipeline();

  tensorflow::tfrt_compiler::RegisterTPULowerClusterToRuntimeOpsPassPipeline();
  tensorflow::tfrt_compiler::
      RegisterNonTPULowerClusterToRuntimeOpsPassPipeline();

  mlir::DialectRegistry registry;
  mlir::RegisterCommonToolingDialects(registry);
  registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  registry.insert<mlir::quantfork::QuantizationForkDialect>();

  return failed(
      mlir::MlirOptMain(argc, argv, "TensorFlow pass driver\n", registry));
}

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

#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"

#include <memory>

#include "mlir/Pass/PassManager.h"  // TF:llvm-project
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/bridge_logger.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

namespace mlir {
namespace TFTPU {

void CreateTPUBridge(OpPassManager &pm) {
  // Run shape inference so that tf_executor/tf_device ops created later will
  // likely to inherit more concrete types.
  pm.addPass(TF::CreateTFShapeInferencePass());
  OpPassManager &func_pm = pm.nest<FuncOp>();
  func_pm.addPass(tf_executor::CreateTFExecutorIslandCoarseningPass());
  func_pm.addPass(CreateTPUClusterFormationPass());
  func_pm.addPass(createCanonicalizerPass());
  // Place DecomposeResourceOpsPass before TFExecutorConstantSinking pass
  // because DecomposeResourceOpsPass uses pattern rewriter which hoists
  // changed constants out of tf_device.Launch.
  func_pm.addPass(TFDevice::CreateDecomposeResourceOpsPass());

  // Run another shape inference pass because resource ecomposition might have
  // created new partial types.
  pm.addPass(TF::CreateTFShapeInferencePass());
  OpPassManager &func_pm2 = pm.nest<FuncOp>();
  func_pm2.addPass(tf_executor::CreateTFExecutorConstantSinkingPass());
  func_pm2.addPass(TFDevice::CreateResourceOpLiftingPass());

  pm.addPass(TF::CreateResourceDeviceInferencePass());
  pm.addPass(TFDevice::CreateClusterOutliningPass());
  pm.addPass(CreateTPUDynamicPaddingMapperPass());
  pm.addPass(TFDevice::CreateAnnotateParameterReplicationPass());
  pm.addPass(CreateTPURewritePass());
  pm.addNestedPass<FuncOp>(TFDevice::CreateReplicateInvariantOpHoistingPass());
  pm.addNestedPass<FuncOp>(CreateFunctionalToExecutorDialectConversionPass());
  pm.addNestedPass<FuncOp>(CreateTPUMergeVariablesWithExecutePass());
  pm.addNestedPass<FuncOp>(CreateBreakUpIslandsPass());
  pm.addNestedPass<FuncOp>(TFDevice::CreateReplicateToIslandPass());
  pm.addNestedPass<FuncOp>(CreateBreakUpIslandsPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
}

tensorflow::Status TPUBridge(ModuleOp module, bool enable_logging) {
  PassManager bridge(module.getContext());

  // Add logger to bridge passmanager.
  if (enable_logging)
    bridge.enableIRPrinting(std::make_unique<tensorflow::BridgeLoggerConfig>());

  // Populate a passmanager with the list of passes that implement the bridge.
  CreateTPUBridge(bridge);

  // Run the bridge on the module, in case of failure, the `diag_handler`
  // converts MLIR errors emitted to the MLIRContext into a tensorflow::Status.
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  LogicalResult result = bridge.run(module);
  (void)result;
  return diag_handler.ConsumeStatus();
}

}  // namespace TFTPU

namespace TF {

tensorflow::Status RunBridgeWithStandardPipeline(ModuleOp module,
                                                 bool enable_logging,
                                                 bool enable_inliner) {
  PassManager bridge(module.getContext());

  // Add logger to bridge passmanager.
  if (enable_logging)
    bridge.enableIRPrinting(std::make_unique<tensorflow::BridgeLoggerConfig>());

  StandardPipelineOptions pipeline_options;
  pipeline_options.enable_inliner.setValue(enable_inliner);
  CreateTFStandardPipeline(bridge, pipeline_options);
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  LogicalResult result = bridge.run(module);
  (void)result;
  return diag_handler.ConsumeStatus();
}

}  // namespace TF
}  // namespace mlir

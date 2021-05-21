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

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/bridge_logger.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

namespace mlir {
namespace {
// Add logger to bridge passmanager.
void EnableLogging(PassManager *pm) {
  // Print the whole module after each pass, which requires disabling
  // multi-threading as well.
  pm->getContext()->disableMultithreading();
  pm->enableIRPrinting(std::make_unique<tensorflow::BridgeLoggerConfig>(
      /*print_module_scope=*/true));
  pm->enableTiming(std::make_unique<tensorflow::BridgeTimingConfig>());
}
}  // namespace

namespace TFTPU {

namespace {
tensorflow::Status RunTPUBridge(
    ModuleOp module, bool enable_logging,
    llvm::function_ref<void(OpPassManager &pm)> pipeline_builder) {
  PassManager bridge(module.getContext());
  ::tensorflow::applyTensorflowAndCLOptions(bridge);
  if (enable_logging || VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("tpu_bridge_before", module);
    if (VLOG_IS_ON(2)) EnableLogging(&bridge);
  }

  // Populate a passmanager with the list of passes that implement the bridge.
  pipeline_builder(bridge);

  // Add set of passes to lower back to graph (from tf_executor).
  TF::AddGraphExportLoweringPasses(bridge);

  // Run the bridge on the module, in case of failure, the `diag_handler`
  // converts MLIR errors emitted to the MLIRContext into a tensorflow::Status.
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  LogicalResult result = bridge.run(module);
  (void)result;
  if (enable_logging || VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile("tpu_bridge_after", module);
  return diag_handler.ConsumeStatus();
}
}  // namespace

void CreateTPUBridgePipeline(OpPassManager &pm) {
  // The following ops must be preserved regardless of reachability. Ideally,
  // all graphs should have control dependencies to enforce this but this is
  // currently not the case (see b/177478741).
  const llvm::SmallVector<std::string, 4> ops_to_preserve = {
      "tf.TPUReplicateMetadata", "tf.TPUCompilationResult",
      "tf.TPUReplicatedOutput"};
  pm.addNestedPass<FuncOp>(
      tf_executor::CreateTFExecutorGraphPruningPass(ops_to_preserve));
  // It is assumed at this stage there are no V1 control flow ops as Graph
  // functionalization is ran before import. Ops can be lifted out of
  // tf_executor dialect islands/graphs.
  pm.addNestedPass<FuncOp>(CreateExecutorDialectToFunctionalConversionPass());
  // Run shape inference so that tf_executor/tf_device ops created later will
  // likely to inherit more concrete types.
  pm.addPass(TF::CreateTFShapeInferencePass());
  pm.addNestedPass<FuncOp>(CreateTPUReorderReplicateAndPartitionedInputsPass());
  pm.addPass(CreateTPUClusterFormationPass());
  pm.addPass(CreateOutsideCompiledToHostLaunchPass());
  pm.addNestedPass<FuncOp>(TFDevice::CreateDeviceAttributeToLaunchPass());
  // Place DecomposeResourceOpsPass before TFExecutorConstantSinking pass
  // because DecomposeResourceOpsPass uses pattern rewriter which hoists
  // changed constants out of tf_device.Launch.
  pm.addPass(TFDevice::CreateDecomposeResourceOpsInClusterPass());
  // Encode this in its own scope so that func_pm is not mistakenly used
  // later on.
  {
    OpPassManager &func_pm = pm.nest<FuncOp>();
    func_pm.addPass(CreateTPUHostComputationExpansionPass());
    func_pm.addPass(CreateTPUUpdateEmbeddingEnqueueOpInputsPass());
  }
  // TODO(b/173622615): This should incrementally be moved down as
  // more passes support this representation and then can be removed once
  // all passes support it.
  pm.addPass(TFDevice::CreateHostLaunchToOutsideCompiledPass());

  // TODO(b/173622615): Once OutsideCompilation is represented by launch op and
  // the remaining passes including Inliner support it, remove this
  // LaunchToDeviceAttributePass. This LaunchToDeviceAttribute pass needs to
  // come before TPUClusterCleanupAttributes pass or else the device attribute
  // will be removed from launch causing an error.
  pm.addNestedPass<FuncOp>(TFDevice::CreateLaunchToDeviceAttributePass());

  // Note that the region-based control-flow produced here still contains
  // function call ops which get inlined by the subsequent inliner pass.
  pm.addPass(TF::CreateTFFunctionalControlFlowToRegions());
  pm.addPass(CreateOutsideCompiledToHostLaunchPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<FuncOp>(
      TF::CreateDropWhileShapeInvariantInDeviceClusterPass());
  // Run another shape inference pass because resource decomposition might have
  // created new partial types. Also, after dropping `shape_invariant` attribute
  // from While/WhileRegion ops within cluster would lead to more precise
  // shapes.
  pm.addPass(TF::CreateTFShapeInferencePass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addPass(CreateTPUClusterCleanupAttributesPass());
  // TODO(b/173622615): This should incrementally be moved down as
  // more passes support this representation and then can be removed once
  // all passes support it.
  pm.addPass(TFDevice::CreateHostLaunchToOutsideCompiledPass());
  pm.addPass(TFDevice::CreateResourceOpLiftingPass());
  // Re-run the canonicalizer pass as some cleanup during resource op lifting
  // pass opens up some opportunities for canonicalization of cluster ops.
  // Specifically, we want to eliminate pass through results from the cluster
  // op.
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  if (tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_merge_control_flow_pass) {
    pm.addPass(TFDevice::CreateMergeControlFlowPass());
  }

  pm.addPass(TFDevice::CreateMarkOpsForOutsideCompilationPass());
  pm.addPass(CreateTPUExtractHeadTailOutsideCompilationPass());
  pm.addPass(CreateTPUExtractOutsideCompilationPass());

  pm.addNestedPass<FuncOp>(TFDevice::CreateClusterConstantSinkingPass());
  pm.addPass(TF::CreateResourceDeviceInferencePass());
  pm.addPass(TFDevice::CreateClusterOutliningPass());
  pm.addPass(CreateTPUResourceReadForWritePass());
  pm.addPass(CreateTPUShardingIdentificationPass());
  pm.addNestedPass<FuncOp>(CreateTPUResourceReadsWritesPartitioningPass());
  pm.addPass(TFDevice::CreateAnnotateParameterReplicationPass());
  pm.addPass(TFDevice::CreateMarkInputOutputAliasesPass());
  pm.addPass(CreateTPURewritePass());
  pm.addPass(createSymbolDCEPass());
  pm.addNestedPass<FuncOp>(TFDevice::CreateReplicateInvariantOpHoistingPass());
  pm.addNestedPass<FuncOp>(CreateTPUMergeVariablesWithExecutePass());
  pm.addNestedPass<FuncOp>(
      TF::CreateHoistReplicateInvariantResourceWritesPass());
  pm.addNestedPass<FuncOp>(CreateTPUColocateCompositeResourceOps());
  pm.addPass(CreateTPUVariableReformattingPass());
  pm.addPass(TF::CreateTFRegionControlFlowToFunctional());
}

void CreateTPUBridgePipelineV1(OpPassManager &pm) {
  pm.addPass(TF::CreateTFShapeInferencePass());
  // For V1 compatibility, we process a module where the graph does not have
  // feeds and fetched. We extract first the TPU computation in a submodule,
  // where it'll be in a function with args and returned values, much more like
  // a TF v2 module. We can then run the usual pipeline on this nested module.
  // Afterward we inline back in the parent module and delete the nested one.
  pm.addPass(tf_executor::CreateTFExecutorTPUV1IslandCoarseningPass());
  pm.addPass(tf_executor::CreateTFExecutorTPUV1IslandOutliningPass());
  OpPassManager &nested_module = pm.nest<ModuleOp>();
  CreateTPUBridgePipeline(nested_module);
  pm.addPass(tf_executor::CreateTFExecutorTPUV1IslandInliningPass());
}

tensorflow::Status TPUBridge(ModuleOp module, bool enable_logging) {
  return RunTPUBridge(module, enable_logging, CreateTPUBridgePipeline);
}
tensorflow::Status TPUBridgeV1Compat(ModuleOp module, bool enable_logging) {
  return RunTPUBridge(module, enable_logging, CreateTPUBridgePipelineV1);
}

}  // namespace TFTPU

namespace TF {

void AddGraphExportLoweringPasses(OpPassManager &pm) {
  auto add_pass = [&](std::unique_ptr<Pass> pass) {
    pm.addNestedPass<FuncOp>(std::move(pass));
    pm.addPass(CreateBreakUpIslandsPass());
  };

  add_pass(CreateFunctionalToExecutorDialectConversionPass());
  add_pass(TFDevice::CreateReplicateToIslandPass());
  add_pass(TFDevice::CreateParallelExecuteToIslandsPass());
  add_pass(TFDevice::CreateLaunchToDeviceAttributePass());
  pm.addNestedPass<FuncOp>(TFTPU::CreateTPUDevicePropagationPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(CreateVerifySuitableForExportPass());
}

tensorflow::Status RunBridgeWithStandardPipeline(ModuleOp module,
                                                 bool enable_logging,
                                                 bool enable_inliner) {
  PassManager bridge(module.getContext());
  if (enable_logging || VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("standard_pipeline_before", module);
    if (VLOG_IS_ON(2)) EnableLogging(&bridge);
  }

  StandardPipelineOptions pipeline_options;
  pipeline_options.enable_inliner.setValue(enable_inliner);
  CreateTFStandardPipeline(bridge, pipeline_options);
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  LogicalResult result = bridge.run(module);
  (void)result;
  if (enable_logging || VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile("standard_pipeline_after", module);
  return diag_handler.ConsumeStatus();
}

}  // namespace TF
}  // namespace mlir

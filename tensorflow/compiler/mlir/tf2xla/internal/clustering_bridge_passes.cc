/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/internal/clustering_bridge_passes.h"

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

using mlir::OpPassManager;
using mlir::func::FuncOp;

// Adds Bridge clustering pipeline passes to the given pass_manager. Does not
// run them.
void AddBridgeClusteringPipelinePasses(OpPassManager& pm,
                                       llvm::StringRef module_name) {
  // The following ops must be preserved regardless of reachability. Ideally,
  // all graphs should have control dependencies to enforce this but this is
  // currently not the case (see b/177478741).
  const llvm::SmallVector<std::string, 4> ops_to_preserve = {
      "tf.TPUReplicateMetadata", "tf.TPUCompilationResult",
      "tf.TPUReplicatedOutput"};
  bool strict_clusters =
      tensorflow::GetMlirCommonFlags()->tf_mlir_enable_strict_clusters;
  pm.addNestedPass<FuncOp>(
      mlir::tf_executor::CreateTFExecutorGraphPruningPass(ops_to_preserve));
  // It is assumed at this stage there are no V1 control flow ops as Graph
  // functionalization is ran before import. Ops can be lifted out of
  // tf_executor dialect islands/graphs.
  pm.addNestedPass<FuncOp>(
      mlir::CreateExecutorDialectToFunctionalConversionPass());
  // Guarantee all functions have one use, which enables more exact shape
  // inference.
  pm.addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  // Run shape inference so that tf_executor/tf_device ops created later will
  // likely to inherit more concrete types.
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addNestedPass<FuncOp>(mlir::TFTPU::CreateTPUPartitionedOpConversionPass());
  pm.addNestedPass<FuncOp>(
      mlir::TFTPU::CreateTPUReorderReplicateAndPartitionedInputsPass());
  pm.addNestedPass<FuncOp>(mlir::TF::CreateDecomposeReduceDatasetPass());
  // Only one of EmbeddingSequencing and EmbeddingPipelining will actually
  // run and the logic is in EmbeddingPipeliningPass. If the pipelining pass
  // runs, embedding attributes are stripped and the sequencing pass will have
  // no effect. If the pipelining pass doesn't run, embedding attributes are
  // preserved and the sequencing rewrite will trigger.
  pm.addPass(mlir::TFDevice::CreateEmbeddingPipeliningPass());
  pm.addPass(mlir::TFDevice::CreateEmbeddingSequencingPass());
  pm.addPass(mlir::TFTPU::CreateTPUClusterFormationPass(strict_clusters));
  // CreateEmbeddingPipeliningPass may have created more functions, but
  // TPUClusterCleanup and OutsideCompiledToHostLaunch need every function to be
  // only called from one cluster. Here, we choose to fix the all-funcs-one-use
  // invariant right before it's needed, not after it's been broken.
  pm.addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  // Run TPU cluster cleanup attributes so ops with no outside compiled
  // attribute have no host device attribute.
  pm.addPass(mlir::TFTPU::CreateTPUClusterCleanupAttributesPass());
  pm.addPass(mlir::TFDevice::CreateOutsideCompiledToHostLaunchPass());
  pm.addNestedPass<FuncOp>(mlir::TFDevice::CreateDeviceAttributeToLaunchPass());
  // Running canonicalizer before decomposing resource ops in cluster helps the
  // latter pass to converge faster as it does not have to spend time folding
  // away dead ops.
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  // Place DecomposeResourceOpsPass before TFExecutorConstantSinking pass
  // because DecomposeResourceOpsPass uses pattern rewriter which hoists
  // changed constants out of tf_device.Launch.
  pm.addPass(mlir::TFDevice::CreateDecomposeResourceOpsInClusterPass());
  // Encode this in its own scope so that func_pm is not mistakenly used
  // later on.
  {
    OpPassManager& func_pm = pm.nest<FuncOp>();
    func_pm.addPass(mlir::TFTPU::CreateTPUHostComputationExpansionPass());
    func_pm.addPass(mlir::TFTPU::CreateTPUUpdateEmbeddingEnqueueOpInputsPass());
  }
  // TODO(b/173622615): This should incrementally be moved down as
  // more passes support this representation and then can be removed once
  // all passes support it.
  pm.addPass(mlir::TFDevice::CreateHostLaunchToOutsideCompiledPass());

  // TODO(b/173622615): Once OutsideCompilation is represented by launch op and
  // the remaining passes including Inliner support it, remove this
  // LaunchToDeviceAttributePass. This LaunchToDeviceAttribute pass needs to
  // come before TPUClusterCleanupAttributes pass or else the device attribute
  // will be removed from launch causing an error.
  pm.addNestedPass<FuncOp>(mlir::TFDevice::CreateLaunchToDeviceAttributePass());

  // TODO(b/173622615): This can be removed once more passes support outside
  // compilation represented by op and conversion back to attribute is removed.
  pm.addPass(mlir::TFDevice::CreateOutsideCompiledToHostLaunchPass());
  // Note that the region-based control-flow produced here still contains
  // function call ops which get inlined by the subsequent inliner pass.
  pm.addPass(mlir::TF::CreateTFFunctionalControlFlowToRegions());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<FuncOp>(
      mlir::TF::CreateDropWhileShapeInvariantInDeviceClusterPass());
  // Run another shape inference pass because resource decomposition might have
  // created new partial types. Also, after dropping `shape_invariant` attribute
  // from While/WhileRegion ops within cluster would lead to more precise
  // shapes.
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::TFTPU::CreateTPUClusterCleanupAttributesPass());
  pm.addPass(mlir::TFDevice::CreateResourceOpLiftingPass());
  // Re-run the canonicalizer pass as some cleanup during resource op lifting
  // pass opens up some opportunities for canonicalization of cluster ops.
  // Specifically, we want to eliminate pass through results from the cluster
  // op.
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  if (tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_merge_control_flow_pass) {
    pm.addPass(mlir::TFDevice::CreateMergeControlFlowPass());
  }

  // TODO(b/173622615): This should incrementally be moved down as
  // more passes support this representation and then can be removed once
  // all passes support it.
  pm.addPass(mlir::TFDevice::CreateHostLaunchToOutsideCompiledPass());

  pm.addPass(mlir::TFDevice::CreateMarkOpsForOutsideCompilationPass());
  pm.addPass(mlir::TFDevice::CreateExtractHeadTailOutsideCompilationPass());
  pm.addPass(mlir::TFDevice::CreateExtractOutsideCompilationPass());
  pm.addNestedPass<FuncOp>(
      mlir::TFDevice::CreateVerifyNoOutsideCompilationMarkersPass());

  pm.addNestedPass<FuncOp>(mlir::TFDevice::CreateClusterConstantSinkingPass());
  pm.addPass(mlir::TF::CreateResourceDeviceInferencePass());
  pm.addPass(mlir::TFDevice::CreateClusterOutliningPass());
  pm.addPass(mlir::TFTPU::CreateTPUResourceReadForWritePass());
  pm.addPass(mlir::TFDevice::CreateMarkInputOutputAliasesPass());
  pm.addPass(mlir::TFTPU::CreateTPUShardingIdentificationPass());
  pm.addNestedPass<FuncOp>(
      mlir::TFTPU::CreateTPUResourceReadsWritesPartitioningPass());
  pm.addPass(mlir::TFDevice::CreateAnnotateParameterReplicationPass());
  pm.addNestedPass<FuncOp>(mlir::TF::CreateRewriteTPUEmbeddingOpsPass());
  pm.addPass(mlir::TFTPU::CreateTPUAnnotateDynamicShapeInputsPass());
  pm.addNestedPass<FuncOp>(
      mlir::TF::CreateHoistReplicateInvariantResourceWritesPass());

  pm.addPass(mlir::TFTPU::CreateTPURewritePass(module_name));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addNestedPass<FuncOp>(mlir::TFDevice::CreateEmbeddingProgramKeyPass());
  pm.addNestedPass<FuncOp>(
      mlir::TFDevice::CreateReplicateInvariantOpHoistingPass());
  pm.addPass(mlir::TFTPU::CreateTPUMergeVariablesWithExecutePass());
  pm.addNestedPass<FuncOp>(
      mlir::TFTPU::CreateExtractTPUCopyWithDynamicShapeOpPass());
  pm.addNestedPass<FuncOp>(
      mlir::TFTPU::CreateTPUColocateCompositeResourceOps());
  if (tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_tpu_variable_runtime_reformatting_pass) {
    pm.addPass(mlir::TFTPU::CreateTPUVariableRuntimeReformattingPass());
  }
}

};  // namespace internal
};  // namespace tf2xla
};  // namespace tensorflow

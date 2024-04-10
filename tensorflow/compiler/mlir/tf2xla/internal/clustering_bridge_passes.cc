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

#include "absl/log/log.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/sparsecore/sparsecore_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

using mlir::OpPassManager;
using mlir::func::FuncOp;

// LINT.IfChange(replicated_bridge_passes)

// Adds replicated Bridge clustering pipeline passes to the given pass_manager.
// Does not run them.
void AddReplicatedBridgeClusteringPipelinePasses(OpPassManager& pm,
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
  pm.addPass(tensorflow::tf2xla::internal::CreateTPUClusterFormationPass(
      strict_clusters));
  // CreateEmbeddingPipeliningPass may have created more functions, but
  // TPUClusterCleanup and OutsideCompiledToHostLaunch need every function to be
  // only called from one cluster. Here, we choose to fix the all-funcs-one-use
  // invariant right before it's needed, not after it's been broken.
  pm.addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  // Run TPU cluster cleanup attributes so ops with no outside compiled
  // attribute have no host device attribute.
  pm.addPass(mlir::TFTPU::CreateTPUClusterCleanupAttributesPass());
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

  // TODO(b/173622615): Once OutsideCompilation is represented by launch op and
  // the remaining passes including Inliner support it, remove this
  // LaunchToDeviceAttributePass. This LaunchToDeviceAttribute pass needs to
  // come before TPUClusterCleanupAttributes pass or else the device attribute
  // will be removed from launch causing an error.
  pm.addNestedPass<FuncOp>(mlir::TFDevice::CreateLaunchToDeviceAttributePass());

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

  pm.addPass(
      tensorflow::tf2xla::internal::CreateMarkOpsForOutsideCompilationPass());
  pm.addPass(tensorflow::tf2xla::internal::
                 CreateExtractHeadTailOutsideCompilationPass());
  pm.addPass(
      tensorflow::tf2xla::internal::CreateExtractOutsideCompilationPass());
  pm.addNestedPass<FuncOp>(
      mlir::TFDevice::CreateVerifyNoOutsideCompilationMarkersPass());

  pm.addNestedPass<FuncOp>(mlir::TFDevice::CreateClusterConstantSinkingPass());
  pm.addPass(mlir::TF::CreateResourceDeviceInferencePass());
  pm.addNestedPass<FuncOp>(
      tensorflow::tf2xla::internal::CreateHoistBroadcastReadPass());
  pm.addNestedPass<FuncOp>(
      tensorflow::tf2xla::internal::CreateXlaBroadcastPass());
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
  // Verifies clustering has conformed with the expected invariants
  pm.addNestedPass<FuncOp>(
      tensorflow::tf2xla::internal::CreateVerifyClusteringPass());
}
// LINT.ThenChange(:non_replicated_bridge_passes)

void NoCanonicalization(OpPassManager& pm) {}

// LINT.IfChange(non_replicated_bridge_passes)

// Same as above but for non-replicated Bridge.
void AddNonReplicatedBridgeClusteringPipelinePasses(OpPassManager& pm) {
  // The following ops must be preserved regardless of reachability. Ideally,
  // all graphs should have control dependencies to enforce this.
  VLOG(2) << "Create TF XLA Bridge pipeline";
  pm.addPass(mlir::TFDevice::CreateXlaValidateInputsPass());
  pm.addNestedPass<FuncOp>(
      mlir::TF::CreateCanonicalizeCompileAndReplicateAttributesPass());
  const llvm::SmallVector<std::string, 4> ops_to_preserve = {};
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
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  // Encapsulate PartitionedCall ops within a cluster so that the composite
  // resource ops can be decomposed.
  pm.addPass(tensorflow::tf2xla::internal::CreateXlaClusterFormationPass());
  // Running canonicalizer before decomposing resource ops in cluster helps the
  // latter pass to converge faster as it does not have to spend time folding
  // away dead ops.
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  // Decompose resource ops.
  pm.addPass(mlir::TFDevice::CreateDecomposeResourceOpsInClusterPass());
  // TODO(b/267193636): Remove this flag when outside compilation
  // for generic pipeline is landed.
  if (tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_generic_outside_compilation) {
    pm.addPass(mlir::TF::CreateTFFunctionalControlFlowToRegions());
  }
  // Run another shape inference pass because resource decomposition might have
  // created new partial types. Also, after dropping `shape_invariant` attribute
  // from While/WhileRegion ops within cluster would lead to more precise
  // shapes.
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  // Inline all the function calls. Do not call canonicalizer to prevent it from
  // moving the definition of any constant operand of ops within a cluster to
  // its outside. This may cause the op to fail to verify after the cluster is
  // outlined, as the constant operand is replaced by an argument.
  pm.addPass(mlir::createInlinerPass({}, NoCanonicalization));
  // Lift resource operations out of device computation. This step needs to be
  // done after inlining.
  pm.addPass(mlir::TFDevice::CreateResourceOpLiftingPass());
  // TODO(b/267193636): Remove this flag when outside compilation
  // for generic pipeline is landed.
  if (tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_generic_outside_compilation) {
    pm.addPass(
        tensorflow::tf2xla::internal::CreateMarkOpsForOutsideCompilationPass());
    pm.addPass(tensorflow::tf2xla::internal::
                   CreateExtractHeadTailOutsideCompilationPass());
    pm.addPass(
        tensorflow::tf2xla::internal::CreateExtractOutsideCompilationPass());
  }
  // Outline clusters into cluster functions.
  pm.addPass(mlir::TFDevice::CreateClusterOutliningPass());
  // Verifies clustering has conformed with the expected invariants
  pm.addNestedPass<FuncOp>(
      tensorflow::tf2xla::internal::CreateVerifyClusteringPass());
}
// LINT.ThenChange(:replicated_bridge_passes)

};  // namespace internal
};  // namespace tf2xla
};  // namespace tensorflow

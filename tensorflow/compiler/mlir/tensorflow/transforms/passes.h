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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {

// Creates a pass that breaks up an island with multiple ops into multiple
// islands, each with a single op.
std::unique_ptr<OperationPass<FuncOp>> CreateBreakUpIslandsPass();

// Creates a pass that converts mlir functions consisting of mlir ops into a
// tf_executor dialect as a single island.
std::unique_ptr<OperationPass<FuncOp>>
CreateFunctionalToExecutorDialectConversionPass();

namespace TF {
// Transforms functional control flow operations in the standard TensorFlow
// dialect to MLIR Control Flow Graph (CFG) form.
std::unique_ptr<OperationPass<FuncOp>> CreateTFFunctionalControlFlowToCFG();

// Materialize the MlirPassthroughOp by replacing it with the MLIR module
// attached as an attribute.
std::unique_ptr<OperationPass<FuncOp>> CreateMaterializePassthroughOpPass();

// Performs Shape Inference on the TensorFlow dialect using the global registry.
std::unique_ptr<OperationPass<ModuleOp>> CreateTFShapeInferencePass();

// Optional pass which will unroll BatchMatMul and use only MatMul
std::unique_ptr<OperationPass<FuncOp>> CreateUnrollBatchMatMulPassPass();

// Optional pass which will map TF BatchMatMul to TF Einsum
std::unique_ptr<OperationPass<FuncOp>> CreateBatchMatMulToEinsumPass();

// Optimizes Tensorflow graph.
std::unique_ptr<OperationPass<FuncOp>> CreateTFOptimizePass();

// Creates pass to rewrite RecvTPUEmbeddingActivationsOp and
// SendTPUEmbeddingGradients ops to internal variants.
std::unique_ptr<OperationPass<FuncOp>> CreateRewriteTPUEmbeddingOps();

// Performs specific fusion for GPU targets.
std::unique_ptr<OperationPass<FuncOp>> CreateGpuOpFusionPass();

struct LayoutOptimizationPipelineOptions
    : public PassPipelineOptions<LayoutOptimizationPipelineOptions> {
  Option<std::string> force_data_format{
      *this, "force-data-format",
      llvm::cl::desc("Force data format for all layout sensitive ops")};
};

// Layout optimization assigns optimal data layout for layout sensitive
// operations, and cancels all redundant transposes.
void CreateLayoutOptimizationPipeline(
    OpPassManager& pm,  // NOLINT - MLIR contract is pass by mutable reference.
    const LayoutOptimizationPipelineOptions& options);

struct StandardPipelineOptions
    : public PassPipelineOptions<StandardPipelineOptions> {
  Option<bool> enable_inliner{*this, "enable-inliner",
                              llvm::cl::desc("Enable inliner."),
                              llvm::cl::init(false)};
};

// Propagates the pass manager with the passes involved in transforming or
// optimizing an MLIR graph without any target specialization.
// NOLINTNEXTLINE - MLIR contract is pass by mutable reference.
void CreateTFStandardPipeline(OpPassManager& pm,
                              const StandardPipelineOptions& options);

// Propagates device attributes of resources from callers to callees.
std::unique_ptr<OperationPass<ModuleOp>> CreateResourceDeviceInferencePass();

// Creates a pass that promotes resource reads/writes in the main function to
// inputs and outputs of the main function, assuming that resource operations
// have already been decomposed and function calls have already been inlined.
// The pass also annotates the input arguments for resources with the indices
// of their aliasing output arguments.
std::unique_ptr<OperationPass<ModuleOp>> CreatePromoteResourcesToArgsPass();

// Creates a pass that promotes tf.VarHandleOp to to resource arguments of where
// resource names are `tf_saved_model.bound_input` symbol argument attributes
// for all functions.
std::unique_ptr<OperationPass<ModuleOp>>
CreatePromoteVarHandlesToSavedModelArgsPass();

// Creates a pass that converts readonly reference variables to the
// corresponding resource variables.
std::unique_ptr<OperationPass<FuncOp>>
CreateConvertReadonlyReferenceVariablesToResourceVariablesPass();

// Marks function visibility using tf.entry_function specification. That is,
// functions with tf.entry_function attributes are marked with public
// visibility while the other functions are marked with private visibility.
LogicalResult MarkFunctionVisibilityUsingEntryFunctionSpecification(
    ModuleOp module);
// Creates a pass that uses tf.entry_function specification to mark function
// visibility.
std::unique_ptr<OperationPass<ModuleOp>>
CreateMarkFunctionVisibilityUsingEntryFunctionSpecificationPass();

// Creates a pass that marks the main function with public visibility, while
// other functions are marked with private visibility.
std::unique_ptr<OperationPass<ModuleOp>>
CreateMarkOnlyMainFunctionWithPublicVisibilityPass();

// Creates a simple device assignment pass on TF dialect for CoreRT use case.
std::unique_ptr<OperationPass<FuncOp>> CreateSimpleTFDeviceAssignmentPass(
    llvm::StringRef default_device);

// Performs resource lifting on the function body to hoist resource variable
// accesses outside all control flow statements.
LogicalResult ResourceLiftingForFunctionalControlFlow(FuncOp function);

// Converts stack ops into operations on local variables, which can later be
// removed by resource lifting. Requires known maximum sizes of stacks and
// known element shapes of push ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateStackOpsDecompositionPass();

// Converts tensor list operations into operations on buffers and sizes. Needs
// static shapes and known max element count.
std::unique_ptr<OperationPass<ModuleOp>> CreateTensorListOpsDecompositionPass();

// Converts tensor array ops into operations on local variables, which can later
// be removed by resource lifting. Requires known sizes and known element shapes
// (either defined in TensorArrayV3 or implied in the first write).
std::unique_ptr<OperationPass<ModuleOp>>
CreateTensorArrayOpsDecompositionPass();

// Create a pass that legalize HLO to TF dialect.
std::unique_ptr<OperationPass<FuncOp>> CreateLegalizeHloToTfPass();
}  // namespace TF

namespace TFControlFlow {
// Raises from the "TensorFlow Control Flow" dialect to the standard TensorFlow
// dialect.
std::unique_ptr<OperationPass<FuncOp>> CreateRaiseTFControlFlowPass();

}  // namespace TFControlFlow

namespace tf_executor {
class GraphOp;

// Returns a pass that folds switch nodes with constant predicates.
std::unique_ptr<OperationPass<FuncOp>> CreateSwitchFoldPass();

// Creates a pass to merge IslandOps from TFExecutor dialect.
std::unique_ptr<OperationPass<FuncOp>> CreateTFExecutorIslandCoarseningPass();

// Creates a pass to merge IslandOps for operation marked for execution on TPU.
// This is a V1 backward compatibility.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorTPUV1IslandCoarseningPass();

// Creates a pass to outlining TPU clusters from single IslandOp into a nested
// module suitable for being processed as-if it was a V2 module.
// This is a V1 backward compatibility.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorTPUV1IslandOutliningPass();

// Creates a pass to inline calls to the nested TPU module, this reverses the
// effect of the `TFExecutorTPUV1IslandOutlining` pass above.
// This is a V1 backward compatibility.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorTPUV1IslandInliningPass();

// Creates a pass to prune tf_executor.graph from dead nodes.
std::unique_ptr<OperationPass<FuncOp>> CreateTFExecutorGraphPruningPass();

// Prunes unreachable operations of a tf_executor.graph operation.
void PruneGraph(GraphOp graph);

// Sink `tf.Const` operations in the LaunchOp region using them. This is
// performed in order to limit the number of values implicitly captured in this
// region before outlining.
std::unique_ptr<OperationPass<FuncOp>> CreateTFExecutorConstantSinkingPass();

}  // namespace tf_executor

namespace TFDevice {
// Creates a pass that forms clusters from instructions that are assigned to
// same device.
std::unique_ptr<OperationPass<FuncOp>> CreateClusterFormationPass();

// Creates a pass that outlines regions of tf_device.launch operations.
std::unique_ptr<OperationPass<ModuleOp>> CreateClusterOutliningPass();

// A pass that decomposes composite resource operations into primitive ones like
// ReadVariableOp, AssignVariableOp and other computations to facilitate
// transformations like resource op lifting.
std::unique_ptr<OperationPass<FuncOp>> CreateDecomposeResourceOpsPass();

// Creates a pass that lifts operations on external resource variables from
// device computation nested in `tf_device::LaunchOp` out so that resource
// variable load operations are all before device computation while resource
// variable store operations are all after device computation. After this pass,
// device computation no longer interacts with external resource variables.
std::unique_ptr<OperationPass<ModuleOp>> CreateResourceOpLiftingPass();

// Lifts resource operations from tf_device.launch_func ops nested in `op`
// outside. Returns a failure if there are remaining resource-type values that
// can not be lifted.
LogicalResult LiftResourceOps(Operation* op);

// Creates a pass that hoists invariant operations in a `tf_device.replicate`.
std::unique_ptr<OperationPass<FuncOp>> CreateReplicateInvariantOpHoistingPass();

// Creates a pass that forms replica `tf_executor.island` from a single
// `tf_device.replicate` island.
std::unique_ptr<OperationPass<FuncOp>> CreateReplicateToIslandPass();

// Creates a pass that creates `tf_executor.island` from a single
// `tf_device.parallel_execute` island.
std::unique_ptr<OperationPass<FuncOp>> CreateParallelExecuteToIslandsPass();

// Creates a pass that annotates whether a LaunchFuncOp's parameters have the
// same data across replicas.
std::unique_ptr<OperationPass<ModuleOp>>
CreateAnnotateParameterReplicationPass();

// Creates a pass that hoists a `tf_device.launch` body and assigns a `device`
// attribute to each TensorFlow dialect op in the body based on the `device`
// attribute on the `tf_device.launch`.
std::unique_ptr<OperationPass<FuncOp>> CreateLaunchToDeviceAttributePass();
}  // namespace TFDevice

namespace TFTPU {
// Creates a pass that forms clusters from operations of the same
// `_tpu_replicate` attribute.
std::unique_ptr<OperationPass<FuncOp>> CreateTPUClusterFormationPass();

// Creates a pass that allows TPU program inputs to have layouts determined at
// run time.
std::unique_ptr<OperationPass<FuncOp>> CreateTPUDynamicLayoutPass();

// Creates a pass that remaps and assigns padding map from a
// `tf_device.launch_func` `padding_map` attribute to its encapsulated function.
std::unique_ptr<OperationPass<ModuleOp>> CreateTPUDynamicPaddingMapperPass();

// Creates a pass that rewrites `tf_device.launch_func` on TPUs into TPU runtime
// ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateTPURewritePass();

// Creates a pass that identifies XLASharding ops in launch op for TPU
// computation.
std::unique_ptr<OperationPass<ModuleOp>> CreateTPUShardingIdentificationPass();

// Creates a pass that merges device variable reads/updates into the surrounded
// TPUExecute node. This allows the execute node to perform in-place variable
// updates.
std::unique_ptr<OperationPass<FuncOp>> CreateTPUMergeVariablesWithExecutePass();

// Creates a pass that adds ops which perform formatting on variables at
// run-time according to compilation result.
std::unique_ptr<OperationPass<ModuleOp>> CreateTPUVariableReformattingPass();

// Creates a pass that extracts outside compilation (CPU ops inside TPU cluster)
// at head/tail of TPU cluster to run before/after TPU computation.
std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUExtractHeadTailOutsideCompilationPass();

// Creates a pass that extract outside compilation (CPU ops inside TPU cluster)
// ops to a separate parallel_execute region to run on CPU.
std::unique_ptr<OperationPass<FuncOp>> CreateTPUExtractOutsideCompilationPass();

// Populates the supplied passmanager with the passes required to run the
void CreateTPUBridgePipeline(OpPassManager& pm);

// Populates the supplied passmanager with the passes required to run the
// bridge in V1 mode.
void CreateTPUBridgePipelineV1(OpPassManager& pm);

}  // namespace TFTPU

namespace tf_saved_model {

// Creates a pass that optimizes tf_saved_model.global_tensor ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateOptimizeGlobalTensorsPass();

// Creates a pass that freezes tf_saved_model.global_tensor ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateFreezeGlobalTensorsPass();

// Creates a pass that uses tf_saved_model dialect linkage information
// to mark function visibility. That is, exported functions are marked with
// public visibility while the other functions are marked with private
// visibility.
std::unique_ptr<OperationPass<ModuleOp>>
CreateMarkFunctionVisibilityUsingSavedModelLinkagePass();

}  // namespace tf_saved_model

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_PASSES_H_
